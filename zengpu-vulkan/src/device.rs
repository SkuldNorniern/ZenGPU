//! Vulkan logical device and buffer operations (plan M1 / G2).

use std::sync::{Arc, Mutex};

use ash::{Device, vk};
use zengpu_hal::{
    AddressMode, BufferDesc, BufferHandle, BufferUsage, DeviceRequest, FilterMode, Format,
    GpuDevice, GpuError, HalCapabilities, MemoryUsage, Result, SamplerDesc, SamplerHandle,
    SlotMap, TextureDesc, TextureHandle, UsageError, marker,
};

use crate::instance::VulkanShared;

/// Shared logical device state — owned by `Arc` so swapchains can hold a ref.
pub(crate) struct VulkanDeviceInner {
    pub shared: Arc<VulkanShared>,
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub dual_src_blend: bool,
}

// ash::Device is Send + Sync; vk::PhysicalDevice is a u64.
unsafe impl Send for VulkanDeviceInner {}
unsafe impl Sync for VulkanDeviceInner {}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    usage: BufferUsage,
    mapped: *mut u8,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

pub(crate) struct VulkanTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub extent: vk::Extent2D,
}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

/// Vulkan logical device implementing [`GpuDevice`].
pub struct VulkanDevice {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    buffers: Mutex<SlotMap<marker::Buffer, VulkanBuffer>>,
    pub(crate) textures: Mutex<SlotMap<marker::Texture, VulkanTexture>>,
    samplers: Mutex<SlotMap<marker::Sampler, vk::Sampler>>,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub(crate) fn create(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        _req: DeviceRequest,
        extra_extensions: &[*const i8],
    ) -> Result<Self> {
        let mut extensions = extra_extensions.to_vec();
        let supports_portability = unsafe {
            shared
                .instance
                .enumerate_device_extension_properties(physical)
        }
        .is_ok_and(|available| {
            available.iter().any(|extension| unsafe {
                std::ffi::CStr::from_ptr(extension.extension_name.as_ptr())
                    == ash::khr::portability_subset::NAME
            })
        });
        if supports_portability {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let queue_family = compute_queue_family(&shared.instance, physical)
            .ok_or_else(|| GpuError::Backend("no compute queue family".to_string()))?;

        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let supported_features =
            unsafe { shared.instance.get_physical_device_features(physical) };
        let dual_src_blend = supported_features.dual_src_blend == vk::TRUE;
        let features = vk::PhysicalDeviceFeatures {
            shader_sampled_image_array_dynamic_indexing: vk::TRUE,
            dual_src_blend: if dual_src_blend { vk::TRUE } else { vk::FALSE },
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: if extensions.is_empty() {
                std::ptr::null()
            } else {
                extensions.as_ptr()
            },
            p_enabled_features: &features,
            ..Default::default()
        };

        let device = unsafe {
            shared
                .instance
                .create_device(physical, &device_create_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateDevice: {e}")))?
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            inner: Arc::new(VulkanDeviceInner {
                shared,
                device,
                physical,
                queue_family,
                queue,
                dual_src_blend,
            }),
            buffers: Mutex::new(SlotMap::new()),
            textures: Mutex::new(SlotMap::new()),
            samplers: Mutex::new(SlotMap::new()),
        })
    }

    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(shared, physical, req, &[])
    }

    pub(crate) fn new_with_swapchain(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(
            shared,
            physical,
            req,
            &[ash::khr::swapchain::NAME.as_ptr()],
        )
    }

    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        let mem_props = unsafe {
            self.inner
                .shared
                .instance
                .get_physical_device_memory_properties(self.inner.physical)
        };
        (0..mem_props.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
        })
    }

    /// Submit a one-shot command buffer that records work via `f`, then waits
    /// for completion.  Used for staging uploads (G3) and layout transitions.
    pub(crate) fn one_shot_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: self.inner.queue_family,
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            ..Default::default()
        };
        let pool = unsafe {
            self.inner
                .device
                .create_command_pool(&pool_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateCommandPool: {e}")))?
        };

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let cmd = unsafe {
            match self.inner.device.allocate_command_buffers(&alloc_info) {
                Ok(v) => v[0],
                Err(e) => {
                    self.inner.device.destroy_command_pool(pool, None);
                    return Err(GpuError::Backend(format!("vkAllocateCommandBuffers: {e}")));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        if let Err(e) = unsafe { self.inner.device.begin_command_buffer(cmd, &begin_info) } {
            unsafe { self.inner.device.destroy_command_pool(pool, None) };
            return Err(GpuError::Backend(format!("vkBeginCommandBuffer: {e}")));
        }

        let record_result = record(&self.inner.device, cmd);

        unsafe {
            let _ = self.inner.device.end_command_buffer(cmd);
        }

        if let Err(e) = record_result {
            unsafe { self.inner.device.destroy_command_pool(pool, None) };
            return Err(e);
        }

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        let submit_result = unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::Backend(format!("vkQueueSubmit: {e}")))
        };
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            self.inner.device.destroy_command_pool(pool, None);
        }

        submit_result
    }

    /// Raw `vk::ImageView` for a HAL texture handle. For user-side pipelines that
    /// need to bind textures into descriptor sets (e.g. bindless arrays).
    pub fn texture_view(&self, handle: TextureHandle) -> Option<vk::ImageView> {
        self.textures.lock().unwrap().get(handle).map(|t| t.view)
    }

    /// Raw `vk::Sampler` for a HAL sampler handle.
    pub fn sampler_vk(&self, handle: SamplerHandle) -> Option<vk::Sampler> {
        self.samplers.lock().unwrap().get(handle).map(|s| *s)
    }
}

fn filter_to_vk(f: FilterMode) -> vk::Filter {
    match f {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

fn address_to_vk(a: AddressMode) -> vk::SamplerAddressMode {
    match a {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

pub(crate) fn hal_format_to_vk(format: Format) -> vk::Format {
    match format {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
        Format::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
    }
}

fn compute_queue_family(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(physical) }
        .into_iter()
        .enumerate()
        .find_map(|(i, f)| {
            if f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                Some(i as u32)
            } else {
                None
            }
        })
}

fn buffer_usage_to_vk(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsage::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(BufferUsage::TRANSFER_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::TRANSFER_DST) || usage.contains(BufferUsage::READBACK) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if flags.is_empty() {
        flags = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    }
    flags
}

fn memory_usage_to_vk(usage: MemoryUsage) -> vk::MemoryPropertyFlags {
    match usage {
        MemoryUsage::GpuOnly | MemoryUsage::Pooled => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryUsage::Upload
        | MemoryUsage::CpuToGpu
        | MemoryUsage::Transient
        | MemoryUsage::Persistent => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUsage::Readback => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
        }
    }
}

fn stale(handle: BufferHandle, buffers: &SlotMap<marker::Buffer, VulkanBuffer>) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: buffers.generation_at(handle.index()).unwrap_or(u32::MAX),
    })
}

impl GpuDevice for VulkanDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let vk_usage = buffer_usage_to_vk(desc.usage);
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size,
            usage: vk_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            self.inner
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateBuffer: {e}")))?
        };

        let mem_reqs =
            unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };

        let preferred_flags = memory_usage_to_vk(desc.memory);
        let type_index = self
            .find_memory_type(mem_reqs.memory_type_bits, preferred_flags)
            .or_else(|| {
                if preferred_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
                    self.find_memory_type(
                        mem_reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                } else {
                    None
                }
            });

        let type_index = match type_index {
            Some(i) => i,
            None => {
                unsafe { self.inner.device.destroy_buffer(buffer, None) };
                return Err(GpuError::OutOfMemory(desc.memory));
            }
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index: type_index,
            ..Default::default()
        };

        let memory = unsafe {
            match self.inner.device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(_) => {
                    self.inner.device.destroy_buffer(buffer, None);
                    return Err(GpuError::OutOfMemory(desc.memory));
                }
            }
        };

        if let Err(e) = unsafe {
            self.inner.device.bind_buffer_memory(buffer, memory, 0)
        } {
            unsafe {
                self.inner.device.destroy_buffer(buffer, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::Backend(format!("vkBindBufferMemory: {e}")));
        }

        let is_host_visible = preferred_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            || matches!(desc.memory, MemoryUsage::Readback);

        let mapped = if is_host_visible {
            match unsafe {
                self.inner.device.map_memory(
                    memory,
                    0,
                    desc.size,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(ptr) => ptr as *mut u8,
                Err(e) => {
                    unsafe {
                        self.inner.device.destroy_buffer(buffer, None);
                        self.inner.device.free_memory(memory, None);
                    }
                    return Err(GpuError::Backend(format!("vkMapMemory: {e}")));
                }
            }
        } else {
            std::ptr::null_mut()
        };

        Ok(self.buffers.lock().unwrap().insert(VulkanBuffer {
            buffer,
            memory,
            size: desc.size,
            usage: desc.usage,
            mapped,
        }))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;

        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "write_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(data.len()).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..{end} exceeds buffer size {}",
                buf.size
            ))));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.mapped.add(start), data.len());
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;

        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "read_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(len as usize).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..{end} exceeds buffer size {}",
                buf.size
            ))));
        }
        let mut out = vec![0u8; len as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.mapped.add(start), out.as_mut_ptr(), len as usize);
        }
        Ok(out)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buf) = buffers.remove(buffer) {
            unsafe {
                if !buf.mapped.is_null() {
                    self.inner.device.unmap_memory(buf.memory);
                }
                self.inner.device.destroy_buffer(buf.buffer, None);
                self.inner.device.free_memory(buf.memory, None);
            }
        }
    }

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        let format = hal_format_to_vk(desc.format);
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D { width: desc.width, height: desc.height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let image = unsafe {
            self.inner.device.create_image(&image_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateImage: {e}")))?
        };
        let mem_reqs = unsafe { self.inner.device.get_image_memory_requirements(image) };
        let type_index = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let type_index = match type_index {
            Some(i) => i,
            None => {
                unsafe { self.inner.device.destroy_image(image, None) };
                return Err(GpuError::OutOfMemory(MemoryUsage::GpuOnly));
            }
        };
        let memory = unsafe {
            match self.inner.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: mem_reqs.size,
                    memory_type_index: type_index,
                    ..Default::default()
                },
                None,
            ) {
                Ok(m) => m,
                Err(_) => {
                    self.inner.device.destroy_image(image, None);
                    return Err(GpuError::OutOfMemory(MemoryUsage::GpuOnly));
                }
            }
        };
        if let Err(e) = unsafe { self.inner.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                self.inner.device.destroy_image(image, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::Backend(format!("vkBindImageMemory: {e}")));
        }
        let view = unsafe {
            match self.inner.device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.inner.device.destroy_image(image, None);
                    self.inner.device.free_memory(memory, None);
                    return Err(GpuError::Backend(format!("vkCreateImageView: {e}")));
                }
            }
        };
        Ok(self.textures.lock().unwrap().insert(VulkanTexture {
            image,
            view,
            memory,
            extent: vk::Extent2D { width: desc.width, height: desc.height },
        }))
    }

    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        let (image, extent) = {
            let textures = self.textures.lock().unwrap();
            let tex = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("upload_texture_data: stale texture handle".to_string())
            })?;
            (tex.image, tex.extent)
        };

        let staging = self.create_buffer(zengpu_hal::BufferDesc {
            size: data.len() as u64,
            usage: zengpu_hal::BufferUsage::TRANSFER_SRC,
            memory: MemoryUsage::Upload,
        })?;
        self.write_buffer(staging, 0, data)?;

        let staging_vk = {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(staging).map(|b| b.buffer).unwrap()
        };

        self.one_shot_submit(|dev, cmd| {
            let to_transfer = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                dev.cmd_copy_buffer_to_image(
                    cmd,
                    staging_vk,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    }],
                );
            }
            let to_shader = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_shader],
                );
            }
            Ok(())
        })?;

        self.destroy_buffer(staging);
        Ok(())
    }

    fn destroy_texture(&self, texture: TextureHandle) {
        let mut textures = self.textures.lock().unwrap();
        if let Some(tex) = textures.remove(texture) {
            unsafe {
                self.inner.device.destroy_image_view(tex.view, None);
                self.inner.device.destroy_image(tex.image, None);
                self.inner.device.free_memory(tex.memory, None);
            }
        }
    }

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle> {
        let min = filter_to_vk(desc.min_filter);
        let mag = filter_to_vk(desc.mag_filter);
        let addr = address_to_vk(desc.address);
        let info = vk::SamplerCreateInfo {
            mag_filter: mag,
            min_filter: min,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: addr,
            address_mode_v: addr,
            address_mode_w: addr,
            max_lod: vk::LOD_CLAMP_NONE,
            ..Default::default()
        };
        let sampler = unsafe {
            self.inner.device.create_sampler(&info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateSampler: {e}")))?
        };
        Ok(self.samplers.lock().unwrap().insert(sampler))
    }

    fn destroy_sampler(&self, sampler: SamplerHandle) {
        let mut samplers = self.samplers.lock().unwrap();
        if let Some(s) = samplers.remove(sampler) {
            unsafe { self.inner.device.destroy_sampler(s, None) };
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let mut buffers = self.buffers.lock().unwrap();
        for buf in buffers.drain() {
            unsafe {
                if !buf.mapped.is_null() {
                    self.inner.device.unmap_memory(buf.memory);
                }
                self.inner.device.destroy_buffer(buf.buffer, None);
                self.inner.device.free_memory(buf.memory, None);
            }
        }
        let mut textures = self.textures.lock().unwrap();
        for tex in textures.drain() {
            unsafe {
                self.inner.device.destroy_image_view(tex.view, None);
                self.inner.device.destroy_image(tex.image, None);
                self.inner.device.free_memory(tex.memory, None);
            }
        }
        let mut samplers = self.samplers.lock().unwrap();
        for s in samplers.drain() {
            unsafe { self.inner.device.destroy_sampler(s, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};
    use crate::instance::VulkanInstance;

    fn try_device() -> Option<Box<dyn GpuDevice>> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    fn rw_desc(size: u64) -> BufferDesc {
        BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [1, 2, 3, 4]);
        assert_eq!(dev.read_buffer(h, 2, 2).unwrap(), [3, 4]);
    }

    #[test]
    fn read_without_readback_usage_fails() {
        let Some(dev) = try_device() else { return };
        let h = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage { needed: "READBACK", .. })
        ));
    }

    #[test]
    fn use_after_destroy_is_stale() {
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.destroy_buffer(h);
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::StaleHandle { .. })
        ));
    }

    #[test]
    fn out_of_bounds_write_fails() {
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));
    }

    #[test]
    fn reports_graphics_and_compute() {
        let Some(dev) = try_device() else { return };
        assert!(dev.capabilities().graphics);
        assert!(dev.capabilities().compute);
    }
}
