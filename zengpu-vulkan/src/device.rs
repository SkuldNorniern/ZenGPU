//! Vulkan logical device and buffer operations.

use std::sync::{Arc, Mutex};

use ash::{Device, khr, vk};
use zengpu_hal::{
    AddressMode, Bindings, BlendMode, BufferDesc, BufferHandle, BufferUsage, ComputePipelineDesc,
    DeviceRequest, FilterMode, Format, GpuDevice, GpuError, GraphicsDevice, GraphicsPipelineDesc,
    HalCapabilities, MemoryUsage, PipelineHandle, PrimitiveTopology, Result, SamplerDesc,
    SamplerHandle, Scalar, ShaderDesc, ShaderHandle, SlotMap, StepMode, SurfaceConfig,
    TargetHandle, TextureDesc, TextureHandle, TextureUsage, UsageError, VertexFormat,
    WindowHandles, marker,
};

use crate::command_list::{CmdListPool, VulkanCommandList};
use crate::instance::VulkanShared;
use crate::surface::VulkanSurface;

/// Maximum number of storage buffers in the bindless descriptor table.
const MAX_BINDLESS_BUFFERS: u32 = 4096;

/// Maximum number of combined image samplers in the bindless texture table.
const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Descriptor pool + layout + set for the bindless SSBO table.
struct BindlessState {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

// SAFETY: the contained Vulkan handles are all u64 values; we control their
// lifetimes through VulkanDevice (which is Send+Sync).
unsafe impl Send for BindlessState {}
unsafe impl Sync for BindlessState {}

/// A compute or graphics pipeline. Both kinds share one [`PipelineHandle`]
/// slotmap (the HAL has a single `PipelineHandle` type for both).
pub(crate) enum VulkanPipeline {
    Compute {
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    },
    Graphics {
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    },
}

impl VulkanPipeline {
    /// The raw pipeline and layout, regardless of kind.
    pub(crate) fn handles(&self) -> (vk::Pipeline, vk::PipelineLayout) {
        match *self {
            VulkanPipeline::Compute { layout, pipeline }
            | VulkanPipeline::Graphics { layout, pipeline } => (pipeline, layout),
        }
    }
}

/// A render target registered for use as a [`RenderPassDesc`](zengpu_hal::RenderPassDesc)
/// attachment — a swapchain image or an offscreen texture. `layout` tracks the
/// image's current `vk::ImageLayout` so [`VulkanCommandList`](crate::command_list::VulkanCommandList)
/// can emit the right barriers for dynamic rendering (no render-pass objects
/// to do this automatically).
pub(crate) struct VulkanRenderTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
    #[allow(dead_code)]
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

/// Shared logical device state — owned by `Arc` so swapchains can hold a ref.
pub(crate) struct VulkanDeviceInner {
    pub shared: Arc<VulkanShared>,
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub dual_src_blend: bool,
    /// `VK_KHR_dynamic_rendering` loader — the unified graphics path (D17/GU)
    /// records render passes via `cmd_begin_rendering`/`cmd_end_rendering`,
    /// with no `vk::RenderPass`/`vk::Framebuffer` objects.
    pub dynamic_rendering: khr::dynamic_rendering::Device,
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

pub(crate) struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub usage: BufferUsage,
    pub mapped: *mut u8,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

pub(crate) struct VulkanTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

/// Vulkan logical device implementing [`GpuDevice`].
pub struct VulkanDevice {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    pub(crate) buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    pub(crate) textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    samplers: Mutex<SlotMap<marker::Sampler, vk::Sampler>>,
    shaders: Mutex<SlotMap<marker::Shader, vk::ShaderModule>>,
    pub(crate) pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    /// Render targets (swapchain images, offscreen textures) recordable
    /// commands can attach to. Shared with [`VulkanCommandList`](crate::command_list::VulkanCommandList).
    pub(crate) render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    bindless: BindlessState,
    pub(crate) cmd_list_pool: Arc<CmdListPool>,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    /// Cloneable access to the raw Vulkan device context used by render targets,
    /// frame graphs, and engine-side graphics resources.
    pub fn context(&self) -> crate::swapchain::DeviceContext {
        crate::swapchain::DeviceContext::from_inner(Arc::clone(&self.inner))
    }

    /// Wait until all work submitted to this logical device has completed.
    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.inner
                .device
                .device_wait_idle()
                .map_err(|e| GpuError::Backend(format!("device_wait_idle: {e}")))
        }
    }

    pub(crate) fn create(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        _req: DeviceRequest,
        extra_extensions: &[*const i8],
    ) -> Result<Self> {
        let mut extensions = extra_extensions.to_vec();
        let available_extensions = unsafe {
            shared
                .instance
                .enumerate_device_extension_properties(physical)
        }
        .map_err(|e| GpuError::Backend(format!("enumerate device extensions: {e}")))?;
        let supports_extension = |name: &std::ffi::CStr| {
            available_extensions.iter().any(|extension| unsafe {
                std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) == name
            })
        };
        if supports_extension(ash::khr::portability_subset::NAME) {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        // The unified graphics API (D17/GU) records render passes via
        // dynamic rendering — no vk::RenderPass/Framebuffer objects.
        if !supports_extension(khr::dynamic_rendering::NAME) {
            return Err(GpuError::Backend(
                "GPU does not support VK_KHR_dynamic_rendering".to_string(),
            ));
        }
        extensions.push(khr::dynamic_rendering::NAME.as_ptr());

        let queue_family = compute_queue_family(&shared.instance, physical)
            .ok_or_else(|| GpuError::Backend("no compute queue family".to_string()))?;

        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let supported_features = unsafe { shared.instance.get_physical_device_features(physical) };
        let dual_src_blend = supported_features.dual_src_blend == vk::TRUE;

        // Enable Vulkan 1.2 descriptor-indexing features for bindless resources.
        let mut desc_idx = vk::PhysicalDeviceDescriptorIndexingFeatures {
            shader_storage_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
            descriptor_binding_storage_buffer_update_after_bind: vk::TRUE,
            descriptor_binding_sampled_image_update_after_bind: vk::TRUE,
            descriptor_binding_partially_bound: vk::TRUE,
            runtime_descriptor_array: vk::TRUE,
            ..Default::default()
        };
        let mut dynamic_rendering_feat = vk::PhysicalDeviceDynamicRenderingFeatures {
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };
        desc_idx.p_next = &mut dynamic_rendering_feat as *mut _ as *mut std::ffi::c_void;
        let mut features2 = vk::PhysicalDeviceFeatures2 {
            features: vk::PhysicalDeviceFeatures {
                shader_sampled_image_array_dynamic_indexing: vk::TRUE,
                dual_src_blend: if dual_src_blend { vk::TRUE } else { vk::FALSE },
                ..Default::default()
            },
            ..Default::default()
        };
        features2.p_next = &mut desc_idx as *mut _ as *mut std::ffi::c_void;

        let device_create_info = vk::DeviceCreateInfo {
            p_next: &features2 as *const _ as *const std::ffi::c_void,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: if extensions.is_empty() {
                std::ptr::null()
            } else {
                extensions.as_ptr()
            },
            p_enabled_features: std::ptr::null(), // must be null when using features2 pNext
            ..Default::default()
        };

        let device = unsafe {
            shared
                .instance
                .create_device(physical, &device_create_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateDevice: {e}")))?
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let dynamic_rendering = khr::dynamic_rendering::Device::new(&shared.instance, &device);

        let inner = Arc::new(VulkanDeviceInner {
            shared,
            device,
            physical,
            queue_family,
            queue,
            dual_src_blend,
            dynamic_rendering,
        });

        let bindless = create_bindless(&inner.device)?;
        let cmd_list_pool = Arc::new(CmdListPool::new(Arc::clone(&inner))?);

        Ok(Self {
            inner,
            buffers: Arc::new(Mutex::new(SlotMap::new())),
            textures: Arc::new(Mutex::new(SlotMap::new())),
            samplers: Mutex::new(SlotMap::new()),
            shaders: Mutex::new(SlotMap::new()),
            pipelines: Arc::new(Mutex::new(SlotMap::new())),
            render_targets: Arc::new(Mutex::new(SlotMap::new())),
            bindless,
            cmd_list_pool,
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
        Self::create(shared, physical, req, &[ash::khr::swapchain::NAME.as_ptr()])
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
    /// for completion. Used for staging uploads and layout transitions.
    pub(crate) fn one_shot_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let cmd = self.cmd_list_pool.acquire()?;

        let record_result = record(&self.inner.device, cmd);

        unsafe {
            let _ = self.inner.device.end_command_buffer(cmd);
        }

        if let Err(e) = record_result {
            self.cmd_list_pool.release(cmd);
            return Err(e);
        }

        let fence = unsafe {
            self.inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| GpuError::Backend(format!("vkCreateFence: {e}")))?
        };
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        let mut submitted = false;
        let submit_result = unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &[submit_info], fence)
                .map_err(|e| GpuError::Backend(format!("vkQueueSubmit: {e}")))
        };
        if submit_result.is_ok() {
            submitted = true;
        }
        let wait_result = submit_result.and_then(|()| unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| GpuError::Backend(format!("vkWaitForFences: {e}")))
        });
        unsafe {
            self.inner.device.destroy_fence(fence, None);
        }
        if !submitted || wait_result.is_ok() {
            self.cmd_list_pool.release(cmd);
        }

        wait_result
    }

    /// Register a STORAGE buffer in the bindless SSBO table at its slot index.
    /// Called automatically by `create_buffer` for `STORAGE`-flagged buffers.
    fn bind_buffer_to_bindless(&self, slot: u32, buffer: vk::Buffer, size: u64) {
        let info = vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: if size == 0 { vk::WHOLE_SIZE } else { size },
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.bindless.set,
            dst_binding: 0,
            dst_array_element: slot,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &info,
            ..Default::default()
        };
        unsafe {
            self.inner.device.update_descriptor_sets(&[write], &[]);
        }
    }

    /// Register `texture` (sampled with `sampler`) in the bindless
    /// combined-image-sampler table, at `texture`'s own slot index, for use
    /// as a [`Bindings::textures`] index in [`RenderCommands::bind`](zengpu_hal::RenderCommands::bind).
    /// The descriptor declares `SHADER_READ_ONLY_OPTIMAL`; the image must be
    /// in that layout by the time it is sampled — true after
    /// [`GpuDevice::upload_texture_data`], or after a render pass with
    /// [`zengpu_hal::ColorAttachment::sample_after`] for a render-target texture.
    pub fn bind_texture(&self, texture: TextureHandle, sampler: SamplerHandle) -> Option<u32> {
        let view = self.textures.lock().unwrap().get(texture)?.view;
        let vk_sampler = *self.samplers.lock().unwrap().get(sampler)?;
        let info = vk::DescriptorImageInfo {
            sampler: vk_sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.bindless.set,
            dst_binding: 1,
            dst_array_element: texture.index(),
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &info,
            ..Default::default()
        };
        unsafe {
            self.inner.device.update_descriptor_sets(&[write], &[]);
        }
        Some(texture.index())
    }
}

fn create_bindless(dev: &ash::Device) -> Result<BindlessState> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_BINDLESS_BUFFERS,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_BINDLESS_TEXTURES,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
    ];
    let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND; 2];
    let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
        binding_count: binding_flags.len() as u32,
        p_binding_flags: binding_flags.as_ptr(),
        ..Default::default()
    };
    let layout = unsafe {
        dev.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                p_next: &mut flags_info as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            },
            None,
        )
        .map_err(|e| GpuError::Backend(format!("bindless layout: {e}")))?
    };

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_BINDLESS_BUFFERS,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_BINDLESS_TEXTURES,
        },
    ];
    let pool = unsafe {
        dev.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
                max_sets: 1,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            },
            None,
        )
        .map_err(|e| {
            dev.destroy_descriptor_set_layout(layout, None);
            GpuError::Backend(format!("bindless pool: {e}"))
        })?
    };

    let set = unsafe {
        dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            ..Default::default()
        })
        .map_err(|e| {
            dev.destroy_descriptor_pool(pool, None);
            dev.destroy_descriptor_set_layout(layout, None);
            GpuError::Backend(format!("bindless set alloc: {e}"))
        })?[0]
    };

    Ok(BindlessState { layout, pool, set })
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

        let mem_reqs = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };

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

        if let Err(e) = unsafe { self.inner.device.bind_buffer_memory(buffer, memory, 0) } {
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
                self.inner
                    .device
                    .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
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

        let vk_buf = buffer; // Copy before moving into struct
        let handle = self.buffers.lock().unwrap().insert(VulkanBuffer {
            buffer: vk_buf,
            memory,
            size: desc.size,
            usage: desc.usage,
            mapped,
        });
        if desc.usage.contains(BufferUsage::STORAGE) {
            self.bind_buffer_to_bindless(handle.index(), vk_buf, desc.size);
        }
        Ok(handle)
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
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("range {start}..{end} exceeds buffer size {}", buf.size),
            )));
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
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("range {start}..{end} exceeds buffer size {}", buf.size),
            )));
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
        let mut usage = vk::ImageUsageFlags::empty();
        if desc.usage.contains(TextureUsage::SAMPLED) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.contains(TextureUsage::RENDER_TARGET) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if desc.usage.contains(TextureUsage::DEPTH_STENCIL) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if desc.usage.contains(TextureUsage::TRANSFER_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(TextureUsage::TRANSFER_DST) {
            usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let image = unsafe {
            self.inner
                .device
                .create_image(&image_info, None)
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
            format,
            extent: vk::Extent2D {
                width: desc.width,
                height: desc.height,
            },
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
            self.inner
                .device
                .create_sampler(&info, None)
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

    // ── Compute ───────────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        if desc.spirv.len() % 4 != 0 {
            return Err(GpuError::ShaderCompile(
                "SPIR-V byte length must be a multiple of 4".to_string(),
            ));
        }
        let words: Vec<u32> = desc
            .spirv
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let module = unsafe {
            self.inner
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: desc.spirv.len(),
                        p_code: words.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| GpuError::ShaderCompile(format!("vkCreateShaderModule: {e}")))?
        };
        Ok(self.shaders.lock().unwrap().insert(module))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let mut shaders = self.shaders.lock().unwrap();
        if let Some(m) = shaders.remove(shader) {
            unsafe {
                self.inner.device.destroy_shader_module(m, None);
            }
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let shader_module = {
            let shaders = self.shaders.lock().unwrap();
            *shaders
                .get(desc.shader)
                .ok_or_else(|| GpuError::PipelineCreation("stale shader handle".to_string()))?
        };

        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 128, // 32 u32 slots: buffer indices + scalars
        };
        let layout = unsafe {
            self.inner
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        set_layout_count: 1,
                        p_set_layouts: &self.bindless.layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &pc_range,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| GpuError::PipelineCreation(format!("vkCreatePipelineLayout: {e}")))?
        };

        let entry = std::ffi::CString::new(desc.entry)
            .map_err(|e| GpuError::PipelineCreation(format!("entry name nul: {e}")))?;
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        };
        let result = unsafe {
            self.inner.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo {
                    stage,
                    layout,
                    ..Default::default()
                }],
                None,
            )
        };
        match result {
            Ok(pipelines) => Ok(self
                .pipelines
                .lock()
                .unwrap()
                .insert(VulkanPipeline::Compute {
                    layout,
                    pipeline: pipelines[0],
                })),
            Err((_, e)) => {
                unsafe {
                    self.inner.device.destroy_pipeline_layout(layout, None);
                }
                Err(GpuError::PipelineCreation(format!(
                    "vkCreateComputePipelines: {e}"
                )))
            }
        }
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(p) = pipelines.remove(pipeline) {
            let (pipeline, layout) = p.handles();
            unsafe {
                self.inner.device.destroy_pipeline(pipeline, None);
                self.inner.device.destroy_pipeline_layout(layout, None);
            }
        }
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        if grid.contains(&0) {
            return Err(GpuError::Dispatch(format!(
                "dispatch grid dimensions must be non-zero, got {grid:?}"
            )));
        }

        let (vk_pipeline, vk_layout) = {
            let pipelines = self.pipelines.lock().unwrap();
            let p = pipelines
                .get(pipeline)
                .ok_or_else(|| GpuError::Dispatch("stale pipeline handle".to_string()))?;
            match p {
                VulkanPipeline::Compute { layout, pipeline } => (*pipeline, *layout),
                VulkanPipeline::Graphics { .. } => {
                    return Err(GpuError::Dispatch(
                        "dispatch called with a graphics pipeline handle".to_string(),
                    ));
                }
            }
        };

        // Pack push constants: [buffer_indices, scalars], each as 4 bytes.
        let mut pc = [0u8; 128];
        let mut pc_len = 0usize;
        let mut push_pc = |bytes: [u8; 4]| -> Result<()> {
            if pc_len + 4 > pc.len() {
                return Err(GpuError::Dispatch(format!(
                    "push constants exceed {} bytes",
                    pc.len()
                )));
            }
            pc[pc_len..pc_len + 4].copy_from_slice(&bytes);
            pc_len += 4;
            Ok(())
        };
        for &idx in bindings.buffers {
            push_pc(idx.to_ne_bytes())?;
        }
        for scalar in bindings.scalars {
            push_pc(match scalar {
                Scalar::U32(v) => v.to_ne_bytes(),
                Scalar::I32(v) => v.to_ne_bytes(),
                Scalar::F32(v) => v.to_bits().to_ne_bytes(),
            })?;
        }

        let bindless_set = self.bindless.set;
        self.one_shot_submit(move |dev, cmd| {
            unsafe {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    vk_layout,
                    0,
                    &[bindless_set],
                    &[],
                );
                if pc_len != 0 {
                    dev.cmd_push_constants(
                        cmd,
                        vk_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &pc[..pc_len],
                    );
                }
                dev.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
            }
            Ok(())
        })
    }
}

impl VulkanDevice {
    /// Create a graphics pipeline via `VK_KHR_dynamic_rendering` — no
    /// `vk::RenderPass`/`vk::Framebuffer` objects. Part of the unified
    /// graphics API (D17/GU); see [`zengpu_hal::GraphicsDevice::create_graphics_pipeline`].
    pub(crate) fn create_graphics_pipeline_impl(
        &self,
        desc: GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle> {
        let (vert_module, frag_module) = {
            let shaders = self.shaders.lock().unwrap();
            let vert = *shaders.get(desc.vertex_shader).ok_or_else(|| {
                GpuError::PipelineCreation("stale vertex shader handle".to_string())
            })?;
            let frag = *shaders.get(desc.fragment_shader).ok_or_else(|| {
                GpuError::PipelineCreation("stale fragment shader handle".to_string())
            })?;
            (vert, frag)
        };

        let entry = std::ffi::CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
        ];

        // One binding per vertex layout; binding index = slice position, which
        // is the `slot` passed to set_vertex_buffer. Attributes carry the binding
        // of the layout they belong to so multiple streams (e.g. per-vertex quad
        // + per-instance data) coexist.
        let bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_layouts
            .iter()
            .enumerate()
            .map(|(i, layout)| vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: layout.stride,
                input_rate: step_mode_to_vk(layout.step_mode),
            })
            .collect();
        let attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_layouts
            .iter()
            .enumerate()
            .flat_map(|(i, layout)| {
                layout
                    .attributes
                    .iter()
                    .map(move |a| vk::VertexInputAttributeDescription {
                        location: a.location,
                        binding: i as u32,
                        format: vertex_format_to_vk(a.format),
                        offset: a.offset,
                    })
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: bindings.len() as u32,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: attributes.len() as u32,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: topology_to_vk(desc.topology),
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterization = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: sample_count_to_vk(desc.samples),
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: if desc.depth.test { vk::TRUE } else { vk::FALSE },
            depth_write_enable: if desc.depth.write {
                vk::TRUE
            } else {
                vk::FALSE
            },
            depth_compare_op: vk::CompareOp::LESS,
            ..Default::default()
        };

        let blend_att = blend_mode_to_vk(desc.blend);
        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &blend_att,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let color_format = hal_format_to_vk(desc.color_format);
        let depth_format = desc
            .depth_format
            .map(hal_format_to_vk)
            .unwrap_or(vk::Format::UNDEFINED);
        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 1,
            p_color_attachment_formats: &color_format,
            depth_attachment_format: depth_format,
            ..Default::default()
        };

        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 128, // 32 u32 slots: buffer indices + scalars
        };
        let layout = unsafe {
            self.inner
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        set_layout_count: 1,
                        p_set_layouts: &self.bindless.layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &pc_range,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| GpuError::PipelineCreation(format!("vkCreatePipelineLayout: {e}")))?
        };

        let create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &rendering_info as *const _ as *const std::ffi::c_void,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout,
            ..Default::default()
        };

        let result = unsafe {
            self.inner.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        };
        match result {
            Ok(pipelines) => Ok(self
                .pipelines
                .lock()
                .unwrap()
                .insert(VulkanPipeline::Graphics {
                    layout,
                    pipeline: pipelines[0],
                })),
            Err((_, e)) => {
                unsafe {
                    self.inner.device.destroy_pipeline_layout(layout, None);
                }
                Err(GpuError::PipelineCreation(format!(
                    "vkCreateGraphicsPipelines: {e}"
                )))
            }
        }
    }

    /// Acquire a pooled, reset-reusable [`VulkanCommandList`] and begin
    /// recording. Part of the unified graphics API (D17/GU); see
    /// [`zengpu_hal::GraphicsDevice::create_command_list`].
    pub(crate) fn create_command_list_impl(&self) -> Result<VulkanCommandList> {
        let cmd = self.cmd_list_pool.acquire()?;
        Ok(VulkanCommandList::new(
            Arc::clone(&self.inner),
            Arc::clone(&self.cmd_list_pool),
            cmd,
            Arc::clone(&self.pipelines),
            Arc::clone(&self.render_targets),
            Arc::clone(&self.buffers),
            self.bindless.set,
        ))
    }

    /// Register `depth`'s image/view as a render target for use as
    /// [`zengpu_hal::DepthAttachment::target`]. Call again with the recreated
    /// [`crate::DepthTarget`] after a resize and use [`unregister_render_target`](Self::unregister_render_target)
    /// to drop the stale handle.
    pub fn register_depth_target(&self, depth: &crate::depth_target::DepthTarget) -> TargetHandle {
        let (width, height) = depth.extent();
        self.render_targets
            .lock()
            .unwrap()
            .insert(VulkanRenderTarget {
                image: depth.image(),
                view: depth.view(),
                format: crate::depth_target::DEPTH_FORMAT,
                extent: vk::Extent2D { width, height },
                layout: vk::ImageLayout::UNDEFINED,
            })
    }

    /// Register `texture`'s image/view as a render target for use as
    /// [`zengpu_hal::ColorAttachment::target`]. `texture` must have been
    /// created with [`TextureUsage::RENDER_TARGET`]. Use
    /// [`unregister_render_target`](Self::unregister_render_target) to drop
    /// the handle when the texture is destroyed. Returns `None` for a stale
    /// `texture` handle.
    pub fn register_color_target(&self, texture: TextureHandle) -> Option<TargetHandle> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(texture)?;
        Some(
            self.render_targets
                .lock()
                .unwrap()
                .insert(VulkanRenderTarget {
                    image: tex.image,
                    view: tex.view,
                    format: tex.format,
                    extent: tex.extent,
                    layout: vk::ImageLayout::UNDEFINED,
                }),
        )
    }

    /// Drop a render-target registration created by [`register_depth_target`](Self::register_depth_target)
    /// or [`register_color_target`](Self::register_color_target).
    pub fn unregister_render_target(&self, handle: TargetHandle) {
        self.render_targets.lock().unwrap().remove(handle);
    }
}

impl GraphicsDevice for VulkanDevice {
    type Surface = VulkanSurface;
    type CommandList = VulkanCommandList;

    fn create_surface(
        &self,
        window: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self::Surface> {
        VulkanSurface::new(self, window, config)
    }

    fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc<'_>) -> Result<PipelineHandle> {
        self.create_graphics_pipeline_impl(desc)
    }

    fn create_command_list(&self) -> Result<Self::CommandList> {
        self.create_command_list_impl()
    }

    fn supports_dual_source_blending(&self) -> bool {
        self.inner.dual_src_blend
    }
}

fn vertex_format_to_vk(f: VertexFormat) -> vk::Format {
    match f {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Uint8x4Unorm => vk::Format::R8G8B8A8_UNORM,
    }
}

fn step_mode_to_vk(s: StepMode) -> vk::VertexInputRate {
    match s {
        StepMode::Vertex => vk::VertexInputRate::VERTEX,
        StepMode::Instance => vk::VertexInputRate::INSTANCE,
    }
}

fn topology_to_vk(t: PrimitiveTopology) -> vk::PrimitiveTopology {
    match t {
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
    }
}

fn blend_mode_to_vk(b: BlendMode) -> vk::PipelineColorBlendAttachmentState {
    match b {
        BlendMode::Opaque => vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        },
        BlendMode::AlphaBlend => vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        },
        BlendMode::DualSourceAlpha => vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC1_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::SRC1_ALPHA,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        },
    }
}

fn sample_count_to_vk(samples: u32) -> vk::SampleCountFlags {
    match samples {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
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
        let mut shaders = self.shaders.lock().unwrap();
        for m in shaders.drain() {
            unsafe {
                self.inner.device.destroy_shader_module(m, None);
            }
        }
        let mut pipelines = self.pipelines.lock().unwrap();
        for p in pipelines.drain() {
            let (pipeline, layout) = p.handles();
            unsafe {
                self.inner.device.destroy_pipeline(pipeline, None);
                self.inner.device.destroy_pipeline_layout(layout, None);
            }
        }
        unsafe {
            // Pool destruction also frees all descriptor sets from the pool.
            self.inner
                .device
                .destroy_descriptor_pool(self.bindless.pool, None);
            self.inner
                .device
                .destroy_descriptor_set_layout(self.bindless.layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::VulkanInstance;
    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};

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
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "READBACK",
                ..
            })
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
