//! Vulkan logical device and buffer operations.

use std::{
    any::Any,
    ffi::{CStr, CString, c_void},
    ptr::{copy_nonoverlapping, null, null_mut},
    sync::{Arc, Mutex},
    time::Duration,
};

use ash::{Device, ext, khr, vk};
use zengpu_hal::{
    Bindings, BufferDesc, BufferHandle, BufferUsage, CompareFn, ComputeOp, ComputePipelineDesc,
    DeviceLimits, DeviceRequest, DispatchOp, Features, FilterMode, GpuDevice, GpuError,
    GpuSubmission, GraphicsDevice, GraphicsPipelineDesc, HalCapabilities, MemoryUsage,
    PipelineHandle, PolygonMode, Result, SamplerDesc, SamplerHandle, Scalar, ShaderDesc,
    ShaderHandle, ShaderSource, SlotMap, Submission, SubmissionStatus, SurfaceConfig,
    TargetHandle, TexDim, TextureDesc, TextureHandle, TextureUsage, UsageError, WindowHandles,
    marker,
};

mod dispatch;
mod format;
mod graphics;
mod hal_impl;
mod resources;

use format::*;

use crate::command_list::{COLOR_SUBRESOURCE, CmdListPool, VulkanCommandList};
use crate::depth_target::{DEPTH_FORMAT, DepthTarget};
use crate::instance::VulkanShared;
use crate::offscreen::OffscreenTarget;
use crate::surface::VulkanSurface;
use crate::swapchain::{DeviceContext, map_vk_err};

/// Maximum number of storage buffers in the bindless descriptor table.
const MAX_BINDLESS_BUFFERS: u32 = 4096;

/// Maximum number of combined image samplers in the bindless texture table.
const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Vulkan guarantees at least 128 bytes of push constants. Keeping compute's
/// ABI at that portable floor avoids rejecting otherwise conforming devices.
const COMPUTE_PUSH_CONSTANT_BYTES: usize = 128;

/// Descriptor pool + layout + set for the bindless SSBO table.
struct BindlessState {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    buffer_capacity: u32,
    texture_capacity: u32,
    bound_textures: Mutex<Vec<bool>>,
}

// SAFETY: the contained Vulkan handles are all u64 values; we control their
// lifetimes through VulkanDevice (which is Send+Sync).
unsafe impl Send for BindlessState {}
unsafe impl Sync for BindlessState {}

struct FencePool {
    inner: Arc<VulkanDeviceInner>,
    free: Mutex<Vec<vk::Fence>>,
}

struct VulkanSubmissionState {
    fence: Option<vk::Fence>,
    cmd: Option<vk::CommandBuffer>,
}

enum DeferredVulkanResource {
    Buffer(u32, VulkanBuffer),
    Texture(u32, VulkanTexture),
    Sampler(u32, vk::Sampler),
    Pipeline(u32, VulkanPipeline),
}

#[derive(Default)]
struct VulkanLifetimeState {
    in_flight: usize,
    deferred: Vec<DeferredVulkanResource>,
}

/// Fence-backed Vulkan submission. Completion releases its pooled fence and
/// command buffer exactly once; a timed-out wait leaves both owned by this
/// handle so a later poll/wait is safe.
struct VulkanSubmission {
    cycle_id: u64,
    inner: Arc<VulkanDeviceInner>,
    fence_pool: Arc<FencePool>,
    cmd_pool: Arc<CmdListPool>,
    buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    samplers: Arc<Mutex<SlotMap<marker::Sampler, vk::Sampler>>>,
    pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    lifetime: Arc<Mutex<VulkanLifetimeState>>,
    state: Mutex<VulkanSubmissionState>,
}

impl VulkanSubmission {
    fn complete_locked(&self, state: &mut VulkanSubmissionState) {
        if let Some(fence) = state.fence.take() {
            self.fence_pool.release(fence);
        }
        if let Some(cmd) = state.cmd.take() {
            self.cmd_pool.release(cmd);
        }
        self.release_in_flight();
    }

    fn timeout_ns(timeout: Duration) -> u64 {
        timeout.as_nanos().min(u128::from(u64::MAX)) as u64
    }

    fn release_in_flight(&self) {
        let deferred = {
            let mut lifetime = self.lifetime.lock().unwrap();
            debug_assert!(lifetime.in_flight > 0);
            lifetime.in_flight = lifetime.in_flight.saturating_sub(1);
            if lifetime.in_flight == 0 {
                std::mem::take(&mut lifetime.deferred)
            } else {
                Vec::new()
            }
        };
        destroy_deferred_vulkan(
            &self.inner,
            &self.buffers,
            &self.textures,
            &self.samplers,
            &self.pipelines,
            deferred,
        );
    }
}

impl GpuSubmission for VulkanSubmission {
    fn cycle_id(&self) -> u64 {
        self.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        let mut state = self.state.lock().unwrap();
        let Some(fence) = state.fence else {
            return Ok(SubmissionStatus::Complete);
        };
        let signaled = unsafe {
            self.inner
                .device
                .get_fence_status(fence)
                .map_err(|e| map_vk_err("vkGetFenceStatus", e))?
        };
        if signaled {
            self.complete_locked(&mut state);
            Ok(SubmissionStatus::Complete)
        } else {
            Ok(SubmissionStatus::Pending)
        }
    }

    fn wait(&self, timeout: Duration) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let Some(fence) = state.fence else {
            return Ok(());
        };
        let result = unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, Self::timeout_ns(timeout))
        };
        match result {
            Ok(()) => {
                self.complete_locked(&mut state);
                Ok(())
            }
            Err(vk::Result::TIMEOUT) => Err(GpuError::Timeout),
            Err(e) => Err(map_vk_err("vkWaitForFences", e)),
        }
    }
}

impl Drop for VulkanSubmission {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap();
        let Some(fence) = state.fence else {
            return;
        };
        // Dropping a pending token must not recycle resources still in use by
        // the GPU. A caller that needs a non-blocking control thread retains
        // timed-out tokens and reaps them away from that thread.
        let waited = unsafe { self.inner.device.wait_for_fences(&[fence], true, u64::MAX) };
        if waited.is_ok() {
            if let Some(fence) = state.fence.take() {
                self.fence_pool.release(fence);
            }
            if let Some(cmd) = state.cmd.take() {
                self.cmd_pool.release(cmd);
            }
        } else {
            // A device-lost/error path cannot safely recycle these objects.
            // Destroying the fence is legal after device loss; the command
            // pool remains alive and is reclaimed with the device.
            if let Some(fence) = state.fence.take() {
                unsafe { self.inner.device.destroy_fence(fence, None) };
            }
            state.cmd.take();
        }
        self.release_in_flight();
    }
}

impl FencePool {
    fn new(inner: Arc<VulkanDeviceInner>) -> Self {
        Self {
            inner,
            free: Mutex::new(Vec::new()),
        }
    }

    fn acquire(&self) -> Result<vk::Fence> {
        let fence = self.free.lock().unwrap().pop();
        match fence {
            Some(fence) => {
                unsafe {
                    self.inner
                        .device
                        .reset_fences(&[fence])
                        .map_err(|e| GpuError::Backend(format!("vkResetFences: {e}")))?;
                }
                Ok(fence)
            }
            None => unsafe {
                self.inner
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| GpuError::Backend(format!("vkCreateFence: {e}")))
            },
        }
    }

    fn release(&self, fence: vk::Fence) {
        self.free.lock().unwrap().push(fence);
    }
}

impl Drop for FencePool {
    fn drop(&mut self) {
        let mut free = self.free.lock().unwrap();
        for fence in free.drain(..) {
            unsafe {
                self.inner.device.destroy_fence(fence, None);
            }
        }
    }
}

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
/// image's current `vk::ImageLayout` so [`VulkanCommandList`](VulkanCommandList)
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
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// Vulkan queues require external host synchronization. Every internal
    /// submit/present operation using `queue` must hold this mutex.
    pub queue_lock: Mutex<()>,
    pub features: Features,
    pub limits: DeviceLimits,
    pub graphics: bool,
    pub dual_src_blend: bool,
    pub fill_mode_non_solid: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
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
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub usage: vk::ImageUsageFlags,
}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

fn destroy_deferred_vulkan(
    inner: &VulkanDeviceInner,
    buffers: &Mutex<SlotMap<marker::Buffer, VulkanBuffer>>,
    textures: &Mutex<SlotMap<marker::Texture, VulkanTexture>>,
    samplers: &Mutex<SlotMap<marker::Sampler, vk::Sampler>>,
    pipelines: &Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>,
    deferred: Vec<DeferredVulkanResource>,
) {
    for resource in deferred {
        match resource {
            DeferredVulkanResource::Buffer(index, buffer) => {
                unsafe {
                    if !buffer.mapped.is_null() {
                        inner.device.unmap_memory(buffer.memory);
                    }
                    inner.device.destroy_buffer(buffer.buffer, None);
                    inner.device.free_memory(buffer.memory, None);
                }
                buffers.lock().unwrap().release_retired(index);
            }
            DeferredVulkanResource::Texture(index, texture) => {
                unsafe {
                    inner.device.destroy_image_view(texture.view, None);
                    inner.device.destroy_image(texture.image, None);
                    inner.device.free_memory(texture.memory, None);
                }
                textures.lock().unwrap().release_retired(index);
            }
            DeferredVulkanResource::Sampler(index, sampler) => {
                unsafe { inner.device.destroy_sampler(sampler, None) };
                samplers.lock().unwrap().release_retired(index);
            }
            DeferredVulkanResource::Pipeline(index, pipeline) => {
                let (pipeline, layout) = pipeline.handles();
                unsafe {
                    inner.device.destroy_pipeline(pipeline, None);
                    inner.device.destroy_pipeline_layout(layout, None);
                }
                pipelines.lock().unwrap().release_retired(index);
            }
        }
    }
}

/// Vulkan logical device implementing [`GpuDevice`].
pub struct VulkanDevice {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    pub(crate) buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    pub(crate) textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    samplers: Arc<Mutex<SlotMap<marker::Sampler, vk::Sampler>>>,
    shaders: Mutex<SlotMap<marker::Shader, vk::ShaderModule>>,
    pub(crate) pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    /// Render targets (swapchain images, offscreen textures) recordable
    /// commands can attach to. Shared with [`VulkanCommandList`](VulkanCommandList).
    pub(crate) render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    bindless: BindlessState,
    pipeline_cache: vk::PipelineCache,
    pub(crate) cmd_list_pool: Arc<CmdListPool>,
    fence_pool: Arc<FencePool>,
    lifetime: Arc<Mutex<VulkanLifetimeState>>,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub(crate) fn create(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
        extra_extensions: &[*const i8],
        needs_graphics: bool,
    ) -> Result<Self> {
        let mut extensions = extra_extensions.to_vec();
        let available_extensions = unsafe {
            shared
                .instance
                .enumerate_device_extension_properties(physical)
        }
        .map_err(|e| GpuError::Backend(format!("enumerate device extensions: {e}")))?;
        let supports_extension = |name: &CStr| {
            available_extensions.iter().any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr()) == name
            })
        };
        for &extension in extra_extensions {
            let name = unsafe { CStr::from_ptr(extension) };
            if !supports_extension(name) {
                return Err(GpuError::Backend(format!(
                    "required Vulkan device extension is unavailable: {}",
                    name.to_string_lossy()
                )));
            }
        }
        if supports_extension(khr::portability_subset::NAME) {
            extensions.push(khr::portability_subset::NAME.as_ptr());
        }
        // Dynamic rendering is a graphics-only requirement. Headless compute
        // devices must not be rejected for an unrelated extension.
        if needs_graphics {
            if !supports_extension(khr::dynamic_rendering::NAME) {
                return Err(GpuError::Backend(
                    "GPU does not support VK_KHR_dynamic_rendering".to_string(),
                ));
            }
            extensions.push(khr::dynamic_rendering::NAME.as_ptr());
        }
        let shader_atomic_float_extension = supports_extension(ext::shader_atomic_float::NAME);

        let queue_family =
            queue_family(&shared.instance, physical, needs_graphics).ok_or_else(|| {
                let kind = if needs_graphics {
                    "graphics"
                } else {
                    "compute"
                };
                GpuError::Backend(format!("no {kind} queue family"))
            })?;
        let device_limits = physical_device_limits(&shared.instance, physical, queue_family);

        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let mut supported_desc_idx = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut supported_dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut supported_atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
        supported_desc_idx.p_next = &mut supported_dynamic_rendering as *mut _ as *mut c_void;
        supported_dynamic_rendering.p_next = &mut supported_atomic_float as *mut _ as *mut c_void;
        let mut supported_features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut supported_desc_idx as *mut _ as *mut c_void,
            ..Default::default()
        };
        unsafe {
            shared
                .instance
                .get_physical_device_features2(physical, &mut supported_features2);
        }

        let descriptor_indexing = supported_desc_idx
            .shader_storage_buffer_array_non_uniform_indexing
            == vk::TRUE
            && supported_desc_idx.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
            && supported_desc_idx.descriptor_binding_storage_buffer_update_after_bind == vk::TRUE
            && supported_desc_idx.descriptor_binding_sampled_image_update_after_bind == vk::TRUE
            && supported_desc_idx.descriptor_binding_partially_bound == vk::TRUE
            && supported_desc_idx.runtime_descriptor_array == vk::TRUE;
        if !descriptor_indexing {
            return Err(GpuError::UnsupportedFeatures(Features::DESCRIPTOR_INDEXING));
        }
        if needs_graphics && supported_dynamic_rendering.dynamic_rendering != vk::TRUE {
            return Err(GpuError::Unsupported(
                "Vulkan dynamic rendering feature is unavailable".to_string(),
            ));
        }

        // Only treat features the current backend actually exposes as
        // available to DeviceRequest. This
        // deliberately rejects hardware features whose HAL implementation is
        // not present yet instead of silently accepting the request.
        let mut available_features = Features::COMPUTE | Features::DESCRIPTOR_INDEXING;
        if needs_graphics {
            available_features |= Features::GRAPHICS;
        }
        let missing_required = req.required.difference(available_features);
        if !missing_required.is_empty() {
            return Err(GpuError::UnsupportedFeatures(missing_required));
        }

        let shader_atomic_float = shader_atomic_float_extension
            && supported_atomic_float.shader_buffer_float32_atomic_add == vk::TRUE;
        if shader_atomic_float {
            extensions.push(ext::shader_atomic_float::NAME.as_ptr());
        }

        let supported_features = supported_features2.features;
        let dual_src_blend = needs_graphics && supported_features.dual_src_blend == vk::TRUE;
        let fill_mode_non_solid =
            needs_graphics && supported_features.fill_mode_non_solid == vk::TRUE;
        let sampler_anisotropy =
            needs_graphics && supported_features.sampler_anisotropy == vk::TRUE;
        let max_sampler_anisotropy = unsafe {
            shared
                .instance
                .get_physical_device_properties(physical)
                .limits
                .max_sampler_anisotropy
        };

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
        let mut atomic_float_feat = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT {
            shader_buffer_float32_atomic_add: vk::TRUE,
            ..Default::default()
        };
        if shader_atomic_float && needs_graphics {
            dynamic_rendering_feat.p_next = &mut atomic_float_feat as *mut _ as *mut c_void;
        }
        desc_idx.p_next = if needs_graphics {
            &mut dynamic_rendering_feat as *mut _ as *mut c_void
        } else if shader_atomic_float {
            &mut atomic_float_feat as *mut _ as *mut c_void
        } else {
            null_mut()
        };
        let mut features2 = vk::PhysicalDeviceFeatures2 {
            features: vk::PhysicalDeviceFeatures {
                shader_sampled_image_array_dynamic_indexing: vk::TRUE,
                dual_src_blend: if dual_src_blend { vk::TRUE } else { vk::FALSE },
                fill_mode_non_solid: if fill_mode_non_solid {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                sampler_anisotropy: if sampler_anisotropy {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                ..Default::default()
            },
            ..Default::default()
        };
        features2.p_next = &mut desc_idx as *mut _ as *mut c_void;

        let device_create_info = vk::DeviceCreateInfo {
            p_next: &features2 as *const _ as *const c_void,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: if extensions.is_empty() {
                null()
            } else {
                extensions.as_ptr()
            },
            p_enabled_features: null(), // must be null when using features2 pNext
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
        let memory_properties = unsafe {
            shared
                .instance
                .get_physical_device_memory_properties(physical)
        };

        let inner = Arc::new(VulkanDeviceInner {
            shared,
            device,
            physical,
            memory_properties,
            queue_family,
            queue,
            queue_lock: Mutex::new(()),
            features: available_features,
            limits: device_limits,
            graphics: needs_graphics,
            dual_src_blend,
            fill_mode_non_solid,
            sampler_anisotropy,
            max_sampler_anisotropy,
            dynamic_rendering,
        });

        let bindless = create_bindless(&inner.device, inner.limits)?;
        let cmd_list_pool = Arc::new(CmdListPool::new(Arc::clone(&inner))?);
        let fence_pool = Arc::new(FencePool::new(Arc::clone(&inner)));
        let pipeline_cache = unsafe {
            inner
                .device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .map_err(|e| GpuError::Backend(format!("vkCreatePipelineCache: {e}")))?
        };

        Ok(Self {
            inner,
            buffers: Arc::new(Mutex::new(SlotMap::new())),
            textures: Arc::new(Mutex::new(SlotMap::new())),
            samplers: Arc::new(Mutex::new(SlotMap::new())),
            shaders: Mutex::new(SlotMap::new()),
            pipelines: Arc::new(Mutex::new(SlotMap::new())),
            render_targets: Arc::new(Mutex::new(SlotMap::new())),
            bindless,
            pipeline_cache,
            cmd_list_pool,
            fence_pool,
            lifetime: Arc::new(Mutex::new(VulkanLifetimeState::default())),
        })
    }

    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(shared, physical, req, &[], false)
    }

    pub(crate) fn new_headless_graphics(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(shared, physical, req, &[], true)
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
            &[khr::swapchain::NAME.as_ptr()],
            true,
        )
    }

}

fn bindless_capacities(limits: DeviceLimits) -> Result<(u32, u32)> {
    let total = limits.max_update_after_bind_descriptors;
    let buffer_capacity = MAX_BINDLESS_BUFFERS
        .min(limits.max_storage_buffers)
        .min(total.saturating_sub(1));
    let texture_capacity = MAX_BINDLESS_TEXTURES
        .min(limits.max_sampled_textures)
        .min(total.saturating_sub(buffer_capacity));
    if buffer_capacity == 0 || texture_capacity == 0 {
        return Err(GpuError::Unsupported(format!(
            "bindless descriptor limits are too small: storage={}, textures={}, total={total}",
            limits.max_storage_buffers, limits.max_sampled_textures
        )));
    }
    Ok((buffer_capacity, texture_capacity))
}

fn create_bindless(dev: &ash::Device, limits: DeviceLimits) -> Result<BindlessState> {
    let (buffer_capacity, texture_capacity) = bindless_capacities(limits)?;
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: texture_capacity,
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
                p_next: &mut flags_info as *mut _ as *mut c_void,
                ..Default::default()
            },
            None,
        )
        .map_err(|e| GpuError::Backend(format!("bindless layout: {e}")))?
    };

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: texture_capacity,
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

    Ok(BindlessState {
        layout,
        pool,
        set,
        buffer_capacity,
        texture_capacity,
        bound_textures: Mutex::new(vec![false; texture_capacity as usize]),
    })
}

pub(crate) fn queue_family(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    needs_graphics: bool,
) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let supports = |f: &vk::QueueFamilyProperties, flags| f.queue_flags.contains(flags);

    if needs_graphics {
        families
            .iter()
            .enumerate()
            .find(|(_, f)| supports(f, vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            .or_else(|| {
                families
                    .iter()
                    .enumerate()
                    .find(|(_, f)| supports(f, vk::QueueFlags::GRAPHICS))
            })
            .map(|(i, _)| i as u32)
    } else {
        families
            .iter()
            .enumerate()
            .find(|(_, f)| {
                supports(f, vk::QueueFlags::COMPUTE) && !supports(f, vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                families
                    .iter()
                    .enumerate()
                    .find(|(_, f)| supports(f, vk::QueueFlags::COMPUTE))
            })
            .map(|(i, _)| i as u32)
    }
}

pub(crate) fn physical_device_limits(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
) -> DeviceLimits {
    let mut descriptor = vk::PhysicalDeviceDescriptorIndexingProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2 {
        p_next: &mut descriptor as *mut _ as *mut c_void,
        ..Default::default()
    };
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let limits = properties.properties.limits;
    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let timestamp_supported = queues
        .get(queue_family as usize)
        .is_some_and(|queue| queue.timestamp_valid_bits > 0);
    DeviceLimits {
        max_workgroup_size: limits.max_compute_work_group_size,
        max_workgroup_invocations: limits.max_compute_work_group_invocations,
        max_dispatch_size: limits.max_compute_work_group_count,
        max_storage_buffer_range: u64::from(limits.max_storage_buffer_range),
        max_push_constant_size: limits.max_push_constants_size,
        max_storage_buffers: descriptor
            .max_per_stage_descriptor_update_after_bind_storage_buffers
            .min(descriptor.max_descriptor_set_update_after_bind_storage_buffers),
        max_sampled_textures: descriptor
            .max_per_stage_descriptor_update_after_bind_sampled_images
            .min(descriptor.max_descriptor_set_update_after_bind_sampled_images),
        max_update_after_bind_descriptors: descriptor
            .max_update_after_bind_descriptors_in_all_pools,
        max_memory_allocations: limits.max_memory_allocation_count,
        timestamp_supported,
        timestamp_period_ns: limits.timestamp_period,
    }
}

fn stale(handle: BufferHandle, buffers: &SlotMap<marker::Buffer, VulkanBuffer>) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: buffers.generation_at(handle.index()).unwrap_or(u32::MAX),
    })
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Resources referenced by submitted work must not be destroyed until
        // the device is idle. Synchronous HAL calls normally guarantee this,
        // but graphics/raw-context users and error paths may still have work
        // pending when the owning VulkanDevice is dropped.
        let _queue = self
            .inner
            .queue_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
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
            self.inner
                .device
                .destroy_pipeline_cache(self.pipeline_cache, None);
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
    use std::ops::Deref;

    use super::*;
    use crate::instance::VulkanInstance;
    use zengpu_hal::{AdapterRequest, AddressMode, BorderColor, DeviceRequest, Format, GpuInstance};

    struct TestDevice {
        dev: Box<dyn GpuDevice>,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl Deref for TestDevice {
        type Target = dyn GpuDevice;

        fn deref(&self) -> &Self::Target {
            self.dev.as_ref()
        }
    }

    fn try_device() -> Option<TestDevice> {
        let guard = crate::test_gpu_lock();
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        let dev = adapter.open(DeviceRequest::default()).ok()?;
        Some(TestDevice { dev, _guard: guard })
    }

    fn try_graphics_device() -> Option<TestDevice> {
        let guard = crate::test_gpu_lock();
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_vulkan_adapter()?;
        let dev = adapter.open_headless(DeviceRequest::default()).ok()?;
        Some(TestDevice {
            dev: Box::new(dev),
            _guard: guard,
        })
    }

    fn rw_desc(size: u64) -> BufferDesc {
        BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        }
    }

    fn u32s_to_bytes(words: &[u32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        }
    }

    #[test]
    fn compute_push_constant_abi_uses_vulkan_portable_minimum() {
        assert_eq!(COMPUTE_PUSH_CONSTANT_BYTES, 128);
    }

    #[test]
    fn bindless_capacities_respect_each_device_limit() {
        let limits = DeviceLimits {
            max_storage_buffers: 16,
            max_sampled_textures: 8,
            max_update_after_bind_descriptors: 12,
            ..Default::default()
        };
        assert_eq!(bindless_capacities(limits).unwrap(), (11, 1));
        assert!(
            bindless_capacities(DeviceLimits {
                max_storage_buffers: 1,
                max_sampled_textures: 1,
                max_update_after_bind_descriptors: 1,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    #[ignore = "explicit NVIDIA/Vulkan logical-device churn stress test"]
    fn repeated_device_create_use_drop_stress() {
        let _guard = crate::test_gpu_lock();
        for cycle in 0..100u32 {
            let inst = VulkanInstance::new().expect("create Vulkan instance");
            let adapter = inst
                .request_adapter(AdapterRequest::default())
                .expect("find Vulkan adapter");
            let dev = adapter
                .open(DeviceRequest::default())
                .expect("open Vulkan device");
            let src = dev
                .create_buffer(BufferDesc {
                    size: 4096,
                    usage: BufferUsage::TRANSFER_SRC,
                    memory: MemoryUsage::Upload,
                })
                .expect("create upload buffer");
            let dst = dev
                .create_buffer(BufferDesc {
                    size: 4096,
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                    memory: MemoryUsage::Readback,
                })
                .expect("create readback buffer");
            let pattern = [cycle as u8; 4096];
            dev.write_buffer(src, 0, &pattern).expect("write upload");
            dev.copy_buffer(src, 0, dst, 0, 4096).expect("copy buffers");
            assert_eq!(
                dev.read_buffer(dst, 0, 4096).expect("read back"),
                pattern,
                "cycle {cycle}"
            );
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
    fn device_local_buffer_staging_round_trip() {
        const SIZE: usize = 4 * 1024 * 1024;
        let Some(dev) = try_device() else { return };
        let upload = dev
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::TRANSFER_SRC,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let gpu = dev
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let readback = dev
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: MemoryUsage::Readback,
            })
            .unwrap();
        let data: Vec<u8> = (0..SIZE)
            .map(|i| (i as u32).wrapping_mul(31).wrapping_add(7) as u8)
            .collect();
        dev.write_buffer(upload, 0, &data).unwrap();
        assert!(dev.write_buffer(gpu, 0, &[1]).is_err());
        dev.copy_buffer(upload, 0, gpu, 0, SIZE as u64).unwrap();
        dev.copy_buffer(gpu, 0, readback, 0, SIZE as u64).unwrap();
        assert_eq!(dev.read_buffer(readback, 0, SIZE as u64).unwrap(), data);

        dev.destroy_buffer(upload);
        dev.destroy_buffer(gpu);
        dev.destroy_buffer(readback);
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
    fn trait_open_reports_compute_only() {
        let Some(dev) = try_device() else { return };
        assert!(!dev.capabilities().graphics);
        assert!(dev.capabilities().compute);
        assert!(
            dev.capabilities()
                .features
                .contains(Features::COMPUTE | Features::DESCRIPTOR_INDEXING)
        );
        let limits = dev.limits();
        assert!(limits.max_workgroup_invocations >= 128);
        assert!(limits.max_workgroup_size[0] >= 128);
        assert!(limits.max_dispatch_size.iter().all(|value| *value > 0));
        assert!(limits.max_storage_buffer_range > 0);
        assert!(limits.max_push_constant_size >= COMPUTE_PUSH_CONSTANT_BYTES as u32);
        assert!(limits.max_storage_buffers >= MAX_BINDLESS_BUFFERS);
    }

    #[test]
    fn required_unimplemented_feature_is_rejected() {
        let _guard = crate::test_gpu_lock();
        let Some(adapter) = VulkanInstance::new()
            .ok()
            .and_then(|instance| instance.request_vulkan_adapter())
        else {
            return;
        };
        let err = match adapter.open_headless(DeviceRequest {
            required: Features::TIMESTAMPS,
            ..Default::default()
        }) {
            Ok(_) => panic!("unsupported required feature unexpectedly succeeded"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GpuError::UnsupportedFeatures(features) if features == Features::TIMESTAMPS
        ));
    }

    #[test]
    fn optional_unimplemented_feature_is_ignored() {
        let _guard = crate::test_gpu_lock();
        let Some(adapter) = VulkanInstance::new()
            .ok()
            .and_then(|instance| instance.request_vulkan_adapter())
        else {
            return;
        };
        let dev = adapter
            .open_headless(DeviceRequest {
                optional: Features::TIMESTAMPS,
                ..Default::default()
            })
            .unwrap();
        assert!(dev.capabilities().graphics);
    }

    #[test]
    fn compute_device_rejects_required_graphics() {
        let _guard = crate::test_gpu_lock();
        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = instance.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let err = match adapter.open(DeviceRequest {
            required: Features::GRAPHICS,
            ..Default::default()
        }) {
            Ok(_) => panic!("compute-only device unexpectedly accepted graphics"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GpuError::UnsupportedFeatures(features) if features == Features::GRAPHICS
        ));
    }

    #[test]
    fn graphics_device_accepts_required_backend_features() {
        let _guard = crate::test_gpu_lock();
        let Some(adapter) = VulkanInstance::new()
            .ok()
            .and_then(|instance| instance.request_vulkan_adapter())
        else {
            return;
        };
        let dev = adapter
            .open_headless(DeviceRequest {
                required: Features::COMPUTE | Features::GRAPHICS | Features::DESCRIPTOR_INDEXING,
                ..Default::default()
            })
            .unwrap();
        assert!(dev.capabilities().compute);
        assert!(dev.capabilities().graphics);
    }

    #[test]
    fn create_sampler_with_full_desc() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let sampler = dev
            .create_sampler(SamplerDesc {
                min_filter: FilterMode::Linear,
                mag_filter: FilterMode::Linear,
                mip_filter: FilterMode::Linear,
                address: AddressMode::Repeat,
                anisotropy: 16,
                lod_min: 0.0,
                lod_max: 4.0,
                compare: Some(CompareFn::LessEqual),
                border: BorderColor::OpaqueBlack,
            })
            .unwrap();
        dev.destroy_sampler(sampler);
    }

    fn tex_desc(dimension: zengpu_hal::TexDim) -> TextureDesc {
        TextureDesc {
            width: 8,
            height: 8,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension,
            mip_levels: 1,
            array_layers: 1,
        }
    }

    #[test]
    fn create_texture_2d_array() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let tex = dev
            .create_texture(TextureDesc {
                array_layers: 4,
                ..tex_desc(zengpu_hal::TexDim::D2)
            })
            .unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn create_texture_3d() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let tex = dev
            .create_texture(TextureDesc {
                depth: 4,
                ..tex_desc(zengpu_hal::TexDim::D3)
            })
            .unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn create_texture_cube_requires_six_layers() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let err = dev
            .create_texture(TextureDesc {
                array_layers: 3,
                ..tex_desc(zengpu_hal::TexDim::Cube)
            })
            .unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));

        let tex = dev
            .create_texture(TextureDesc {
                array_layers: 6,
                ..tex_desc(zengpu_hal::TexDim::Cube)
            })
            .unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn upload_texture_data_region_targets_specific_mip() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let tex = dev
            .create_texture(TextureDesc {
                mip_levels: 2,
                ..tex_desc(zengpu_hal::TexDim::D2)
            })
            .unwrap();
        // Mip 0 is 8x8, mip 1 is 4x4.
        dev.upload_texture_data(tex, &vec![0xFFu8; 8 * 8 * 4])
            .unwrap();
        dev.upload_texture_data_region(tex, 1, 0, &[0x80u8; 4 * 4 * 4])
            .unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn generate_mipmaps_without_transfer_usage_fails() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let tex = dev
            .create_texture(TextureDesc {
                mip_levels: 4,
                usage: TextureUsage::SAMPLED,
                ..tex_desc(zengpu_hal::TexDim::D2)
            })
            .unwrap();
        let err = dev.generate_mipmaps(tex).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "TRANSFER_SRC | TRANSFER_DST",
                ..
            })
        ));
        dev.destroy_texture(tex);
    }

    #[test]
    fn generate_mipmaps_builds_full_chain() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let tex = dev
            .create_texture(TextureDesc {
                mip_levels: 4,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                ..tex_desc(zengpu_hal::TexDim::D2)
            })
            .unwrap();
        dev.upload_texture_data(tex, &vec![0xFFu8; 8 * 8 * 4])
            .unwrap();
        dev.generate_mipmaps(tex).unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn mixed_copy_dispatch_submission_round_trips_device_local_data() {
        use zengpu_spirv::{ZslShader, zsl};

        const DOUBLE_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel double(inp: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len { out[i] = inp[i] + inp[i] }
            }
        );
        let Some(dev) = try_device() else { return };
        let shader = dev.create_shader(DOUBLE_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        const N: u32 = 64;
        let size = N as u64 * 4;
        let upload = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_SRC,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let input = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let output = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let readback = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: MemoryUsage::Readback,
            })
            .unwrap();
        let values: Vec<f32> = (0..N).map(|i| i as f32 - 17.0).collect();
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
        dev.write_buffer(upload, 0, bytes).unwrap();
        let submission = dev
            .submit_compute_ops(
                0x4d_49_58_45_44,
                &[
                    ComputeOp::CopyBuffer(zengpu_hal::BufferCopyOp {
                        src: upload,
                        src_offset: 0,
                        dst: input,
                        dst_offset: 0,
                        len: size,
                    }),
                    ComputeOp::Dispatch(DispatchOp {
                        pipeline,
                        bindings: Bindings {
                            buffers: &[input.index(), output.index()],
                            textures: &[],
                            scalars: &[Scalar::U32(N)],
                        },
                        grid: [1, 1, 1],
                    }),
                    ComputeOp::CopyBuffer(zengpu_hal::BufferCopyOp {
                        src: output,
                        src_offset: 0,
                        dst: readback,
                        dst_offset: 0,
                        len: size,
                    }),
                ],
            )
            .unwrap();
        assert_eq!(submission.cycle_id(), 0x4d_49_58_45_44);
        submission.wait(Duration::from_secs(5)).unwrap();
        let actual = dev.read_buffer(readback, 0, size).unwrap();
        for (index, bytes) in actual.chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_ne_bytes(bytes.try_into().unwrap()),
                values[index] * 2.0
            );
        }
        dev.destroy_buffer(upload);
        dev.destroy_buffer(input);
        dev.destroy_buffer(output);
        dev.destroy_buffer(readback);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    #[test]
    fn dispatch_batch_chains_ops_with_visible_writes() {
        use zengpu_spirv::{ZslShader, zsl};

        const ADD_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel add(a: device buffer<f32>, b: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len { out[i] = a[i] + b[i] }
            }
        );
        const RELU_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel relu(inp: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len { out[i] = max(inp[i], 0.0) }
            }
        );

        let Some(dev) = try_device() else { return };

        let add_shader = dev.create_shader(ADD_ZSL.spirv_desc()).unwrap();
        let add_pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: add_shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let relu_shader = dev.create_shader(RELU_ZSL.spirv_desc()).unwrap();
        let relu_pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: relu_shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();

        let storage_rw = |size: u64| BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        let n: u32 = 8;
        let a = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        let b = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        let sum = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        let out = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();

        let a_vals: Vec<f32> = (0..n).map(|i| i as f32 - 4.0).collect();
        let b_vals: Vec<f32> = (0..n).map(|_| -10.0).collect();
        dev.write_buffer(
            a,
            0,
            u32s_to_bytes(unsafe {
                std::slice::from_raw_parts(a_vals.as_ptr() as *const u32, a_vals.len())
            }),
        )
        .unwrap();
        dev.write_buffer(
            b,
            0,
            u32s_to_bytes(unsafe {
                std::slice::from_raw_parts(b_vals.as_ptr() as *const u32, b_vals.len())
            }),
        )
        .unwrap();

        // sum = a + b (all negative); out = relu(sum) (all zero), batched as
        // one submission. The implicit barrier between ops must make `sum`'s
        // write visible to the relu dispatch's read.
        let submission = dev
            .submit_batch(
                0x4d_50_50_49,
                &[
                    DispatchOp {
                        pipeline: add_pipeline,
                        bindings: Bindings {
                            buffers: &[a.index(), b.index(), sum.index()],
                            textures: &[],
                            scalars: &[Scalar::U32(n)],
                        },
                        grid: [n.div_ceil(64), 1, 1],
                    },
                    DispatchOp {
                        pipeline: relu_pipeline,
                        bindings: Bindings {
                            buffers: &[sum.index(), out.index()],
                            textures: &[],
                            scalars: &[Scalar::U32(n)],
                        },
                        grid: [n.div_ceil(64), 1, 1],
                    },
                ],
            )
            .unwrap();
        assert_eq!(submission.cycle_id(), 0x4d_50_50_49);
        let second_out = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        let second_submission = dev
            .submit(
                0x4d_50_50_4a,
                relu_pipeline,
                Bindings {
                    buffers: &[a.index(), second_out.index()],
                    textures: &[],
                    scalars: &[Scalar::U32(n)],
                },
                [n.div_ceil(64), 1, 1],
            )
            .unwrap();
        assert!(dev.write_buffer(a, 0, &[0; 4]).is_err());
        // Destruction while pending invalidates handles immediately but must
        // not recycle descriptor/pipeline slots until this token completes.
        let retired_sum_index = sum.index();
        dev.destroy_buffer(sum);
        dev.destroy_pipeline(add_pipeline);
        assert!(dev.read_buffer(sum, 0, 4).is_err());
        let allocated_while_pending = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        assert_ne!(allocated_while_pending.index(), retired_sum_index);
        // A zero-duration wait is always bounded. A sufficiently fast GPU may
        // already be complete; otherwise timeout is distinct and retryable.
        let zero_wait = submission.wait(Duration::ZERO);
        assert!(zero_wait.is_ok() || matches!(zero_wait, Err(GpuError::Timeout)));
        submission.wait(Duration::from_secs(5)).unwrap();
        assert_eq!(submission.poll().unwrap(), SubmissionStatus::Complete);
        let allocated_after_first = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        assert_ne!(allocated_after_first.index(), retired_sum_index);
        second_submission.wait(Duration::from_secs(5)).unwrap();
        assert_eq!(
            second_submission.poll().unwrap(),
            SubmissionStatus::Complete
        );
        let allocated_after_completion = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        assert_eq!(allocated_after_completion.index(), retired_sum_index);

        let bytes = dev.read_buffer(out, 0, n as u64 * 4).unwrap();
        let result: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n as usize) };
        assert_eq!(result, &[0.0; 8]);

        dev.destroy_shader(add_shader);
        dev.destroy_pipeline(relu_pipeline);
        dev.destroy_shader(relu_shader);
        dev.destroy_buffer(allocated_while_pending);
        dev.destroy_buffer(allocated_after_first);
        dev.destroy_buffer(allocated_after_completion);
        dev.destroy_buffer(second_out);
    }

    #[test]
    fn dispatch_rejects_invalid_bindless_buffer_indices() {
        use zengpu_spirv::{ZslShader, zsl};

        const FILL_ZSL: ZslShader = zsl!(
            @workgroup_size(1)
            kernel fill(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = 1.0
            }
        );

        let Some(dev) = try_device() else { return };
        let shader = dev.create_shader(FILL_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [1, 1, 1],
            })
            .unwrap();

        let non_storage = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[non_storage.index()],
                    ..Default::default()
                },
                [1, 1, 1],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "STORAGE",
                ..
            })
        ));
        dev.destroy_buffer(non_storage);

        let storage = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let stale_index = storage.index();
        dev.destroy_buffer(storage);
        let err = dev
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[stale_index],
                    ..Default::default()
                },
                [1, 1, 1],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(message))
                if message.contains("not live")
        ));

        let err = dev
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[dev.limits().max_storage_buffers],
                    ..Default::default()
                },
                [1, 1, 1],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(message))
                if message.contains("capacity") || message.contains("not live")
        ));

        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    #[test]
    fn dispatch_rejects_destroyed_bound_texture_index() {
        use zengpu_spirv::{ZslShader, zsl};

        const FILL_ZSL: ZslShader = zsl!(
            @workgroup_size(1)
            kernel fill(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = 1.0
            }
        );

        let Some(dev) = try_graphics_device() else {
            return;
        };
        let shader = dev.create_shader(FILL_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [1, 1, 1],
            })
            .unwrap();
        let output = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let texture = dev
            .create_texture(tex_desc(zengpu_hal::TexDim::D2))
            .unwrap();
        let sampler = dev.create_sampler(SamplerDesc::default()).unwrap();
        let device = dev
            .dev
            .as_any()
            .downcast_ref::<VulkanDevice>()
            .expect("graphics helper must contain VulkanDevice");
        let texture_index = device.bind_texture(texture, sampler).unwrap();
        dev.destroy_texture(texture);

        let err = dev
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[output.index()],
                    textures: &[texture_index],
                    scalars: &[],
                },
                [1, 1, 1],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(message))
                if message.contains("not live and bound")
        ));

        dev.destroy_sampler(sampler);
        dev.destroy_buffer(output);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    #[test]
    fn concurrent_dispatches_share_the_queue_safely() {
        use std::sync::{Arc, Barrier};
        use zengpu_spirv::{ZslShader, zsl};

        const FILL_ZSL: ZslShader = zsl!(
            push P { value: f32 }
            @workgroup_size(64)
            kernel fill(out: device mut buffer<f32>, p: P, id: global_id) {
                out[id.x] = p.value
            }
        );

        let Some(dev) = try_device() else { return };
        let TestDevice {
            dev,
            _guard: test_guard,
        } = dev;
        let dev: Arc<dyn GpuDevice> = Arc::from(dev);
        let shader = dev.create_shader(FILL_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();

        const THREADS: usize = 2;
        const LEN: u64 = 256;
        let desc = BufferDesc {
            size: LEN * 4,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        let outputs: Vec<_> = (0..THREADS)
            .map(|_| dev.create_buffer(desc).unwrap())
            .collect();
        let start = Arc::new(Barrier::new(THREADS));
        let workers: Vec<_> = outputs
            .iter()
            .copied()
            .enumerate()
            .map(|(worker, output)| {
                let dev = Arc::clone(&dev);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let buffers = [output.index()];
                    let scalars = [Scalar::F32(worker as f32 + 0.5)];
                    start.wait();
                    dev.dispatch(
                        pipeline,
                        Bindings {
                            buffers: &buffers,
                            textures: &[],
                            scalars: &scalars,
                        },
                        [LEN.div_ceil(64) as u32, 1, 1],
                    )
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("dispatch worker panicked").unwrap();
        }

        for (worker, output) in outputs.iter().copied().enumerate() {
            let bytes = dev.read_buffer(output, 0, LEN * 4).unwrap();
            let expected = worker as f32 + 0.5;
            for value in bytes.chunks_exact(4) {
                assert_eq!(f32::from_ne_bytes(value.try_into().unwrap()), expected);
            }
            dev.destroy_buffer(output);
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        drop(dev);
        drop(test_guard);
    }

    #[test]
    fn zsl_trigonometry_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const TRIG_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel trig(src: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len {
                    let x = src[i]
                    out[i] = sin(x) + cos(x) + tan(x)
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 128u32;
        let size = u64::from(n) * 4;
        let desc = BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        let src = dev.create_buffer(desc).unwrap();
        let out = dev.create_buffer(desc).unwrap();
        let input: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 / n as f32).collect();
        dev.write_buffer(
            src,
            0,
            u32s_to_bytes(unsafe {
                std::slice::from_raw_parts(input.as_ptr() as *const u32, input.len())
            }),
        )
        .unwrap();

        let shader = dev.create_shader(TRIG_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &[src.index(), out.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();

        let raw = dev.read_buffer(out, 0, size).unwrap();
        for (i, value) in input.iter().copied().enumerate() {
            let got = f32::from_ne_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            let expected = value.sin() + value.cos() + value.tan();
            assert!((got - expected).abs() < 2e-5, "out[{i}] mismatch");
        }

        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(src);
        dev.destroy_buffer(out);
    }

    #[test]
    fn zsl_integer_buffers_round_trip_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const TYPED_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len {
                    out_u[i] = src_u[i]
                    out_i[i] = src_i[i]
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 128u32;
        let size = u64::from(n) * 4;
        let desc = BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        let src_u = dev.create_buffer(desc).unwrap();
        let src_i = dev.create_buffer(desc).unwrap();
        let out_u = dev.create_buffer(desc).unwrap();
        let out_i = dev.create_buffer(desc).unwrap();
        let input_u: Vec<u32> = (0..n)
            .map(|value| value.wrapping_mul(2_654_435_761))
            .collect();
        let input_i: Vec<i32> = (0..n as i32).map(|value| value - 64).collect();
        let bytes_u = u32s_to_bytes(&input_u);
        let bytes_i = u32s_to_bytes(unsafe {
            std::slice::from_raw_parts(input_i.as_ptr() as *const u32, input_i.len())
        });
        dev.write_buffer(src_u, 0, bytes_u).unwrap();
        dev.write_buffer(src_i, 0, bytes_i).unwrap();

        let shader = dev.create_shader(TYPED_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &[src_u.index(), src_i.index(), out_u.index(), out_i.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();

        assert_eq!(dev.read_buffer(out_u, 0, size).unwrap(), bytes_u);
        assert_eq!(dev.read_buffer(out_i, 0, size).unwrap(), bytes_i);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(src_u);
        dev.destroy_buffer(src_i);
        dev.destroy_buffer(out_u);
        dev.destroy_buffer(out_i);
    }

    #[test]
    fn zsl_finite_classification_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const FINITE_ZSL: ZslShader = zsl!(
            push P { len: u32 }
            @workgroup_size(64)
            kernel classify(src: device buffer<f32>, out: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.len {
                    let x = src[i]
                    if isfinite(x) {
                        out[i] = 1
                    } else {
                        if isnan(x) {
                            out[i] = 2
                        } else {
                            if isinf(x) { out[i] = 3 } else { out[i] = 4 }
                        }
                    }
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let input = [0.0f32, -12.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let expected = [1u32, 1, 2, 3, 3];
        let size = std::mem::size_of_val(&input) as u64;
        let desc = BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        let src = dev.create_buffer(desc).unwrap();
        let out = dev.create_buffer(desc).unwrap();
        dev.write_buffer(
            src,
            0,
            u32s_to_bytes(unsafe {
                std::slice::from_raw_parts(input.as_ptr() as *const u32, input.len())
            }),
        )
        .unwrap();

        let shader = dev.create_shader(FINITE_ZSL.spirv_desc()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &[src.index(), out.index()],
                scalars: &[Scalar::U32(input.len() as u32)],
                textures: &[],
            },
            [1, 1, 1],
        )
        .unwrap();

        let raw = dev.read_buffer(out, 0, size).unwrap();
        let actual: Vec<u32> = raw
            .chunks_exact(4)
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(src);
        dev.destroy_buffer(out);
    }
}
