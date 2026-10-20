
use std::{
    any::Any,
    collections::HashMap,
    ffi::{CStr, CString, c_void},
    fs,
    io::ErrorKind,
    mem,
    path::PathBuf,
    ptr::{copy_nonoverlapping, null, null_mut},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use ash::{Device, ext, khr, vk};
use zengpu_hal::{
    Bindings, BufferDesc, BufferHandle, BufferUsage, CompareFn, ComputeOp, ComputePipelineDesc,
    DeviceLimits, DeviceRequest, DispatchOp, Features, FilterMode, GpuDevice, GpuError,
    GpuSubmission, GraphicsDevice, GraphicsPipelineDesc, HalCapabilities, MemoryBudget,
    MemoryUsage, PipelineHandle, PolygonMode, QueryPoolHandle, Result, SamplerDesc, SamplerHandle,
    Scalar, ShaderDesc, ShaderHandle, ShaderSource, SlotMap, Submission, SubmissionStatus,
    SurfaceConfig, TargetHandle, TexDim, TextureDesc, TextureHandle, TextureUsage, UsageError,
    WindowHandles, marker,
};

mod dispatch;
mod format;
mod graphics;
mod hal_impl;
mod resources;
mod transfer;

use format::*;

use crate::command_list::{COLOR_SUBRESOURCE, CmdListPool, VulkanCommandList};
use crate::depth_target::{DEPTH_FORMAT, DepthTarget};
use crate::instance::VulkanShared;
use crate::offscreen::OffscreenTarget;
use crate::surface::VulkanSurface;
use crate::swapchain::{DeviceContext, map_vk_err};

/// Max storage buffers per bindless SSBO table (one per element type).
const MAX_BINDLESS_BUFFERS: u32 = 4096;

const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Bindless SSBO table for `buffer<f32>` access. Every `STORAGE` buffer is registered here,
/// whatever its ZSL element type: buffers are untyped at the HAL.
const SSBO_BINDING_F32: u32 = 0;

/// Combined-image-sampler table. Numbered here only because it shares the descriptor set layout.
const SSBO_BINDING_TEXTURES: u32 = 1;

/// Bindless SSBO table for `buffer<u32>`, same slot indices as the f32 table: `uint[]` access
/// without bitcasting.
const SSBO_BINDING_U32: u32 = 2;

/// Bindless SSBO table for `buffer<i32>`, same slot indices as the f32 table.
const SSBO_BINDING_I32: u32 = 3;

/// Storage-image table: each `STORAGE` texture's mip 0, in `GENERAL` layout, at its own slot.
const STORAGE_IMAGE_BINDING: u32 = 4;

/// SSBO tables sharing the storage-buffer budget (`f32`/`u32`/`i32`). Append-only: new tables take
/// new binding numbers, no renumbering.
const SSBO_TYPED_TABLE_COUNT: u32 = 3;

/// Keep compute's push-constant ABI at the 128-byte Vulkan minimum, so conforming devices are
/// not rejected.
const COMPUTE_PUSH_CONSTANT_BYTES: usize = 128;

struct BindlessState {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    buffer_capacity: u32,
    texture_capacity: u32,
    /// Storage-image table slots; 0 if compute cannot store to images.
    storage_capacity: u32,
    bound_textures: Mutex<Vec<bool>>,
}

// SAFETY: handles are u64 values, lifetimes controlled through `VulkanDevice`.
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

impl VulkanSubmissionState {
    const DONE: Self = Self {
        fence: None,
        cmd: None,
    };
}

/// Submissions dropped while their work still ran. `reap_orphans` recycles their fence, command
/// buffer and deferred frees once the work finishes.
type Orphans = Arc<Mutex<Vec<VulkanSubmission>>>;

enum DeferredVulkanResource {
    Buffer(u32, VulkanBuffer),
    Texture(u32, VulkanTexture),
    Sampler(u32, vk::Sampler),
    Pipeline(u32, VulkanPipeline),
}

/// Queue work, numbered in submission order. Once `n` is finished, everything before it is too.
///
/// Every submission takes a number, so a resource is freed only after the work that could
/// still use it.
#[derive(Default)]
pub(crate) struct QueueSerials {
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl QueueSerials {
    /// Numbers a submission that reached the queue. Call with the queue lock held, right after the
    /// submit, so numbers follow queue order.
    pub(crate) fn next(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub(crate) fn complete(&self, serial: u64) {
        self.completed.fetch_max(serial, Ordering::AcqRel);
    }

    pub(crate) fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    pub(crate) fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    pub(crate) fn idle(&self) -> bool {
        self.completed() >= self.submitted()
    }
}

#[derive(Default)]
struct VulkanLifetimeState {
    /// Destroyed while work pending: freed when that work finishes.
    deferred: Vec<(u64, DeferredVulkanResource)>,
    /// Last submission per buffer (by bindless index).
    buffer_use: HashMap<u32, u64>,
}

impl VulkanLifetimeState {
    fn buffer_busy(&self, index: u32, completed: u64) -> bool {
        self.buffer_use
            .get(&index)
            .is_some_and(|serial| *serial > completed)
    }

    fn mark_used(&mut self, indices: &[u32], serial: u64) {
        for &index in indices {
            self.buffer_use.insert(index, serial);
        }
    }

    fn take_ready(&mut self, completed: u64) -> Vec<DeferredVulkanResource> {
        self.buffer_use.retain(|_, serial| *serial > completed);
        let (ready, waiting): (Vec<_>, Vec<_>) = mem::take(&mut self.deferred)
            .into_iter()
            .partition(|(serial, _)| *serial <= completed);
        self.deferred = waiting;
        ready.into_iter().map(|(_, resource)| resource).collect()
    }
}

/// Fence-backed submission. Completion releases its fence and command buffer once. After a
/// timed-out wait this handle still owns both, so a later poll or wait is safe.
struct VulkanSubmission {
    cycle_id: u64,
    serial: u64,
    inner: Arc<VulkanDeviceInner>,
    fence_pool: Arc<FencePool>,
    cmd_pool: Arc<CmdListPool>,
    buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    samplers: Arc<Mutex<SlotMap<marker::Sampler, vk::Sampler>>>,
    pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    lifetime: Arc<Mutex<VulkanLifetimeState>>,
    orphans: Orphans,
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
        self.inner.serials.complete(self.serial);
        let deferred = self
            .lifetime
            .lock()
            .unwrap()
            .take_ready(self.inner.serials.completed());
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
        let mut state = mem::replace(self.state.get_mut().unwrap(), VulkanSubmissionState::DONE);
        let Some(fence) = state.fence else {
            return;
        };
        match unsafe { self.inner.device.get_fence_status(fence) } {
            Ok(true) => self.complete_locked(&mut state),
            Ok(false) => {
                let orphan = self.orphan(state);
                self.orphans.lock().unwrap().push(orphan);
            }
            Err(_) => {
                unsafe { self.inner.device.destroy_fence(fence, None) };
                self.release_in_flight();
            }
        }
    }
}

impl VulkanSubmission {
    /// Copy of this submission holding `state`, outlives the original.
    fn orphan(&self, state: VulkanSubmissionState) -> Self {
        Self {
            cycle_id: self.cycle_id,
            serial: self.serial,
            inner: Arc::clone(&self.inner),
            fence_pool: Arc::clone(&self.fence_pool),
            cmd_pool: Arc::clone(&self.cmd_pool),
            buffers: Arc::clone(&self.buffers),
            textures: Arc::clone(&self.textures),
            samplers: Arc::clone(&self.samplers),
            pipelines: Arc::clone(&self.pipelines),
            lifetime: Arc::clone(&self.lifetime),
            orphans: Arc::clone(&self.orphans),
            state: Mutex::new(state),
        }
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

/// Compute or graphics pipeline. Both share one `PipelineHandle` slotmap.
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
    pub(crate) fn handles(&self) -> (vk::Pipeline, vk::PipelineLayout) {
        match *self {
            VulkanPipeline::Compute { layout, pipeline }
            | VulkanPipeline::Graphics { layout, pipeline } => (pipeline, layout),
        }
    }
}

/// Render target for a `RenderPassDesc` attachment: swapchain image or offscreen texture.
/// `layout` tracks the current `vk::ImageLayout` for barriers: dynamic rendering has no
/// render pass to do it.
pub(crate) struct VulkanRenderTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
    #[allow(dead_code)]
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
    /// Layout after `sample_after`: the texture's resting layout.
    pub resting: vk::ImageLayout,
    /// Texture behind a color target, to tell whether it holds texels (an upload may have given it
    /// some). `None` for swapchain images and depth targets.
    pub texture: Option<TextureHandle>,
}

pub(crate) struct VulkanQueryPool {
    pub pool: vk::QueryPool,
    pub count: u32,
}

/// Shared device state. `Arc`, so swapchains hold a ref.
pub(crate) struct VulkanDeviceInner {
    pub shared: Arc<VulkanShared>,
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub budget_enabled: bool,
    pub swapchain_enabled: bool,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// Queue needs external sync: every submit or present on `queue` holds this mutex.
    pub queue_lock: Mutex<()>,
    /// Numbers submissions to `queue` for deferred frees.
    pub serials: QueueSerials,
    /// Set once the device is seen lost; it never comes back.
    pub lost: AtomicBool,
    /// Unsignalled fence, whose status is the one thing that reports device loss.
    pub probe: vk::Fence,
    pub features: Features,
    pub limits: DeviceLimits,
    pub graphics: bool,
    pub dual_src_blend: bool,
    pub fill_mode_non_solid: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    /// `Depth24PlusStencil8` here: `D24_UNORM_S8_UINT` if present, else `D32_SFLOAT_S8_UINT`. AMD
    /// exposes no D24S8. "24 plus" allows the wider depth.
    pub depth24_stencil8: vk::Format,
    /// `(min, max)` subgroup sizes a compute pipeline may require (`VK_EXT_subgroup_size_control`).
    pub compute_subgroup_sizes: Option<(u32, u32)>,
    pub coop_matrix: bool,
    /// `VK_KHR_pipeline_executable_properties` enabled: stats-capture pipelines report
    /// resource use.
    pub pipeline_executable_info: bool,
    /// `VK_KHR_dynamic_rendering` loader: `cmd_begin_rendering`/`cmd_end_rendering`, no
    /// `vk::RenderPass` or `vk::Framebuffer`.
    pub dynamic_rendering: khr::dynamic_rendering::Device,
    /// `VK_EXT_debug_utils` commands, if the instance enabled it.
    pub debug_utils: Option<ext::debug_utils::Device>,
}

unsafe impl Send for VulkanDeviceInner {}
unsafe impl Sync for VulkanDeviceInner {}

/// Depth-stencil format for `Format::Depth24PlusStencil8`: first of D24S8, D32S8 usable as an
/// optimal-tiling depth attachment.
fn pick_depth24_stencil8(instance: &ash::Instance, physical: vk::PhysicalDevice) -> vk::Format {
    [
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
    ]
    .into_iter()
    .find(|&format| {
        let props = unsafe { instance.get_physical_device_format_properties(physical, format) };
        props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    })
    .unwrap_or(vk::Format::D24_UNORM_S8_UINT)
}

impl VulkanDeviceInner {
    pub(crate) fn vk_format(&self, format: zengpu_hal::Format) -> vk::Format {
        match format {
            zengpu_hal::Format::Depth24PlusStencil8 => self.depth24_stencil8,
            other => hal_format_to_vk(other),
        }
    }

    /// Device-local heap sizes with the driver's budget and usage. `None` without
    /// `VK_EXT_memory_budget`.
    pub(crate) fn memory_budget(&self) -> Option<MemoryBudget> {
        if !self.budget_enabled {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let (count, heaps) = {
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                self.shared
                    .instance
                    .get_physical_device_memory_properties2(self.physical, &mut props);
            }
            (
                props.memory_properties.memory_heap_count as usize,
                props.memory_properties.memory_heaps,
            )
        };
        let mut out = MemoryBudget {
            device_local: 0,
            budget: 0,
            usage: 0,
        };
        for (i, heap) in heaps.iter().enumerate().take(count) {
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                out.device_local = out.device_local.saturating_add(heap.size);
                out.budget = out.budget.saturating_add(budget.heap_budget[i]);
                out.usage = out.usage.saturating_add(budget.heap_usage[i]);
            }
        }
        Some(out)
    }

    pub(crate) fn is_lost(&self) -> bool {
        if self.lost.load(Ordering::Acquire) {
            return true;
        }
        let status = unsafe { self.device.get_fence_status(self.probe) };
        let lost = matches!(status, Err(vk::Result::ERROR_DEVICE_LOST));
        if lost {
            self.lost.store(true, Ordering::Release);
        }
        lost
    }
}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.probe, None);
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
    pub texel_bytes: u32,
    /// Per array layer, one bit per mip that holds texels in the resting layout. A partial write to
    /// a level not ready leaves the rest undefined.
    pub ready: Vec<u32>,
    /// Resting layout between operations: `SHADER_READ_ONLY_OPTIMAL` if only sampled, else
    /// `GENERAL` (storage is written while others sample; read-only needs `SAMPLED` usage).
    pub resting: vk::ImageLayout,
    /// Mip 0 view for the storage-image table, when `view` has more levels than a storage
    /// descriptor takes. Else null.
    pub storage_view: vk::ImageView,
}

impl VulkanTexture {
    pub(crate) fn resting_layout(usage: vk::ImageUsageFlags) -> vk::ImageLayout {
        if usage.contains(vk::ImageUsageFlags::SAMPLED)
            && !usage.contains(vk::ImageUsageFlags::STORAGE)
        {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.storage_view != vk::ImageView::null() {
                device.destroy_image_view(self.storage_view, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }

    pub(crate) fn is_ready(&self, mip_level: u32, layer: u32) -> bool {
        self.ready
            .get(layer as usize)
            .is_some_and(|bits| mip_level < 32 && bits & (1 << mip_level) != 0)
    }

    pub(crate) fn mark_ready(&mut self, mip_level: u32, layer: u32) {
        if let Some(bits) = self.ready.get_mut(layer as usize)
            && mip_level < 32
        {
            *bits |= 1 << mip_level;
        }
    }
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
                unsafe { texture.destroy(&inner.device) };
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

/// Vulkan logical device.
///
/// Drop every surface, offscreen target and command list from this device before the device. Debug
/// builds panic on a violation. Release builds silently destroy shared resources under survivors.
pub struct VulkanDevice {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    pub(crate) buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    pub(crate) textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    samplers: Arc<Mutex<SlotMap<marker::Sampler, vk::Sampler>>>,
    shaders: Mutex<SlotMap<marker::Shader, (vk::ShaderModule, bool)>>,
    pub(crate) pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    /// Render targets (swapchain images, offscreen textures) that commands attach to.
    pub(crate) render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    pub(crate) query_pools: Arc<Mutex<SlotMap<marker::QueryPool, VulkanQueryPool>>>,
    bindless: BindlessState,
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<PathBuf>,
    pub(crate) cmd_list_pool: Arc<CmdListPool>,
    fence_pool: Arc<FencePool>,
    lifetime: Arc<Mutex<VulkanLifetimeState>>,
    /// Reused for host-visible uploads.
    staging: Mutex<transfer::StagingRing>,
    transfer_slots: Mutex<Option<dispatch::TransferSlots>>,
    orphans: Orphans,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    /// Dropped submissions whose work is done. Drop them after unlocking: a drop can add to
    /// this list.
    pub(crate) fn reap_orphans(&self) {
        let done: Vec<VulkanSubmission> = {
            let mut orphans = self.orphans.lock().unwrap();
            let (done, running): (Vec<_>, Vec<_>) = mem::take(&mut *orphans)
                .into_iter()
                .partition(|orphan| !matches!(orphan.poll(), Ok(SubmissionStatus::Pending)));
            *orphans = running;
            done
        };
        drop(done);
    }

    /// Marks the device lost as a driver reset would: `is_lost` reports it, surface acquire and
    /// present fail with `DeviceLost`. For testing recovery without hanging a GPU. No way back.
    pub fn simulate_loss(&self) {
        self.inner.lost.store(true, Ordering::Release);
    }

    pub(crate) fn buffer_busy(&self, index: u32) -> bool {
        let completed = self.inner.serials.completed();
        self.lifetime.lock().unwrap().buffer_busy(index, completed)
    }

    pub(crate) fn collect_garbage(&self) {
        let ready = self
            .lifetime
            .lock()
            .unwrap()
            .take_ready(self.inner.serials.completed());
        if ready.is_empty() {
            return;
        }
        destroy_deferred_vulkan(
            &self.inner,
            &self.buffers,
            &self.textures,
            &self.samplers,
            &self.pipelines,
            ready,
        );
    }

    pub(crate) fn create(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
        extra_extensions: &[*const i8],
        needs_graphics: bool,
    ) -> Result<Self> {
        let pipeline_cache_path = req.pipeline_cache_path.clone();
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
        let mut supported_subgroup_size =
            vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::default();
        supported_desc_idx.p_next = &mut supported_dynamic_rendering as *mut _ as *mut c_void;
        supported_dynamic_rendering.p_next = &mut supported_atomic_float as *mut _ as *mut c_void;
        supported_atomic_float.p_next = &mut supported_subgroup_size as *mut _ as *mut c_void;
        let mut supported_coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut supported_storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut supported_storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
        let mut supported_f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut supported_wg_explicit =
            vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
        let mut supported_pipe_exec =
            vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
        supported_subgroup_size.p_next = &mut supported_coop as *mut _ as *mut c_void;
        supported_coop.p_next = &mut supported_storage16 as *mut _ as *mut c_void;
        supported_storage16.p_next = &mut supported_storage8 as *mut _ as *mut c_void;
        supported_storage8.p_next = &mut supported_f16i8 as *mut _ as *mut c_void;
        supported_f16i8.p_next = &mut supported_wg_explicit as *mut _ as *mut c_void;
        supported_wg_explicit.p_next = &mut supported_pipe_exec as *mut _ as *mut c_void;
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
        let budget_enabled = supports_extension(ext::memory_budget::NAME);
        if budget_enabled {
            extensions.push(ext::memory_budget::NAME.as_ptr());
        }
        let compute_subgroup_sizes = if supports_extension(ext::subgroup_size_control::NAME)
            && supported_subgroup_size.subgroup_size_control == vk::TRUE
        {
            let mut sizes = vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceProperties2 {
                p_next: &mut sizes as *mut _ as *mut c_void,
                ..Default::default()
            };
            unsafe {
                shared
                    .instance
                    .get_physical_device_properties2(physical, &mut props)
            };
            sizes
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                .then_some((sizes.min_subgroup_size, sizes.max_subgroup_size))
        } else {
            None
        };
        if compute_subgroup_sizes.is_some() {
            extensions.push(ext::subgroup_size_control::NAME.as_ptr());
        }

        let supported_features = supported_features2.features;
        let storage_images = supported_features.shader_storage_image_read_without_format
            == vk::TRUE
            && supported_features.shader_storage_image_write_without_format == vk::TRUE
            && supported_features.shader_storage_image_array_dynamic_indexing == vk::TRUE
            && supported_desc_idx.descriptor_binding_storage_image_update_after_bind == vk::TRUE;
        let mut device_limits = device_limits;
        if !storage_images {
            device_limits.max_storage_textures = 0;
        }
        let on = |enabled: bool| if enabled { vk::TRUE } else { vk::FALSE };
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

        let mut desc_idx = vk::PhysicalDeviceDescriptorIndexingFeatures {
            shader_storage_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
            descriptor_binding_storage_buffer_update_after_bind: vk::TRUE,
            descriptor_binding_sampled_image_update_after_bind: vk::TRUE,
            descriptor_binding_storage_image_update_after_bind: on(storage_images),
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
                shader_storage_image_array_dynamic_indexing: on(storage_images),
                shader_storage_image_read_without_format: on(storage_images),
                shader_storage_image_write_without_format: on(storage_images),
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
        let mut subgroup_size_feat = vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT {
            subgroup_size_control: vk::TRUE,
            ..Default::default()
        };
        if compute_subgroup_sizes.is_some() {
            subgroup_size_feat.p_next = features2.p_next;
            features2.p_next = &mut subgroup_size_feat as *mut _ as *mut c_void;
        }
        let f16i8_supported = supported_storage16.storage_buffer16_bit_access == vk::TRUE
            && supported_storage8.storage_buffer8_bit_access == vk::TRUE
            && supported_f16i8.shader_float16 == vk::TRUE
            && supported_f16i8.shader_int8 == vk::TRUE;
        let coop_supported = supports_extension(khr::cooperative_matrix::NAME)
            && supported_coop.cooperative_matrix == vk::TRUE
            && f16i8_supported;
        let mut coop_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
            cooperative_matrix: vk::TRUE,
            ..Default::default()
        };
        let mut storage16_feat = vk::PhysicalDevice16BitStorageFeatures {
            storage_buffer16_bit_access: vk::TRUE,
            ..Default::default()
        };
        let mut storage8_feat = vk::PhysicalDevice8BitStorageFeatures {
            storage_buffer8_bit_access: vk::TRUE,
            ..Default::default()
        };
        let mut f16i8_feat = vk::PhysicalDeviceShaderFloat16Int8Features {
            shader_float16: vk::TRUE,
            shader_int8: vk::TRUE,
            ..Default::default()
        };
        if f16i8_supported {
            f16i8_feat.p_next = features2.p_next;
            storage8_feat.p_next = &mut f16i8_feat as *mut _ as *mut c_void;
            storage16_feat.p_next = &mut storage8_feat as *mut _ as *mut c_void;
            features2.p_next = &mut storage16_feat as *mut _ as *mut c_void;
        }
        let wg_explicit_supported = supports_extension(khr::workgroup_memory_explicit_layout::NAME)
            && supported_wg_explicit.workgroup_memory_explicit_layout == vk::TRUE;
        let pipe_exec_supported = supports_extension(khr::pipeline_executable_properties::NAME)
            && supported_pipe_exec.pipeline_executable_info == vk::TRUE;
        let mut pipe_exec_feat = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR {
            pipeline_executable_info: vk::TRUE,
            ..Default::default()
        };
        if pipe_exec_supported {
            extensions.push(khr::pipeline_executable_properties::NAME.as_ptr());
            pipe_exec_feat.p_next = features2.p_next;
            features2.p_next = &mut pipe_exec_feat as *mut _ as *mut c_void;
        }
        let mut wg_explicit_feat = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR {
            workgroup_memory_explicit_layout: vk::TRUE,
            workgroup_memory_explicit_layout8_bit_access: supported_wg_explicit
                .workgroup_memory_explicit_layout8_bit_access,
            workgroup_memory_explicit_layout16_bit_access: supported_wg_explicit
                .workgroup_memory_explicit_layout16_bit_access,
            ..Default::default()
        };
        if wg_explicit_supported {
            extensions.push(khr::workgroup_memory_explicit_layout::NAME.as_ptr());
            wg_explicit_feat.p_next = features2.p_next;
            features2.p_next = &mut wg_explicit_feat as *mut _ as *mut c_void;
        }
        if coop_supported {
            extensions.push(khr::cooperative_matrix::NAME.as_ptr());
            coop_feat.p_next = features2.p_next;
            features2.p_next = &mut coop_feat as *mut _ as *mut c_void;
        }

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
            p_enabled_features: null(),
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

        let debug_utils = shared
            .debug_utils_enabled()
            .then(|| ext::debug_utils::Device::new(&shared.instance, &device));
        let probe = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| map_vk_err("vkCreateFence", e))?;
        let depth24_stencil8 = pick_depth24_stencil8(&shared.instance, physical);
        let inner = Arc::new(VulkanDeviceInner {
            shared,
            device,
            physical,
            memory_properties,
            budget_enabled,
            swapchain_enabled: extra_extensions
                .iter()
                .any(|&name| unsafe { CStr::from_ptr(name) == khr::swapchain::NAME }),
            queue_family,
            queue,
            queue_lock: Mutex::new(()),
            serials: QueueSerials::default(),
            lost: AtomicBool::new(false),
            probe,
            features: available_features,
            limits: device_limits,
            graphics: needs_graphics,
            dual_src_blend,
            fill_mode_non_solid,
            sampler_anisotropy,
            max_sampler_anisotropy,
            depth24_stencil8,
            compute_subgroup_sizes,
            coop_matrix: coop_supported,
            pipeline_executable_info: pipe_exec_supported,
            dynamic_rendering,
            debug_utils,
        });

        let bindless = create_bindless(&inner.device, inner.limits)?;
        let cmd_list_pool = Arc::new(CmdListPool::new(Arc::clone(&inner))?);
        let fence_pool = Arc::new(FencePool::new(Arc::clone(&inner)));
        let initial_cache_data =
            pipeline_cache_path
                .as_deref()
                .and_then(|path| match fs::read(path) {
                    Ok(data) if !data.is_empty() => Some(data),
                    Ok(_) => None,
                    Err(e) if e.kind() == ErrorKind::NotFound => None,
                    Err(e) => {
                        log::warn!(
                            "[zengpu-vulkan] failed to read pipeline cache {}: {e}",
                            path.display()
                        );
                        None
                    }
                });
        let cache_info = initial_cache_data
            .as_deref()
            .map_or_else(vk::PipelineCacheCreateInfo::default, |data| {
                vk::PipelineCacheCreateInfo::default().initial_data(data)
            });
        let pipeline_cache = match unsafe { inner.device.create_pipeline_cache(&cache_info, None) }
        {
            Ok(cache) => cache,
            Err(e) if initial_cache_data.is_some() => {
                log::warn!(
                    "[zengpu-vulkan] persisted pipeline cache was rejected ({e}); starting empty"
                );
                unsafe {
                    inner
                        .device
                        .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                }
                .map_err(|e| GpuError::Backend(format!("vkCreatePipelineCache: {e}")))?
            }
            Err(e) => return Err(GpuError::Backend(format!("vkCreatePipelineCache: {e}"))),
        };

        Ok(Self {
            inner,
            buffers: Arc::new(Mutex::new(SlotMap::new())),
            textures: Arc::new(Mutex::new(SlotMap::new())),
            samplers: Arc::new(Mutex::new(SlotMap::new())),
            shaders: Mutex::new(SlotMap::new()),
            pipelines: Arc::new(Mutex::new(SlotMap::new())),
            render_targets: Arc::new(Mutex::new(SlotMap::new())),
            query_pools: Arc::new(Mutex::new(SlotMap::new())),
            bindless,
            pipeline_cache,
            pipeline_cache_path,
            cmd_list_pool,
            fence_pool,
            lifetime: Arc::new(Mutex::new(VulkanLifetimeState::default())),
            staging: Mutex::new(transfer::StagingRing::default()),
            transfer_slots: Mutex::new(None),
            orphans: Arc::default(),
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
    /// Texture slots in the shared bindless budget, capped by `MAX_BINDLESS_TEXTURES`.
    pub fn bindless_texture_capacity(&self) -> u32 {
        self.bindless.texture_capacity
    }

    /// Texture slots in use from the shared budget; compare with `MAX_BINDLESS_TEXTURES`.
    pub fn bindless_texture_used(&self) -> u32 {
        self.bindless
            .bound_textures
            .lock()
            .unwrap()
            .iter()
            .filter(|&&bound| bound)
            .count() as u32
    }

    /// Storage-buffer slots in the shared bindless budget, capped by `MAX_BINDLESS_BUFFERS`.
    pub fn bindless_buffer_capacity(&self) -> u32 {
        self.bindless.buffer_capacity
    }

    /// Storage-buffer slots in use; compare with `MAX_BINDLESS_BUFFERS`.
    pub fn bindless_buffer_used(&self) -> u32 {
        let buffers = self.buffers.lock().unwrap();
        (0..self.bindless.buffer_capacity)
            .filter_map(|index| buffers.get_by_slot_index(index))
            .filter(|buffer| buffer.usage.contains(BufferUsage::STORAGE))
            .count() as u32
    }

    /// Waits for the GPU to finish every timestamp in `pool`. Error for a stale handle.
    pub fn resolve_query_pool(&self, pool: QueryPoolHandle) -> Result<Vec<u64>> {
        let query_pools = self.query_pools.lock().unwrap();
        let query_pool = query_pools
            .get(pool)
            .ok_or_else(|| GpuError::Backend("stale query pool handle".to_string()))?;
        let mut timestamps = vec![0; query_pool.count as usize];
        unsafe {
            self.inner
                .device
                .get_query_pool_results(
                    query_pool.pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|e| GpuError::Backend(format!("vkGetQueryPoolResults: {e}")))?;
        }
        Ok(timestamps)
    }
}

fn bindless_capacities(limits: DeviceLimits) -> Result<(u32, u32, u32)> {
    let total = limits.max_update_after_bind_descriptors;
    let buffer_capacity = MAX_BINDLESS_BUFFERS
        .min(limits.max_storage_buffers / SSBO_TYPED_TABLE_COUNT)
        .min(total.saturating_sub(1) / SSBO_TYPED_TABLE_COUNT);
    let texture_capacity = MAX_BINDLESS_TEXTURES
        .min(limits.max_sampled_textures)
        .min(total.saturating_sub(buffer_capacity * SSBO_TYPED_TABLE_COUNT));
    if buffer_capacity == 0 || texture_capacity == 0 {
        return Err(GpuError::Unsupported(format!(
            "bindless descriptor limits are too small: storage={}, textures={}, total={total}",
            limits.max_storage_buffers, limits.max_sampled_textures
        )));
    }
    let storage_capacity = MAX_BINDLESS_TEXTURES
        .min(limits.max_storage_textures)
        .min(total - buffer_capacity * SSBO_TYPED_TABLE_COUNT - texture_capacity);
    Ok((buffer_capacity, texture_capacity, storage_capacity))
}

fn create_bindless(dev: &ash::Device, limits: DeviceLimits) -> Result<BindlessState> {
    let (buffer_capacity, texture_capacity, storage_capacity) = bindless_capacities(limits)?;
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding {
            binding: SSBO_BINDING_F32,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: SSBO_BINDING_TEXTURES,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: texture_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: SSBO_BINDING_U32,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: SSBO_BINDING_I32,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
    ];
    if storage_capacity != 0 {
        bindings.push(vk::DescriptorSetLayoutBinding {
            binding: STORAGE_IMAGE_BINDING,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: storage_capacity,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        });
    }
    let binding_flags = vec![
        vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        bindings.len()
    ];
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

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffer_capacity * SSBO_TYPED_TABLE_COUNT,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: texture_capacity,
        },
    ];
    if storage_capacity != 0 {
        pool_sizes.push(vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: storage_capacity,
        });
    }
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
        storage_capacity,
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
        max_storage_textures: descriptor
            .max_per_stage_descriptor_update_after_bind_storage_images
            .min(descriptor.max_descriptor_set_update_after_bind_storage_images),
        max_update_after_bind_descriptors: descriptor
            .max_update_after_bind_descriptors_in_all_pools,
        max_memory_allocations: limits.max_memory_allocation_count,
        timestamp_supported,
        timestamp_period_ns: limits.timestamp_period,
        compute_units: 0,
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
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        self.reap_orphans();
        if let Some(slots) = self.transfer_slots.get_mut().ok().and_then(Option::take) {
            slots.destroy(&self.inner.device);
        }
        #[cfg(debug_assertions)]
        {
            let mut outstanding = Vec::new();
            if Arc::strong_count(&self.inner) > 3 {
                outstanding.push("inner");
            }
            if Arc::strong_count(&self.buffers) > 1 {
                outstanding.push("buffers");
            }
            if Arc::strong_count(&self.textures) > 1 {
                outstanding.push("textures");
            }
            if Arc::strong_count(&self.samplers) > 1 {
                outstanding.push("samplers");
            }
            if Arc::strong_count(&self.pipelines) > 1 {
                outstanding.push("pipelines");
            }
            if Arc::strong_count(&self.render_targets) > 1 {
                outstanding.push("render_targets");
            }
            if Arc::strong_count(&self.query_pools) > 1 {
                outstanding.push("query_pools");
            }
            if Arc::strong_count(&self.cmd_list_pool) > 1 {
                outstanding.push("cmd_list_pool");
            }
            if Arc::strong_count(&self.fence_pool) > 1 {
                outstanding.push("fence_pool");
            }
            if Arc::strong_count(&self.lifetime) > 1 {
                outstanding.push("lifetime");
            }
            if !outstanding.is_empty() {
                panic!(
                    "VulkanDevice dropped with outstanding shared owners of {}: drop Surface, OffscreenTarget, and command-list handles before dropping the device",
                    outstanding.join(", ")
                );
            }
        }

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
            unsafe { tex.destroy(&self.inner.device) };
        }
        let mut samplers = self.samplers.lock().unwrap();
        for s in samplers.drain() {
            unsafe { self.inner.device.destroy_sampler(s, None) };
        }
        let mut shaders = self.shaders.lock().unwrap();
        for (m, _) in shaders.drain() {
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
        let mut query_pools = self.query_pools.lock().unwrap();
        for query_pool in query_pools.drain() {
            unsafe { self.inner.device.destroy_query_pool(query_pool.pool, None) };
        }
        unsafe {
            if let Some(path) = &self.pipeline_cache_path {
                match self
                    .inner
                    .device
                    .get_pipeline_cache_data(self.pipeline_cache)
                {
                    Ok(data) => {
                        if let Err(e) = fs::write(path, data) {
                            log::warn!(
                                "[zengpu-vulkan] failed to persist pipeline cache {}: {e}",
                                path.display()
                            );
                        }
                    }
                    Err(e) => log::warn!(
                        "[zengpu-vulkan] failed to retrieve pipeline cache data for {}: {e}",
                        path.display()
                    ),
                }
            }
            self.inner
                .device
                .destroy_pipeline_cache(self.pipeline_cache, None);
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
    use std::collections::VecDeque;
    use std::mem::size_of_val;
    use std::num::NonZeroIsize;
    use std::slice::from_raw_parts;
    use std::sync::MutexGuard;
    use std::thread::spawn;
    use std::{env, ops::Deref, process};

    use super::*;
    use crate::instance::VulkanInstance;
    use zengpu_hal::{
        AdapterRequest, Bindings, BlendState, ColorAttachment, ColorTargetState, DepthState,
        DeviceRequest, FilterMode, Format, GpuInstance, GraphicsDevice, GraphicsPipelineDesc,
        LoadOp, PipelineHandle, PrimitiveTopology, RasterState, RenderCommands, RenderPassDesc,
        SamplerDesc, Scalar, TargetHandle, TexDim, TextureHandle, TextureRegion, Viewport,
        ViewportScissor,
    };

    struct TestDevice {
        dev: Box<dyn GpuDevice>,
        _guard: MutexGuard<'static, ()>,
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

    #[test]
    fn invalid_texture_mips_return_errors_without_poisoning_the_device() {
        use zengpu_hal::TextureWrite;
        let Some(dev) = try_device() else { return };
        let tex = dev
            .create_texture(TextureDesc {
                width: 1,
                height: 1,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        for mip_level in [1, 32, u32::MAX] {
            let mut region = TextureRegion::rect(0, 0, 1, 1);
            region.mip_level = mip_level;
            assert!(
                dev.write_textures(&[TextureWrite {
                    texture: tex,
                    region,
                    data: &[0; 4],
                    bytes_per_row: 4,
                }])
                .is_err()
            );
        }
        dev.upload_texture_data(tex, &[0; 4]).unwrap();
        dev.destroy_texture(tex);
    }

    #[test]
    fn headless_devices_reject_surfaces_before_touching_native_handles() {
        let _guard = crate::test_gpu_lock();
        let Ok(instance) = VulkanInstance::new_with_surface() else {
            return;
        };
        let Some(adapter) = instance.request_graphics_adapter() else {
            return;
        };
        let Ok(device) = adapter.open_headless(DeviceRequest::default()) else {
            return;
        };
        let handles = zengpu_hal::WindowHandles::from_raw(
            zen_window_handle::WindowHandle::Win32(zen_window_handle::Win32WindowHandle {
                hwnd: NonZeroIsize::new(1).unwrap(),
                hinstance: None,
            }),
            zen_window_handle::DisplayHandle::Windows,
        );
        let outcome = device.create_surface(
            &handles,
            zengpu_hal::SurfaceConfig {
                width: 64,
                height: 64,
                format: Format::Bgra8Unorm,
                present_mode: zengpu_hal::PresentMode::Fifo,
            },
        );
        assert!(matches!(outcome, Err(GpuError::Backend(message))
            if message.contains("open_with_surface")));
        device.wait_idle().expect("device remains usable");
    }

    #[test]
    fn texture_regions_round_trip_without_touching_the_rest() {
        let Some(dev) = try_device() else {
            return;
        };
        if !dev.capabilities().transfers.texture_region_write {
            return;
        }
        let tex = dev
            .create_texture(TextureDesc {
                width: 4,
                height: 4,
                depth: 1,
                format: zengpu_hal::Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let red = [255u8, 0, 0, 255];
        let green = [0u8, 255, 0, 255];
        let blue = [0u8, 0, 255, 255];
        dev.upload_texture_data(tex, &red.repeat(16)).unwrap();
        let mut wide = Vec::new();
        for _ in 0..2 {
            wide.extend_from_slice(&green.repeat(2));
            wide.extend_from_slice(&[9, 9, 9, 9]);
        }
        let write = zengpu_hal::TextureWrite {
            texture: tex,
            region: zengpu_hal::TextureRegion::rect(1, 1, 2, 2),
            data: &wide,
            bytes_per_row: 12,
        };
        let wrote = dev.write_textures(&[write]).unwrap();
        let src = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::TRANSFER_SRC,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        dev.write_buffer(src, 0, &blue).unwrap();
        let copied = dev
            .copy_buffer_to_texture(src, 0, 4, tex, zengpu_hal::TextureRegion::rect(0, 0, 1, 1))
            .unwrap();
        let out = dev.create_buffer(rw_desc(64)).unwrap();
        let read = dev
            .copy_texture_region_to_buffer(
                tex,
                zengpu_hal::TextureRegion::rect(0, 0, 4, 4),
                out,
                0,
                16,
            )
            .unwrap();
        read.wait(Duration::from_secs(5)).unwrap();
        wrote.wait(Duration::from_secs(5)).unwrap();
        copied.wait(Duration::from_secs(5)).unwrap();
        let px = dev.read_buffer(out, 0, 64).unwrap();
        let at = |x: usize, y: usize| {
            [
                px[(y * 4 + x) * 4],
                px[(y * 4 + x) * 4 + 1],
                px[(y * 4 + x) * 4 + 2],
                px[(y * 4 + x) * 4 + 3],
            ]
        };
        assert_eq!(at(0, 0), blue);
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            assert_eq!(at(x, y), green, "({x}, {y}) is in the tile");
        }
        for (x, y) in [(3, 0), (0, 3), (3, 3), (3, 1), (1, 3)] {
            assert_eq!(at(x, y), red, "({x}, {y}) kept its texel");
        }
        let outside = zengpu_hal::TextureWrite {
            region: zengpu_hal::TextureRegion::rect(3, 3, 2, 2),
            ..write
        };
        assert!(dev.write_textures(&[outside]).is_err());
        dev.destroy_buffer(src);
        dev.destroy_buffer(out);
        dev.destroy_texture(tex);
    }

    /// Dropping a pending submission hands it over instead of waiting. The next submit recycles it.
    #[test]
    fn dropping_pending_work_is_kept_until_done() {
        let Some(dev) = try_device() else {
            return;
        };
        let vk_dev = dev.as_any().downcast_ref::<VulkanDevice>().unwrap();
        if !dev.capabilities().transfers.buffer_texture_copy {
            return;
        }
        let tex = dev
            .create_texture(TextureDesc {
                width: 4096,
                height: 4096,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let size = 4096 * 4096 * 4;
        let out = dev.create_buffer(rw_desc(size)).unwrap();
        let region = zengpu_hal::TextureRegion::rect(0, 0, 4096, 4096);
        let copy = dev
            .copy_texture_region_to_buffer(tex, region, out, 0, 4096 * 4)
            .unwrap();
        let pending = matches!(copy.poll(), Ok(SubmissionStatus::Pending));
        drop(copy);
        if pending {
            assert_eq!(
                vk_dev.orphans.lock().unwrap().len(),
                1,
                "kept, not waited for"
            );
        }
        unsafe { vk_dev.inner.device.device_wait_idle().unwrap() };
        vk_dev.reap_orphans();
        assert!(
            vk_dev.orphans.lock().unwrap().is_empty(),
            "recycled once done"
        );
        dev.destroy_buffer(out);
        dev.destroy_texture(tex);
    }

    #[test]
    fn readback_polls_while_frames_keep_going() {
        let Some(dev) = try_device() else {
            return;
        };
        if !dev.capabilities().transfers.buffer_texture_copy {
            return;
        }
        let desc = |width, height| TextureDesc {
            width,
            height,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        };
        let tex = dev.create_texture(desc(256, 256)).unwrap();
        let scratch = dev.create_texture(desc(16, 16)).unwrap();
        let pixels: Vec<u8> = (0..256 * 256u32).flat_map(|i| i.to_le_bytes()).collect();
        dev.upload_texture_data(tex, &pixels).unwrap();
        let tile = vec![0x7f; 16 * 16 * 4];
        let out = dev
            .create_buffer(rw_desc(u64::try_from(pixels.len()).unwrap()))
            .unwrap();

        let read = dev
            .copy_texture_region_to_buffer(
                tex,
                zengpu_hal::TextureRegion::rect(0, 0, 256, 256),
                out,
                0,
                256 * 4,
            )
            .unwrap();
        let mut frame_work: Vec<Submission> = Vec::new();
        let mut status = read.poll().unwrap();
        let mut frames = 0;
        while status == SubmissionStatus::Pending {
            frames += 1;
            assert!(frames < 100_000, "readback still pending");
            frame_work.push(
                dev.write_textures(&[zengpu_hal::TextureWrite {
                    texture: scratch,
                    region: zengpu_hal::TextureRegion::rect(0, 0, 16, 16),
                    data: &tile,
                    bytes_per_row: 64,
                }])
                .unwrap(),
            );
            frame_work.retain(|work| work.poll().unwrap() == SubmissionStatus::Pending);
            status = read.poll().unwrap();
        }
        let mut back = vec![0u8; pixels.len()];
        dev.read_buffer_into(out, 0, &mut back).unwrap();
        assert!(back == pixels, "readback bytes differ from the upload");
        for work in frame_work {
            work.wait(Duration::from_secs(5)).unwrap();
        }
        dev.destroy_buffer(out);
        dev.destroy_texture(scratch);
        dev.destroy_texture(tex);
    }

    #[test]
    fn a_simulated_loss_sticks() {
        let _guard = crate::test_gpu_lock();
        let Some(dev) = VulkanInstance::new()
            .ok()
            .and_then(|i| i.request_vulkan_adapter())
            .and_then(|a| a.open_headless(DeviceRequest::default()).ok())
        else {
            return;
        };
        assert!(!GpuDevice::is_lost(&dev));
        dev.simulate_loss();
        assert!(GpuDevice::is_lost(&dev));
    }

    /// Unmappable buffers, through staging: across chunk boundaries and at odd offsets.
    #[test]
    fn gpu_only_buffer_round_trips_through_staging() {
        let Some(dev) = try_device() else {
            return;
        };
        let size = (40u64 << 20) + 12;
        let buf = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let data: Vec<u8> = (0..size).map(|i| (i % 253) as u8).collect();
        dev.write_buffer(buf, 0, &data).unwrap();
        assert_eq!(dev.read_buffer(buf, 0, size).unwrap(), data);
        dev.write_buffer(buf, 7, &[1, 2, 3]).unwrap();
        let mut window = [0u8; 5];
        dev.read_buffer_into(buf, 6, &mut window).unwrap();
        assert_eq!(window, [data[6], 1, 2, 3, data[10]]);
        let inline: Vec<u8> = (0..64).map(|i| 200 - i).collect();
        dev.write_buffer(buf, 4096, &inline).unwrap();
        assert_eq!(dev.read_buffer(buf, 4092, 72).unwrap()[4..68], inline[..]);
        assert!(dev.read_buffer(buf, size - 2, 4).is_err(), "past the end");
        dev.destroy_buffer(buf);
    }

    /// Without ReBAR a large `CpuToGpu` buffer must still be made in host memory, not fail on the
    /// small aperture.
    #[test]
    fn large_cpu_to_gpu_buffer_survives_small_bar() {
        let Some(dev) = try_device() else {
            return;
        };
        let size = 64u64 << 20;
        let buf = dev
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::CpuToGpu,
            })
            .unwrap();
        let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        dev.write_buffer(buf, 0, &data).unwrap();
        let mut back = vec![0u8; 4096];
        dev.read_buffer_into(buf, size - 4096, &mut back).unwrap();
        assert_eq!(&back[..], &data[data.len() - 4096..]);
        dev.destroy_buffer(buf);
    }

    /// AMD has no D24S8: `Depth24PlusStencil8` must still be creatable, backed by D32S8.
    #[test]
    fn depth24_plus_stencil8_is_creatable_everywhere() {
        let Some(dev) = try_device() else {
            return;
        };
        let backing = dev
            .as_any()
            .downcast_ref::<VulkanDevice>()
            .unwrap()
            .inner
            .depth24_stencil8;
        assert!(
            backing == vk::Format::D24_UNORM_S8_UINT || backing == vk::Format::D32_SFLOAT_S8_UINT,
            "{backing:?}"
        );
        let tex = dev
            .create_texture(zengpu_hal::TextureDesc {
                width: 64,
                height: 64,
                depth: 1,
                format: zengpu_hal::Format::Depth24PlusStencil8,
                usage: zengpu_hal::TextureUsage::RENDER_TARGET,
                samples: 1,
                dimension: zengpu_hal::TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        dev.destroy_texture(tex);
        assert_eq!(
            crate::from_vk_format(backing),
            Some(zengpu_hal::Format::Depth24PlusStencil8)
        );
    }

    /// Mandatory color formats: sampled and drawn into.
    #[test]
    fn colour_formats_report_their_uses() {
        let Some(dev) = try_device() else {
            return;
        };
        for format in zengpu_hal::Format::ALL {
            let caps = dev.format_caps(format);
            if format.is_depth_stencil() {
                assert!(caps.render_target, "{format:?} as depth");
                continue;
            }
            assert!(caps.sampled && caps.render_target, "{format:?}: {caps:?}");
        }
        let half = dev.format_caps(zengpu_hal::Format::Rgba16Float);
        assert!(
            half.filterable && half.blendable && half.storage,
            "{half:?}"
        );
    }

    /// Upload, region write and readback keep every byte, at each format's texel size.
    #[test]
    fn colour_formats_round_trip_through_region_writes() {
        let Some(dev) = try_device() else {
            return;
        };
        if !dev.capabilities().transfers.buffer_texture_copy {
            return;
        }
        for format in zengpu_hal::Format::ALL {
            if format.is_depth_stencil() || !dev.format_caps(format).sampled {
                continue;
            }
            round_trip(&*dev, format);
        }
    }

    fn round_trip(dev: &dyn GpuDevice, format: zengpu_hal::Format) {
        const SIZE: u32 = 8;
        let texel = format.bytes_per_texel();
        let row = SIZE * texel;
        let tex = dev
            .create_texture(TextureDesc {
                width: SIZE,
                height: SIZE,
                depth: 1,
                format,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_DST
                    | TextureUsage::TRANSFER_SRC,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let base: Vec<u8> = (0..row * SIZE).map(|i| (i % 0x61) as u8).collect();
        dev.upload_texture_data(tex, &base).unwrap();
        let patch = vec![0x3Cu8; (4 * 4 * texel) as usize];
        let write = zengpu_hal::TextureWrite {
            texture: tex,
            region: zengpu_hal::TextureRegion::rect(2, 2, 4, 4),
            data: &patch,
            bytes_per_row: 4 * texel,
        };
        dev.write_textures(&[write])
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();
        let size = u64::from(row * SIZE);
        let out = dev.create_buffer(rw_desc(size)).unwrap();
        dev.copy_texture_region_to_buffer(
            tex,
            zengpu_hal::TextureRegion::rect(0, 0, SIZE, SIZE),
            out,
            0,
            row,
        )
        .unwrap()
        .wait(Duration::from_secs(5))
        .unwrap();
        let got = dev.read_buffer(out, 0, size).unwrap();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let at = ((y * row + x * texel) as usize)..((y * row + (x + 1) * texel) as usize);
                let inside = (2..6).contains(&x) && (2..6).contains(&y);
                let want: &[u8] = if inside {
                    &patch[..texel as usize]
                } else {
                    &base[at.clone()]
                };
                assert_eq!(&got[at], want, "{format:?} at ({x}, {y})");
            }
        }
        dev.destroy_buffer(out);
        dev.destroy_texture(tex);
    }

    const CLEAR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    const TEXELS: [[f32; 4]; 2] = [[0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25]];

    const QUAD_VS: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
        vertex vs(vi: vertex_index) -> (f32x4, f32x2) {
            let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
            (f32x4(corner * 2.0 - 1.0, 0.0, 1.0), corner)
        }
    );
    const HALF_RED_FS: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
        fragment fs() -> f32x4 { f32x4(1.0, 0.0, 0.0, 0.5) }
    );
    const SAMPLE_FS: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
        push P { slot: u32 }
        fragment fs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
            tex[p.slot].sample(uv)
        }
    );

    #[derive(Clone, Copy)]
    enum Lane {
        Unorm8,
        Unorm16,
        Half,
        Float,
    }

    fn lane(format: Format) -> Lane {
        match format {
            Format::Rgba8Unorm
            | Format::Rgba8UnormSrgb
            | Format::Bgra8Unorm
            | Format::Bgra8UnormSrgb
            | Format::R8Unorm => Lane::Unorm8,
            Format::Rgba16Unorm | Format::R16Unorm => Lane::Unorm16,
            Format::Rgba16Float | Format::R16Float => Lane::Half,
            _ => Lane::Float,
        }
    }

    fn lane_bytes(lane: Lane) -> usize {
        match lane {
            Lane::Unorm8 => 1,
            Lane::Unorm16 | Lane::Half => 2,
            Lane::Float => 4,
        }
    }

    /// Stored channels: 1 for red-only formats, else 4.
    fn channels(format: Format) -> usize {
        usize::try_from(format.bytes_per_texel()).unwrap() / lane_bytes(lane(format))
    }

    /// Max absolute readback error for `format`.
    fn tolerance(format: Format) -> f32 {
        match lane(format) {
            Lane::Unorm8 => 2.0 / 255.0,
            Lane::Unorm16 => 1e-4,
            Lane::Half => 1e-3,
            Lane::Float => 1e-6,
        }
    }

    fn is_srgb(format: Format) -> bool {
        matches!(format, Format::Rgba8UnormSrgb | Format::Bgra8UnormSrgb)
    }

    fn is_bgra(format: Format) -> bool {
        matches!(format, Format::Bgra8Unorm | Format::Bgra8UnormSrgb)
    }

    fn half_to_f32(bits: u16) -> f32 {
        let exp = i32::from((bits >> 10) & 0x1f);
        let frac = f32::from(bits & 0x3ff);
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        if exp == 0 {
            sign * frac * 2f32.powi(-24)
        } else {
            sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
        }
    }

    /// f16 bits by truncating the mantissa: exact for normal test values.
    fn f32_to_half(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) + 15 - 127;
        let frac = (bits >> 13) & 0x3ff;
        u16::try_from(sign | (exp << 10) | frac).unwrap()
    }

    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_srgb(c: f32) -> f32 {
        if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    fn decode_lane(lane: Lane, bytes: &[u8]) -> f32 {
        match lane {
            Lane::Unorm8 => f32::from(bytes[0]) / 255.0,
            Lane::Unorm16 => f32::from(u16::from_le_bytes(bytes.try_into().unwrap())) / 65535.0,
            Lane::Half => half_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())),
            Lane::Float => f32::from_le_bytes(bytes.try_into().unwrap()),
        }
    }

    fn nearest_level(c: f32, max: u16) -> u16 {
        let dist = |level: u16| (f32::from(level) / f32::from(max) - c).abs();
        (0..=max)
            .min_by(|a, b| dist(*a).total_cmp(&dist(*b)))
            .unwrap()
    }

    fn encode_lane(lane: Lane, c: f32) -> Vec<u8> {
        match lane {
            Lane::Unorm8 => vec![u8::try_from(nearest_level(c, 255)).unwrap()],
            Lane::Unorm16 => nearest_level(c, 65535).to_le_bytes().to_vec(),
            Lane::Half => f32_to_half(c).to_le_bytes().to_vec(),
            Lane::Float => c.to_le_bytes().to_vec(),
        }
    }

    /// Linear RGBA of one texel. One-channel formats give (r, 0, 0, 1).
    fn decode(format: Format, texel: &[u8]) -> [f32; 4] {
        let lane = lane(format);
        let mut out = [0.0, 0.0, 0.0, 1.0];
        for (i, bytes) in texel.chunks_exact(lane_bytes(lane)).enumerate() {
            out[i] = decode_lane(lane, bytes);
        }
        if is_bgra(format) {
            out.swap(0, 2);
        }
        if is_srgb(format) {
            for c in &mut out[..3] {
                *c = srgb_to_linear(*c);
            }
        }
        out
    }

    fn encode(format: Format, rgba: [f32; 4]) -> Vec<u8> {
        let order = if is_bgra(format) {
            [2, 1, 0, 3]
        } else {
            [0, 1, 2, 3]
        };
        (0..channels(format))
            .flat_map(|i| {
                let mut c = rgba[order[i]];
                if is_srgb(format) && i < 3 {
                    c = linear_to_srgb(c);
                }
                encode_lane(lane(format), c)
            })
            .collect()
    }

    fn assert_texels(format: Format, got: &[[f32; 4]], want: &[[f32; 4]]) {
        let tol = tolerance(format);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            for c in 0..channels(format) {
                assert!(
                    (g[c] - w[c]).abs() <= tol,
                    "{format:?} texel {i} channel {c}: got {g:?}, want {w:?}"
                );
            }
        }
    }

    /// `w` x `h` render target, primed so readback starts from the shader-read layout.
    fn render_target(
        device: &VulkanDevice,
        format: Format,
        w: u32,
        h: u32,
    ) -> (TextureHandle, TargetHandle) {
        let tex = device
            .create_texture(TextureDesc {
                width: w,
                height: h,
                depth: 1,
                format,
                usage: TextureUsage::RENDER_TARGET
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let zeros = vec![0u8; (w * h * format.bytes_per_texel()) as usize];
        device.upload_texture_data(tex, &zeros).unwrap();
        let target = device.register_color_target(tex).unwrap();
        (tex, target)
    }

    fn read_texels(
        device: &VulkanDevice,
        tex: TextureHandle,
        format: Format,
        w: u32,
        h: u32,
    ) -> Vec<[f32; 4]> {
        let row = w * format.bytes_per_texel();
        let len = u64::from(row * h);
        let buf = device.create_buffer(rw_desc(len)).unwrap();
        device
            .copy_texture_region_to_buffer(tex, TextureRegion::rect(0, 0, w, h), buf, 0, row)
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();
        let raw = device.read_buffer(buf, 0, len).unwrap();
        device.destroy_buffer(buf);
        let texel = usize::try_from(format.bytes_per_texel()).unwrap();
        raw.chunks_exact(texel).map(|t| decode(format, t)).collect()
    }

    fn quad_pipeline(
        device: &VulkanDevice,
        fs: &zengpu_spirv::ZslShader,
        format: Format,
        blend: Option<BlendState>,
    ) -> PipelineHandle {
        let vs_shader = device.create_shader(QUAD_VS.spirv_desc().unwrap()).unwrap();
        let fs_shader = device.create_shader(fs.spirv_desc().unwrap()).unwrap();
        let pipeline = device
            .create_graphics_pipeline(GraphicsPipelineDesc {
                vertex_shader: vs_shader,
                fragment_shader: fs_shader,
                vertex_layouts: &[],
                topology: PrimitiveTopology::TriangleList,
                color_targets: &[ColorTargetState { format, blend }],
                depth_format: None,
                depth: DepthState::default(),
                stencil: None,
                raster: RasterState::default(),
                samples: 1,
            })
            .unwrap();
        device.destroy_shader(vs_shader);
        device.destroy_shader(fs_shader);
        pipeline
    }

    fn draw_quad(
        device: &VulkanDevice,
        target: TargetHandle,
        load: LoadOp,
        pipeline: PipelineHandle,
        scalars: &[Scalar],
        (w, h): (u16, u16),
    ) {
        let mut cmd = device.create_command_list().unwrap();
        let color = [ColorAttachment {
            target,
            load,
            store: true,
            sample_after: true,
        }];
        cmd.begin_render_pass(&RenderPassDesc {
            color: &color,
            depth: None,
        });
        cmd.set_pipeline(pipeline);
        cmd.set_viewport_scissor(ViewportScissor {
            viewport: Viewport {
                x: 0.0,
                y: 0.0,
                width: f32::from(w),
                height: f32::from(h),
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: None,
        });
        cmd.bind(Bindings {
            scalars,
            ..Default::default()
        });
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        device.submit_and_wait(cmd).unwrap();
    }

    fn clear_round_trip(device: &VulkanDevice, format: Format) {
        let (tex, target) = render_target(device, format, 2, 2);
        let mut cmd = device.create_command_list().unwrap();
        let color = [ColorAttachment {
            target,
            load: LoadOp::Clear(CLEAR),
            store: true,
            sample_after: true,
        }];
        cmd.begin_render_pass(&RenderPassDesc {
            color: &color,
            depth: None,
        });
        cmd.end_render_pass();
        device.submit_and_wait(cmd).unwrap();
        let got = read_texels(device, tex, format, 2, 2);
        assert_texels(format, &got, &[CLEAR; 4]);
        device.destroy_texture(tex);
    }

    fn blend_round_trip(device: &VulkanDevice, format: Format) {
        let (tex, target) = render_target(device, format, 2, 2);
        let pipeline = quad_pipeline(device, &HALF_RED_FS, format, Some(BlendState::ALPHA_BLEND));
        draw_quad(
            device,
            target,
            LoadOp::clear_rgb(0.0, 0.0, 1.0),
            pipeline,
            &[],
            (2, 2),
        );
        let got = read_texels(device, tex, format, 2, 2);
        assert_texels(format, &got, &[[0.5, 0.0, 0.5, 1.0]; 4]);
        device.destroy_pipeline(pipeline);
        device.destroy_texture(tex);
    }

    fn sample_round_trip(device: &VulkanDevice, format: Format) {
        let src = device
            .create_texture(TextureDesc {
                width: 2,
                height: 1,
                depth: 1,
                format,
                usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let bytes: Vec<u8> = TEXELS.iter().flat_map(|t| encode(format, *t)).collect();
        device.upload_texture_data(src, &bytes).unwrap();
        let sampler = device
            .create_sampler(SamplerDesc {
                min_filter: FilterMode::Nearest,
                mag_filter: FilterMode::Nearest,
                mip_filter: FilterMode::Nearest,
                ..SamplerDesc::default()
            })
            .unwrap();
        let slot = device.bind_texture(src, sampler).unwrap();

        let out_format = Format::Rgba32Float;
        let (out, target) = render_target(device, out_format, 2, 1);
        let pipeline = quad_pipeline(device, &SAMPLE_FS, out_format, None);
        draw_quad(
            device,
            target,
            LoadOp::Clear([0.0; 4]),
            pipeline,
            &[Scalar::U32(slot)],
            (2, 1),
        );
        let got = read_texels(device, out, out_format, 2, 1);
        assert_texels(format, &got, &TEXELS);
        device.destroy_pipeline(pipeline);
        device.destroy_texture(out);
        device.destroy_sampler(sampler);
        device.destroy_texture(src);
    }

    fn vulkan(dev: &TestDevice) -> &VulkanDevice {
        dev.dev
            .as_any()
            .downcast_ref::<VulkanDevice>()
            .expect("vulkan device")
    }

    #[test]
    fn colour_formats_clear_to_their_value() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let device = vulkan(&dev);
        for format in Format::ALL {
            if !format.is_depth_stencil() && device.format_caps(format).render_target {
                clear_round_trip(device, format);
            }
        }
    }

    /// A pass into a never-uploaded target still reads back: it marks the texture as holding
    /// texels, so the copy does not start from `UNDEFINED`.
    #[test]
    fn rendered_and_uploaded_texels_survive_readback_and_load() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let device = vulkan(&dev);
        let format = Format::Rgba8Unorm;
        let pass = |target, load| {
            let mut cmd = device.create_command_list().unwrap();
            cmd.begin_render_pass(&RenderPassDesc {
                color: &[ColorAttachment {
                    target,
                    load,
                    store: true,
                    sample_after: true,
                }],
                depth: None,
            });
            cmd.end_render_pass();
            device.submit_and_wait(cmd).unwrap();
        };

        let offscreen = OffscreenTarget::new(device, format, 2, 2).unwrap();
        pass(offscreen.target_handle(), LoadOp::Clear(CLEAR));
        let got = read_texels(device, offscreen.texture_handle(), format, 2, 2);
        assert_texels(format, &got, &[CLEAR; 4]);
        pass(offscreen.target_handle(), LoadOp::Load);
        let got = read_texels(device, offscreen.texture_handle(), format, 2, 2);
        assert_texels(format, &got, &[CLEAR; 4]);
        drop(offscreen);

        let tex = device
            .create_texture(TextureDesc {
                width: 2,
                height: 2,
                depth: 1,
                format,
                usage: TextureUsage::RENDER_TARGET
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let target = device.register_color_target(tex).unwrap();
        device.upload_texture_data(tex, &[255u8; 16]).unwrap();
        pass(target, LoadOp::Load);
        let got = read_texels(device, tex, format, 2, 2);
        assert_texels(format, &got, &[[1.0; 4]; 4]);
        device.unregister_render_target(target);
        device.destroy_texture(tex);
    }

    #[test]
    fn colour_formats_alpha_blend() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let device = vulkan(&dev);
        for format in Format::ALL {
            let caps = device.format_caps(format);
            if !format.is_depth_stencil() && caps.render_target && caps.blendable {
                blend_round_trip(device, format);
            }
        }
    }

    #[test]
    fn colour_formats_sample_nearest() {
        let Some(dev) = try_graphics_device() else {
            return;
        };
        let device = vulkan(&dev);
        if !device.format_caps(Format::Rgba32Float).render_target {
            return;
        }
        for format in Format::ALL {
            if !format.is_depth_stencil() && device.format_caps(format).sampled {
                sample_round_trip(device, format);
            }
        }
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
        unsafe { from_raw_parts(words.as_ptr() as *const u8, size_of_val(words)) }
    }

    #[test]
    fn bindless_capacities_respect_each_device_limit() {
        let limits = DeviceLimits {
            max_storage_buffers: 16,
            max_sampled_textures: 8,
            max_update_after_bind_descriptors: 12,
            ..Default::default()
        };
        assert_eq!(bindless_capacities(limits).unwrap(), (3, 3, 0));
        let roomy = DeviceLimits {
            max_storage_textures: 64,
            max_update_after_bind_descriptors: 1000,
            ..limits
        };
        assert_eq!(bindless_capacities(roomy).unwrap(), (5, 8, 64));
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
    fn timestamp_query_pool_writes_and_resolves() {
        let _guard = crate::test_gpu_lock();
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = inst.request_vulkan_adapter() else {
            return;
        };
        let Ok(device) = adapter.open_headless(DeviceRequest::default()) else {
            return;
        };

        let pool = device.create_query_pool(1).unwrap();
        let mut commands = device.create_command_list().unwrap();
        commands.write_timestamp(pool, 0);
        device.submit_and_wait(commands).unwrap();

        let timestamps = device.resolve_query_pool(pool).unwrap();
        assert_eq!(timestamps.len(), 1);
        assert_ne!(timestamps[0], 0);
    }

    #[test]
    fn pipeline_cache_is_persisted_to_disk() {
        use std::time::{SystemTime, UNIX_EPOCH};
        use zengpu_spirv::{ZslShader, zsl};

        const CACHE_TEST_ZSL: ZslShader = zsl!(
            @workgroup_size(1)
            kernel cache_test() {}
        );

        let _guard = crate::test_gpu_lock();
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = inst.request_vulkan_adapter() else {
            return;
        };
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "zengpu-pipeline-cache-{}-{unique}.bin",
            process::id()
        ));
        let Ok(device) = adapter.open_headless(DeviceRequest {
            pipeline_cache_path: Some(path.clone()),
            ..Default::default()
        }) else {
            return;
        };
        let shader = device
            .create_shader(CACHE_TEST_ZSL.spirv_desc().unwrap())
            .unwrap();
        device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [1, 1, 1],
            })
            .unwrap();
        drop(device);

        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
        fs::remove_file(path).unwrap();
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
        dev.write_buffer(gpu, 0, &[1]).unwrap();
        dev.copy_buffer(upload, 0, gpu, 0, SIZE as u64).unwrap();
        dev.copy_buffer(gpu, 0, readback, 0, SIZE as u64).unwrap();
        assert_eq!(dev.read_buffer(readback, 0, SIZE as u64).unwrap(), data);

        dev.destroy_buffer(upload);
        dev.destroy_buffer(gpu);
        dev.destroy_buffer(readback);
    }

    #[test]
    fn buffer_misuse_is_reported() {
        let Some(dev) = try_device() else { return };
        let no_readback = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let err = dev.read_buffer(no_readback, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "READBACK",
                ..
            })
        ));
        dev.destroy_buffer(no_readback);

        let h = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));
        dev.destroy_buffer(h);
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::StaleHandle { .. })
        ));
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

        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Some(any_adapter) = instance.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let err = match any_adapter.open(DeviceRequest {
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
                optional: Features::TIMESTAMPS,
                ..Default::default()
            })
            .unwrap();
        assert!(dev.capabilities().compute);
        assert!(dev.capabilities().graphics);
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
    fn textures_destroyed_while_in_flight_are_recovered() {
        let Some(dev) = try_device() else {
            return;
        };
        if !dev.capabilities().transfers.texture_region_write {
            return;
        }
        let desc = TextureDesc {
            width: 256,
            height: 256,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        };
        let pixels = vec![0x7Fu8; 256 * 256 * 4];
        let write_full = |tex| zengpu_hal::TextureWrite {
            texture: tex,
            region: zengpu_hal::TextureRegion::rect(0, 0, 256, 256),
            data: &pixels,
            bytes_per_row: 256 * 4,
        };
        let mut window: VecDeque<Submission> = VecDeque::new();
        for _ in 0..200 {
            let tex = dev.create_texture(desc).unwrap();
            window.push_back(dev.write_textures(&[write_full(tex)]).unwrap());
            dev.destroy_texture(tex);
            if window.len() == 8 {
                window
                    .pop_front()
                    .unwrap()
                    .wait(Duration::from_secs(5))
                    .unwrap();
            }
        }
        for work in window {
            work.wait(Duration::from_secs(5)).unwrap();
        }
        let fresh = dev.create_texture(desc).unwrap();
        dev.write_textures(&[write_full(fresh)])
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();
        dev.destroy_texture(fresh);
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
        let mut region = TextureRegion::rect(0, 0, 4, 4);
        region.mip_level = 1;
        let out = dev.create_buffer(rw_desc(4 * 4 * 4)).unwrap();
        dev.copy_texture_region_to_buffer(tex, region, out, 0, 4 * 4)
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();
        assert!(
            dev.read_buffer(out, 0, 4 * 4 * 4)
                .unwrap()
                .iter()
                .all(|&b| b == 0xFF)
        );
        dev.destroy_buffer(out);
        dev.destroy_texture(tex);
    }

    /// Host cost of one trivial one-workgroup dispatch.
    ///
    /// Run with `cargo test --release -p zengpu-vulkan dispatch_overhead -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        use std::time::{Duration, Instant};

        use zengpu_spirv::{ZslShader, zsl};

        const TOUCH: ZslShader = zsl!(
            @workgroup_size(64)
            kernel touch(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = 1.0
            }
        );
        let Some(dev) = try_device() else { return };
        let shader = dev.create_shader(TOUCH.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let out = dev
            .create_buffer(BufferDesc {
                size: 4096,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bindings = Bindings {
            buffers: &[out.index()],
            ..Bindings::default()
        };
        dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
        const N: u32 = 2000;
        let per = |d: Duration| d.as_secs_f64() * 1e6 / f64::from(N);

        let t = Instant::now();
        for _ in 0..N {
            dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
        }
        println!(
            "vulkan: single dispatch + sync   {:.2} us",
            per(t.elapsed())
        );

        let ops: Vec<DispatchOp<'_>> = (0..N)
            .map(|_| DispatchOp {
                pipeline,
                bindings,
                grid: [1, 1, 1],
            })
            .collect();
        let t = Instant::now();
        let sub = dev.submit_batch(0, &ops).unwrap();
        println!(
            "vulkan: enqueue N dispatches (CPU) {:.2} us",
            per(t.elapsed())
        );
        sub.wait(Duration::from_secs(10)).unwrap();

        let t = Instant::now();
        dev.dispatch_batch(&ops).unwrap();
        println!(
            "vulkan: dispatch_batch N         {:.2} us",
            per(t.elapsed())
        );

        program_destroy(&*dev, pipeline, shader, out);
    }

    fn program_destroy(
        dev: &dyn GpuDevice,
        pipeline: PipelineHandle,
        shader: ShaderHandle,
        out: BufferHandle,
    ) {
        dev.destroy_buffer(out);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    #[test]
    fn pipeline_stats_order_tiny_and_heavy_kernels() {
        use zengpu_spirv::{ZslShader, zsl};

        const TINY: ZslShader = zsl!(
            @workgroup_size(64)
            kernel tiny(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = 1.0
            }
        );
        const HEAVY: ZslShader = zsl!(
            @workgroup_size(64)
            kernel heavy(inp: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                let i = id.x
                let x0 = inp[i]
                let x1 = x0 * 1.5
                let x2 = x1 + x0
                let x3 = x2 * x1
                let x4 = x3 + x2
                let x5 = x4 * x3
                let x6 = x5 + x4
                let x7 = x6 * x5
                let x8 = x7 + x6
                let x9 = x8 * x7
                let x10 = x9 + x8
                let x11 = x10 * x9
                let x12 = x11 + x10
                let x13 = x12 * x11
                let x14 = x13 + x12
                let x15 = x14 * x13
                out[i] = x0 * x1 + x2 * x3 + x4 * x5 + x6 * x7 + x8 * x9 + x10 * x11 + x12 * x13 + x14 * x15
            }
        );
        let Some(dev) = try_device() else { return };
        let capture = zengpu_hal::ComputePipelineHints {
            capture_stats: true,
            ..Default::default()
        };
        let stats = |shader: &ZslShader, entry: &str| {
            let module = dev.create_shader(shader.spirv_desc().unwrap()).unwrap();
            let pipeline = dev
                .create_compute_pipeline_with(
                    ComputePipelineDesc {
                        shader: module,
                        entry,
                        block: [64, 1, 1],
                    },
                    capture,
                )
                .unwrap();
            dev.pipeline_stats(pipeline)
        };
        let tiny = stats(&TINY, "main").expect("captured stats");
        let heavy = stats(&HEAVY, "main").expect("captured stats");
        assert!(tiny.vgprs > 0 && tiny.sgprs > 0, "{tiny:?}");
        assert!(heavy.vgprs > tiny.vgprs, "{heavy:?} vs {tiny:?}");
        assert_eq!(tiny.scratch_bytes, 0);
        assert!(tiny.max_threads >= 64);
        assert!(tiny.waves_per_simd.is_some());
        println!("vulkan stats: tiny {tiny:?} heavy {heavy:?}");
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
        let shader = dev.create_shader(DOUBLE_ZSL.spirv_desc().unwrap()).unwrap();
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
        let bytes = unsafe { from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
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
                            images: &[],
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
        for (index, bytes) in actual.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(f32::from_ne_bytes(*bytes), values[index] * 2.0);
        }
        dev.destroy_buffer(upload);
        dev.destroy_buffer(input);
        dev.destroy_buffer(output);
        dev.destroy_buffer(readback);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    #[test]
    fn compute_chains_texture_passes_in_one_submission() {
        use zengpu_spirv::{ZslShader, zsl};

        const INVERT: ZslShader = zsl!(
            push P { w: u32, h: u32 }
            @workgroup_size(8, 8)
            kernel invert(src: texture, dst: image, p: P, id: global_id) {
                let x = id.x
                let y = id.y
                if x < p.w && y < p.h {
                    dst.store(x, y, 1.0 - src.load(x, y).r, 1.0 - src.load(x, y).g, 1.0 - src.load(x, y).b, src.load(x, y).a)
                }
            }
        );
        const COMBINE: ZslShader = zsl!(
            @workgroup_size(8, 8)
            kernel combine(inv: image, smooth: texture, out: image, id: global_id) {
                let x = id.x
                let y = id.y
                if x < out.width() && y < out.height() {
                    let u = (x + 0.5) / out.width()
                    let v = (y + 0.5) / out.height()
                    let r = (inv.load(x, y).r + smooth.sample(u, v).r) * 0.5
                    out.store(x, y, r, inv.load(x, y).g, smooth.sample_level(u, v, 0.0).b, 1.0)
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let vk_dev = dev.as_any().downcast_ref::<VulkanDevice>().unwrap();
        if vk_dev.limits().max_storage_textures == 0 {
            return;
        }
        let pipeline = |shader: &ZslShader| {
            let shader = dev.create_shader(shader.spirv_desc().unwrap()).unwrap();
            dev.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [8, 8, 1],
            })
            .unwrap()
        };
        let (invert, combine) = (pipeline(&INVERT), pipeline(&COMBINE));
        let (w, h) = (5u32, 3u32);
        let texture = |format: Format, usage: TextureUsage| {
            dev.create_texture(TextureDesc {
                width: w,
                height: h,
                depth: 1,
                format,
                usage,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
        };
        let src = texture(
            Format::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        )
        .unwrap();
        let mid = texture(
            Format::Rgba16Float,
            TextureUsage::STORAGE | TextureUsage::SAMPLED,
        )
        .unwrap();
        let out = texture(
            Format::Rgba32Float,
            TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC,
        )
        .unwrap();
        let texels: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 % 256) as u8).collect();
        dev.upload_texture_data(src, &texels).unwrap();
        let nearest = dev.create_sampler(SamplerDesc::default()).unwrap();
        let src_slot = vk_dev.bind_texture(src, nearest).unwrap();
        let mid_slot = vk_dev.bind_texture(mid, nearest).unwrap();

        let src_textures = [src_slot];
        let mid_images = [mid.index()];
        let dimensions = [Scalar::U32(w), Scalar::U32(h)];
        let mid_textures = [mid_slot];
        let out_images = [mid.index(), out.index()];
        let ops = [
            ComputeOp::Dispatch(DispatchOp {
                pipeline: invert,
                bindings: Bindings {
                    buffers: &[],
                    textures: &src_textures,
                    images: &mid_images,
                    scalars: &dimensions,
                },
                grid: [1, 1, 1],
            }),
            ComputeOp::Dispatch(DispatchOp {
                pipeline: combine,
                bindings: Bindings {
                    buffers: &[],
                    textures: &mid_textures,
                    images: &out_images,
                    scalars: &[],
                },
                grid: [1, 1, 1],
            }),
        ];
        dev.submit_compute_ops(1, &ops)
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();

        let got = read_texels(vk_dev, out, Format::Rgba32Float, w, h);
        for (i, (texel, src)) in got.iter().zip(texels.as_chunks::<4>().0.iter()).enumerate() {
            let inv = |c: u8| 1.0 - f32::from(c) / 255.0;
            let want = [inv(src[0]), inv(src[1]), inv(src[2]), 1.0];
            for (g, w) in texel.iter().zip(want) {
                assert!((g - w).abs() < 2e-3, "texel {i}: {texel:?} != {want:?}");
            }
        }

        let not_storage = Bindings {
            buffers: &[],
            textures: &[src_slot],
            images: &[src.index()],
            scalars: &[Scalar::U32(w), Scalar::U32(h)],
        };
        assert!(dev.dispatch(invert, not_storage, [1, 1, 1]).is_err());
        assert!(
            texture(
                Format::Rgba8Unorm,
                TextureUsage::STORAGE | TextureUsage::RENDER_TARGET
            )
            .is_err()
        );
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

        let add_shader = dev.create_shader(ADD_ZSL.spirv_desc().unwrap()).unwrap();
        let add_pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: add_shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let relu_shader = dev.create_shader(RELU_ZSL.spirv_desc().unwrap()).unwrap();
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
            u32s_to_bytes(unsafe { from_raw_parts(a_vals.as_ptr() as *const u32, a_vals.len()) }),
        )
        .unwrap();
        dev.write_buffer(
            b,
            0,
            u32s_to_bytes(unsafe { from_raw_parts(b_vals.as_ptr() as *const u32, b_vals.len()) }),
        )
        .unwrap();

        let submission = dev
            .submit_batch(
                0x4d_50_50_49,
                &[
                    DispatchOp {
                        pipeline: add_pipeline,
                        bindings: Bindings {
                            buffers: &[a.index(), b.index(), sum.index()],
                            textures: &[],
                            images: &[],
                            scalars: &[Scalar::U32(n)],
                        },
                        grid: [n.div_ceil(64), 1, 1],
                    },
                    DispatchOp {
                        pipeline: relu_pipeline,
                        bindings: Bindings {
                            buffers: &[sum.index(), out.index()],
                            textures: &[],
                            images: &[],
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
                    images: &[],
                    scalars: &[Scalar::U32(n)],
                },
                [n.div_ceil(64), 1, 1],
            )
            .unwrap();
        assert!(dev.write_buffer(a, 0, &[0; 4]).is_err());
        let unrelated = dev.create_buffer(storage_rw(16)).unwrap();
        dev.write_buffer(unrelated, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(unrelated, 0, 4).unwrap(), vec![1, 2, 3, 4]);
        dev.destroy_buffer(unrelated);
        let retired_sum_index = sum.index();
        dev.destroy_buffer(sum);
        dev.destroy_pipeline(add_pipeline);
        assert!(dev.read_buffer(sum, 0, 4).is_err());
        let allocated_while_pending = dev.create_buffer(storage_rw(n as u64 * 4)).unwrap();
        assert_ne!(allocated_while_pending.index(), retired_sum_index);
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
        dev.write_buffer(a, 0, &[0; 4]).unwrap();

        let bytes = dev.read_buffer(out, 0, n as u64 * 4).unwrap();
        let result: &[f32] = unsafe { from_raw_parts(bytes.as_ptr() as *const f32, n as usize) };
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
        let shader = dev.create_shader(FILL_ZSL.spirv_desc().unwrap()).unwrap();
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

        let output = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let texture = dev.create_texture(tex_desc(TexDim::D2)).unwrap();
        let sampler = dev.create_sampler(SamplerDesc::default()).unwrap();
        let texture_index = vulkan(&dev).bind_texture(texture, sampler).unwrap();
        dev.destroy_texture(texture);
        let err = dev
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[output.index()],
                    textures: &[texture_index],
                    images: &[],
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
        let shader = dev.create_shader(FILL_ZSL.spirv_desc().unwrap()).unwrap();
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
                spawn(move || {
                    let buffers = [output.index()];
                    let scalars = [Scalar::F32(worker as f32 + 0.5)];
                    start.wait();
                    dev.dispatch(
                        pipeline,
                        Bindings {
                            buffers: &buffers,
                            textures: &[],
                            images: &[],
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
            for value in bytes.as_chunks::<4>().0.iter() {
                assert_eq!(f32::from_ne_bytes(*value), expected);
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
            u32s_to_bytes(unsafe { from_raw_parts(input.as_ptr() as *const u32, input.len()) }),
        )
        .unwrap();

        let shader = dev.create_shader(TRIG_ZSL.spirv_desc().unwrap()).unwrap();
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
                images: &[],
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
    fn zsl_dot4_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const DOT4_ZSL: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel dot4(x: device buffer<i32>, y: device buffer<u32>, k: device buffer<i32>, z: device buffer<u32>, out_i: device mut buffer<i32>, out_iu: device mut buffer<i32>, out_u: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    out_i[i] = dot4_i8(x[i], x[i ^ 1], k[i])
                    out_iu[i] = dot4_iu8(x[i], y[i], k[i])
                    out_u[i] = dot4_u8(y[i], y[i ^ 1], z[i])
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 256u32;
        let size = u64::from(n) * 4;
        let input = BufferDesc {
            size,
            usage: BufferUsage::STORAGE,
            memory: MemoryUsage::Upload,
        };
        let output = BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        const EDGE: [u8; 4] = [0x80, 0x7f, 0xff, 0x00];
        let word = |i: u32| {
            u32::from_le_bytes([
                EDGE[(i % 4) as usize],
                EDGE[((i / 4 + 1) % 4) as usize],
                EDGE[((i / 2 + 2) % 4) as usize],
                (i.wrapping_mul(0x9e37_79b9) >> 24) as u8,
            ])
        };
        let xs: Vec<u32> = (0..n).map(word).collect();
        let ys: Vec<u32> = (0..n).map(|i| word(i + 7)).collect();
        let ks: Vec<u32> = (0..n).map(|i| i32::MAX as u32 - i % 100).collect();
        let zs: Vec<u32> = (0..n).map(|i| u32::MAX - i % 100).collect();
        let inputs: Vec<_> = [&xs, &ys, &ks, &zs]
            .iter()
            .map(|data| {
                let buf = dev.create_buffer(input).unwrap();
                dev.write_buffer(buf, 0, u32s_to_bytes(data)).unwrap();
                buf
            })
            .collect();
        let outs: Vec<_> = (0..3).map(|_| dev.create_buffer(output).unwrap()).collect();

        let shader = dev.create_shader(DOT4_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = inputs.iter().chain(&outs).map(|b| b.index()).collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[Scalar::U32(n)],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();

        let words = |b| -> Vec<u32> {
            dev.read_buffer(b, 0, size)
                .unwrap()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got_i, got_iu, got_u) = (words(outs[0]), words(outs[1]), words(outs[2]));
        let s8 = |w: u32, l: u32| ((w >> (8 * l)) as u8) as i8 as i32;
        let u8l = |w: u32, l: u32| ((w >> (8 * l)) & 0xff) as i32;
        for i in 0..n as usize {
            let (x, x2, y, y2) = (xs[i], xs[i ^ 1], ys[i], ys[i ^ 1]);
            let (mut want_i, mut want_iu, mut want_u) = (ks[i] as i32, ks[i] as i32, zs[i]);
            for l in 0..4 {
                want_i = want_i.wrapping_add(s8(x, l).wrapping_mul(s8(x2, l)));
                want_iu = want_iu.wrapping_add(s8(x, l).wrapping_mul(u8l(y, l)));
                want_u = want_u.wrapping_add((u8l(y, l) * u8l(y2, l)) as u32);
            }
            assert_eq!(got_i[i] as i32, want_i, "dot4_i8 at {i}");
            assert_eq!(got_iu[i] as i32, want_iu, "dot4_iu8 at {i}");
            assert_eq!(got_u[i], want_u, "dot4_u8 at {i}");
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for b in inputs.into_iter().chain(outs) {
            dev.destroy_buffer(b);
        }
    }

    /// f16 results: products and sums combine unrounded, then round once.
    mod packed_ref {
        #[derive(Clone, Copy)]
        enum Fin {
            Zero(bool),
            Num(bool, i128, i32),
            Inf(bool),
            Nan,
        }

        fn classify(h: u16) -> Fin {
            let neg = h >> 15 == 1;
            let exp = (h >> 10) & 0x1f;
            let frac = i128::from(h & 0x3ff);
            match exp {
                0x1f if frac == 0 => Fin::Inf(neg),
                0x1f => Fin::Nan,
                0 if frac == 0 => Fin::Zero(neg),
                0 => Fin::Num(neg, frac, -24),
                _ => Fin::Num(neg, frac | 0x400, i32::from(exp) - 25),
            }
        }

        fn sign(f: Fin) -> bool {
            match f {
                Fin::Zero(s) | Fin::Inf(s) | Fin::Num(s, ..) => s,
                Fin::Nan => false,
            }
        }

        fn mul(a: Fin, b: Fin) -> Fin {
            let s = sign(a) ^ sign(b);
            match (a, b) {
                (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
                (Fin::Inf(_), Fin::Zero(_)) | (Fin::Zero(_), Fin::Inf(_)) => Fin::Nan,
                (Fin::Inf(_), _) | (_, Fin::Inf(_)) => Fin::Inf(s),
                (Fin::Zero(_), _) | (_, Fin::Zero(_)) => Fin::Zero(s),
                (Fin::Num(_, m1, e1), Fin::Num(_, m2, e2)) => Fin::Num(s, m1 * m2, e1 + e2),
            }
        }

        fn add(a: Fin, b: Fin) -> Fin {
            match (a, b) {
                (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
                (Fin::Inf(x), Fin::Inf(y)) => {
                    if x == y {
                        Fin::Inf(x)
                    } else {
                        Fin::Nan
                    }
                }
                (Fin::Inf(x), _) | (_, Fin::Inf(x)) => Fin::Inf(x),
                (Fin::Zero(x), Fin::Zero(y)) => Fin::Zero(x && y),
                (Fin::Zero(_), o) | (o, Fin::Zero(_)) => o,
                (Fin::Num(sa, ma, ea), Fin::Num(sb, mb, eb)) => {
                    let e = ea.min(eb);
                    let va = (if sa { -ma } else { ma }) << (ea - e);
                    let vb = (if sb { -mb } else { mb }) << (eb - e);
                    let s = va + vb;
                    if s == 0 {
                        Fin::Zero(false)
                    } else {
                        Fin::Num(s < 0, s.abs(), e)
                    }
                }
            }
        }

        /// `m * 2^e` (m > 0) to f16, nearest even.
        fn round(neg: bool, m: u128, e: i32) -> u16 {
            let bitlen = (128 - m.leading_zeros()) as i32;
            let drop = (bitlen - 11).max(-24 - e);
            let (mut q, mut qe) = if drop <= 0 {
                (m << (-drop) as u32, e + drop)
            } else if drop >= 127 {
                (0, -24)
            } else {
                let d = drop as u32;
                let mut q = m >> d;
                let rem = m & ((1u128 << d) - 1);
                let half = 1u128 << (d - 1);
                if rem > half || (rem == half && q & 1 == 1) {
                    q += 1;
                }
                (q, e + drop)
            };
            let sign = u16::from(neg) << 15;
            if q == 0 {
                return sign;
            }
            while q >= 2048 {
                q >>= 1;
                qe += 1;
            }
            if q >= 1024 {
                let t = qe + 10;
                if t > 15 {
                    return sign | 0x7c00;
                }
                sign | (((t + 15) as u16) << 10) | (q as u16 - 1024)
            } else {
                sign | q as u16
            }
        }

        fn bits(f: Fin) -> u16 {
            match f {
                Fin::Nan => 0x7e00,
                Fin::Inf(s) => (u16::from(s) << 15) | 0x7c00,
                Fin::Zero(s) => u16::from(s) << 15,
                Fin::Num(s, m, e) => round(s, m as u128, e),
            }
        }

        pub fn add16(a: u16, b: u16) -> u16 {
            bits(add(classify(a), classify(b)))
        }

        pub fn mul16(a: u16, b: u16) -> u16 {
            bits(mul(classify(a), classify(b)))
        }

        pub fn fma16(a: u16, b: u16, c: u16) -> u16 {
            bits(add(mul(classify(a), classify(b)), classify(c)))
        }

        fn to_f32(h: u16) -> f32 {
            match classify(h) {
                Fin::Nan => f32::NAN,
                Fin::Inf(s) => {
                    if s {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                }
                Fin::Zero(s) => {
                    if s {
                        -0.0
                    } else {
                        0.0
                    }
                }
                Fin::Num(s, m, e) => {
                    let v = m as f32 * 2f32.powi(e);
                    if s { -v } else { v }
                }
            }
        }

        /// Products exact in f32, then two f32 adds.
        pub fn dot2(a: u32, b: u32, acc: f32) -> f32 {
            let p0 = to_f32(a as u16) * to_f32(b as u16);
            let p1 = to_f32((a >> 16) as u16) * to_f32((b >> 16) as u16);
            (acc + p0) + p1
        }

        pub fn lanes(a: u32, b: u32, f: impl Fn(u16, u16) -> u16) -> u32 {
            u32::from(f(a as u16, b as u16))
                | (u32::from(f((a >> 16) as u16, (b >> 16) as u16)) << 16)
        }

        pub fn lanes3(a: u32, b: u32, c: u32, f: impl Fn(u16, u16, u16) -> u16) -> u32 {
            u32::from(f(a as u16, b as u16, c as u16))
                | (u32::from(f((a >> 16) as u16, (b >> 16) as u16, (c >> 16) as u16)) << 16)
        }

        /// Bit-exact per f16 lane, except that any NaN matches any NaN.
        pub fn same_f16x2(got: u32, want: u32) -> bool {
            let lane_ok = |g: u16, w: u16| {
                let nan = |h: u16| (h & 0x7fff) > 0x7c00;
                (nan(g) && nan(w)) || g == w
            };
            lane_ok(got as u16, want as u16) && lane_ok((got >> 16) as u16, (want >> 16) as u16)
        }

        pub fn pk_add_u16(a: u32, b: u32) -> u32 {
            lanes(a, b, |x, y| x.wrapping_add(y))
        }

        pub fn pk_mul_u16(a: u32, b: u32) -> u32 {
            lanes(a, b, |x, y| x.wrapping_mul(y))
        }

        pub fn pk_mad_u16(a: u32, b: u32, c: u32) -> u32 {
            lanes3(a, b, c, |x, y, z| x.wrapping_mul(y).wrapping_add(z))
        }

        pub fn funnel_shr(hi: u32, lo: u32, s: u32) -> u32 {
            ((u64::from(hi) << 32 | u64::from(lo)) >> (s & 31)) as u32
        }

        /// CUDA `__byte_perm` with the 3-bit selector and no sign mode.
        pub fn byte_perm(a: u32, b: u32, sel: u32) -> u32 {
            let src = u64::from(a) | (u64::from(b) << 32);
            (0..4).fold(0u32, |out, i| {
                let k = (sel >> (4 * i)) & 7;
                let byte = ((src >> (8 * k)) & 0xff) as u32;
                out | (byte << (8 * i))
            })
        }
    }

    #[test]
    fn zsl_packed_ops_run_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const PACKED_ZSL: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel packed(x: device buffer<u32>, y: device buffer<u32>, z: device buffer<u32>, f: device buffer<f32>, o1: device mut buffer<u32>, o2: device mut buffer<u32>, o3: device mut buffer<u32>, o4: device mut buffer<u32>, o5: device mut buffer<u32>, o6: device mut buffer<u32>, o7: device mut buffer<u32>, o8: device mut buffer<u32>, of: device mut buffer<f32>, o9: device mut buffer<u32>, o10: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    o1[i] = pk_add_u16(x[i], y[i])
                    o2[i] = pk_mul_u16(x[i], y[i])
                    o3[i] = pk_mad_u16(x[i], y[i], z[i])
                    o4[i] = pk_add_f16(x[i], y[i])
                    o5[i] = pk_mul_f16(x[i], y[i])
                    o6[i] = pk_fma_f16(x[i], y[i], z[i])
                    o7[i] = funnel_shr(x[i], y[i], z[i])
                    o8[i] = byte_perm(x[i], y[i], z[i])
                    of[i] = dot2_f16(x[i], y[i], f[i])
                    o9[i] = pk_mul_f16(x[i], 0x3c003c00u32)
                    o10[i] = pk_add_u16(x[i], 0x3c003c00u32) + (-2147483648) + 0b1010 + 0o17 + 0xffffffffu32
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 1024u32;
        let size = u64::from(n) * 4;
        let input = BufferDesc {
            size,
            usage: BufferUsage::STORAGE,
            memory: MemoryUsage::Upload,
        };
        let output = BufferDesc {
            size,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        };
        const F16_EDGE: [u16; 16] = [
            0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff,
            0x3c00, 0xbc00, 0x3555, 0x4000, 0x0200,
        ];
        let half = |s: u32| -> u16 {
            if s.is_multiple_of(3) {
                (s.wrapping_mul(0x9e37_79b9) >> 16) as u16
            } else {
                F16_EDGE[(s % 16) as usize]
            }
        };
        let lanes = |s: u32| u32::from(half(s)) | (u32::from(half(s.wrapping_add(7919))) << 16);
        let mut xs: Vec<u32> = (0..n).map(lanes).collect();
        let mut ys: Vec<u32> = (0..n).map(|i| lanes(i + 4096)).collect();
        let mut zs: Vec<u32> = (0..n)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 31,
                2 => 32,
                3 => 0x7654_3210,
                _ => lanes(i + 8192),
            })
            .collect();
        xs[0] = (xs[0] & 0xffff_0000) | 0x3c01;
        ys[0] = (ys[0] & 0xffff_0000) | 0x3bff;
        zs[0] = (zs[0] & 0xffff_0000) | 0x0009;
        let fs: Vec<f32> = (0..n)
            .map(|i| match i % 4 {
                0 => 0.0,
                1 => -0.0,
                2 => i as f32 * 0.37 - 100.0,
                _ => f32::from_bits(lanes(i + 123)),
            })
            .collect();
        let f_bits: Vec<u32> = fs.iter().map(|v| v.to_bits()).collect();
        let inputs: Vec<_> = [&xs, &ys, &zs, &f_bits]
            .iter()
            .map(|data| {
                let buf = dev.create_buffer(input).unwrap();
                dev.write_buffer(buf, 0, u32s_to_bytes(data)).unwrap();
                buf
            })
            .collect();
        let outs: Vec<_> = (0..11)
            .map(|_| dev.create_buffer(output).unwrap())
            .collect();

        let shader = dev.create_shader(PACKED_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = inputs.iter().chain(&outs).map(|b| b.index()).collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[Scalar::U32(n)],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();

        let words = |b| -> Vec<u32> {
            dev.read_buffer(b, 0, size)
                .unwrap()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let got: Vec<Vec<u32>> = outs.iter().map(|b| words(*b)).collect();
        use packed_ref as r;
        for i in 0..n as usize {
            let (x, y, z) = (xs[i], ys[i], zs[i]);
            assert!(
                r::same_f16x2(got[9][i], r::lanes(x, 0x3c00_3c00, r::mul16)),
                "pk_mul_f16 with a hex literal at {i}: x={x:08x}"
            );
            assert_eq!(
                got[10][i],
                r::pk_add_u16(x, 0x3c00_3c00)
                    .wrapping_add(0x8000_0000)
                    .wrapping_add(10)
                    .wrapping_add(15)
                    .wrapping_add(0xffff_ffff),
                "literal expression at {i}"
            );
            let want = [
                r::pk_add_u16(x, y),
                r::pk_mul_u16(x, y),
                r::pk_mad_u16(x, y, z),
                r::lanes(x, y, r::add16),
                r::lanes(x, y, r::mul16),
                r::lanes3(x, y, z, r::fma16),
                r::funnel_shr(x, y, z),
                r::byte_perm(x, y, z),
            ];
            for (k, w) in want.iter().enumerate() {
                let f16_out = (3..=5).contains(&k);
                let ok = if f16_out {
                    r::same_f16x2(got[k][i], *w)
                } else {
                    got[k][i] == *w
                };
                assert!(
                    ok,
                    "output {k} at {i}: got {:08x} want {w:08x} x={x:08x} y={y:08x} z={z:08x}",
                    got[k][i]
                );
            }
            let want_f = r::dot2(x, y, fs[i]);
            let got_f = f32::from_bits(got[8][i]);
            assert!(
                (want_f.is_nan() && got_f.is_nan()) || want_f.to_bits() == got_f.to_bits(),
                "dot2_f16 at {i}: got {got_f:?} want {want_f:?}"
            );
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for b in inputs.into_iter().chain(outs) {
            dev.destroy_buffer(b);
        }
    }

    #[test]
    fn zsl_bitwise_ops_unpack_nibbles_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const NIBBLES_ZSL: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel nibbles(packed: device buffer<u32>, out: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    let w = packed[i >> 3]
                    out[i] = ((w >> ((i & 7) << 2)) & 15) | ((i ^ 1) << 8)
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 1000u32;
        let words: Vec<u32> = (0..n.div_ceil(8))
            .map(|i| i.wrapping_mul(0x9e37_79b9))
            .collect();
        let packed = dev
            .create_buffer(BufferDesc {
                size: words.len() as u64 * 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        let out = dev
            .create_buffer(BufferDesc {
                size: u64::from(n) * 4,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        dev.write_buffer(packed, 0, u32s_to_bytes(&words)).unwrap();
        let shader = dev
            .create_shader(NIBBLES_ZSL.spirv_desc().unwrap())
            .unwrap();
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
                buffers: &[packed.index(), out.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();
        let raw = dev.read_buffer(out, 0, u64::from(n) * 4).unwrap();
        for i in 0..n {
            let at = i as usize * 4;
            let got = u32::from_ne_bytes(raw[at..at + 4].try_into().unwrap());
            let want = ((words[(i >> 3) as usize] >> ((i & 7) << 2)) & 15) | ((i ^ 1) << 8);
            assert_eq!(got, want, "at {i}");
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(packed);
        dev.destroy_buffer(out);
    }

    /// f32 to nearest f16, ties to even, exact in f64. 65520 and up round to inf. NaN, inf,
    /// zero pass.
    fn f16_round_ref(x: f32) -> f32 {
        if !x.is_finite() || x == 0.0 {
            return x;
        }
        let mag = f64::from(x.abs());
        if mag >= 65520.0 {
            return f32::INFINITY.copysign(x);
        }
        let scale = 2f64.powf(mag.log2().floor() - 10.0);
        let rounded = (mag / scale).round_ties_even() * scale;
        (rounded as f32).copysign(x)
    }

    #[test]
    fn zsl_scalar_features_run_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const FEATURES_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel scalar_features(ua: device buffer<u32>, ia: device buffer<i32>, fa: device buffer<f32>, sb: device buffer<i8>, ub: device buffer<u8>, ou: device mut buffer<u32>, oi: device mut buffer<i32>, of: device mut buffer<f32>, ob: device mut buffer<i8>, oub: device mut buffer<u8>, oh: device mut buffer<i32>, oh2: device mut buffer<i32>, ofr: device mut buffer<f32>, osel: device mut buffer<i32>, id: global_id) {
                let i = id.x
                ou[i] = ua[i] % 7u
                oi[i] = ia[i] % 5
                of[i] = fa[i] % 2.5
                ob[i] = sb[i] + 1
                oub[i] = ub[i] + 1
                oh[i] = i32(fa[i] * 3.0)
                oh2[i] = i32(ua[i])
                ofr[i] = f16_round(fa[i])
                osel[i] = select(ia[i] < 0, ia[i] * 2, ia[i])
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 256usize;
        let ua: Vec<u32> = (0..n as u32).map(|i| i * 37 + 5).collect();
        let ia: Vec<i32> = (0..n as i32).map(|i| i * 13 - 1500).collect();
        let fa: Vec<f32> = (0..n)
            .map(|i| match i % 16 {
                0 => 3.0e9,
                1 => -3.0e9,
                _ => i as f32 * 0.37 - 40.0,
            })
            .collect();
        let sb: Vec<u8> = (0..n as i32).map(|i| (i * 7 - 500) as i8 as u8).collect();
        let ub: Vec<u8> = (0..n).map(|i| (i * 11 % 256) as u8).collect();
        let u32_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |data: &[u8], usage: BufferUsage| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size: data.len() as u64,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let inputs = [
            make(&u32_bytes(&ua), input),
            make(&i32_bytes(&ia), input),
            make(&f32_bytes(&fa), input),
            make(&sb, input),
            make(&ub, input),
        ];
        let out_sizes = [1024u64, 1024, 1024, 256, 256, 1024, 1024, 1024, 1024];
        let outs: Vec<_> = out_sizes
            .iter()
            .map(|&s| make(&vec![0u8; s as usize], output))
            .collect();

        let shader = dev
            .create_shader(FEATURES_ZSL.spirv_desc().unwrap())
            .unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [32, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = inputs
            .iter()
            .chain(outs.iter())
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [8, 1, 1],
        )
        .unwrap();
        let read = |b: zengpu_hal::BufferHandle, len: u64| dev.read_buffer(b, 0, len).unwrap();
        let words = |b: zengpu_hal::BufferHandle| -> Vec<u32> {
            read(b, 1024)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ou = words(outs[0]);
        let oi: Vec<i32> = words(outs[1]).into_iter().map(|w| w as i32).collect();
        let of: Vec<f32> = words(outs[2]).into_iter().map(f32::from_bits).collect();
        let ob = read(outs[3], 256);
        let oub = read(outs[4], 256);
        let oh: Vec<i32> = words(outs[5]).into_iter().map(|w| w as i32).collect();
        let oh2: Vec<i32> = words(outs[6]).into_iter().map(|w| w as i32).collect();
        let ofr: Vec<f32> = words(outs[7]).into_iter().map(f32::from_bits).collect();
        let osel: Vec<i32> = words(outs[8]).into_iter().map(|w| w as i32).collect();
        for i in 0..n {
            assert_eq!(ou[i], ua[i] % 7, "u32 % at {i}");
            assert_eq!(oi[i], ia[i] % 5, "i32 % at {i}");
            assert!(
                of[i] == fa[i] % 2.5,
                "f32 % at {i}: {} vs {}",
                of[i],
                fa[i] % 2.5
            );
            assert_eq!(ob[i], sb[i].wrapping_add(1), "i8 store at {i}");
            assert_eq!(oub[i], ub[i].wrapping_add(1), "u8 store at {i}");
            assert_eq!(oh[i], (fa[i] * 3.0) as i32, "i32(f32) at {i}");
            assert_eq!(oh2[i], ua[i] as i32, "i32(u32) at {i}");
            let want_round = f16_round_ref(fa[i]);
            assert!(
                ofr[i] == want_round || (ofr[i].is_nan() && want_round.is_nan()),
                "f16_round at {i}: {} vs {want_round}",
                ofr[i]
            );
            let want_sel = if ia[i] < 0 { ia[i] * 2 } else { ia[i] };
            assert_eq!(osel[i], want_sel, "select at {i}");
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for b in inputs.into_iter().chain(outs) {
            dev.destroy_buffer(b);
        }
    }

    /// `frag_load_a/b` from f32 shared: each subgroup stages its tile in f16 workgroup memory.
    #[test]
    fn zsl_frag_ab_from_shared_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const SHARED_AB_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel frag_shared_ab(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, out: device mut buffer<f32>) {
                shared sa: array<f32, 256>
                shared sb: array<f32, 256>
                let lane = local_id().x
                for i in 0..8 {
                    sa[lane + i * 32] = a[lane + i * 32]
                    sb[lane + i * 32] = b[lane + i * 32]
                }
                barrier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(sb, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let d = mma(fa, fb, acc)
                frag_store(out, 0, 16, d)
            }
        );

        let Some(dev) = try_device() else { return };
        let fa: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let fb: Vec<f32> = (0..256)
            .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
            .collect();
        let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |data: &[u8], usage: BufferUsage| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size: data.len() as u64,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(&f32_bytes(&fa), input);
        let b = make(&f32_bytes(&fb), input);
        let c = make(&f32_bytes(&fc), input);
        let out = make(&[0u8; 1024], output);

        let shader = dev
            .create_shader(SHARED_AB_ZSL.spirv_desc().unwrap())
            .unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [32, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = [a, b, c, out].iter().map(|b| b.index()).collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let got: Vec<f32> = dev
            .read_buffer(out, 0, 1024)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for m in 0..16 {
            for n in 0..16 {
                let mut s = fc[m * 16 + n];
                for k in 0..16 {
                    s += fa[m * 16 + k] * fb[k * 16 + n];
                }
                let at = m * 16 + n;
                assert!(
                    (got[at] - s).abs() <= 1e-3 * s.abs().max(1.0),
                    "shared a/b mma at ({m}, {n}): got {} want {s}",
                    got[at]
                );
            }
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, b, c, out] {
            dev.destroy_buffer(buf);
        }
    }

    /// `u32x4` buffer elements on SPIR-V. Scalar tail past the last full vector.
    #[test]
    fn zsl_u32x4_copy_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const VEC_COPY_ZSL: ZslShader = zsl!(
            push P { n: u32, tail: u32 }
            @workgroup_size(64)
            kernel vec_copy(src: device buffer<u32x4>, out: device mut buffer<u32x4>, words: device buffer<u32>, out_words: device mut buffer<u32>, p: P, id: global_id) {
                if id.x < p.n {
                    let v = src[id.x]
                    out[id.x] = u32x4(v.w, v.z, v.y + v.x, v.x)
                }
                if id.x < p.tail {
                    out_words[p.n * 4 + id.x] = words[p.n * 4 + id.x]
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let (n, tail) = (5u32, 3u32);
        let words: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let src_words: Vec<u32> = (0..20u32)
            .map(|i| i.wrapping_mul(40_503) ^ 0x9e37)
            .collect();
        let as_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |data: &[u8], usage: BufferUsage| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size: data.len() as u64,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let src = make(&as_bytes(&src_words), input);
        let wbuf = make(&as_bytes(&words), input);
        let out = make(&as_bytes(&[0u32; 20]), output);
        let out_words = make(&as_bytes(&[0u32; 32]), output);

        let shader = dev
            .create_shader(VEC_COPY_ZSL.spirv_desc().unwrap())
            .unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = [src, out, wbuf, out_words]
            .iter()
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[Scalar::U32(n), Scalar::U32(tail)],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let read = |b: zengpu_hal::BufferHandle, count: usize| -> Vec<u32> {
            dev.read_buffer(b, 0, (count * 4) as u64)
                .unwrap()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let got = read(out, 20);
        for i in 0..n as usize {
            let v = &src_words[i * 4..i * 4 + 4];
            let want = [v[3], v[2], v[1].wrapping_add(v[0]), v[0]];
            assert_eq!(&got[i * 4..i * 4 + 4], &want, "vector {i}");
        }
        let got_words = read(out_words, 32);
        let tail_start = (n * 4) as usize;
        for i in 0..tail as usize {
            assert_eq!(
                got_words[tail_start + i],
                words[tail_start + i],
                "tail word {i}"
            );
        }
        assert!(got_words[..tail_start].iter().all(|&w| w == 0));
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [src, wbuf, out, out_words] {
            dev.destroy_buffer(buf);
        }
    }

    /// `C = A*B + C` on one 16x16x16 tile, f16 and packed i8. Needs those configs (RADV gfx12).
    #[test]
    fn zsl_frag_mma_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const FRAG_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel frag_gemm(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, out: device mut buffer<f32>, ba: device buffer<u32>, bb: device buffer<u32>, ci: device buffer<i32>, oi: device mut buffer<i32>, zout: device mut buffer<f32>) {
                let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let d = mma(fa, fb, acc)
                frag_store(out, 0, 16, d)
                let zacc: frag<f32, 16, 16, acc> = frag_zero_acc()
                let dz = mma(fa, fb, zacc)
                frag_store(zout, 0, 16, dz)
                let ia: frag<i8, 16, 16, a> = frag_load_a(ba, 0, 16)
                let ib: frag<i8, 16, 16, b> = frag_load_b(bb, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                let id2 = mma(ia, ib, iacc)
                frag_store(oi, 0, 16, id2)
            }
        );

        let Some(dev) = try_device() else { return };
        let f16_bits = |x: f32| -> u16 {
            if x == 0.0 {
                return 0;
            }
            let b = x.to_bits();
            let sign = (b >> 16) & 0x8000;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = (b >> 13) & 0x3ff;
            (sign | ((e as u32) << 10) | m) as u16
        };
        let fa: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let fb: Vec<f32> = (0..256)
            .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
            .collect();
        let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
        let ba8: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
        let bb8: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
        let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
        let f16_bytes =
            |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let packed: Vec<u8> = ba8
            .chunks_exact(4)
            .chain(bb8.chunks_exact(4))
            .flat_map(|c| c.iter().map(|&b| b as u8).collect::<Vec<u8>>())
            .collect();
        let (pa, pb) = packed.split_at(256);
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(512, input, &f16_bytes(&fa));
        let b = make(512, input, &f16_bytes(&fb));
        let c = make(1024, input, &f32_bytes(&fc));
        let ba = make(256, input, pa);
        let bb = make(256, input, pb);
        let ci_buf = make(1024, input, &i32_bytes(&ci));
        let out = make(1024, output, &[0u8; 1024]);
        let oi = make(1024, output, &[0u8; 1024]);
        let zout = make(1024, output, &[0u8; 1024]);

        let shader = dev.create_shader(FRAG_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [32, 1, 1],
            })
            .unwrap();
        let slots: Vec<u32> = [a, b, c, out, ba, bb, ci_buf, oi, zout]
            .iter()
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let floats = |b| -> Vec<f32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ints = |b| -> Vec<i32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got_f, got_i, got_z) = (floats(out), ints(oi), floats(zout));
        for m in 0..16 {
            for n in 0..16 {
                let mut s = fc[m * 16 + n];
                let mut si = ci[m * 16 + n];
                for k in 0..16 {
                    s += fa[m * 16 + k] * fb[k * 16 + n];
                    si = si.wrapping_add(i32::from(ba8[m * 16 + k]) * i32::from(bb8[k * 16 + n]));
                }
                let at = m * 16 + n;
                assert!(
                    (got_f[at] - s).abs() <= 1e-3 * s.abs().max(1.0),
                    "f16 mma at ({m}, {n}): got {} want {s}",
                    got_f[at]
                );
                assert_eq!(got_i[at], si, "i8 mma at ({m}, {n})");
                let p: f32 = (0..16).map(|k| fa[m * 16 + k] * fb[k * 16 + n]).sum();
                assert!(
                    (got_z[at] - p).abs() <= 1e-3 * p.abs().max(1.0),
                    "zero-acc mma at ({m}, {n}): got {} want {p}",
                    got_z[at]
                );
            }
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, b, c, out, ba, bb, ci_buf, oi, zout] {
            dev.destroy_buffer(buf);
        }
    }

    /// Stores narrow into shared arrays on RADV (RNE for f16 and bf16).
    #[test]
    fn zsl_typed_shared_round_trips_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const TYPED_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel typed_rt(a: device buffer<f32>, ci: device buffer<i32>, ui: device buffer<u8>, ub: device buffer<u32>, oh: device mut buffer<f32>, oq: device mut buffer<i32>, ou: device mut buffer<u32>, ow: device mut buffer<f32>, on: device mut buffer<u32>) {
                shared sh: array<f16, 256>
                shared sq: array<i8, 256>
                shared su: array<u8, 256>
                shared sw: array<bf16, 256>
                shared sn: array<u32, 256>
                let lane = local_id().x
                for i in 0..8 {
                    sh[lane + i * 32] = a[lane + i * 32]
                    sq[lane + i * 32] = ci[lane + i * 32]
                    su[lane + i * 32] = ui[lane + i * 32]
                    sw[lane + i * 32] = a[lane + i * 32]
                    sn[lane + i * 32] = ub[lane + i * 32]
                }
                barrier()
                for i in 0..8 {
                    oh[lane + i * 32] = sh[lane + i * 32]
                    oq[lane + i * 32] = sq[lane + i * 32]
                    ou[lane + i * 32] = su[lane + i * 32]
                    ow[lane + i * 32] = sw[lane + i * 32]
                    on[lane + i * 32] = sn[lane + i * 32]
                }
            }
        );

        let f16_round = |x: f32| -> f32 {
            if !x.is_finite() || x == 0.0 {
                return x;
            }
            let a = x.abs();
            if a >= 65520.0 {
                return f32::INFINITY.copysign(x);
            }
            let e = a.log2().floor().max(-14.0);
            let q = 2f32.powf(e - 10.0);
            ((a / q).round_ties_even() * q).copysign(x)
        };
        let bf16_round = |x: f32| -> f32 {
            if !x.is_finite() {
                return x;
            }
            let b = x.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        };

        let Some(dev) = try_device() else { return };
        let a_vals: Vec<f32> = (0..256)
            .map(|i| {
                [
                    0.1,
                    65504.0,
                    65520.0,
                    1.0 + 2f32.powi(-11),
                    1.0 + 3.0 * 2f32.powi(-11),
                    1e-5,
                    -2.5e-7,
                    123.456,
                    2.75,
                    -0.0,
                ][i % 10]
            })
            .collect();
        let ci: Vec<i32> = (0..256i32).map(|i| i * 37 - 4000).collect();
        let ui: Vec<u8> = (0..256u32).map(|i| (i * 53 % 256) as u8).collect();
        let ub: Vec<u32> = (0..256u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u32_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(1024, input, &f32_bytes(&a_vals));
        let c = make(1024, input, &i32_bytes(&ci));
        let u8s = make(256, input, &ui);
        let uw = make(1024, input, &u32_bytes(&ub));
        let oh = make(1024, output, &[0u8; 1024]);
        let oq = make(1024, output, &[0u8; 1024]);
        let ou = make(1024, output, &[0u8; 1024]);
        let ow = make(1024, output, &[0u8; 1024]);
        let on = make(1024, output, &[0u8; 1024]);

        let shader = dev.create_shader(TYPED_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let slots: Vec<u32> = [a, c, u8s, uw, oh, oq, ou, ow, on]
            .iter()
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let read = |b| dev.read_buffer(b, 0, 1024).unwrap();
        let floats = |b| -> Vec<f32> {
            read(b)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ints = |b| -> Vec<i32> {
            read(b)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let uints = |b| -> Vec<u32> {
            read(b)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got_h, got_q, got_u) = (floats(oh), ints(oq), uints(ou));
        let (got_w, got_n) = (floats(ow), uints(on));
        for i in 0..256 {
            assert_eq!(
                got_h[i].to_bits(),
                f16_round(a_vals[i]).to_bits(),
                "f16 shared at {i}"
            );
            assert_eq!(got_q[i], i32::from(ci[i] as i8), "i8 narrowing at {i}");
            assert_eq!(got_u[i], u32::from(ui[i]), "u8 shared at {i}");
            assert_eq!(
                got_w[i].to_bits(),
                bf16_round(a_vals[i]).to_bits(),
                "bf16 shared at {i}"
            );
            assert_eq!(got_n[i], ub[i], "u32 shared at {i}");
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, c, u8s, uw, oh, oq, ou, ow, on] {
            dev.destroy_buffer(buf);
        }
    }

    /// `frag_load_a/b` load straight from f16 and i8 shared arrays, with no f16 stage.
    #[test]
    fn zsl_frag_from_typed_shared_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const SHARED_FRAG: ZslShader = zsl!(
            @workgroup_size(32)
            kernel shared_frag(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, ba: device buffer<i32>, bb: device buffer<i32>, ci: device buffer<i32>, out: device mut buffer<f32>, oi: device mut buffer<i32>) {
                shared sa: array<f16, 256>
                shared sb: array<f16, 256>
                shared qa: array<i8, 256>
                shared qb: array<i8, 256>
                let lane = local_id().x
                for i in 0..8 {
                    sa[lane + i * 32] = a[lane + i * 32]
                    sb[lane + i * 32] = b[lane + i * 32]
                    qa[lane + i * 32] = ba[lane + i * 32]
                    qb[lane + i * 32] = bb[lane + i * 32]
                }
                barrier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(sb, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                frag_store(out, 0, 16, mma(fa, fb, acc))
                let ia: frag<i8, 16, 16, a> = frag_load_a(qa, 0, 16)
                let ib: frag<i8, 16, 16, b> = frag_load_b(qb, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                frag_store(oi, 0, 16, mma(ia, ib, iacc))
            }
        );

        let Some(dev) = try_device() else { return };
        let f16_bits = |x: f32| -> u16 {
            if x == 0.0 {
                return 0;
            }
            let b = x.to_bits();
            let sign = (b >> 16) & 0x8000;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = (b >> 13) & 0x3ff;
            (sign | ((e as u32) << 10) | m) as u16
        };
        let fa: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let fb: Vec<f32> = (0..256)
            .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
            .collect();
        let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
        let ba8: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
        let bb8: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
        let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
        let f16_bytes =
            |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(512, input, &f16_bytes(&fa));
        let b = make(512, input, &f16_bytes(&fb));
        let c = make(1024, input, &f32_bytes(&fc));
        let ba32: Vec<i32> = ba8.iter().map(|&x| i32::from(x)).collect();
        let bb32: Vec<i32> = bb8.iter().map(|&x| i32::from(x)).collect();
        let ba = make(1024, input, &i32_bytes(&ba32));
        let bb = make(1024, input, &i32_bytes(&bb32));
        let ci_buf = make(1024, input, &i32_bytes(&ci));
        let out = make(1024, output, &[0u8; 1024]);
        let oi = make(1024, output, &[0u8; 1024]);

        let shader = dev
            .create_shader(SHARED_FRAG.spirv_desc().unwrap())
            .unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let slots: Vec<u32> = [a, b, c, ba, bb, ci_buf, out, oi]
            .iter()
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let floats = |b| -> Vec<f32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ints = |b| -> Vec<i32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got_f, got_i) = (floats(out), ints(oi));
        for m in 0..16 {
            for n in 0..16 {
                let mut s = fc[m * 16 + n];
                let mut si = ci[m * 16 + n];
                for k in 0..16 {
                    s += fa[m * 16 + k] * fb[k * 16 + n];
                    si = si.wrapping_add(i32::from(ba8[m * 16 + k]) * i32::from(bb8[k * 16 + n]));
                }
                let at = m * 16 + n;
                assert!(
                    (got_f[at] - s).abs() <= 1e-3 * s.abs().max(1.0),
                    "f16 mma from shared at ({m}, {n}): got {} want {s}",
                    got_f[at]
                );
                assert_eq!(got_i[at], si, "i8 mma from shared at ({m}, {n})");
            }
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, b, c, ba, bb, ci_buf, out, oi] {
            dev.destroy_buffer(buf);
        }
    }

    /// `frag_load_b_t` on RADV: N x K row-major f16 or packed i8 B, loaded as a ColumnMajor
    /// cooperative load.
    #[test]
    fn zsl_frag_load_b_t_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const BT_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel frag_bt(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, ba: device buffer<u32>, bb: device buffer<u32>, ci: device buffer<i32>, out: device mut buffer<f32>, oi: device mut buffer<i32>) {
                let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b_t(b, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                frag_store(out, 0, 16, mma(fa, fb, acc))
                let ia: frag<i8, 16, 16, a> = frag_load_a(ba, 0, 16)
                let ib: frag<i8, 16, 16, b> = frag_load_b_t(bb, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                frag_store(oi, 0, 16, mma(ia, ib, iacc))
            }
        );

        let Some(dev) = try_device() else { return };
        let f16_bits = |x: f32| -> u16 {
            if x == 0.0 {
                return 0;
            }
            let b = x.to_bits();
            let sign = (b >> 16) & 0x8000;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = (b >> 13) & 0x3ff;
            (sign | ((e as u32) << 10) | m) as u16
        };
        let fa: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let fb: Vec<f32> = (0..256)
            .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
            .collect();
        let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
        let ba8: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
        let bb8: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
        let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
        let f16_bytes =
            |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let pack = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(512, input, &f16_bytes(&fa));
        let b = make(512, input, &f16_bytes(&fb));
        let c = make(1024, input, &f32_bytes(&fc));
        let ba = make(256, input, &pack(&ba8));
        let bb = make(256, input, &pack(&bb8));
        let ci_buf = make(1024, input, &i32_bytes(&ci));
        let out = make(1024, output, &[0u8; 1024]);
        let oi = make(1024, output, &[0u8; 1024]);

        let shader = dev.create_shader(BT_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let slots: Vec<u32> = [a, b, c, ba, bb, ci_buf, out, oi]
            .iter()
            .map(|b| b.index())
            .collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let floats = |b| -> Vec<f32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ints = |b| -> Vec<i32> {
            dev.read_buffer(b, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got_f, got_i) = (floats(out), ints(oi));
        for m in 0..16 {
            for n in 0..16 {
                let mut s = fc[m * 16 + n];
                let mut si = ci[m * 16 + n];
                for k in 0..16 {
                    s += fa[m * 16 + k] * fb[n * 16 + k];
                    si = si.wrapping_add(i32::from(ba8[m * 16 + k]) * i32::from(bb8[n * 16 + k]));
                }
                let at = m * 16 + n;
                assert!(
                    (got_f[at] - s).abs() <= 1e-3 * s.abs().max(1.0),
                    "f16 b_t at ({m}, {n}): got {} want {s}",
                    got_f[at]
                );
                assert_eq!(got_i[at], si, "i8 b_t at ({m}, {n})");
            }
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, b, c, ba, bb, ci_buf, out, oi] {
            dev.destroy_buffer(buf);
        }
    }

    /// Compares bit builtins on RADV with the CPU: f16 decode and RNE pack on subnormal, inf, NaN
    /// and tie inputs.
    #[test]
    fn zsl_bit_builtins_run_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const BITS_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel bits(x: device buffer<u32>, y: device buffer<f32>, z: device buffer<f32>, lo: device mut buffer<f32>, hi: device mut buffer<f32>, packed: device mut buffer<u32>, bf: device mut buffer<f32>, bu: device mut buffer<u32>, bi: device mut buffer<i32>, sb: device mut buffer<i32>, ub: device mut buffer<u32>, id: global_id) {
                lo[id.x] = f16_lo(x[id.x])
                hi[id.x] = f16_hi(x[id.x])
                packed[id.x] = pack_f16x2(y[id.x], z[id.x])
                bf[id.x] = bitcast_f32(x[id.x])
                bu[id.x] = bitcast_u32(y[id.x])
                bi[id.x] = bitcast_i32(x[id.x])
                sb[id.x] = sbyte(x[id.x], id.x)
                ub[id.x] = byte(x[id.x], id.x)
            }
        );
        const HALVES: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00,
            0x3c00, 0xbc00, 0x3555, 0x0200, 0x7d00,
        ];
        const F32S: [f32; 16] = [
            0.0,
            -0.0,
            1.000_488_3,
            1.001_464_8,
            65520.0,
            2.980_232_2e-8,
            8.940_697e-8,
            1.0,
            -2.5,
            65504.0,
            f32::INFINITY,
            f32::NAN,
            1.0e-8,
            -1.0e-8,
            0.333_333_34,
            3.0e-5,
        ];
        const F32S_F16: [u16; 16] = [
            0x0000, 0x8000, 0x3c00, 0x3c02, 0x7c00, 0x0000, 0x0002, 0x3c00, 0xc100, 0x7bff, 0x7c00,
            0x7e00, 0x0000, 0x8000, 0x3555, 0x01f7,
        ];
        let f16_decode = |h: u16| -> f32 {
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let e = i32::from((h >> 10) & 0x1f);
            let m = f32::from(h & 0x3ff);
            if e == 0 {
                sign * m * 2f32.powi(-24)
            } else if e == 31 {
                if m == 0.0 {
                    sign * f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
            }
        };

        let Some(dev) = try_device() else { return };
        let n = 256usize;
        let xs: Vec<u32> = (0..n)
            .map(|i| u32::from(HALVES[i % 16]) | (u32::from(HALVES[(i * 7 + 3) % 16]) << 16))
            .collect();
        let ys: Vec<f32> = (0..n).map(|i| F32S[i % 16]).collect();
        let zs: Vec<f32> = (0..n).map(|i| F32S[(i * 5 + 1) % 16]).collect();
        let words = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let floats = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let size = (n * 4) as u64;
        let x = make(size, input, &words(&xs));
        let y = make(size, input, &floats(&ys));
        let z = make(size, input, &floats(&zs));
        let outs: Vec<_> = (0..8).map(|_| make(size, output, &[0u8; 1024])).collect();
        let shader = dev.create_shader(BITS_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let mut slots = vec![x.index(), y.index(), z.index()];
        slots.extend(outs.iter().map(|b| b.index()));
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [(n / 32) as u32, 1, 1],
        )
        .unwrap();
        let bytes = |b| dev.read_buffer(b, 0, size).unwrap();
        let floats_of = |b| -> Vec<f32> {
            bytes(b)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let words_of = |b| -> Vec<u32> {
            bytes(b)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let got_lo = floats_of(outs[0]);
        let got_hi = floats_of(outs[1]);
        let got_packed = words_of(outs[2]);
        let got_bf = floats_of(outs[3]);
        let got_bu = words_of(outs[4]);
        let got_bi = words_of(outs[5]);
        let got_sb = words_of(outs[6]);
        let got_ub = words_of(outs[7]);
        for i in 0..n {
            let x_bits = xs[i];
            let (lo_h, hi_h) = ((x_bits & 0xffff) as u16, (x_bits >> 16) as u16);
            let want_lo = f16_decode(lo_h);
            let want_hi = f16_decode(hi_h);
            assert!(
                if want_lo.is_nan() {
                    got_lo[i].is_nan()
                } else {
                    got_lo[i].to_bits() == want_lo.to_bits()
                },
                "f16_lo {lo_h:#06x}: {}",
                got_lo[i]
            );
            assert!(
                if want_hi.is_nan() {
                    got_hi[i].is_nan()
                } else {
                    got_hi[i].to_bits() == want_hi.to_bits()
                },
                "f16_hi {hi_h:#06x}: {}",
                got_hi[i]
            );
            let k = i % 16;
            let k2 = (i * 5 + 1) % 16;
            let (g_lo, g_hi) = (
                (got_packed[i] & 0xffff) as u16,
                (got_packed[i] >> 16) as u16,
            );
            for (got, want) in [(g_lo, F32S_F16[k]), (g_hi, F32S_F16[k2])] {
                let ok = if want == 0x7e00 {
                    got & 0x7fff > 0x7c00
                } else {
                    got == want
                };
                assert!(ok, "pack_f16x2 at {i}: got {got:#06x} want {want:#06x}");
            }
            assert_eq!(got_bf[i].to_bits(), x_bits, "bitcast_f32 at {i}");
            assert_eq!(got_bu[i], ys[i].to_bits(), "bitcast_u32 at {i}");
            assert_eq!(got_bi[i], x_bits, "bitcast_i32 at {i}");
            let lane = (i % 4) as u32;
            let byte = (x_bits >> (8 * lane)) & 0xff;
            assert_eq!(got_ub[i], byte, "byte at {i}");
            assert_eq!(
                got_sb[i] as i32,
                i32::from(byte as u8 as i8),
                "sbyte at {i}"
            );
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [x, y, z].into_iter().chain(outs) {
            dev.destroy_buffer(buf);
        }
    }

    /// `view` shared arrays on RADV: u32x4 stores through an f16 view, barrier, f16 loads, one mma.
    #[test]
    fn zsl_shared_view_mma_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const VIEW_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel view_mma(a: device buffer<u32x4>, b: device buffer<u32x4>, c: device buffer<f32>, out: device mut buffer<f32>) {
                shared sa: array<f16, 256>
                shared sb: array<f16, 256>
                view sa4: array<u32x4> = sa
                view sb4: array<u32x4> = sb
                let lane = local_id().x
                sa4[lane] = a[lane]
                sb4[lane] = b[lane]
                barrier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b_t(sb, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let done: frag<f32, 16, 16, acc> = mma(fa, fb, acc)
                frag_store(out, 0, 16, done)
            }
        );

        let Some(dev) = try_device() else { return };
        let f16_bits = |x: f32| -> u16 {
            if x == 0.0 {
                return 0;
            }
            let b = x.to_bits();
            let sign = (b >> 16) & 0x8000;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = (b >> 13) & 0x3ff;
            (sign | ((e as u32) << 10) | m) as u16
        };
        let fa: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let fb: Vec<f32> = (0..256)
            .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
            .collect();
        let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
        let f16_bytes =
            |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect() };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let input = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let a = make(512, input, &f16_bytes(&fa));
        let b = make(512, input, &f16_bytes(&fb));
        let c = make(1024, input, &f32_bytes(&fc));
        let out = make(1024, output, &[0u8; 1024]);

        let shader = dev.create_shader(VIEW_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let slots: Vec<u32> = [a, b, c, out].iter().map(|b| b.index()).collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let got: Vec<f32> = dev
            .read_buffer(out, 0, 1024)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for m in 0..16 {
            for n in 0..16 {
                let mut s = fc[m * 16 + n];
                for k in 0..16 {
                    s += fa[m * 16 + k] * fb[n * 16 + k];
                }
                let at = m * 16 + n;
                assert!(
                    (got[at] - s).abs() <= 1e-3 * s.abs().max(1.0),
                    "view mma at ({m}, {n}): got {} want {s}",
                    got[at]
                );
            }
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [a, b, c, out] {
            dev.destroy_buffer(buf);
        }
    }

    /// One buffer, two params (u32x4 writes, f16 reads after a barrier), on RADV.
    #[test]
    fn zsl_one_buffer_two_element_types_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const ALIAS_ZSL: ZslShader = zsl!(
            @workgroup_size(32)
            kernel alias_buf(src: device buffer<u32x4>, x: device mut buffer<u32x4>, xh: device buffer<f16>, out: device mut buffer<f32>, id: global_id) {
                x[id.x] = src[id.x]
                barrier()
                for j in 0..8 {
                    out[id.x * 8 + j] = xh[id.x * 8 + j]
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let vals: Vec<f32> = (0..256)
            .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
            .collect();
        let f16_bits = |x: f32| -> u16 {
            if x == 0.0 {
                return 0;
            }
            let b = x.to_bits();
            let sign = (b >> 16) & 0x8000;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = (b >> 13) & 0x3ff;
            (sign | ((e as u32) << 10) | m) as u16
        };
        let src_bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&x| f16_bits(x).to_le_bytes())
            .collect();
        let make = |size: u64, usage: BufferUsage, data: &[u8]| {
            let buf = dev
                .create_buffer(BufferDesc {
                    size,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(buf, 0, data).unwrap();
            buf
        };
        let rw = BufferUsage::STORAGE;
        let output = BufferUsage::STORAGE | BufferUsage::READBACK;
        let src = make(512, rw, &src_bytes);
        let x = make(512, rw, &[0u8; 512]);
        let out = make(1024, output, &[0u8; 1024]);

        let shader = dev.create_shader(ALIAS_ZSL.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry: "main",
                    block: [32, 1, 1],
                },
                zengpu_hal::ComputePipelineHints {
                    subgroup_size: Some(32),
                    capture_stats: false,
                },
            )
            .unwrap();
        let slots = [src.index(), x.index(), x.index(), out.index()];
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let got: Vec<f32> = dev
            .read_buffer(out, 0, 1024)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (h, (&g, &v)) in got.iter().zip(&vals).enumerate() {
            assert_eq!(g, v, "half {h}");
        }
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        for buf in [src, x, out] {
            dev.destroy_buffer(buf);
        }
    }

    #[test]
    fn compute_caps_are_reported_on_radv() {
        let Some(dev) = try_device() else { return };
        let caps = dev.compute_caps();
        let (lo, hi) = caps.subgroup_sizes;
        assert!(lo <= caps.default_subgroup && caps.default_subgroup <= hi);
        assert!(
            caps.shared_bytes >= 32 * 1024,
            "shared {}",
            caps.shared_bytes
        );
        assert!(
            caps.max_workgroup >= 256,
            "workgroup {}",
            caps.max_workgroup
        );
        for shape in &caps.mma {
            assert_eq!(
                (shape.m, shape.n, shape.k, shape.subgroup),
                (16, 16, 16, 32)
            );
        }
        assert!(
            caps.mma
                .iter()
                .any(|s| s.a == zengpu_hal::DType::F16 && s.acc == zengpu_hal::DType::F32)
        );
        assert!(
            caps.mma
                .iter()
                .any(|s| s.a == zengpu_hal::DType::I8 && s.acc == zengpu_hal::DType::I32)
        );
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
        let size = size_of_val(&input) as u64;
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
            u32s_to_bytes(unsafe { from_raw_parts(input.as_ptr() as *const u32, input.len()) }),
        )
        .unwrap();

        let shader = dev.create_shader(FINITE_ZSL.spirv_desc().unwrap()).unwrap();
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
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();

        let raw = dev.read_buffer(out, 0, size).unwrap();
        let actual: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_ne_bytes(*bytes))
            .collect();
        assert_eq!(actual, expected);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(src);
        dev.destroy_buffer(out);
    }

    /// Little-endian bytes of `v`, for a storage buffer.
    fn zsl_f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_ne_bytes()).collect()
    }

    fn zsl_bytes_f32(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect()
    }

    fn zsl_bytes_u32(b: &[u8]) -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c))
            .collect()
    }

    /// Bindings: `inputs`, then `outputs` (byte sizes, zero-filled).
    fn run_zsl(
        dev: &dyn GpuDevice,
        shader: &zengpu_spirv::ZslShader,
        block: [u32; 3],
        groups: [u32; 3],
        inputs: &[Vec<u8>],
        outputs: &[u64],
        scalars: &[Scalar],
    ) -> Vec<Vec<u8>> {
        let shader_h = dev.create_shader(shader.spirv_desc().unwrap()).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: shader_h,
                entry: "main",
                block,
            })
            .unwrap();
        let mut bufs = Vec::new();
        for bytes in inputs {
            let b = dev
                .create_buffer(BufferDesc {
                    size: bytes.len() as u64,
                    usage: BufferUsage::STORAGE,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(b, 0, bytes).unwrap();
            bufs.push(b);
        }
        for &size in outputs {
            let b = dev
                .create_buffer(BufferDesc {
                    size,
                    usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(b, 0, &vec![0u8; size as usize]).unwrap();
            bufs.push(b);
        }
        let slots: Vec<u32> = bufs.iter().map(|b| b.index()).collect();
        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars,
                textures: &[],
                images: &[],
            },
            groups,
        )
        .unwrap();
        let results = outputs
            .iter()
            .enumerate()
            .map(|(k, &size)| dev.read_buffer(bufs[inputs.len() + k], 0, size).unwrap())
            .collect();
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader_h);
        for b in bufs {
            dev.destroy_buffer(b);
        }
        results
    }

    #[test]
    fn zsl_mul_add_f32_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const FMA3: ZslShader = zsl!(
            @workgroup_size(64)
            kernel fma3(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                out[id.x] = mul_add_f32(a[id.x], b[id.x], c[id.x])
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 1024u32;
        let val = |i: u32, k: u32| {
            (i.wrapping_mul(2_654_435_761).wrapping_add(k) % 2001) as f32 / 100.0 - 10.0
        };
        let a: Vec<f32> = (0..n).map(|i| val(i, 1)).collect();
        let b: Vec<f32> = (0..n).map(|i| val(i, 2)).collect();
        let c: Vec<f32> = (0..n).map(|i| val(i, 3)).collect();
        let out = run_zsl(
            &*dev,
            &FMA3,
            [64, 1, 1],
            [n.div_ceil(64), 1, 1],
            &[zsl_f32_bytes(&a), zsl_f32_bytes(&b), zsl_f32_bytes(&c)],
            &[u64::from(n) * 4],
            &[],
        );
        let got = zsl_bytes_f32(&out[0]);
        for i in 0..n as usize {
            let want = a[i].mul_add(b[i], c[i]);
            assert_eq!(got[i].to_bits(), want.to_bits(), "fma at {i}");
        }
    }

    #[test]
    fn zsl_scatter_add_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const SCATTER: ZslShader = zsl!(
            push Scatter { n: u32 }
            @workgroup_size(256)
            kernel scatter_add(idx: device buffer<f32>, val: device buffer<f32>, out: device mut buffer<f32>, p: Scatter, id: global_id) {
                let i = id.x
                if i < p.n {
                    atomic_add(out, idx[i], val[i])
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let (n, bins) = (1000u32, 16u32);
        let idx: Vec<f32> = (0..n).map(|i| ((i * 7) % bins) as f32).collect();
        let val: Vec<f32> = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
        let out = run_zsl(
            &*dev,
            &SCATTER,
            [256, 1, 1],
            [n.div_ceil(256), 1, 1],
            &[zsl_f32_bytes(&idx), zsl_f32_bytes(&val)],
            &[u64::from(bins) * 4],
            &[Scalar::U32(n)],
        );
        let mut want = vec![0f32; bins as usize];
        for i in 0..n as usize {
            want[idx[i] as usize] += val[i];
        }
        assert_eq!(zsl_bytes_f32(&out[0]), want);
    }

    #[test]
    fn zsl_workgroup_reduce_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const WG_REDUCE: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(256)
            kernel workgroup_reduce(src: device buffer<f32>, out_add: device mut buffer<f32>, out_min: device mut buffer<f32>, out_max: device mut buffer<f32>, p: P, id: global_id) {
                if group_id().x < p.n {
                    let value = src[id.x]
                    out_add[id.x] = workgroup_reduce_add(value)
                    out_min[id.x] = workgroup_reduce_min(value)
                    out_max[id.x] = workgroup_reduce_max(value)
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let groups = 2u32;
        let n = groups * 256;
        let src: Vec<f32> = (0..n).map(|i| ((i * 37) % 61) as f32 - 30.0).collect();
        let out = run_zsl(
            &*dev,
            &WG_REDUCE,
            [256, 1, 1],
            [groups, 1, 1],
            &[zsl_f32_bytes(&src)],
            &[u64::from(n) * 4; 3],
            &[Scalar::U32(groups)],
        );
        let (add, min, max) = (
            zsl_bytes_f32(&out[0]),
            zsl_bytes_f32(&out[1]),
            zsl_bytes_f32(&out[2]),
        );
        for i in 0..n as usize {
            let g = &src[(i / 256) * 256..(i / 256 + 1) * 256];
            let sum: f32 = g.iter().sum();
            let lo = g.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = g.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!((add[i], min[i], max[i]), (sum, lo, hi), "lane {i}");
        }
    }

    #[test]
    fn zsl_subgroup_ops_run_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const QUERY: ZslShader = zsl!(
            @workgroup_size(256)
            kernel query(out_size: device mut buffer<u32>, out_id: device mut buffer<u32>, out_lane: device mut buffer<u32>, id: global_id) {
                out_size[id.x] = subgroup_size()
                out_id[id.x] = subgroup_id()
                out_lane[id.x] = subgroup_local_id()
            }
        );
        const PROOF: ZslShader = zsl!(
            @workgroup_size(256)
            kernel proof(out_reduce_add: device mut buffer<u32>, out_scan_inclusive: device mut buffer<u32>, out_scan_exclusive: device mut buffer<u32>, out_reduce_min: device mut buffer<u32>, out_reduce_max: device mut buffer<u32>, id: global_id) {
                out_reduce_add[id.x] = subgroup_reduce_add(1u32)
                out_scan_inclusive[id.x] = subgroup_scan_inclusive_add(1u32)
                out_scan_exclusive[id.x] = subgroup_scan_exclusive_add(1u32)
                out_reduce_min[id.x] = subgroup_reduce_min(subgroup_local_id())
                out_reduce_max[id.x] = subgroup_reduce_max(subgroup_local_id())
            }
        );
        const XOR: ZslShader = zsl!(
            @workgroup_size(256)
            kernel shuffle_xor(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = subgroup_shuffle_xor(id.x * 1.0, 1u32)
            }
        );

        let Some(dev) = try_device() else { return };
        let n = 256u32;
        let bytes = u64::from(n) * 4;
        let q = run_zsl(&*dev, &QUERY, [256, 1, 1], [1, 1, 1], &[], &[bytes; 3], &[]);
        let (size, id, lane) = (
            zsl_bytes_u32(&q[0]),
            zsl_bytes_u32(&q[1]),
            zsl_bytes_u32(&q[2]),
        );
        let w = size[0];
        assert!(w == 32 || w == 64, "subgroup size {w}");
        for i in 0..n as usize {
            assert_eq!(
                (size[i], id[i], lane[i]),
                (w, i as u32 / w, i as u32 % w),
                "query at {i}"
            );
        }
        let p = run_zsl(&*dev, &PROOF, [256, 1, 1], [1, 1, 1], &[], &[bytes; 5], &[]);
        for i in 0..n as usize {
            let l = i as u32 % w;
            let got: Vec<u32> = p.iter().map(|o| zsl_bytes_u32(o)[i]).collect();
            assert_eq!(got, [w, l + 1, l, 0, w - 1], "proof at {i}");
        }
        let x = run_zsl(&*dev, &XOR, [256, 1, 1], [1, 1, 1], &[], &[bytes], &[]);
        for (i, v) in zsl_bytes_f32(&x[0]).into_iter().enumerate() {
            assert_eq!(v, (i ^ 1) as f32, "shuffle_xor at {i}");
        }
    }

    #[test]
    fn zsl_tiled_gemm_runs_through_vulkan() {
        use zengpu_spirv::{ZslShader, zsl};

        const GEMM: ZslShader = zsl!(
            push Shape { m: u32, n: u32, k: u32 }
            @workgroup_size(16, 16)
            kernel tiled_gemm(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: Shape) {
                shared as_tile: array<f32, 256>
                shared bs_tile: array<f32, 256>
                let lx = local_id().x
                let ly = local_id().y
                let row = group_id().y * 16 + ly
                let col = group_id().x * 16 + lx
                let sum: f32 = 0.0
                let tile_count = (p.k + 15) / 16
                for tile in 0..tile_count {
                    let ak = tile * 16 + lx
                    let bk = tile * 16 + ly
                    if row < p.m && ak < p.k {
                        as_tile[ly * 16 + lx] = a[row * p.k + ak]
                    } else {
                        as_tile[ly * 16 + lx] = 0.0
                    }
                    if bk < p.k && col < p.n {
                        bs_tile[ly * 16 + lx] = b[bk * p.n + col]
                    } else {
                        bs_tile[ly * 16 + lx] = 0.0
                    }
                    barrier()
                    for q in 0..16 {
                        sum = sum + as_tile[ly * 16 + q] * bs_tile[q * 16 + lx]
                    }
                    barrier()
                }
                if row < p.m && col < p.n {
                    c[row * p.n + col] = sum
                }
            }
        );

        let Some(dev) = try_device() else { return };
        let (m, n, k) = (32u32, 32u32, 32u32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
        let out = run_zsl(
            &*dev,
            &GEMM,
            [16, 16, 1],
            [n / 16, m / 16, 1],
            &[zsl_f32_bytes(&a), zsl_f32_bytes(&b)],
            &[u64::from(m * n) * 4],
            &[Scalar::U32(m), Scalar::U32(n), Scalar::U32(k)],
        );
        let got = zsl_bytes_f32(&out[0]);
        for r in 0..m as usize {
            for c in 0..n as usize {
                let want: f32 = (0..k as usize)
                    .map(|q| a[r * k as usize + q] * b[q * n as usize + c])
                    .sum();
                assert_eq!(got[r * n as usize + c], want, "c[{r}][{c}]");
            }
        }
    }
}
