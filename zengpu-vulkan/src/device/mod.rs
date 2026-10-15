//! Vulkan logical device and buffer operations.

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

/// Maximum number of storage buffers in each bindless SSBO table (per element
/// type — see [`SSBO_BINDING_F32`]/[`SSBO_BINDING_U32`]/[`SSBO_BINDING_I32`]).
const MAX_BINDLESS_BUFFERS: u32 = 4096;

/// Maximum number of combined image samplers in the bindless texture table.
const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Bindless SSBO table for `buffer<f32>` element access. Every `STORAGE`
/// buffer is registered here regardless of its ZSL-declared element type —
/// buffers are untyped at the HAL level (see [`crate::device::hal_impl`]).
const SSBO_BINDING_F32: u32 = 0;

/// Bindless combined-image-sampler table (unrelated to buffers; numbered here
/// only because it shares the same descriptor set layout).
const SSBO_BINDING_TEXTURES: u32 = 1;

/// Bindless SSBO table for `buffer<u32>` element access, same slot indices as
/// [`SSBO_BINDING_F32`] — added alongside it so a kernel's typed SPIR-V can
/// read/write `uint[]` directly instead of bitcasting through the f32 table.
const SSBO_BINDING_U32: u32 = 2;

/// Bindless SSBO table for `buffer<i32>` element access, same slot indices as
/// [`SSBO_BINDING_F32`].
const SSBO_BINDING_I32: u32 = 3;

/// Bindless storage-image table: every `STORAGE` texture's mip 0, in the
/// `GENERAL` layout, at its own slot index, for compute `image` params.
const STORAGE_IMAGE_BINDING: u32 = 4;

/// Number of distinct-element-type SSBO tables sharing the storage-buffer
/// descriptor budget (`f32`/`u32`/`i32` today). Append-only: a future bf16/f16
/// table adds a new binding number here without renumbering existing ones.
const SSBO_TYPED_TABLE_COUNT: u32 = 3;

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
    /// Slots in the storage-image table; 0 when the device cannot store to
    /// images from compute.
    storage_capacity: u32,
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

impl VulkanSubmissionState {
    const DONE: Self = Self {
        fence: None,
        cmd: None,
    };
}

/// Submissions dropped while their work was still running. Their fence,
/// command buffer and deferred frees are recycled once the work finishes,
/// which [`VulkanDevice::reap_orphans`] checks on the next submission,
/// instead of the drop waiting for the GPU.
type Orphans = Arc<Mutex<Vec<VulkanSubmission>>>;

enum DeferredVulkanResource {
    Buffer(u32, VulkanBuffer),
    Texture(u32, VulkanTexture),
    Sampler(u32, vk::Sampler),
    Pipeline(u32, VulkanPipeline),
}

/// Work submitted to the device's queue, numbered in submission order.
///
/// One queue signals in the order work was submitted, so once the work
/// numbered `n` is seen finished, everything numbered before it is too.
/// Every submission takes a number, compute and graphics alike, which is
/// what lets a resource be freed once the work that could still be using
/// it is done, and not before.
#[derive(Default)]
pub(crate) struct QueueSerials {
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl QueueSerials {
    /// Numbers a submission that has just reached the queue. Call it with
    /// the queue lock held, right after the submit succeeded, so numbers
    /// follow submission order.
    pub(crate) fn next(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Notes that the work numbered `serial` has finished.
    pub(crate) fn complete(&self, serial: u64) {
        self.completed.fetch_max(serial, Ordering::AcqRel);
    }

    pub(crate) fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    pub(crate) fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Whether every submission so far has finished.
    pub(crate) fn idle(&self) -> bool {
        self.completed() >= self.submitted()
    }
}

/// Resources waiting on the GPU, and which work last used which buffer.
#[derive(Default)]
struct VulkanLifetimeState {
    /// Destroyed while work was pending: freed once the work numbered with
    /// it has finished.
    deferred: Vec<(u64, DeferredVulkanResource)>,
    /// The last submission each buffer (by bindless index) was used by.
    buffer_use: HashMap<u32, u64>,
}

impl VulkanLifetimeState {
    /// Whether work not yet finished uses buffer `index`.
    fn buffer_busy(&self, index: u32, completed: u64) -> bool {
        self.buffer_use
            .get(&index)
            .is_some_and(|serial| *serial > completed)
    }

    /// Notes that submission `serial` uses the buffers at `indices`.
    fn mark_used(&mut self, indices: &[u32], serial: u64) {
        for &index in indices {
            self.buffer_use.insert(index, serial);
        }
    }

    /// What can be freed now that the work up to `completed` is done.
    fn take_ready(&mut self, completed: u64) -> Vec<DeferredVulkanResource> {
        self.buffer_use.retain(|_, serial| *serial > completed);
        let (ready, waiting): (Vec<_>, Vec<_>) = mem::take(&mut self.deferred)
            .into_iter()
            .partition(|(serial, _)| *serial <= completed);
        self.deferred = waiting;
        ready.into_iter().map(|(_, resource)| resource).collect()
    }
}

/// Fence-backed Vulkan submission. Completion releases its pooled fence and
/// command buffer exactly once; a timed-out wait leaves both owned by this
/// handle so a later poll/wait is safe.
struct VulkanSubmission {
    cycle_id: u64,
    /// This work's number in the queue.
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
            // Still running: the GPU may be using what this submission owns,
            // so it is kept, not waited for, and recycled once done.
            Ok(false) => {
                let orphan = self.orphan(state);
                self.orphans.lock().unwrap().push(orphan);
            }
            Err(_) => {
                // A device-lost/error path cannot safely recycle these
                // objects. Destroying the fence is legal after device loss;
                // the command pool remains alive and is reclaimed with the
                // device.
                unsafe { self.inner.device.destroy_fence(fence, None) };
                self.release_in_flight();
            }
        }
    }
}

impl VulkanSubmission {
    /// A copy of this submission holding `state`, to outlive the original.
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
    /// Where [`zengpu_hal::ColorAttachment::sample_after`] leaves it: the
    /// texture's resting layout, so samples and copies find it there.
    pub resting: vk::ImageLayout,
}

pub(crate) struct VulkanQueryPool {
    pub pool: vk::QueryPool,
    pub count: u32,
}

/// Shared logical device state — owned by `Arc` so swapchains can hold a ref.
pub(crate) struct VulkanDeviceInner {
    pub shared: Arc<VulkanShared>,
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Whether `VK_EXT_memory_budget` was enabled, so budget queries work.
    pub budget_enabled: bool,
    /// Whether this logical device enabled `VK_KHR_swapchain`.
    pub swapchain_enabled: bool,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// Vulkan queues require external host synchronization. Every internal
    /// submit/present operation using `queue` must hold this mutex.
    pub queue_lock: Mutex<()>,
    /// Numbers every submission to `queue`, for freeing resources once the
    /// work that used them is done.
    pub serials: QueueSerials,
    /// Set once the device is seen lost; it never comes back.
    pub lost: AtomicBool,
    /// An unsignalled fence asked about to learn whether the device is
    /// lost, which is the one thing a fence status reports regardless.
    pub probe: vk::Fence,
    pub features: Features,
    pub limits: DeviceLimits,
    pub graphics: bool,
    pub dual_src_blend: bool,
    pub fill_mode_non_solid: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    /// What `Format::Depth24PlusStencil8` means here: `D24_UNORM_S8_UINT`
    /// where the device has it, else `D32_SFLOAT_S8_UINT`. AMD drivers expose
    /// no D24S8 at all; "24 plus" allows the wider depth.
    pub depth24_stencil8: vk::Format,
    /// `VK_KHR_dynamic_rendering` loader — the unified graphics path (D17/GU)
    /// records render passes via `cmd_begin_rendering`/`cmd_end_rendering`,
    /// with no `vk::RenderPass`/`vk::Framebuffer` objects.
    pub dynamic_rendering: khr::dynamic_rendering::Device,
    /// Device-level `VK_EXT_debug_utils` commands, when the instance enabled
    /// the extension for validation and debugging.
    pub debug_utils: Option<ext::debug_utils::Device>,
}

// ash::Device is Send + Sync; vk::PhysicalDevice is a u64.
unsafe impl Send for VulkanDeviceInner {}
unsafe impl Sync for VulkanDeviceInner {}

/// The depth-stencil format backing `Format::Depth24PlusStencil8`: the first
/// of D24S8 and D32S8 usable as an optimal-tiling depth attachment.
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
    /// The Vulkan format this device uses for `format`.
    pub(crate) fn vk_format(&self, format: zengpu_hal::Format) -> vk::Format {
        match format {
            zengpu_hal::Format::Depth24PlusStencil8 => self.depth24_stencil8,
            other => hal_format_to_vk(other),
        }
    }

    /// Device-local heap sizes, with the budget and usage the driver reports
    /// for them. `None` unless `VK_EXT_memory_budget` was enabled.
    pub(crate) fn memory_budget(&self) -> Option<MemoryBudget> {
        if !self.budget_enabled {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        // Copy the heap table out so the chained borrow of `budget` ends here.
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

    /// Whether the device is lost.
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
    /// Bytes per texel of its format.
    pub texel_bytes: u32,
    /// Per array layer, one bit per mip level that holds texels and sits in
    /// the resting layout. A write to part of a level that is not ready
    /// leaves the rest of that level undefined.
    pub ready: Vec<u32>,
    /// The layout every ready level sits in between operations:
    /// `SHADER_READ_ONLY_OPTIMAL` for a texture only sampled, else `GENERAL`:
    /// a storage texture is written by compute while others sample it, and
    /// the read-only layout needs `SAMPLED` usage.
    pub resting: vk::ImageLayout,
    /// A view of mip 0 alone, for the storage-image table, when `view` has
    /// more levels than a storage descriptor takes; else null.
    pub storage_view: vk::ImageView,
}

impl VulkanTexture {
    /// The resting layout of a texture with `usage`.
    pub(crate) fn resting_layout(usage: vk::ImageUsageFlags) -> vk::ImageLayout {
        if usage.contains(vk::ImageUsageFlags::SAMPLED)
            && !usage.contains(vk::ImageUsageFlags::STORAGE)
        {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        }
    }

    /// Frees the texture's views, image and memory.
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

/// Vulkan logical device implementing [`GpuDevice`].
///
/// Drop every [`VulkanSurface`], [`OffscreenTarget`], and outstanding command
/// list derived from this device before dropping the device itself. Debug
/// builds panic on a violation; release builds silently destroy shared Vulkan
/// resources out from under any survivor, which is a real use-after-free.
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
    pub(crate) query_pools: Arc<Mutex<SlotMap<marker::QueryPool, VulkanQueryPool>>>,
    bindless: BindlessState,
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<PathBuf>,
    pub(crate) cmd_list_pool: Arc<CmdListPool>,
    fence_pool: Arc<FencePool>,
    lifetime: Arc<Mutex<VulkanLifetimeState>>,
    /// Reused host-visible memory uploads pass through.
    staging: Mutex<transfer::StagingRing>,
    /// Staging for host access to buffers the host cannot map.
    transfer_slots: Mutex<Option<dispatch::TransferSlots>>,
    orphans: Orphans,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    /// Recycles what dropped submissions held, for those whose work is done.
    /// Done ones are dropped after the list is unlocked, since a drop can
    /// add to it.
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

    /// Marks the device lost, as a driver reset would: [`is_lost`] says so,
    /// and a surface's acquire and present fail with
    /// [`GpuError::DeviceLost`]. For
    /// testing recovery without hanging a real GPU; there is no way back.
    ///
    /// [`is_lost`]: zengpu_hal::GpuDevice::is_lost
    pub fn simulate_loss(&self) {
        self.inner.lost.store(true, Ordering::Release);
    }

    /// Whether work that has not finished uses buffer `index`.
    pub(crate) fn buffer_busy(&self, index: u32) -> bool {
        let completed = self.inner.serials.completed();
        self.lifetime.lock().unwrap().buffer_busy(index, completed)
    }

    /// Frees whatever destroyed resources the finished work no longer needs.
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
        // Optional: without it the device reports no memory budget.
        let budget_enabled = supports_extension(ext::memory_budget::NAME);
        if budget_enabled {
            extensions.push(ext::memory_budget::NAME.as_ptr());
        }

        let supported_features = supported_features2.features;
        // Compute stores to any float format through one `Unknown`-format
        // image type, reached by a push-constant slot.
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

        // Enable Vulkan 1.2 descriptor-indexing features for bindless resources.
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
    /// Total texture slots available to consumers sharing this device's
    /// bindless budget, capped by `MAX_BINDLESS_TEXTURES`.
    pub fn bindless_texture_capacity(&self) -> u32 {
        self.bindless.texture_capacity
    }

    /// Texture slots currently used by consumers sharing this device's
    /// bindless budget; compare with `MAX_BINDLESS_TEXTURES`.
    pub fn bindless_texture_used(&self) -> u32 {
        self.bindless
            .bound_textures
            .lock()
            .unwrap()
            .iter()
            .filter(|&&bound| bound)
            .count() as u32
    }

    /// Total storage-buffer slots available to consumers sharing this device's
    /// bindless budget, capped by `MAX_BINDLESS_BUFFERS`.
    pub fn bindless_buffer_capacity(&self) -> u32 {
        self.bindless.buffer_capacity
    }

    /// Live storage-buffer slots used by consumers sharing this device's
    /// bindless budget; compare with `MAX_BINDLESS_BUFFERS`.
    pub fn bindless_buffer_used(&self) -> u32 {
        let buffers = self.buffers.lock().unwrap();
        (0..self.bindless.buffer_capacity)
            .filter_map(|index| buffers.get_by_slot_index(index))
            .filter(|buffer| buffer.usage.contains(BufferUsage::STORAGE))
            .count() as u32
    }

    /// Resolve every timestamp in `pool`, waiting for the GPU results to become
    /// available. Returns an error for a stale query-pool handle.
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

/// Slots in the buffer, sampled-texture and storage-image tables.
fn bindless_capacities(limits: DeviceLimits) -> Result<(u32, u32, u32)> {
    let total = limits.max_update_after_bind_descriptors;
    // `buffer_capacity` is a *per-table* slot count; there are
    // `SSBO_TYPED_TABLE_COUNT` STORAGE_BUFFER tables sharing one descriptor
    // pool, so the device's storage-buffer and total-descriptor budgets must
    // be divided across all of them up front.
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
    // Without storage images the binding is left out: its update-after-bind
    // flag would need a feature the device lacks.
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
        // Work still running when its submission was dropped finishes here,
        // so what it held is recycled before the device goes.
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
            // `inner` is also retained by the device-owned command-list and
            // fence pools, so its no-external-owner baseline is three.
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
            unsafe { tex.destroy(&self.inner.device) };
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
    use std::collections::VecDeque;
    use std::mem::size_of_val;
    use std::num::NonZeroIsize;
    use std::slice::from_raw_parts;
    use std::sync::MutexGuard;
    use std::thread::{sleep, spawn};
    use std::time::Instant;
    use std::{env, ops::Deref, process};

    use super::*;
    use crate::instance::VulkanInstance;
    use zengpu_hal::{
        AdapterRequest, AddressMode, Bindings, BlendState, BorderColor, ColorAttachment,
        ColorTargetState, DepthState, DeviceRequest, FilterMode, Format, GpuAdapter, GpuInstance,
        GraphicsDevice, GraphicsPipelineDesc, LoadOp, PipelineHandle, PrimitiveTopology,
        RasterState, RenderCommands, RenderPassDesc, SamplerDesc, Scalar, TargetHandle, TexDim,
        TextureHandle, TextureRegion, Viewport, ViewportScissor,
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
        // Validation errors must leave resource locks and subsequent uploads usable.
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
        // A 2x2 green tile taken out of a 3-texel-wide image, into (1, 1).
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
        // One blue texel from a buffer into (0, 0), on the device.
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
        // A write outside the texture is refused, not clipped.
        let outside = zengpu_hal::TextureWrite {
            region: zengpu_hal::TextureRegion::rect(3, 3, 2, 2),
            ..write
        };
        assert!(dev.write_textures(&[outside]).is_err());
        dev.destroy_buffer(src);
        dev.destroy_buffer(out);
        dev.destroy_texture(tex);
    }

    #[test]
    fn memory_budget_grows_with_device_allocations() {
        const MIB: u64 = 1024 * 1024;
        let Some(dev) = try_device() else { return };
        let Some(before) = dev.memory_budget() else {
            // VK_EXT_memory_budget is not available here; nothing to check.
            return;
        };
        assert!(before.budget <= before.device_local);
        assert!(before.usage > 0);
        let tex = dev
            .create_texture(TextureDesc {
                width: 4096,
                height: 4096,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .expect("create 64 MiB texture");
        // Drivers refresh the budget lazily, so poll for a little while.
        let mut after = before;
        for _ in 0..20 {
            after = dev.memory_budget().expect("budget stays available");
            if after.usage >= before.usage + 32 * MIB {
                break;
            }
            sleep(Duration::from_millis(50));
        }
        if after.usage < before.usage + 32 * MIB {
            // The driver did not move its counter within the poll window.
            eprintln!(
                "memory usage did not grow: {} -> {} bytes",
                before.usage, after.usage
            );
        }
        assert!(after.budget <= after.device_local);
        assert!(after.usage > 0);
        dev.destroy_texture(tex);
    }

    /// Dropping a submission whose work is still running hands it over
    /// instead of waiting, and the next submission recycles it once done.
    #[test]
    fn dropping_pending_work_does_not_wait() {
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
        let started = Instant::now();
        let pending = matches!(copy.poll(), Ok(SubmissionStatus::Pending));
        drop(copy);
        assert!(
            started.elapsed() < Duration::from_millis(5),
            "the drop waited"
        );
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
        // One distinct texel per pixel, so a misplaced row fails the check.
        let pixels: Vec<u8> = (0..256 * 256u32).flat_map(|i| i.to_le_bytes()).collect();
        dev.upload_texture_data(tex, &pixels).unwrap();
        let tile = vec![0x7f; 16 * 16 * 4];
        let out = dev
            .create_buffer(rw_desc(u64::try_from(pixels.len()).unwrap()))
            .unwrap();

        let started = Instant::now();
        let read = dev
            .copy_texture_region_to_buffer(
                tex,
                zengpu_hal::TextureRegion::rect(0, 0, 256, 256),
                out,
                0,
                256 * 4,
            )
            .unwrap();
        // Work submitted each frame is kept until it completes: dropping a
        // pending submission waits for the GPU.
        let mut frame_work: Vec<Submission> = Vec::new();
        let mut status = read.poll().unwrap();
        while status == SubmissionStatus::Pending {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "readback still pending after 5 s"
            );
            let frame = Instant::now();
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
            assert!(
                frame.elapsed() < Duration::from_millis(50),
                "a frame and its poll took {:?}",
                frame.elapsed()
            );
        }
        assert!(started.elapsed() < Duration::from_secs(5));
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
    fn a_working_device_is_not_lost() {
        let Some(dev) = try_device() else {
            return;
        };
        assert!(!dev.is_lost());
        assert!(!dev.is_lost(), "asking twice changes nothing");
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

    #[test]
    fn adapters_are_found_again_by_uuid() {
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        for adapter in inst.vulkan_adapters() {
            let again = inst
                .vulkan_adapter_by_uuid(adapter.uuid())
                .expect("still there");
            assert_eq!(again.info().name, adapter.info().name);
        }
        assert!(inst.vulkan_adapter_by_uuid([0xAB; 16]).is_none());
    }

    #[test]
    fn a_texture_pool_hands_back_what_was_released() {
        let Some(dev) = try_device() else {
            return;
        };
        let desc = TextureDesc {
            width: 64,
            height: 64,
            depth: 1,
            format: zengpu_hal::Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        };
        let tile = 64 * 64 * 4;
        let mut pool = zengpu_hal::TexturePool::new(2 * tile);
        let a = pool.acquire(&*dev, desc).unwrap();
        pool.release(&*dev, a);
        assert_eq!(pool.free_bytes(), tile);
        let again = pool.acquire(&*dev, desc).unwrap();
        assert_eq!(again, a, "the released texture comes back");
        assert_eq!(pool.free_bytes(), 0);
        // Past the budget the oldest free texture goes.
        let (b, c) = (
            pool.acquire(&*dev, desc).unwrap(),
            pool.acquire(&*dev, desc).unwrap(),
        );
        for t in [again, b, c] {
            pool.release(&*dev, t);
        }
        assert_eq!(pool.free_bytes(), 2 * tile);
        pool.clear(&*dev);
        assert_eq!((pool.free_bytes(), pool.live()), (0, 0));
    }

    #[test]
    fn many_tile_writes_reuse_staging() {
        let Some(dev) = try_device() else {
            return;
        };
        if !dev.capabilities().transfers.staged_uploads {
            return;
        }
        let tex = dev
            .create_texture(TextureDesc {
                width: 256,
                height: 256,
                depth: 1,
                format: zengpu_hal::Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let tile = vec![7u8; 64 * 64 * 4];
        // Far more tiles than the ring holds at once: spans have to come back.
        let mut pending = Vec::new();
        for i in 0..400u32 {
            let (x, y) = ((i % 4) * 64, ((i / 4) % 4) * 64);
            let write = zengpu_hal::TextureWrite {
                texture: tex,
                region: zengpu_hal::TextureRegion::rect(x, y, 64, 64),
                data: &tile,
                bytes_per_row: 64 * 4,
            };
            pending.push(dev.write_textures(&[write]).unwrap());
            if pending.len() > 8 {
                pending.remove(0).wait(Duration::from_secs(5)).unwrap();
            }
        }
        for submission in pending {
            submission.wait(Duration::from_secs(5)).unwrap();
        }
        dev.destroy_texture(tex);
    }

    /// Device-local buffers the host cannot map are written and read
    /// through staging, across chunk boundaries and at odd offsets.
    #[test]
    fn gpu_only_buffer_round_trips_through_staging() {
        let Some(dev) = try_device() else {
            return;
        };
        let size = (40u64 << 20) + 12; // spans three 16 MiB staging chunks
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
        // Aligned and small: sent inline with vkCmdUpdateBuffer.
        let inline: Vec<u8> = (0..64).map(|i| 200 - i).collect();
        dev.write_buffer(buf, 4096, &inline).unwrap();
        assert_eq!(dev.read_buffer(buf, 4092, 72).unwrap()[4..68], inline[..]);
        assert!(dev.read_buffer(buf, size - 2, 4).is_err(), "past the end");
        dev.destroy_buffer(buf);
    }

    /// Without Resizable BAR the device-local host-visible heap is a 256 MiB
    /// aperture with a few MiB to spare; a large `CpuToGpu` buffer must still
    /// be made (in host memory) rather than fail.
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

    /// AMD drivers have no D24S8; the format must still be creatable as a
    /// depth target, backed by D32S8 there.
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

    /// Every colour format a desktop Vulkan GPU must take: sampled and drawn
    /// into, as the spec requires of these.
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

    /// Upload, a region write and readback keep every byte, for every colour
    /// format, at its own texel size.
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
        // Bytes below 0x70 make no float NaN or infinity in any of the formats.
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
    /// Sampled texels, left to right: 2x1 source texture.
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

    /// How one channel of a format is stored.
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

    /// Stored channels: one for the red-only formats, otherwise four.
    fn channels(format: Format) -> usize {
        usize::try_from(format.bytes_per_texel()).unwrap() / lane_bytes(lane(format))
    }

    /// Absolute error a readback of `format` may show.
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

    /// Half bits by truncating the mantissa: exact for the normal test values.
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

    /// The unorm level (0..=`max`) nearest to `c`.
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

    /// One texel as linear RGBA. A one-channel format gives (r, 0, 0, 1).
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

    /// The bytes of one texel holding linear RGBA `rgba`.
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

    /// A `w` x `h` render target, primed so its readback starts from the
    /// shader-read layout.
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

    /// Every texel of a `w` x `h` texture, decoded to linear RGBA.
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

    /// A pipeline drawing `vs` and `fs` into one `format` target.
    fn quad_pipeline(
        device: &VulkanDevice,
        fs: &zengpu_spirv::ZslShader,
        format: Format,
        blend: Option<BlendState>,
    ) -> PipelineHandle {
        let vs_shader = device.create_shader(QUAD_VS.spirv_desc()).unwrap();
        let fs_shader = device.create_shader(fs.spirv_desc()).unwrap();
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

    /// Draws a full-screen quad over `target` and waits for it.
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
        // Half-red at alpha 0.5 over opaque blue.
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

    /// Clearing each colour render target reads back its clear value.
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

    /// Source-over alpha blending of half-red over opaque blue, per blendable format.
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

    /// Nearest sampling at each texel centre returns the uploaded value.
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
        // buffer_capacity = 4096.min(16/3=5).min((12-1)/3=3) = 3 (per-table);
        // texture_capacity = 1024.min(8).min(12 - 3*3=3) = 3, leaving no
        // storage images.
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
    fn bindless_usage_tracks_bound_textures_and_storage_buffers() {
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

        assert!(device.bindless_texture_capacity() > 0);
        assert!(device.bindless_buffer_capacity() > 0);
        assert_eq!(device.bindless_texture_used(), 0);
        assert_eq!(device.bindless_buffer_used(), 0);

        let texture = device
            .create_texture(TextureDesc {
                width: 1,
                height: 1,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .unwrap();
        let sampler = device.create_sampler(SamplerDesc::default()).unwrap();
        assert_eq!(device.bind_texture(texture, sampler), Some(texture.index()));
        assert_eq!(device.bindless_texture_used(), 1);

        let buffer = device
            .create_buffer(BufferDesc {
                size: 16,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        assert_eq!(device.bindless_buffer_used(), 1);

        device.destroy_buffer(buffer);
        device.destroy_sampler(sampler);
        device.destroy_texture(texture);
        assert_eq!(device.bindless_buffer_used(), 0);
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
        let shader = device.create_shader(CACHE_TEST_ZSL.spirv_desc()).unwrap();
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
        // Host writes reach device-local memory whether or not it is mapped
        // (staged when it is not); the copy below overwrites this byte.
        dev.write_buffer(gpu, 0, &[1]).unwrap();
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
            // Destroyed while the write may still be running on the GPU.
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

        // Sampled 8-bit texels in, inverted into a half-float image.
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
        // The first pass's image read back both ways, loaded as an image and
        // sampled as a texture at the texel's centre.
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
            let shader = dev.create_shader(shader.spirv_desc()).unwrap();
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
                // Half floats hold the middle pass.
                assert!((g - w).abs() < 2e-3, "texel {i}: {texel:?} != {want:?}");
            }
        }

        // An image slot must hold a storage texture.
        let not_storage = Bindings {
            buffers: &[],
            textures: &[src_slot],
            images: &[src.index()],
            scalars: &[Scalar::U32(w), Scalar::U32(h)],
        };
        assert!(dev.dispatch(invert, not_storage, [1, 1, 1]).is_err());
        // Storage textures are compute's: not drawn into, one 2D layer.
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
            u32s_to_bytes(unsafe { from_raw_parts(a_vals.as_ptr() as *const u32, a_vals.len()) }),
        )
        .unwrap();
        dev.write_buffer(
            b,
            0,
            u32s_to_bytes(unsafe { from_raw_parts(b_vals.as_ptr() as *const u32, b_vals.len()) }),
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
        // Only buffers the pending work uses are held: another one stays
        // writable and readable while it runs.
        let unrelated = dev.create_buffer(storage_rw(16)).unwrap();
        dev.write_buffer(unrelated, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(unrelated, 0, 4).unwrap(), vec![1, 2, 3, 4]);
        dev.destroy_buffer(unrelated);
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
        // Finished work lets go of its buffers.
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
        let bytes_i =
            u32s_to_bytes(unsafe { from_raw_parts(input_i.as_ptr() as *const u32, input_i.len()) });
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
                images: &[],
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
}
