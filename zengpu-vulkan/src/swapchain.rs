//! Shared Vulkan swapchain plumbing used by every surface type, plus
//! per-platform surface creation.
//!
//! [`Swapchain`] owns the surface, swapchain, image views, command pool and
//! per-frame-in-flight command buffers, and sync objects (acquire/submit/
//! present + frame-index bookkeeping). It does **not** own anything shaped by
//! a particular render pass (render pass, framebuffers, depth buffers,
//! pipelines, per-surface vertex/instance buffers) stay with the consumer or a
//! renderer crate layered above ZenGPU.
//!
//! Consumers should place a `swapchain: Swapchain` field **last** in their
//! struct: Rust drops fields in declaration order, so the consumer's own
//! `Drop` (which destroys its render-pass-shaped resources, built from
//! `swapchain`'s image views) runs before `swapchain`'s `Drop` (which
//! destroys the image views/swapchain/surface/sync those resources were
//! built from).

use std::mem;
use std::sync::{Arc, Mutex};

use ash::{Device, khr, vk};
use zengpu_hal::{GpuError, PresentMode, Result, SurfaceError};

use crate::device::{VulkanDevice, VulkanDeviceInner};
use crate::instance::VulkanShared;
#[cfg(target_os = "macos")]
use crate::metal_layer::metal_layer_for;
#[cfg(target_os = "macos")]
use ash::ext;

/// Per-swapchain-image state that's rebuilt on resize / surface loss.
struct SwapchainResources {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
}

struct FrameState {
    current: usize,
    /// The queue serial of the frame last submitted from each slot, so the
    /// slot's fence wait can say that work has finished.
    serials: Vec<u64>,
}

/// Result of [`Swapchain::begin_frame`].
#[derive(Copy, Clone)]
pub enum BeginFrame {
    /// Window is minimised (zero extent) — skip this frame entirely.
    Skip,
    /// The swapchain was just rebuilt — the caller must rebuild any
    /// resources derived from `image_views()`/`extent()` (framebuffers,
    /// depth targets) and skip this frame.
    Recreated,
    /// Normal frame: `current` selects the per-frame-in-flight resources
    /// (command buffer, sync objects, instance buffers), `index` selects the
    /// acquired swapchain image (and its framebuffer).
    Image { current: usize, index: u32 },
}

/// Shared surface/swapchain/sync/command-pool plumbing. See module docs.
pub struct Swapchain {
    inner: Arc<VulkanDeviceInner>,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    format: vk::Format,
    present_mode: vk::PresentModeKHR,
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frames_in_flight: usize,
    resources: Mutex<SwapchainResources>,
    frame_state: Mutex<FrameState>,
}

// Safety: all mutable cross-frame state is behind Mutex; ash types are Send+Sync.
unsafe impl Send for Swapchain {}
unsafe impl Sync for Swapchain {}

impl Swapchain {
    pub fn new(
        device: &VulkanDevice,
        handles: &zengpu_hal::WindowHandles,
        config: zengpu_hal::SurfaceConfig,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let inner = Arc::clone(&device.inner);
        let shared = Arc::clone(&inner.shared);
        if !shared.surface_extensions {
            return Err(GpuError::Backend(
                "Swapchain::new requires VulkanInstance::new_with_surface()".to_string(),
            ));
        }
        let surface_loader = khr::surface::Instance::new(&shared.entry, &shared.instance);
        let swapchain_loader = khr::swapchain::Device::new(&shared.instance, &inner.device);
        let surface = create_platform_surface(&shared, handles)?;
        // Destroys what was made so far if any step below fails.
        let mut made = Partial {
            device: inner.device.clone(),
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            image_views: Vec::new(),
            cmd_pool: vk::CommandPool::null(),
            sync: FrameSync::default(),
        };

        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                inner.physical,
                inner.queue_family,
                surface,
            )
        }
        .map_err(|e| GpuError::Backend(format!("surface support query: {e}")))?;
        if !supports_present {
            // On MoltenVK, vkGetPhysicalDeviceSurfaceSupportKHR is unreliable —
            // the single graphics+compute queue family does present in practice
            // even when the query reports otherwise. Warn and let swapchain
            // creation be the real arbiter (it returns a clear error if present
            // is genuinely unsupported). On real Vulkan, trust the query.
            #[cfg(target_os = "macos")]
            {
                log::warn!(
                    "[zengpu-vulkan] queue family {} reports no present support \
                     (MoltenVK query is unreliable); proceeding with swapchain creation",
                    inner.queue_family
                );
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err(GpuError::Backend(format!(
                    "selected queue family {} cannot present to this surface",
                    inner.queue_family
                )));
            }
        }

        let present_mode = pick_present_mode(
            &surface_loader,
            inner.physical,
            surface,
            config.present_mode,
        )?;
        let format = pick_format(&surface_loader, inner.physical, surface)?;

        let (swapchain, images, extent) = create_swapchain(
            &surface_loader,
            &swapchain_loader,
            inner.physical,
            surface,
            format,
            present_mode,
            config.width,
            config.height,
            vk::SwapchainKHR::null(),
        )?;
        made.swapchain = swapchain;
        made.image_views = create_image_views(&inner.device, &images, format)?;
        made.cmd_pool = create_command_pool(&inner.device, inner.queue_family)?;
        let cmd_buffers = allocate_cmd_buffers(&inner.device, made.cmd_pool, frames_in_flight)?;
        made.sync = create_sync(&inner.device, frames_in_flight)?;

        // Everything is made: the swapchain owns it from here.
        let surface = mem::replace(&mut made.surface, vk::SurfaceKHR::null());
        let swapchain = mem::replace(&mut made.swapchain, vk::SwapchainKHR::null());
        let image_views = mem::take(&mut made.image_views);
        let cmd_pool = mem::replace(&mut made.cmd_pool, vk::CommandPool::null());
        let FrameSync {
            image_available,
            render_finished,
            in_flight,
        } = mem::take(&mut made.sync);
        drop(made);

        Ok(Self {
            inner,
            surface_loader,
            swapchain_loader,
            surface,
            format,
            present_mode,
            cmd_pool,
            cmd_buffers,
            image_available,
            render_finished,
            in_flight,
            frames_in_flight,
            resources: Mutex::new(SwapchainResources {
                swapchain,
                images,
                image_views,
                extent,
            }),
            frame_state: Mutex::new(FrameState {
                current: 0,
                serials: vec![0; frames_in_flight],
            }),
        })
    }

    /// Raw device handles for building render-pass-shaped resources on top of
    /// this swapchain. See [`DeviceContext`].
    pub fn context(&self) -> DeviceContext {
        DeviceContext {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn format(&self) -> zengpu_hal::Format {
        crate::from_vk_format(self.format).expect("swapchain has unsupported vk::Format")
    }

    pub fn extent(&self) -> (u32, u32) {
        let e = self.resources.lock().unwrap().extent;
        (e.width, e.height)
    }

    pub(crate) fn raw_format(&self) -> vk::Format {
        self.format
    }

    pub(crate) fn raw_extent(&self) -> vk::Extent2D {
        self.resources.lock().unwrap().extent
    }

    pub fn image_views(&self) -> Vec<vk::ImageView> {
        self.resources.lock().unwrap().image_views.clone()
    }

    pub fn image_count(&self) -> usize {
        self.resources.lock().unwrap().images.len()
    }

    pub fn images(&self) -> Vec<vk::Image> {
        self.resources.lock().unwrap().images.clone()
    }

    pub fn cmd_buffer(&self, current: usize) -> vk::CommandBuffer {
        self.cmd_buffers[current]
    }

    /// Wait for the next frame-in-flight slot, then acquire a swapchain
    /// image. Returns [`BeginFrame::Skip`] for a minimised window or
    /// [`BeginFrame::Recreated`] if the swapchain was just rebuilt — in
    /// either case the caller skips this frame (no [`Swapchain::end_frame`] call).
    pub fn begin_frame(&self) -> Result<BeginFrame> {
        let (current, last_serial) = {
            let state = self.frame_state.lock().unwrap();
            (state.current, state.serials[state.current])
        };

        unsafe {
            self.inner
                .device
                .wait_for_fences(&[self.in_flight[current]], true, u64::MAX)
                .map_err(|e| map_vk_err("wait_for_fences", e))?;
        }
        // The frame this slot last drew has finished, and the resources it
        // used can be freed.
        self.inner.serials.complete(last_serial);

        let mut res = self.resources.lock().unwrap();
        if res.extent.width == 0 || res.extent.height == 0 {
            return Ok(BeginFrame::Skip);
        }

        let index = match unsafe {
            self.swapchain_loader.acquire_next_image(
                res.swapchain,
                u64::MAX,
                self.image_available[current],
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let (w, h) = (res.extent.width, res.extent.height);
                self.recreate(&mut res, w, h)?;
                return Ok(BeginFrame::Recreated);
            }
            Err(e) => return Err(map_surface_err(e)),
        };

        unsafe {
            self.inner
                .device
                .reset_fences(&[self.in_flight[current]])
                .map_err(|e| GpuError::Backend(format!("reset_fences: {e}")))?;
        }

        Ok(BeginFrame::Image { current, index })
    }

    /// Submit `cmd` (recorded by the caller into `frame`'s command buffer)
    /// and present. Returns `true` if the swapchain was recreated, in which
    /// case the caller must rebuild framebuffers/depth before the next frame.
    /// No-op (returns `Ok(false)`) for [`BeginFrame::Skip`]/[`BeginFrame::Recreated`].
    pub fn end_frame(&self, frame: &BeginFrame, cmd: vk::CommandBuffer) -> Result<bool> {
        let (current, index) = match *frame {
            BeginFrame::Image { current, index } => (current, index),
            _ => return Ok(false),
        };

        let wait_semaphores = [self.image_available[current]];
        let signal_semaphores = [self.render_finished[current]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: cmd_bufs.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };
        let serial = {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.inner
                    .device
                    .queue_submit(self.inner.queue, &[submit_info], self.in_flight[current])
                    .map_err(|e| map_vk_err("queue_submit", e))?;
            }
            self.inner.serials.next()
        };
        self.frame_state.lock().unwrap().serials[current] = serial;

        let mut res = self.resources.lock().unwrap();
        let swapchains = [res.swapchain];
        let image_indices = [index];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };
        let present_result = {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.swapchain_loader
                    .queue_present(self.inner.queue, &present_info)
            }
        };
        let recreated = match present_result {
            Ok(_suboptimal) => false,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let (w, h) = (res.extent.width, res.extent.height);
                self.recreate(&mut res, w, h)?;
                true
            }
            Err(e) => return Err(map_surface_err(e)),
        };
        drop(res);

        self.frame_state.lock().unwrap().current = (current + 1) % self.frames_in_flight;
        Ok(recreated)
    }

    /// Force a swapchain recreation (e.g. after a resize). Safe to call when
    /// the window is minimised — the new extent may be zero.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        let mut res = self.resources.lock().unwrap();
        self.recreate(&mut res, width, height)
    }

    /// Recreate the swapchain + image views in place. Keeps the surface,
    /// format, and present mode.
    fn recreate(&self, res: &mut SwapchainResources, width: u32, height: u32) -> Result<()> {
        let _queue = self.inner.queue_lock.lock().unwrap();
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        let old_swapchain = res.swapchain;
        let (swapchain, images, extent) = create_swapchain(
            &self.surface_loader,
            &self.swapchain_loader,
            self.inner.physical,
            self.surface,
            self.format,
            self.present_mode,
            width,
            height,
            old_swapchain,
        )?;

        // The old swapchain and views stay until the new ones are all made, so
        // a failure leaves `res` as it was instead of pointing at freed
        // handles.
        let image_views = match create_image_views(&self.inner.device, &images, self.format) {
            Ok(views) => views,
            Err(e) => {
                unsafe { self.swapchain_loader.destroy_swapchain(swapchain, None) };
                return Err(e);
            }
        };
        unsafe {
            for &iv in &res.image_views {
                self.inner.device.destroy_image_view(iv, None);
            }
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        res.swapchain = swapchain;
        res.images = images;
        res.image_views = image_views;
        res.extent = extent;
        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let dev = &self.inner.device;
        unsafe {
            let res = self.resources.lock().unwrap();
            for &iv in &res.image_views {
                dev.destroy_image_view(iv, None);
            }
            self.swapchain_loader.destroy_swapchain(res.swapchain, None);
            drop(res);

            dev.free_command_buffers(self.cmd_pool, &self.cmd_buffers);
            dev.destroy_command_pool(self.cmd_pool, None);
            for i in 0..self.frames_in_flight {
                dev.destroy_semaphore(self.image_available[i], None);
                dev.destroy_semaphore(self.render_finished[i], None);
                dev.destroy_fence(self.in_flight[i], None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

/// Clean accessor over the raw Vulkan handles a consumer needs to build
/// render-pass-shaped resources (render pass, pipelines, framebuffers, depth
/// targets, vertex/index buffers) on top of a [`Swapchain`]. Obtained from
/// [`Swapchain::context`].
///
/// Intentionally raw for now: it hands back `ash`/`vk` handles plus the
/// physical-device memory properties, leaving buffer/image allocation to the
/// caller. Convenience allocators may be layered on later.
#[derive(Clone)]
pub struct DeviceContext {
    pub(crate) inner: Arc<VulkanDeviceInner>,
}

// Safety: same as VulkanDeviceInner — ash handles are Send+Sync.
unsafe impl Send for DeviceContext {}
unsafe impl Sync for DeviceContext {}

impl DeviceContext {
    pub(crate) fn from_inner(inner: Arc<VulkanDeviceInner>) -> Self {
        Self { inner }
    }

    /// Logical device — create render passes, pipelines, framebuffers, images.
    pub(crate) fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Instance — physical-device queries.
    pub fn instance(&self) -> &ash::Instance {
        &self.inner.shared.instance
    }

    /// Physical device backing the logical device.
    pub fn physical(&self) -> vk::PhysicalDevice {
        self.inner.physical
    }

    /// Queue used for submission/present.
    ///
    /// This is a raw escape hatch. Callers that submit directly must provide
    /// their own external synchronization; ZenGPU's internal submissions use
    /// the device-wide queue lock.
    pub fn queue(&self) -> vk::Queue {
        self.inner.queue
    }

    /// Queue-family index of [`queue`](Self::queue).
    pub fn queue_family(&self) -> u32 {
        self.inner.queue_family
    }

    /// Whether `device` is the logical device represented by this context.
    pub fn is_device(&self, device: &VulkanDevice) -> bool {
        Arc::ptr_eq(&self.inner, &device.inner)
    }

    /// Whether the selected physical device supports dual-source blending.
    pub fn supports_dual_source_blending(&self) -> bool {
        self.inner.dual_src_blend
    }

    /// Whether the selected physical device supports non-solid fill modes
    /// (`PolygonMode::Line`/`Point`).
    pub fn supports_non_solid_fill(&self) -> bool {
        self.inner.fill_mode_non_solid
    }

    /// Physical-device memory properties, for picking a memory type when
    /// allocating buffers/images.
    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.inner
                .shared
                .instance
                .get_physical_device_memory_properties(self.inner.physical)
        }
    }

    pub(crate) fn inner_arc(&self) -> Arc<VulkanDeviceInner> {
        Arc::clone(&self.inner)
    }

    /// Submit a one-shot command buffer — records work via `f`, then waits for
    /// completion. Useful for staging uploads and layout transitions in user-side
    /// surfaces that build on top of [`Swapchain`].
    pub fn one_shot_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<()>,
    {
        let dev = &self.inner.device;
        let pool = unsafe {
            dev.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: self.inner.queue_family,
                    flags: vk::CommandPoolCreateFlags::TRANSIENT,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("one_shot create_command_pool: {e}")))?
        };
        let cmd = unsafe {
            match dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            }) {
                Ok(v) => v[0],
                Err(e) => {
                    dev.destroy_command_pool(pool, None);
                    return Err(GpuError::Backend(format!(
                        "one_shot allocate_command_buffers: {e}"
                    )));
                }
            }
        };
        if let Err(e) = unsafe {
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
        } {
            unsafe { dev.destroy_command_pool(pool, None) };
            return Err(GpuError::Backend(format!(
                "one_shot begin_command_buffer: {e}"
            )));
        }
        let result = record(dev, cmd);
        unsafe {
            let _ = dev.end_command_buffer(cmd);
        }
        if let Err(e) = result {
            unsafe { dev.destroy_command_pool(pool, None) };
            return Err(e);
        }
        let _queue = self.inner.queue_lock.lock().unwrap();
        let submit_result = unsafe {
            dev.queue_submit(
                self.inner.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                vk::Fence::null(),
            )
            .map_err(|e| map_vk_err("one_shot queue_submit", e))
        };
        if submit_result.is_ok() {
            self.inner.serials.next();
        }
        unsafe {
            if dev.device_wait_idle().is_ok() {
                self.inner.serials.complete(self.inner.serials.submitted());
            }
            dev.destroy_command_pool(pool, None);
        }
        submit_result
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub(crate) fn map_surface_err(e: vk::Result) -> GpuError {
    match e {
        vk::Result::ERROR_DEVICE_LOST => GpuError::DeviceLost,
        vk::Result::ERROR_OUT_OF_DATE_KHR => SurfaceError::Outdated.into(),
        vk::Result::ERROR_SURFACE_LOST_KHR => SurfaceError::Lost.into(),
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            SurfaceError::OutOfMemory.into()
        }
        _ => GpuError::Backend(format!("surface error: {e}")),
    }
}

pub(crate) fn map_vk_err(op: &str, e: vk::Result) -> GpuError {
    match e {
        vk::Result::ERROR_DEVICE_LOST => GpuError::DeviceLost,
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
            GpuError::OutOfMemory(zengpu_hal::MemoryUsage::Upload)
        }
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            GpuError::OutOfMemory(zengpu_hal::MemoryUsage::GpuOnly)
        }
        _ => GpuError::Backend(format!("{op}: {e}")),
    }
}

fn pick_format(
    surface_loader: &khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::Format> {
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface formats: {e}")))?
    };
    // Prefer a non-sRGB BGRA format so straight sRGB colour bytes pass through
    // unchanged (matching the CPU rasterizer). Fall back to whatever's first.
    formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.first())
        .map(|f| f.format)
        .ok_or_else(|| GpuError::Backend("no surface formats".to_string()))
}

fn pick_present_mode(
    surface_loader: &khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    requested: PresentMode,
) -> Result<vk::PresentModeKHR> {
    let modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface present modes: {e}")))?
    };
    let desired = match requested {
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    Ok(if modes.contains(&desired) {
        desired
    } else {
        vk::PresentModeKHR::FIFO // guaranteed available
    })
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain(
    surface_loader: &khr::surface::Instance,
    swapchain_loader: &khr::swapchain::Device,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    format: vk::Format,
    present_mode: vk::PresentModeKHR,
    width: u32,
    height: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D)> {
    let caps = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface capabilities: {e}")))?
    };
    let color_space = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface formats: {e}")))?
    }
    .iter()
    .find(|f| f.format == format)
    .map(|f| f.color_space)
    .unwrap_or(vk::ColorSpaceKHR::SRGB_NONLINEAR);

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    // Minimised window: zero extent. Return an empty swapchain handle so the
    // caller can detect and skip presenting.
    if extent.width == 0 || extent.height == 0 {
        return Ok((vk::SwapchainKHR::null(), Vec::new(), extent));
    }

    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: image_count,
        image_format: format,
        image_color_space: color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        pre_transform: caps.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        old_swapchain,
        ..Default::default()
    };
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateSwapchainKHR: {e}")))?
    };
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|e| GpuError::Backend(format!("get_swapchain_images: {e}")))?
    };
    Ok((swapchain, images, extent))
}

pub(crate) fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let mut views = Vec::with_capacity(images.len());
    for &image in images {
        let info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        match unsafe { device.create_image_view(&info, None) } {
            Ok(view) => views.push(view),
            Err(e) => {
                for view in views {
                    unsafe { device.destroy_image_view(view, None) };
                }
                return Err(GpuError::Backend(format!("create_image_view: {e}")));
            }
        }
    }
    Ok(views)
}

fn create_command_pool(device: &ash::Device, queue_family: u32) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo {
        queue_family_index: queue_family,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    unsafe {
        device
            .create_command_pool(&info, None)
            .map_err(|e| GpuError::Backend(format!("create_command_pool: {e}")))
    }
}

fn allocate_cmd_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    count: usize,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: count as u32,
        ..Default::default()
    };
    unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| GpuError::Backend(format!("allocate_command_buffers: {e}")))
    }
}

/// A frame slot's semaphores and fence, for every slot.
#[derive(Default)]
struct FrameSync {
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
}

impl FrameSync {
    /// # Safety
    /// Every handle must be from `device` and unused.
    unsafe fn destroy(self, device: &Device) {
        unsafe {
            for semaphore in self.image_available.into_iter().chain(self.render_finished) {
                device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight {
                device.destroy_fence(fence, None);
            }
        }
    }
}

fn create_sync(device: &Device, count: usize) -> Result<FrameSync> {
    let sem_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo {
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    let mut sync = FrameSync {
        image_available: Vec::with_capacity(count),
        render_finished: Vec::with_capacity(count),
        in_flight: Vec::with_capacity(count),
    };
    for _ in 0..count {
        let made = unsafe {
            device
                .create_semaphore(&sem_info, None)
                .and_then(|available| {
                    sync.image_available.push(available);
                    let finished = device.create_semaphore(&sem_info, None)?;
                    sync.render_finished.push(finished);
                    device.create_fence(&fence_info, None)
                })
        };
        match made {
            Ok(fence) => sync.in_flight.push(fence),
            Err(e) => {
                unsafe { sync.destroy(device) };
                return Err(GpuError::Backend(format!("create sync objects: {e}")));
            }
        }
    }
    Ok(sync)
}

/// What [`Swapchain::new`] has made so far. Dropped before it finishes, it
/// destroys all of it; null handles and empty lists are what it gave away.
struct Partial {
    device: Device,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    cmd_pool: vk::CommandPool,
    sync: FrameSync,
}

impl Drop for Partial {
    fn drop(&mut self) {
        unsafe {
            mem::take(&mut self.sync).destroy(&self.device);
            if self.cmd_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.cmd_pool, None);
            }
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
            }
        }
    }
}

// ── Per-platform surface creation ───────────────────────────────────────────

#[cfg(target_os = "windows")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use zen_window_handle::WindowHandle;

    let win32_loader = khr::win32_surface::Instance::new(&shared.entry, &shared.instance);

    let WindowHandle::Win32(win32) = handles.window else {
        return Err(GpuError::Backend(
            "expected Win32 window handle on Windows".to_string(),
        ));
    };

    let create_info = vk::Win32SurfaceCreateInfoKHR {
        hwnd: win32.hwnd.get() as vk::HANDLE,
        hinstance: win32
            .hinstance
            .map(|h| h.get() as vk::HINSTANCE)
            .unwrap_or(0isize as vk::HINSTANCE),
        ..Default::default()
    };

    unsafe {
        win32_loader
            .create_win32_surface(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateWin32SurfaceKHR: {e}")))
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use zen_window_handle::{DisplayHandle, WindowHandle};

    match (handles.window, handles.display) {
        (WindowHandle::Xcb(window), DisplayHandle::Xcb(display)) => {
            let connection = display.connection.ok_or_else(|| {
                GpuError::Backend("XCB display handle has no connection".to_string())
            })?;
            let loader = khr::xcb_surface::Instance::new(&shared.entry, &shared.instance);
            let create_info = vk::XcbSurfaceCreateInfoKHR {
                connection: connection.as_ptr(),
                window: window.window.get(),
                ..Default::default()
            };
            unsafe {
                loader
                    .create_xcb_surface(&create_info, None)
                    .map_err(|e| GpuError::Backend(format!("vkCreateXcbSurfaceKHR: {e}")))
            }
        }
        (WindowHandle::Wayland(window), DisplayHandle::Wayland(display)) => {
            let loader = khr::wayland_surface::Instance::new(&shared.entry, &shared.instance);
            let create_info = vk::WaylandSurfaceCreateInfoKHR {
                display: display.display.as_ptr(),
                surface: window.surface.as_ptr(),
                ..Default::default()
            };
            unsafe {
                loader
                    .create_wayland_surface(&create_info, None)
                    .map_err(|e| GpuError::Backend(format!("vkCreateWaylandSurfaceKHR: {e}")))
            }
        }
        _ => Err(GpuError::Backend(
            "expected matching XCB or Wayland window/display handles on Linux".to_string(),
        )),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use zen_window_handle::WindowHandle;

    let WindowHandle::AppKit(appkit) = handles.window else {
        return Err(GpuError::Backend(
            "expected AppKit window handle on macOS".to_string(),
        ));
    };
    // SAFETY: the handle's contract is a live NSView, used on the main
    // thread like the window it belongs to.
    let layer = unsafe { metal_layer_for(appkit.ns_view.as_ptr().cast())? };
    let loader = ext::metal_surface::Instance::new(&shared.entry, &shared.instance);
    let create_info = vk::MetalSurfaceCreateInfoEXT {
        p_layer: layer.cast(),
        ..Default::default()
    };
    unsafe {
        loader
            .create_metal_surface(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateMetalSurfaceEXT: {e}")))
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub(crate) fn create_platform_surface(
    _shared: &VulkanShared,
    _handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    Err(GpuError::Backend(
        "unsupported surface platform".to_string(),
    ))
}
