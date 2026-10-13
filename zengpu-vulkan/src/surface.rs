//! Presentable surface over [`Swapchain`]. Callers use only `configure`, `resize`, `acquire` and
//! `present`, never `vk::SwapchainKHR` or framebuffers.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ash::vk;
use zengpu_hal::{
    Acquire, Frame, GpuError, Result, SlotMap, Surface, SurfaceConfig, TargetHandle, WindowHandles,
    marker,
};

use crate::command_list::{COLOR_SUBRESOURCE, CmdListPool, VulkanCommandList};
use crate::device::{VulkanDevice, VulkanDeviceInner, VulkanRenderTarget};
use crate::swapchain::{BeginFrame, Swapchain};

/// Frames in flight. Three slots absorb one slow frame and bound the latency.
const FRAMES_IN_FLIGHT: usize = 3;

/// Swapchain image acquired for rendering, with its render-target handle.
pub struct VulkanFrame {
    state: BeginFrame,
    target: TargetHandle,
}

impl Frame for VulkanFrame {
    fn target(&self) -> TargetHandle {
        self.target
    }
}

/// Presentable surface over [`Swapchain`]. Drop before its `VulkanDevice`.
pub struct VulkanSurface {
    inner: Arc<VulkanDeviceInner>,
    swapchain: Swapchain,
    render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    cmd_list_pool: Arc<CmdListPool>,
    /// One render-target handle per swapchain image, rebuilt on resize/recreate.
    targets: Mutex<Vec<TargetHandle>>,
    /// Command buffer for in-flight slot `current`. Returned to the pool once that slot's
    /// fence signals.
    pending_cmd: Mutex<Vec<Option<vk::CommandBuffer>>>,
}

impl VulkanSurface {
    pub(crate) fn new(
        device: &VulkanDevice,
        handles: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self> {
        let swapchain = Swapchain::new(device, handles, config, FRAMES_IN_FLIGHT)?;
        let surface = Self {
            inner: Arc::clone(&device.inner),
            swapchain,
            render_targets: Arc::clone(&device.render_targets),
            cmd_list_pool: Arc::clone(&device.cmd_list_pool),
            targets: Mutex::new(Vec::new()),
            pending_cmd: Mutex::new(vec![None; FRAMES_IN_FLIGHT]),
        };
        surface.rebuild_targets();
        Ok(surface)
    }

    /// Re-register the swapchain's images as render targets, dropping the old ones. Called on
    /// creation and on any recreation reported by `begin_frame`/`end_frame`.
    fn rebuild_targets(&self) {
        let images = self.swapchain.images();
        let image_views = self.swapchain.image_views();
        let format = self.swapchain.raw_format();
        let extent = self.swapchain.raw_extent();

        let mut render_targets = self.render_targets.lock().unwrap();
        let mut targets = self.targets.lock().unwrap();
        for handle in targets.drain(..) {
            render_targets.remove(handle);
        }
        for (&image, &view) in images.iter().zip(&image_views) {
            targets.push(render_targets.insert(VulkanRenderTarget {
                image,
                view,
                format,
                extent,
                layout: vk::ImageLayout::UNDEFINED,
                resting: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }));
        }
    }

    /// Barrier `target` to `PRESENT_SRC_KHR` if needed. Dynamic rendering makes no automatic
    /// transitions: do this before presenting.
    fn transition_to_present(&self, cmd: vk::CommandBuffer, target: TargetHandle) {
        let mut render_targets = self.render_targets.lock().unwrap();
        let Some(rt) = render_targets.get_mut(target) else {
            return;
        };
        if rt.layout == vk::ImageLayout::PRESENT_SRC_KHR {
            return;
        }
        let barrier = vk::ImageMemoryBarrier {
            old_layout: rt.layout,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::empty(),
            image: rt.image,
            subresource_range: COLOR_SUBRESOURCE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..Default::default()
        };
        rt.layout = vk::ImageLayout::PRESENT_SRC_KHR;
        drop(render_targets);
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
}

impl Surface for VulkanSurface {
    type Frame = VulkanFrame;
    type CommandList = VulkanCommandList;

    fn configure(&self, config: SurfaceConfig) -> Result<()> {
        self.swapchain.resize(config.width, config.height)?;
        self.rebuild_targets();
        Ok(())
    }

    fn resize(&self, width: u32, height: u32) -> Result<()> {
        self.swapchain.resize(width, height)?;
        self.rebuild_targets();
        Ok(())
    }

    fn size(&self) -> (u32, u32) {
        self.swapchain.extent()
    }

    fn acquire(&self) -> Result<Acquire<VulkanFrame>> {
        if self.inner.lost.load(Ordering::Acquire) {
            return Err(GpuError::DeviceLost);
        }
        match self.swapchain.begin_frame()? {
            BeginFrame::Skip => Ok(Acquire::Skip),
            BeginFrame::Recreated => {
                self.rebuild_targets();
                Ok(Acquire::Skip)
            }
            BeginFrame::Image { current, index } => {
                if let Some(cmd) = self.pending_cmd.lock().unwrap()[current].take() {
                    self.cmd_list_pool.release(cmd);
                }
                let target = self.targets.lock().unwrap()[index as usize];
                Ok(Acquire::Frame(VulkanFrame {
                    state: BeginFrame::Image { current, index },
                    target,
                }))
            }
        }
    }

    fn present(&self, frame: VulkanFrame, list: VulkanCommandList) -> Result<()> {
        if self.inner.lost.load(Ordering::Acquire) {
            return Err(GpuError::DeviceLost);
        }
        let BeginFrame::Image { current, .. } = frame.state else {
            return Err(GpuError::Backend(
                "VulkanSurface::present called with a non-Image frame".to_string(),
            ));
        };

        let cmd = list.raw();
        self.transition_to_present(cmd, frame.target);
        unsafe {
            self.inner
                .device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }

        let recreated = self.swapchain.end_frame(&frame.state, cmd)?;
        self.pending_cmd.lock().unwrap()[current] = Some(cmd);
        if recreated {
            self.rebuild_targets();
        }
        Ok(())
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        let mut render_targets = self.render_targets.lock().unwrap();
        for handle in self.targets.lock().unwrap().drain(..) {
            render_targets.remove(handle);
        }
    }
}
