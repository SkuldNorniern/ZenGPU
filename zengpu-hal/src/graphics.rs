//! Backend-neutral graphics contract. There are no backend types here, so passes and draws are
//! recorded without naming `vk::*`. Associated types are used instead of `Box<dyn>`, so recording
//! does not allocate. Only graphics-capable backends implement it.

use core::ops::Range;

use crate::command::Bindings;
use crate::desc::{GraphicsPipelineDesc, SurfaceConfig};
use crate::error::Result;
use crate::handle::{BufferHandle, PipelineHandle, TargetHandle};
use crate::surface::WindowHandles;
use crate::types::{Rect, Viewport};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp {
    /// Clear to the given RGBA value for color, or to the depth in component 0 for depth.
    Clear([f32; 4]),
    Load,
    /// Contents are undefined, and the pass overwrites them.
    DontCare,
}

impl LoadOp {
    pub const fn clear_rgb(r: f32, g: f32, b: f32) -> Self {
        LoadOp::Clear([r, g, b, 1.0])
    }

    pub const fn clear_depth(depth: f32) -> Self {
        LoadOp::Clear([depth, 0.0, 0.0, 0.0])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorAttachment {
    /// The swapchain `Frame::target` or an offscreen target. The backend resolves it.
    pub target: TargetHandle,
    pub load: LoadOp,
    /// `false` means a DONT_CARE store.
    pub store: bool,
    /// When `true`, transitions the target to a shader-readable layout at pass end, so a later pass
    /// can sample it bindlessly.
    pub sample_after: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct DepthAttachment {
    pub target: TargetHandle,
    pub load: LoadOp,
    pub store: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderPassDesc<'a> {
    pub color: &'a [ColorAttachment],
    pub depth: Option<DepthAttachment>,
}

/// `None` means the whole viewport. The scissor doubles as a damage rect.
#[derive(Debug, Clone, Copy)]
pub struct ViewportScissor {
    pub viewport: Viewport,
    pub scissor: Option<Rect>,
}

/// Implemented by graphics backends only.
pub trait GraphicsDevice: crate::GpuDevice {
    type Surface: Surface;
    /// Pooled, so recording allocates nothing in steady state.
    type CommandList: RenderCommands;

    fn create_surface(
        &self,
        window: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self::Surface>;

    fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> Result<PipelineHandle>;

    /// Pooled list that records straight into the backend command buffer.
    fn create_command_list(&self) -> Result<Self::CommandList>;
}

/// The frame was skipped: the window is minimized, or the surface was recreated.
///
/// The caller drops the frame.
pub enum Acquire<F> {
    Frame(F),
    Skip,
}

/// Owns the swapchain and rebuilds it on resize or loss.
pub trait Surface {
    type Frame: Frame;
    type CommandList: RenderCommands;

    fn configure(&self, config: SurfaceConfig) -> Result<()>;

    fn resize(&self, width: u32, height: u32) -> Result<()>;

    /// Current backbuffer size in pixels.
    fn size(&self) -> (u32, u32);

    fn acquire(&self) -> Result<Acquire<Self::Frame>>;

    /// The windowed path owns its synchronization. There is no public fence.
    fn present(&self, frame: Self::Frame, list: Self::CommandList) -> Result<()>;
}

/// Per-frame token. The rest of its state is internal to the backend.
pub trait Frame {
    fn target(&self) -> TargetHandle;
}

/// Command list. Methods append in order, and `&mut self` is the only synchronization needed.
///
/// Separate lists can record on any thread.
pub trait RenderCommands {
    fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>);

    fn set_pipeline(&mut self, pipeline: PipelineHandle);

    fn set_viewport_scissor(&mut self, vs: ViewportScissor);

    fn bind(&mut self, bindings: Bindings<'_>);

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle);

    fn set_index_buffer(&mut self, buffer: BufferHandle);

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);

    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>);

    fn end_render_pass(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_helpers() {
        assert_eq!(LoadOp::clear_rgb(0.1, 0.2, 0.3), LoadOp::Clear([0.1, 0.2, 0.3, 1.0]));
        assert_eq!(LoadOp::clear_depth(1.0), LoadOp::Clear([1.0, 0.0, 0.0, 0.0]));
    }
}
