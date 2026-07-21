//! Backend-neutral graphics contract — the unified graphics API's foundation
//! (plan D17 / §15.5). Lives in `zengpu-hal`; `zengpu-vulkan` implements it;
//! the `zengpu` facade re-exports it. **No backend types appear here** — a
//! consumer records a render pass and draws without ever naming `vk::*`/`ash`.
//!
//! Associated types (not `Box<dyn ..>`) keep the surface allocation-free and
//! monomorphized — recording must not churn the heap (memory efficiency is a
//! first-class constraint). Only graphics-capable backends implement these; a
//! compute-only backend (CPU reference) does not.

use core::ops::Range;

use crate::command::Bindings;
use crate::desc::{GraphicsPipelineDesc, SurfaceConfig};
use crate::error::Result;
use crate::handle::{BufferHandle, PipelineHandle, TargetHandle};
use crate::surface::WindowHandles;
use crate::types::{Rect, Viewport};

// ── Render-pass value types ────────────────────────────────────────────────

/// What happens to an attachment's existing contents at the start of a pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp {
    /// Clear to the given RGBA value (color) or depth in component 0 (depth).
    Clear([f32; 4]),
    /// Preserve and read back the prior contents.
    Load,
    /// Contents are undefined; the pass overwrites them.
    DontCare,
}

impl LoadOp {
    /// Clear a color attachment to an opaque RGB color.
    pub const fn clear_rgb(r: f32, g: f32, b: f32) -> Self {
        LoadOp::Clear([r, g, b, 1.0])
    }

    /// Clear a depth attachment to `depth` (stored in component 0).
    pub const fn clear_depth(depth: f32) -> Self {
        LoadOp::Clear([depth, 0.0, 0.0, 0.0])
    }
}

/// One color attachment in a render pass: the target plus how it is loaded and
/// whether the result is kept.
#[derive(Debug, Clone, Copy)]
pub struct ColorAttachment {
    /// Render target to draw into — a swapchain [`Frame::target`] or an offscreen
    /// target. Backend-neutral; the backend resolves it.
    pub target: TargetHandle,
    pub load: LoadOp,
    /// Whether to store the result (`false` = `DONT_CARE` store).
    pub store: bool,
    /// If `true`, transition this target to a shader-readable layout when the
    /// pass ends, so a later pass can sample it as a bindless texture.
    /// This guarantee covers same-device, same-queue submission ordering only;
    /// no cross-submission semaphore or fence is exposed. Render-then-sample
    /// back-to-back is safe, while external queues, independent devices, and
    /// out-of-order submissions require caller-managed synchronization.
    pub sample_after: bool,
}

/// The depth/stencil attachment in a render pass.
#[derive(Debug, Clone, Copy)]
pub struct DepthAttachment {
    pub target: TargetHandle,
    pub load: LoadOp,
    pub store: bool,
}

/// Describes a render pass to begin on a command list: color attachments
/// (usually one) and an optional depth attachment. No backend types appear.
#[derive(Debug, Clone, Copy)]
pub struct RenderPassDesc<'a> {
    pub color: &'a [ColorAttachment],
    pub depth: Option<DepthAttachment>,
}

/// A viewport plus an optional scissor (the scissor doubles as a damage rect);
/// `None` scissor covers the whole viewport.
#[derive(Debug, Clone, Copy)]
pub struct ViewportScissor {
    pub viewport: Viewport,
    pub scissor: Option<Rect>,
}

// ── Behavioral contract ────────────────────────────────────────────────────

/// A graphics-capable device — extends [`GpuDevice`](crate::GpuDevice) with
/// surface/pipeline/command-list creation. Only graphics backends implement it.
pub trait GraphicsDevice: crate::GpuDevice {
    /// The presentable surface this backend produces.
    type Surface: Surface;
    /// The command recorder this backend produces (pooled & reusable — recording
    /// allocates nothing in steady state).
    type CommandList: RenderCommands;

    /// Create a presentable surface for `window`, configured by `config`.
    fn create_surface(
        &self,
        window: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self::Surface>;

    /// Create a graphics pipeline from a backend-neutral descriptor.
    fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> Result<PipelineHandle>;

    /// Acquire a reusable command list. Backends draw from a pool; the returned
    /// list records straight into a backend command buffer with no intermediate
    /// allocation.
    fn create_command_list(&self) -> Result<Self::CommandList>;

    /// Whether the device supports [`BlendMode::DualSourceAlpha`](crate::desc::BlendMode::DualSourceAlpha)
    /// (`dualSrcBlend`). Coverage-based text rendering falls back to
    /// [`BlendMode::AlphaBlend`](crate::desc::BlendMode::AlphaBlend) where this is `false`.
    fn supports_dual_source_blending(&self) -> bool;

    /// Whether the device supports [`PolygonMode::Line`](crate::desc::PolygonMode::Line) /
    /// [`PolygonMode::Point`](crate::desc::PolygonMode::Point) (`fillModeNonSolid`).
    /// [`create_graphics_pipeline`](Self::create_graphics_pipeline) fails if a
    /// non-solid fill mode is requested where this is `false`.
    fn supports_non_solid_fill(&self) -> bool;
}

/// An acquired-frame result: a real frame, or skip (minimized window, or the
/// surface just recreated itself internally — the caller drops the frame).
pub enum Acquire<F> {
    /// A frame ready to render into.
    Frame(F),
    /// No frame this tick — skip rendering (do not record/present).
    Skip,
}

/// A presentable surface: configure, acquire a frame, render, present. The
/// surface owns its swapchain and rebuilds it internally on resize/loss, so the
/// caller never touches framebuffers.
pub trait Surface {
    /// The per-frame token this surface hands out.
    type Frame: Frame;
    /// The command list type accepted by [`present`](Self::present).
    type CommandList: RenderCommands;

    /// (Re)configure size/format/present-mode.
    fn configure(&self, config: SurfaceConfig) -> Result<()>;

    /// Force a resize/recreate to `width`×`height`.
    fn resize(&self, width: u32, height: u32) -> Result<()>;

    /// Current backbuffer size in pixels.
    fn size(&self) -> (u32, u32);

    /// Acquire the next frame, or [`Acquire::Skip`].
    fn acquire(&self) -> Result<Acquire<Self::Frame>>;

    /// Submit the recorded `list` for `frame` and present — the windowed path
    /// owns its own sync, so no public fence is exposed here.
    fn present(&self, frame: Self::Frame, list: Self::CommandList) -> Result<()>;
}

/// A per-frame token. Carries the acquired image as a render target; the rest of
/// its state is backend-internal.
pub trait Frame {
    /// The acquired backbuffer as a render target, for [`ColorAttachment::target`].
    fn target(&self) -> TargetHandle;
}

/// Records draw commands into a (pooled) backend command buffer. Methods append
/// in order; no intermediate command list is built. `&mut self` is the only
/// synchronization needed — different lists record independently on any thread.
pub trait RenderCommands {
    /// Begin a named debug group in GPU debugging and profiling tools.
    fn push_debug_group(&mut self, label: &str);

    /// End the current named debug group.
    fn pop_debug_group(&mut self);

    /// Insert a named marker in GPU debugging and profiling tools.
    fn insert_debug_label(&mut self, label: &str);

    /// Begin a render pass over the given attachments.
    fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>);

    /// Bind the graphics pipeline used by subsequent draws.
    fn set_pipeline(&mut self, pipeline: PipelineHandle);

    /// Set the viewport and (optional) scissor.
    fn set_viewport_scissor(&mut self, vs: ViewportScissor);

    /// Bind bindless texture/buffer indices + inline scalars (push constants).
    fn bind(&mut self, bindings: Bindings<'_>);

    /// Bind a vertex buffer to `slot`.
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle);

    /// Bind the index buffer (32-bit indices).
    fn set_index_buffer(&mut self, buffer: BufferHandle);

    /// Draw `vertices` (non-indexed) for each instance in `instances`.
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);

    /// Draw `indices` (indexed) for each instance in `instances`.
    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>);

    /// Issue `draw_count` non-indexed indirect draws from `buffer`, beginning
    /// at `offset` bytes with entries `stride` bytes apart. Each entry has the
    /// `VkDrawIndirectCommand` layout `[vertex_count, instance_count,
    /// first_vertex, first_instance]` of `u32`s (16 bytes). Pass a zero stride
    /// for tightly packed entries.
    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32);

    /// Issue `draw_count` indexed indirect draws from `buffer`, beginning at
    /// `offset` bytes with entries `stride` bytes apart. Each entry has the
    /// `VkDrawIndexedIndirectCommand` layout `[index_count, instance_count,
    /// first_index, vertex_offset, first_instance]` (20 bytes), where
    /// `vertex_offset` is a signed `i32` and the other fields are `u32`. Pass a
    /// zero stride for tightly packed entries.
    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    );

    /// End the current render pass.
    fn end_render_pass(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_helpers() {
        assert_eq!(
            LoadOp::clear_rgb(0.1, 0.2, 0.3),
            LoadOp::Clear([0.1, 0.2, 0.3, 1.0])
        );
        assert_eq!(
            LoadOp::clear_depth(1.0),
            LoadOp::Clear([1.0, 0.0, 0.0, 0.0])
        );
    }
}
