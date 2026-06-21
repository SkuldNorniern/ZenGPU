//! Backend-independent device trait. Object-safe: `Box<dyn GpuDevice>` or `Arc<dyn GpuDevice>`
//! select at runtime.

use crate::command::{Bindings, DispatchOp};
use crate::desc::{BufferDesc, ComputePipelineDesc, SamplerDesc, ShaderDesc, TextureDesc};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::HalCapabilities;
use crate::types::Features;

/// `Send + Sync`, so worker threads can allocate and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Returns the backend device for backend-specific calls, e.g. a Vulkan swapchain
    /// from a `VulkanDevice`.
    fn as_any(&self) -> &dyn core::any::Any;

    fn capabilities(&self) -> HalCapabilities;

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Requires `BufferUsage::READBACK`.
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Uploads tightly packed pixels in the creation format (RGBA8: 4 bytes per pixel) and blocks
    /// until done.
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Nothing may still reference the image, as with `destroy_buffer`. A stale handle is a no-op.
    fn destroy_texture(&self, texture: TextureHandle);

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    fn destroy_sampler(&self, sampler: SamplerHandle);

    /// When `false`, `create_sampler` treats any anisotropy as `1` instead of failing. This is a
    /// quality setting, not a hard requirement.
    fn supports_anisotropic_filtering(&self) -> bool {
        false
    }

    /// The default returns `UnsupportedFeatures`, since there is no compute.
    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    fn destroy_shader(&self, _shader: ShaderHandle) {}

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    fn destroy_pipeline(&self, _pipeline: PipelineHandle) {}

    /// Dispatches a kernel and blocks until the GPU is done.
    ///
    /// `bindings.buffers` are bindless SSBO slot indices, indexed with `nonuniformEXT`.
    /// `bindings.scalars` become push constants, packed after the indices. `grid` is the workgroup
    /// count per axis.
    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Dispatches a batch of ops as one submission that blocks once, instead of one per op. Chained
    /// kernels (e.g. `relu(add(a, b))`) should use this instead of a `dispatch` loop.
    ///
    /// The default calls `dispatch` per op, so unimplemented backends stay correct.
    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        for op in ops {
            self.dispatch(op.pipeline, op.bindings, op.grid)?;
        }
        Ok(())
    }
}
