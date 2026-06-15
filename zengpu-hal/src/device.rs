//! The device trait: backend-independent handle onto a GPU (or the CPU reference).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn GpuDevice>` and
//! selected at runtime.

use crate::command::Bindings;
use crate::desc::{BufferDesc, ComputePipelineDesc, SamplerDesc, ShaderDesc, TextureDesc};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::HalCapabilities;
use crate::types::Features;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker threads can allocate
/// and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Returns the backend device for backend-specific calls, e.g. a Vulkan swapchain
    /// from a `VulkanDevice`.
    fn as_any(&self) -> &dyn core::any::Any;

    fn capabilities(&self) -> HalCapabilities;

    /// Returned handle is generational.
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Buffer must be created with [`crate::BufferUsage::READBACK`].
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Uploads pixel data to a texture. `data` must be tightly packed in the creation format (RGBA8
    /// is 4 bytes per pixel). Blocks until the upload completes.
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Nothing may still reference the image, as with `destroy_buffer`. A stale handle is a no-op.
    fn destroy_texture(&self, texture: TextureHandle);

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    fn destroy_sampler(&self, sampler: SamplerHandle);

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

    /// Blocks until GPU work completes.
    ///
    /// `bindings.buffers` are slot indices into the bindless SSBO table, indexed with
    /// `nonuniformEXT`. `bindings.scalars` become push constants, packed after the buffer indices.
    /// `grid` is the workgroup count per axis.
    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }
}
