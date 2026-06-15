//! The device trait — the backend-independent handle onto a GPU (or the CPU
//! reference). This is the first slice of the split HAL; it grows as
//! backends come online (compute dispatch, graphics passes, surfaces).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn
//! GpuDevice>` and selected at runtime.

use crate::command::Bindings;
use crate::desc::{BufferDesc, ComputePipelineDesc, SamplerDesc, ShaderDesc, TextureDesc};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::HalCapabilities;
use crate::types::Features;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker
/// threads can allocate and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Downcast to the concrete backend type. Required for backend-specific
    /// operations (e.g. creating a Vulkan swapchain from a VulkanDevice).
    fn as_any(&self) -> &dyn core::any::Any;

    /// Which HAL capabilities this backend implements.
    fn capabilities(&self) -> HalCapabilities;

    /// Create a buffer. The returned handle is generational.
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    /// Upload `data` into `buffer` starting at byte `offset`.
    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Read `len` bytes from `buffer` starting at byte `offset`. The buffer must
    /// have been created with [`crate::BufferUsage::READBACK`].
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Destroy `buffer`, invalidating its handle. A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    // ── Texture API ───────────────────────────────────────────────────────────

    /// Allocate a GPU-resident texture. Data must be uploaded separately via
    /// [`Self::upload_texture_data`].
    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Upload `data` to the texture.  `data` must be tightly packed pixels in
    /// the format specified at creation (RGBA8 = 4 bytes per pixel).  Blocks
    /// until the upload is complete; an asynchronous path may be added later.
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Destroy a texture, invalidating its handle. A stale handle is a no-op.
    /// The caller must ensure no surface / pipeline is still referencing the
    /// underlying image (the same contract as `destroy_buffer`).
    fn destroy_texture(&self, texture: TextureHandle);

    /// Create a texture sampler.
    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    /// Destroy a sampler, invalidating its handle.
    fn destroy_sampler(&self, sampler: SamplerHandle);

    // ── Compute API ───────────────────────────────────────────────────────────

    /// Upload a SPIR-V shader module. Returns a handle usable for pipeline
    /// creation. Default impl returns `UnsupportedFeatures` on backends that do
    /// not support compute.
    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Destroy a shader module.
    fn destroy_shader(&self, _shader: ShaderHandle) {}

    /// Create a compute pipeline from a shader module + entry point.
    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Destroy a compute pipeline.
    fn destroy_pipeline(&self, _pipeline: PipelineHandle) {}

    /// Synchronously dispatch a compute pipeline. Blocks until GPU work
    /// completes. Asynchronous dispatch is deferred.
    ///
    /// `bindings.buffers` contains slot indices into the bindless SSBO
    /// table; the shader indexes them with `nonuniformEXT`. `bindings.scalars`
    /// become push-constant values (packed after the buffer indices). `grid` is
    /// the workgroup count along each axis.
    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }
}
