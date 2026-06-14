//! The device trait — the backend-independent handle onto a GPU (or the CPU
//! reference). This is the first slice of the split-HAL (plan §20); it grows as
//! backends come online (compute dispatch, graphics passes, surfaces).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn
//! GpuDevice>` and selected at runtime.

use crate::desc::{BufferDesc, SamplerDesc, TextureDesc};
use crate::error::Result;
use crate::handle::{BufferHandle, SamplerHandle, TextureHandle};
use crate::request::HalCapabilities;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker
/// threads can allocate and upload concurrently (plan D5).
pub trait GpuDevice: Send + Sync {
    /// Downcast to the concrete backend type. Required for backend-specific
    /// operations (e.g. creating a Vulkan swapchain from a VulkanDevice).
    fn as_any(&self) -> &dyn core::any::Any;

    /// Which HAL shapes this backend implements (plan §4 / D1).
    fn capabilities(&self) -> HalCapabilities;

    /// Create a buffer. The returned handle is generational (plan §5 / D3).
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    /// Upload `data` into `buffer` starting at byte `offset`.
    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Read `len` bytes from `buffer` starting at byte `offset`. The buffer must
    /// have been created with [`crate::BufferUsage::READBACK`] (plan §9).
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Destroy `buffer`, invalidating its handle. A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    // ── Texture API (plan G3) ─────────────────────────────────────────────────

    /// Allocate a GPU-resident texture. Data must be uploaded separately via
    /// [`Self::upload_texture_data`].
    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Upload `data` to the texture.  `data` must be tightly packed pixels in
    /// the format specified at creation (RGBA8 = 4 bytes per pixel).  Blocks
    /// until the upload is complete (G3 scope; async path is plan D6).
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Destroy a texture, invalidating its handle. A stale handle is a no-op.
    /// The caller must ensure no surface / pipeline is still referencing the
    /// underlying image (the same contract as `destroy_buffer`).
    fn destroy_texture(&self, texture: TextureHandle);

    /// Create a texture sampler.
    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    /// Destroy a sampler, invalidating its handle.
    fn destroy_sampler(&self, sampler: SamplerHandle);
}
