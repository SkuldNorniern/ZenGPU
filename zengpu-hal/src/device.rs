//! The device trait — the backend-independent handle onto a GPU (or the CPU
//! reference). This is the first slice of the split-HAL (plan §20); it grows as
//! backends come online (compute dispatch, graphics passes, surfaces).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn
//! GpuDevice>` and selected at runtime.

use crate::desc::BufferDesc;
use crate::error::Result;
use crate::handle::BufferHandle;
use crate::request::HalCapabilities;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker
/// threads can allocate and upload concurrently (plan D5).
pub trait GpuDevice: Send + Sync {
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
}
