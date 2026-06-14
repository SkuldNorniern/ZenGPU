//! The device trait: backend-independent handle onto a GPU (or the CPU reference).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn GpuDevice>` and
//! selected at runtime.

use crate::desc::BufferDesc;
use crate::error::Result;
use crate::handle::BufferHandle;
use crate::request::HalCapabilities;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker threads can allocate
/// and upload concurrently.
pub trait GpuDevice: Send + Sync {
    fn capabilities(&self) -> HalCapabilities;

    /// Returned handle is generational.
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Buffer must be created with [`crate::BufferUsage::READBACK`].
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);
}
