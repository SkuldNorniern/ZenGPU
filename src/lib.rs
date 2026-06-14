//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! Re-exports the public API of the internal crates. Foundation types live in [`hal`].

/// Backend-independent types, handles and errors.
pub use zengpu_hal as hal;

pub use zengpu_hal::{
    BackendPreference, BufferHandle, BufferUsage, DType, Features, Format, GpuError, MemoryUsage,
    PipelineHandle, PowerPreference, PresentMode, Result, SurfaceError, TextureHandle, UsageError,
};
