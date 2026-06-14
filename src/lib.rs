//! ZenGPU — a native-first GPU runtime for Rust (graphics + general compute).
//!
//! This is the main crate; it re-exports the stable public API from the
//! internal crates. See the project plan for the architecture and the numbered
//! decision record.
//!
//! At this stage only the shared-core foundation ([`hal`]) exists; the graphics
//! and compute runtimes are being built against it (see `status.md`).

/// The backend-independent foundation: types, handles, and errors.
pub use zengpu_hal as hal;

// Flat re-exports of the most-used foundation items, so downstream code can
// write `zengpu::BufferHandle` rather than `zengpu::hal::BufferHandle`.
pub use zengpu_hal::{
    BackendPreference, BufferHandle, BufferUsage, DType, Features, Format, GpuError, MemoryUsage,
    PipelineHandle, PowerPreference, PresentMode, Result, SurfaceError, TextureHandle, UsageError,
};
