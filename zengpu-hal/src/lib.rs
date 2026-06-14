//! ZenGPU HAL — the backend-independent foundation.
//!
//! This crate holds the shared-core pieces that everything else builds on, with
//! no GPU and no backend dependencies:
//!
//! - [`error`] — structured, enum-based errors (plan §21).
//! - [`handle`] — generational-index [`SlotMap`] and typed resource handles,
//!   which give the validation layer use-after-free detection (plan §5 / D3).
//! - [`types`] — backend selection, memory/usage classes, feature flags,
//!   formats, dtypes (plan §7, §22) — all backend- and consumer-neutral (D10).
//!
//! Split-HAL traits (graphics + compute) land here as the backends come online.

#![forbid(unsafe_code)]

mod error;
mod handle;
mod types;

pub use error::{GpuError, Result, SurfaceError, UsageError};
pub use handle::{
    BufferHandle, Handle, PipelineHandle, SamplerHandle, ShaderHandle, SlotMap, SurfaceHandle,
    TargetHandle, TextureHandle, marker,
};
pub use types::{
    BackendPreference, BufferUsage, DType, Features, Format, MemoryUsage, PowerPreference,
    PresentMode,
};
