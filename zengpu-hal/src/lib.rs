//! ZenGPU HAL: the backend-independent foundation. No GPU and no backend dependencies.
//!
//! - [`error`]: structured, enum-based errors.
//! - [`handle`]: generational-index [`SlotMap`] and typed resource handles, with use-after-free
//!    detection.
//! - [`types`]: backend selection, memory/usage classes, feature flags, formats, dtypes.

#![forbid(unsafe_code)]

mod desc;
mod error;
mod handle;
mod types;

pub use desc::{
    AddressMode, BlendMode, BufferDesc, ComputePipelineDesc, DepthState, FilterMode,
    GraphicsPipelineDesc, PrimitiveTopology, RenderTargetDesc, SamplerDesc, ShaderDesc,
    SurfaceConfig, TextureDesc, VertexAttribute, VertexFormat, VertexLayout,
};
pub use error::{GpuError, Result, SurfaceError, UsageError};
pub use handle::{
    BufferHandle, Handle, PipelineHandle, SamplerHandle, ShaderHandle, SlotMap, SurfaceHandle,
    TargetHandle, TextureHandle, marker,
};
pub use types::{
    BackendPreference, BufferUsage, DType, Features, Format, MemoryUsage, PowerPreference,
    PresentMode, Rect, TextureUsage, Viewport,
};
