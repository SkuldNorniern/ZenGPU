//! ZenGPU HAL — the backend-independent foundation.
//!
//! This crate holds the shared-core pieces that everything else builds on, with
//! no GPU and no backend dependencies:
//!
//! - [`error`] — structured, enum-based errors.
//! - [`handle`] — generational-index [`SlotMap`] and typed resource handles,
//!   which give the validation layer use-after-free detection.
//! - [`types`] — backend selection, memory/usage classes, feature flags,
//!   formats, dtypes — all backend- and consumer-neutral.
//!
//! Split-HAL traits (graphics + compute) land here as the backends come online.

#![forbid(unsafe_code)]

mod adapter;
mod command;
mod desc;
mod device;
mod error;
mod handle;
mod request;
mod surface;
mod types;

pub use adapter::{AdapterInfo, DeviceType, GpuAdapter, GpuInstance};
pub use command::{Bindings, Scalar};
pub use device::GpuDevice;
pub use request::{AdapterRequest, DeviceRequest, HalCapabilities};
pub use surface::WindowHandles;

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
