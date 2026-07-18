//! ZenGPU HAL — the backend-independent foundation.
//!
//! This crate holds the shared-core pieces that everything else builds on, with
//! no GPU and no backend dependencies:
//!
//! - [`GpuError`] — structured, enum-based errors.
//! - [`SlotMap`] and typed resource handles,
//!   which give the validation layer use-after-free detection.
//! - Backend selection, memory/usage classes, feature flags,
//!   formats, dtypes — all backend- and consumer-neutral.
//!
//! Split-HAL traits (graphics + compute) land here as the backends come online.

#![forbid(unsafe_code)]

mod adapter;
mod command;
mod desc;
mod device;
mod error;
mod graphics;
mod handle;
mod request;
mod submission;
mod surface;
mod types;

pub use adapter::{AdapterInfo, DeviceType, GpuAdapter, GpuInstance};
pub use command::{Bindings, BufferCopyOp, ComputeOp, DispatchOp, Scalar};
pub use device::GpuDevice;
pub use graphics::{
    Acquire, ColorAttachment, DepthAttachment, Frame, GraphicsDevice, LoadOp, RenderCommands,
    RenderPassDesc, Surface, ViewportScissor,
};
pub use request::{AdapterRequest, DeviceLimits, DeviceRequest, HalCapabilities};
pub use submission::{CompletedSubmission, GpuSubmission, Submission, SubmissionStatus};
pub use surface::WindowHandles;

pub use desc::{
    AddressMode, BlendMode, BorderColor, BufferDesc, CompareFn, ComputePipelineDesc, CullMode,
    DepthState, FilterMode, FrontFace, GraphicsPipelineDesc, PolygonMode, PrimitiveTopology,
    RasterState, RenderTargetDesc, SamplerDesc, ShaderDesc, ShaderSource, StepMode, SurfaceConfig,
    TexDim, TextureDesc, VertexAttribute, VertexFormat, VertexLayout,
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
