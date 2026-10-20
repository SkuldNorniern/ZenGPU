//! ZenGPU HAL: backend-independent core with no GPU and no backend dependencies.
//! - [`GpuError`]: structured errors.
//! - [`SlotMap`] and typed handles: use-after-free detection.
//! - Backend selection, memory and usage classes, feature flags, formats, dtypes.

#![forbid(unsafe_code)]

mod adapter;
mod caps;
mod command;
mod desc;
mod device;
mod error;
mod graphics;
mod handle;
mod pool;
mod request;
mod submission;
mod surface;
mod types;

pub use adapter::{AdapterInfo, DeviceType, GpuAdapter, GpuInstance};
pub use caps::{ComputeCaps, ElemType, MmaShape};
pub use command::{Bindings, BufferCopyOp, ComputeOp, DispatchOp, Scalar};
pub use device::GpuDevice;
pub use graphics::{
    Acquire, ColorAttachment, DepthAttachment, Frame, GraphicsDevice, LoadOp, RenderCommands,
    RenderPassDesc, Surface, ViewportScissor,
};
pub use pool::TexturePool;
pub use request::{
    AdapterRequest, DeviceLimits, DeviceRequest, FormatCaps, HalCapabilities, MemoryBudget,
    TransferCaps,
};
pub use submission::{
    CompletedReadback, CompletedSubmission, GpuReadback, GpuSubmission, ReadSubmission, Submission,
    SubmissionStatus,
};
pub use surface::WindowHandles;

pub use desc::{
    AddressMode, BlendComponent, BlendFactor, BlendMode, BlendOp, BlendState, BorderColor,
    BufferDesc, ColorTargetState, CompareFn, ComputePipelineDesc, ComputePipelineHints, CullMode,
    DepthState, FilterMode, FrontFace, GraphicsPipelineDesc, PolygonMode, PrimitiveTopology,
    RasterState, RenderTargetDesc, SamplerDesc, ShaderDesc, ShaderSource, StencilFaceState,
    StencilOp, StencilState, StepMode, SurfaceConfig, TexDim, TextureDesc, TextureRegion,
    TextureWrite, VertexAttribute, VertexFormat, VertexLayout,
};
pub use error::{GpuError, Result, SurfaceError, UsageError};
pub use handle::{
    BufferHandle, Handle, PipelineHandle, QueryPoolHandle, SamplerHandle, ShaderHandle, SlotMap,
    SurfaceHandle, TargetHandle, TextureHandle, marker,
};
pub use types::{
    BackendPreference, BufferUsage, DType, Features, Format, MemoryUsage, PowerPreference,
    PresentMode, Rect, TextureUsage, Viewport,
};
