//! ZenGPU HAL: backend-independent core with no GPU and no backend dependencies.
//! - [`GpuError`]: structured errors.
//! - [`SlotMap`] and typed handles: use-after-free detection.
//! - Backend selection, memory and usage classes, feature flags, formats, dtypes.

#![forbid(unsafe_code)]

mod adapter;
mod command;
mod desc;
mod device;
mod error;
mod graphics;
mod handle;
mod request;
mod surface;
mod types;

pub use adapter::{AdapterInfo, DeviceType, GpuAdapter, GpuInstance};
pub use command::{Bindings, DispatchOp, Scalar};
pub use device::GpuDevice;
pub use graphics::{
    Acquire, ColorAttachment, DepthAttachment, Frame, GraphicsDevice, LoadOp, RenderCommands,
    RenderPassDesc, Surface, ViewportScissor,
};
pub use request::{AdapterRequest, DeviceLimits, DeviceRequest, HalCapabilities};
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
