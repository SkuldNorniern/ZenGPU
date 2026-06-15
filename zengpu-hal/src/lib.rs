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
