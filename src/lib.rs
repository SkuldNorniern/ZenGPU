//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! Re-exports the public API of the internal crates, so most consumers only need `zengpu`.
//! Sub-crates are reachable under their module names ([`vulkan`], [`compute`], [`blas`], `cpu`)
//! or as `zengpu-hal`, `zengpu-vulkan`, etc.
//!
//! # Cargo features
//!
//! - `vulkan` (default): [`VulkanInstance`], [`VulkanDevice`], swapchains, frame-graph.
//! - `compute` (default): [`DeviceArray`], [`BufferPool`].
//! - `blas` (default): [`GemmKernel`].
//! - `cpu`: [`CpuDevice`]. Conformance oracle, not a product fallback.

/// Backend-independent types, handles and errors. Always available.
pub use zengpu_hal as hal;

pub use zengpu_hal::{
    AdapterInfo, AdapterRequest, AddressMode, BackendPreference, Bindings, BlendMode, BufferDesc,
    BufferHandle, BufferUsage, ComputePipelineDesc, DType, DepthState, DeviceRequest, DeviceType,
    Features, FilterMode, Format, GpuAdapter, GpuDevice, GpuError, GpuInstance,
    GraphicsPipelineDesc, HalCapabilities, MemoryUsage, PipelineHandle, PowerPreference,
    PresentMode, PrimitiveTopology, Rect, RenderTargetDesc, Result, SamplerDesc, SamplerHandle,
    Scalar, ShaderDesc, ShaderHandle, SurfaceConfig, SurfaceError, TargetHandle, TextureDesc,
    TextureHandle, TextureUsage, UsageError, VertexAttribute, VertexFormat, VertexLayout, Viewport,
    WindowHandles,
};

/// Vulkan 1.2+ backend, graphics + compute.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{
    AttachmentUsage, BeginFrame, DepthTarget, DeviceContext, FrameGraph, OffscreenTarget,
    ResourceId, SampledImageView, Swapchain, VulkanAdapter, VulkanDevice, VulkanInstance,
    DEPTH_FORMAT,
};

/// CPU reference backend. Conformance oracle.
#[cfg(feature = "cpu")]
pub use zengpu_cpu as cpu;
#[cfg(feature = "cpu")]
pub use zengpu_cpu::{CpuAdapter, CpuDevice, CpuInstance};

/// Device arrays, pooled allocation, and elementwise kernels.
#[cfg(feature = "compute")]
pub use zengpu_compute as compute;
#[cfg(feature = "compute")]
pub use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};

/// Portable GEMM fallback.
#[cfg(feature = "blas")]
pub use zengpu_blas as blas;
#[cfg(feature = "blas")]
pub use zengpu_blas::GemmKernel;
