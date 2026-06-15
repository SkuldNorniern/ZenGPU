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
    AdapterInfo, AdapterRequest, AddressMode, Acquire, BackendPreference, Bindings, BlendMode,
    BufferDesc, BufferHandle, BufferUsage, ColorAttachment, ComputePipelineDesc, DType,
    DepthAttachment, DepthState, DeviceRequest, DeviceType, Features, FilterMode, Format, Frame,
    GpuAdapter, GpuDevice, GpuError, GpuInstance, GraphicsDevice, GraphicsPipelineDesc,
    HalCapabilities, LoadOp, MemoryUsage, PipelineHandle, PowerPreference, PresentMode,
    PrimitiveTopology, Rect, RenderCommands, RenderPassDesc, RenderTargetDesc, Result,
    SamplerDesc, SamplerHandle, Scalar, ShaderDesc, ShaderHandle, Surface, SurfaceConfig,
    SurfaceError, TargetHandle, TextureDesc, TextureHandle, TextureUsage, UsageError,
    VertexAttribute, VertexFormat, VertexLayout, Viewport, ViewportScissor, WindowHandles,
};

/// Vulkan 1.2+ backend, graphics + compute.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{
    AttachmentUsage, BeginFrame, DepthTarget, DeviceContext, FrameGraph, OffscreenTarget,
    ResourceId, SampledImageView, Swapchain, VulkanAdapter, VulkanCommandList, VulkanDevice,
    VulkanFrame, VulkanInstance, VulkanSurface, DEPTH_FORMAT,
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

/// Vulkan device with surface/swapchain support, ready for [`Swapchain::new`].
#[cfg(feature = "vulkan")]
pub fn open_vulkan_with_surface() -> Result<VulkanDevice> {
    VulkanInstance::new_with_surface()?
        .request_vulkan_adapter()
        .ok_or_else(|| GpuError::Backend("no Vulkan adapter found".into()))?
        .open_with_surface(DeviceRequest::default())
}

/// Headless, compute only. For swapchains and presents use [`open_vulkan_with_surface`].
#[cfg(feature = "vulkan")]
pub fn open_vulkan() -> Result<Box<dyn GpuDevice>> {
    VulkanInstance::new()?
        .request_adapter(AdapterRequest::default())
        .ok_or_else(|| GpuError::Backend("no Vulkan adapter found".into()))?
        .open(DeviceRequest::default())
}
