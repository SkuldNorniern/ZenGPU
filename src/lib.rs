//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! The public API of the internal crates, re-exported. Sub-crates are reachable as modules
//! ([`vulkan`], [`compute`], [`blas`], `cpu`) or as `zengpu-hal`, `zengpu-vulkan`, etc.
//!
//! # Cargo features
//!
//! - `vulkan` (default): [`VulkanInstance`], [`VulkanDevice`], swapchains, frame-graph.
//! - `compute` (default): [`DeviceArray`], [`BufferPool`].
//! - `blas` (default): [`GemmKernel`].
//! - `cpu`: [`CpuDevice`], for conformance tests, not production fallback.

pub mod log;

/// Backend-independent types, handles and errors. Always available.
pub use zengpu_hal as hal;

/// Compile GLSL and ZSL to SPIR-V at build time. Re-exports `zengpu_spirv`.
pub use zengpu_spirv as spirv;

/// GLSL or ZSL to SPIR-V at build time. See [`zengpu_spirv::zengpu_spirv`].
#[macro_export]
macro_rules! zengpu_spirv {
    ($($tt:tt)*) => {
        ::zengpu_spirv::zengpu_spirv!($($tt)*)
    };
}

pub use zengpu_hal::{
    Acquire, AdapterInfo, AdapterRequest, AddressMode, BackendPreference, Bindings, BlendMode,
    BufferDesc, BufferHandle, BufferUsage, ColorAttachment, ComputePipelineDesc, DType,
    DepthAttachment, DepthState, DeviceRequest, DeviceType, Features, FilterMode, Format, Frame,
    GpuAdapter, GpuDevice, GpuError, GpuInstance, GraphicsDevice, GraphicsPipelineDesc,
    HalCapabilities, LoadOp, MemoryUsage, PipelineHandle, PowerPreference, PresentMode,
    PrimitiveTopology, Rect, RenderCommands, RenderPassDesc, RenderTargetDesc, Result, SamplerDesc,
    SamplerHandle, Scalar, ShaderDesc, ShaderHandle, StepMode, Surface, SurfaceConfig,
    SurfaceError, TargetHandle, TextureDesc, TextureHandle, TextureUsage, UsageError,
    VertexAttribute, VertexFormat, VertexLayout, Viewport, ViewportScissor, WindowHandles,
};

/// Vulkan 1.2+ backend, graphics + compute.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{
    AttachmentUsage, BeginFrame, DEPTH_FORMAT, DepthTarget, DeviceContext, FrameGraph,
    OffscreenTarget, ResourceId, Swapchain, VulkanAdapter, VulkanCommandList, VulkanDevice,
    VulkanFrame, VulkanInstance, VulkanSurface,
};

/// CPU reference backend for conformance tests.
#[cfg(feature = "cpu")]
pub use zengpu_cpu as cpu;
#[cfg(feature = "cpu")]
pub use zengpu_cpu::{CpuAdapter, CpuDevice, CpuInstance};

/// Device arrays, pooled allocation, and elementwise kernels.
#[cfg(feature = "compute")]
pub use zengpu_compute as compute;
#[cfg(feature = "compute")]
pub use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};

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

/// Headless, compute only. For windowed rendering use [`open_vulkan_with_surface`].
#[cfg(feature = "vulkan")]
pub fn open_vulkan() -> Result<VulkanDevice> {
    VulkanInstance::new()?
        .request_vulkan_adapter()
        .ok_or_else(|| GpuError::Backend("no Vulkan adapter found".into()))?
        .open_headless(DeviceRequest::default())
}
