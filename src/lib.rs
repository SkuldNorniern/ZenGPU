//! ZenGPU — a native-first GPU runtime for Rust (graphics + general compute).
//!
//! This is the main crate; it re-exports the public API of the internal
//! crates so most consumers can depend on just `zengpu` and write
//! `zengpu::VulkanInstance`, `zengpu::DeviceArray`, etc. For detailed work
//! (e.g. backend-internal types), the sub-crates remain available wholesale
//! under their module names ([`vulkan`], [`compute`], [`blas`], [`cpu`]), or
//! directly as `zengpu-hal`, `zengpu-vulkan`, etc.
//!
//! # Cargo features
//!
//! - `vulkan` (default) — the Vulkan backend ([`vulkan`]): [`VulkanInstance`],
//!   [`VulkanDevice`], swapchains, frame-graph.
//! - `compute` (default) — device arrays, pooled allocation, and elementwise
//!   kernels ([`compute`]): [`DeviceArray`], [`BufferPool`].
//! - `blas` (default) — GEMM compute kernel on top of `compute` ([`blas`]):
//!   [`GemmKernel`].
//! - `cpu` — the CPU reference backend ([`cpu`]): [`CpuDevice`]. This is the
//!   conformance oracle, not a product fallback — most consumers don't need it.

/// The backend-independent foundation: types, handles, and errors (always
/// available).
pub use zengpu_hal as hal;

// Flat re-exports of the most-used foundation items, so downstream code can
// write `zengpu::BufferHandle` rather than `zengpu::hal::BufferHandle`.
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

/// The Vulkan backend — graphics + compute on Vulkan 1.2+.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{
    AttachmentUsage, BeginFrame, CircleInstance, DEPTH_FORMAT, DepthTarget, DeviceContext, DrawRef,
    Frame2d, FrameGraph, GradientInstance, IMAGE_SLOTS, ImageInstance, OffscreenTarget,
    RectInstance, ResourceId, SampledImageView, Swapchain, TextInstance, Vulkan2dSurface,
    VulkanAdapter, VulkanDevice, VulkanInstance,
};

/// The CPU reference backend — the conformance oracle.
#[cfg(feature = "cpu")]
pub use zengpu_cpu as cpu;
#[cfg(feature = "cpu")]
pub use zengpu_cpu::{CpuAdapter, CpuDevice, CpuInstance};

/// Device arrays, pooled allocation, and elementwise kernels.
#[cfg(feature = "compute")]
pub use zengpu_compute as compute;
#[cfg(feature = "compute")]
pub use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};

/// GEMM compute kernel — the portable matmul fallback.
#[cfg(feature = "blas")]
pub use zengpu_blas as blas;
#[cfg(feature = "blas")]
pub use zengpu_blas::GemmKernel;
