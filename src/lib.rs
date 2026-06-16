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

/// Shader macro — compile GLSL (and later ZSL/WGSL) to SPIR-V at build time.
/// Re-exports `zengpu_spirv` crate for sub-crate access.
pub use zengpu_spirv as spirv;

/// Compile shader source to SPIR-V at build time. GLSL in step 1; ZSL and
/// WGSL support will be added transparently in later steps.
#[macro_export]
macro_rules! zengpu_spirv {
    ($($tt:tt)*) => {
        ::zengpu_spirv::zengpu_spirv!($($tt)*)
    };
}

// Flat re-exports of the most-used foundation items, so downstream code can
// write `zengpu::BufferHandle` rather than `zengpu::hal::BufferHandle`.
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

/// The Vulkan backend — graphics + compute on Vulkan 1.2+.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{
    AttachmentUsage, BeginFrame, DEPTH_FORMAT, DepthTarget, DeviceContext, FrameGraph,
    OffscreenTarget, ResourceId, Swapchain, VulkanAdapter, VulkanCommandList, VulkanDevice,
    VulkanFrame, VulkanInstance, VulkanSurface,
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

// ── Convenience entry points ──────────────────────────────────────────────────
//
// These bridge instance + adapter selection + device open into single calls so
// consumers never need to know which subcrate drives each step.

/// Open the best available Vulkan device with surface/swapchain support.
///
/// Replaces the three-line `VulkanInstance::new_with_surface()` +
/// `request_vulkan_adapter()` + `open_with_surface()` pattern. Returns a
/// [`VulkanDevice`] ready for [`Swapchain::new`].
#[cfg(feature = "vulkan")]
pub fn open_vulkan_with_surface() -> Result<VulkanDevice> {
    VulkanInstance::new_with_surface()?
        .request_vulkan_adapter()
        .ok_or_else(|| GpuError::Backend("no Vulkan adapter found".into()))?
        .open_with_surface(DeviceRequest::default())
}

/// Open the best available Vulkan device without surface support (headless /
/// compute). For windowed rendering use [`open_vulkan_with_surface`].
#[cfg(feature = "vulkan")]
pub fn open_vulkan() -> Result<VulkanDevice> {
    VulkanInstance::new()?
        .request_vulkan_adapter()
        .ok_or_else(|| GpuError::Backend("no Vulkan adapter found".into()))?
        .open_headless(DeviceRequest::default())
}
