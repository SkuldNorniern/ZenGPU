//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! # Quick start
//!
//! ```no_run
//! use zengpu::{Instance, AdapterRequest, DeviceRequest, BufferDesc, BufferUsage, MemoryUsage};
//!
//! let instance = Instance::new();
//! let adapter  = instance.request_adapter(AdapterRequest::default()).expect("no GPU found");
//! let device   = adapter.open(DeviceRequest::default()).expect("device creation failed");
//!
//! let buf = device.create_buffer(BufferDesc {
//!     size:   1024,
//!     usage:  BufferUsage::STORAGE,
//!     memory: MemoryUsage::CpuToGpu,
//! }).unwrap();
//! device.write_buffer(buf, 0, &[0u8; 1024]).unwrap();
//! ```
//!
//! # Cargo features
//!
//! | Feature   | Default | Description |
//! |-----------|---------|-------------|
//! | `vulkan`  | yes     | Vulkan 1.2+ backend (graphics + compute). |
//! | `compute` | yes     | Device arrays, pooled allocation, elementwise kernels. |
//! | `blas`    | yes     | GEMM kernel on top of `compute`. |
//! | `cuda`    | no      | CUDA Driver API backend (NVIDIA, compute only). |
//! | `cpu`     | no      | CPU reference backend for conformance tests. |
//!
//! # Backend-specific access
//!
//! [`Device::as_vulkan`] returns `&VulkanDevice` for swapchain / frame-graph
//! work; [`Device::as_cpu`] gives access to the CPU conformance oracle.
//! The sub-crates are re-exported under [`vulkan`], [`compute`], [`blas`],
//! [`cpu`], [`cuda`] for power users who need their full surface.

pub mod adapter;
pub mod device;
pub mod instance;
pub mod log;

pub use adapter::Adapter;
pub use device::Device;
pub use instance::Instance;

// ── HAL foundation (always available) ────────────────────────────────────────

/// Backend-independent HAL types and traits.
pub use zengpu_hal as hal;

/// Compile GLSL / ZSL to SPIR-V at build time.
pub use zengpu_spirv as spirv;

/// Compile GLSL or ZSL source to SPIR-V at build time.
/// See [`zengpu_spirv::zengpu_spirv`] for full documentation.
#[macro_export]
macro_rules! zengpu_spirv {
    ($($tt:tt)*) => {
        ::zengpu_spirv::zengpu_spirv!($($tt)*)
    };
}

// Flat re-exports of the shared vocabulary so callers write `zengpu::BufferHandle`
// rather than `zengpu::hal::BufferHandle`.
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

// ── Backend sub-crates (power-user access) ───────────────────────────────────

/// Vulkan backend: graphics + compute on Vulkan 1.2+.
///
/// Access concrete types here when you need Vulkan-specific APIs
/// (swapchains, frame-graph, render passes). For most use cases
/// [`Device::as_vulkan`] is sufficient.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;

/// CPU reference backend for conformance tests.
#[cfg(feature = "cpu")]
pub use zengpu_cpu as cpu;

/// Device arrays, pooled allocation, and elementwise kernels.
#[cfg(feature = "compute")]
pub use zengpu_compute as compute;

/// GEMM compute kernel: the portable matmul fallback.
#[cfg(feature = "blas")]
pub use zengpu_blas as blas;

/// CUDA Driver API compute backend (NVIDIA GPUs, compute HAL only).
#[cfg(feature = "cuda")]
pub use zengpu_cuda as cuda;
