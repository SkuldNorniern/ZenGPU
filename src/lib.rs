//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! # Quick start
//!
//! ```no_run
//! use zengpu::{Instance, AdapterRequest, BufferDesc, BufferUsage, MemoryUsage};
//!
//! // Explicitly opt in to the backends you need.
//! let instance = Instance::builder()
//!     .vulkan_with_surface()          // Err if Vulkan loader absent
//!     .expect("Vulkan unavailable")
//!     .build();
//!
//! let adapter = instance
//!     .request_adapter(AdapterRequest::default())
//!     .expect("no suitable GPU found");
//!
//! let device = adapter.open_default().expect("device creation failed");
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
pub mod detect;
pub mod device;
pub mod instance;
pub mod log;

pub use adapter::Adapter;
pub use detect::{BackendAvailability, detect_backends};
pub use device::Device;
pub use instance::Instance;

// ── HAL foundation (always available) ────────────────────────────────────────

/// Backend-independent HAL types and traits.
pub use zengpu_hal as hal;

/// ZSL shader macros and the [`ZslShader`] type.
pub use zengpu_spirv as spirv;

/// Compile native ZSL source to all GPU backends at build time.
/// See [`zengpu_spirv::zsl`] for the syntax.
pub use zengpu_spirv::zsl;

/// All-backend compiled form of a ZSL shader — produced by [`zsl!`].
pub use zengpu_spirv::ZslShader;

// Flat re-exports of the shared vocabulary so callers write `zengpu::BufferHandle`
// rather than `zengpu::hal::BufferHandle`.
pub use zengpu_hal::{
    Acquire, AdapterInfo, AdapterRequest, AddressMode, BackendPreference, Bindings, BlendMode,
    BufferCopyOp, BufferDesc, BufferHandle, BufferUsage, ColorAttachment, ComputeOp,
    ComputePipelineDesc, DType, DepthAttachment, DepthState, DeviceLimits, DeviceRequest,
    DeviceType, DispatchOp, Features, FilterMode, Format, Frame, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuSubmission, GraphicsDevice, GraphicsPipelineDesc, HalCapabilities, LoadOp,
    MemoryUsage, PipelineHandle, PowerPreference, PresentMode, PrimitiveTopology, Rect,
    RenderCommands, RenderPassDesc, RenderTargetDesc, Result, SamplerDesc, SamplerHandle, Scalar,
    ShaderDesc, ShaderHandle, StepMode, Submission, SubmissionStatus, Surface, SurfaceConfig,
    SurfaceError, TargetHandle, TextureDesc, TextureHandle, TextureUsage, UsageError,
    VertexAttribute, VertexFormat, VertexLayout, Viewport, ViewportScissor, WindowHandles,
};

// ── Flat re-exports for the most commonly used extension types ────────────────
// These let callers write `zengpu::BufferPool` instead of
// `zengpu::compute::BufferPool`, matching the pattern for HAL types above.

/// Flat re-export: `DeviceArray`, `BufferPool`, and `ElementwiseKernels` from
/// the compute extension.
#[cfg(feature = "compute")]
pub use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};

/// Flat re-export: `GemmKernel` from the BLAS extension.
#[cfg(feature = "blas")]
pub use zengpu_blas::GemmKernel;

/// Flat re-export: concrete Vulkan types required by windowed / backend-aware
/// code. For most use cases the [`Instance`] builder is sufficient.
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{BeginFrame, DeviceContext, Swapchain, VulkanDevice, VulkanInstance};

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

/// Apple Metal backend (macOS/iOS, graphics + compute).
#[cfg(feature = "metal")]
pub use zengpu_metal as metal;

/// AMD ROCm/HIP compute backend.
#[cfg(feature = "hip")]
pub use zengpu_hip as hip;

/// DirectX 12 backend (Windows, graphics + compute).
#[cfg(feature = "dx12")]
pub use zengpu_dx12 as dx12;
