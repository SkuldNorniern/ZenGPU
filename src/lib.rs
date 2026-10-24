//! ZenGPU: a native-first GPU runtime for Rust (graphics + general compute).
//!
//! # Cargo features
//!
//! | Feature   | Default | Description |
//! |-----------|---------|-------------|
//! | `vulkan`  | yes     | Vulkan 1.2+ backend (graphics + compute). |
//! | `metal`   | yes     | Metal backend (macOS, graphics + compute); finds nothing elsewhere. |
//! | `compute` | yes     | Device arrays, pooled allocation, elementwise kernels. |
//! | `cuda`    | no      | CUDA Driver API backend (NVIDIA, compute only). |
//! | `cpu`     | no      | CPU reference backend for conformance tests. |
//! | `hip`     | no      | HIP/ROCm backend (AMD, compute only). |
//! | `dx12`    | no      | DirectX 12 backend (Windows), not yet implemented. |
//!
//! # Backends
//!
//! [`native`] is Metal on macOS, Vulkan elsewhere. One `zsl!` shader serves both
//! through [`native::shader`].
//! Use [`Device::as_vulkan`] for swapchain and frame-graph work, and `Device::as_cpu` for the
//! CPU conformance oracle.
//! Sub-crates are re-exported under [`vulkan`], [`compute`], `cpu`, `cuda`,
//! [`metal`], `hip`, `dx12`.
//!
//! ```no_run
//! use zengpu::{Instance, AdapterRequest, BufferDesc, BufferUsage, MemoryUsage};
//!
//! # #[cfg(feature = "vulkan")] {
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
//! # }
//! ```

pub mod adapter;
pub mod detect;
pub mod device;
pub mod instance;
pub mod log;
#[cfg(any(
    feature = "vulkan",
    all(any(target_os = "macos", target_os = "ios"), feature = "metal")
))]
pub mod native;

pub use adapter::Adapter;
pub use detect::{BackendAvailability, detect_backends};
pub use device::Device;
pub use instance::Instance;

pub use zengpu_hal as hal;

pub use zengpu_spirv as spirv;

/// `zsl!`: compile ZSL to all backends at build time. Syntax: [`zengpu_spirv::zsl`].
pub use zengpu_spirv::zsl;

/// Compiled shader for all backends, from [`zsl!`].
pub use zengpu_spirv::ZslShader;

pub use zengpu_hal::{
    Acquire, AdapterInfo, AdapterRequest, AddressMode, BackendPreference, Bindings, BlendComponent,
    BlendFactor, BlendMode, BlendOp, BlendState, BufferCopyOp, BufferDesc, BufferHandle,
    BufferUsage, ColorAttachment, ColorTargetState, ComputeOp, ComputePipelineDesc, DType,
    DepthAttachment, DepthState, DeviceLimits, DeviceRequest, DeviceType, DispatchOp, Features,
    FilterMode, Format, Frame, GpuAdapter, GpuDevice, GpuError, GpuInstance, GpuSubmission,
    GraphicsDevice, GraphicsPipelineDesc, HalCapabilities, LoadOp, MemoryUsage, PipelineHandle,
    PowerPreference, PresentMode, PrimitiveTopology, Rect, RenderCommands, RenderPassDesc,
    RenderTargetDesc, Result, SamplerDesc, SamplerHandle, Scalar, ShaderDesc, ShaderHandle,
    StencilFaceState, StencilOp, StencilState, StepMode, Submission, SubmissionStatus, Surface,
    SurfaceConfig, SurfaceError, TargetHandle, TextureDesc, TextureHandle, TextureRegion,
    TextureUsage, TextureWrite, TransferCaps, UsageError, VertexAttribute, VertexFormat,
    VertexLayout, Viewport, ViewportScissor, WindowHandles,
};

#[cfg(feature = "compute")]
pub use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};

/// Concrete Vulkan types for windowed, backend-aware code. Most code needs only [`Instance`].
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan::{BeginFrame, DeviceContext, Swapchain, VulkanDevice, VulkanInstance};

/// Vulkan backend (1.2+, graphics + compute). Most code: use [`Device::as_vulkan`].
#[cfg(feature = "vulkan")]
pub use zengpu_vulkan as vulkan;

/// CPU reference backend for conformance tests.
#[cfg(feature = "cpu")]
pub use zengpu_cpu as cpu;

/// Device arrays, pooled allocation, and elementwise kernels.
#[cfg(feature = "compute")]
pub use zengpu_compute as compute;

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
