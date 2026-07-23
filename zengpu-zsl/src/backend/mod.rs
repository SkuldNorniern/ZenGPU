//! ZSL lowering backends — each turns the front-end's parsed entry point into a
//! concrete target. SPIR-V (Vulkan) and MSL (Metal) are both IR consumers; a
//! backend is a consumer of the shared front-end, not the language model.

// MSL backend: consumed by the Metal HAL backend; allow unused until then.
#[cfg(feature = "metal")]
#[allow(dead_code, unused_imports)]
pub mod msl;

#[cfg(feature = "spirv")]
pub mod spirv;

// HIP C++ backend: consumed by zengpu-hip's hipRTC path via the `zsl_hip!` macro.
#[cfg(feature = "hip")]
#[allow(dead_code, unused_imports)]
pub mod hip;

// CUDA C++ backend: consumed by zengpu-cuda's NVRTC path via the `zsl_cuda!` macro.
#[cfg(feature = "cuda")]
#[allow(dead_code, unused_imports)]
pub mod cuda;
