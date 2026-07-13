//! ZenGPU shader macro — compile native ZSL to all backends at build time.
//!
//! Use [`zsl!`] to produce a [`ZslShader`] containing SPIR-V (Vulkan), HIP C++
//! (ROCm), MSL (Metal), and CUDA C++ (NVIDIA), all compiled at build time from
//! the same ZSL source. At runtime, select the right form with
//! [`ZslShader::for_backend`].
//!
//! ```ignore
//! use zengpu_spirv::{zsl, ZslShader};
//!
//! const KERNEL: ZslShader = zsl!(
//!     push P { n: u32, scale: f32 }
//!     @workgroup_size(64)
//!     kernel scale(src: device buffer<f32>, dst: device mut buffer<f32>, p: P, id: global_id) {
//!         let i = id.x
//!         if i < p.n { dst[i] = src[i] * p.scale }
//!     }
//! );
//! ```

use std::slice;

use zengpu_hal::{BackendPreference, ShaderDesc};

/// All-backend compiled form of a ZSL shader.
///
/// Produced by [`zsl!`] at build time. Each field holds the compiled output for
/// one target; use [`for_backend`](ZslShader::for_backend) to pick the right one
/// at runtime.
pub struct ZslShader {
    /// SPIR-V words for Vulkan (and any other SPIR-V consumer).
    pub spv: &'static [u32],
    /// HIP C++ source for AMD ROCm (compiled at runtime via hipRTC).
    pub hip: &'static str,
    /// Metal Shading Language source for Apple Metal.
    pub msl: &'static str,
    /// CUDA C++ source for NVIDIA CUDA (compiled at runtime via NVRTC).
    pub cuda: &'static str,
}

impl ZslShader {
    /// Return the right [`ShaderDesc`] and entry-point name for `backend`.
    ///
    /// Falls back to SPIR-V for any backend that is not HIP, Metal, or CUDA.
    pub fn for_backend(&self, backend: BackendPreference) -> (ShaderDesc<'_>, &'static str) {
        match backend {
            BackendPreference::Hip => (ShaderDesc::hip(self.hip), "zsl_kernel"),
            BackendPreference::Metal => (ShaderDesc::msl(self.msl), "zsl_main"),
            BackendPreference::Cuda => (ShaderDesc::cuda_src(self.cuda), "zsl_kernel"),
            _ => (self.spirv_desc(), "main"),
        }
    }

    /// Convenience: SPIR-V [`ShaderDesc`] for Vulkan-only callers.
    pub fn spirv_desc(&self) -> ShaderDesc<'_> {
        let bytes =
            unsafe { slice::from_raw_parts(self.spv.as_ptr() as *const u8, self.spv.len() * 4) };
        ShaderDesc::spirv(bytes)
    }
}

/// Compile **native ZSL** source to a [`ZslShader`] — all backends built at
/// compile time. See the crate-level docs for usage.
pub use zengpu_zsl::zsl;

/// A column-major 4×4 float matrix for use in `#[derive(ZslPushConst)]` structs.
///
/// Wrap any `mat4` value (glam, nalgebra, hand-rolled) as `ZslMat4(m.to_cols_array())`
/// before passing it to `to_scalars()`. The 16 floats are laid out column-major,
/// matching SPIR-V's `ColMajor` decoration.
#[derive(Debug, Clone, Copy)]
pub struct ZslMat4(pub [f32; 16]);

/// Private re-exports referenced by code emitted by `#[derive(ZslPushConst)]`.
/// Semver-exempt; do not use directly.
#[doc(hidden)]
pub mod _zsl_priv {
    pub use super::ZslMat4;
    pub use zengpu_hal::Scalar;
}

/// Derive `to_scalars()` for a push-constant struct.
///
/// Fields may be `u32`, `i32`, `f32`, or [`ZslMat4`]. The generated method
/// returns a fixed-size array of [`zengpu_hal::Scalar`] in field order,
/// ready for `Bindings::scalars` in a dispatch call. `ZslMat4` expands to
/// 16 consecutive `Scalar::F32` entries in column-major order.
///
/// # Example
///
/// ```ignore
/// use zengpu_spirv::{ZslPushConst, ZslMat4};
///
/// #[derive(ZslPushConst)]
/// struct ScalePush {
///     len: u32,
///     scale: f32,
/// }
///
/// let push = ScalePush { len: 1024, scale: 2.0 };
/// device.dispatch(pipeline, Bindings {
///     scalars: &push.to_scalars(),
///     ..Default::default()
/// }, [16, 1, 1])?;
/// ```
pub use zengpu_zsl::ZslPushConst;
