//! Shader macro that compiles native ZSL to SPIR-V, HIP C++, MSL and CUDA C++ at build time.
//!
//! [`zsl!`] gives a [`ZslShader`] with all four forms from one source. Pick one at runtime with
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

/// Compiled form of a ZSL shader, with one field per target.
/// [`for_backend`](ZslShader::for_backend) picks one at runtime.
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
    /// Returns the [`ShaderDesc`] and entry-point name for `backend`.
    ///
    /// Falls back to SPIR-V for any backend other than HIP, Metal or CUDA.
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

/// Compiles native ZSL to a `ZslShader` for all backends at build time.
pub use zengpu_zsl::zsl;

/// Column-major 4x4 for `#[derive(ZslPushConst)]`. Wrap as `ZslMat4(m.to_cols_array())`. Matches
/// SPIR-V `ColMajor`.
#[derive(Debug, Clone, Copy)]
pub struct ZslMat4(pub [f32; 16]);

/// Private re-exports referenced by code emitted by `#[derive(ZslPushConst)]`. Semver-exempt; do
/// not use directly.
#[doc(hidden)]
pub mod _zsl_priv {
    pub use super::ZslMat4;
    pub use zengpu_hal::Scalar;
}

/// Converts a push-constant struct to scalars, for `Bindings::scalars`.
///
/// Fields are `u32`, `i32`, `f32` or `ZslMat4`. Returns `[zengpu_hal::Scalar; N]` in field order.
/// `ZslMat4` is 16 `Scalar::F32` in column-major order.
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
