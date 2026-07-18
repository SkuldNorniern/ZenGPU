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

#[cfg(test)]
extern crate self as zengpu_spirv;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod zsl_native_tests {
    //! Native ZSL syntax through the `zsl!` macro (no syn/quote in the pipeline).

    use crate::{ZslShader, zsl};

    const SPIRV_MAGIC: u32 = 0x0723_0203;

    #[test]
    fn native_saxpy_compiles_to_spirv() {
        const SHADER: ZslShader = zsl!(
            push SaxpyPush { n: u32, alpha: f32 }
            @workgroup_size(256)
            kernel saxpy(
                push: SaxpyPush,
                x: device buffer<f32>,
                y: device mut buffer<f32>,
                id: global_id,
            ) {
                let i = id.x
                if i < push.n {
                    y[i] = push.alpha * x[i] + y[i]
                }
            }
        );
        assert!(!SHADER.spv.is_empty());
        assert_eq!(SHADER.spv[0], SPIRV_MAGIC, "valid SPIR-V header");
        #[cfg(feature = "hip")]
        assert!(SHADER.hip.contains("zsl_kernel"));
    }

    #[test]
    fn native_vertex_mvp() {
        const SHADER: ZslShader = zsl!(
            push Mvp { mvp: mat4x4<f32> }
            vertex vs(@location(0) pos: f32x3, m: Mvp) -> f32x4 {
                m.mvp * pos.extend(1.0)
            }
        );
        assert_eq!(SHADER.spv[0], SPIRV_MAGIC);
    }

    #[test]
    fn native_vertex_with_varying() {
        const SHADER: ZslShader = zsl!(
            vertex vs(@location(0) pos: f32x4, @location(1) uv: f32x2) -> (f32x4, f32x2) {
                (pos, uv)
            }
        );
        assert_eq!(SHADER.spv[0], SPIRV_MAGIC);
    }

    #[test]
    fn native_fragment_color() {
        const SHADER: ZslShader = zsl!(
            fragment fs(@location(0) color: f32x4) -> f32x4 {
                color * 0.5
            }
        );
        assert_eq!(SHADER.spv[0], SPIRV_MAGIC);
    }

    #[test]
    fn native_copy_with_module_and_workgroup() {
        const SHADER: ZslShader = zsl!(
            module zengpu.examples.copy
            @workgroup_size(64)
            kernel copy(
                src: device buffer<f32>,
                dst: device mut buffer<f32>,
                id: global_id,
            ) {
                let i = id.x
                dst[i] = src[i]
            }
        );
        assert_eq!(SHADER.spv[0], SPIRV_MAGIC);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod push_const_derive_tests {
    use crate::{_zsl_priv::Scalar, ZslMat4, ZslPushConst};

    #[derive(ZslPushConst)]
    struct ScalePush {
        len: u32,
        offset: i32,
        scale: f32,
    }

    #[derive(ZslPushConst)]
    struct AllU32 {
        a: u32,
        b: u32,
        c: u32,
    }

    #[derive(ZslPushConst)]
    struct Empty {}

    #[test]
    fn to_scalars_types_and_order() {
        let p = ScalePush {
            len: 1024,
            offset: -7,
            scale: 2.5,
        };
        let s = p.to_scalars();
        assert_eq!(s[0], Scalar::U32(1024));
        assert_eq!(s[1], Scalar::I32(-7));
        assert_eq!(s[2], Scalar::F32(2.5));
    }

    #[test]
    fn to_scalars_all_u32() {
        let p = AllU32 { a: 3, b: 5, c: 7 };
        let [a, b, c] = p.to_scalars();
        assert_eq!(a, Scalar::U32(3));
        assert_eq!(b, Scalar::U32(5));
        assert_eq!(c, Scalar::U32(7));
    }

    #[test]
    fn to_scalars_empty() {
        let p = Empty {};
        let s: [Scalar; 0] = p.to_scalars();
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn to_scalars_coerces_to_slice() {
        let p = AllU32 { a: 1, b: 2, c: 3 };
        let scalars = p.to_scalars();
        let slice: &[Scalar] = &scalars;
        assert_eq!(slice.len(), 3);
    }

    #[derive(ZslPushConst)]
    struct MvpPush {
        mvp: ZslMat4,
    }

    #[derive(ZslPushConst)]
    struct MixedPush {
        scale: f32,
        mvp: ZslMat4,
        tint: u32,
    }

    #[test]
    fn mat4_expands_to_16_scalars() {
        let identity = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let p = MvpPush {
            mvp: ZslMat4(identity),
        };
        let s = p.to_scalars();
        assert_eq!(s.len(), 16);
        assert_eq!(s[0], Scalar::F32(1.0));
        assert_eq!(s[5], Scalar::F32(1.0));
        assert_eq!(s[10], Scalar::F32(1.0));
        assert_eq!(s[15], Scalar::F32(1.0));
        assert_eq!(s[1], Scalar::F32(0.0));
        assert_eq!(s[4], Scalar::F32(0.0));
    }

    #[test]
    fn mixed_push_mat4_ordering() {
        let identity = [1.0f32; 16];
        let p = MixedPush {
            scale: 2.0,
            mvp: ZslMat4(identity),
            tint: 255,
        };
        let s = p.to_scalars();
        assert_eq!(s.len(), 18);
        assert_eq!(s[0], Scalar::F32(2.0));
        assert_eq!(s[1], Scalar::F32(1.0));
        assert_eq!(s[16], Scalar::F32(1.0));
        assert_eq!(s[17], Scalar::U32(255));
    }
}
