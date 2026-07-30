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
//!
//! [`ZslShader::into_compiled`] bridges to the newer [`CompiledModule`]/
//! [`BackendArtifact`] API, which represents an arbitrary set of compiled
//! backends (including a future DirectX target) instead of a fixed 4-field
//! struct, and reports a missing backend as a typed [`BackendUnavailable`]
//! error instead of silently substituting SPIR-V.
//!
//! `ZslShader`, [`ZslShader::for_backend`], and the current native ZSL source
//! syntax are accepted for this crate's present versioned release. A future
//! release will emit deprecation diagnostics for both, then remove them
//! together in favor of [`CompiledModule`] and the syntax changes tracked in
//! the ZSL language design plan. New code should prefer `CompiledModule` over
//! `ZslShader` where practical.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::slice;

use zengpu_hal::{BackendPreference, ShaderDesc};

// The `zsl!` and `ZslPushConst` expansions name `::zengpu_spirv::...`; this
// lets the inline tests resolve those paths from inside this crate.
#[cfg(test)]
extern crate self as zengpu_spirv;

/// A buffer element's scalar kind, as carried by [`ZslShader`]'s reflection
/// metadata. Independent of `zengpu-zsl`'s internal `TypeId` — `zengpu-zsl` is
/// a `proc-macro = true` crate and cannot export ordinary types for another
/// crate to name, so this is this crate's own small, `const`-friendly mirror
/// of the same information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    F32,
    U32,
    I32,
    /// A graphics push-constant `mat4x4<f32>` param (never a buffer element —
    /// buffers are always scalar).
    Mat4,
}

/// What one push-constant-block member actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// A `device buffer<elem>` param. The push-constant member itself always
    /// holds a bindless-table `u32` slot index, not `elem` — `elem` and
    /// `binding` describe what the *buffer* holds and which bindless table
    /// (matching `zengpu-vulkan`'s `SSBO_BINDING_F32`=0/`SSBO_BINDING_U32`=2/
    /// `SSBO_BINDING_I32`=3) it routes to.
    BufferIndex { elem: ScalarKind, binding: u32 },
    /// A push-constant scalar (or graphics `mat4x4<f32>`) param, stored as
    /// its own declared type.
    Scalar { kind: ScalarKind },
}

/// One param's push-constant layout, as embedded in a `const ZslShader`.
#[derive(Debug, Clone, Copy)]
pub struct ParamLayout {
    pub name: &'static str,
    /// Byte offset within the push-constant block.
    pub offset: u32,
    /// Size in bytes (4 for a bindless index/scalar; 64 for `mat4x4<f32>`).
    pub size: u32,
    pub kind: ParamKind,
}

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
    /// Push-constant layout + bindless-binding metadata for this entry
    /// point's params, in declaration order (buffer params first, then
    /// scalar params — matching every backend's own push-constant-block
    /// construction).
    pub reflection: &'static [ParamLayout],
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

/// Identifies one ZSL lowering target.
///
/// Ordered so `CompiledModule::artifacts` (a `BTreeMap`) iterates in a stable,
/// predictable sequence; the order itself carries no other meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendId {
    Spirv,
    Hip,
    Cuda,
    Metal,
    Dx,
}

impl BackendId {
    /// The `zengpu-zsl`/`zengpu-spirv` cargo feature that must be enabled to
    /// compile this backend's emitter. Used in [`BackendUnavailable`] so a
    /// caller knows exactly what to turn on.
    pub fn required_cargo_feature(self) -> &'static str {
        match self {
            BackendId::Spirv => "spirv",
            BackendId::Hip => "hip",
            BackendId::Cuda => "cuda",
            BackendId::Metal => "metal",
            BackendId::Dx => "dx",
        }
    }
}

/// One backend's compiled output.
///
/// `DxHlsl`/`Dxil` are placeholders: no `zengpu-zsl` backend emits them yet, so
/// no `CompiledModule` in this codebase constructs those variants today.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendArtifact {
    /// SPIR-V words for Vulkan (and any other SPIR-V consumer).
    Spirv(Vec<u32>),
    /// HIP C++ source for AMD ROCm (compiled at runtime via hipRTC).
    HipSource(String),
    /// CUDA C++ source for NVIDIA CUDA (compiled at runtime via NVRTC).
    CudaSource(String),
    /// Metal Shading Language source for Apple Metal.
    MetalSource(String),
    /// HLSL source for DirectX (compiled offline/at-runtime via DXC/FXC).
    /// Not produced by any backend yet.
    DxHlsl(String),
    /// Compiled DXIL bytecode for DirectX 12. Not produced by any backend yet.
    Dxil(Vec<u8>),
}

/// Returned when a [`CompiledModule`] is asked for a backend it does not have
/// an artifact for — either because that emitter wasn't compiled in (cargo
/// feature off) or because this particular shader has nothing for that stage
/// (e.g. a graphics module has no HIP output). Never silently substitutes
/// another backend's artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendUnavailable {
    pub backend: BackendId,
    pub required_cargo_feature: &'static str,
}

impl Display for BackendUnavailable {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "no {:?} artifact available (enable the \"{}\" cargo feature, and confirm this shader targets that backend)",
            self.backend, self.required_cargo_feature
        )
    }
}

impl Error for BackendUnavailable {}

/// An arbitrary set of compiled backend artifacts for one ZSL module.
///
/// Unlike [`ZslShader`]'s fixed 4 fields, this can represent any subset of
/// backends (including a future DX one) without every caller needing to know
/// about every backend up front.
#[derive(Debug, Clone, Default)]
pub struct CompiledModule {
    pub artifacts: BTreeMap<BackendId, BackendArtifact>,
}

impl CompiledModule {
    /// Look up the compiled artifact for `backend`.
    ///
    /// Returns [`BackendUnavailable`] rather than substituting a different
    /// backend's output — callers that want a fallback must ask for one
    /// explicitly.
    pub fn artifact(&self, backend: BackendId) -> Result<&BackendArtifact, BackendUnavailable> {
        self.artifacts
            .get(&backend)
            .ok_or_else(|| BackendUnavailable {
                backend,
                required_cargo_feature: backend.required_cargo_feature(),
            })
    }
}

impl ZslShader {
    /// Bridge to the new [`CompiledModule`] API.
    ///
    /// An artifact is included only when its `ZslShader` field is non-empty —
    /// today an empty `hip`/`msl`/`cuda` string legitimately means either that
    /// backend's cargo feature was off, or this particular shader is a
    /// graphics module that backend doesn't lower (matching existing
    /// `compile_zsl_all` behavior). `spv` is always included since SPIR-V is
    /// always compiled by default.
    pub fn into_compiled(&self) -> CompiledModule {
        let mut artifacts = BTreeMap::new();
        artifacts.insert(BackendId::Spirv, BackendArtifact::Spirv(self.spv.to_vec()));
        if !self.hip.is_empty() {
            artifacts.insert(
                BackendId::Hip,
                BackendArtifact::HipSource(self.hip.to_string()),
            );
        }
        if !self.cuda.is_empty() {
            artifacts.insert(
                BackendId::Cuda,
                BackendArtifact::CudaSource(self.cuda.to_string()),
            );
        }
        if !self.msl.is_empty() {
            artifacts.insert(
                BackendId::Metal,
                BackendArtifact::MetalSource(self.msl.to_string()),
            );
        }
        CompiledModule { artifacts }
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
        // Emitted only when this crate asks zengpu-zsl for HIP.
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

    // ── ZslMat4 ──────────────────────────────────────────────────────────────────

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
            1.0f32, 0.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
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
        // off-diagonal must be 0
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
        // 1 (scale) + 16 (mvp) + 1 (tint) = 18
        assert_eq!(s.len(), 18);
        assert_eq!(s[0], Scalar::F32(2.0));
        assert_eq!(s[1], Scalar::F32(1.0)); // mvp[0]
        assert_eq!(s[16], Scalar::F32(1.0)); // mvp[15]
        assert_eq!(s[17], Scalar::U32(255));
    }
}
