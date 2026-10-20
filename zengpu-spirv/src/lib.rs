//! Compiles native ZSL to all backends at build time.
//!
//! [`zsl!`] produces a [`ZslShader`] with SPIR-V (Vulkan), HIP C++ (ROCm), MSL (Metal) and CUDA C++
//! (NVIDIA). Pick one at runtime with [`ZslShader::for_backend`].
//!
//! Each backend lowers independently. A failed lowering is an `Err` field with its diagnostic, not
//! a build failure. `@backends(hip, cuda)` before the entry point makes the listed backends a hard
//! build error.
//!
//! [`ZslShader::into_compiled`] gives a [`CompiledModule`]. A missing backend returns
//! [`BackendUnavailable`], never a substituted artifact.
//!
//! `ZslShader`, `for_backend` and the current ZSL syntax are supported. A later release adds
//! deprecation diagnostics before removing them. Prefer [`CompiledModule`] in new code.
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

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::slice;

use zengpu_hal::{
    BackendPreference, GpuDevice, GpuError, Result as HalResult, ShaderDesc, ShaderHandle,
};

#[cfg(test)]
extern crate self as zengpu_spirv;

/// Buffer element kind in `ZslShader` reflection. Own copy: `zengpu-zsl` is a proc-macro crate and
/// exports no types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    F32,
    U32,
    I32,
    /// A `buffer<f16>` element (buffers only; packed two per word on Vulkan).
    F16,
    /// A `buffer<bf16>` element (buffers only; packed two per word on Vulkan).
    Bf16,
    /// A graphics push-constant `mat4x4<f32>`. Never a buffer element.
    Mat4,
    /// A graphics push-constant `f32x2` param.
    Vec2,
    /// A graphics push-constant `f32x4` param.
    Vec4,
}

/// What one push-constant-block member actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Buffer param: the push member is a `u32` slot in the table for `binding`, not `elem`.
    /// `binding` is an `SSBO_BINDING_*` in zengpu-vulkan.
    BufferIndex { elem: ScalarKind, binding: u32 },
    /// A `u32` slot in the sampled-texture table at `binding`.
    TextureIndex { binding: u32 },
    /// A `u32` slot in the storage-image table at `binding`.
    ImageIndex { binding: u32 },
    /// A push-constant scalar, stored as its declared type.
    Scalar { kind: ScalarKind },
}

/// Push-constant layout of one param in a `const ZslShader`.
#[derive(Debug, Clone, Copy)]
pub struct ParamLayout {
    pub name: &'static str,
    pub offset: u32,
    /// Size in bytes (4 for a bindless index/scalar; 64 for `mat4x4<f32>`).
    pub size: u32,
    pub kind: ParamKind,
}

/// All backends for one ZSL shader, from `zsl!`. A field is `Err` if the target has no output (not
/// expressible, or its feature is off). Use `for_backend` at runtime.
pub struct ZslShader {
    /// SPIR-V words for Vulkan (and any other SPIR-V consumer).
    pub spv: Result<&'static [u32], &'static str>,
    /// HIP C++ source for AMD ROCm (compiled at runtime via hipRTC).
    pub hip: Result<&'static str, &'static str>,
    /// Metal Shading Language source for Apple Metal.
    pub msl: Result<&'static str, &'static str>,
    /// CUDA C++ source for NVIDIA CUDA (compiled at runtime via NVRTC).
    pub cuda: Result<&'static str, &'static str>,
    /// Push-constant layout and binding metadata. Buffer params come first, then scalars, as every
    /// backend builds it.
    pub reflection: &'static [ParamLayout],
}

impl ZslShader {
    /// Falls back to SPIR-V for backends other than HIP, Metal or CUDA.
    pub fn for_backend(
        &self,
        backend: BackendPreference,
    ) -> HalResult<(ShaderDesc<'_>, &'static str)> {
        match backend {
            BackendPreference::Hip => Ok((ShaderDesc::hip(self.hip_src()?), "zsl_kernel")),
            BackendPreference::Metal => Ok((ShaderDesc::msl(self.msl_src()?), "zsl_main")),
            BackendPreference::Cuda => Ok((ShaderDesc::cuda_src(self.cuda_src()?), "zsl_kernel")),
            _ => Ok((self.spirv_desc()?, "main")),
        }
    }

    /// `Err` if there is no artifact, or its feature is off.
    pub fn create(&self, device: &dyn GpuDevice) -> HalResult<(ShaderHandle, &'static str)> {
        let (desc, entry) = self.for_backend(device.backend())?;
        Ok((device.create_shader(desc)?, entry))
    }

    /// Convenience: SPIR-V [`ShaderDesc`] for Vulkan-only callers.
    pub fn spirv_desc(&self) -> HalResult<ShaderDesc<'_>> {
        let words = self.spirv_words()?;
        let bytes = unsafe { slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        Ok(ShaderDesc::spirv(bytes))
    }

    pub fn spirv_words(&self) -> HalResult<&'static [u32]> {
        self.spv.map_err(|msg| unavailable(BackendId::Spirv, msg))
    }

    pub fn hip_src(&self) -> HalResult<&'static str> {
        self.hip.map_err(|msg| unavailable(BackendId::Hip, msg))
    }

    pub fn msl_src(&self) -> HalResult<&'static str> {
        self.msl.map_err(|msg| unavailable(BackendId::Metal, msg))
    }

    pub fn cuda_src(&self) -> HalResult<&'static str> {
        self.cuda.map_err(|msg| unavailable(BackendId::Cuda, msg))
    }
}

fn unavailable(backend: BackendId, msg: &str) -> GpuError {
    GpuError::ShaderCompile(format!(
        "zsl: no {} artifact for this shader: {msg}",
        backend.name()
    ))
}

/// ZSL lowering target. Variants are ordered so `CompiledModule::artifacts` iterates stably; the
/// order carries no other meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendId {
    Spirv,
    Hip,
    Cuda,
    Metal,
    Dx,
}

impl BackendId {
    /// Name in `@backends(...)` and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            BackendId::Spirv => "spirv",
            BackendId::Hip => "hip",
            BackendId::Cuda => "cuda",
            BackendId::Metal => "metal",
            BackendId::Dx => "dx",
        }
    }

    /// Cargo feature for this backend's emitter, named in `BackendUnavailable`.
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

/// Output of one backend. `DxHlsl` and `Dxil` are placeholders that no emitter produces.
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
    /// HLSL source for DirectX. Placeholder, no emitter.
    DxHlsl(String),
    /// DXIL for DirectX 12. Placeholder, no emitter.
    Dxil(Vec<u8>),
}

/// No artifact for the requested backend: feature off, or nothing for this stage (graphics has no
/// HIP). Never substitutes another backend's artifact.
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

/// Any set of backend artifacts for one ZSL module. Unlike the fixed fields of `ZslShader`, adding
/// a backend needs no caller changes.
#[derive(Debug, Clone, Default)]
pub struct CompiledModule {
    pub artifacts: BTreeMap<BackendId, BackendArtifact>,
}

impl CompiledModule {
    /// Returns the artifact for `backend`. Never substitutes another backend's output. Ask for a
    /// fallback explicitly.
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
    /// `Err` fields have no artifact. `artifact()` reports `BackendUnavailable`.
    pub fn into_compiled(&self) -> CompiledModule {
        let mut artifacts = BTreeMap::new();
        if let Ok(words) = self.spv {
            artifacts.insert(BackendId::Spirv, BackendArtifact::Spirv(words.to_vec()));
        }
        if let Ok(src) = self.hip {
            artifacts.insert(BackendId::Hip, BackendArtifact::HipSource(src.to_string()));
        }
        if let Ok(src) = self.cuda {
            artifacts.insert(
                BackendId::Cuda,
                BackendArtifact::CudaSource(src.to_string()),
            );
        }
        if let Ok(src) = self.msl {
            artifacts.insert(
                BackendId::Metal,
                BackendArtifact::MetalSource(src.to_string()),
            );
        }
        CompiledModule { artifacts }
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
        assert!(!SHADER.spv.unwrap().is_empty());
        assert_eq!(SHADER.spv.unwrap()[0], SPIRV_MAGIC, "valid SPIR-V header");
        #[cfg(feature = "hip")]
        assert!(SHADER.hip.unwrap().contains("zsl_kernel"));
    }

    #[test]
    fn native_vertex_mvp() {
        const SHADER: ZslShader = zsl!(
            push Mvp { mvp: mat4x4<f32> }
            vertex vs(@location(0) pos: f32x3, m: Mvp) -> f32x4 {
                m.mvp * pos.extend(1.0)
            }
        );
        assert_eq!(SHADER.spv.unwrap()[0], SPIRV_MAGIC);
    }

    #[test]
    fn native_vertex_with_varying() {
        const SHADER: ZslShader = zsl!(
            vertex vs(@location(0) pos: f32x4, @location(1) uv: f32x2) -> (f32x4, f32x2) {
                (pos, uv)
            }
        );
        assert_eq!(SHADER.spv.unwrap()[0], SPIRV_MAGIC);
    }

    #[test]
    fn native_fragment_color() {
        const SHADER: ZslShader = zsl!(
            fragment fs(@location(0) color: f32x4) -> f32x4 {
                color * 0.5
            }
        );
        assert_eq!(SHADER.spv.unwrap()[0], SPIRV_MAGIC);
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
        assert_eq!(SHADER.spv.unwrap()[0], SPIRV_MAGIC);
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
