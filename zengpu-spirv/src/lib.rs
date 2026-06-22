//! ZenGPU shader macro — compile native ZSL to SPIR-V at build time.
//!
//! ZSL is ZenGPU's own shader language with its own lexer/parser/lowerer (no
//! `syn`/`quote`/`inline-spirv`). Use [`zsl!`] for compute and vertex/fragment:
//!
//! ```ignore
//! const VS: &[u32] = zengpu_spirv::zsl!(
//!     push P { mvp: mat4x4<f32> }
//!     vertex vs(@location(0) pos: f32x3, p: P) -> f32x4 {
//!         p.mvp * pos.extend(1.0)
//!     }
//! );
//! ```

/// Compile **native ZSL** source to SPIR-V (`&[u32]`) — ZenGPU's own
/// lexer/parser/lowerer. Compute + vertex/fragment. See [`zengpu_zsl::zsl`].
pub use zengpu_zsl::zsl;

/// Compile **native ZSL** source to Metal Shading Language (`&str`) for the
/// Metal backend (`ShaderDesc::msl`). Compute kernels today. See [`zengpu_zsl::zsl_msl`].
pub use zengpu_zsl::zsl_msl;

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
