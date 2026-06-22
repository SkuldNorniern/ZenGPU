//! ZenGPU shader macro — compile GLSL and ZSL to SPIR-V at build time.
//!
//! # GLSL
//!
//! Pass a GLSL source string with a stage token, forwarded to `inline_spirv`:
//!
//! ```ignore
//! use zengpu_spirv::zengpu_spirv;
//!
//! const VERT: &[u32] = zengpu_spirv!(
//!     r#"
//!     #version 450
//!     void main() { gl_Position = vec4(0.0); }
//!     "#,
//!     vert,
//!     vulkan1_0
//! );
//! ```
//!
//! # ZSL
//!
//! Pass a `#[vertex]`, `#[fragment]`, or `#[compute]` annotated Rust function.
//! ZSL is a Rust-flavored shader language compiled directly to SPIR-V:
//!
//! ```ignore
//! const VERT: &[u32] = zengpu_spirv!(
//!     #[vertex]
//!     fn vs_main(#[location(0)] in_pos: Vec3, mvp: Mat4) -> Vec4 {
//!         mvp * in_pos.extend(1.0)
//!     }
//! );
//! ```

/// Re-export so that `$crate::inline_spirv` resolves inside `zengpu_spirv!`.
#[doc(hidden)]
pub use inline_spirv;

/// Compile **native ZSL** source to SPIR-V (`&[u32]`) — ZenGPU's own
/// lexer/parser/lowerer, no `syn`/`quote`. Compute + vertex/fragment. See
/// [`zengpu_zsl::zsl`] for syntax.
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

/// Compile shader source to SPIR-V at build time.
///
/// # GLSL
/// Pass a GLSL source string with a stage token, identical to `inline_spirv!`:
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(r#"#version 450 ..."#, vert, vulkan1_0);
/// ```
///
/// # ZSL
/// Pass a Rust-flavored function annotated with a stage attribute:
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(
///     #[vertex]
///     fn vs_main(#[location(0)] in_pos: Vec3, mvp: Mat4) -> Vec4 {
///         mvp * in_pos.extend(1.0)
///     }
/// );
/// ```
#[macro_export]
macro_rules! zengpu_spirv {
    // GLSL path: string literal + stage token (forwarded to inline_spirv).
    // ZSL now has its own native macro, [`zsl!`]; this GLSL path is transitional.
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
