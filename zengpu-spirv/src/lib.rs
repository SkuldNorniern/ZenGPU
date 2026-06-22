//! Shader macro that compiles GLSL or ZSL to SPIR-V at build time.
//!
//! # GLSL
//!
//! A GLSL string with a stage token, forwarded to `inline_spirv`:
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
//! A `#[vertex]`, `#[fragment]` or `#[compute]` annotated Rust function, lowered directly to
//! SPIR-V:
//!
//! ```ignore
//! const VERT: &[u32] = zengpu_spirv!(
//!     #[vertex]
//!     fn vs_main(#[location(0)] in_pos: Vec3, mvp: Mat4) -> Vec4 {
//!         mvp * in_pos.extend(1.0)
//!     }
//! );
//! ```

#[doc(hidden)]
pub use inline_spirv;

#[doc(hidden)]
pub use zengpu_zsl;

/// /// Native ZSL to SPIR-V words (compute kernels). See [`zengpu_zsl::zsl`] for syntax.
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

/// Compile shader source to SPIR-V at build time.
///
/// # GLSL
/// Same as `inline_spirv!`:
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(r#"#version 450 ..."#, vert, vulkan1_0);
/// ```
///
/// # ZSL
/// A stage-attributed Rust function:
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
    (#[$attr:meta] $($rest:tt)*) => {
        $crate::zengpu_zsl::zsl_spirv!(#[$attr] $($rest)*)
    };
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
