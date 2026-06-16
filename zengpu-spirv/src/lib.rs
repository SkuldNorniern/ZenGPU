//! ZenGPU shader macro: compiles GLSL to SPIR-V at build time.
//!
//! `zengpu_spirv!` accepts the same syntax as `inline_spirv::inline_spirv!`:
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

#[doc(hidden)]
pub use inline_spirv;

/// Keeps `$crate::zengpu_spirv_proc` resolvable inside `zengpu_spirv!`.
#[doc(hidden)]
pub use zengpu_spirv_proc;

/// Private re-exports referenced by code emitted by `#[derive(ZslPushConst)]`. Semver-exempt; do
/// not use directly.
#[doc(hidden)]
pub mod _zsl_priv {
    pub use zengpu_hal::Scalar;
}

/// Derive `to_scalars()` for a push-constant struct. Fields must be `u32`, `i32` or `f32`.
///
/// Returns [`zengpu_hal::Scalar`]s in field order, for `Bindings::scalars`.
///
/// # Example
///
/// ```ignore
/// use zengpu_spirv::ZslPushConst;
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
pub use zengpu_spirv_proc::ZslPushConst;

/// Compile GLSL source to SPIR-V at build time.
///
/// # GLSL
///
/// Pass a GLSL source string with a stage token, same as `inline_spirv!`:
///
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(r#"#version 450 ..."#, vert, vulkan1_0);
/// ```
///
/// # ZSL
///
/// Pass a Rust-flavored function with a stage attribute:
///
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(
///     #[vertex]
///     fn vs_main(in_pos: Vec3) -> Vec4 { ... }
/// );
/// ```
#[macro_export]
macro_rules! zengpu_spirv {
    (#[$attr:meta] $($rest:tt)*) => {
        $crate::zengpu_spirv_proc::zsl_spirv!(#[$attr] $($rest)*)
    };
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
