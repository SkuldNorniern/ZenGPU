//! Shader macro that compiles native ZSL to SPIR-V at build time.
//!
//! Has its own lexer, parser and lowerer, with no `syn`, `quote` or `inline-spirv`. [`zsl!`]
//! handles compute and vertex/fragment shaders:
//!
//! ```ignore
//! const VS: &[u32] = zengpu_spirv::zsl!(
//!     push P { mvp: mat4x4<f32> }
//!     vertex vs(@location(0) pos: f32x3, p: P) -> f32x4 {
//!         p.mvp * pos.extend(1.0)
//!     }
//! );
//! ```

/// Native ZSL to SPIR-V words (compute and vertex/fragment). See [`zengpu_zsl::zsl`].
pub use zengpu_zsl::zsl;

/// Native ZSL to MSL source (`&str`), with the same syntax as [`zsl`]. Compute kernels only.
///
/// The kernel is named `zsl_main`. Buffers bind at `[[buffer(0..n)]]`, push scalars at
/// `[[buffer(n)]]`, and the thread id at `[[thread_position_in_grid]]`.
pub use zengpu_zsl::zsl_msl;

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
