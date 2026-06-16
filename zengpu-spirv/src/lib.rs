//! ZenGPU shader macro — compile GLSL (and later ZSL/WGSL) to SPIR-V at build
//! time.
//!
//! # Step 1 — GLSL shim
//!
//! `zengpu_spirv!` currently accepts the same syntax as `inline_spirv::inline_spirv!`:
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
//! # Roadmap
//!
//! - Step 3+: ZSL (Rust-flavored) input via `#[vertex]`/`#[fragment]`/`#[compute]`
//!   function attributes — auto-detected by the macro, compiled through the ZSL
//!   pipeline rather than shaderc.
//! - Later: WGSL input via a `wgsl` stage token, routed through `naga`.

/// Re-export so that `$crate::inline_spirv` resolves inside `zengpu_spirv!`.
#[doc(hidden)]
pub use inline_spirv;

/// Compile shader source to SPIR-V at build time.
///
/// Currently accepts GLSL in the same form as `inline_spirv::inline_spirv!`.
/// ZSL and WGSL support will be added in later steps without changing the
/// call-site syntax.
#[macro_export]
macro_rules! zengpu_spirv {
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
