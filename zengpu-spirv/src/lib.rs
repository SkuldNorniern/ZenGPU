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

/// Re-export so that `$crate::zengpu_spirv_proc` resolves inside `zengpu_spirv!`.
#[doc(hidden)]
pub use zengpu_spirv_proc;

/// Compile shader source to SPIR-V at build time.
///
/// # GLSL (current)
/// Pass a GLSL source string with a stage token, identical to `inline_spirv!`:
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(r#"#version 450 ..."#, vert, vulkan1_0);
/// ```
///
/// # ZSL (step 3+)
/// Pass a Rust-flavored function annotated with a stage attribute:
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv!(
///     #[vertex]
///     fn vs_main(in_pos: Vec3) -> Vec4 { ... }
/// );
/// ```
/// ZSL parsing is set up; codegen lands in step 3.
#[macro_export]
macro_rules! zengpu_spirv {
    // ZSL path: input starts with an outer attribute (#[vertex/fragment/compute] fn ...)
    (#[$attr:meta] $($rest:tt)*) => {
        $crate::zengpu_spirv_proc::zsl_spirv!(#[$attr] $($rest)*)
    };
    // GLSL path: string literal + stage token (existing inline_spirv behaviour)
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
