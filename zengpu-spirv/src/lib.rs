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

/// Compile GLSL to SPIR-V at build time, same form as `inline_spirv::inline_spirv!`.
#[macro_export]
macro_rules! zengpu_spirv {
    ($($tt:tt)*) => {
        $crate::inline_spirv::inline_spirv!($($tt)*)
    };
}
