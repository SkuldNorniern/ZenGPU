//! IR → Metal Shading Language (MSL) text.
//!
//! A second backend alongside `crate::backend::spirv`, consuming the same IR.
//! Metal's `MTLLibrary` needs MSL source, not SPIR-V, so this emits MSL text;
//! the Metal HAL backend compiles it with `new_library_with_source`.
//!
//! ABI (so the Metal dispatch code can bind to match): each `device buffer<f32>`
//! param binds directly to `[[buffer(i)]]` in declaration order; push-block
//! scalars are packed into a `constant Push&` at `[[buffer(n_buffers)]]`; the
//! thread id is `[[thread_position_in_grid]]`.

pub mod compute;

pub use compute::lower_compute;

/// Emitted MSL plus the metadata the Metal backend needs to bind and dispatch.
pub struct MslShader {
    /// The MSL source text.
    pub source: String,
    /// The kernel/function entry-point name to look up in the `MTLLibrary`.
    pub entry: &'static str,
    /// Number of `device buffer<f32>` params, bound at `[[buffer(0..n)]]`.
    pub buffer_count: u32,
    /// Whether a `constant Push&` argument is present (at `[[buffer(n)]]`).
    pub has_push: bool,
    /// Compute threadgroup size.
    pub local_size: [u32; 3],
}

/// The single entry-point name used for every emitted MSL function.
pub(crate) const ENTRY: &str = "zsl_main";
