//! Metal Shading Language text, for `MTLLibrary` via `new_library_with_source`.
//!
//! ABI: each `device buffer<f32>` param binds to `[[buffer(i)]]` in declaration
//! order; push scalars are in a `constant Push&` at `[[buffer(n_buffers)]]`;
//! thread id is `[[thread_position_in_grid]]`.

pub mod compute;

pub use compute::lower_compute;

pub struct MslShader {
    pub source: String,
    pub entry: &'static str,
    /// Number of `device buffer<f32>` params, bound at `[[buffer(0..n)]]`.
    pub buffer_count: u32,
    /// Whether a `constant Push&` argument is present (at `[[buffer(n)]]`).
    pub has_push: bool,
    pub local_size: [u32; 3],
}

/// The single entry-point name used for every emitted MSL function.
pub(crate) const ENTRY: &str = "zsl_main";
