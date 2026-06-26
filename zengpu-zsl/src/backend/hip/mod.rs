pub mod compute;

pub use compute::lower_compute;

pub const ENTRY: &str = "zsl_kernel";

pub struct HipShader {
    pub source: String,
    pub entry: &'static str,
    pub buffer_count: u32,
    pub has_scalars: bool,
    pub local_size: [u32; 3],
}
