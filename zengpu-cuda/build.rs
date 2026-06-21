fn main() {
    // Locate the CUDA toolkit installation for informational purposes.
    // The crate loads libcuda at runtime via libloading — a missing toolkit is
    // not a build error; enumerate_adapters just returns empty.
    let cuda_root = std::env::var("CUDA_PATH").ok().or_else(|| {
        if cfg!(unix) && std::path::Path::new("/usr/local/cuda").exists() {
            Some("/usr/local/cuda".into())
        } else {
            None
        }
    });

    if let Some(root) = cuda_root {
        println!("cargo:rustc-env=ZENGPU_CUDA_TOOLKIT_ROOT={root}");
    }
}
