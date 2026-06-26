fn main() {
    // cuda-oxide's build.rs reads CUDA_LIB_PATH to find cuda.lib / libcuda.so.
    // The NVIDIA installer on Windows sets CUDA_PATH but not CUDA_LIB_PATH, so
    // derive and emit a link-search path here to cover that case.
    if std::env::var("CUDA_LIB_PATH").is_err() {
        if let Ok(root) = std::env::var("CUDA_PATH") {
            let lib_dir = if cfg!(windows) {
                format!("{root}\\lib\\x64")
            } else {
                format!("{root}/lib64")
            };
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
    }
    // NVRTC for runtime compilation of CUDA C++ sources.
    println!("cargo:rustc-link-lib=nvrtc");
}
