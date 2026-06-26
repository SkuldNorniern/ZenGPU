fn main() {
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
    println!("cargo:rustc-link-lib=nvrtc");
}
