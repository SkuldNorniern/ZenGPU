use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // cuda-oxide's build.rs reads CUDA_LIB_PATH to find cuda.lib / libcuda.so.
    // The NVIDIA installer on Windows sets CUDA_PATH but not CUDA_LIB_PATH, so
    // derive and emit a link-search path here to cover that case.
    let mut library_paths = Vec::new();
    if let Some(path) = env::var_os("CUDA_LIB_PATH") {
        library_paths.push(PathBuf::from(path));
    }
    if let Some(root) = env::var_os("CUDA_PATH") {
        let root = PathBuf::from(root);
        library_paths.push(if cfg!(windows) {
            root.join("lib").join("x64")
        } else {
            root.join("lib64")
        });
    }
    if !cfg!(windows) {
        library_paths.extend([
            PathBuf::from("/usr/local/cuda/lib64"),
            PathBuf::from("/usr/local/cuda-11.3/lib64"),
            PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            PathBuf::from("/usr/lib64"),
            PathBuf::from("/usr/lib"),
        ]);
    }
    for path in &library_paths {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    let has_cuda_driver = library_paths
        .iter()
        .any(|path| has_library(path, &["libcuda.so", "libcuda.dylib", "cuda.lib"]));
    if !has_cuda_driver {
        cc::Build::new().file("src/cuda_stub.c").compile("cuda");
    }

    let has_nvrtc = library_paths
        .iter()
        .any(|path| has_library(path, &["libnvrtc.so", "libnvrtc.dylib", "nvrtc.lib"]));
    if !has_nvrtc {
        cc::Build::new().file("src/nvrtc_stub.c").compile("nvrtc");
    }
    // NVRTC for runtime compilation of CUDA C++ sources.
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=src/cuda_stub.c");
    println!("cargo:rerun-if-changed=src/nvrtc_stub.c");
}

fn has_library(path: &Path, names: &[&str]) -> bool {
    names.iter().any(|name| path.join(name).exists())
}
