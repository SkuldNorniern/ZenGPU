fn main() {
    // Find ROCm installation.
    let rocm_path = std::env::var("ROCM_PATH")
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    // HIP runtime — provides hipMalloc, hipMemcpy, hipModuleLaunch*, etc.
    println!("cargo:rustc-link-lib=amdhip64");
    // hipRTC — runtime kernel compilation.
    println!("cargo:rustc-link-lib=hiprtc");

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}
