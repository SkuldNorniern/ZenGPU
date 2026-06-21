fn main() {
    // Link the Metal and Foundation frameworks on Apple platforms.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        || std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios")
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
