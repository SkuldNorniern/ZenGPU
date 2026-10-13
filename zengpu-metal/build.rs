use std::env;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        || env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios")
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
