use std::env;

fn main() {
    // objc 0.2's `msg_send!` expands to a check for a `cargo-clippy` feature,
    // and that check is compiled here, in the calling crate. Declare the value
    // so the expansion does not trip `unexpected_cfgs`.
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");

    // Link the Metal and Foundation frameworks on Apple platforms.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        || env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios")
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
