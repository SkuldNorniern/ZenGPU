fn main() {
    // objc 0.2's `msg_send!` expands to a check for a `cargo-clippy` feature,
    // and that check is compiled here, in the calling crate. Declare the value
    // so the expansion does not trip `unexpected_cfgs`.
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");
}
