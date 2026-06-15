# Release Process

## 0.0.1 Checklist

1. Run `cargo test --workspace --all-targets --all-features`.
2. Run `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
3. Set `RUSTDOCFLAGS=-D warnings` for the shell, then run
   `cargo doc --workspace --all-features --no-deps`.
4. Run `cargo package --workspace --no-verify` to inspect all archives before
   the workspace crates exist on crates.io.
5. Confirm the repository is clean and the release commit is pushed.
6. Create and push the signed or annotated tag `v0.0.1`.
7. Publish crates in dependency order:

```text
zengpu-hal
zengpu-cpu
zengpu-vulkan
zengpu-compute
zengpu-blas
zengpu-conformance
zengpu
```

Use `cargo publish --dry-run -p <crate>` immediately before each publish. Cargo
must see each newly published dependency on crates.io before packaging the next
dependent crate, so allow for index propagation between steps.

8. Create the GitHub release from `CHANGELOG.md`.

Do not publish from a dirty worktree or use `--allow-dirty` for the real release.
