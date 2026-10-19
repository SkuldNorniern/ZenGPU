# Releasing ZenGPU

The tag workflow creates a GitHub release after the reusable CI workflow passes
on Windows, Linux and macOS, including the Rust 1.88 checks. It does not publish
crates to crates.io.

Before cutting a tag, run:

```sh
cargo fmt --all -- --check
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +1.88.0 check --workspace --all-targets --all-features
```

With a current stable Cargo, `cargo package --workspace` assembles and verifies
the crates together using a temporary local registry. This can catch packaging
failures before any dependencies have been published. CI also checks workspace
package assembly on Linux. Keep each crate's `LICENSE` copy in sync with the
workspace `LICENSE`.

Backend tests can return early when the driver or adapter is missing. A passing
run on a machine without that backend does not establish hardware correctness.
Run the backend regressions on actual CUDA, HIP and Vulkan hardware, and the
Metal release check from the README on a real Mac. For HIP, include the
`library_kernels` integration tests; for CUDA, include the async transfer, cache,
and GEMM tests. Keep the adapter and test output with the release evidence.

Verify all package versions and the root dependency requirements before
publishing. `zen-window-handle` has its own version; it does not inherit the
workspace version. Publish dependencies before their consumers in this order
(including dependencies needed to verify packaged tests/examples):

1. `zen-window-handle`
2. `zengpu-hal`, `zengpu-spv`, `zengpu-zsl`
3. `zengpu-spirv`
4. `zengpu-compute`, then `zengpu-blas`
5. `zengpu-cpu`, `zengpu-vulkan`, `zengpu-metal`, `zengpu-hip`, `zengpu-cuda`,
   `zengpu-dx12`
6. `zengpu-conformance`
7. `zengpu`

For each crate, use `cargo publish -p <crate> --dry-run` first, then publish it
and wait for its version to become available in the registry before proceeding
to consumers. A consumer's dry run cannot resolve an unpublished dependency
from the registry, even though the dependency builds in the local workspace.
Check the packaged file list, especially licenses and any build-script inputs,
with `cargo package -p <crate> --list`.

Once the intended crates are available and validation is complete, push a
version tag matching the workspace version (for example `v0.0.1`). Use a
prerelease version and matching tag if that is the intended channel. Check the
generated notes for features, fixes and performance changes, and update Aurea's
ZenGPU revision or version requirement to the tested release.
