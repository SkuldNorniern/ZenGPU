# ZenGPU

Pre-alpha GPU runtime for Rust: one device model for graphics and compute,
and ZSL, a Rust-flavoured shader language that compiles to every backend.

> Pre-alpha. APIs still move before `0.1.0`.

## Quick Start

```toml
[dependencies]
zengpu = "0.0.1"
```

```rust,ignore
use zengpu::{ZslShader, zsl};

const SCALE: ZslShader = zsl!(
    push Push { n: u32, scale: f32 }
    @workgroup_size(64)
    kernel scale(p: Push, src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
        let i = id.x
        if i < p.n {
            dst[i] = src[i] * p.scale
        }
    }
);
```

One `zsl!` gives SPIR-V, HIP, CUDA and MSL; `ZslShader::for_backend` picks one.

## Support

| Backend | Graphics | Compute | Platforms |
| --- | --- | --- | --- |
| Vulkan 1.2 (`vulkan`, default) | yes | yes | Windows, Linux, macOS via MoltenVK |
| Metal (`metal`) | yes | yes | macOS, iOS |
| HIP/ROCm (`hip`) | no | yes | Linux |
| CUDA (`cuda`) | no | yes | Linux, Windows |
| CPU (`cpu`) | no | yes | all, for conformance tests |
| DirectX 12 (`dx12`) | skeleton | skeleton | Windows |

Other features: `compute` (default) for device arrays, pooling and
elementwise kernels, `blas` for GEMM.

ZSL covers compute, vertex and fragment shaders: bindless buffers
(`u32`/`i32`/`f32`/`bf16`/`f16`), push constants, vectors and matrices,
`if`/`for`, shared memory, atomics, subgroup and workgroup reductions.
Graphics adds swizzles, constant arrays, `vertex_index`/`frag_coord`, bindless
texture sampling, `discard` and dual-source output.

Limits: no structs, modules or recursion in ZSL yet. SPIR-V lacks
`bf16`/`f16` storage. Dispatch and readback are synchronous. Built-in kernels
are `f32` only.

## Examples

```bash
cargo run --example vec_add
cargo run --example cube
cargo run --release --example heavy_compute
```

More in [`examples/`](examples/).

## Crates

Depend on `zengpu`. Subcrates: `zengpu-hal` (types and traits), one per
backend (`zengpu-vulkan`, `-metal`, `-hip`, `-cuda`, `-dx12`, `-cpu`),
`zengpu-compute`, `zengpu-blas`, `zengpu-spirv` and `zengpu-zsl` (shaders),
`zengpu-spv` (SPIR-V checks), `zengpu-conformance`.

## Building

Rust 1.85+. Vulkan needs a driver with descriptor indexing. On macOS:

```sh
brew install vulkan-loader molten-vk
ln -sf /opt/homebrew/lib/libvulkan.dylib /usr/local/lib/libvulkan.dylib
ln -sf /opt/homebrew/lib/libvulkan.1.dylib /usr/local/lib/libvulkan.1.dylib
```

## License

[Apache-2.0](LICENSE).
