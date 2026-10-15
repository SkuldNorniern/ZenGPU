# ZenGPU

Pre-alpha GPU runtime for Rust: one device model for graphics and compute,
and ZSL, a Rust-flavoured shader language that compiles to every backend.

> Pre-alpha. APIs still move before `0.1.0`.

Resource handles preserve a process-unique table identity as well as their slot
and generation. Handles from another device, including a replacement device,
are rejected. Keep the complete typed handle in wrappers; code that reconstructs
one must use `Handle::from_owned_parts(index, generation, owner)`.
`Handle::from_parts` now creates an invalid, unowned sentinel. Raw bindless slot
indices in `Bindings` remain local to the submitting device.

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
`zengpu::native` names the platform's graphics backend (Metal on macOS, Vulkan elsewhere) and `native::shader` picks its artifact.

## Support

| Backend | Graphics | Compute | Platforms |
| --- | --- | --- | --- |
| Vulkan 1.2 (`vulkan`, default) | yes | yes | Windows, Linux, macOS via MoltenVK |
| Metal (`metal`, default) | yes | yes | macOS |
| HIP/ROCm (`hip`) | no | yes | Linux |
| CUDA (`cuda`) | no | yes | Linux, Windows |
| CPU (`cpu`) | no | yes | all, for conformance tests |
| DirectX 12 (`dx12`) | skeleton | skeleton | Windows |

Other features: `compute` (default) for device arrays, pooling and
elementwise kernels, `blas` for GEMM.

HIP loads ROCm at run time, so building needs no ROCm and an app built with
`hip` still starts where ROCm is missing (the backend then has no adapters).
It needs ROCm 5.0 or newer at run time and compiles kernels with hipRTC for the exact
gfx target, so RDNA 1 (RX 5000), RDNA 2 laptop parts and other GPUs missing
from AMD's support list run without `HSA_OVERRIDE_GFX_VERSION`. On RDNA 1/2 it
sets `HSA_ENABLE_SDMA=0` before `hipInit` unless you set it yourself or set
`ZENGPU_HIP_NO_ENV_WORKAROUNDS=1`. Peer-to-peer copies are on for CDNA and off
for consumer RDNA; `ZENGPU_HIP_P2P=1` or `0` overrides that.
Set `DeviceRequest::pipeline_cache_path` to keep compiled HIP kernels in
`<path>.hip/`: a kernel then loads in under a millisecond instead of the
~250 ms hipRTC takes.
Waiting threads block rather than spin (`ZENGPU_HIP_SPIN_WAIT=1` restores
HIP's spinning), and on ROCm 6+ buffers of 2 MiB and up come from HIP's
stream-ordered pool, so freeing one never waits for running kernels
(`ZENGPU_HIP_NO_MEMPOOL=1` turns that off).

ZSL covers compute, vertex and fragment shaders: bindless buffers
(`u32`/`i32`/`f32`/`bf16`/`f16`), push constants, vectors and matrices,
`if`/`for`, shared memory, atomics, subgroup and workgroup reductions.
Graphics adds swizzles, constant arrays, `vertex_index`/`frag_coord`, bindless
texture sampling, `discard` and dual-source output. Clip space is Vulkan's (+y down) on every backend; MSL output flips y to match.

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

Rust 1.88+. Vulkan needs a driver with descriptor indexing.

Metal needs nothing extra. For Vulkan on macOS:

```sh
brew install vulkan-loader molten-vk
```

Before publishing Metal support, run the offscreen graphics and transfer
regressions on a real Mac. This mode fails if no Metal adapter opens and prints
the selected adapter:

```sh
ZENGPU_REQUIRE_METAL=1 cargo test -p zengpu-metal graphics_tests -- --test-threads=1 --nocapture
```

## License

[Apache-2.0](LICENSE).
