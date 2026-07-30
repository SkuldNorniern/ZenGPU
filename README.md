# ZenGPU

ZenGPU is a pre-alpha GPU runtime for Rust. It provides a shared device model
for graphics and general compute: buffers, textures, shaders, compute
pipelines, render targets, surfaces, and command recording use the same backend
foundation.

Version `0.0.1` is pre-alpha. APIs are expected to change before `0.1.0`.

## What 0.0.1 Includes

- A backend-independent HAL with typed generational handles, resource
  descriptors, structured errors, and object-safe compute traits.
- A split graphics contract for graphics-capable devices: surfaces, frames,
  render targets, graphics pipelines, and allocation-conscious command lists
  that record directly into backend command buffers.
- Vulkan 1.2 graphics and compute through `zengpu-vulkan`.
- Vulkan swapchains, offscreen targets, depth targets, and a lightweight frame
  graph with automatic image-layout barriers.
- Same-device zero-copy handoff from rendered targets to sampled-image slots.
- Compute-only backends for HIP/ROCm (`zengpu-hip`) and CUDA (`zengpu-cuda`),
  plus a native Metal backend (`zengpu-metal`, graphics and compute, macOS/
  iOS), all driven from the same ZSL source. A DirectX 12 backend
  (`zengpu-dx12`) exists as an inert skeleton.
- A deterministic CPU backend used for conformance tests.
- `DeviceArray`, pooled allocation, `f32` add/ReLU kernels, and portable `f32`
  GEMM.
- Small SPIR-V tooling for disassembly and structural diagnostics during shader
  bring-up.
- `zengpu_spirv!`, a shader macro that can compile GLSL through `inline-spirv`
  or ZSL through the local Rust-flavored shader pipeline. ZSL compiles to
  SPIR-V, HIP C++, CUDA C++, and MSL from one source, with a unified type
  system (`u32`/`i32`/`f32`/`bf16`/`f16` buffer storage), typed bindless
  storage buffers, and subgroup/workgroup cooperative reductions
  (`subgroup_reduce_*`, `subgroup_shuffle_*`, `workgroup_reduce_*`) for
  cross-lane and cross-workgroup math.

ZenGPU does not include scene, ECS, asset, editor, tensor-graph, or application
types. Consumer crates define planning and presentation policy; ZenGPU provides
execution, resources, synchronization, and backend translation.

## Installation

The default feature set enables Vulkan and compute helpers:

```toml
[dependencies]
zengpu = "0.0.1"
```

Feature flags:

- `vulkan` (default): Vulkan graphics and compute backend.
- `compute` (default): `DeviceArray`, `BufferPool`, and elementwise kernels.
- `blas`: portable GEMM; implies `compute`.
- `cpu`: CPU reference backend for conformance tests.
- `hip`: HIP/ROCm compute backend.
- `cuda`: CUDA compute backend.
- `metal`: native Metal compute/graphics backend (macOS).
- `dx12`: DirectX 12 backend (currently an inert skeleton).

Foundation-only users can disable defaults:

```toml
zengpu = { version = "0.0.1", default-features = false }
```

### macOS

Vulkan runs on macOS through [MoltenVK](https://github.com/KhronosGroup/MoltenVK):

```sh
brew install vulkan-loader molten-vk
# Put the loader on macOS's default dylib search path (one-time, no sudo):
ln -sf /opt/homebrew/lib/libvulkan.dylib   /usr/local/lib/libvulkan.dylib
ln -sf /opt/homebrew/lib/libvulkan.1.dylib /usr/local/lib/libvulkan.1.dylib
```

After that, examples and tests run with no extra env (the Homebrew loader
auto-discovers the MoltenVK ICD):

```sh
cargo run --release --example cube
```

The native Apple Metal backend (`metal` feature) needs no setup.

## Minimal Vulkan Compute

This round-trips a host-visible buffer through the backend-independent device
interface:

```rust,no_run
use zengpu::{
    AdapterRequest, BufferDesc, BufferUsage, DeviceRequest, GpuAdapter, GpuInstance,
    MemoryUsage, VulkanInstance,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = VulkanInstance::new()?;
    let adapter = instance
        .request_adapter(AdapterRequest::default())
        .ok_or("no Vulkan adapter")?;
    let device = adapter.open(DeviceRequest::default())?;

    let buffer = device.create_buffer(BufferDesc {
        size: 4,
        usage: BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::Upload,
    })?;
    device.write_buffer(buffer, 0, &[1, 2, 3, 4])?;
    assert_eq!(device.read_buffer(buffer, 0, 4)?, [1, 2, 3, 4]);
    device.destroy_buffer(buffer);
    Ok(())
}
```

## Shader Input

The `zengpu_spirv!` macro accepts either GLSL or ZSL.

```rust,ignore
use zengpu::zengpu_spirv;

const VERT: &[u32] = zengpu_spirv!(
    r#"
    #version 450
    void main() { gl_Position = vec4(0.0); }
    "#,
    vert,
    vulkan1_0
);
```

ZSL is a Rust-flavored shader language compiled by `zengpu-zsl`: from one ZSL
source, `zsl!` produces SPIR-V (Vulkan), HIP C++ (ROCm), CUDA C++ (NVIDIA), and
MSL (Metal) simultaneously; pick the right compiled form for the active
backend at runtime with `ZslShader::for_backend`.

This ZSL compute kernel scales one buffer into another:

```rust,ignore
use zengpu_spirv::zsl;

const SCALE_ZSL: ZslShader = zsl!(
    push Push { n: u32, scale: f32 }
    @workgroup_size(64)
    kernel scale(
        p: Push,
        src: device buffer<f32>,
        dst: device mut buffer<f32>,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            dst[i] = src[i] * p.scale
        }
    }
);
```

The equivalent GLSL uses ZenGPU's bindless storage-buffer table and push
constants explicitly:

```rust,ignore
const SCALE_GLSL: &[u32] = zengpu_spirv!(
    r#"
    #version 450
    layout(local_size_x = 64) in;

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform Push {
        uint src;
        uint dst;
        uint len;
        float scale;
    } pc;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.len) {
            g_bufs[pc.dst].data[i] = g_bufs[pc.src].data[i] * pc.scale;
        }
    }
    "#,
    comp,
    vulkan1_0
);
```

ZSL supports compute, vertex, and fragment entry points; typed bindless
storage buffers (`u32`/`i32`/`f32`/`bf16`/`f16`); push constants; scalars,
vectors, and matrices; arithmetic and comparisons; `if`/`for` control flow;
workgroup-shared memory and barriers; atomics; and subgroup/workgroup
collectives (`subgroup_reduce_add/min/max/and/or`, `subgroup_scan_inclusive/
exclusive_add`, `subgroup_shuffle*`, `subgroup_any/all/ballot`,
`workgroup_reduce_add/min/max`) for cooperative cross-lane and
cross-workgroup reductions. It remains smaller than GLSL/HLSL in absolute
feature count (no user-defined structs, modules, or recursion yet) but covers
real compute and graphics workloads today, not just experimentation.

## Examples

Run examples from the `ZenGPU` directory:

```bash
cargo run --example vec_add
cargo run --example op_graph_lower
cargo run --release --example heavy_compute
cargo run --example cube
```

- `vec_add`: upload buffers, dispatch a bindless compute shader, read results.
- `op_graph_lower`: shows how a consumer graph could lower to `DeviceArray`,
  elementwise kernels, and GEMM.
- `heavy_compute`: sustained GEMM workload for checking the compute path under
  heavier GPU use; tune with `ZENGPU_HEAVY_DIM` and `ZENGPU_HEAVY_REPS`.
- `cube`: create a Vulkan surface and render a windowed graphics workload.

## Workspace Crates

| Crate | Purpose |
|---|---|
| `zengpu` | Main facade and recommended dependency |
| `zengpu-hal` | Backend-independent types, handles, traits, descriptors, and errors |
| `zengpu-vulkan` | Vulkan 1.2 graphics and compute backend |
| `zengpu-hip` | HIP/ROCm compute backend |
| `zengpu-cuda` | CUDA compute backend |
| `zengpu-metal` | Native Metal compute/graphics backend |
| `zengpu-dx12` | DirectX 12 backend (inert skeleton) |
| `zengpu-cpu` | CPU reference backend |
| `zengpu-compute` | Resident arrays, pooling, and elementwise kernels |
| `zengpu-blas` | Portable GEMM kernel |
| `zengpu-conformance` | Cross-backend conformance harness |
| `zengpu-spirv` | Public shader macro and push-constant helpers |
| `zengpu-spv` | SPIR-V decoding, disassembly, and structural validation |
| `zengpu-zsl` | ZSL parsing and lowering to SPIR-V, HIP, CUDA, and MSL |

Most users should depend on `zengpu`. The subcrates are available for backend
work, conformance, or macro internals.

## Design Boundaries

- ZenGPU executes work; higher layers decide what work should exist and in what
  application-level order.
- Graphics consumers bring their own windows, renderers, painters, scenes, text
  systems, and asset models.
- Compute consumers bring their own tensors, graphs, schedulers, and fusion
  policies.
- Public APIs stay backend-neutral. Vulkan is the first backend, not the shape
  every caller must copy.

## Current Limitations

- Vulkan and Metal are the only backends with graphics support; HIP and CUDA
  are compute-only. DirectX 12 is an inert skeleton (no device creation yet).
  ZSL's SPIR-V bf16/f16 typed storage is not yet implemented (HIP/CUDA/MSL
  already have it); `dot_f32` and workgroup `any`/`all`/scan are not yet
  implemented on any backend.
- Dispatch and readback are synchronous.
- Built-in elementwise and GEMM kernels support `f32` only (ZSL itself has
  wider type support: `u32`/`i32`/`bf16`/`f16` storage).
- The CPU backend is intended for correctness testing rather than production
  fallback use.
- ZSL has no user-defined structs, modules, or recursion yet; GLSL remains
  available through `inline-spirv` for anything outside ZSL's current scope.
- Async readback, deferred destruction, and broader memory-pool policy are
  planned optimization areas.
- Resource synchronization and lifetime validation are still being expanded.
- Vulkan requires a Vulkan 1.2-capable driver with descriptor indexing.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
