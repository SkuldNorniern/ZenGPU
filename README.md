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
- A deterministic CPU backend used for conformance tests.
- `DeviceArray`, pooled allocation, `f32` add/ReLU kernels, and portable `f32`
  GEMM.
- `zengpu_spirv!`, a shader macro that can compile GLSL through `inline-spirv`
  or ZSL through the local Rust-flavored shader pipeline.

ZenGPU does not include scene, ECS, asset, editor, tensor-graph, or application
types. Consumer crates define planning and presentation policy; ZenGPU provides
execution, resources, synchronization, and backend translation.

## Installation

The default feature set enables Vulkan, compute helpers, and BLAS:

```toml
[dependencies]
zengpu = "0.0.1"
```

Feature flags:

- `vulkan` (default): Vulkan graphics and compute backend.
- `compute` (default): `DeviceArray`, `BufferPool`, and elementwise kernels.
- `blas` (default): portable GEMM; implies `compute`.
- `cpu`: CPU reference backend for conformance tests.

Foundation-only users can disable defaults:

```toml
zengpu = { version = "0.0.1", default-features = false }
```

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

ZSL is a Rust-flavored shader input compiled directly to SPIR-V by
`zengpu-zsl`:

```rust,ignore
const COMPUTE: &[u32] = zengpu_spirv!(
    #[compute(local_size_x = 64)]
    fn cs_main(#[global_invocation_id] gid: UVec3) {
        let i = gid.x;
    }
);
```

This ZSL compute kernel scales one buffer into another:

```rust,ignore
const SCALE_ZSL: &[u32] = zengpu_spirv!(
    #[compute(local_size_x = 64)]
    fn cs_scale(src: Buf<f32>, dst: BufMut<f32>, len: u32, scale: f32) {
        let i: u32 = global_id().x;
        if i < len {
            dst[i] = src[i] * scale;
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

The ZSL path currently supports the subset used by ZenGPU tests and renderer
experiments: compute, vertex, and fragment entry points; SSBOs; push constants;
scalars; vectors; matrices; arithmetic; comparisons; and selected control flow.
It is available for experimentation and is still limited.

## Examples

Run examples from the `ZenGPU` directory:

```bash
cargo run --example vec_add
cargo run --example op_graph_lower
cargo run --example cube
```

- `vec_add`: upload buffers, dispatch a bindless compute shader, read results.
- `op_graph_lower`: shows how a consumer graph could lower to `DeviceArray`,
  elementwise kernels, and GEMM.
- `cube`: create a Vulkan surface and render a windowed graphics workload.

## Workspace Crates

| Crate | Purpose |
|---|---|
| `zengpu` | Main facade and recommended dependency |
| `zengpu-hal` | Backend-independent types, handles, traits, descriptors, and errors |
| `zengpu-vulkan` | Vulkan 1.2 graphics and compute backend |
| `zengpu-cpu` | CPU reference backend |
| `zengpu-compute` | Resident arrays, pooling, and elementwise kernels |
| `zengpu-blas` | Portable GEMM kernel |
| `zengpu-conformance` | Cross-backend conformance harness |
| `zengpu-spirv` | Public shader macro and push-constant helpers |
| `zengpu-zsl` | Proc-macro internals for ZSL parsing and SPIR-V lowering |

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

- Vulkan is the only GPU backend.
- Dispatch and readback are synchronous.
- Built-in elementwise and GEMM kernels support `f32` only.
- The CPU backend is intended for correctness testing rather than production
  fallback use.
- ZSL is intentionally small and incomplete; GLSL remains available through
  `inline-spirv`.
- Async readback, deferred destruction, and broader memory-pool policy are
  planned optimization areas.
- Resource synchronization and lifetime validation are still being expanded.
- Vulkan requires a Vulkan 1.2-capable driver with descriptor indexing.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
