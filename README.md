# ZenGPU

ZenGPU is an experimental native-first GPU runtime for Rust. Version `0.0.1`
provides Vulkan graphics and compute, a CPU conformance backend, resident device
arrays, elementwise compute kernels, and a portable `f32` GEMM kernel.

The project is pre-alpha. APIs are expected to change before `0.1.0`.

## What 0.0.1 Includes

- Backend-independent resource descriptions, typed generational handles, and
  the object-safe `GpuDevice` compute interface.
- Vulkan 1.2 compute with descriptor-indexed storage buffers.
- Vulkan swapchains, offscreen and depth targets, and a lightweight frame graph
  with automatic image-layout barriers.
- Zero-copy sampling of a rendered offscreen image by another renderer on the
  same logical device.
- A deterministic CPU compute backend used as the conformance oracle.
- `DeviceArray`, pooled allocation, `f32` add/ReLU kernels, and portable `f32`
  GEMM.

ZenGPU deliberately does not contain scene, ECS, asset, editor, tensor-graph, or
application types. Those belong in consumer crates.

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
- `cpu`: CPU reference backend.

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

See the repository examples for compute dispatch, graph lowering, and Vulkan
graphics:

```text
cargo run --example vec_add
cargo run --example op_graph_lower
cargo run --example cube
```

## Workspace Crates

| Crate | Purpose |
|---|---|
| `zengpu` | Main facade and recommended dependency |
| `zengpu-hal` | Backend-independent types, handles, errors, and traits |
| `zengpu-vulkan` | Vulkan 1.2 graphics and compute backend |
| `zengpu-cpu` | CPU reference backend |
| `zengpu-compute` | Resident arrays, pooling, and elementwise kernels |
| `zengpu-blas` | Portable GEMM kernel |
| `zengpu-conformance` | Cross-backend conformance harness |

## Current Limitations

- Vulkan is the only GPU backend.
- Dispatch and readback are synchronous.
- Built-in elementwise and GEMM kernels support `f32` only.
- Graphics APIs are currently Vulkan-specific rather than exposed through a
  complete backend-independent graphics trait.
- Renderers and UI painters are consumer-side layers built on ZenGPU's generic
  swapchain, target, and synchronization primitives.
- Resource synchronization and lifetime validation are still being expanded.
- Vulkan requires a Vulkan 1.2-capable driver with descriptor indexing.

## Relationship to Other Projects

[Aurea](https://github.com/SkuldNorniern/aurea) is the first graphics consumer.
Compute-graph and ML libraries can lower their own operations to ZenGPU
dispatches and `DeviceArray`s. These consumers do not shape ZenGPU's public
types.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
