# ZenGPU

**A native-first GPU runtime for Rust — graphics *and* general compute under one
device, behind a clean, backend-independent API.**


ZenGPU is one runtime for GPU **rendering** and GPU **compute**: a shared
device / memory / queue / synchronization core, with two workload-shaped
abstraction layers on top — a **graphics HAL** (3D-capable, with an automatic
frame-graph) and a **general-purpose compute HAL** (bindless dispatch, async
readback). It targets **native** backends first (Vulkan, then D3D12 / Metal /
CUDA / HIP) and is **not** bound to the WebGPU portability spec.

> **Status: design / pre-alpha.** [`plan.md`](plan.md) is the source of truth —
> the full architecture, the decision record, and the milestones. The code is
> being built against it; the API snippets below show the *planned* shape.

---

## Why ZenGPU

- **Graphics + compute, one device.** A render frame and a compute dispatch share
  the same device, allocator, queues, and synchronization — no two runtimes to
  reconcile.
- **Native-first, not a WebGPU clone.** The API is shaped by native backends and
  free to use their features (timeline semaphores, descriptor indexing, tensor
  cores, mesh shaders) instead of sanding down to a portable-web subset.
- **General-purpose.** No consumer-specific types in the public API. ZenGPU takes
  a *raw window handle* (so any windowing library works) and exposes *plain GPU
  compute* (so any library — physics, simulation, image processing, ML — can use
  it). It is the first GPU backend for the [aurea](https://github.com/SkuldNorniern/aurea)
  GUI toolkit, but nothing in the API knows that.
- **Strict and clear.** Buffers, textures, pipelines, queues, fences, and explicit
  memory usage stay visible. Safe defaults, explicit escape hatches, a validation
  mode that catches use-after-free, missing barriers, and wrong resource state.
- **Verifiable across backends.** A CPU reference backend is the conformance
  oracle: every compute op is checked GPU-vs-CPU within tolerance in CI.

---

## Architecture

```text
            ┌──────────────────────────────────────────────┐
            │                 Shared Core                   │
            │  Instance · Adapter · Device · Memory · Queue │
            │  Fence/Event · Slotmap registry · Validation  │
            └──────────────────────────────────────────────┘
                   ▲                              ▲
        ┌──────────┘                              └──────────┐
  ┌──────────────────┐                       ┌──────────────────┐
  │   Graphics HAL   │                       │   Compute HAL    │
  │  (3D-capable)    │                       │ (general-purpose)│
  │  + frame-graph   │                       │  bindless, async │
  └──────────────────┘                       └──────────────────┘
                   │                              │
            ┌──────┴──────────────────────────────┴──────┐
            │   Backends: Vulkan · CPU (now)              │
            │   D3D12 · Metal · CUDA · HIP (planned)      │
            └─────────────────────────────────────────────┘
```

- **Split HAL.** Graphics and compute share only the core primitives. A backend
  may implement one or both: Vulkan implements both, D3D12 graphics-only, CUDA
  compute-only. This keeps native compute (CUDA) from being forced into a
  graphics-shaped model.
- **The render layer (frame-graph) sits above the graphics HAL**; a game engine
  (ECS, scene, assets, physics) sits above *that* — ZenGPU renders, the engine
  decides what to render.
- **Tensor graphs sit above the compute HAL**, in [Laminax](https://github.com/SkuldNorniern).
  ZenGPU executes nodes; it does not plan them.

See [`plan.md`](plan.md) for the full design and the numbered decision record.

---

## Usage (planned API)

### General compute — usable by any library

No tensors, no graph, no ML assumptions — just upload, dispatch, read back:

```rust
use zengpu::*;

fn main() -> Result<()> {
    let instance = Instance::new();
    let device = instance.request_device(DeviceRequest {
        backend:  BackendPreference::Auto,        // native: Vulkan / D3D12 / Metal / CUDA
        power:    PowerPreference::HighPerformance,
        features: Features::COMPUTE,
    })?;

    let input  = device.create_buffer_init(&data, BufferUsage::STORAGE)?;
    let output = device.create_buffer(BufferDesc {
        size:   data.len() * 4,
        usage:  BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::GpuOnly,
    })?;

    let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
        shader: device.load_spirv(include_bytes!("vec_add.spv"))?,
        entry:  "main",
    })?;

    let mut cmd = device.create_command_list()?;
    cmd.dispatch(pipeline, Bindings {
        buffers: &[input.index(), output.index()],     // bindless indices
        scalars: &[Scalar::U32(data.len() as u32)],
        grid:    [data.len().div_ceil(256) as u32, 1, 1],
    });
    let done = device.queue().submit(cmd)?;

    let result = output.map_async(done)?.wait()?;       // async readback
    println!("{:?}", &result[..8]);
    Ok(())
}
```

### Graphics — raw-handle surface, frame-graph

The app owns the window; ZenGPU owns the swapchain. `window` is anything that
exposes a raw window handle (aurea, winit, SDL, your own):

```rust
let surface = instance.create_surface(&window)?;
surface.configure(&device, SurfaceConfig {
    format, width, height,
    present_mode: PresentMode::Fifo,        // vsync
});

// per frame:
let frame = surface.acquire()?;             // swapchain image, fence-backed
let mut g = FrameGraph::new();
g.pass("main")
    .color(frame.target())
    .draw(|pass| {
        pass.set_pipeline(triangle_pipeline);
        pass.draw(0..3, 0..1);
    });
device.run_graph(&g)?;                       // ZenGPU orders passes + inserts barriers
frame.present();
```

---

## Backends

Native-first. WebGPU/WASM is **not** a target and does not shape the design.

| Backend     | Kind      | Graphics | Compute | Status                         |
|-------------|-----------|:--------:|:-------:|--------------------------------|
| Vulkan      | native    |    ✅    |   ✅    | Tier 0 — in progress           |
| CPU         | reference |    —     |   ✅    | Tier 0 — conformance oracle    |
| D3D12       | native    |    ✅    |   —     | Tier 1 — planned (Windows)     |
| Metal       | native    |    ✅    |   ✅    | Tier 1 — planned (Apple)       |
| CUDA        | native    |    —     |   ✅    | Tier 1 — planned (cuBLAS/ML)   |
| HIP / ROCm  | native    |    —     |   ✅    | Tier 2 — planned (AMD)         |
| WebGPU/WASM | —         |    —     |   —     | **Not a target**               |

---

## Workspace

```text
zengpu            main crate, stable public API
zengpu-hal        shared-core + split-HAL traits (graphics + compute shapes)
zengpu-graphics   surface/swapchain, render targets, graphics pipelines
zengpu-render     automatic frame-graph (passes, transient aliasing, barriers)
zengpu-compute    general compute: buffers, dispatch, bindless, async readback
zengpu-blas       BLAS bridge (vendor libs + portable fallback kernels)
zengpu-compiler   compiler kernel ABI (bindless module/metadata launch)
zengpu-vulkan     Vulkan backend (graphics + compute)
zengpu-cpu        CPU reference backend (compute oracle)
```

---

## Design principles (the short version)

ZenGPU's architecture is a set of recorded decisions ([`plan.md` §0](plan.md)):

- **Split HAL** — graphics and compute shapes, so native compute (CUDA) is never
  second-class.
- **Generational-index handles** — cheap `Copy` handles with use-after-free
  detection.
- **Bindless** — descriptor-indexing throughout; keeps the compiler ABI stable.
- **Multi-threaded recording**, **async readback**, **automatic frame-graph**,
  **multi-surface + offscreen targets** — all baseline.
- **CPU reference = conformance oracle** — the multi-backend claim is only as real
  as the GPU-vs-CPU test.
- **Native-first** — borrow wgpu's *ergonomics* (raw-handle surfaces, text out of
  core), never the WebGPU *spec*.

---

## Project relationships

- **[aurea](https://github.com/SkuldNorniern/aurea)** — GUI toolkit; ZenGPU's
  first graphics consumer, on the path to a game engine. aurea provides the
  window; ZenGPU provides the GPU.
- **Laminax / [cetana](https://github.com/SkuldNorniern/cetana)** — tensor graph
  and ML library; the primary compute consumer. Laminax owns the op graph; ZenGPU
  executes the nodes.
- **[lamina](https://github.com/SkuldNorniern/lamina)** — compiler; emits GPU
  kernels that ZenGPU launches through the bindless compiler ABI.

These are consumers, not constraints: the public API carries none of their types.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
