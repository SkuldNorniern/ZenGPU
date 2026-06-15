# Changelog

All notable changes to ZenGPU are documented here.

## [0.0.1] - 2026-06-15

Initial pre-alpha release.

### Added

- Backend-independent HAL types, typed generational handles, and structured
  errors.
- Object-safe compute device interface with buffers, textures, samplers,
  shaders, pipelines, and synchronous dispatch.
- Vulkan 1.2 backend with bindless compute, generic swapchains, offscreen and
  depth targets, and a lightweight frame graph.
- Same-device zero-copy handoff from rendered targets to sampled-image slots.
- CPU reference backend and CPU-versus-Vulkan conformance tests.
- Resident device arrays, pooled allocation, `f32` add/ReLU kernels, and
  portable `f32` GEMM.
- Root `zengpu` facade with feature-gated Vulkan, CPU, compute, and BLAS APIs.

[0.0.1]: https://github.com/SkuldNorniern/ZenGPU/releases/tag/v0.0.1
