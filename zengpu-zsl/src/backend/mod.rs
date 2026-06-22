//! ZSL lowering backends — each turns the front-end's parsed entry point into a
//! concrete target. SPIR-V (Vulkan) and MSL (Metal) are both IR consumers; a
//! backend is a consumer of the shared front-end, not the language model.

// MSL backend: consumed by the Metal HAL backend (wired next); allow unused
// until then so the lib build stays warning-free.
#[allow(dead_code, unused_imports)]
pub mod msl;
pub mod spirv;
