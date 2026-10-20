//! ZSL to SPIR-V: `builder` emits words; per-stage lowerers are `compute` and `graphics`.

mod builder;
pub mod compute;
pub mod graphics_ir;

pub use compute::lower_compute;
pub use graphics_ir::lower_graphics;
