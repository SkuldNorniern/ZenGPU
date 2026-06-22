//! ZSL → SPIR-V backend: a [`builder`] that emits SPIR-V words, plus the
//! per-stage lowerers ([`compute`], [`graphics`]).

pub mod builder;
pub mod compute;
pub mod graphics;
pub mod graphics_ir;

pub use compute::lower_compute;
pub use graphics::{lower_fragment, lower_vertex};
