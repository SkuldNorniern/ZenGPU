//! ZSL lowering backends. Each consumes the shared front-end IR; none defines the language.

#[allow(dead_code, unused_imports)]
pub mod msl;
pub mod spirv;
