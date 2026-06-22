//! ZSL front-end: AST and the type system.
//!
//! Backend-neutral — shared by every lowering backend (SPIR-V today, MSL next).
//! This is the parsing/typing half of the toolchain; lowering to a specific
//! target lives under `crate::backend`.

pub mod ast;
pub mod types;
