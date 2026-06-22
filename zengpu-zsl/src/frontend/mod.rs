//! ZSL front-end: AST and the type system.
//!
//! Backend-neutral — shared by every lowering backend (SPIR-V today, MSL next).
//! This is the parsing/typing half of the toolchain; lowering to a specific
//! target lives under `crate::backend`.

pub mod ast;
pub mod parse;
pub mod types;

// Native ZSL frontend (no syn/quote). Wired into the macro shell as the parser
// lands; `allow(dead_code)` until then.
#[allow(dead_code)]
pub mod lex;
#[allow(dead_code)]
pub mod parser;
