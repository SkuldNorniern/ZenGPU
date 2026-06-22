//! ZSL front-end: AST and the type system.
//!
//! Backend-neutral — shared by every lowering backend (SPIR-V today, MSL next).
//! This is the parsing/typing half of the toolchain; lowering to a specific
//! target lives under `crate::backend`.

pub mod ast;
pub mod parse;
pub mod types;

// Native ZSL frontend (no syn/quote): lexer + parser feeding the IR.
pub mod lex;
pub mod parser;
