//! ZSL front-end: native lexer + parser (no `syn`/`quote`).
//!
//! Tokenizes and parses native ZSL source into the backend-neutral IR
//! (`crate::ir`); lowering to a target lives under `crate::backend`.

pub mod lex;
pub mod parser;
