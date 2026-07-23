//! ZSL front-end: native lexer and parser.
//!
//! Produces the backend-neutral IR (`crate::ir`). Lowering lives in `crate::backend`.

pub mod diagnostic;
pub mod lex;
pub mod parser;
