//! ZSL lowering backends: each turns the parsed entry point into one target. They consume the
//! shared front end and do not define the language.

pub mod spirv;
