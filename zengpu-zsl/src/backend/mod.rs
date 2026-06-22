//! ZSL lowering backends — each turns the front-end's parsed entry point into a
//! concrete target. SPIR-V is the first backend; MSL follows. A backend is a
//! consumer of the shared front-end, not the language model.

pub mod spirv;
