//! ZSL lowering backends. Each consumes the shared front-end IR; none defines the language.

#[cfg(feature = "metal")]
#[allow(dead_code, unused_imports)]
pub mod msl;

#[cfg(feature = "spirv")]
pub mod spirv;

#[cfg(feature = "hip")]
#[allow(dead_code, unused_imports)]
pub mod hip;

#[cfg(feature = "cuda")]
#[allow(dead_code, unused_imports)]
pub mod cuda;
