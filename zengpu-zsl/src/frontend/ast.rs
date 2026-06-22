//! ZSL AST — parsed representation of a ZSL shader entry point.
//!
//! Data structures only; the `syn` → AST parsing lives in `parse.rs`.

use syn::Ident;

use crate::frontend::types::ZslType;

/// Shader stage of a ZSL entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
            Stage::Compute => "compute",
        }
    }
}

/// A single entry-point parameter with its resolved ZSL type and optional
/// location/builtin binding.
#[derive(Debug)]
pub struct ZslParam {
    pub ident: Ident,
    pub ty: ZslType,
    pub location: Option<u32>,
    #[allow(dead_code)]
    pub builtin: Option<String>,
}

/// The fully-parsed ZSL entry point — ready for lowering.
#[derive(Debug)]
pub struct ZslEntryPoint {
    // stage/ret read by the lib.rs dispatch; dead_code lint fires because the
    // fields are pub and the lint doesn't see cross-module reads in proc-macros.
    #[allow(dead_code)]
    pub stage: Stage,
    pub params: Vec<ZslParam>,
    pub ret: ZslType,
}
