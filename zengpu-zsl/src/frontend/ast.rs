//! ZSL entry-point AST. Data only; `syn` parsing is in `parse.rs`.

use syn::Ident;

use crate::frontend::types::ZslType;

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

/// Entry-point param: resolved ZSL type, plus optional location/builtin binding.
#[derive(Debug)]
pub struct ZslParam {
    pub ident: Ident,
    pub ty: ZslType,
    pub location: Option<u32>,
    #[allow(dead_code)]
    pub builtin: Option<String>,
}

#[derive(Debug)]
pub struct ZslEntryPoint {
    #[allow(dead_code)]
    pub stage: Stage,
    pub params: Vec<ZslParam>,
    pub ret: ZslType,
}
