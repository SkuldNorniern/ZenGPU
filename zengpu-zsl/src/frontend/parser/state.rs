use std::collections::HashMap;

use crate::ir::node::IrStmt;
use crate::ir::types::TypeId;
use crate::ir::{GfxTy, Mutability};

/// An `out` param. The call-site argument must be a plain, pre-declared local that receives
/// the writes.
#[derive(Clone)]
pub(super) struct FnParamDef {
    pub(super) name: String,
    pub(super) is_out: bool,
    pub(super) ty: TypeId,
}

/// Locals declared by the function itself (`let` and loop variables), not its params.
///
/// Renamed fresh per call.
pub(super) struct FunctionDef {
    pub(super) params: Vec<FnParamDef>,
    pub(super) locals: Vec<(String, TypeId)>,
    pub(super) body: Vec<IrStmt>,
}

pub(super) struct Ctx {
    pub(super) buffers: HashMap<String, BufferSymbol>,
    pub(super) scalars: HashMap<String, TypeId>,
    pub(super) push_param: Option<String>,
    pub(super) id_param: Option<String>,
    pub(super) locals: HashMap<String, TypeId>,
    pub(super) locals_order: Vec<(String, TypeId)>,
    pub(super) shared: HashMap<String, PendingShared>,
    pub(super) shared_order: Vec<String>,
}

impl Ctx {
    pub(super) fn type_ctx(&self) -> crate::ir::typeck::ComputeTypeCtx {
        crate::ir::typeck::ComputeTypeCtx {
            locals: self.locals.clone(),
            scalars: self.scalars.clone(),
            buffers: self
                .buffers
                .iter()
                .map(|(k, v)| (k.clone(), v.elem))
                .collect(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BufferSymbol {
    pub(super) writable: bool,
    pub(super) elem: TypeId,
}

pub(super) struct PendingShared {
    pub(super) name: String,
    pub(super) elem: TypeId,
    pub(super) len: u32,
}

pub(super) struct PendingParam {
    pub(super) name: String,
    pub(super) kind: PendingParamKind,
}

pub(super) enum PendingParamKind {
    Buffer {
        elem: TypeId,
        mutability: Mutability,
    },
    Scalar(TypeId),
}

/// Parser-side `@location` input, type not yet interned. Public counterpart: `ir::GfxInput`.
pub(super) struct PendingGfxInput {
    pub(super) name: String,
    pub(super) location: u32,
    pub(super) ty: GfxTy,
}

/// Parser-side push-constant scalar/matrix param, type not yet interned. Public: `ir::GfxScalar`.
pub(super) struct PendingGfxScalar {
    pub(super) name: String,
    pub(super) ty: GfxTy,
}

pub(super) struct GfxCtx {
    pub(super) inputs: HashMap<String, GfxTy>,
    pub(super) scalars: HashMap<String, GfxTy>,
    pub(super) push_param: Option<String>,
    pub(super) buffers: HashMap<String, bool>,
    pub(super) locals: HashMap<String, GfxTy>,
    pub(super) locals_order: Vec<(String, GfxTy)>,
}
