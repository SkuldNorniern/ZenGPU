use std::collections::HashMap;

use crate::ir::node::IrStmt;
use crate::ir::{BufElem, GfxTy, Mutability, ScalarTy};

/// An `out` param. The call-site argument must be a plain, pre-declared local that receives
/// the writes.
#[derive(Clone)]
pub(super) struct FnParamDef {
    pub(super) name: String,
    pub(super) is_out: bool,
    pub(super) ty: ScalarTy,
}

/// Locals declared by the function itself (`let` and loop variables), not its params.
///
/// Renamed fresh per call.
pub(super) struct FunctionDef {
    pub(super) params: Vec<FnParamDef>,
    pub(super) locals: Vec<(String, ScalarTy)>,
    pub(super) body: Vec<IrStmt>,
}

pub(super) struct Ctx {
    pub(super) buffers: HashMap<String, BufferSymbol>,
    pub(super) scalars: HashMap<String, ScalarTy>,
    pub(super) push_param: Option<String>,
    pub(super) id_param: Option<String>,
    pub(super) locals: HashMap<String, ScalarTy>,
    pub(super) locals_order: Vec<(String, ScalarTy)>,
    pub(super) shared: HashMap<String, PendingShared>,
    pub(super) shared_order: Vec<String>,
}

#[derive(Clone, Copy)]
pub(super) struct BufferSymbol {
    pub(super) writable: bool,
    pub(super) elem: BufElem,
}

/// Parser-side workgroup array, `elem` not yet interned. Public: `ir::SharedDecl`.
pub(super) struct PendingShared {
    pub(super) name: String,
    pub(super) elem: ScalarTy,
    pub(super) len: u32,
}

/// Parser-side compute param, type interned later in `parse_kernel_rest`. Public: `ir::Param`.
pub(super) struct PendingParam {
    pub(super) name: String,
    pub(super) kind: PendingParamKind,
}

pub(super) enum PendingParamKind {
    Buffer {
        elem: BufElem,
        mutability: Mutability,
    },
    Scalar(ScalarTy),
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
