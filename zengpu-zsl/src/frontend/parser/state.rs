use std::collections::HashMap;

use crate::ir::node::IrStmt;
use crate::ir::{BufElem, GfxTy, Mutability, ScalarTy};

/// One scalar `fn` parameter: `name: ty` (in, by value) or `out name: ty`
/// (out — the call-site argument must be a plain pre-declared local; writes
/// to the parameter inside the body become writes to that local).
#[derive(Clone)]
pub(super) struct FnParamDef {
    pub(super) name: String,
    pub(super) is_out: bool,
    pub(super) ty: ScalarTy,
}

/// A parsed helper function, ready to be inlined by name-substitution at each
/// call site. `locals` are the function's own `let`/`for`-loop-var locals
/// (not its params), each renamed fresh per call to avoid collisions.
pub(super) struct FunctionDef {
    pub(super) params: Vec<FnParamDef>,
    pub(super) locals: Vec<(String, ScalarTy)>,
    pub(super) body: Vec<IrStmt>,
}

/// Resolved symbol tables for the compute body being parsed.
pub(super) struct Ctx {
    pub(super) buffers: HashMap<String, BufferSymbol>,
    pub(super) scalars: HashMap<String, ScalarTy>, // push-field name → type
    pub(super) push_param: Option<String>,         // the `push:` param name
    pub(super) id_param: Option<String>,           // the `id: global_id` param name
    pub(super) locals: HashMap<String, ScalarTy>,  // name → type (accumulated)
    pub(super) locals_order: Vec<(String, ScalarTy)>,
    pub(super) shared: HashMap<String, PendingShared>,
    pub(super) shared_order: Vec<String>,
}

#[derive(Clone, Copy)]
pub(super) struct BufferSymbol {
    pub(super) writable: bool,
    pub(super) elem: BufElem,
}

/// A workgroup-shared array as accumulated during parsing, before its `elem`
/// is interned into the module's `TypeArena`. Distinct from the public
/// `ir::SharedDecl` (whose `elem` is a `TypeId`) so parsing itself needs no
/// arena access — only the final construction in `parse_kernel_rest` does.
pub(super) struct PendingShared {
    pub(super) name: String,
    pub(super) elem: ScalarTy,
    pub(super) len: u32,
}

/// A compute entry-point parameter as accumulated during parsing, before its
/// scalar/buffer-element type is interned into the module's `TypeArena`.
/// Distinct from the public `ir::Param`/`ir::ParamKind` (whose payload is a
/// `TypeId`) so parsing itself needs no arena access — only the final
/// construction in `parse_kernel_rest` does.
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

/// A `@location` input as accumulated during parsing, before its type is
/// interned into the module's `TypeArena`. Distinct from the public
/// `ir::GfxInput` (whose `ty` is a `TypeId`).
pub(super) struct PendingGfxInput {
    pub(super) name: String,
    pub(super) location: u32,
    pub(super) ty: GfxTy,
}

/// A push-constant scalar/matrix param as accumulated during parsing, before
/// its type is interned. Distinct from the public `ir::GfxScalar`.
pub(super) struct PendingGfxScalar {
    pub(super) name: String,
    pub(super) ty: GfxTy,
}

/// Resolved symbol tables for a graphics (vertex/fragment) body.
pub(super) struct GfxCtx {
    pub(super) inputs: HashMap<String, GfxTy>, // @location input name → type
    pub(super) scalars: HashMap<String, GfxTy>, // push-field name → type
    pub(super) push_param: Option<String>,     // the push param name
    pub(super) buffers: HashMap<String, bool>, // name → writable
    pub(super) locals: HashMap<String, GfxTy>,
    pub(super) locals_order: Vec<(String, GfxTy)>,
}
