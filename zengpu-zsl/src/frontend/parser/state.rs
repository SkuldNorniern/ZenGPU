use std::collections::{HashMap, HashSet};

use crate::ir::node::IrStmt;
use crate::ir::typeck::{ComputeTypeCtx, GraphicsTypeCtx};
use crate::ir::types::TypeId;
use crate::ir::{GfxSlot, Mutability};

/// One scalar `fn` parameter: `name: ty` (in, by value) or `out name: ty`
/// (out — the call-site argument must be a plain pre-declared local; writes
/// to the parameter inside the body become writes to that local).
#[derive(Clone)]
pub(super) struct FnParamDef {
    pub(super) name: String,
    pub(super) is_out: bool,
    pub(super) ty: TypeId,
}

/// A parsed helper function, ready to be inlined by name-substitution at each
/// call site. `locals` are the function's own `let`/`for`-loop-var locals
/// (not its params), each renamed fresh per call to avoid collisions.
pub(super) struct FunctionDef {
    pub(super) params: Vec<FnParamDef>,
    pub(super) locals: Vec<(String, TypeId)>,
    pub(super) body: Vec<IrStmt>,
}

/// Resolved symbol tables for the compute body being parsed.
pub(super) struct Ctx {
    pub(super) buffers: HashMap<String, BufferSymbol>,
    pub(super) scalars: HashMap<String, TypeId>, // push-field name → type
    pub(super) push_param: Option<String>,       // the `push:` param name
    pub(super) id_param: Option<String>,         // the `id: global_id` param name
    pub(super) locals: HashMap<String, TypeId>,  // name → type (accumulated)
    pub(super) locals_order: Vec<(String, TypeId)>,
    pub(super) shared: HashMap<String, PendingShared>,
    pub(super) shared_order: Vec<String>,
    /// `texture` and `image` param names → whether it is an `image`.
    pub(super) textures: HashMap<String, bool>,
}

impl Ctx {
    pub(super) fn type_ctx(&self) -> ComputeTypeCtx {
        ComputeTypeCtx {
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

/// A workgroup-shared array as accumulated during parsing.
pub(super) struct PendingShared {
    // Mirrors `SharedDecl::name`; retained for future shared-array consumers.
    #[allow(dead_code)]
    pub(super) name: String,
    pub(super) elem: TypeId,
    pub(super) len: u32,
}

/// A compute entry-point parameter as accumulated during parsing.
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
    Texture,
    Image,
}

/// A graphics input as accumulated during parsing.
pub(super) struct PendingGfxInput {
    pub(super) name: String,
    pub(super) slot: GfxSlot,
    pub(super) ty: TypeId,
}

/// A push-constant scalar/matrix param as accumulated during parsing.
pub(super) struct PendingGfxScalar {
    pub(super) name: String,
    pub(super) ty: TypeId,
}

/// Resolved symbol tables for a graphics (vertex/fragment) body.
pub(super) struct GfxCtx {
    pub(super) is_fragment: bool,
    pub(super) inputs: HashMap<String, TypeId>, // @location input name → type
    pub(super) scalars: HashMap<String, TypeId>, // push-field name → type
    pub(super) push_param: Option<String>,      // the push param name
    pub(super) buffers: HashMap<String, bool>,  // name → writable
    pub(super) textures: Option<String>,        // the `textures` param name
    pub(super) locals: HashMap<String, TypeId>,
    pub(super) locals_order: Vec<(String, TypeId)>,
    /// Locals bound to an array literal, read only through `name[i]`.
    pub(super) arrays: HashSet<String>,
}

impl GfxCtx {
    pub(super) fn type_ctx(&self) -> GraphicsTypeCtx {
        GraphicsTypeCtx {
            inputs: self.inputs.clone(),
            scalars: self.scalars.clone(),
            locals: self.locals.clone(),
        }
    }
}
