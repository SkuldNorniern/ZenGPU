use std::collections::{HashMap, HashSet};

use crate::ir::node::IrStmt;
use crate::ir::typeck::{ComputeTypeCtx, GraphicsTypeCtx};
use crate::ir::types::TypeId;
use crate::ir::{GfxSlot, Mutability};

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
            shared: self
                .shared
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
    Texture,
    Image,
}

pub(super) struct PendingGfxInput {
    pub(super) name: String,
    pub(super) slot: GfxSlot,
    pub(super) ty: TypeId,
}

pub(super) struct PendingGfxScalar {
    pub(super) name: String,
    pub(super) ty: TypeId,
}

pub(super) struct GfxCtx {
    pub(super) is_fragment: bool,
    pub(super) inputs: HashMap<String, TypeId>,
    pub(super) scalars: HashMap<String, TypeId>,
    pub(super) push_param: Option<String>,
    pub(super) buffers: HashMap<String, bool>,
    pub(super) textures: Option<String>,
    pub(super) locals: HashMap<String, TypeId>,
    pub(super) locals_order: Vec<(String, TypeId)>,
    /// Array-literal locals, readable only as `name[i]`.
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
