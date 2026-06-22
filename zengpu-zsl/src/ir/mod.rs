//! Backend-neutral ZSL IR — the `zen.md` semantic model.
//!
//! Built once from the front-end AST (`build`), then consumed by each backend
//! (`crate::backend::spirv` today, MSL next). Per `zen.md`: *"SPIR-V can be the
//! first backend. It should not be the language model."*
//!
//! This slice models the compute entry point. Graphics (vertex/fragment,
//! varyings, vertex input/output structs) extends `EntryKind`/`Param` next.

pub mod build;
pub mod node;

use node::IrStmt;

use crate::frontend::types::BufElem;

/// Scalar value / declaration type. Compute values are scalar today.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarTy {
    U32,
    F32,
    Bool,
}

/// Mutability of a resource parameter (`zen.md`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mutability {
    Read,
    ReadWrite,
}

/// A resolved entry-point parameter, in source declaration order.
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
}

/// What an entry-point parameter binds to.
pub enum ParamKind {
    /// `device buffer<elem>` — a bindless storage buffer. `elem`/`mutability`
    /// are carried for reflection and the MSL backend; the SPIR-V backend only
    /// handles `f32` and enforces writability in `ir::build`.
    Buffer {
        #[allow(dead_code)]
        elem: BufElem,
        #[allow(dead_code)]
        mutability: Mutability,
    },
    /// A push-constant scalar (`u32`/`f32`).
    Scalar(ScalarTy),
}

/// The kind of entry point, plus stage-specific facts.
pub enum EntryKind {
    Compute { local_size: [u32; 3] },
}

/// A parsed, resolved entry point ready for any backend.
pub struct Entry {
    pub kind: EntryKind,
    pub params: Vec<Param>,
    /// Locals in declaration order (the SPIR-V backend hoists these).
    pub locals: Vec<(String, ScalarTy)>,
    pub body: Vec<IrStmt>,
}

/// A ZSL module — one entry point for now.
pub struct Module {
    pub entry: Entry,
}
