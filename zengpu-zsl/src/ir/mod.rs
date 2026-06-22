//! Backend-neutral ZSL IR. Built once from the AST, consumed by each backend. Covers compute and
//! graphics entries.

pub mod build;
pub mod node;

use node::IrStmt;

use crate::frontend::types::BufElem;

/// Scalar value / declaration type. Compute values are scalar only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarTy {
    U32,
    F32,
    Bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mutability {
    Read,
    ReadWrite,
}

pub struct Param {
    pub name: String,
    pub kind: ParamKind,
}

pub enum ParamKind {
    /// Bindless storage buffer (`device buffer<elem>`).
    ///
    /// SPIR-V supports only `f32`. Writability is checked in `ir::build`. `elem` and `mutability`
    /// are kept for reflection and MSL.
    Buffer {
        #[allow(dead_code)]
        elem: BufElem,
        #[allow(dead_code)]
        mutability: Mutability,
    },
    /// A push-constant scalar of type `u32` or `f32`.
    Scalar(ScalarTy),
}

pub enum EntryKind {
    Compute { local_size: [u32; 3] },
}

pub struct Entry {
    pub kind: EntryKind,
    pub params: Vec<Param>,
    /// Local variables in declaration order. The SPIR-V backend hoists them to the entry block.
    pub locals: Vec<(String, ScalarTy)>,
    pub body: Vec<IrStmt>,
}

pub struct Module {
    pub entry: Entry,
}
