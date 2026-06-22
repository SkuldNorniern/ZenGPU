//! Backend-neutral ZSL IR. Built once from the AST, consumed by each backend. Covers compute and
//! graphics entries.

pub mod build;
pub mod node;

use node::{IrExpr, IrStmt};

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

#[derive(Debug)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub enum EntryKind {
    Compute { local_size: [u32; 3] },
}

#[derive(Debug)]
pub struct Entry {
    pub kind: EntryKind,
    pub params: Vec<Param>,
    /// Local variables in declaration order. The SPIR-V backend hoists them to the entry block.
    pub locals: Vec<(String, ScalarTy)>,
    pub body: Vec<IrStmt>,
}

#[derive(Debug)]
pub struct Module {
    pub entry: Entry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxTy {
    F32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

#[derive(Debug)]
pub struct GfxInput {
    pub name: String,
    pub location: u32,
    pub ty: GfxTy,
}

#[derive(Debug)]
pub struct GfxScalar {
    pub name: String,
    pub ty: GfxTy,
}

#[derive(Debug)]
pub struct GraphicsEntry {
    pub is_fragment: bool,
    /// `@location` inputs. The backend sorts them by location.
    pub inputs: Vec<GfxInput>,
    pub scalar_params: Vec<GfxScalar>,
    /// Names of the `buffer<f32>` params. The parser checks writability.
    pub buf_params: Vec<String>,
    /// Vertex varying output types. Empty for fragment and position-only vertex stages.
    pub varyings: Vec<GfxTy>,
    pub locals: Vec<(String, GfxTy)>,
    pub body: Vec<IrStmt>,
    /// Vertex: `[position, varyings…]`. Fragment: `[color]`.
    pub ret: Vec<IrExpr>,
}

#[derive(Debug)]
pub struct GraphicsModule {
    pub entry: GraphicsEntry,
}
