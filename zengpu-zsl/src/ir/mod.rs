//! Backend-neutral ZSL IR. Built once from the AST, consumed by each backend. Covers compute and
//! graphics entries.

pub mod arena;
pub mod node;
pub mod typeck;
pub mod types;
pub mod validate;

use arena::TypeArena;
use node::{IrExpr, IrStmt};
use types::TypeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BufElem {
    F32,
    I32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
}

/// Scalar value / declaration type. Compute values are scalar only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarTy {
    U32,
    I32,
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
    /// `device buffer<elem>`: bindless storage buffer. `elem` is a `TypeId` in the module's `types`
    /// arena. `mutability` is for reflection and MSL. SPIR-V only handles `f32` and checks
    /// writability in `ir::build`.
    Buffer {
        elem: TypeId,
        mutability: Mutability,
    },
    /// Push-constant scalar (`u32`/`f32`). Backends convert with `scalar_ty_of`.
    Scalar(TypeId),
}

#[derive(Debug)]
pub enum EntryKind {
    Compute { local_size: [u32; 3] },
}

#[derive(Debug)]
pub struct Entry {
    pub kind: EntryKind,
    pub params: Vec<Param>,
    pub shared: Vec<SharedDecl>,
    /// Locals in declaration order (SPIR-V hoists them). Ids are in `types`; backends convert
    /// with `scalar_ty_of`.
    pub locals: Vec<(String, TypeId)>,
    pub body: Vec<IrStmt>,
}

#[derive(Debug)]
pub struct SharedDecl {
    pub name: String,
    /// Interned in `Module::types`.
    ///
    /// Not yet read by any backend, which all hardcode f32 shared memory. Meant for future non-f32
    /// shared arrays.
    pub elem: TypeId,
    pub len: u32,
}

#[derive(Debug)]
pub struct Module {
    pub entry: Entry,
    pub types: TypeArena,
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
