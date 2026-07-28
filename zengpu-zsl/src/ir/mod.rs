//! Backend-neutral ZSL IR. Built once from the AST, consumed by each backend. Covers compute and
//! graphics entries.

pub mod arena;
pub mod binding;
pub mod capability;
pub mod layout;
pub mod node;
pub mod reflection;
pub mod typeck;
pub mod types;
pub mod validate;

use arena::TypeArena;
use node::{IrExpr, IrStmt};
use types::TypeId;

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
    /// `elem` is interned in `Module::types`. SPIR-V only handles `f32` elements. Writability is
    /// checked in `ir::build`.
    Buffer {
        elem: TypeId,
        mutability: Mutability,
    },
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
    /// In declaration order. SPIR-V hoists these declarations. TypeIds live in `Module::types`.
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

#[derive(Debug)]
pub struct GfxInput {
    pub name: String,
    pub location: u32,
    pub ty: TypeId,
}

#[derive(Debug)]
pub struct GfxScalar {
    pub name: String,
    pub ty: TypeId,
}

#[derive(Debug)]
pub struct GraphicsEntry {
    pub is_fragment: bool,
    /// `@location` inputs. The backend sorts them by location.
    pub inputs: Vec<GfxInput>,
    pub scalar_params: Vec<GfxScalar>,
    /// Names of the `buffer<f32>` params. The parser checks writability.
    pub buf_params: Vec<String>,
    /// Vertex varying outputs. Empty for fragment shaders and for position-only vertex shaders.
    pub varyings: Vec<TypeId>,
    pub locals: Vec<(String, TypeId)>,
    pub body: Vec<IrStmt>,
    /// Vertex: `[position, varyings…]`. Fragment: `[color]`.
    pub ret: Vec<IrExpr>,
}

#[derive(Debug)]
pub struct GraphicsModule {
    pub entry: GraphicsEntry,
    pub types: TypeArena,
}
