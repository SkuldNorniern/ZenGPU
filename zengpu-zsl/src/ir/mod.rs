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

use node::{IrExpr, IrStmt};

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
#[derive(Debug)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
}

/// What an entry-point parameter binds to.
#[derive(Debug)]
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
#[derive(Debug)]
pub enum EntryKind {
    Compute { local_size: [u32; 3] },
}

/// A parsed, resolved entry point ready for any backend.
#[derive(Debug)]
pub struct Entry {
    pub kind: EntryKind,
    pub params: Vec<Param>,
    /// Locals in declaration order (the SPIR-V backend hoists these).
    pub locals: Vec<(String, ScalarTy)>,
    pub body: Vec<IrStmt>,
}

/// A ZSL module — one entry point for now.
#[derive(Debug)]
pub struct Module {
    pub entry: Entry,
}

// ── Graphics IR ────────────────────────────────────────────────────────────────
//
// Vertex/fragment shaders work over a richer value universe (vectors, matrices)
// than compute's scalars, so they carry their own type enum and entry shape;
// `IrExpr`/`IrStmt` nodes are shared. Unifying `ScalarTy` and `GfxTy` into one
// `zen.md` `Type` is a later consolidation.

/// Graphics value / declaration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxTy {
    F32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

/// A `@location` vertex/fragment input.
#[derive(Debug)]
pub struct GfxInput {
    pub name: String,
    pub location: u32,
    pub ty: GfxTy,
}

/// A push-constant scalar/matrix param (`f32`/`u32`/`mat4x4`).
#[derive(Debug)]
pub struct GfxScalar {
    pub name: String,
    pub ty: GfxTy,
}

/// A parsed, resolved vertex or fragment entry point.
#[derive(Debug)]
pub struct GraphicsEntry {
    pub is_fragment: bool,
    /// `@location` inputs, in source declaration order (the backend sorts by location).
    pub inputs: Vec<GfxInput>,
    /// Push-constant scalar/matrix params, in declaration order.
    pub scalar_params: Vec<GfxScalar>,
    /// `buffer<f32>`/`mut buffer<f32>` param names, in declaration order
    /// (writability already enforced in the parser).
    pub buf_params: Vec<String>,
    /// Vertex varying output types (empty for fragment / position-only vertex).
    pub varyings: Vec<GfxTy>,
    /// Locals in declaration order, with inferred types.
    pub locals: Vec<(String, GfxTy)>,
    /// Leading statements (everything before the tail expression).
    pub body: Vec<IrStmt>,
    /// Tail outputs: `[position]` / `[position, varyings…]` for vertex,
    /// `[color]` for fragment.
    pub ret: Vec<IrExpr>,
}

/// A graphics module — one vertex or fragment entry point.
#[derive(Debug)]
pub struct GraphicsModule {
    pub entry: GraphicsEntry,
}
