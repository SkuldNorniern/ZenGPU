//! Backend-neutral ZSL IR — the `zen.md` semantic model.
//!
//! Built once from the front-end AST (`build`), then consumed by each backend
//! (`crate::backend::spirv` today, MSL next). Per `zen.md`: *"SPIR-V can be the
//! first backend. It should not be the language model."*
//!
//! This slice models the compute entry point. Graphics (vertex/fragment,
//! varyings, vertex input/output structs) extends `EntryKind`/`Param` next.

pub mod arena;
pub mod binding;
pub mod capability;
pub mod layout;
pub mod node;
pub mod reflection;
pub mod typeck;
pub mod types;
pub mod uniformity;
pub mod validate;

use arena::TypeArena;
use node::{IrExpr, IrStmt};
use types::TypeId;

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
    /// `device buffer<elem>` — a bindless storage buffer. `elem` is interned
    /// in the owning [`Module`]'s `types` arena. `mutability` is carried for
    /// reflection and the MSL backend; the SPIR-V backend only handles `f32`
    /// and enforces writability in `ir::build`.
    Buffer {
        elem: TypeId,
        #[allow(dead_code)] // Read by the feature-gated consumers described above.
        mutability: Mutability,
    },
    /// A push-constant scalar (`u32`/`f32`), interned in the owning
    /// [`Module`]'s `types` arena.
    Scalar(TypeId),
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
    /// Workgroup-shared arrays declared in the kernel body.
    pub shared: Vec<SharedDecl>,
    /// Locals in declaration order (the SPIR-V backend hoists these). Each
    /// [`TypeId`] is interned in the owning [`Module`]'s `types` arena.
    pub locals: Vec<(String, TypeId)>,
    pub body: Vec<IrStmt>,
}

/// A fixed-size workgroup-shared array.
#[derive(Debug)]
pub struct SharedDecl {
    pub name: String,
    /// Interned in the owning [`Module`]'s `types` arena. No backend reads
    /// this field's value today (every backend hardcodes `f32`/`float` shared
    /// memory regardless); it exists for future non-f32 shared arrays.
    #[allow(dead_code)] // Retained for the future use described above.
    pub elem: TypeId,
    pub len: u32,
}

/// A ZSL module — one entry point for now.
#[derive(Debug)]
pub struct Module {
    pub entry: Entry,
    /// Interns every [`TypeId`] referenced by `entry.locals`/`entry.shared`.
    pub types: TypeArena,
}

// ── Graphics IR ────────────────────────────────────────────────────────────────
//
// Vertex/fragment shaders work over a richer value universe (vectors, matrices)
// than compute's scalars, so they carry their own entry shape; `IrExpr`/
// `IrStmt` nodes are shared. Values are `zen.md` `Type`s throughout.

/// A vertex/fragment input: a `@location` attribute or varying, or a value
/// the pipeline supplies.
#[derive(Debug)]
pub struct GfxInput {
    pub name: String,
    pub slot: GfxSlot,
    /// Interned in the owning [`GraphicsModule`]'s `types` arena.
    pub ty: TypeId,
}

/// Where a graphics input comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxSlot {
    Location(u32),
    Builtin(GfxBuiltin),
}

/// A pipeline-supplied graphics input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxBuiltin {
    /// `vertex_index`: the vertex's index in its draw (`u32`, vertex only).
    VertexIndex,
    /// `instance_index`: the instance's index in its draw (`u32`, vertex only).
    InstanceIndex,
    /// `frag_coord`: the fragment's window position (`f32x4`, fragment only).
    FragCoord,
}

impl GfxBuiltin {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "vertex_index" => Self::VertexIndex,
            "instance_index" => Self::InstanceIndex,
            "frag_coord" => Self::FragCoord,
            _ => return None,
        })
    }

    pub fn in_fragment(self) -> bool {
        matches!(self, Self::FragCoord)
    }
}

/// A push-constant scalar/matrix param (`f32`/`u32`/`mat4x4`).
#[derive(Debug)]
pub struct GfxScalar {
    pub name: String,
    /// Interned in the owning [`GraphicsModule`]'s `types` arena.
    pub ty: TypeId,
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
    /// The `textures` param's name: the bindless texture table, sampled as
    /// `name[slot].sample(uv)`.
    pub textures: Option<String>,
    /// Vertex varying output types (empty for fragment / position-only
    /// vertex). Interned in the owning [`GraphicsModule`]'s `types` arena.
    pub varyings: Vec<TypeId>,
    /// Locals in declaration order, with inferred types.
    pub locals: Vec<(String, TypeId)>,
    /// Leading statements (everything before the tail expression).
    pub body: Vec<IrStmt>,
    /// Tail outputs: `[position]` / `[position, varyings…]` for vertex,
    /// one per [`GraphicsEntry::outputs`] for fragment.
    pub ret: Vec<IrExpr>,
    /// Fragment outputs, one per `ret` value (empty for vertex).
    pub outputs: Vec<GfxOutput>,
}

/// A fragment output: its type and the colour attachment it writes.
#[derive(Debug, Clone, Copy)]
pub struct GfxOutput {
    pub ty: TypeId,
    pub location: u32,
    /// `@blend_src(n)`: which blend source it is at its location. Source 1
    /// feeds dual-source blending.
    pub blend_src: u32,
}

/// A graphics module — one vertex or fragment entry point.
#[derive(Debug)]
pub struct GraphicsModule {
    pub entry: GraphicsEntry,
    /// Interns every [`TypeId`] referenced by `entry.inputs`/`entry.
    /// scalar_params`/`entry.varyings`/`entry.locals`.
    pub types: TypeArena,
}
