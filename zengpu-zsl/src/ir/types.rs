//! The unified ZSL semantic type model (`zen.md`/design-decision §2.1).
//!
//! `Type` is the single value-type universe for both compute scalars and
//! graphics vectors/matrices/buffer element types. The parser and every
//! backend hold and compare [`TypeId`]s natively; [`crate::ir::arena::
//! TypeArena`] is the only place `Type` values are constructed or inspected.

/// An interned type. Obtained only from a [`crate::ir::arena::TypeArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub(crate) u32);

/// Identifies a nominal struct type. Unused until user-defined structs land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub(crate) u32);

/// Identifies a compile-time constant, used by [`ArrayLen::Specialization`].
/// Unused until specialization constants land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(pub(crate) u32);

// Full aspirational type model added ahead of its consumers; several variants aren't constructible yet.
/// Integer bit width.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntWidth {
    W8,
    W16,
    W32,
    W64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// Floating-point storage/arithmetic format.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatFormat {
    F16,
    Bf16,
    F32,
    F64,
    Fp8E4M3,
    Fp8E5M2,
}

/// ZSL memory/address space (`zen/plan.md`: "Memory space should be explicit
/// where it affects behavior").
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    Private,
    Workgroup,
    Device,
    Uniform,
    Push,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixMajor {
    Column,
    Row,
}

/// Array length: a fixed constant, a specialization-constant-dependent size,
/// or unbounded (the trailing member of a storage buffer).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArrayLen {
    Const(u32),
    Specialization(ConstId),
    Runtime,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDim {
    D1,
    D2,
    D3,
    Cube,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureClass {
    Sampled,
    Storage { access: Access },
    Depth,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerKind {
    Filtering,
    NonFiltering,
    Comparison,
}

/// The unified ZSL type. Recursive members refer to an interned [`TypeId`]
/// rather than boxing `Type` directly, so `Type` itself stays a plain,
/// hashable, structurally-comparable value — see
/// [`crate::ir::arena::TypeArena`] for how instances of this enum are
/// created and deduplicated.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Bool,
    Int {
        width: IntWidth,
        signedness: Signedness,
    },
    Float {
        format: FloatFormat,
    },
    Vector {
        element: TypeId,
        lanes: u8,
    },
    Matrix {
        element: TypeId,
        rows: u8,
        columns: u8,
        major: MatrixMajor,
    },
    Array {
        element: TypeId,
        length: ArrayLen,
    },
    Struct(StructId),
    Bitset {
        bits: u32,
    },
    Ref {
        space: AddressSpace,
        access: Access,
        pointee: TypeId,
    },
    Buffer {
        space: AddressSpace,
        access: Access,
        element: TypeId,
    },
    Texture {
        dim: TextureDim,
        arrayed: bool,
        multisampled: bool,
        class: TextureClass,
        texel: TypeId,
    },
    Sampler(SamplerKind),
}
