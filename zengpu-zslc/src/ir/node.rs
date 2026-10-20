//! Backend-neutral, symbol-resolved nodes. Identifiers are classified (local, scalar param, buffer)
//! and structure is checked in `build`. Type inference and conversions are per backend.

use crate::ir::types::TypeId;

#[derive(Debug)]
pub enum IrExpr {
    /// Typed as `u32`.
    LitU32(u32),
    /// Typed as `f32`.
    LitF32(f32),
    Local(String),
    ScalarParam(String),
    BufferLoad { buf: String, index: Box<IrExpr> },
    SharedLoad { name: String, index: Box<IrExpr> },
    /// Component 0, 1 or 2. Compute only.
    GlobalId(u32),
    LocalId(u32),
    GroupId(u32),
    /// Texel channel 0-3 (r, g, b, a). Compute only. See [`TexelOp`] for `args`.
    Texel {
        tex: String,
        op: TexelOp,
        args: Vec<IrExpr>,
        component: u32,
    },
    /// Mip 0 size along `axis` (0 is width, 1 is height). Compute only.
    TexSize { tex: String, axis: u32 },
    Input(String),
    FieldAccess { base: Box<IrExpr>, component: u32 },
    /// Picks `lanes.len()` (2-4) components from `base`. Graphics only.
    Swizzle { base: Box<IrExpr>, lanes: Vec<u32> },
    /// Values of the same type, bound by a `let`. Graphics only.
    ArrayLit(Vec<IrExpr>),
    Index { array: String, index: Box<IrExpr> },
    /// An `f32x4` sampled from the texture in slot `index`. Graphics only.
    Sample {
        index: Box<IrExpr>,
        uv: Box<IrExpr>,
        lod: Option<Box<IrExpr>>,
    },
    /// `dim` is 2, 3 or 4. `ty` is `None` after parsing and is filled by
    /// `resolve_construct_types_expr` before any backend sees it.
    Construct {
        ty: Option<TypeId>,
        dim: u8,
        args: Vec<IrExpr>,
    },
    /// `ty` is the `frag<...>` type produced. See [`FragSrc`].
    FragLoad {
        ty: TypeId,
        src: FragSrc,
        offset: Box<IrExpr>,
        ld: Box<IrExpr>,
        /// `frag_load_b_t`: B is N x K row-major, (k, n) at `offset + n * ld + k`.
        trans: bool,
    },
    Builtin { func: BuiltinFn, args: Vec<IrExpr> },
    Neg(Box<IrExpr>),
    Binary {
        op: IrBinOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
}

#[derive(Debug)]
pub enum IrStmt {
    /// `name` is pre-declared in `Entry::locals`.
    Let { name: String, init: IrExpr },
    AssignLocal { name: String, value: IrExpr },
    AssignBuffer {
        buf: String,
        index: IrExpr,
        value: IrExpr,
    },
    /// Mutable device buffers only.
    AtomicAdd {
        buf: String,
        index: IrExpr,
        value: IrExpr,
    },
    /// Six values: x, y, r, g, b and a. Compute only.
    ImageStore { image: String, args: Vec<IrExpr> },
    AssignShared {
        name: String,
        index: IrExpr,
        value: IrExpr,
    },
    Barrier,
    If {
        cond: IrExpr,
        then: Vec<IrStmt>,
        else_: Option<Vec<IrStmt>>,
    },
    /// `hi` is exclusive.
    For {
        var: String,
        lo: IrExpr,
        hi: IrExpr,
        body: Vec<IrStmt>,
    },
    /// see [`FragSrc`].
    FragStore {
        src: FragSrc,
        offset: IrExpr,
        ld: IrExpr,
        value: IrExpr,
    },
    Eval(IrExpr),
    Discard,
}

/// A buffer param or shared array. Compute only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragSrc {
    Buffer(String),
    Shared(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelOp {
    /// Unfiltered read at mip 0 with `u32` coords. Works on `texture` and `image`.
    Load,
    /// Sampler-filtered read at mip 0 with normalized `f32` coords. `texture` only.
    Sample,
    /// `Sample` at mip `lod`.
    SampleLevel,
}

/// Maps to GLSL.std.450 and the Metal stdlib.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFn {
    U32,
    /// f32 to i32 truncates, maps NaN to 0 and saturates. u32 to i32 reinterprets the bits.
    I32,
    /// Rounds to the nearest f16, ties to even. The result stays f32: overflow gives inf, and NaN
    /// stays NaN. Compute only.
    F16Round,
    F32,
    Step,
    Smoothstep,
    Abs,
    Sign,
    Exp,
    /// Approximate `exp`. Uses `__expf` on HIP and CUDA, and `exp` on SPIR-V.
    FastExp,
    /// Nearest integer, ties to even.
    RoundEven,
    Tanh,
    Sin,
    Cos,
    Tan,
    IsNan,
    IsInf,
    IsFinite,
    Log,
    Sqrt,
    /// Reciprocal square root. Uses `rsqrtf` on HIP and CUDA.
    Rsqrt,
    Floor,
    Ceil,
    Fract,
    Min,
    Max,
    Pow,
    Clamp,
    Mix,
    Normalize,
    Length,
    Dot,
    Extend,
    /// Fused `a * b + acc` with a single rounding. This is the F32-accumulate primitive for
    /// low-precision products. Compute only.
    MulAddF32,
    /// Four-lane signed byte dot product. Results wrap without saturation. Compute only.
    Dot4I8,
    /// Four-lane unsigned byte dot product, otherwise as `dot4_i8`.
    Dot4U8,
    Dot4IU8,
    /// Per lane `(a + b) mod 2^16`, with lane 0 in bits 0-15. Compute only.
    PkAddU16,
    PkMulU16,
    PkMadU16,
    /// Per-lane f16 add, rounding to nearest even.
    PkAddF16,
    PkMulF16,
    /// Per-lane fused f16 `a * b + c`.
    PkFmaF16,
    /// Approximate. The error is within 1e-4 of the largest term.
    Dot2F16,
    /// low 32 bits of `(hi:lo) >> (s & 31)`.
    FunnelShr,
    /// Like `__byte_perm` without sign mode. Byte i is picked by selector nibble i from `b:a`.
    BytePerm,
    /// Low 16 bits read as f16, widened exactly. Compute only.
    F16Lo,
    /// High 16 bits read as f16, widened exactly.
    F16Hi,
    /// Packs `lo` into the low half, rounded to f16 with ties to even. Compute only.
    PackF16x2,
    BitcastF32,
    /// overload by argument type.
    BitcastU32,
    BitcastI32,
    /// Byte `i & 3`, zero extended.
    Byte,
    /// Byte `i & 3`, sign extended.
    SByte,
    /// Both arms are evaluated, with no branch. The types must match.
    Select,
    /// Subgroup index within the workgroup. A query, not a collective, so it is legal in any
    /// control flow.
    SubgroupId,
    /// Lane index within the subgroup. A query, legal in any control flow.
    SubgroupLocalId,
    SubgroupSize,
    /// A collective. It must occur in subgroup-uniform control flow (`ir::uniformity`).
    SubgroupAny,
    /// A collective. Compute only.
    SubgroupAll,
    /// Bit i is `predicate` on lane i. Bits at or above `subgroup_size()` are zero.
    ///
    /// A collective. Compute only.
    SubgroupBallot,
    /// Returns `value` from `lane`. A collective, which needs subgroup-uniform control
    /// flow (`ir::uniformity`).
    SubgroupBroadcast,
    /// A collective. Compute only.
    SubgroupShuffle,
    /// Returns `value` from lane `this ^ mask`. A collective. Compute only.
    SubgroupShuffleXor,
    /// Returns `value` from a lower lane. A collective. Compute only.
    SubgroupShuffleUp,
    /// Returns `value` from a higher lane. A collective. Compute only.
    SubgroupShuffleDown,
    /// All-reduce add over f32, u32 and i32. A collective. Compute only.
    SubgroupReduceAdd,
    /// All-reduce min over f32, u32 and i32. A collective. Compute only.
    SubgroupReduceMin,
    /// All-reduce max over f32, u32 and i32. A collective. Compute only.
    SubgroupReduceMax,
    /// All-reduce bitwise AND. A collective. Compute only.
    SubgroupReduceAnd,
    /// All-reduce bitwise OR. A collective. Compute only.
    SubgroupReduceOr,
    /// Inclusive prefix sum over f32, u32 and i32. A collective. Compute only.
    SubgroupScanInclusiveAdd,
    /// Exclusive prefix sum over f32, u32 and i32. A collective. Compute only.
    SubgroupScanExclusiveAdd,
    /// Workgroup-wide reduction over all threads, via shared memory and barriers. Needs no
    /// subgroup capability.
    ///
    /// Requires a power-of-two thread count.
    WorkgroupReduceAdd,
    WorkgroupReduceMin,
    WorkgroupReduceMax,
    /// A zero fragment. Its type comes from the `let` annotation or the `mma` accumulator slot.
    FragZeroAcc,
    Mma,
    /// `frag_len(f) -> u32` gives the number of elements each lane holds of `f`.
    FragLen,
    FragRow,
    FragCol,
    FragGet,
    FragSet,
}

impl BuiltinFn {
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFn::U32 => "u32",
            BuiltinFn::F32 => "f32",
            BuiltinFn::Step => "step",
            BuiltinFn::Smoothstep => "smoothstep",
            BuiltinFn::Abs => "abs",
            BuiltinFn::Sign => "sign",
            BuiltinFn::Exp => "exp",
            BuiltinFn::FastExp => "fast_exp",
            BuiltinFn::RoundEven => "round_even",
            BuiltinFn::Tanh => "tanh",
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            BuiltinFn::Tan => "tan",
            BuiltinFn::IsNan => "isnan",
            BuiltinFn::IsInf => "isinf",
            BuiltinFn::IsFinite => "isfinite",
            BuiltinFn::Log => "log",
            BuiltinFn::Sqrt => "sqrt",
            BuiltinFn::Rsqrt => "rsqrt",
            BuiltinFn::Floor => "floor",
            BuiltinFn::Ceil => "ceil",
            BuiltinFn::Fract => "fract",
            BuiltinFn::Min => "min",
            BuiltinFn::Max => "max",
            BuiltinFn::Pow => "pow",
            BuiltinFn::Clamp => "clamp",
            BuiltinFn::Mix => "mix",
            BuiltinFn::Normalize => "normalize",
            BuiltinFn::Length => "length",
            BuiltinFn::Dot => "dot",
            BuiltinFn::Extend => "extend",
            BuiltinFn::MulAddF32 => "mul_add_f32",
            BuiltinFn::Dot4I8 => "dot4_i8",
            BuiltinFn::Dot4U8 => "dot4_u8",
            BuiltinFn::Dot4IU8 => "dot4_iu8",
            BuiltinFn::PkAddU16 => "pk_add_u16",
            BuiltinFn::PkMulU16 => "pk_mul_u16",
            BuiltinFn::PkMadU16 => "pk_mad_u16",
            BuiltinFn::PkAddF16 => "pk_add_f16",
            BuiltinFn::PkMulF16 => "pk_mul_f16",
            BuiltinFn::PkFmaF16 => "pk_fma_f16",
            BuiltinFn::Dot2F16 => "dot2_f16",
            BuiltinFn::FunnelShr => "funnel_shr",
            BuiltinFn::BytePerm => "byte_perm",
            BuiltinFn::F16Lo => "f16_lo",
            BuiltinFn::F16Hi => "f16_hi",
            BuiltinFn::PackF16x2 => "pack_f16x2",
            BuiltinFn::BitcastF32 => "bitcast_f32",
            BuiltinFn::BitcastU32 => "bitcast_u32",
            BuiltinFn::BitcastI32 => "bitcast_i32",
            BuiltinFn::Byte => "byte",
            BuiltinFn::SByte => "sbyte",
            BuiltinFn::Select => "select",
            BuiltinFn::I32 => "i32",
            BuiltinFn::F16Round => "f16_round",
            BuiltinFn::SubgroupId => "subgroup_id",
            BuiltinFn::SubgroupLocalId => "subgroup_local_id",
            BuiltinFn::SubgroupSize => "subgroup_size",
            BuiltinFn::SubgroupAny => "subgroup_any",
            BuiltinFn::SubgroupAll => "subgroup_all",
            BuiltinFn::SubgroupBallot => "subgroup_ballot",
            BuiltinFn::SubgroupBroadcast => "subgroup_broadcast",
            BuiltinFn::SubgroupShuffle => "subgroup_shuffle",
            BuiltinFn::SubgroupShuffleXor => "subgroup_shuffle_xor",
            BuiltinFn::SubgroupShuffleUp => "subgroup_shuffle_up",
            BuiltinFn::SubgroupShuffleDown => "subgroup_shuffle_down",
            BuiltinFn::SubgroupReduceAdd => "subgroup_reduce_add",
            BuiltinFn::SubgroupReduceMin => "subgroup_reduce_min",
            BuiltinFn::SubgroupReduceMax => "subgroup_reduce_max",
            BuiltinFn::SubgroupReduceAnd => "subgroup_reduce_and",
            BuiltinFn::SubgroupReduceOr => "subgroup_reduce_or",
            BuiltinFn::SubgroupScanInclusiveAdd => "subgroup_scan_inclusive_add",
            BuiltinFn::SubgroupScanExclusiveAdd => "subgroup_scan_exclusive_add",
            BuiltinFn::WorkgroupReduceAdd => "workgroup_reduce_add",
            BuiltinFn::WorkgroupReduceMin => "workgroup_reduce_min",
            BuiltinFn::WorkgroupReduceMax => "workgroup_reduce_max",
            BuiltinFn::FragZeroAcc => "frag_zero_acc",
            BuiltinFn::Mma => "mma",
            BuiltinFn::FragLen => "frag_len",
            BuiltinFn::FragRow => "frag_row",
            BuiltinFn::FragCol => "frag_col",
            BuiltinFn::FragGet => "frag_get",
            BuiltinFn::FragSet => "frag_set",
        }
    }
}

/// True if any statement or expression calls `subgroup_id()`,
/// `subgroup_local_id()` or `subgroup_size()`.
///
/// This decides whether a kernel declares subgroup builtins (SPIR-V `GroupNonUniform`, MSL
/// simdgroup params).
pub fn stmts_use_subgroup_query(stmts: &[IrStmt]) -> bool {
    stmts.iter().any(stmt_uses_subgroup_query)
}

fn stmt_uses_subgroup_query(stmt: &IrStmt) -> bool {
    match stmt {
        IrStmt::Let { init, .. } | IrStmt::AssignLocal { value: init, .. } | IrStmt::Eval(init) => {
            expr_uses_subgroup_query(init)
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => {
            expr_uses_subgroup_query(index) || expr_uses_subgroup_query(value)
        }
        IrStmt::ImageStore { args, .. } => args.iter().any(expr_uses_subgroup_query),
        IrStmt::FragStore {
            offset, ld, value, ..
        } => {
            expr_uses_subgroup_query(offset)
                || expr_uses_subgroup_query(ld)
                || expr_uses_subgroup_query(value)
        }
        IrStmt::Barrier | IrStmt::Discard => false,
        IrStmt::If { cond, then, else_ } => {
            expr_uses_subgroup_query(cond)
                || stmts_use_subgroup_query(then)
                || else_.as_deref().is_some_and(stmts_use_subgroup_query)
        }
        IrStmt::For { lo, hi, body, .. } => {
            expr_uses_subgroup_query(lo)
                || expr_uses_subgroup_query(hi)
                || stmts_use_subgroup_query(body)
        }
    }
}

fn expr_uses_subgroup_query(expr: &IrExpr) -> bool {
    match expr {
        IrExpr::Builtin { func, args } => {
            matches!(
                func,
                BuiltinFn::SubgroupId | BuiltinFn::SubgroupLocalId | BuiltinFn::SubgroupSize
            ) || args.iter().any(expr_uses_subgroup_query)
        }
        IrExpr::BufferLoad { index, .. } | IrExpr::SharedLoad { index, .. } => {
            expr_uses_subgroup_query(index)
        }
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } => {
            expr_uses_subgroup_query(base)
        }
        IrExpr::Construct { args, .. } | IrExpr::ArrayLit(args) => {
            args.iter().any(expr_uses_subgroup_query)
        }
        IrExpr::Index { index, .. } => expr_uses_subgroup_query(index),
        IrExpr::Sample { index, uv, lod } => {
            expr_uses_subgroup_query(index)
                || expr_uses_subgroup_query(uv)
                || lod.as_deref().is_some_and(expr_uses_subgroup_query)
        }
        IrExpr::Texel { args, .. } => args.iter().any(expr_uses_subgroup_query),
        IrExpr::FragLoad { offset, ld, .. } => {
            expr_uses_subgroup_query(offset) || expr_uses_subgroup_query(ld)
        }
        IrExpr::Neg(inner) => expr_uses_subgroup_query(inner),
        IrExpr::Binary { lhs, rhs, .. } => {
            expr_uses_subgroup_query(lhs) || expr_uses_subgroup_query(rhs)
        }
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::Local(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_)
        | IrExpr::TexSize { .. } => false,
    }
}

/// True if any `workgroup_reduce_add/min/max` is called.
///
/// This decides whether the kernel gets the f32 shared scratch array.
pub fn stmts_use_workgroup_reduce(stmts: &[IrStmt]) -> bool {
    stmts.iter().any(stmt_uses_workgroup_reduce)
}

fn stmt_uses_workgroup_reduce(stmt: &IrStmt) -> bool {
    match stmt {
        IrStmt::Let { init, .. } | IrStmt::AssignLocal { value: init, .. } | IrStmt::Eval(init) => {
            expr_uses_workgroup_reduce(init)
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => {
            expr_uses_workgroup_reduce(index) || expr_uses_workgroup_reduce(value)
        }
        IrStmt::ImageStore { args, .. } => args.iter().any(expr_uses_workgroup_reduce),
        IrStmt::FragStore {
            offset, ld, value, ..
        } => {
            expr_uses_workgroup_reduce(offset)
                || expr_uses_workgroup_reduce(ld)
                || expr_uses_workgroup_reduce(value)
        }
        IrStmt::Barrier | IrStmt::Discard => false,
        IrStmt::If { cond, then, else_ } => {
            expr_uses_workgroup_reduce(cond)
                || stmts_use_workgroup_reduce(then)
                || else_.as_deref().is_some_and(stmts_use_workgroup_reduce)
        }
        IrStmt::For { lo, hi, body, .. } => {
            expr_uses_workgroup_reduce(lo)
                || expr_uses_workgroup_reduce(hi)
                || stmts_use_workgroup_reduce(body)
        }
    }
}

fn expr_uses_workgroup_reduce(expr: &IrExpr) -> bool {
    match expr {
        IrExpr::Builtin { func, args } => {
            matches!(
                func,
                BuiltinFn::WorkgroupReduceAdd
                    | BuiltinFn::WorkgroupReduceMin
                    | BuiltinFn::WorkgroupReduceMax
            ) || args.iter().any(expr_uses_workgroup_reduce)
        }
        IrExpr::BufferLoad { index, .. } | IrExpr::SharedLoad { index, .. } => {
            expr_uses_workgroup_reduce(index)
        }
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } => {
            expr_uses_workgroup_reduce(base)
        }
        IrExpr::Construct { args, .. } | IrExpr::ArrayLit(args) => {
            args.iter().any(expr_uses_workgroup_reduce)
        }
        IrExpr::Index { index, .. } => expr_uses_workgroup_reduce(index),
        IrExpr::Sample { index, uv, lod } => {
            expr_uses_workgroup_reduce(index)
                || expr_uses_workgroup_reduce(uv)
                || lod.as_deref().is_some_and(expr_uses_workgroup_reduce)
        }
        IrExpr::Texel { args, .. } => args.iter().any(expr_uses_workgroup_reduce),
        IrExpr::FragLoad { offset, ld, .. } => {
            expr_uses_workgroup_reduce(offset) || expr_uses_workgroup_reduce(ld)
        }
        IrExpr::Neg(inner) => expr_uses_workgroup_reduce(inner),
        IrExpr::Binary { lhs, rhs, .. } => {
            expr_uses_workgroup_reduce(lhs) || expr_uses_workgroup_reduce(rhs)
        }
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::Local(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_)
        | IrExpr::TexSize { .. } => false,
    }
}

/// Returns the (vote, ballot, shuffle, shuffle_relative, arithmetic) flags from one walk over
/// the module.
pub fn stmts_use_subgroup_collectives(stmts: &[IrStmt]) -> (bool, bool, bool, bool, bool) {
    stmts
        .iter()
        .fold((false, false, false, false, false), |acc, stmt| {
            merge_subgroup_collective_uses(acc, stmt_uses_subgroup_collectives(stmt))
        })
}

fn merge_subgroup_collective_uses(
    a: (bool, bool, bool, bool, bool),
    b: (bool, bool, bool, bool, bool),
) -> (bool, bool, bool, bool, bool) {
    (a.0 || b.0, a.1 || b.1, a.2 || b.2, a.3 || b.3, a.4 || b.4)
}

fn stmt_uses_subgroup_collectives(stmt: &IrStmt) -> (bool, bool, bool, bool, bool) {
    match stmt {
        IrStmt::Let { init, .. } | IrStmt::AssignLocal { value: init, .. } | IrStmt::Eval(init) => {
            expr_uses_subgroup_collectives(init)
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => merge_subgroup_collective_uses(
            expr_uses_subgroup_collectives(index),
            expr_uses_subgroup_collectives(value),
        ),
        IrStmt::ImageStore { args, .. } => args
            .iter()
            .fold((false, false, false, false, false), |acc, e| {
                merge_subgroup_collective_uses(acc, expr_uses_subgroup_collectives(e))
            }),
        IrStmt::FragStore {
            offset, ld, value, ..
        } => merge_subgroup_collective_uses(
            merge_subgroup_collective_uses(
                expr_uses_subgroup_collectives(offset),
                expr_uses_subgroup_collectives(ld),
            ),
            expr_uses_subgroup_collectives(value),
        ),
        IrStmt::Barrier | IrStmt::Discard => (false, false, false, false, false),
        IrStmt::If { cond, then, else_ } => {
            let mut uses = merge_subgroup_collective_uses(
                expr_uses_subgroup_collectives(cond),
                stmts_use_subgroup_collectives(then),
            );
            if let Some(else_block) = else_ {
                uses = merge_subgroup_collective_uses(
                    uses,
                    stmts_use_subgroup_collectives(else_block),
                );
            }
            uses
        }
        IrStmt::For { lo, hi, body, .. } => merge_subgroup_collective_uses(
            merge_subgroup_collective_uses(
                expr_uses_subgroup_collectives(lo),
                expr_uses_subgroup_collectives(hi),
            ),
            stmts_use_subgroup_collectives(body),
        ),
    }
}

fn expr_uses_subgroup_collectives(expr: &IrExpr) -> (bool, bool, bool, bool, bool) {
    match expr {
        IrExpr::Builtin { func, args } => {
            let own = (
                matches!(func, BuiltinFn::SubgroupAny | BuiltinFn::SubgroupAll),
                matches!(
                    func,
                    BuiltinFn::SubgroupBallot | BuiltinFn::SubgroupBroadcast
                ),
                matches!(
                    func,
                    BuiltinFn::SubgroupShuffle | BuiltinFn::SubgroupShuffleXor
                ),
                matches!(
                    func,
                    BuiltinFn::SubgroupShuffleUp | BuiltinFn::SubgroupShuffleDown
                ),
                matches!(
                    func,
                    BuiltinFn::SubgroupReduceAdd
                        | BuiltinFn::SubgroupReduceMin
                        | BuiltinFn::SubgroupReduceMax
                        | BuiltinFn::SubgroupReduceAnd
                        | BuiltinFn::SubgroupReduceOr
                        | BuiltinFn::SubgroupScanInclusiveAdd
                        | BuiltinFn::SubgroupScanExclusiveAdd
                ),
            );
            args.iter().fold(own, |acc, arg| {
                merge_subgroup_collective_uses(acc, expr_uses_subgroup_collectives(arg))
            })
        }
        IrExpr::BufferLoad { index, .. } | IrExpr::SharedLoad { index, .. } => {
            expr_uses_subgroup_collectives(index)
        }
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } => {
            expr_uses_subgroup_collectives(base)
        }
        IrExpr::Construct { args, .. } => args
            .iter()
            .fold((false, false, false, false, false), |acc, arg| {
                merge_subgroup_collective_uses(acc, expr_uses_subgroup_collectives(arg))
            }),
        IrExpr::Neg(inner) | IrExpr::Index { index: inner, .. } => {
            expr_uses_subgroup_collectives(inner)
        }
        IrExpr::Texel { args: elems, .. } | IrExpr::ArrayLit(elems) => elems
            .iter()
            .fold((false, false, false, false, false), |acc, e| {
                merge_subgroup_collective_uses(acc, expr_uses_subgroup_collectives(e))
            }),
        IrExpr::Sample { index, uv, lod } => {
            let uses = merge_subgroup_collective_uses(
                expr_uses_subgroup_collectives(index),
                expr_uses_subgroup_collectives(uv),
            );
            match lod {
                Some(lod) => {
                    merge_subgroup_collective_uses(uses, expr_uses_subgroup_collectives(lod))
                }
                None => uses,
            }
        }
        IrExpr::FragLoad { offset, ld, .. } => merge_subgroup_collective_uses(
            expr_uses_subgroup_collectives(offset),
            expr_uses_subgroup_collectives(ld),
        ),
        IrExpr::Binary { lhs, rhs, .. } => merge_subgroup_collective_uses(
            expr_uses_subgroup_collectives(lhs),
            expr_uses_subgroup_collectives(rhs),
        ),
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::Local(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_)
        | IrExpr::TexSize { .. } => (false, false, false, false, false),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// The sign follows the dividend. Both operands must have the same type, as for `/`.
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    /// Unsigned integers only. `>>` is a logical shift, not an arithmetic one.
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}
