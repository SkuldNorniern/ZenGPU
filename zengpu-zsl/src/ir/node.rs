//! ZSL IR — expression and statement nodes.
//!
//! Backend-neutral and symbol-resolved: identifiers are already classified as
//! locals, scalar params, or buffers, and structural validity (buffer writes,
//! known builtins) is checked during `build`. Value-type inference and the
//! emission of conversions are left to each backend.

use crate::ir::types::TypeId;

/// A ZSL expression.
#[derive(Debug)]
pub enum IrExpr {
    /// Integer literal (`u32`-typed).
    LitU32(u32),
    /// Float literal (`f32`-typed).
    LitF32(f32),
    /// A declared local variable.
    Local(String),
    /// A push-constant scalar parameter.
    ScalarParam(String),
    /// `buf[index]` — a bindless storage-buffer element load.
    BufferLoad { buf: String, index: Box<IrExpr> },
    /// `name[index]` — an element load from workgroup-shared memory.
    SharedLoad { name: String, index: Box<IrExpr> },
    /// `global_id().{x|y|z}` → component `0`/`1`/`2` (compute only).
    GlobalId(u32),
    /// `local_id().{x|y|z}` (compute only).
    LocalId(u32),
    /// `group_id().{x|y|z}` (compute only).
    GroupId(u32),
    /// Channel `component` (0..4 for r/g/b/a) of a texel of the `texture` or
    /// `image` param `tex` (compute only). See [`TexelOp`] for `args`.
    Texel {
        tex: String,
        op: TexelOp,
        args: Vec<IrExpr>,
        component: u32,
    },
    /// `tex.width()` / `tex.height()` → `u32`: the size of mip 0 along `axis`
    /// (0 or 1) of the `texture` or `image` param `tex` (compute only).
    TexSize { tex: String, axis: u32 },
    /// A graphics `@location` vertex/fragment input.
    Input(String),
    /// Component access `expr.{x|y|z|w}` → scalar at component `0`/`1`/`2`/`3`.
    FieldAccess { base: Box<IrExpr>, component: u32 },
    /// A multi-lane swizzle `expr.xy`/`expr.rgb`/… → a vector of
    /// `lanes.len()` (2 to 4) components picked from `base` (graphics only).
    Swizzle { base: Box<IrExpr>, lanes: Vec<u32> },
    /// `[a, b, …]`: an array of same-typed values, bound by a `let`
    /// (graphics only).
    ArrayLit(Vec<IrExpr>),
    /// `array[index]`: an element of a local array (graphics only).
    Index { array: String, index: Box<IrExpr> },
    /// `table[index].sample(uv)` / `.sample_level(uv, lod)` → `f32x4`: the
    /// texture in bindless slot `index`, sampled at `uv` (graphics only).
    Sample {
        index: Box<IrExpr>,
        uv: Box<IrExpr>,
        lod: Option<Box<IrExpr>>,
    },
    /// A vector constructor `f32xN(args…)`. `dim` is `2`/`3`/`4`; `ty` is
    /// `None` immediately after parsing (a `TypeArena` isn't available deep
    /// inside expression parsing) and is filled to `Some(..)` by
    /// `frontend::parser`'s `resolve_construct_types_expr` pass before the
    /// module is returned — always `Some` by the time any backend sees it.
    Construct {
        ty: Option<TypeId>,
        dim: u8,
        args: Vec<IrExpr>,
    },
    /// A built-in math call (`abs`, `min`, `clamp`, `dot`, `extend`, …).
    Builtin { func: BuiltinFn, args: Vec<IrExpr> },
    /// Unary negation.
    Neg(Box<IrExpr>),
    /// A binary operation.
    Binary {
        op: IrBinOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
}

/// A ZSL statement.
#[derive(Debug)]
pub enum IrStmt {
    /// `let name = init;` — `name` is pre-declared in `Entry::locals`.
    Let { name: String, init: IrExpr },
    /// `name = value;`
    AssignLocal { name: String, value: IrExpr },
    /// `buf[index] = value;`
    AssignBuffer {
        buf: String,
        index: IrExpr,
        value: IrExpr,
    },
    /// `atomic_add(buf, index, value)` on a mutable device buffer.
    AtomicAdd {
        buf: String,
        index: IrExpr,
        value: IrExpr,
    },
    /// `image.store(x, y, r, g, b, a)`: one texel of the `image` param
    /// `image`; `args` are those six values (compute only).
    ImageStore { image: String, args: Vec<IrExpr> },
    /// `name[index] = value;` for workgroup-shared memory.
    AssignShared {
        name: String,
        index: IrExpr,
        value: IrExpr,
    },
    /// A workgroup execution and memory barrier.
    Barrier,
    /// `if cond { then } [else { else_ }]`
    If {
        cond: IrExpr,
        then: Vec<IrStmt>,
        else_: Option<Vec<IrStmt>>,
    },
    /// `for var in lo..hi { body }` (exclusive range).
    For {
        var: String,
        lo: IrExpr,
        hi: IrExpr,
        body: Vec<IrStmt>,
    },
    /// A bare expression evaluated for effect.
    Eval(IrExpr),
    /// `discard`: drops the fragment, writing nothing (fragment only).
    Discard,
}

/// How an [`IrExpr::Texel`] reads its texel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelOp {
    /// `tex.load(x, y)`: the texel at integer `u32` coordinates in mip 0,
    /// unfiltered. Works on a `texture` or an `image`.
    Load,
    /// `tex.sample(u, v)`: filtered by the texture's sampler at normalized
    /// `f32` coordinates, in mip 0. `texture` only.
    Sample,
    /// `tex.sample_level(u, v, lod)`: as `Sample`, at mip level `lod`.
    SampleLevel,
}

/// Built-in math functions (map to GLSL.std.450 / Metal stdlib).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFn {
    U32,
    /// `f32(x)` — a `u32` converted to `f32` (graphics only).
    F32,
    /// `step(edge, x)` — `0.0` below `edge`, else `1.0` (graphics only).
    Step,
    /// `smoothstep(lo, hi, x)` — Hermite ease between `lo` and `hi`
    /// (graphics only).
    Smoothstep,
    Abs,
    Sign,
    Exp,
    Tanh,
    Sin,
    Cos,
    Tan,
    IsNan,
    IsInf,
    IsFinite,
    Log,
    Sqrt,
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
    /// `dot(a, b)` → scalar. Formerly the dedicated `IrExpr::Dot` node.
    Dot,
    /// `vec3.extend(scalar)` → `f32x4`. Formerly the dedicated
    /// `IrExpr::Extend` node.
    Extend,
    /// `mul_add_f32(a, b, acc) -> f32` — a single-rounding fused multiply-add
    /// (`a * b + acc` computed with one rounding step, not two). Compute-only;
    /// the design doc's F32-accumulate primitive for low-precision products.
    MulAddF32,
    /// `subgroup_id() -> u32` — this thread's subgroup index within its
    /// workgroup. Compute-only, legal in any control flow (a query, not a
    /// collective — no uniformity requirement).
    SubgroupId,
    /// `subgroup_local_id() -> u32` — this thread's lane index within its
    /// subgroup. Compute-only, legal in any control flow.
    SubgroupLocalId,
    /// `subgroup_size() -> u32` — the subgroup width for this dispatch.
    /// Compute-only, legal in any control flow.
    SubgroupSize,
    /// `subgroup_any(predicate: bool) -> bool` — true if `predicate` is true
    /// for any active lane. Compute-only; a collective — must occur in
    /// subgroup-uniform control flow (`ir::uniformity`).
    SubgroupAny,
    /// `subgroup_all(predicate: bool) -> bool` — true if `predicate` is true
    /// for every active lane. Compute-only; a collective.
    SubgroupAll,
    /// `subgroup_ballot(predicate: bool) -> bitset<128>` — bit `i` is
    /// `predicate` evaluated on lane `i`; bits at or above `subgroup_size()`
    /// are zero. Compute-only; a collective.
    SubgroupBallot,
    /// `subgroup_broadcast(value: f32, lane: u32) -> f32` — returns `value`
    /// from `lane`. Compute-only; a collective — must occur in
    /// subgroup-uniform control flow (`ir::uniformity`).
    SubgroupBroadcast,
    /// `subgroup_shuffle(value: f32, lane: u32) -> f32` — returns `value`
    /// from `lane`. Compute-only; a collective.
    SubgroupShuffle,
    /// `subgroup_shuffle_xor(value: f32, mask: u32) -> f32` — returns `value`
    /// from the lane selected by XORing this lane with `mask`. Compute-only;
    /// a collective.
    SubgroupShuffleXor,
    /// `subgroup_shuffle_up(value: f32, delta: u32) -> f32` — returns `value`
    /// from a lower-numbered lane. Compute-only; a collective.
    SubgroupShuffleUp,
    /// `subgroup_shuffle_down(value: f32, delta: u32) -> f32` — returns
    /// `value` from a higher-numbered lane. Compute-only; a collective.
    SubgroupShuffleDown,
    /// `subgroup_reduce_add(value) -> typeof(value)` — all-reduces an f32,
    /// u32, or i32 value with addition. Compute-only; a collective.
    SubgroupReduceAdd,
    /// `subgroup_reduce_min(value) -> typeof(value)` — all-reduces an f32,
    /// u32, or i32 value with minimum. Compute-only; a collective.
    SubgroupReduceMin,
    /// `subgroup_reduce_max(value) -> typeof(value)` — all-reduces an f32,
    /// u32, or i32 value with maximum. Compute-only; a collective.
    SubgroupReduceMax,
    /// `subgroup_reduce_and(value: u32) -> u32` — all-reduces with bitwise
    /// AND. Compute-only; a collective.
    SubgroupReduceAnd,
    /// `subgroup_reduce_or(value: u32) -> u32` — all-reduces with bitwise OR.
    /// Compute-only; a collective.
    SubgroupReduceOr,
    /// `subgroup_scan_inclusive_add(value) -> typeof(value)` — inclusive
    /// prefix sum of an f32, u32, or i32 value. Compute-only; a collective.
    SubgroupScanInclusiveAdd,
    /// `subgroup_scan_exclusive_add(value) -> typeof(value)` — exclusive
    /// prefix sum of an f32, u32, or i32 value. Compute-only; a collective.
    SubgroupScanExclusiveAdd,
    /// `workgroup_reduce_add(value: f32) -> f32` — all-threads-in-workgroup
    /// reduce via shared memory + barriers (portable, no subgroup capability
    /// needed). Requires a power-of-two workgroup thread count.
    WorkgroupReduceAdd,
    /// `workgroup_reduce_min(value: f32) -> f32` — see `WorkgroupReduceAdd`.
    WorkgroupReduceMin,
    /// `workgroup_reduce_max(value: f32) -> f32` — see `WorkgroupReduceAdd`.
    WorkgroupReduceMax,
}

impl BuiltinFn {
    /// The ZSL source spelling, used in diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFn::U32 => "u32",
            BuiltinFn::F32 => "f32",
            BuiltinFn::Step => "step",
            BuiltinFn::Smoothstep => "smoothstep",
            BuiltinFn::Abs => "abs",
            BuiltinFn::Sign => "sign",
            BuiltinFn::Exp => "exp",
            BuiltinFn::Tanh => "tanh",
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            BuiltinFn::Tan => "tan",
            BuiltinFn::IsNan => "isnan",
            BuiltinFn::IsInf => "isinf",
            BuiltinFn::IsFinite => "isfinite",
            BuiltinFn::Log => "log",
            BuiltinFn::Sqrt => "sqrt",
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
        }
    }
}

/// Whether any statement (recursively, including nested expressions) calls
/// `subgroup_id()`/`subgroup_local_id()`/`subgroup_size()` — shared by any
/// backend that needs to conditionally declare subgroup-related
/// capabilities/inputs only for kernels that actually use one (SPIR-V's
/// `GroupNonUniform` capability + builtin input variables, MSL's
/// simdgroup-attributed kernel parameters).
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

/// Whether the statement tree calls any `workgroup_reduce_add/min/max` —
/// used to decide whether a kernel needs the extra system-synthesized
/// f32 shared-memory scratch array declared at all.
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

/// Whether the statement tree uses subgroup vote, ballot, shuffle,
/// relative-shuffle, and/or arithmetic collectives.
///
/// The returned tuple is `(uses_vote, uses_ballot, uses_shuffle,
/// uses_shuffle_relative, uses_arithmetic)`, allowing SPIR-V lowering to declare the five
/// independently-required capabilities in one walk.
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

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    /// Unsigned-integer-only: `&`, `|`, `^`, `<<`, `>>` (logical, not
    /// arithmetic, shift). SPIR-V only today; other backends reject them.
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}
