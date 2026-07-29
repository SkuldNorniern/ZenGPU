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
    /// A graphics `@location` vertex/fragment input.
    Input(String),
    /// Component access `expr.{x|y|z|w}` → scalar at component `0`/`1`/`2`/`3`.
    FieldAccess { base: Box<IrExpr>, component: u32 },
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
}

/// Built-in math functions (map to GLSL.std.450 / Metal stdlib).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFn {
    U32,
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
}

impl BuiltinFn {
    /// The ZSL source spelling, used in diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFn::U32 => "u32",
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
        IrStmt::Barrier => false,
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
        IrExpr::FieldAccess { base, .. } => expr_uses_subgroup_query(base),
        IrExpr::Construct { args, .. } => args.iter().any(expr_uses_subgroup_query),
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
        | IrExpr::Input(_) => false,
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
