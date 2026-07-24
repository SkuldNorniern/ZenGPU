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
    /// Superseded by [`IrExpr::Construct`]; the parser no longer constructs
    /// this. Kept until a follow-up commit confirms no remaining consumers.
    VecConstruct { dim: u8, args: Vec<IrExpr> },
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
    /// Superseded by `Builtin { func: BuiltinFn::Extend, .. }`; the parser no
    /// longer constructs this. Kept until a follow-up commit confirms no
    /// remaining consumers.
    Extend {
        base: Box<IrExpr>,
        scalar: Box<IrExpr>,
    },
    /// Superseded by `Builtin { func: BuiltinFn::Dot, .. }`; the parser no
    /// longer constructs this. Kept until a follow-up commit confirms no
    /// remaining consumers.
    Dot { a: Box<IrExpr>, b: Box<IrExpr> },
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
        }
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
