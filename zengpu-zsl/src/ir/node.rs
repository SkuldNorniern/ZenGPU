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
    Input(String),
    FieldAccess { base: Box<IrExpr>, component: u32 },
    /// `dim` is 2, 3 or 4. `ty` is `None` after parsing and is filled by
    /// `resolve_construct_types_expr` before any backend sees it.
    Construct {
        ty: Option<TypeId>,
        dim: u8,
        args: Vec<IrExpr>,
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
    Eval(IrExpr),
}

/// Maps to GLSL.std.450 and the Metal stdlib.
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
    Dot,
    Extend,
}

impl BuiltinFn {
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
    /// Unsigned-only: `&`, `|`, `^`, `<<`, `>>` (logical shift). SPIR-V only; other backends
    /// reject them.
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}
