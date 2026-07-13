//! Backend-neutral, symbol-resolved nodes. Identifiers are classified (local, scalar param, buffer)
//! and structure is checked in `build`. Type inference and conversions are per backend.

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
    /// A vector constructor `f32xN(args…)` (`dim` is `2`/`3`/`4`).
    VecConstruct { dim: u8, args: Vec<IrExpr> },
    /// `vec3.extend(scalar)` returns an `f32x4`.
    Extend {
        base: Box<IrExpr>,
        scalar: Box<IrExpr>,
    },
    /// `dot(a, b)` returns a scalar.
    Dot { a: Box<IrExpr>, b: Box<IrExpr> },
    /// A built-in math call, such as `abs`, `min` or `clamp`.
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
    Abs,
    Sign,
    Exp,
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
}

impl BuiltinFn {
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFn::Abs => "abs",
            BuiltinFn::Sign => "sign",
            BuiltinFn::Exp => "exp",
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
}
