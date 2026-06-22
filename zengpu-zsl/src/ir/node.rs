//! ZSL IR — expression and statement nodes.
//!
//! Backend-neutral and symbol-resolved: identifiers are already classified as
//! locals, scalar params, or buffers, and structural validity (buffer writes,
//! known builtins) is checked during `build`. Value-type inference and the
//! emission of conversions are left to each backend.

/// A ZSL expression.
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
    /// `global_id().{x|y|z}` → component `0`/`1`/`2`.
    GlobalId(u32),
    /// A built-in math call (`abs`, `min`, `clamp`, …).
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFn {
    Abs,
    Sign,
    Sqrt,
    Floor,
    Ceil,
    Fract,
    Min,
    Max,
    Pow,
    Clamp,
    Mix,
}

/// Binary operators.
#[derive(Clone, Copy, PartialEq, Eq)]
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
