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
    /// `global_id().{x|y|z}` maps to component `0`, `1` or `2`.
    GlobalId(u32),
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
