use super::*;

pub(super) fn builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "u32" => BuiltinFn::U32,
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
        "tanh" => BuiltinFn::Tanh,
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        "tan" => BuiltinFn::Tan,
        "isnan" => BuiltinFn::IsNan,
        "isinf" => BuiltinFn::IsInf,
        "isfinite" => BuiltinFn::IsFinite,
        "log" => BuiltinFn::Log,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        _ => return None,
    })
}

pub(super) fn buffer_elem_name(elem: TypeId, arena: &crate::ir::arena::TypeArena) -> &'static str {
    use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type};
    match arena.get(elem) {
        Type::Float {
            format: FloatFormat::F32,
        } => "f32",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => "u32",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => "i32",
        _ => "unsupported",
    }
}

pub(super) fn gfx_builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
        "tanh" => BuiltinFn::Tanh,
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        "tan" => BuiltinFn::Tan,
        "log" => BuiltinFn::Log,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        "normalize" => BuiltinFn::Normalize,
        "length" => BuiltinFn::Length,
        _ => return None,
    })
}

pub(super) fn is_gfx_vec(t: GfxTy) -> bool {
    matches!(t, GfxTy::Vec2 | GfxTy::Vec3 | GfxTy::Vec4)
}

/// Graphics expression type. Must match the SPIR-V backend's `GVal` inference so `let` types agree.
pub(super) fn infer_gfx_ty(expr: &IrExpr, ctx: &GfxCtx) -> GfxTy {
    match expr {
        IrExpr::LitU32(_) => GfxTy::U32,
        IrExpr::LitF32(_) => GfxTy::F32,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::Input(n) => ctx.inputs.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::FieldAccess { .. } => GfxTy::F32,
        IrExpr::Construct { dim, .. } => match dim {
            2 => GfxTy::Vec2,
            3 => GfxTy::Vec3,
            _ => GfxTy::Vec4,
        },
        IrExpr::BufferLoad { .. } => GfxTy::F32,
        IrExpr::GlobalId(_) => GfxTy::U32,
        IrExpr::SharedLoad { .. } => GfxTy::F32,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => GfxTy::U32,
        IrExpr::Builtin { func, args } => match func {
            BuiltinFn::U32 => GfxTy::U32,
            BuiltinFn::Length => GfxTy::F32,
            BuiltinFn::Dot => GfxTy::F32,
            BuiltinFn::Extend => match args.first().map(|a| infer_gfx_ty(a, ctx)) {
                Some(GfxTy::Vec2) => GfxTy::Vec3,
                _ => GfxTy::Vec4,
            },
            _ => args
                .first()
                .map(|a| infer_gfx_ty(a, ctx))
                .unwrap_or(GfxTy::F32),
        },
        IrExpr::Neg(e) => infer_gfx_ty(e, ctx),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => GfxTy::F32,
            _ => {
                let l = infer_gfx_ty(lhs, ctx);
                let r = infer_gfx_ty(rhs, ctx);
                match (l, r) {
                    (GfxTy::Mat4, x) if is_gfx_vec(x) => x,
                    (a, b) if a == b => a,
                    (GfxTy::F32 | GfxTy::U32, x) if is_gfx_vec(x) => x,
                    (x, GfxTy::F32 | GfxTy::U32) if is_gfx_vec(x) => x,
                    _ if l == GfxTy::F32 || r == GfxTy::F32 => GfxTy::F32,
                    _ => GfxTy::U32,
                }
            }
        },
    }
}
