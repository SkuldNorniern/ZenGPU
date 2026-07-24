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

/// Infer the scalar type of an expression (compute is scalar-only). Mirrors the
/// SPIR-V backend's value-type inference so a `let`'s declared type matches.
pub(super) fn infer_ty(expr: &IrExpr, ctx: &Ctx) -> ScalarTy {
    match expr {
        IrExpr::LitU32(_) => ScalarTy::U32,
        IrExpr::LitF32(_) => ScalarTy::F32,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(ScalarTy::U32),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(ScalarTy::F32),
        IrExpr::GlobalId(_) => ScalarTy::U32,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => ScalarTy::U32,
        IrExpr::BufferLoad { buf, .. } => ctx
            .buffers
            .get(buf)
            .map(|buffer| scalar_for_buffer(buffer.elem))
            .unwrap_or(ScalarTy::F32),
        IrExpr::SharedLoad { .. } => ScalarTy::F32,
        IrExpr::Builtin { func, .. } => match func {
            BuiltinFn::U32 => ScalarTy::U32,
            BuiltinFn::IsNan | BuiltinFn::IsInf | BuiltinFn::IsFinite => ScalarTy::Bool,
            _ => ScalarTy::F32,
        },
        IrExpr::Neg(e) => infer_ty(e, ctx),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => ScalarTy::Bool,
            _ => {
                let l = infer_ty(lhs, ctx);
                let r = infer_ty(rhs, ctx);
                if l == ScalarTy::F32 || r == ScalarTy::F32 {
                    ScalarTy::F32
                } else if l == ScalarTy::I32 || r == ScalarTy::I32 {
                    ScalarTy::I32
                } else {
                    ScalarTy::U32
                }
            }
        },
        // Graphics-only IR nodes are never produced by the compute parser.
        _ => ScalarTy::U32,
    }
}

pub(super) fn scalar_for_buffer(elem: BufElem) -> ScalarTy {
    match elem {
        BufElem::F32 => ScalarTy::F32,
        BufElem::U32 => ScalarTy::U32,
        BufElem::I32 => ScalarTy::I32,
        _ => unreachable!("non-scalar buffer element reached compute parser"),
    }
}

pub(super) fn buffer_elem_name(elem: BufElem) -> &'static str {
    match elem {
        BufElem::F32 => "f32",
        BufElem::U32 => "u32",
        BufElem::I32 => "i32",
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

/// Infer a graphics expression's type, mirroring the SPIR-V backend's `GVal`
/// inference so a `let`'s declared type matches what the backend computes.
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
            | IrBinOp::Or => GfxTy::F32, // invalid in value position; fallback
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
