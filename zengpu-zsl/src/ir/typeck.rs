//! Type checking over already-parsed compute/graphics IR, producing
//! [`crate::ir::types::TypeId`] values.
//!
//! The compute parser uses [`type_of_compute_expr`] to type expressions and
//! store compute locals, parameters, and buffer elements natively as
//! [`crate::ir::types::TypeId`]. The graphics checker remains a standalone
//! equivalent of `crate::frontend::parser`'s private `infer_gfx_ty`; only its
//! own tests and the parser comparison test call it today.

use std::collections::HashMap;

use crate::ir::ScalarTy;
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr};
use crate::ir::types::TypeId;

/// Symbol tables needed to type a compute [`IrExpr`], mirroring the shape of
/// `crate::frontend::parser`'s private `Ctx` (which this does not reuse or
/// depend on — it is a standalone equivalent).
#[derive(Debug, Default)]
pub struct ComputeTypeCtx {
    pub locals: HashMap<String, TypeId>,
    pub scalars: HashMap<String, TypeId>,
    pub buffers: HashMap<String, TypeId>,
}

/// Type a compute expression, interning canonical scalar types into `arena`.
pub fn type_of_compute_expr(expr: &IrExpr, ctx: &ComputeTypeCtx, arena: &mut TypeArena) -> TypeId {
    let u32_id = arena.intern_scalar(ScalarTy::U32);
    let i32_id = arena.intern_scalar(ScalarTy::I32);
    let f32_id = arena.intern_scalar(ScalarTy::F32);
    let bool_id = arena.intern_scalar(ScalarTy::Bool);
    match expr {
        IrExpr::LitU32(_) => u32_id,
        IrExpr::LitF32(_) => f32_id,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(u32_id),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(f32_id),
        IrExpr::GlobalId(_) => u32_id,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => u32_id,
        IrExpr::BufferLoad { buf, .. } => ctx.buffers.get(buf).copied().unwrap_or(f32_id),
        IrExpr::SharedLoad { .. } => f32_id,
        IrExpr::Builtin { func, .. } => match func {
            BuiltinFn::U32 => u32_id,
            BuiltinFn::IsNan | BuiltinFn::IsInf | BuiltinFn::IsFinite => bool_id,
            _ => f32_id,
        },
        IrExpr::Neg(e) => type_of_compute_expr(e, ctx, arena),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => bool_id,
            _ => {
                let l = type_of_compute_expr(lhs, ctx, arena);
                let r = type_of_compute_expr(rhs, ctx, arena);
                if l == f32_id || r == f32_id {
                    f32_id
                } else if l == i32_id || r == i32_id {
                    i32_id
                } else {
                    u32_id
                }
            }
        },
        // Graphics-only nodes never appear in compute IR; same fallback as before.
        _ => u32_id,
    }
}

/// Symbol tables needed to type a graphics [`IrExpr`], mirroring the shape of
/// `crate::frontend::parser`'s private `GfxCtx`.
#[derive(Debug, Default)]
pub struct GraphicsTypeCtx {
    pub inputs: HashMap<String, crate::ir::GfxTy>,
    pub scalars: HashMap<String, crate::ir::GfxTy>,
    pub locals: HashMap<String, crate::ir::GfxTy>,
}

/// Type a graphics expression, interning the result into `arena`. Mirrors
/// `crate::frontend::parser`'s private `infer_gfx_ty` exactly.
pub fn type_of_graphics_expr(expr: &IrExpr, ctx: &GraphicsTypeCtx, arena: &mut TypeArena) -> TypeId {
    use crate::ir::GfxTy;

    match expr {
        IrExpr::LitU32(_) => arena.intern_gfx_ty(GfxTy::U32),
        IrExpr::LitF32(_) => arena.intern_gfx_ty(GfxTy::F32),
        IrExpr::Local(n) => arena.intern_gfx_ty(ctx.locals.get(n).copied().unwrap_or(GfxTy::F32)),
        IrExpr::ScalarParam(n) => {
            arena.intern_gfx_ty(ctx.scalars.get(n).copied().unwrap_or(GfxTy::F32))
        }
        IrExpr::Input(n) => arena.intern_gfx_ty(ctx.inputs.get(n).copied().unwrap_or(GfxTy::F32)),
        IrExpr::FieldAccess { .. } => arena.intern_gfx_ty(GfxTy::F32),
        IrExpr::Construct { dim, .. } => {
            let ty = match dim {
                2 => GfxTy::Vec2,
                3 => GfxTy::Vec3,
                _ => GfxTy::Vec4,
            };
            arena.intern_gfx_ty(ty)
        }
        IrExpr::BufferLoad { .. } => arena.intern_gfx_ty(GfxTy::F32),
        IrExpr::GlobalId(_) => arena.intern_gfx_ty(GfxTy::U32),
        IrExpr::SharedLoad { .. } => arena.intern_gfx_ty(GfxTy::F32),
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => arena.intern_gfx_ty(GfxTy::U32),
        IrExpr::Builtin { func, args } => match func {
            BuiltinFn::U32 => arena.intern_gfx_ty(GfxTy::U32),
            BuiltinFn::Length => arena.intern_gfx_ty(GfxTy::F32),
            BuiltinFn::Dot => arena.intern_gfx_ty(GfxTy::F32),
            BuiltinFn::Extend => {
                let base_ty = args.first().map(|a| type_of_graphics_expr(a, ctx, arena));
                let vec2_id = arena.intern_gfx_ty(GfxTy::Vec2);
                let ty = if base_ty == Some(vec2_id) { GfxTy::Vec3 } else { GfxTy::Vec4 };
                arena.intern_gfx_ty(ty)
            }
            _ => match args.first() {
                Some(a) => type_of_graphics_expr(a, ctx, arena),
                None => arena.intern_gfx_ty(GfxTy::F32),
            },
        },
        IrExpr::Neg(e) => type_of_graphics_expr(e, ctx, arena),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => arena.intern_gfx_ty(GfxTy::F32),
            _ => {
                let l = type_of_graphics_expr(lhs, ctx, arena);
                let r = type_of_graphics_expr(rhs, ctx, arena);
                let mat4_id = arena.intern_gfx_ty(GfxTy::Mat4);
                let f32_id = arena.intern_gfx_ty(GfxTy::F32);
                let u32_id = arena.intern_gfx_ty(GfxTy::U32);
                let vec_ids = [
                    arena.intern_gfx_ty(GfxTy::Vec2),
                    arena.intern_gfx_ty(GfxTy::Vec3),
                    arena.intern_gfx_ty(GfxTy::Vec4),
                ];
                let is_vec = |id: TypeId| vec_ids.contains(&id);
                if l == mat4_id && is_vec(r) {
                    r
                } else if l == r {
                    l
                } else if (l == f32_id || l == u32_id) && is_vec(r) {
                    r
                } else if (r == f32_id || r == u32_id) && is_vec(l) {
                    l
                } else if l == f32_id || r == f32_id {
                    f32_id
                } else {
                    u32_id
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::GfxTy;
    use crate::ir::types::Type;

    #[test]
    fn compute_literal_and_local_types() {
        let mut arena = TypeArena::new();
        let mut ctx = ComputeTypeCtx::default();
        ctx.locals
            .insert("x".into(), arena.intern_scalar(ScalarTy::F32));

        let lit = type_of_compute_expr(&IrExpr::LitU32(1), &ctx, &mut arena);
        assert_eq!(
            *arena.get(lit),
            Type::Int {
                width: crate::ir::types::IntWidth::W32,
                signedness: crate::ir::types::Signedness::Unsigned
            }
        );

        let local = type_of_compute_expr(&IrExpr::Local("x".into()), &ctx, &mut arena);
        assert_eq!(local, arena.intern_scalar(ScalarTy::F32));
    }

    #[test]
    fn compute_comparison_is_bool() {
        let mut arena = TypeArena::new();
        let ctx = ComputeTypeCtx::default();
        let expr = IrExpr::Binary {
            op: IrBinOp::Lt,
            lhs: Box::new(IrExpr::LitU32(1)),
            rhs: Box::new(IrExpr::LitU32(2)),
        };
        let ty = type_of_compute_expr(&expr, &ctx, &mut arena);
        assert_eq!(ty, arena.intern_scalar(ScalarTy::Bool));
    }

    #[test]
    fn compute_arithmetic_promotes_to_f32() {
        let mut arena = TypeArena::new();
        let ctx = ComputeTypeCtx::default();
        let expr = IrExpr::Binary {
            op: IrBinOp::Add,
            lhs: Box::new(IrExpr::LitU32(1)),
            rhs: Box::new(IrExpr::LitF32(2.0)),
        };
        let ty = type_of_compute_expr(&expr, &ctx, &mut arena);
        assert_eq!(ty, arena.intern_scalar(ScalarTy::F32));
    }

    #[test]
    fn graphics_vec_construct_and_mat4_vec4_promotion() {
        let mut arena = TypeArena::new();
        let ctx = GraphicsTypeCtx::default();
        let vec4 = IrExpr::Construct {
            ty: None,
            dim: 4,
            args: vec![
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(1.0),
            ],
        };
        let vec4_ty = type_of_graphics_expr(&vec4, &ctx, &mut arena);
        assert_eq!(vec4_ty, arena.intern_gfx_ty(GfxTy::Vec4));

        let mat_mul = IrExpr::Binary {
            op: IrBinOp::Mul,
            lhs: Box::new(IrExpr::ScalarParam("mvp".into())), // typed F32 by empty ctx default; see below
            rhs: Box::new(vec4),
        };
        // Force the lhs to read as Mat4 by seeding ctx.scalars.
        let mut ctx2 = GraphicsTypeCtx::default();
        ctx2.scalars.insert("mvp".into(), GfxTy::Mat4);
        let promoted = type_of_graphics_expr(&mat_mul, &ctx2, &mut arena);
        assert_eq!(promoted, arena.intern_gfx_ty(GfxTy::Vec4));
    }
}
