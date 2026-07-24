//! A standalone type checker over already-parsed compute/graphics IR,
//! computing [`crate::ir::types::TypeId`] instead of the legacy `ScalarTy`/
//! `GfxTy` enums.
//!
//! This is a genuinely independent reimplementation of the type-inference
//! logic in `crate::frontend::parser`'s private `infer_ty`/`infer_gfx_ty`
//! (used internally by the parser to type `let` locals) — not a wrapper
//! around them. `crate::frontend::parser`'s own test module has comparison
//! tests proving both agree on representative expressions. Nothing outside
//! this module's own tests and those comparison tests calls the functions
//! here yet: the parser, `Entry`/`GraphicsEntry`, and every backend are
//! unaffected and keep using `ScalarTy`/`GfxTy` exactly as today.

use std::collections::HashMap;

use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr};
use crate::ir::types::{Type, TypeId};
use crate::ir::{BufElem, ScalarTy};

/// Symbol tables needed to type a compute [`IrExpr`], mirroring the shape of
/// `crate::frontend::parser`'s private `Ctx` (which this does not reuse or
/// depend on — it is a standalone equivalent).
#[derive(Debug, Default)]
pub struct ComputeTypeCtx {
    pub locals: HashMap<String, ScalarTy>,
    pub scalars: HashMap<String, ScalarTy>,
    pub buffers: HashMap<String, BufElem>,
}

/// Type a compute expression, interning the result into `arena`. Mirrors
/// `crate::frontend::parser`'s private `infer_ty` exactly; see that
/// function's doc reference in this module's own doc comment.
pub fn type_of_compute_expr(expr: &IrExpr, ctx: &ComputeTypeCtx, arena: &mut TypeArena) -> TypeId {
    match expr {
        IrExpr::LitU32(_) => arena.intern_scalar(ScalarTy::U32),
        IrExpr::LitF32(_) => arena.intern_scalar(ScalarTy::F32),
        IrExpr::Local(n) => {
            let ty = ctx.locals.get(n).copied().unwrap_or(ScalarTy::U32);
            arena.intern_scalar(ty)
        }
        IrExpr::ScalarParam(n) => {
            let ty = ctx.scalars.get(n).copied().unwrap_or(ScalarTy::F32);
            arena.intern_scalar(ty)
        }
        IrExpr::GlobalId(_) => arena.intern_scalar(ScalarTy::U32),
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => arena.intern_scalar(ScalarTy::U32),
        IrExpr::BufferLoad { buf, .. } => {
            let ty = match ctx.buffers.get(buf) {
                Some(elem) => scalar_for_buffer(*elem),
                None => ScalarTy::F32,
            };
            arena.intern_scalar(ty)
        }
        IrExpr::SharedLoad { .. } => arena.intern_scalar(ScalarTy::F32),
        IrExpr::Builtin { func, .. } => {
            let ty = match func {
                BuiltinFn::U32 => ScalarTy::U32,
                BuiltinFn::IsNan | BuiltinFn::IsInf | BuiltinFn::IsFinite => ScalarTy::Bool,
                _ => ScalarTy::F32,
            };
            arena.intern_scalar(ty)
        }
        IrExpr::Neg(e) => type_of_compute_expr(e, ctx, arena),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => arena.intern_scalar(ScalarTy::Bool),
            _ => {
                let l = type_of_compute_expr(lhs, ctx, arena);
                let r = type_of_compute_expr(rhs, ctx, arena);
                let f32_id = arena.intern_scalar(ScalarTy::F32);
                let i32_id = arena.intern_scalar(ScalarTy::I32);
                if l == f32_id || r == f32_id {
                    f32_id
                } else if l == i32_id || r == i32_id {
                    i32_id
                } else {
                    arena.intern_scalar(ScalarTy::U32)
                }
            }
        },
        // Graphics-only nodes never appear in compute IR; same fallback as
        // the private `infer_ty`.
        _ => arena.intern_scalar(ScalarTy::U32),
    }
}

/// Mirrors `crate::frontend::parser`'s private `scalar_for_buffer`: only
/// scalar buffer element types are valid in a compute context.
fn scalar_for_buffer(elem: BufElem) -> ScalarTy {
    match elem {
        BufElem::F32 => ScalarTy::F32,
        BufElem::U32 => ScalarTy::U32,
        BufElem::I32 => ScalarTy::I32,
        _ => unreachable!("non-scalar buffer element reached compute type checker"),
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

    #[test]
    fn compute_literal_and_local_types() {
        let mut arena = TypeArena::new();
        let mut ctx = ComputeTypeCtx::default();
        ctx.locals.insert("x".into(), ScalarTy::F32);

        let lit = type_of_compute_expr(&IrExpr::LitU32(1), &ctx, &mut arena);
        assert_eq!(*arena.get(lit), Type::Int { width: crate::ir::types::IntWidth::W32, signedness: crate::ir::types::Signedness::Unsigned });

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
