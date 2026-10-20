//! Typing for compute and graphics IR, producing `TypeId`s.
//!
//! Parsers use it to store locals, params and buffer elements as `TypeId`.

use std::collections::HashMap;

use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr};
use crate::ir::types::{ArrayLen, FloatFormat, FragUse, IntWidth, Signedness, Type, TypeId};

/// Symbol tables for typing a compute expression. Standalone, not shared with the parser's `Ctx`.
#[derive(Debug, Default)]
pub struct ComputeTypeCtx {
    pub locals: HashMap<String, TypeId>,
    pub scalars: HashMap<String, TypeId>,
    pub buffers: HashMap<String, TypeId>,
}

pub fn frag_parts(arena: &TypeArena, ty: TypeId) -> Option<(TypeId, FragUse)> {
    match arena.get(ty) {
        Type::Frag { element, usage, .. } => Some((*element, *usage)),
        _ => None,
    }
}

pub fn i8_type() -> Type {
    Type::Int {
        width: IntWidth::W8,
        signedness: Signedness::Signed,
    }
}

pub fn frag16_id(arena: &mut TypeArena, element: TypeId, usage: FragUse) -> TypeId {
    arena.intern(Type::Frag {
        element,
        rows: 16,
        cols: 16,
        usage,
    })
}

pub fn type_of_compute_expr(expr: &IrExpr, ctx: &ComputeTypeCtx, arena: &mut TypeArena) -> TypeId {
    let u32_id = arena.u32_id();
    let i32_id = arena.i32_id();
    let f32_id = arena.f32_id();
    let bool_id = arena.bool_id();
    let bf16_id = arena.bf16_id();
    let f16_id = arena.f16_id();
    let bitset128_id = arena.bitset128_id();
    match expr {
        IrExpr::LitU32(_) => u32_id,
        IrExpr::LitF32(_) => f32_id,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(u32_id),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(f32_id),
        IrExpr::GlobalId(_) => u32_id,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => u32_id,
        IrExpr::BufferLoad { buf, .. } => {
            let elem = ctx.buffers.get(buf).copied().unwrap_or(f32_id);
            if elem == bf16_id || elem == f16_id {
                f32_id
            } else {
                elem
            }
        }
        IrExpr::SharedLoad { .. } => f32_id,
        IrExpr::Texel { .. } => f32_id,
        IrExpr::TexSize { .. } => u32_id,
        IrExpr::FragLoad { ty, .. } => *ty,
        IrExpr::Builtin { func, args } => match func {
            BuiltinFn::Mma => {
                let int_operands = args
                    .first()
                    .map(|a| type_of_compute_expr(a, ctx, arena))
                    .and_then(|t| frag_parts(arena, t))
                    .is_some_and(|(e, _)| arena.get(e) == &i8_type());
                let element = if int_operands { i32_id } else { f32_id };
                frag16_id(arena, element, FragUse::Acc)
            }
            BuiltinFn::FragZeroAcc => frag16_id(arena, f32_id, FragUse::Acc),
            BuiltinFn::FragSet => args
                .first()
                .map(|a| type_of_compute_expr(a, ctx, arena))
                .unwrap_or(f32_id),
            BuiltinFn::FragGet => {
                let element = args
                    .first()
                    .map(|a| type_of_compute_expr(a, ctx, arena))
                    .and_then(|t| frag_parts(arena, t))
                    .map(|(e, _)| e);
                match element {
                    Some(e) if arena.get(e) == &i8_type() => i32_id,
                    Some(e)
                        if arena.get(e)
                            == &Type::Float {
                                format: FloatFormat::F16,
                            } =>
                    {
                        f32_id
                    }
                    Some(e) => e,
                    None => f32_id,
                }
            }
            BuiltinFn::FragLen | BuiltinFn::FragRow | BuiltinFn::FragCol => u32_id,
            BuiltinFn::U32
            | BuiltinFn::SubgroupId
            | BuiltinFn::SubgroupLocalId
            | BuiltinFn::SubgroupSize => u32_id,
            BuiltinFn::Dot4I8 | BuiltinFn::Dot4IU8 => i32_id,
            BuiltinFn::Dot4U8 => u32_id,
            BuiltinFn::PkAddU16
            | BuiltinFn::PkMulU16
            | BuiltinFn::PkMadU16
            | BuiltinFn::PkAddF16
            | BuiltinFn::PkMulF16
            | BuiltinFn::PkFmaF16
            | BuiltinFn::FunnelShr
            | BuiltinFn::BytePerm => u32_id,
            BuiltinFn::IsNan
            | BuiltinFn::IsInf
            | BuiltinFn::IsFinite
            | BuiltinFn::SubgroupAny
            | BuiltinFn::SubgroupAll => bool_id,
            BuiltinFn::SubgroupBallot => bitset128_id,
            BuiltinFn::SubgroupReduceAdd
            | BuiltinFn::SubgroupReduceMin
            | BuiltinFn::SubgroupReduceMax
            | BuiltinFn::SubgroupReduceAnd
            | BuiltinFn::SubgroupReduceOr
            | BuiltinFn::SubgroupScanInclusiveAdd
            | BuiltinFn::SubgroupScanExclusiveAdd => args
                .first()
                .map(|a| type_of_compute_expr(a, ctx, arena))
                .unwrap_or(f32_id),
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
        _ => u32_id,
    }
}

/// Symbol tables for typing a graphics expression.
#[derive(Debug, Default)]
pub struct GraphicsTypeCtx {
    pub inputs: HashMap<String, TypeId>,
    pub scalars: HashMap<String, TypeId>,
    pub locals: HashMap<String, TypeId>,
}

pub fn type_of_graphics_expr(
    expr: &IrExpr,
    ctx: &GraphicsTypeCtx,
    arena: &mut TypeArena,
) -> TypeId {
    let u32_id = arena.u32_id();
    let f32_id = arena.f32_id();
    let vec2_id = arena.vec_id(2);
    let vec3_id = arena.vec_id(3);
    let vec4_id = arena.vec_id(4);
    let mat4_id = arena.mat4_id();

    match expr {
        IrExpr::LitU32(_) => u32_id,
        IrExpr::LitF32(_) => f32_id,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(f32_id),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(f32_id),
        IrExpr::Input(n) => ctx.inputs.get(n).copied().unwrap_or(f32_id),
        IrExpr::FieldAccess { .. } => f32_id,
        IrExpr::Sample { .. } => vec4_id,
        IrExpr::ArrayLit(elems) => {
            let element = elems
                .first()
                .map(|e| type_of_graphics_expr(e, ctx, arena))
                .unwrap_or(f32_id);
            arena.intern(Type::Array {
                element,
                length: ArrayLen::Const(elems.len() as u32),
            })
        }
        IrExpr::Index { array, .. } => match ctx.locals.get(array).map(|&t| arena.get(t)) {
            Some(Type::Array { element, .. }) => *element,
            _ => f32_id,
        },
        IrExpr::Swizzle { lanes, .. } => match lanes.len() {
            2 => vec2_id,
            3 => vec3_id,
            _ => vec4_id,
        },
        IrExpr::Construct { dim, .. } => match dim {
            2 => vec2_id,
            3 => vec3_id,
            _ => vec4_id,
        },
        IrExpr::BufferLoad { .. } => f32_id,
        IrExpr::GlobalId(_) => u32_id,
        IrExpr::SharedLoad { .. } | IrExpr::Texel { .. } | IrExpr::FragLoad { .. } => f32_id,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) | IrExpr::TexSize { .. } => u32_id,
        IrExpr::Builtin { func, args } => match func {
            BuiltinFn::U32 => u32_id,
            BuiltinFn::F32 | BuiltinFn::Length => f32_id,
            BuiltinFn::Step | BuiltinFn::Smoothstep => args
                .last()
                .map(|a| type_of_graphics_expr(a, ctx, arena))
                .unwrap_or(f32_id),
            BuiltinFn::Dot => f32_id,
            BuiltinFn::Extend => {
                let base_ty = args.first().map(|a| type_of_graphics_expr(a, ctx, arena));
                if base_ty == Some(vec2_id) {
                    vec3_id
                } else {
                    vec4_id
                }
            }
            _ => args
                .first()
                .map(|a| type_of_graphics_expr(a, ctx, arena))
                .unwrap_or(f32_id),
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
            | IrBinOp::Or => f32_id,
            _ => {
                let l = type_of_graphics_expr(lhs, ctx, arena);
                let r = type_of_graphics_expr(rhs, ctx, arena);
                let vec_ids = [vec2_id, vec3_id, vec4_id];
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
    use crate::ir::types::{IntWidth, Signedness, Type};

    #[test]
    fn compute_literal_and_local_types() {
        let mut arena = TypeArena::new();
        let mut ctx = ComputeTypeCtx::default();
        ctx.locals.insert("x".into(), arena.f32_id());

        let lit = type_of_compute_expr(&IrExpr::LitU32(1), &ctx, &mut arena);
        assert_eq!(
            *arena.get(lit),
            Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned
            }
        );

        let local = type_of_compute_expr(&IrExpr::Local("x".into()), &ctx, &mut arena);
        assert_eq!(local, arena.f32_id());
    }

    #[test]
    fn compute_buffer_load_narrows_bf16_to_f32() {
        let mut arena = TypeArena::new();
        let mut ctx = ComputeTypeCtx::default();
        ctx.buffers.insert("x".into(), arena.bf16_id());

        let load = IrExpr::BufferLoad {
            buf: "x".into(),
            index: Box::new(IrExpr::LitU32(0)),
        };
        let ty = type_of_compute_expr(&load, &ctx, &mut arena);
        assert_eq!(ty, arena.f32_id());
    }

    #[test]
    fn compute_buffer_load_narrows_f16_to_f32() {
        let mut arena = TypeArena::new();
        let mut ctx = ComputeTypeCtx::default();
        ctx.buffers.insert("x".into(), arena.f16_id());

        let load = IrExpr::BufferLoad {
            buf: "x".into(),
            index: Box::new(IrExpr::LitU32(0)),
        };
        let ty = type_of_compute_expr(&load, &ctx, &mut arena);
        assert_eq!(ty, arena.f32_id());
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
        assert_eq!(ty, arena.bool_id());
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
        assert_eq!(ty, arena.f32_id());
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
        assert_eq!(vec4_ty, arena.vec_id(4));

        let mat_mul = IrExpr::Binary {
            op: IrBinOp::Mul,
            lhs: Box::new(IrExpr::ScalarParam("mvp".into())),
            rhs: Box::new(vec4),
        };
        let mut ctx2 = GraphicsTypeCtx::default();
        ctx2.scalars.insert("mvp".into(), arena.mat4_id());
        let promoted = type_of_graphics_expr(&mat_mul, &ctx2, &mut arena);
        assert_eq!(promoted, arena.vec_id(4));
    }
}
