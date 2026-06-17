//! ZSL → SPIR-V lowering for vertex and fragment shaders.
//!
//! Supports: `#[location(N)]` inputs (f32/Vec2/Vec3/Vec4), scalar push
//! constants, Vec2/Vec3/Vec4 constructors, component access (.x/.y/.z/.w),
//! and scalar arithmetic. Return type must be Vec4.

use std::collections::HashMap;

use proc_macro2::Span;
use syn::{
    BinOp, Block, Expr, ExprBinary, ExprCall, ExprField, ExprLit, ExprMethodCall, ExprPath,
    ExprTuple, Lit, Member, Stmt, spanned::Spanned,
};

use crate::ast::{ZslEntryPoint, ZslParam};
use crate::spirv::{Id, SpvBuilder, builtin, deco, sc};
use crate::types::ZslType;

// ── Public entry points ───────────────────────────────────────────────────────

pub fn lower_vertex(entry: &ZslEntryPoint, body: &Block) -> Result<Vec<u32>, (Span, String)> {
    lower_graphics(entry, body, false)
}

pub fn lower_fragment(entry: &ZslEntryPoint, body: &Block) -> Result<Vec<u32>, (Span, String)> {
    lower_graphics(entry, body, true)
}

// ── Value type ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GvTy {
    F32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

struct GVal {
    id: Id,
    ty: GvTy,
}

struct InputVar {
    var: Id,
    ty: GvTy,
    elem_ty: Id,
}

#[derive(Clone)]
struct GfxScalarInfo {
    pc_index: u32,
    ty: Id,
    gty: GvTy,
}

struct GfxLocal {
    ptr: Id,
    ty: GvTy,
    elem_ty: Id,
    #[allow(dead_code)]
    ptr_ty: Id,
}

struct GfxCtx<'a> {
    spv: &'a mut SpvBuilder,
    t_f32: Id,
    t_u32: Id,
    t_vec2: Id,
    t_vec3: Id,
    t_vec4: Id,
    t_mat4: Id,
    inputs: HashMap<String, InputVar>,
    scalar_params: HashMap<String, GfxScalarInfo>,
    pc_var: Option<Id>,
    t_ptr_pc_u32: Id,
    t_ptr_pc_f32: Id,
    t_ptr_pc_mat4: Id,
    #[allow(dead_code)]
    const_0_u32: Id,
    locals: HashMap<String, GfxLocal>,
}

impl GfxCtx<'_> {
    fn spv_elem_ty(&self, ty: GvTy) -> Id {
        match ty {
            GvTy::F32 => self.t_f32,
            GvTy::U32 => self.t_u32,
            GvTy::Vec2 => self.t_vec2,
            GvTy::Vec3 => self.t_vec3,
            GvTy::Vec4 => self.t_vec4,
            GvTy::Mat4 => self.t_mat4,
        }
    }
}

// ── Core lowering ─────────────────────────────────────────────────────────────

fn lower_graphics(
    entry: &ZslEntryPoint,
    body: &Block,
    is_fragment: bool,
) -> Result<Vec<u32>, (Span, String)> {
    let mut spv = SpvBuilder::new();

    spv.capability_shader();
    spv.memory_model_logical_glsl450();

    // ── Core types ────────────────────────────────────────────────────────────
    let t_void = spv.type_void();
    let t_f32 = spv.type_float(32);
    let t_u32 = spv.type_int(32, false);
    let t_vec2 = spv.type_vector(t_f32, 2);
    let t_vec3 = spv.type_vector(t_f32, 3);
    let t_vec4 = spv.type_vector(t_f32, 4);
    let t_mat4 = spv.type_matrix(t_vec4, 4);

    // ── Return type: parse varyings from tuple ────────────────────────────────
    // Vertex: Vec4 → position only; (Vec4, T…) → position + varyings at loc 0,1,…
    // Fragment: always Vec4.
    let ret_varyings: Vec<ZslType> = if is_fragment {
        vec![]
    } else {
        match &entry.ret {
            ZslType::Vec4 => vec![],
            ZslType::Tuple(elems) => {
                if elems.is_empty() || elems[0] != ZslType::Vec4 {
                    return Err((
                        Span::call_site(),
                        "ZSL: vertex tuple return must start with Vec4 (position)".into(),
                    ));
                }
                elems[1..].to_vec()
            }
            other => {
                return Err((
                    Span::call_site(),
                    format!(
                        "ZSL: vertex return must be Vec4 or (Vec4, …) tuple; got `{}`",
                        other.display()
                    ),
                ));
            }
        }
    };

    // ── Classify params ───────────────────────────────────────────────────────
    let loc_params: Vec<&ZslParam> = entry
        .params
        .iter()
        .filter(|p| p.location.is_some())
        .collect();
    let scalar_params: Vec<&ZslParam> = entry
        .params
        .iter()
        .filter(|p| {
            p.location.is_none() && matches!(p.ty, ZslType::U32 | ZslType::F32 | ZslType::Mat4)
        })
        .collect();

    // ── Input pointer types ───────────────────────────────────────────────────
    let t_ptr_in_f32 = spv.type_pointer(sc::INPUT, t_f32);
    let t_ptr_in_vec2 = spv.type_pointer(sc::INPUT, t_vec2);
    let t_ptr_in_vec3 = spv.type_pointer(sc::INPUT, t_vec3);
    let t_ptr_in_vec4 = spv.type_pointer(sc::INPUT, t_vec4);

    // ── Input variables ───────────────────────────────────────────────────────
    // Collect in location order so the interface list is deterministic.
    let mut loc_params_sorted = loc_params.clone();
    loc_params_sorted.sort_by_key(|p| p.location.unwrap());

    let mut input_vars: HashMap<String, InputVar> = HashMap::new();
    let mut interface: Vec<Id> = Vec::new();

    for param in &loc_params_sorted {
        let loc = param.location.unwrap();
        let gty = gv_ty_from_zsl(&param.ty).ok_or_else(|| {
            (
                Span::call_site(),
                format!(
                    "ZSL: unsupported input type `{}` for `{}`; use f32/Vec2/Vec3/Vec4",
                    param.ty.display(),
                    param.ident
                ),
            )
        })?;
        let (spv_ptr_ty, elem_ty) = match gty {
            GvTy::F32 | GvTy::U32 => (t_ptr_in_f32, t_f32),
            GvTy::Vec2 => (t_ptr_in_vec2, t_vec2),
            GvTy::Vec3 => (t_ptr_in_vec3, t_vec3),
            GvTy::Vec4 => (t_ptr_in_vec4, t_vec4),
            GvTy::Mat4 => unreachable!(), // gv_ty_from_zsl never returns Mat4 for inputs
        };
        let var = spv.global_variable(spv_ptr_ty, sc::INPUT);
        spv.decorate(var, deco::LOCATION, &[loc]);
        interface.push(var);
        input_vars.insert(
            param.ident.to_string(),
            InputVar {
                var,
                ty: gty,
                elem_ty,
            },
        );
    }

    // ── Output variable (position / fragment color) ───────────────────────────
    let t_ptr_out_vec4 = spv.type_pointer(sc::OUTPUT, t_vec4);
    let out_var = spv.global_variable(t_ptr_out_vec4, sc::OUTPUT);
    if is_fragment {
        spv.decorate(out_var, deco::LOCATION, &[0]);
    } else {
        spv.decorate(out_var, deco::BUILT_IN, &[builtin::POSITION]);
    }
    interface.push(out_var);

    // ── Varying output variables (vertex only) ────────────────────────────────
    let mut varying_out_vars: Vec<(Id, GvTy)> = Vec::new();
    for (loc, vty) in ret_varyings.iter().enumerate() {
        let gty = gv_ty_from_zsl(vty).ok_or_else(|| {
            (
                Span::call_site(),
                format!(
                    "ZSL: unsupported varying type `{}`; use f32/Vec2/Vec3/Vec4",
                    vty.display()
                ),
            )
        })?;
        let spv_elem = gvty_to_spv_id(gty, t_f32, t_u32, t_vec2, t_vec3, t_vec4);
        let t_ptr = spv.type_pointer(sc::OUTPUT, spv_elem);
        let var = spv.global_variable(t_ptr, sc::OUTPUT);
        spv.decorate(var, deco::LOCATION, &[loc as u32]);
        interface.push(var);
        varying_out_vars.push((var, gty));
    }

    // ── Push-constant block (params without location: u32/f32/Mat4) ─────────────
    let pc_var = if !scalar_params.is_empty() {
        let pc_members: Vec<Id> = scalar_params
            .iter()
            .map(|p| match p.ty {
                ZslType::U32 => t_u32,
                ZslType::F32 => t_f32,
                ZslType::Mat4 => t_mat4,
                _ => unreachable!(),
            })
            .collect();
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);
        let mut offset: u32 = 0;
        for (i, p) in scalar_params.iter().enumerate() {
            match p.ty {
                ZslType::U32 | ZslType::F32 => {
                    spv.member_decorate(t_pc_struct, i as u32, deco::OFFSET, &[offset]);
                    offset += 4;
                }
                ZslType::Mat4 => {
                    offset = (offset + 15) & !15; // align to 16
                    spv.member_decorate(t_pc_struct, i as u32, deco::OFFSET, &[offset]);
                    spv.member_decorate(t_pc_struct, i as u32, deco::COL_MAJOR, &[]);
                    spv.member_decorate(t_pc_struct, i as u32, deco::MATRIX_STRIDE, &[16]);
                    offset += 64;
                }
                _ => unreachable!(),
            }
        }
        let t_ptr_pc = spv.type_pointer(sc::PUSH_CONSTANT, t_pc_struct);
        Some(spv.global_variable(t_ptr_pc, sc::PUSH_CONSTANT))
    } else {
        None
    };

    // ── Entry point + execution mode ──────────────────────────────────────────
    let t_fn = spv.type_function(t_void, &[]);
    let fn_id = spv.fresh_id();
    let fn_name = entry.ident.to_string();
    if is_fragment {
        spv.entry_point_fragment(fn_id, &fn_name, &interface);
        spv.execution_mode_origin_upper_left(fn_id);
    } else {
        spv.entry_point_vertex(fn_id, &fn_name, &interface);
    }

    // ── Constants ─────────────────────────────────────────────────────────────
    let const_0_u32 = spv.constant_u32(t_u32, 0);

    // ── Scalar/mat param map ──────────────────────────────────────────────────
    let scalar_param_map: HashMap<String, GfxScalarInfo> = scalar_params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (ty, gty) = match p.ty {
                ZslType::U32 => (t_u32, GvTy::U32),
                ZslType::Mat4 => (t_mat4, GvTy::Mat4),
                _ => (t_f32, GvTy::F32),
            };
            (
                p.ident.to_string(),
                GfxScalarInfo {
                    pc_index: i as u32,
                    ty,
                    gty,
                },
            )
        })
        .collect();

    // ── Push-constant pointer types ───────────────────────────────────────────
    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    let mut t_ptr_pc_mat4 = Id(0);
    if pc_var.is_some() {
        for p in &scalar_params {
            match p.ty {
                ZslType::U32 if t_ptr_pc_u32 == Id(0) => {
                    t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
                }
                ZslType::F32 if t_ptr_pc_f32 == Id(0) => {
                    t_ptr_pc_f32 = spv.type_pointer(sc::PUSH_CONSTANT, t_f32);
                }
                ZslType::Mat4 if t_ptr_pc_mat4 == Id(0) => {
                    t_ptr_pc_mat4 = spv.type_pointer(sc::PUSH_CONSTANT, t_mat4);
                }
                _ => {}
            }
        }
    }

    // ── Begin function, hoist all OpVariable to entry block ───────────────────
    spv.begin_function(t_void, fn_id, t_fn);
    spv.label();

    let local_decls = collect_gfx_locals(body)?;
    let mut t_ptr_func_f32 = Id(0);
    let mut t_ptr_func_u32 = Id(0);
    let mut t_ptr_func_vec2 = Id(0);
    let mut t_ptr_func_vec3 = Id(0);
    let mut t_ptr_func_vec4 = Id(0);
    let mut locals: HashMap<String, GfxLocal> = HashMap::new();
    for (name, gty) in &local_decls {
        let elem_ty = match gty {
            GvTy::F32 => t_f32,
            GvTy::U32 => t_u32,
            GvTy::Vec2 => t_vec2,
            GvTy::Vec3 => t_vec3,
            GvTy::Vec4 => t_vec4,
            GvTy::Mat4 => {
                return Err((
                    Span::call_site(),
                    "ZSL: Mat4 local variables are not supported".into(),
                ));
            }
        };
        let ptr_ty_slot = match gty {
            GvTy::F32 => &mut t_ptr_func_f32,
            GvTy::U32 => &mut t_ptr_func_u32,
            GvTy::Vec2 => &mut t_ptr_func_vec2,
            GvTy::Vec3 => &mut t_ptr_func_vec3,
            GvTy::Vec4 => &mut t_ptr_func_vec4,
            GvTy::Mat4 => unreachable!(),
        };
        if *ptr_ty_slot == Id(0) {
            *ptr_ty_slot = spv.type_pointer(sc::FUNCTION, elem_ty);
        }
        let ptr = spv.op_variable(*ptr_ty_slot, sc::FUNCTION);
        locals.insert(
            name.clone(),
            GfxLocal {
                ptr,
                ty: *gty,
                elem_ty,
                ptr_ty: *ptr_ty_slot,
            },
        );
    }

    let mut ctx = GfxCtx {
        spv: &mut spv,
        t_f32,
        t_u32,
        t_vec2,
        t_vec3,
        t_vec4,
        t_mat4,
        inputs: input_vars,
        scalar_params: scalar_param_map,
        pc_var,
        t_ptr_pc_u32,
        t_ptr_pc_f32,
        t_ptr_pc_mat4,
        const_0_u32,
        locals,
    };

    lower_gfx_body(&mut ctx, body, out_var, &varying_out_vars)?;

    ctx.spv.op_return();
    ctx.spv.end_function();

    Ok(spv.finish())
}

// ── Body / statement lowering ─────────────────────────────────────────────────

fn lower_gfx_body(
    ctx: &mut GfxCtx<'_>,
    body: &Block,
    out_var: Id,
    varying_outs: &[(Id, GvTy)],
) -> Result<(), (Span, String)> {
    if body.stmts.is_empty() {
        return Err((
            Span::call_site(),
            "ZSL: vertex/fragment body must have a tail expression".into(),
        ));
    }
    let n = body.stmts.len();
    let (stmts, tail) = body.stmts.split_at(n - 1);
    for stmt in stmts {
        lower_gfx_stmt(ctx, stmt)?;
    }
    match &tail[0] {
        Stmt::Expr(Expr::Tuple(ExprTuple { elems, .. }), None) if !varying_outs.is_empty() => {
            let expected = 1 + varying_outs.len();
            if elems.len() != expected {
                return Err((
                    Span::call_site(),
                    format!(
                        "ZSL: return tuple has {} elements, expected {} (position + {} varyings)",
                        elems.len(),
                        expected,
                        varying_outs.len()
                    ),
                ));
            }
            // First element → gl_Position
            let pos = lower_gfx_expr(ctx, &elems[0])?;
            if pos.ty != GvTy::Vec4 {
                return Err((
                    elems[0].span(),
                    "ZSL: first tuple element (position) must be Vec4".into(),
                ));
            }
            ctx.spv.op_store(out_var, pos.id);
            // Remaining elements → varying outputs
            for (i, (var_id, expected_gty)) in varying_outs.iter().enumerate() {
                let val = lower_gfx_expr(ctx, &elems[i + 1])?;
                if val.ty != *expected_gty {
                    return Err((
                        elems[i + 1].span(),
                        format!(
                            "ZSL: varying[{i}] type mismatch: expected {:?}, got {:?}",
                            expected_gty, val.ty
                        ),
                    ));
                }
                ctx.spv.op_store(*var_id, val.id);
            }
            Ok(())
        }
        Stmt::Expr(expr, None) => {
            let val = lower_gfx_expr(ctx, expr)?;
            if val.ty != GvTy::Vec4 {
                return Err((
                    expr.span(),
                    format!("ZSL: shader return type must be Vec4, got {:?}", val.ty),
                ));
            }
            ctx.spv.op_store(out_var, val.id);
            Ok(())
        }
        other => Err((
            Span::call_site(),
            format!(
                "ZSL: last statement must be a tail expression (no semicolon); got `{}`",
                quote::quote!(#other)
            ),
        )),
    }
}

fn lower_gfx_stmt(ctx: &mut GfxCtx<'_>, stmt: &Stmt) -> Result<(), (Span, String)> {
    match stmt {
        Stmt::Local(local) => {
            let ident = gfx_local_ident(local)?;
            if let Some(init) = &local.init {
                let gty = ctx.locals.get(&ident).map(|l| l.ty).unwrap_or(GvTy::F32);
                let val = lower_gfx_expr(ctx, &init.expr)?;
                let coerced = coerce_gfx(ctx, val, gty, init.expr.span())?;
                let ptr = ctx
                    .locals
                    .get(&ident)
                    .ok_or_else(|| {
                        (
                            Span::call_site(),
                            format!("ZSL: undeclared local `{ident}`"),
                        )
                    })?
                    .ptr;
                ctx.spv.op_store(ptr, coerced);
            }
            Ok(())
        }
        Stmt::Expr(expr, Some(_)) => {
            lower_gfx_expr(ctx, expr)?;
            Ok(())
        }
        other => Err((
            Span::call_site(),
            format!("ZSL: unsupported statement `{}`", quote::quote!(#other)),
        )),
    }
}

// ── Expression lowering ───────────────────────────────────────────────────────

fn lower_gfx_expr(ctx: &mut GfxCtx<'_>, expr: &Expr) -> Result<GVal, (Span, String)> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit), ..
        }) => {
            let v: u32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of u32 range", lit)))?;
            let id = ctx.spv.constant_u32(ctx.t_u32, v);
            Ok(GVal { id, ty: GvTy::U32 })
        }

        Expr::Lit(ExprLit {
            lit: Lit::Float(lit),
            ..
        }) => {
            let v: f32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of f32 range", lit)))?;
            let id = ctx.spv.constant_f32(ctx.t_f32, v);
            Ok(GVal { id, ty: GvTy::F32 })
        }

        Expr::Path(ExprPath { path, .. }) => {
            let ident = path
                .get_ident()
                .ok_or_else(|| (path.span(), "ZSL: expected simple identifier".into()))?
                .to_string();

            if let Some(local) = ctx.locals.get(&ident) {
                let (ty, ptr, elem_ty) = (local.ty, local.ptr, local.elem_ty);
                let id = ctx.spv.op_load(elem_ty, ptr);
                return Ok(GVal { id, ty });
            }

            if let Some(info) = ctx.inputs.get(&ident) {
                let (ty, var, elem_ty) = (info.ty, info.var, info.elem_ty);
                let id = ctx.spv.op_load(elem_ty, var);
                return Ok(GVal { id, ty });
            }

            if let Some(info) = ctx.scalar_params.get(&ident).cloned() {
                let pc_var = ctx
                    .pc_var
                    .ok_or_else(|| (path.span(), "ZSL: no push constant block".into()))?;
                let pc_ptr_ty = match info.gty {
                    GvTy::U32 => ctx.t_ptr_pc_u32,
                    GvTy::F32 => ctx.t_ptr_pc_f32,
                    GvTy::Mat4 => ctx.t_ptr_pc_mat4,
                    _ => {
                        return Err((path.span(), "ZSL: unsupported push-constant type".into()));
                    }
                };
                if pc_ptr_ty == Id(0) {
                    return Err((
                        path.span(),
                        "ZSL: push-constant pointer type not allocated".into(),
                    ));
                }
                let pc_idx = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
                let chain = ctx.spv.op_access_chain(pc_ptr_ty, pc_var, &[pc_idx]);
                let id = ctx.spv.op_load(info.ty, chain);
                return Ok(GVal { id, ty: info.gty });
            }

            Err((path.span(), format!("ZSL: unknown identifier `{ident}`")))
        }

        // Component access: expr.x / .y / .z / .w
        Expr::Field(ExprField {
            base,
            member: Member::Named(field),
            ..
        }) => {
            let composite = lower_gfx_expr(ctx, base)?;
            let index = match field.to_string().as_str() {
                "x" => 0u32,
                "y" => 1,
                "z" => 2,
                "w" => 3,
                other => {
                    return Err((
                        field.span(),
                        format!("ZSL: unknown field `.{other}`; use .x/.y/.z/.w"),
                    ));
                }
            };
            let t_f32 = ctx.t_f32;
            let id = ctx.spv.op_composite_extract(t_f32, composite.id, &[index]);
            Ok(GVal { id, ty: GvTy::F32 })
        }

        // Vec4(a,b,c,d) / Vec3(a,b,c) / Vec2(a,b)
        Expr::Call(ExprCall { func, args, .. }) => {
            let Expr::Path(p) = &**func else {
                return Err((
                    func.span(),
                    "ZSL: unsupported call; expected Vec2/Vec3/Vec4".into(),
                ));
            };
            let Some(ctor) = p.path.get_ident() else {
                return Err((
                    func.span(),
                    "ZSL: unsupported call; expected Vec2/Vec3/Vec4".into(),
                ));
            };
            let (expected, gty, spv_ty) = match ctor.to_string().as_str() {
                "Vec4" => (4usize, GvTy::Vec4, ctx.t_vec4),
                "Vec3" => (3, GvTy::Vec3, ctx.t_vec3),
                "Vec2" => (2, GvTy::Vec2, ctx.t_vec2),
                other => {
                    return Err((
                        ctor.span(),
                        format!("ZSL: unknown function `{other}`; use Vec2/Vec3/Vec4 constructors"),
                    ));
                }
            };
            if args.len() != expected {
                return Err((
                    func.span(),
                    format!("ZSL: {ctor} takes {expected} args, got {}", args.len()),
                ));
            }
            let mut components: Vec<Id> = Vec::with_capacity(expected);
            for arg in args {
                let v = lower_gfx_expr(ctx, arg)?;
                let f32_id = scalar_to_f32(ctx, v);
                components.push(f32_id);
            }
            let id = ctx.spv.op_composite_construct(spv_ty, &components);
            Ok(GVal { id, ty: gty })
        }

        // Scalar binary ops
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let lhs = lower_gfx_expr(ctx, left)?;
            let rhs = lower_gfx_expr(ctx, right)?;
            gfx_binary_arith(ctx, lhs, rhs, op)
        }

        // vec3.extend(f32) → Vec4
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            if method != "extend" {
                return Err((
                    method.span(),
                    format!("ZSL: unknown method `.{method}()`; use `.extend(f32)` on Vec3"),
                ));
            }
            if args.len() != 1 {
                return Err((
                    method.span(),
                    "ZSL: `.extend()` takes exactly one argument".into(),
                ));
            }
            let base = lower_gfx_expr(ctx, receiver)?;
            if base.ty != GvTy::Vec3 {
                return Err((
                    receiver.span(),
                    "ZSL: `.extend()` requires a Vec3 receiver".into(),
                ));
            }
            let ext = lower_gfx_expr(ctx, &args[0])?;
            let ext_id = scalar_to_f32(ctx, ext);
            let t_f32 = ctx.t_f32;
            let x = ctx.spv.op_composite_extract(t_f32, base.id, &[0]);
            let y = ctx.spv.op_composite_extract(t_f32, base.id, &[1]);
            let z = ctx.spv.op_composite_extract(t_f32, base.id, &[2]);
            let t_vec4 = ctx.t_vec4;
            let id = ctx.spv.op_composite_construct(t_vec4, &[x, y, z, ext_id]);
            Ok(GVal { id, ty: GvTy::Vec4 })
        }

        Expr::Paren(ep) => lower_gfx_expr(ctx, &ep.expr),

        other => Err((
            other.span(),
            format!("ZSL: unsupported expression `{}`", quote::quote!(#other)),
        )),
    }
}

// ── Local variable pre-scan ───────────────────────────────────────────────────

fn collect_gfx_locals(block: &Block) -> Result<Vec<(String, GvTy)>, (Span, String)> {
    let mut out = Vec::new();
    for stmt in &block.stmts {
        if let Stmt::Local(local) = stmt {
            let ident = gfx_local_ident(local)?;
            let gty = if let syn::Pat::Type(pt) = &local.pat {
                let zty = ZslType::from_syn(&pt.ty)?;
                gv_ty_from_zsl(&zty).ok_or_else(|| {
                    (
                        pt.ty.span(),
                        format!(
                            "ZSL: unsupported local type `{}`; use f32/u32/Vec2/Vec3/Vec4",
                            zty.display()
                        ),
                    )
                })?
            } else {
                GvTy::F32
            };
            out.push((ident, gty));
        }
    }
    Ok(out)
}

fn gfx_local_ident(local: &syn::Local) -> Result<String, (Span, String)> {
    match &local.pat {
        syn::Pat::Type(pt) => match &*pt.pat {
            syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
            p => Err((
                p.span(),
                "ZSL: let binding must be a simple identifier".into(),
            )),
        },
        syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
        p => Err((
            p.span(),
            "ZSL: let binding must be a simple identifier".into(),
        )),
    }
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

fn gfx_binary_arith(
    ctx: &mut GfxCtx<'_>,
    lhs: GVal,
    rhs: GVal,
    op: &BinOp,
) -> Result<GVal, (Span, String)> {
    if lhs.ty == GvTy::Mat4 {
        if !matches!(op, BinOp::Mul(_)) {
            return Err((op.span(), "ZSL: Mat4 only supports `*`".into()));
        }
        if rhs.ty != GvTy::Vec4 {
            return Err((op.span(), "ZSL: Mat4 * requires Vec4 on the right".into()));
        }
        let t_vec4 = ctx.t_vec4;
        let id = ctx.spv.op_matrix_times_vector(t_vec4, lhs.id, rhs.id);
        return Ok(GVal { id, ty: GvTy::Vec4 });
    }
    if !matches!(lhs.ty, GvTy::F32 | GvTy::U32) {
        return Err((
            op.span(),
            "ZSL: binary arithmetic only on scalar types (f32, u32)".into(),
        ));
    }
    let (lhs, rhs) = gfx_unify_scalars(ctx, lhs, rhs);
    let ty_id = ctx.spv_elem_ty(lhs.ty);
    let id = match op {
        BinOp::Add(_) => {
            if lhs.ty == GvTy::F32 {
                ctx.spv.op_fadd(ty_id, lhs.id, rhs.id)
            } else {
                ctx.spv.op_iadd(ty_id, lhs.id, rhs.id)
            }
        }
        BinOp::Sub(_) => {
            if lhs.ty == GvTy::F32 {
                ctx.spv.op_fsub(ty_id, lhs.id, rhs.id)
            } else {
                ctx.spv.op_isub(ty_id, lhs.id, rhs.id)
            }
        }
        BinOp::Mul(_) => {
            if lhs.ty == GvTy::F32 {
                ctx.spv.op_fmul(ty_id, lhs.id, rhs.id)
            } else {
                ctx.spv.op_imul(ty_id, lhs.id, rhs.id)
            }
        }
        BinOp::Div(_) => {
            if lhs.ty == GvTy::F32 {
                ctx.spv.op_fdiv(ty_id, lhs.id, rhs.id)
            } else {
                ctx.spv.op_udiv(ty_id, lhs.id, rhs.id)
            }
        }
        other => return Err((other.span(), "ZSL: unsupported op; use +, -, *, /".into())),
    };
    Ok(GVal { id, ty: lhs.ty })
}

fn gfx_unify_scalars(ctx: &mut GfxCtx<'_>, lhs: GVal, rhs: GVal) -> (GVal, GVal) {
    if lhs.ty == rhs.ty {
        return (lhs, rhs);
    }
    let lhs_id = scalar_to_f32(ctx, lhs);
    let rhs_id = scalar_to_f32(ctx, rhs);
    (
        GVal {
            id: lhs_id,
            ty: GvTy::F32,
        },
        GVal {
            id: rhs_id,
            ty: GvTy::F32,
        },
    )
}

// ── Coercions ─────────────────────────────────────────────────────────────────

fn scalar_to_f32(ctx: &mut GfxCtx<'_>, val: GVal) -> Id {
    if val.ty == GvTy::F32 {
        return val.id;
    }
    let t = ctx.t_f32;
    ctx.spv.op_convert_u_to_f(t, val.id)
}

fn coerce_gfx(
    ctx: &mut GfxCtx<'_>,
    val: GVal,
    target: GvTy,
    span: Span,
) -> Result<Id, (Span, String)> {
    if val.ty == target {
        return Ok(val.id);
    }
    match (val.ty, target) {
        (GvTy::U32, GvTy::F32) => {
            let t = ctx.t_f32;
            Ok(ctx.spv.op_convert_u_to_f(t, val.id))
        }
        (GvTy::F32, GvTy::U32) => {
            let t = ctx.t_u32;
            Ok(ctx.spv.op_convert_f_to_u(t, val.id))
        }
        _ => Err((
            span,
            format!("ZSL: cannot coerce {:?} to {:?}", val.ty, target),
        )),
    }
}

// ── Type helpers ──────────────────────────────────────────────────────────────

fn gv_ty_from_zsl(zty: &ZslType) -> Option<GvTy> {
    match zty {
        ZslType::F32 => Some(GvTy::F32),
        ZslType::U32 => Some(GvTy::U32),
        ZslType::Vec2 => Some(GvTy::Vec2),
        ZslType::Vec3 => Some(GvTy::Vec3),
        ZslType::Vec4 => Some(GvTy::Vec4),
        _ => None,
    }
}

fn gvty_to_spv_id(gty: GvTy, t_f32: Id, t_u32: Id, t_vec2: Id, t_vec3: Id, t_vec4: Id) -> Id {
    match gty {
        GvTy::F32 => t_f32,
        GvTy::U32 => t_u32,
        GvTy::Vec2 => t_vec2,
        GvTy::Vec3 => t_vec3,
        GvTy::Vec4 => t_vec4,
        GvTy::Mat4 => unreachable!(), // Mat4 is never a vertex varying
    }
}
