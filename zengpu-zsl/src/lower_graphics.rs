//! ZSL → SPIR-V lowering for vertex and fragment shaders.
//!
//! Supports: `#[location(N)]` inputs (f32/Vec2/Vec3/Vec4), push constants
//! (f32/u32/Mat4), Vec2/Vec3/Vec4 constructors, component access (.x/.y/.z/.w),
//! `vec.extend(f32)`, scalar and vector arithmetic, Mat4*Vec4, vec*scalar,
//! scalar*vec, `dot(a,b)`, unary negation, comparison operators (`<>/<=/>=/==/!=`),
//! logical operators (`&&`/`||`), `if`/`else` control flow, variable reassignment,
//! and GLSL.std.450 built-ins (`abs`/`sign`/`sqrt`/`floor`/`ceil`/`fract`/
//! `normalize`/`length`/`min`/`max`/`pow`/`clamp`/`mix`).

/// GLSL.std.450 extended instruction opcodes used by ZSL.
mod glsl_op {
    pub const F_ABS: u32 = 4;
    pub const F_SIGN: u32 = 6;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const POW: u32 = 26;
    pub const SQRT: u32 = 31;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const F_MIX: u32 = 46;
    pub const LENGTH: u32 = 66;
    pub const NORMALIZE: u32 = 69;
}

use std::collections::HashMap;

use proc_macro2::Span;
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprBlock, ExprCall, ExprField, ExprIf,
    ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath, ExprTuple, ExprUnary, Lit, Member,
    Stmt, UnOp, spanned::Spanned,
};

use crate::frontend::ast::{ZslEntryPoint, ZslParam};
use crate::frontend::types::ZslType;
use crate::spirv::{Id, SpvBuilder, builtin, deco, sc};

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

#[derive(Clone, Copy)]
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

/// A `Buf<f32>` or `BufMut<f32>` parameter in a graphics shader.
///
/// The buffer index is passed via push constant at field `pc_index`
/// (already offset by any preceding buf indices in the PC struct).
struct GfxBufInfo {
    pc_index: u32,
    writable: bool,
}

struct GfxCtx<'a> {
    spv: &'a mut SpvBuilder,
    t_f32: Id,
    t_u32: Id,
    t_bool: Id,
    t_vec2: Id,
    t_vec3: Id,
    t_vec4: Id,
    t_mat4: Id,
    glsl_ext: Id,
    inputs: HashMap<String, InputVar>,
    scalar_params: HashMap<String, GfxScalarInfo>,
    /// Buffer params (Buf<f32> / BufMut<f32>) keyed by parameter name.
    buf_params: HashMap<String, GfxBufInfo>,
    /// Bindless SSBO array global variable (None when no buf params).
    g_bufs_var: Option<Id>,
    /// Number of buffer params (= number of u32 indices prepended to the PC block).
    #[allow(dead_code)]
    n_buf_params: u32,
    /// Pointer into the SSBO element array (StorageBuffer storage class, f32 element).
    t_ptr_ssbo_f32: Id,
    pc_var: Option<Id>,
    t_ptr_pc_u32: Id,
    t_ptr_pc_f32: Id,
    t_ptr_pc_mat4: Id,
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
    let glsl_ext = spv.ext_inst_import_glsl();
    spv.memory_model_logical_glsl450();

    // ── Core types ────────────────────────────────────────────────────────────
    let t_void = spv.type_void();
    let t_f32 = spv.type_float(32);
    let t_u32 = spv.type_int(32, false);
    let t_bool = spv.type_bool();
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
    let buf_params: Vec<&ZslParam> = entry
        .params
        .iter()
        .filter(|p| matches!(p.ty, ZslType::Buf(_) | ZslType::BufMut(_)))
        .collect();

    // Validate buf element types — only f32 is supported in the current lowering.
    for p in &buf_params {
        let elem = match &p.ty {
            ZslType::Buf(e) | ZslType::BufMut(e) => e,
            _ => unreachable!(),
        };
        if !matches!(elem, crate::frontend::types::BufElem::F32) {
            return Err((
                Span::call_site(),
                format!(
                    "ZSL: `{}` — only `Buf<f32>` / `BufMut<f32>` are supported in \
                     vertex/fragment shaders; found `Buf<{}>` for `{}`",
                    p.ident,
                    elem.display(),
                    p.ident,
                ),
            ));
        }
    }

    // Buf params need the RuntimeDescriptorArray capability (same as compute).
    if !buf_params.is_empty() {
        spv.capability_runtime_descriptor_array();
    }

    let n_buf_params = buf_params.len() as u32;

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

    // ── SSBO types — bindless model (same as compute) ────────────────────────
    // layout(set=0, binding=0) buffer Buf { float data[]; } g_bufs[];
    let (g_bufs_var, t_ptr_ssbo_f32) = if !buf_params.is_empty() {
        let t_ra_f32 = spv.type_runtime_array(t_f32);
        spv.decorate(t_ra_f32, deco::ARRAY_STRIDE, &[4]);
        let t_struct_buf = spv.type_struct(&[t_ra_f32]);
        spv.decorate(t_struct_buf, deco::BLOCK, &[]);
        spv.member_decorate(t_struct_buf, 0, deco::OFFSET, &[0]);
        let t_ra_struct = spv.type_runtime_array(t_struct_buf);
        let t_ptr_ra_struct = spv.type_pointer(sc::STORAGE_BUFFER, t_ra_struct);
        let t_ptr_elem = spv.type_pointer(sc::STORAGE_BUFFER, t_f32);
        let var = spv.global_variable(t_ptr_ra_struct, sc::STORAGE_BUFFER);
        spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
        spv.decorate(var, deco::BINDING, &[0]);
        (Some(var), t_ptr_elem)
    } else {
        (None, Id(0))
    };

    // ── Push-constant block: buf indices (u32) first, then u32/f32/Mat4 ──────
    // Having buf params means we always have a PC block (for the indices),
    // even if scalar_params is empty.
    let pc_var = if !buf_params.is_empty() || !scalar_params.is_empty() {
        // n_buf_params u32 entries (buffer slot indices), then scalar params.
        let mut pc_members: Vec<Id> = (0..n_buf_params).map(|_| t_u32).collect();
        pc_members.extend(scalar_params.iter().map(|p| match p.ty {
            ZslType::U32 => t_u32,
            ZslType::F32 => t_f32,
            ZslType::Mat4 => t_mat4,
            _ => unreachable!(),
        }));
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);

        // Buf-index fields: 4 bytes each at offsets 0, 4, 8, ...
        let mut offset: u32 = 0;
        for i in 0..n_buf_params {
            spv.member_decorate(t_pc_struct, i, deco::OFFSET, &[offset]);
            offset += 4;
        }
        // Scalar fields: same alignment rules as before, but member index is shifted.
        for (i, p) in scalar_params.iter().enumerate() {
            let member = n_buf_params + i as u32;
            match p.ty {
                ZslType::U32 | ZslType::F32 => {
                    spv.member_decorate(t_pc_struct, member, deco::OFFSET, &[offset]);
                    offset += 4;
                }
                ZslType::Mat4 => {
                    offset = (offset + 15) & !15; // align to 16
                    spv.member_decorate(t_pc_struct, member, deco::OFFSET, &[offset]);
                    spv.member_decorate(t_pc_struct, member, deco::COL_MAJOR, &[]);
                    spv.member_decorate(t_pc_struct, member, deco::MATRIX_STRIDE, &[16]);
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
    if is_fragment {
        spv.entry_point_fragment(fn_id, "main", &interface);
        spv.execution_mode_origin_upper_left(fn_id);
    } else {
        spv.entry_point_vertex(fn_id, "main", &interface);
    }

    // ── Constants ─────────────────────────────────────────────────────────────
    let const_0_u32 = spv.constant_u32(t_u32, 0);

    // ── Scalar/mat param map (pc_index offset by n_buf_params) ───────────────
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
                    // Scalar members are AFTER the buf-index members in the PC struct.
                    pc_index: n_buf_params + i as u32,
                    ty,
                    gty,
                },
            )
        })
        .collect();

    // ── Buffer param map ──────────────────────────────────────────────────────
    let buf_param_map: HashMap<String, GfxBufInfo> = buf_params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                p.ident.to_string(),
                GfxBufInfo {
                    pc_index: i as u32, // first n_buf_params fields in PC struct
                    writable: matches!(p.ty, ZslType::BufMut(_)),
                },
            )
        })
        .collect();

    // ── Push-constant pointer types ───────────────────────────────────────────
    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    let mut t_ptr_pc_mat4 = Id(0);
    if pc_var.is_some() {
        // Buf indices are u32 — need u32 PC pointer when there are buf params.
        if n_buf_params > 0 {
            t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
        }
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

    // Build a name→type map from all parameters so collect_gfx_locals can infer
    // the types of unannotated `let` bindings (e.g. `let v = model * pos.extend(1.0)`).
    let param_types: HashMap<String, GvTy> = {
        let mut m = HashMap::new();
        for p in &entry.params {
            // Location inputs use gv_ty_from_zsl; scalars/matrices are a superset.
            let gty = match &p.ty {
                ZslType::F32  => Some(GvTy::F32),
                ZslType::U32  => Some(GvTy::U32),
                ZslType::Vec2 => Some(GvTy::Vec2),
                ZslType::Vec3 => Some(GvTy::Vec3),
                ZslType::Vec4 => Some(GvTy::Vec4),
                ZslType::Mat4 => Some(GvTy::Mat4),
                _ => None,
            };
            if let Some(gty) = gty {
                m.insert(p.ident.to_string(), gty);
            }
        }
        m
    };
    let local_decls = collect_gfx_locals(body, &param_types)?;
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
        t_bool,
        t_vec2,
        t_vec3,
        t_vec4,
        t_mat4,
        glsl_ext,
        inputs: input_vars,
        scalar_params: scalar_param_map,
        buf_params: buf_param_map,
        g_bufs_var,
        n_buf_params,
        t_ptr_ssbo_f32,
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
        // if cond { ... } or if cond { ... } else { ... }
        // Bare (no semicolon) or with semicolon — both are statements.
        Stmt::Expr(
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }),
            _semi,
        ) => {
            let cond_id = lower_gfx_condition(ctx, cond)?;
            let then_label = ctx.spv.fresh_id();
            let merge_label = ctx.spv.fresh_id();

            if let Some((_else_tok, else_expr)) = else_branch {
                let else_label = ctx.spv.fresh_id();
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, then_label, else_label);

                ctx.spv.label_with_id(then_label);
                lower_gfx_block(ctx, then_branch)?;
                ctx.spv.op_branch(merge_label);

                ctx.spv.label_with_id(else_label);
                match else_expr.as_ref() {
                    Expr::Block(ExprBlock { block, .. }) => lower_gfx_block(ctx, block)?,
                    other => {
                        return Err((
                            other.span(),
                            "ZSL: else branch must be a block `{ ... }`".into(),
                        ));
                    }
                }
                ctx.spv.op_branch(merge_label);
            } else {
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, then_label, merge_label);

                ctx.spv.label_with_id(then_label);
                lower_gfx_block(ctx, then_branch)?;
                ctx.spv.op_branch(merge_label);
            }

            ctx.spv.label_with_id(merge_label);
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

        // Function calls: built-ins, Vec constructors
        Expr::Call(ExprCall { func, args, .. }) => {
            let Expr::Path(p) = &**func else {
                return Err((
                    func.span(),
                    "ZSL: unsupported call; expected a built-in or Vec2/Vec3/Vec4".into(),
                ));
            };
            let Some(ctor) = p.path.get_ident() else {
                return Err((
                    func.span(),
                    "ZSL: unsupported call; expected a built-in or Vec2/Vec3/Vec4".into(),
                ));
            };
            let name = ctor.to_string();
            match name.as_str() {
                // dot(a, b) → native OpDot; no GLSL ext needed
                "dot" => {
                    if args.len() != 2 {
                        return Err((func.span(), "ZSL: dot(a, b) takes exactly 2 args".into()));
                    }
                    let a = lower_gfx_expr(ctx, &args[0])?;
                    let b = lower_gfx_expr(ctx, &args[1])?;
                    if !matches!(a.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((args[0].span(), "ZSL: dot() requires Vec2/Vec3/Vec4".into()));
                    }
                    if a.ty != b.ty {
                        return Err((
                            args[1].span(),
                            "ZSL: dot() args must have the same vector type".into(),
                        ));
                    }
                    let t_f32 = ctx.t_f32;
                    let id = ctx.spv.op_dot(t_f32, a.id, b.id);
                    Ok(GVal { id, ty: GvTy::F32 })
                }

                // Unary GLSL built-ins: scalar f32 or any float vector → same type
                "abs" | "sign" | "sqrt" | "floor" | "ceil" | "fract" => {
                    if args.len() != 1 {
                        return Err((func.span(), format!("ZSL: {name}() takes 1 arg")));
                    }
                    let v = lower_gfx_expr(ctx, &args[0])?;
                    if !matches!(v.ty, GvTy::F32 | GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            format!("ZSL: {name}() requires f32/Vec2/Vec3/Vec4"),
                        ));
                    }
                    let opcode = match name.as_str() {
                        "abs" => glsl_op::F_ABS,
                        "sign" => glsl_op::F_SIGN,
                        "sqrt" => glsl_op::SQRT,
                        "floor" => glsl_op::FLOOR,
                        "ceil" => glsl_op::CEIL,
                        "fract" => glsl_op::FRACT,
                        _ => unreachable!(),
                    };
                    let ty_id = ctx.spv_elem_ty(v.ty);
                    let glsl = ctx.glsl_ext;
                    let id = ctx.spv.op_ext_inst(ty_id, glsl, opcode, &[v.id]);
                    Ok(GVal { id, ty: v.ty })
                }

                // normalize(v) → same vec type
                "normalize" => {
                    if args.len() != 1 {
                        return Err((func.span(), "ZSL: normalize() takes 1 arg".into()));
                    }
                    let v = lower_gfx_expr(ctx, &args[0])?;
                    if !matches!(v.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            "ZSL: normalize() requires Vec2/Vec3/Vec4".into(),
                        ));
                    }
                    let ty_id = ctx.spv_elem_ty(v.ty);
                    let glsl = ctx.glsl_ext;
                    let id = ctx
                        .spv
                        .op_ext_inst(ty_id, glsl, glsl_op::NORMALIZE, &[v.id]);
                    Ok(GVal { id, ty: v.ty })
                }

                // length(v) → f32 scalar
                "length" => {
                    if args.len() != 1 {
                        return Err((func.span(), "ZSL: length() takes 1 arg".into()));
                    }
                    let v = lower_gfx_expr(ctx, &args[0])?;
                    if !matches!(v.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            "ZSL: length() requires Vec2/Vec3/Vec4".into(),
                        ));
                    }
                    let t_f32 = ctx.t_f32;
                    let glsl = ctx.glsl_ext;
                    let id = ctx.spv.op_ext_inst(t_f32, glsl, glsl_op::LENGTH, &[v.id]);
                    Ok(GVal { id, ty: GvTy::F32 })
                }

                // min(a, b) / max(a, b) / pow(base, exp) — 2-arg, same f32/vec type
                "min" | "max" | "pow" => {
                    if args.len() != 2 {
                        return Err((func.span(), format!("ZSL: {name}(a, b) takes 2 args")));
                    }
                    let a = lower_gfx_expr(ctx, &args[0])?;
                    let b = lower_gfx_expr(ctx, &args[1])?;
                    if !matches!(a.ty, GvTy::F32 | GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            format!("ZSL: {name}() requires f32/Vec2/Vec3/Vec4"),
                        ));
                    }
                    if a.ty != b.ty {
                        return Err((
                            args[1].span(),
                            format!("ZSL: {name}() args must be the same type"),
                        ));
                    }
                    let opcode = match name.as_str() {
                        "min" => glsl_op::F_MIN,
                        "max" => glsl_op::F_MAX,
                        "pow" => glsl_op::POW,
                        _ => unreachable!(),
                    };
                    let ty_id = ctx.spv_elem_ty(a.ty);
                    let glsl = ctx.glsl_ext;
                    let id = ctx.spv.op_ext_inst(ty_id, glsl, opcode, &[a.id, b.id]);
                    Ok(GVal { id, ty: a.ty })
                }

                // clamp(x, lo, hi) — all same f32/vec type
                "clamp" => {
                    if args.len() != 3 {
                        return Err((func.span(), "ZSL: clamp(x, lo, hi) takes 3 args".into()));
                    }
                    let x = lower_gfx_expr(ctx, &args[0])?;
                    let lo = lower_gfx_expr(ctx, &args[1])?;
                    let hi = lower_gfx_expr(ctx, &args[2])?;
                    if !matches!(x.ty, GvTy::F32 | GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            "ZSL: clamp() requires f32/Vec2/Vec3/Vec4".into(),
                        ));
                    }
                    if x.ty != lo.ty || x.ty != hi.ty {
                        return Err((
                            func.span(),
                            "ZSL: clamp() x, lo, and hi must all be the same type".into(),
                        ));
                    }
                    let ty_id = ctx.spv_elem_ty(x.ty);
                    let glsl = ctx.glsl_ext;
                    let id =
                        ctx.spv
                            .op_ext_inst(ty_id, glsl, glsl_op::F_CLAMP, &[x.id, lo.id, hi.id]);
                    Ok(GVal { id, ty: x.ty })
                }

                // mix(a, b, t) — a and b same type; t same type (scalar or vec)
                "mix" => {
                    if args.len() != 3 {
                        return Err((func.span(), "ZSL: mix(a, b, t) takes 3 args".into()));
                    }
                    let a = lower_gfx_expr(ctx, &args[0])?;
                    let b = lower_gfx_expr(ctx, &args[1])?;
                    let t = lower_gfx_expr(ctx, &args[2])?;
                    if !matches!(a.ty, GvTy::F32 | GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
                        return Err((
                            args[0].span(),
                            "ZSL: mix() requires f32/Vec2/Vec3/Vec4".into(),
                        ));
                    }
                    if a.ty != b.ty {
                        return Err((
                            args[1].span(),
                            "ZSL: mix() a and b must be the same type".into(),
                        ));
                    }
                    if a.ty != t.ty {
                        return Err((
                            args[2].span(),
                            "ZSL: mix() t must be the same type as a and b; \
                             for vec mix with scalar t, use `a * (1.0 - t) + b * t`"
                                .into(),
                        ));
                    }
                    let ty_id = ctx.spv_elem_ty(a.ty);
                    let glsl = ctx.glsl_ext;
                    let id = ctx
                        .spv
                        .op_ext_inst(ty_id, glsl, glsl_op::F_MIX, &[a.id, b.id, t.id]);
                    Ok(GVal { id, ty: a.ty })
                }

                // Vec constructors: Vec2/Vec3/Vec4
                "Vec4" | "Vec3" | "Vec2" => {
                    let (expected, gty, spv_ty) = match name.as_str() {
                        "Vec4" => (4usize, GvTy::Vec4, ctx.t_vec4),
                        "Vec3" => (3, GvTy::Vec3, ctx.t_vec3),
                        "Vec2" => (2, GvTy::Vec2, ctx.t_vec2),
                        _ => unreachable!(),
                    };
                    if args.len() != expected {
                        return Err((
                            func.span(),
                            format!("ZSL: {name} takes {expected} args, got {}", args.len()),
                        ));
                    }
                    let mut components: Vec<Id> = Vec::with_capacity(expected);
                    for arg in args {
                        let v = lower_gfx_expr(ctx, arg)?;
                        components.push(scalar_to_f32(ctx, v));
                    }
                    let id = ctx.spv.op_composite_construct(spv_ty, &components);
                    Ok(GVal { id, ty: gty })
                }

                other => Err((
                    ctor.span(),
                    format!(
                        "ZSL: unknown function `{other}`; built-ins: \
                         dot, abs, sign, sqrt, floor, ceil, fract, normalize, length, \
                         min, max, pow, clamp, mix; constructors: Vec2, Vec3, Vec4"
                    ),
                )),
            }
        }

        // Unary negation: -scalar or -vec
        Expr::Unary(ExprUnary { op, expr, .. }) => match op {
            UnOp::Neg(_) => {
                let val = lower_gfx_expr(ctx, expr)?;
                match val.ty {
                    GvTy::U32 => Err((expr.span(), "ZSL: cannot negate u32".into())),
                    GvTy::Mat4 => Err((expr.span(), "ZSL: cannot negate Mat4".into())),
                    _ => {
                        let ty_id = ctx.spv_elem_ty(val.ty);
                        let id = ctx.spv.op_fnegate(ty_id, val.id);
                        Ok(GVal { id, ty: val.ty })
                    }
                }
            }
            other => Err((other.span(), "ZSL: only unary `-` is supported".into())),
        },

        // Binary ops: scalar arithmetic and vec*scalar
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

        // buf[i]  — read from a Buf<f32> parameter
        Expr::Index(ExprIndex { expr: base, index, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = base.as_ref() {
                if let Some(name) = path.get_ident().map(|i| i.to_string()) {
                    if let Some(info) = ctx.buf_params.get(&name) {
                        let pc_index = info.pc_index;
                        let g_bufs = ctx.g_bufs_var.expect("g_bufs must exist when buf_params is non-empty");
                        let pc_var = ctx.pc_var.ok_or_else(|| {
                            (base.span(), "ZSL: internal error — no PC block for buf param".into())
                        })?;
                        let pc_field = ctx.spv.constant_u32(ctx.t_u32, pc_index);
                        let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
                        let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
                        let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
                        let idx = lower_gfx_expr(ctx, index)?;
                        let idx_id = scalar_to_u32(ctx, idx);
                        let c0 = ctx.const_0_u32;
                        let t_ptr_elem = ctx.t_ptr_ssbo_f32;
                        let ptr_elem = ctx.spv.op_access_chain(t_ptr_elem, g_bufs, &[buf_idx, c0, idx_id]);
                        let t_f32 = ctx.t_f32;
                        let id = ctx.spv.op_load(t_f32, ptr_elem);
                        return Ok(GVal { id, ty: GvTy::F32 });
                    }
                }
            }
            Err((expr.span(), "ZSL: indexing is only supported on Buf<f32> params".into()))
        }

        // x = expr;  or  buf[i] = expr;
        Expr::Assign(ExprAssign { left, right, .. }) => {
            // buf[i] = rhs  — write to a BufMut<f32> parameter
            if let Expr::Index(ExprIndex { expr: base, index, .. }) = left.as_ref() {
                if let Expr::Path(ExprPath { path, .. }) = base.as_ref() {
                    if let Some(name) = path.get_ident().map(|i| i.to_string()) {
                        if let Some(info) = ctx.buf_params.get(&name) {
                            if !info.writable {
                                return Err((
                                    base.span(),
                                    format!("`{name}` is `Buf<f32>` (read-only); use `BufMut<f32>` to write"),
                                ));
                            }
                            let pc_index = info.pc_index;
                            let g_bufs = ctx.g_bufs_var.expect("g_bufs must exist");
                            let pc_var = ctx.pc_var.ok_or_else(|| {
                                (base.span(), "ZSL: internal error — no PC block for buf param".into())
                            })?;
                            let pc_field = ctx.spv.constant_u32(ctx.t_u32, pc_index);
                            let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
                            let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
                            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
                            let idx = lower_gfx_expr(ctx, index)?;
                            let idx_id = scalar_to_u32(ctx, idx);
                            let c0 = ctx.const_0_u32;
                            let t_ptr_elem = ctx.t_ptr_ssbo_f32;
                            let ptr_elem = ctx.spv.op_access_chain(t_ptr_elem, g_bufs, &[buf_idx, c0, idx_id]);
                            let rhs = lower_gfx_expr(ctx, right)?;
                            let rhs_id = scalar_to_f32(ctx, rhs);
                            ctx.spv.op_store(ptr_elem, rhs_id);
                            return Ok(GVal { id: rhs_id, ty: GvTy::F32 });
                        }
                    }
                }
            }

            // name = expr;  — reassignment to an already-declared local
            let ident = match left.as_ref() {
                Expr::Path(p) => p
                    .path
                    .get_ident()
                    .map(|i| i.to_string())
                    .ok_or_else(|| (left.span(), "ZSL: assign target must be a local".into()))?,
                other => return Err((other.span(), "ZSL: assign target must be a local or buf[i]".into())),
            };
            let (ptr, gty) = ctx
                .locals
                .get(&ident)
                .map(|l| (l.ptr, l.ty))
                .ok_or_else(|| (left.span(), format!("ZSL: undeclared local `{ident}`")))?;
            let val = lower_gfx_expr(ctx, right)?;
            let coerced = coerce_gfx(ctx, val, gty, right.span())?;
            ctx.spv.op_store(ptr, coerced);
            Ok(GVal {
                id: coerced,
                ty: gty,
            })
        }

        other => Err((
            other.span(),
            format!("ZSL: unsupported expression `{}`", quote::quote!(#other)),
        )),
    }
}

// ── Local variable pre-scan ───────────────────────────────────────────────────

/// Infer the [`GvTy`] of an expression from its syntax, using `known` as a
/// map of already-typed names (params + earlier locals in the same block).
/// Returns `None` when the expression form is too complex to infer statically.
fn infer_gfx_ty(expr: &Expr, known: &HashMap<String, GvTy>) -> Option<GvTy> {
    match expr {
        Expr::Lit(ExprLit { lit: Lit::Float(_), .. }) => Some(GvTy::F32),
        Expr::Lit(ExprLit { lit: Lit::Int(_), .. }) => Some(GvTy::U32),

        Expr::Path(ExprPath { path, .. }) => {
            let name = path.get_ident()?.to_string();
            known.get(&name).copied()
        }

        Expr::Paren(ExprParen { expr, .. }) => infer_gfx_ty(expr, known),

        Expr::Unary(syn::ExprUnary { expr, .. }) => infer_gfx_ty(expr, known),

        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let recv_ty = infer_gfx_ty(receiver, known)?;
            match method.to_string().as_str() {
                // VecN.extend(f32) → Vec(N+1)
                "extend" if args.len() == 1 => match recv_ty {
                    GvTy::Vec2 => Some(GvTy::Vec3),
                    GvTy::Vec3 => Some(GvTy::Vec4),
                    _ => None,
                },
                // VecN.truncate() → Vec(N-1)
                "truncate" if args.is_empty() => match recv_ty {
                    GvTy::Vec3 => Some(GvTy::Vec2),
                    GvTy::Vec4 => Some(GvTy::Vec3),
                    _ => None,
                },
                // Swizzle shortcuts
                "xyz" | "zyx" if args.is_empty() => match recv_ty {
                    GvTy::Vec4 => Some(GvTy::Vec3),
                    _ => None,
                },
                "xy" | "yx" if args.is_empty() => match recv_ty {
                    GvTy::Vec3 | GvTy::Vec4 => Some(GvTy::Vec2),
                    _ => None,
                },
                // Functions that preserve the receiver type
                "normalize" | "abs" | "floor" | "ceil" | "fract" | "sign" | "sqrt"
                    if args.is_empty() =>
                {
                    Some(recv_ty)
                }
                // Scalar reductions
                "length" | "dot" if args.len() <= 1 => Some(GvTy::F32),
                "cross" if args.len() == 1 => match recv_ty {
                    GvTy::Vec3 => Some(GvTy::Vec3),
                    _ => None,
                },
                _ => None,
            }
        }

        Expr::Call(ExprCall { func, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                let name = path.get_ident()?.to_string();
                match name.as_str() {
                    "vec2" | "Vec2" => Some(GvTy::Vec2),
                    "vec3" | "Vec3" => Some(GvTy::Vec3),
                    "vec4" | "Vec4" => Some(GvTy::Vec4),
                    _ => None,
                }
            } else {
                None
            }
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let lty = infer_gfx_ty(left, known);
            let rty = infer_gfx_ty(right, known);
            match op {
                // Comparisons always produce bool — not a GvTy, skip
                BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_)
                | BinOp::Eq(_) | BinOp::Ne(_) => None,
                _ => match (lty, rty) {
                    // Mat4 * Vec4 → Vec4  (or any VecN for completeness)
                    (Some(GvTy::Mat4), Some(r))
                        if matches!(r, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) =>
                    {
                        Some(r)
                    }
                    // VecN op VecN → VecN
                    (Some(l), Some(r)) if l == r => Some(l),
                    // VecN op scalar or scalar op VecN → VecN
                    (Some(GvTy::F32) | Some(GvTy::U32), Some(r))
                        if matches!(r, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) =>
                    {
                        Some(r)
                    }
                    (Some(l), Some(GvTy::F32) | Some(GvTy::U32))
                        if matches!(l, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) =>
                    {
                        Some(l)
                    }
                    _ => None,
                },
            }
        }

        // Field access: vec.x / vec.y / … → F32; anything else we can't infer
        Expr::Field(ExprField { member: Member::Named(m), .. }) => {
            let name = m.to_string();
            if name.len() == 1 && "xyzw".contains(name.as_str()) {
                Some(GvTy::F32)
            } else {
                None
            }
        }

        _ => None,
    }
}

fn collect_gfx_locals(
    block: &Block,
    param_types: &HashMap<String, GvTy>,
) -> Result<Vec<(String, GvTy)>, (Span, String)> {
    let mut out = Vec::new();
    collect_gfx_locals_block(block, param_types, &mut out)?;
    Ok(out)
}

/// Recursively collect all `let` bindings in a block and any nested `if`
/// branches. All collected locals are emitted as `OpVariable` at the top of
/// the function's first basic block (SPIR-V requires this).
///
/// When a binding has no explicit type annotation, the type is inferred from
/// the initializer expression via [`infer_gfx_ty`]. If inference fails, we
/// fall back to `F32` (the original behaviour) rather than emitting an error,
/// since the real type-mismatch error will surface during lowering anyway.
fn collect_gfx_locals_block(
    block: &Block,
    param_types: &HashMap<String, GvTy>,
    out: &mut Vec<(String, GvTy)>,
) -> Result<(), (Span, String)> {
    // Running map of names→types visible at this point: params + already-seen locals.
    let mut known: HashMap<String, GvTy> = param_types.clone();
    for (name, gty) in out.iter() {
        known.insert(name.clone(), *gty);
    }

    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                let ident = gfx_local_ident(local)?;
                let gty = if let syn::Pat::Type(pt) = &local.pat {
                    // Explicit type annotation — use it.
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
                } else if let Some(init) = &local.init {
                    // No annotation — infer from the initializer; fall back to F32.
                    infer_gfx_ty(&init.expr, &known).unwrap_or(GvTy::F32)
                } else {
                    GvTy::F32
                };
                known.insert(ident.clone(), gty);
                out.push((ident, gty));
            }
            Stmt::Expr(
                Expr::If(ExprIf {
                    then_branch,
                    else_branch,
                    ..
                }),
                _,
            ) => {
                collect_gfx_locals_block(then_branch, param_types, out)?;
                if let Some((_, else_expr)) = else_branch {
                    if let Expr::Block(ExprBlock { block, .. }) = else_expr.as_ref() {
                        collect_gfx_locals_block(block, param_types, out)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
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

// ── Control flow ──────────────────────────────────────────────────────────────

/// Lower a comparison or logical expression to a bool ID for use as a branch condition.
/// Supports `<`, `>`, `<=`, `>=`, `==`, `!=` on f32/u32 and `&&`/`||` of sub-conditions.
fn lower_gfx_condition(ctx: &mut GfxCtx<'_>, expr: &Expr) -> Result<Id, (Span, String)> {
    let Expr::Binary(ExprBinary {
        left, op, right, ..
    }) = expr
    else {
        return Err((
            expr.span(),
            "ZSL: `if` condition must be a comparison (< > <= >= == !=) or logical (&& ||)".into(),
        ));
    };
    // Logical short-circuit operators — recurse on both sides.
    if matches!(op, BinOp::And(_) | BinOp::Or(_)) {
        let l = lower_gfx_condition(ctx, left)?;
        let r = lower_gfx_condition(ctx, right)?;
        let t_bool = ctx.t_bool;
        let id = match op {
            BinOp::And(_) => ctx.spv.op_logical_and(t_bool, l, r),
            BinOp::Or(_) => ctx.spv.op_logical_or(t_bool, l, r),
            _ => unreachable!(),
        };
        return Ok(id);
    }
    let lhs = lower_gfx_expr(ctx, left)?;
    let rhs = lower_gfx_expr(ctx, right)?;
    let t_bool = ctx.t_bool;
    let (lhs, rhs) = gfx_unify_scalars(ctx, lhs, rhs);
    let id = match (op, lhs.ty) {
        (BinOp::Lt(_), GvTy::F32) => ctx.spv.op_ford_lt(t_bool, lhs.id, rhs.id),
        (BinOp::Le(_), GvTy::F32) => ctx.spv.op_ford_le(t_bool, lhs.id, rhs.id),
        (BinOp::Gt(_), GvTy::F32) => ctx.spv.op_ford_gt(t_bool, lhs.id, rhs.id),
        (BinOp::Ge(_), GvTy::F32) => ctx.spv.op_ford_ge(t_bool, lhs.id, rhs.id),
        (BinOp::Eq(_), GvTy::F32) => ctx.spv.op_ford_eq(t_bool, lhs.id, rhs.id),
        (BinOp::Ne(_), GvTy::F32) => ctx.spv.op_ford_ne(t_bool, lhs.id, rhs.id),
        (BinOp::Lt(_), GvTy::U32) => ctx.spv.op_ult(t_bool, lhs.id, rhs.id),
        (BinOp::Le(_), GvTy::U32) => ctx.spv.op_ule(t_bool, lhs.id, rhs.id),
        (BinOp::Gt(_), GvTy::U32) => ctx.spv.op_ugt(t_bool, lhs.id, rhs.id),
        (BinOp::Ge(_), GvTy::U32) => ctx.spv.op_uge(t_bool, lhs.id, rhs.id),
        (BinOp::Eq(_), GvTy::U32) => ctx.spv.op_iequal(t_bool, lhs.id, rhs.id),
        (BinOp::Ne(_), GvTy::U32) => ctx.spv.op_inot_equal(t_bool, lhs.id, rhs.id),
        _ => {
            return Err((
                op.span(),
                "ZSL: unsupported comparison; use <, >, <=, >=, ==, != on f32 or u32".into(),
            ));
        }
    };
    Ok(id)
}

/// Lower statements inside an `if` or `else` branch block.
fn lower_gfx_block(ctx: &mut GfxCtx<'_>, block: &Block) -> Result<(), (Span, String)> {
    for stmt in &block.stmts {
        lower_gfx_stmt(ctx, stmt)?;
    }
    Ok(())
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
    // vec op vec  →  component-wise float op (OpFAdd / FSub / FMul / FDiv)
    if matches!(lhs.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4)
        && matches!(rhs.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4)
    {
        if lhs.ty != rhs.ty {
            return Err((
                op.span(),
                format!(
                    "ZSL: vec op requires matching types; got {:?} and {:?}",
                    lhs.ty, rhs.ty
                ),
            ));
        }
        let ty_id = ctx.spv_elem_ty(lhs.ty);
        let id = match op {
            BinOp::Add(_) => ctx.spv.op_fadd(ty_id, lhs.id, rhs.id),
            BinOp::Sub(_) => ctx.spv.op_fsub(ty_id, lhs.id, rhs.id),
            BinOp::Mul(_) => ctx.spv.op_fmul(ty_id, lhs.id, rhs.id),
            BinOp::Div(_) => ctx.spv.op_fdiv(ty_id, lhs.id, rhs.id),
            other => return Err((other.span(), "ZSL: vec op: use +, -, *, /".into())),
        };
        return Ok(GVal { id, ty: lhs.ty });
    }
    // vec * scalar  or  scalar * vec  →  OpVectorTimesScalar
    if let BinOp::Mul(_) = op {
        let (vec_val, scalar_val) = match (lhs.ty, rhs.ty) {
            (GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4, GvTy::F32 | GvTy::U32) => (lhs, rhs),
            (GvTy::F32 | GvTy::U32, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) => (rhs, lhs),
            _ => (lhs, rhs), // fall through to scalar path
        };
        if matches!(vec_val.ty, GvTy::Vec2 | GvTy::Vec3 | GvTy::Vec4) {
            let scalar_f32 = scalar_to_f32(ctx, scalar_val);
            let vec_ty_id = ctx.spv_elem_ty(vec_val.ty);
            let id = ctx
                .spv
                .op_vector_times_scalar(vec_ty_id, vec_val.id, scalar_f32);
            return Ok(GVal { id, ty: vec_val.ty });
        }
    }
    if !matches!(lhs.ty, GvTy::F32 | GvTy::U32) {
        return Err((
            op.span(),
            "ZSL: binary arithmetic only on scalar or vector types".into(),
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

fn scalar_to_u32(ctx: &mut GfxCtx<'_>, val: GVal) -> Id {
    if val.ty == GvTy::U32 {
        return val.id;
    }
    let t = ctx.t_u32;
    ctx.spv.op_convert_f_to_u(t, val.id)
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
