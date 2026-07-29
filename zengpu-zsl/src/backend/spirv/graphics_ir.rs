//! IR → SPIR-V lowering for vertex and fragment shaders.
//!
//! Consumes a [`GraphicsModule`] produced by the native parser. Dependency-free
//! (no `syn`/`quote`/`proc-macro2`): errors are plain `String`s. Mirrors the
//! emission of the (legacy, syn-based) `graphics` backend, driven by IR nodes.

use crate::ir::GfxInput;

/// GLSL.std.450 extended instruction opcodes.
mod glsl_op {
    pub const F_ABS: u32 = 4;
    pub const F_SIGN: u32 = 6;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TAN: u32 = 15;
    pub const TANH: u32 = 21;
    pub const POW: u32 = 26;
    pub const EXP: u32 = 27;
    pub const LOG: u32 = 28;
    pub const SQRT: u32 = 31;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const F_MIX: u32 = 46;
    pub const LENGTH: u32 = 66;
    pub const NORMALIZE: u32 = 69;
}

use std::collections::HashMap;

use crate::backend::spirv::builder::{Id, SpvBuilder, builtin, deco, sc};
use crate::frontend::diagnostic::Diagnostic;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::TypeId;
use crate::ir::GraphicsModule;

type R<T> = Result<T, String>;

#[derive(Clone, Copy)]
struct GVal {
    id: Id,
    ty: TypeId,
}

struct InputVar {
    var: Id,
    ty: TypeId,
    elem_ty: Id,
}

#[derive(Clone, Copy)]
struct ScalarInfo {
    pc_index: u32,
    ty: Id,
    gty: TypeId,
}

struct LocalVar {
    ptr: Id,
    ty: TypeId,
    elem_ty: Id,
}

struct Ctx<'a> {
    spv: &'a mut SpvBuilder,
    t_f32: Id,
    t_u32: Id,
    t_bool: Id,
    t_vec2: Id,
    t_vec3: Id,
    t_vec4: Id,
    t_mat4: Id,
    /// Canonical `TypeId`s for the six graphics value types, cached once at
    /// the top of lowering (see `lower_graphics_inner`) so every comparison
    /// below is a cheap `TypeId` equality check rather than an arena lookup.
    f32_id: TypeId,
    u32_id: TypeId,
    vec2_id: TypeId,
    vec3_id: TypeId,
    vec4_id: TypeId,
    mat4_id: TypeId,
    glsl_ext: Id,
    inputs: HashMap<String, InputVar>,
    scalar_params: HashMap<String, ScalarInfo>,
    buf_params: HashMap<String, u32>, // name → pc_index
    g_bufs_var: Option<Id>,
    t_ptr_ssbo_f32: Id,
    pc_var: Option<Id>,
    t_ptr_pc_u32: Id,
    t_ptr_pc_f32: Id,
    t_ptr_pc_mat4: Id,
    const_0_u32: Id,
    locals: HashMap<String, LocalVar>,
}

impl Ctx<'_> {
    fn spv_elem_ty(&self, ty: TypeId) -> Id {
        if ty == self.f32_id {
            self.t_f32
        } else if ty == self.u32_id {
            self.t_u32
        } else if ty == self.vec2_id {
            self.t_vec2
        } else if ty == self.vec3_id {
            self.t_vec3
        } else if ty == self.vec4_id {
            self.t_vec4
        } else {
            self.t_mat4
        }
    }
}

/// Lower a graphics [`GraphicsModule`] to SPIR-V words.
pub fn lower_graphics(module: &GraphicsModule) -> Result<Vec<u32>, Diagnostic> {
    lower_graphics_inner(module).map_err(Diagnostic::from)
}

fn lower_graphics_inner(module: &GraphicsModule) -> R<Vec<u32>> {
    let e = &module.entry;
    let is_fragment = e.is_fragment;

    // ── Canonical graphics-value TypeIds ─────────────────────────────────────
    // A local, mutable clone of the module's type arena: reuses every TypeId
    // the parser already interned and adds the six canonical graphics types
    // if a module happens not to reference one. Never written back to
    // `module` — it exists only so this lowering pass has stable `TypeId`s to
    // compare against.
    let mut types = module.types.clone();
    let f32_id = types.f32_id();
    let u32_id = types.u32_id();
    let vec2_id = types.vec_id(2);
    let vec3_id = types.vec_id(3);
    let vec4_id = types.vec_id(4);
    let mat4_id = types.mat4_id();

    let mut spv = SpvBuilder::new();
    spv.capability_shader();
    let glsl_ext = spv.ext_inst_import_glsl();
    spv.memory_model_logical_glsl450();

    let t_void = spv.type_void();
    let t_f32 = spv.type_float(32);
    let t_u32 = spv.type_int(32, false);
    let t_bool = spv.type_bool();
    let t_vec2 = spv.type_vector(t_f32, 2);
    let t_vec3 = spv.type_vector(t_f32, 3);
    let t_vec4 = spv.type_vector(t_f32, 4);
    let t_mat4 = spv.type_matrix(t_vec4, 4);

    let n_buf_params = e.buf_params.len() as u32;
    if !e.buf_params.is_empty() {
        spv.capability_runtime_descriptor_array();
    }

    // Input pointer types (created unconditionally for a deterministic layout).
    let t_ptr_in_f32 = spv.type_pointer(sc::INPUT, t_f32);
    let t_ptr_in_vec2 = spv.type_pointer(sc::INPUT, t_vec2);
    let t_ptr_in_vec3 = spv.type_pointer(sc::INPUT, t_vec3);
    let t_ptr_in_vec4 = spv.type_pointer(sc::INPUT, t_vec4);

    // Input variables, in location order.
    let mut sorted_inputs: Vec<&GfxInput> = e.inputs.iter().collect();
    sorted_inputs.sort_by_key(|i| i.location);
    let mut input_vars: HashMap<String, InputVar> = HashMap::new();
    let mut interface: Vec<Id> = Vec::new();
    for input in &sorted_inputs {
        let input_ty = input.ty;
        let (ptr_ty, elem_ty) = if input_ty == f32_id || input_ty == u32_id {
            (t_ptr_in_f32, t_f32)
        } else if input_ty == vec2_id {
            (t_ptr_in_vec2, t_vec2)
        } else if input_ty == vec3_id {
            (t_ptr_in_vec3, t_vec3)
        } else if input_ty == vec4_id {
            (t_ptr_in_vec4, t_vec4)
        } else {
            return Err("Mat4 inputs are not supported".into());
        };
        let var = spv.global_variable(ptr_ty, sc::INPUT);
        spv.decorate(var, deco::LOCATION, &[input.location]);
        interface.push(var);
        input_vars.insert(
            input.name.clone(),
            InputVar {
                var,
                ty: input_ty,
                elem_ty,
            },
        );
    }

    // Output: position (vertex) or color (fragment).
    let t_ptr_out_vec4 = spv.type_pointer(sc::OUTPUT, t_vec4);
    let out_var = spv.global_variable(t_ptr_out_vec4, sc::OUTPUT);
    if is_fragment {
        spv.decorate(out_var, deco::LOCATION, &[0]);
    } else {
        spv.decorate(out_var, deco::BUILT_IN, &[builtin::POSITION]);
    }
    interface.push(out_var);

    // Varying outputs (vertex only).
    let mut varying_out_vars: Vec<(Id, TypeId)> = Vec::new();
    for (loc, vty_id) in e.varyings.iter().enumerate() {
        let vty = *vty_id;
        let spv_elem = if vty == f32_id {
            t_f32
        } else if vty == u32_id {
            t_u32
        } else if vty == vec2_id {
            t_vec2
        } else if vty == vec3_id {
            t_vec3
        } else if vty == vec4_id {
            t_vec4
        } else {
            return Err("Mat4 varyings are not supported".into());
        };
        let t_ptr = spv.type_pointer(sc::OUTPUT, spv_elem);
        let var = spv.global_variable(t_ptr, sc::OUTPUT);
        spv.decorate(var, deco::LOCATION, &[loc as u32]);
        interface.push(var);
        varying_out_vars.push((var, vty));
    }

    // Bindless SSBO array.
    let (g_bufs_var, t_ptr_ssbo_f32) = if !e.buf_params.is_empty() {
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

    // Push-constant block: buf indices (u32) first, then scalars (u32/f32/mat4).
    let pc_var = if !e.buf_params.is_empty() || !e.scalar_params.is_empty() {
        let mut pc_members: Vec<Id> = (0..n_buf_params).map(|_| t_u32).collect();
        pc_members.extend(e.scalar_params.iter().map(|s| {
            if s.ty == u32_id {
                t_u32
            } else if s.ty == mat4_id {
                t_mat4
            } else {
                t_f32
            }
        }));
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);
        let mut offset: u32 = 0;
        for i in 0..n_buf_params {
            spv.member_decorate(t_pc_struct, i, deco::OFFSET, &[offset]);
            offset += 4;
        }
        for (i, s) in e.scalar_params.iter().enumerate() {
            let member = n_buf_params + i as u32;
            if s.ty == mat4_id {
                offset = (offset + 15) & !15;
                spv.member_decorate(t_pc_struct, member, deco::OFFSET, &[offset]);
                spv.member_decorate(t_pc_struct, member, deco::COL_MAJOR, &[]);
                spv.member_decorate(t_pc_struct, member, deco::MATRIX_STRIDE, &[16]);
                offset += 64;
            } else {
                spv.member_decorate(t_pc_struct, member, deco::OFFSET, &[offset]);
                offset += 4;
            }
        }
        let t_ptr_pc = spv.type_pointer(sc::PUSH_CONSTANT, t_pc_struct);
        Some(spv.global_variable(t_ptr_pc, sc::PUSH_CONSTANT))
    } else {
        None
    };

    // Entry point + execution mode.
    let t_fn = spv.type_function(t_void, &[]);
    let fn_id = spv.fresh_id();
    if is_fragment {
        spv.entry_point_fragment(fn_id, "main", &interface);
        spv.execution_mode_origin_upper_left(fn_id);
    } else {
        spv.entry_point_vertex(fn_id, "main", &interface);
    }

    let const_0_u32 = spv.constant_u32(t_u32, 0);

    let scalar_param_map: HashMap<String, ScalarInfo> = e
        .scalar_params
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (ty, gty) = if s.ty == u32_id {
                (t_u32, u32_id)
            } else if s.ty == mat4_id {
                (t_mat4, mat4_id)
            } else {
                (t_f32, f32_id)
            };
            (
                s.name.clone(),
                ScalarInfo {
                    pc_index: n_buf_params + i as u32,
                    ty,
                    gty,
                },
            )
        })
        .collect();

    let buf_param_map: HashMap<String, u32> = e
        .buf_params
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i as u32))
        .collect();

    // Push-constant pointer types.
    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    let mut t_ptr_pc_mat4 = Id(0);
    if pc_var.is_some() {
        if n_buf_params > 0 {
            t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
        }
        for s in &e.scalar_params {
            if s.ty == u32_id {
                if t_ptr_pc_u32 == Id(0) {
                    t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
                }
            } else if s.ty == mat4_id {
                if t_ptr_pc_mat4 == Id(0) {
                    t_ptr_pc_mat4 = spv.type_pointer(sc::PUSH_CONSTANT, t_mat4);
                }
            } else if t_ptr_pc_f32 == Id(0) {
                t_ptr_pc_f32 = spv.type_pointer(sc::PUSH_CONSTANT, t_f32);
            }
        }
    }

    // Begin function, hoist locals.
    spv.begin_function(t_void, fn_id, t_fn);
    spv.label();

    let mut t_ptr_func_f32 = Id(0);
    let mut t_ptr_func_u32 = Id(0);
    let mut t_ptr_func_vec2 = Id(0);
    let mut t_ptr_func_vec3 = Id(0);
    let mut t_ptr_func_vec4 = Id(0);
    let mut locals: HashMap<String, LocalVar> = HashMap::new();
    for (name, tid) in &e.locals {
        let gty = *tid;
        let (elem_ty, slot) = if gty == f32_id {
            (t_f32, &mut t_ptr_func_f32)
        } else if gty == u32_id {
            (t_u32, &mut t_ptr_func_u32)
        } else if gty == vec2_id {
            (t_vec2, &mut t_ptr_func_vec2)
        } else if gty == vec3_id {
            (t_vec3, &mut t_ptr_func_vec3)
        } else if gty == vec4_id {
            (t_vec4, &mut t_ptr_func_vec4)
        } else {
            return Err("Mat4 local variables are not supported".into());
        };
        if *slot == Id(0) {
            *slot = spv.type_pointer(sc::FUNCTION, elem_ty);
        }
        let ptr = spv.op_variable(*slot, sc::FUNCTION);
        locals.insert(
            name.clone(),
            LocalVar {
                ptr,
                ty: gty,
                elem_ty,
            },
        );
    }

    let mut ctx = Ctx {
        spv: &mut spv,
        t_f32,
        t_u32,
        t_bool,
        t_vec2,
        t_vec3,
        t_vec4,
        t_mat4,
        f32_id,
        u32_id,
        vec2_id,
        vec3_id,
        vec4_id,
        mat4_id,
        glsl_ext,
        inputs: input_vars,
        scalar_params: scalar_param_map,
        buf_params: buf_param_map,
        g_bufs_var,
        t_ptr_ssbo_f32,
        pc_var,
        t_ptr_pc_u32,
        t_ptr_pc_f32,
        t_ptr_pc_mat4,
        const_0_u32,
        locals,
    };

    for stmt in &e.body {
        lower_stmt(&mut ctx, stmt)?;
    }

    // Tail outputs: ret[0] → position/color, ret[1..] → varyings.
    let pos = lower_expr(&mut ctx, &e.ret[0])?;
    if pos.ty != ctx.vec4_id {
        return Err(format!(
            "shader return (position/color) must be f32x4, got {:?}",
            types.get(pos.ty)
        ));
    }
    ctx.spv.op_store(out_var, pos.id);
    for (i, (var_id, expected)) in varying_out_vars.iter().enumerate() {
        let v = lower_expr(&mut ctx, &e.ret[i + 1])?;
        if v.ty != *expected {
            return Err(format!(
                "varying[{i}] type mismatch: expected {:?}, got {:?}",
                types.get(*expected),
                types.get(v.ty)
            ));
        }
        ctx.spv.op_store(*var_id, v.id);
    }

    ctx.spv.op_return();
    ctx.spv.end_function();
    Ok(spv.finish())
}

// ── Statements ─────────────────────────────────────────────────────────────────

fn lower_stmt(ctx: &mut Ctx<'_>, stmt: &IrStmt) -> R<()> {
    match stmt {
        IrStmt::Let { name, init } => {
            let gty = ctx.locals.get(name).map(|l| l.ty).unwrap_or(ctx.f32_id);
            let val = lower_expr(ctx, init)?;
            let coerced = coerce(ctx, val, gty)?;
            let ptr = ctx
                .locals
                .get(name)
                .ok_or_else(|| format!("undeclared local `{name}`"))?
                .ptr;
            ctx.spv.op_store(ptr, coerced);
            Ok(())
        }
        IrStmt::AssignLocal { name, value } => {
            let (ptr, gty) = ctx
                .locals
                .get(name)
                .map(|l| (l.ptr, l.ty))
                .ok_or_else(|| format!("undeclared local `{name}`"))?;
            let val = lower_expr(ctx, value)?;
            let coerced = coerce(ctx, val, gty)?;
            ctx.spv.op_store(ptr, coerced);
            Ok(())
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            let pc_index = *ctx
                .buf_params
                .get(buf)
                .ok_or_else(|| format!("`{buf}` is not a buffer param"))?;
            let g_bufs = ctx.g_bufs_var.ok_or("internal: no g_bufs")?;
            let pc_var = ctx.pc_var.ok_or("internal: no push-constant block")?;
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, pc_index);
            let pc_chain = ctx
                .spv
                .op_access_chain(ctx.t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            let idx = lower_expr(ctx, index)?;
            let idx_id = scalar_to_u32(ctx, idx);
            let c0 = ctx.const_0_u32;
            let ptr_elem =
                ctx.spv
                    .op_access_chain(ctx.t_ptr_ssbo_f32, g_bufs, &[buf_idx, c0, idx_id]);
            let rhs = lower_expr(ctx, value)?;
            let rhs_id = scalar_to_f32(ctx, rhs);
            ctx.spv.op_store(ptr_elem, rhs_id);
            Ok(())
        }
        IrStmt::If { cond, then, else_ } => {
            let cond_id = lower_condition(ctx, cond)?;
            let then_label = ctx.spv.fresh_id();
            let merge_label = ctx.spv.fresh_id();
            if let Some(else_block) = else_ {
                let else_label = ctx.spv.fresh_id();
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, then_label, else_label);
                ctx.spv.label_with_id(then_label);
                for s in then {
                    lower_stmt(ctx, s)?;
                }
                ctx.spv.op_branch(merge_label);
                ctx.spv.label_with_id(else_label);
                for s in else_block {
                    lower_stmt(ctx, s)?;
                }
                ctx.spv.op_branch(merge_label);
            } else {
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, then_label, merge_label);
                ctx.spv.label_with_id(then_label);
                for s in then {
                    lower_stmt(ctx, s)?;
                }
                ctx.spv.op_branch(merge_label);
            }
            ctx.spv.label_with_id(merge_label);
            Ok(())
        }
        IrStmt::Eval(expr) => {
            lower_expr(ctx, expr)?;
            Ok(())
        }
        IrStmt::For { .. } => Err("for-loops are not supported in vertex/fragment shaders".into()),
        IrStmt::AssignShared { .. } | IrStmt::Barrier => {
            Err("workgroup operations are only available in compute shaders".into())
        }
        IrStmt::AtomicAdd { .. } => Err("atomic_add is only available in compute shaders".into()),
    }
}

fn lower_condition(ctx: &mut Ctx<'_>, expr: &IrExpr) -> R<Id> {
    if let IrExpr::Binary { op, lhs, rhs } = expr {
        if matches!(op, IrBinOp::And | IrBinOp::Or) {
            let l = lower_condition(ctx, lhs)?;
            let r = lower_condition(ctx, rhs)?;
            let t_bool = ctx.t_bool;
            return Ok(match op {
                IrBinOp::And => ctx.spv.op_logical_and(t_bool, l, r),
                _ => ctx.spv.op_logical_or(t_bool, l, r),
            });
        }
        if matches!(
            op,
            IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne
        ) {
            let lhs = lower_expr(ctx, lhs)?;
            let rhs = lower_expr(ctx, rhs)?;
            let (lhs, rhs) = unify_scalars(ctx, lhs, rhs);
            let b = ctx.t_bool;
            let id = if lhs.ty == ctx.f32_id {
                match op {
                    IrBinOp::Lt => ctx.spv.op_ford_lt(b, lhs.id, rhs.id),
                    IrBinOp::Le => ctx.spv.op_ford_le(b, lhs.id, rhs.id),
                    IrBinOp::Gt => ctx.spv.op_ford_gt(b, lhs.id, rhs.id),
                    IrBinOp::Ge => ctx.spv.op_ford_ge(b, lhs.id, rhs.id),
                    IrBinOp::Eq => ctx.spv.op_ford_eq(b, lhs.id, rhs.id),
                    IrBinOp::Ne => ctx.spv.op_ford_ne(b, lhs.id, rhs.id),
                    _ => unreachable!(),
                }
            } else if lhs.ty == ctx.u32_id {
                match op {
                    IrBinOp::Lt => ctx.spv.op_ult(b, lhs.id, rhs.id),
                    IrBinOp::Le => ctx.spv.op_ule(b, lhs.id, rhs.id),
                    IrBinOp::Gt => ctx.spv.op_ugt(b, lhs.id, rhs.id),
                    IrBinOp::Ge => ctx.spv.op_uge(b, lhs.id, rhs.id),
                    IrBinOp::Eq => ctx.spv.op_iequal(b, lhs.id, rhs.id),
                    IrBinOp::Ne => ctx.spv.op_inot_equal(b, lhs.id, rhs.id),
                    _ => unreachable!(),
                }
            } else {
                return Err("comparisons require f32 or u32 operands".into());
            };
            return Ok(id);
        }
    }
    Err("`if` condition must be a comparison (< > <= >= == !=) or logical (&& ||)".into())
}

// ── Expressions ────────────────────────────────────────────────────────────────

fn lower_expr(ctx: &mut Ctx<'_>, expr: &IrExpr) -> R<GVal> {
    match expr {
        IrExpr::LitU32(v) => {
            let id = ctx.spv.constant_u32(ctx.t_u32, *v);
            Ok(GVal { id, ty: ctx.u32_id })
        }
        IrExpr::LitF32(v) => {
            let id = ctx.spv.constant_f32(ctx.t_f32, *v);
            Ok(GVal { id, ty: ctx.f32_id })
        }
        IrExpr::Local(name) => {
            let l = ctx
                .locals
                .get(name)
                .ok_or_else(|| format!("unknown local `{name}`"))?;
            let (ty, ptr, elem_ty) = (l.ty, l.ptr, l.elem_ty);
            let id = ctx.spv.op_load(elem_ty, ptr);
            Ok(GVal { id, ty })
        }
        IrExpr::Input(name) => {
            let v = ctx
                .inputs
                .get(name)
                .ok_or_else(|| format!("unknown input `{name}`"))?;
            let (ty, var, elem_ty) = (v.ty, v.var, v.elem_ty);
            let id = ctx.spv.op_load(elem_ty, var);
            Ok(GVal { id, ty })
        }
        IrExpr::ScalarParam(name) => {
            let info = *ctx
                .scalar_params
                .get(name)
                .ok_or_else(|| format!("unknown push field `{name}`"))?;
            let pc_var = ctx.pc_var.ok_or("internal: no push-constant block")?;
            let pc_ptr_ty = if info.gty == ctx.u32_id {
                ctx.t_ptr_pc_u32
            } else if info.gty == ctx.f32_id {
                ctx.t_ptr_pc_f32
            } else if info.gty == ctx.mat4_id {
                ctx.t_ptr_pc_mat4
            } else {
                return Err("unsupported push-constant type".into());
            };
            let pc_idx = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
            let chain = ctx.spv.op_access_chain(pc_ptr_ty, pc_var, &[pc_idx]);
            let id = ctx.spv.op_load(info.ty, chain);
            Ok(GVal { id, ty: info.gty })
        }
        IrExpr::BufferLoad { buf, index } => {
            let pc_index = *ctx
                .buf_params
                .get(buf)
                .ok_or_else(|| format!("`{buf}` is not a buffer param"))?;
            let g_bufs = ctx.g_bufs_var.ok_or("internal: no g_bufs")?;
            let pc_var = ctx.pc_var.ok_or("internal: no push-constant block")?;
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, pc_index);
            let pc_chain = ctx
                .spv
                .op_access_chain(ctx.t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            let idx = lower_expr(ctx, index)?;
            let idx_id = scalar_to_u32(ctx, idx);
            let c0 = ctx.const_0_u32;
            let ptr_elem =
                ctx.spv
                    .op_access_chain(ctx.t_ptr_ssbo_f32, g_bufs, &[buf_idx, c0, idx_id]);
            let id = ctx.spv.op_load(ctx.t_f32, ptr_elem);
            Ok(GVal { id, ty: ctx.f32_id })
        }
        IrExpr::FieldAccess { base, component } => {
            let composite = lower_expr(ctx, base)?;
            let t_f32 = ctx.t_f32;
            let id = ctx
                .spv
                .op_composite_extract(t_f32, composite.id, &[*component]);
            Ok(GVal { id, ty: ctx.f32_id })
        }
        IrExpr::Construct { dim, args, .. } => {
            let (gty, spv_ty) = match dim {
                2 => (ctx.vec2_id, ctx.t_vec2),
                3 => (ctx.vec3_id, ctx.t_vec3),
                _ => (ctx.vec4_id, ctx.t_vec4),
            };
            if args.len() != *dim as usize {
                return Err(format!("f32x{dim} takes {dim} args, got {}", args.len()));
            }
            let mut comps = Vec::with_capacity(args.len());
            for a in args {
                let v = lower_expr(ctx, a)?;
                comps.push(scalar_to_f32(ctx, v));
            }
            let id = ctx.spv.op_composite_construct(spv_ty, &comps);
            Ok(GVal { id, ty: gty })
        }
        IrExpr::Builtin { func, args } => lower_builtin(ctx, *func, args),
        IrExpr::Neg(inner) => {
            let v = lower_expr(ctx, inner)?;
            if v.ty == ctx.u32_id {
                Err("cannot negate u32".into())
            } else if v.ty == ctx.mat4_id {
                Err("cannot negate mat4x4".into())
            } else {
                let ty = ctx.spv_elem_ty(v.ty);
                let id = ctx.spv.op_fnegate(ty, v.id);
                Ok(GVal { id, ty: v.ty })
            }
        }
        IrExpr::Binary { op, lhs, rhs } => {
            let l = lower_expr(ctx, lhs)?;
            let r = lower_expr(ctx, rhs)?;
            lower_arith(ctx, *op, l, r)
        }
        IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::SharedLoad { .. } => {
            Err("workgroup expressions are only available in compute shaders".into())
        }
    }
}

fn lower_builtin(ctx: &mut Ctx<'_>, func: BuiltinFn, args: &[IrExpr]) -> R<GVal> {
    let name = func.name();
    match func {
        BuiltinFn::U32
        | BuiltinFn::IsNan
        | BuiltinFn::IsInf
        | BuiltinFn::IsFinite
        | BuiltinFn::MulAddF32
        | BuiltinFn::SubgroupId
        | BuiltinFn::SubgroupLocalId
        | BuiltinFn::SubgroupSize
        | BuiltinFn::SubgroupAny
        | BuiltinFn::SubgroupAll
        | BuiltinFn::SubgroupBallot
        | BuiltinFn::SubgroupBroadcast
        | BuiltinFn::SubgroupShuffle
        | BuiltinFn::SubgroupShuffleXor
        | BuiltinFn::SubgroupShuffleUp
        | BuiltinFn::SubgroupShuffleDown
        | BuiltinFn::SubgroupReduceAdd
        | BuiltinFn::SubgroupReduceMin
        | BuiltinFn::SubgroupReduceMax
        | BuiltinFn::SubgroupReduceAnd
        | BuiltinFn::SubgroupReduceOr
        | BuiltinFn::SubgroupScanInclusiveAdd
        | BuiltinFn::SubgroupScanExclusiveAdd
        | BuiltinFn::WorkgroupReduceAdd
        | BuiltinFn::WorkgroupReduceMin
        | BuiltinFn::WorkgroupReduceMax => Err(format!(
            "{}() is only available in compute shaders",
            func.name()
        )),
        BuiltinFn::Abs
        | BuiltinFn::Sign
        | BuiltinFn::Exp
        | BuiltinFn::Tanh
        | BuiltinFn::Sin
        | BuiltinFn::Cos
        | BuiltinFn::Tan
        | BuiltinFn::Log
        | BuiltinFn::Sqrt
        | BuiltinFn::Floor
        | BuiltinFn::Ceil
        | BuiltinFn::Fract
        | BuiltinFn::Normalize => {
            if args.len() != 1 {
                return Err(format!("{name}() takes 1 arg"));
            }
            let v = lower_expr(ctx, &args[0])?;
            let ok = if matches!(func, BuiltinFn::Normalize) {
                is_vec(ctx, v.ty)
            } else {
                v.ty == ctx.f32_id || is_vec(ctx, v.ty)
            };
            if !ok {
                return Err(format!("{name}() requires f32/f32x2/f32x3/f32x4"));
            }
            let opcode = match func {
                BuiltinFn::Abs => glsl_op::F_ABS,
                BuiltinFn::Sign => glsl_op::F_SIGN,
                BuiltinFn::Exp => glsl_op::EXP,
                BuiltinFn::Tanh => glsl_op::TANH,
                BuiltinFn::Sin => glsl_op::SIN,
                BuiltinFn::Cos => glsl_op::COS,
                BuiltinFn::Tan => glsl_op::TAN,
                BuiltinFn::Log => glsl_op::LOG,
                BuiltinFn::Sqrt => glsl_op::SQRT,
                BuiltinFn::Floor => glsl_op::FLOOR,
                BuiltinFn::Ceil => glsl_op::CEIL,
                BuiltinFn::Fract => glsl_op::FRACT,
                BuiltinFn::Normalize => glsl_op::NORMALIZE,
                _ => unreachable!(),
            };
            let ty = ctx.spv_elem_ty(v.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(ty, glsl, opcode, &[v.id]);
            Ok(GVal { id, ty: v.ty })
        }
        BuiltinFn::Length => {
            if args.len() != 1 {
                return Err("length() takes 1 arg".into());
            }
            let v = lower_expr(ctx, &args[0])?;
            if !is_vec(ctx, v.ty) {
                return Err("length() requires f32x2/f32x3/f32x4".into());
            }
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(t_f32, glsl, glsl_op::LENGTH, &[v.id]);
            Ok(GVal { id, ty: ctx.f32_id })
        }
        BuiltinFn::Dot => {
            if args.len() != 2 {
                return Err("dot() requires two vectors of the same type".into());
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            if !is_vec(ctx, a.ty) || a.ty != b.ty {
                return Err("dot() requires two vectors of the same type".into());
            }
            let t_f32 = ctx.t_f32;
            let id = ctx.spv.op_dot(t_f32, a.id, b.id);
            Ok(GVal { id, ty: ctx.f32_id })
        }
        BuiltinFn::Extend => {
            if args.len() != 2 {
                return Err(".extend() requires a f32x3 receiver".into());
            }
            let b = lower_expr(ctx, &args[0])?;
            if b.ty != ctx.vec3_id {
                return Err(".extend() requires a f32x3 receiver".into());
            }
            let ext = lower_expr(ctx, &args[1])?;
            let ext_id = scalar_to_f32(ctx, ext);
            let t_f32 = ctx.t_f32;
            let x = ctx.spv.op_composite_extract(t_f32, b.id, &[0]);
            let y = ctx.spv.op_composite_extract(t_f32, b.id, &[1]);
            let z = ctx.spv.op_composite_extract(t_f32, b.id, &[2]);
            let t_vec4 = ctx.t_vec4;
            let id = ctx.spv.op_composite_construct(t_vec4, &[x, y, z, ext_id]);
            Ok(GVal {
                id,
                ty: ctx.vec4_id,
            })
        }
        BuiltinFn::Min | BuiltinFn::Max | BuiltinFn::Pow => {
            if args.len() != 2 {
                return Err(format!("{name}(a, b) takes 2 args"));
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            if !(a.ty == ctx.f32_id || is_vec(ctx, a.ty)) || a.ty != b.ty {
                return Err(format!("{name}() requires matching f32/f32xN args"));
            }
            let opcode = match func {
                BuiltinFn::Min => glsl_op::F_MIN,
                BuiltinFn::Max => glsl_op::F_MAX,
                _ => glsl_op::POW,
            };
            let ty = ctx.spv_elem_ty(a.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(ty, glsl, opcode, &[a.id, b.id]);
            Ok(GVal { id, ty: a.ty })
        }
        BuiltinFn::Clamp => {
            if args.len() != 3 {
                return Err("clamp(x, lo, hi) takes 3 args".into());
            }
            let x = lower_expr(ctx, &args[0])?;
            let lo = lower_expr(ctx, &args[1])?;
            let hi = lower_expr(ctx, &args[2])?;
            if !(x.ty == ctx.f32_id || is_vec(ctx, x.ty)) || x.ty != lo.ty || x.ty != hi.ty {
                return Err("clamp() requires matching f32/f32xN args".into());
            }
            let ty = ctx.spv_elem_ty(x.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(ty, glsl, glsl_op::F_CLAMP, &[x.id, lo.id, hi.id]);
            Ok(GVal { id, ty: x.ty })
        }
        BuiltinFn::Mix => {
            if args.len() != 3 {
                return Err("mix(a, b, t) takes 3 args".into());
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            let t = lower_expr(ctx, &args[2])?;
            if !(a.ty == ctx.f32_id || is_vec(ctx, a.ty)) || a.ty != b.ty || a.ty != t.ty {
                return Err("mix() requires matching f32/f32xN args".into());
            }
            let ty = ctx.spv_elem_ty(a.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(ty, glsl, glsl_op::F_MIX, &[a.id, b.id, t.id]);
            Ok(GVal { id, ty: a.ty })
        }
    }
}

fn lower_arith(ctx: &mut Ctx<'_>, op: IrBinOp, lhs: GVal, rhs: GVal) -> R<GVal> {
    // Mat4 * Vec4.
    if lhs.ty == ctx.mat4_id {
        if !matches!(op, IrBinOp::Mul) {
            return Err("mat4x4 only supports `*`".into());
        }
        if rhs.ty != ctx.vec4_id {
            return Err("mat4x4 * requires f32x4 on the right".into());
        }
        let t_vec4 = ctx.t_vec4;
        let id = ctx.spv.op_matrix_times_vector(t_vec4, lhs.id, rhs.id);
        return Ok(GVal {
            id,
            ty: ctx.vec4_id,
        });
    }
    // Vector op vector (component-wise).
    if is_vec(ctx, lhs.ty) && is_vec(ctx, rhs.ty) {
        if lhs.ty != rhs.ty {
            return Err(format!(
                "vec op requires matching types; got {:?} and {:?}",
                lhs.ty, rhs.ty
            ));
        }
        let ty = ctx.spv_elem_ty(lhs.ty);
        let id = match op {
            IrBinOp::Add => ctx.spv.op_fadd(ty, lhs.id, rhs.id),
            IrBinOp::Sub => ctx.spv.op_fsub(ty, lhs.id, rhs.id),
            IrBinOp::Mul => ctx.spv.op_fmul(ty, lhs.id, rhs.id),
            IrBinOp::Div => ctx.spv.op_fdiv(ty, lhs.id, rhs.id),
            _ => return Err("vec op: use +, -, *, /".into()),
        };
        return Ok(GVal { id, ty: lhs.ty });
    }
    // Vector * scalar / scalar * vector.
    if matches!(op, IrBinOp::Mul) {
        let (vec_val, scalar_val) = match (is_vec(ctx, lhs.ty), is_vec(ctx, rhs.ty)) {
            (true, false) => (lhs, rhs),
            (false, true) => (rhs, lhs),
            _ => (lhs, rhs),
        };
        if is_vec(ctx, vec_val.ty) && !is_vec(ctx, scalar_val.ty) {
            let scalar_f32 = scalar_to_f32(ctx, scalar_val);
            let ty = ctx.spv_elem_ty(vec_val.ty);
            let id = ctx.spv.op_vector_times_scalar(ty, vec_val.id, scalar_f32);
            return Ok(GVal { id, ty: vec_val.ty });
        }
    }
    // Scalar arithmetic.
    if !(lhs.ty == ctx.f32_id || lhs.ty == ctx.u32_id) {
        return Err("binary arithmetic only on scalar or vector types".into());
    }
    let (lhs, rhs) = unify_scalars(ctx, lhs, rhs);
    let ty = ctx.spv_elem_ty(lhs.ty);
    let id = if lhs.ty == ctx.f32_id {
        match op {
            IrBinOp::Add => ctx.spv.op_fadd(ty, lhs.id, rhs.id),
            IrBinOp::Sub => ctx.spv.op_fsub(ty, lhs.id, rhs.id),
            IrBinOp::Mul => ctx.spv.op_fmul(ty, lhs.id, rhs.id),
            IrBinOp::Div => ctx.spv.op_fdiv(ty, lhs.id, rhs.id),
            _ => return Err("unsupported op; use +, -, *, /".into()),
        }
    } else {
        match op {
            IrBinOp::Add => ctx.spv.op_iadd(ty, lhs.id, rhs.id),
            IrBinOp::Sub => ctx.spv.op_isub(ty, lhs.id, rhs.id),
            IrBinOp::Mul => ctx.spv.op_imul(ty, lhs.id, rhs.id),
            IrBinOp::Div => ctx.spv.op_udiv(ty, lhs.id, rhs.id),
            _ => return Err("unsupported op; use +, -, *, /".into()),
        }
    };
    Ok(GVal { id, ty: lhs.ty })
}

// ── Coercions ──────────────────────────────────────────────────────────────────

fn is_vec(ctx: &Ctx<'_>, t: TypeId) -> bool {
    t == ctx.vec2_id || t == ctx.vec3_id || t == ctx.vec4_id
}

fn scalar_to_f32(ctx: &mut Ctx<'_>, v: GVal) -> Id {
    if v.ty == ctx.f32_id {
        return v.id;
    }
    let t = ctx.t_f32;
    ctx.spv.op_convert_u_to_f(t, v.id)
}

fn scalar_to_u32(ctx: &mut Ctx<'_>, v: GVal) -> Id {
    if v.ty == ctx.u32_id {
        return v.id;
    }
    let t = ctx.t_u32;
    ctx.spv.op_convert_f_to_u(t, v.id)
}

fn coerce(ctx: &mut Ctx<'_>, v: GVal, target: TypeId) -> R<Id> {
    if v.ty == target {
        return Ok(v.id);
    }
    if v.ty == ctx.u32_id && target == ctx.f32_id {
        let t = ctx.t_f32;
        Ok(ctx.spv.op_convert_u_to_f(t, v.id))
    } else if v.ty == ctx.f32_id && target == ctx.u32_id {
        let t = ctx.t_u32;
        Ok(ctx.spv.op_convert_f_to_u(t, v.id))
    } else {
        Err(format!(
            "cannot coerce {:?} to {:?}",
            v.ty, target
        ))
    }
}

fn unify_scalars(ctx: &mut Ctx<'_>, lhs: GVal, rhs: GVal) -> (GVal, GVal) {
    if lhs.ty == rhs.ty {
        return (lhs, rhs);
    }
    let l = scalar_to_f32(ctx, lhs);
    let r = scalar_to_f32(ctx, rhs);
    (
        GVal {
            id: l,
            ty: ctx.f32_id,
        },
        GVal {
            id: r,
            ty: ctx.f32_id,
        },
    )
}
