//! SPIR-V lowering of vertex and fragment shaders from IR nodes.
//!
//! Dependency-free: errors are plain `String`s. Mirrors the syn-based `graphics` backend.

use crate::ir::layout::graphics_member;
use crate::ir::{GfxBuiltin, GfxInput, GfxSlot};

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
    pub const ROUND_EVEN: u32 = 2;
    pub const INVERSE_SQRT: u32 = 32;
    pub const LOG: u32 = 28;
    pub const SQRT: u32 = 31;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const U_MIN: u32 = 38;
    pub const U_MAX: u32 = 41;
    pub const U_CLAMP: u32 = 44;
    pub const F_MIX: u32 = 46;
    pub const STEP: u32 = 48;
    pub const SMOOTH_STEP: u32 = 49;
    pub const LENGTH: u32 = 66;
    pub const NORMALIZE: u32 = 69;
}

use std::collections::HashMap;

use crate::backend::spirv::builder::{Id, SpvBuilder, builtin, deco, sc};
use crate::frontend::diagnostic::Diagnostic;
use crate::ir::GraphicsModule;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::{ArrayLen, Type, TypeId};

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

#[derive(Clone, Copy)]
struct TextureTable {
    var: Id,
    t_sampled: Id,
    t_ptr: Id,
}

struct ArrayInfo {
    element: TypeId,
    elem_ty: Id,
    elem_ptr: Id,
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
    /// Canonical `TypeId`s for the six graphics value types, cached in `lower_graphics_inner`.
    f32_id: TypeId,
    u32_id: TypeId,
    vec2_id: TypeId,
    vec3_id: TypeId,
    vec4_id: TypeId,
    mat4_id: TypeId,
    glsl_ext: Id,
    inputs: HashMap<String, InputVar>,
    scalar_params: HashMap<String, ScalarInfo>,
    buf_params: HashMap<String, u32>,
    g_bufs_var: Option<Id>,
    t_ptr_ssbo_f32: Id,
    textures: Option<TextureTable>,
    is_fragment: bool,
    pc_var: Option<Id>,
    pc_ptrs: HashMap<u32, Id>,
    const_0_u32: Id,
    locals: HashMap<String, LocalVar>,
    arrays: HashMap<String, ArrayInfo>,
}

impl Ctx<'_> {
    fn pc_ptr(&mut self, pointee: Id) -> Id {
        if let Some(&ptr) = self.pc_ptrs.get(&pointee.0) {
            return ptr;
        }
        let ptr = self.spv.type_pointer(sc::PUSH_CONSTANT, pointee);
        self.pc_ptrs.insert(pointee.0, ptr);
        ptr
    }

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

pub fn lower_graphics(module: &GraphicsModule) -> Result<Vec<u32>, Diagnostic> {
    lower_graphics_inner(module).map_err(Diagnostic::from)
}

fn lower_graphics_inner(module: &GraphicsModule) -> R<Vec<u32>> {
    let e = &module.entry;
    let is_fragment = e.is_fragment;

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
    let spv_value_ty = |ty: TypeId| {
        if ty == u32_id {
            t_u32
        } else if ty == vec2_id {
            t_vec2
        } else if ty == vec3_id {
            t_vec3
        } else if ty == vec4_id {
            t_vec4
        } else if ty == mat4_id {
            t_mat4
        } else {
            t_f32
        }
    };

    let n_buf_params = e.buf_params.len() as u32;
    if !e.buf_params.is_empty() || e.textures.is_some() {
        spv.capability_runtime_descriptor_array();
    }

    let t_ptr_in_f32 = spv.type_pointer(sc::INPUT, t_f32);
    let t_ptr_in_u32 = spv.type_pointer(sc::INPUT, t_u32);
    let t_ptr_in_vec2 = spv.type_pointer(sc::INPUT, t_vec2);
    let t_ptr_in_vec3 = spv.type_pointer(sc::INPUT, t_vec3);
    let t_ptr_in_vec4 = spv.type_pointer(sc::INPUT, t_vec4);

    let mut sorted_inputs: Vec<&GfxInput> = e.inputs.iter().collect();
    sorted_inputs.sort_by_key(|i| match i.slot {
        GfxSlot::Location(l) => l,
        GfxSlot::Builtin(_) => u32::MAX,
    });
    let mut input_vars: HashMap<String, InputVar> = HashMap::new();
    let mut interface: Vec<Id> = Vec::new();
    for input in &sorted_inputs {
        let input_ty = input.ty;
        let (ptr_ty, elem_ty) = if input_ty == f32_id {
            (t_ptr_in_f32, t_f32)
        } else if input_ty == u32_id {
            (t_ptr_in_u32, t_u32)
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
        match input.slot {
            GfxSlot::Location(location) => {
                spv.decorate(var, deco::LOCATION, &[location]);
                if is_fragment && input_ty == u32_id {
                    spv.decorate(var, deco::FLAT, &[]);
                }
            }
            GfxSlot::Builtin(b) => {
                let which = match b {
                    GfxBuiltin::VertexIndex => builtin::VERTEX_INDEX,
                    GfxBuiltin::InstanceIndex => builtin::INSTANCE_INDEX,
                    GfxBuiltin::FragCoord => builtin::FRAG_COORD,
                };
                spv.decorate(var, deco::BUILT_IN, &[which]);
            }
        }
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

    let mut out_vars: Vec<(Id, TypeId)> = Vec::new();
    if is_fragment {
        let dual_source = e.outputs.iter().any(|o| o.blend_src != 0);
        for o in &e.outputs {
            let t_ptr = spv.type_pointer(sc::OUTPUT, spv_value_ty(o.ty));
            let var = spv.global_variable(t_ptr, sc::OUTPUT);
            spv.decorate(var, deco::LOCATION, &[o.location]);
            if dual_source {
                spv.decorate(var, deco::INDEX, &[o.blend_src]);
            }
            interface.push(var);
            out_vars.push((var, o.ty));
        }
    } else {
        let t_ptr_out_vec4 = spv.type_pointer(sc::OUTPUT, t_vec4);
        let var = spv.global_variable(t_ptr_out_vec4, sc::OUTPUT);
        spv.decorate(var, deco::BUILT_IN, &[builtin::POSITION]);
        interface.push(var);
        out_vars.push((var, vec4_id));
    }

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
        if vty == u32_id {
            spv.decorate(var, deco::FLAT, &[]);
        }
        interface.push(var);
        varying_out_vars.push((var, vty));
    }

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

    let textures = e.textures.as_ref().map(|_| {
        let t_image = spv.type_image_2d(t_f32);
        let t_sampled = spv.type_sampled_image(t_image);
        let t_table = spv.type_runtime_array(t_sampled);
        let t_ptr_table = spv.type_pointer(sc::UNIFORM_CONSTANT, t_table);
        let var = spv.global_variable(t_ptr_table, sc::UNIFORM_CONSTANT);
        spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
        spv.decorate(var, deco::BINDING, &[1]);
        TextureTable {
            var,
            t_sampled,
            t_ptr: spv.type_pointer(sc::UNIFORM_CONSTANT, t_sampled),
        }
    });

    let pc_var = if !e.buf_params.is_empty() || !e.scalar_params.is_empty() {
        let mut pc_members: Vec<Id> = (0..n_buf_params).map(|_| t_u32).collect();
        pc_members.extend(e.scalar_params.iter().map(|s| spv_value_ty(s.ty)));
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);
        let mut offset: u32 = 0;
        for i in 0..n_buf_params {
            spv.member_decorate(t_pc_struct, i, deco::OFFSET, &[offset]);
            offset += 4;
        }
        for (i, s) in e.scalar_params.iter().enumerate() {
            let member = n_buf_params + i as u32;
            let (align, size) = graphics_member(s.ty, &mut types);
            offset = offset.next_multiple_of(align);
            spv.member_decorate(t_pc_struct, member, deco::OFFSET, &[offset]);
            if s.ty == mat4_id {
                spv.member_decorate(t_pc_struct, member, deco::COL_MAJOR, &[]);
                spv.member_decorate(t_pc_struct, member, deco::MATRIX_STRIDE, &[16]);
            }
            offset += size;
        }
        let t_ptr_pc = spv.type_pointer(sc::PUSH_CONSTANT, t_pc_struct);
        Some(spv.global_variable(t_ptr_pc, sc::PUSH_CONSTANT))
    } else {
        None
    };

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
            (
                s.name.clone(),
                ScalarInfo {
                    pc_index: n_buf_params + i as u32,
                    ty: spv_value_ty(s.ty),
                    gty: s.ty,
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

    spv.begin_function(t_void, fn_id, t_fn);
    spv.label();

    let mut t_ptr_func_f32 = Id(0);
    let mut t_ptr_func_u32 = Id(0);
    let mut t_ptr_func_vec2 = Id(0);
    let mut t_ptr_func_vec3 = Id(0);
    let mut t_ptr_func_vec4 = Id(0);
    let mut t_ptr_func_mat4 = Id(0);
    let mut locals: HashMap<String, LocalVar> = HashMap::new();
    let mut arrays: HashMap<String, ArrayInfo> = HashMap::new();
    for (name, tid) in &e.locals {
        let gty = *tid;
        if let Type::Array {
            element,
            length: ArrayLen::Const(n),
        } = *types.get(gty)
        {
            let elem_ty = spv_value_ty(element);
            let len = spv.constant_u32(t_u32, n);
            let t_array = spv.type_array(elem_ty, len);
            let t_ptr = spv.type_pointer_global(sc::FUNCTION, t_array);
            let ptr = spv.op_variable(t_ptr, sc::FUNCTION);
            locals.insert(
                name.clone(),
                LocalVar {
                    ptr,
                    ty: gty,
                    elem_ty: t_array,
                },
            );
            arrays.insert(
                name.clone(),
                ArrayInfo {
                    element,
                    elem_ty,
                    elem_ptr: spv.type_pointer(sc::FUNCTION, elem_ty),
                },
            );
            continue;
        }
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
        } else if gty == mat4_id {
            (t_mat4, &mut t_ptr_func_mat4)
        } else {
            return Err("unsupported local variable type".into());
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
        textures,
        is_fragment,
        pc_var,
        pc_ptrs: HashMap::new(),
        arrays,
        const_0_u32,
        locals,
    };

    for stmt in &e.body {
        lower_stmt(&mut ctx, stmt)?;
    }

    for (i, &(var, expected)) in out_vars.iter().enumerate() {
        let v = lower_expr(&mut ctx, &e.ret[i])?;
        let id = broadcast(&mut ctx, v, expected).map_err(|_| {
            format!(
                "output {i} must be {:?}, got {:?}",
                types.get(expected),
                types.get(v.ty)
            )
        })?;
        ctx.spv.op_store(var, id);
    }
    let first_varying = out_vars.len();
    for (i, (var_id, expected)) in varying_out_vars.iter().enumerate() {
        let v = lower_expr(&mut ctx, &e.ret[first_varying + i])?;
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

fn lower_stmt(ctx: &mut Ctx<'_>, stmt: &IrStmt) -> R<()> {
    match stmt {
        IrStmt::FragStore { .. } => Err("frag_store is only available in compute shaders".into()),
        IrStmt::Let {
            name,
            init: IrExpr::ArrayLit(elems),
        } => {
            let element = ctx
                .arrays
                .get(name)
                .map(|a| a.element)
                .ok_or_else(|| format!("undeclared array `{name}`"))?;
            let (ptr, t_array) = ctx
                .locals
                .get(name)
                .map(|l| (l.ptr, l.elem_ty))
                .ok_or_else(|| format!("undeclared local `{name}`"))?;
            let mut ids = Vec::with_capacity(elems.len());
            for el in elems {
                let v = lower_expr(ctx, el)?;
                ids.push(
                    broadcast(ctx, v, element)
                        .map_err(|_| format!("`{name}` mixes element types"))?,
                );
            }
            let value = ctx.spv.op_composite_construct(t_array, &ids);
            ctx.spv.op_store(ptr, value);
            Ok(())
        }
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
            let pc_ptr = ctx.pc_ptr(ctx.t_u32);
            let pc_chain = ctx.spv.op_access_chain(pc_ptr, pc_var, &[pc_field]);
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
        IrStmt::Discard => {
            ctx.spv.op_kill();
            ctx.spv.label();
            Ok(())
        }
        IrStmt::For { .. } => Err("for-loops are not supported in vertex/fragment shaders".into()),
        IrStmt::AssignShared { .. } | IrStmt::Barrier | IrStmt::SharedBarrier => {
            Err("workgroup operations are only available in compute shaders".into())
        }
        IrStmt::AtomicAdd { .. } => Err("atomic_add is only available in compute shaders".into()),
        IrStmt::ImageStore { .. } => {
            Err("image stores are only available in compute shaders".into())
        }
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

fn lower_expr(ctx: &mut Ctx<'_>, expr: &IrExpr) -> R<GVal> {
    match expr {
        IrExpr::FragLoad { .. } => Err("frag_load is only available in compute shaders".into()),
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
            let pc_ptr_ty = ctx.pc_ptr(info.ty);
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
            let pc_ptr = ctx.pc_ptr(ctx.t_u32);
            let pc_chain = ctx.spv.op_access_chain(pc_ptr, pc_var, &[pc_field]);
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
        IrExpr::Swizzle { base, lanes } => {
            let v = lower_expr(ctx, base)?;
            let width = lanes_of(ctx, v.ty);
            if width == 0 {
                return Err("swizzles need a f32x2/f32x3/f32x4 value".into());
            }
            if lanes.iter().any(|&l| l >= width) {
                return Err(format!("swizzle reaches past a {width}-lane vector"));
            }
            let (gty, spv_ty) = ctx.vec_of(lanes.len() as u32);
            let id = ctx.spv.op_vector_shuffle(spv_ty, v.id, lanes);
            Ok(GVal { id, ty: gty })
        }
        IrExpr::Construct { dim, args, .. } => {
            let (gty, spv_ty) = ctx.vec_of(u32::from(*dim));
            if let [single] = &args[..] {
                let v = lower_expr(ctx, single)?;
                if lanes_of(ctx, v.ty) == 0 {
                    let id = splat(ctx, v, gty);
                    return Ok(GVal { id, ty: gty });
                }
            }
            let mut comps = Vec::with_capacity(args.len());
            let mut filled = 0;
            for a in args {
                let v = lower_expr(ctx, a)?;
                let width = lanes_of(ctx, v.ty);
                if width == 0 {
                    filled += 1;
                    comps.push(scalar_to_f32(ctx, v));
                } else {
                    filled += width;
                    comps.push(v.id);
                }
            }
            if filled != u32::from(*dim) {
                return Err(format!("f32x{dim} needs {dim} components, got {filled}"));
            }
            let id = ctx.spv.op_composite_construct(spv_ty, &comps);
            Ok(GVal { id, ty: gty })
        }
        IrExpr::Builtin { func, args } => lower_builtin(ctx, *func, args),
        IrExpr::ArrayLit(_) => Err("an array is only bound by `let`".into()),
        IrExpr::Index { array, index } => {
            let (element, elem_ty, elem_ptr) = ctx
                .arrays
                .get(array)
                .map(|a| (a.element, a.elem_ty, a.elem_ptr))
                .ok_or_else(|| format!("`{array}` is not an array"))?;
            let base = ctx
                .locals
                .get(array)
                .map(|l| l.ptr)
                .ok_or("internal: no array")?;
            let i = lower_expr(ctx, index)?;
            let i = scalar_to_u32(ctx, i);
            let ptr = ctx.spv.op_access_chain(elem_ptr, base, &[i]);
            let id = ctx.spv.op_load(elem_ty, ptr);
            Ok(GVal { id, ty: element })
        }
        IrExpr::Sample { index, uv, lod } => {
            let table = ctx.textures.ok_or("internal: no texture table")?;
            let slot = lower_expr(ctx, index)?;
            let slot = scalar_to_u32(ctx, slot);
            let uv = lower_expr(ctx, uv)?;
            if uv.ty != ctx.vec2_id {
                return Err("texture coordinates must be f32x2".into());
            }
            let ptr = ctx.spv.op_access_chain(table.t_ptr, table.var, &[slot]);
            let image = ctx.spv.op_load(table.t_sampled, ptr);
            let t_vec4 = ctx.t_vec4;
            let lod = match lod {
                Some(lod) => {
                    let lod = lower_expr(ctx, lod)?;
                    Some(scalar_to_f32(ctx, lod))
                }
                None if ctx.is_fragment => None,
                None => Some(ctx.spv.constant_f32(ctx.t_f32, 0.0)),
            };
            let id = match lod {
                Some(lod) => ctx
                    .spv
                    .op_image_sample_explicit_lod(t_vec4, image, uv.id, lod),
                None => ctx.spv.op_image_sample_implicit_lod(t_vec4, image, uv.id),
            };
            Ok(GVal {
                id,
                ty: ctx.vec4_id,
            })
        }
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
        IrExpr::Texel { .. } | IrExpr::TexSize { .. } => {
            Err("texture and image params are only available in compute shaders; sample the `textures` table".into())
        }
    }
}

fn lower_builtin(ctx: &mut Ctx<'_>, func: BuiltinFn, args: &[IrExpr]) -> R<GVal> {
    let name = func.name();
    match func {
        BuiltinFn::FragZeroAcc
        | BuiltinFn::Mma
        | BuiltinFn::FragLen
        | BuiltinFn::FragRow
        | BuiltinFn::FragCol
        | BuiltinFn::FragGet
        | BuiltinFn::FragSet => {
            Err("fragment builtins are only available in compute shaders".into())
        }
        BuiltinFn::IsNan
        | BuiltinFn::IsInf
        | BuiltinFn::IsFinite
        | BuiltinFn::MulAddF32
        | BuiltinFn::Dot4I8
        | BuiltinFn::Dot4U8
        | BuiltinFn::Dot4IU8
        | BuiltinFn::PkAddU16
        | BuiltinFn::PkMulU16
        | BuiltinFn::PkMadU16
        | BuiltinFn::PkAddF16
        | BuiltinFn::PkMulF16
        | BuiltinFn::PkFmaF16
        | BuiltinFn::Dot2F16
        | BuiltinFn::FunnelShr
        | BuiltinFn::BytePerm
        | BuiltinFn::F16Lo
        | BuiltinFn::F16Hi
        | BuiltinFn::PackF16x2
        | BuiltinFn::BitcastF32
        | BuiltinFn::BitcastU32
        | BuiltinFn::BitcastI32
        | BuiltinFn::Byte
        | BuiltinFn::SByte
        | BuiltinFn::Select
        | BuiltinFn::I32
        | BuiltinFn::F16Round
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
        BuiltinFn::U32 | BuiltinFn::F32 => {
            if args.len() != 1 {
                return Err(format!("{name}() takes 1 arg"));
            }
            let v = lower_expr(ctx, &args[0])?;
            if matches!(func, BuiltinFn::U32) {
                let id = scalar_to_u32(ctx, v);
                Ok(GVal { id, ty: ctx.u32_id })
            } else {
                let id = scalar_to_f32(ctx, v);
                Ok(GVal { id, ty: ctx.f32_id })
            }
        }
        BuiltinFn::Step | BuiltinFn::Smoothstep => {
            let want = if matches!(func, BuiltinFn::Step) {
                2
            } else {
                3
            };
            if args.len() != want {
                return Err(format!("{name}() takes {want} args"));
            }
            let x = lower_expr(ctx, &args[want - 1])?;
            if !(x.ty == ctx.f32_id || is_vec(ctx, x.ty)) {
                return Err(format!("{name}() requires f32/f32xN values"));
            }
            let mut ops = Vec::with_capacity(want);
            for a in &args[..want - 1] {
                let v = lower_expr(ctx, a)?;
                ops.push(broadcast(ctx, v, x.ty)?);
            }
            ops.push(x.id);
            let opcode = if want == 2 {
                glsl_op::STEP
            } else {
                glsl_op::SMOOTH_STEP
            };
            let ty = ctx.spv_elem_ty(x.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(ty, glsl, opcode, &ops);
            Ok(GVal { id, ty: x.ty })
        }
        BuiltinFn::Abs
        | BuiltinFn::Sign
        | BuiltinFn::Exp
        | BuiltinFn::FastExp
        | BuiltinFn::RoundEven
        | BuiltinFn::Tanh
        | BuiltinFn::Sin
        | BuiltinFn::Cos
        | BuiltinFn::Tan
        | BuiltinFn::Log
        | BuiltinFn::Sqrt
        | BuiltinFn::Rsqrt
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
                BuiltinFn::Exp | BuiltinFn::FastExp => glsl_op::EXP,
                BuiltinFn::RoundEven => glsl_op::ROUND_EVEN,
                BuiltinFn::Tanh => glsl_op::TANH,
                BuiltinFn::Sin => glsl_op::SIN,
                BuiltinFn::Cos => glsl_op::COS,
                BuiltinFn::Tan => glsl_op::TAN,
                BuiltinFn::Log => glsl_op::LOG,
                BuiltinFn::Sqrt => glsl_op::SQRT,
                BuiltinFn::Rsqrt => glsl_op::INVERSE_SQRT,
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
            if a.ty == ctx.u32_id && b.ty == ctx.u32_id && func != BuiltinFn::Pow {
                let op = if func == BuiltinFn::Min {
                    glsl_op::U_MIN
                } else {
                    glsl_op::U_MAX
                };
                let (t, glsl) = (ctx.t_u32, ctx.glsl_ext);
                let id = ctx.spv.op_ext_inst(t, glsl, op, &[a.id, b.id]);
                return Ok(GVal { id, ty: a.ty });
            }
            let (a, b) = widen_pair(ctx, a, b)?;
            if !(a.ty == ctx.f32_id || is_vec(ctx, a.ty)) {
                return Err(format!("{name}() requires u32 or f32/f32xN args"));
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
            if x.ty == ctx.u32_id {
                let lo = lower_expr(ctx, &args[1])?;
                let hi = lower_expr(ctx, &args[2])?;
                if lo.ty != ctx.u32_id || hi.ty != ctx.u32_id {
                    return Err("clamp() of a u32 needs u32 bounds".into());
                }
                let (t, glsl) = (ctx.t_u32, ctx.glsl_ext);
                let id = ctx
                    .spv
                    .op_ext_inst(t, glsl, glsl_op::U_CLAMP, &[x.id, lo.id, hi.id]);
                return Ok(GVal { id, ty: x.ty });
            }
            if !(x.ty == ctx.f32_id || is_vec(ctx, x.ty)) {
                return Err("clamp() requires u32 or f32/f32xN args".into());
            }
            let lo = lower_expr(ctx, &args[1])?;
            let lo = broadcast(ctx, lo, x.ty)?;
            let hi = lower_expr(ctx, &args[2])?;
            let hi = broadcast(ctx, hi, x.ty)?;
            let ty = ctx.spv_elem_ty(x.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(ty, glsl, glsl_op::F_CLAMP, &[x.id, lo, hi]);
            Ok(GVal { id, ty: x.ty })
        }
        BuiltinFn::Mix => {
            if args.len() != 3 {
                return Err("mix(a, b, t) takes 3 args".into());
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            let (a, b) = widen_pair(ctx, a, b)?;
            if !(a.ty == ctx.f32_id || is_vec(ctx, a.ty)) {
                return Err("mix() requires f32/f32xN args".into());
            }
            let t = lower_expr(ctx, &args[2])?;
            let t = broadcast(ctx, t, a.ty)?;
            let ty = ctx.spv_elem_ty(a.ty);
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(ty, glsl, glsl_op::F_MIX, &[a.id, b.id, t]);
            Ok(GVal { id, ty: a.ty })
        }
    }
}

fn lower_arith(ctx: &mut Ctx<'_>, op: IrBinOp, lhs: GVal, rhs: GVal) -> R<GVal> {
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
    if is_vec(ctx, lhs.ty) != is_vec(ctx, rhs.ty) && !matches!(op, IrBinOp::Mul) {
        let (lhs, rhs) = widen_pair(ctx, lhs, rhs)?;
        return lower_arith(ctx, op, lhs, rhs);
    }
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
    if !(lhs.ty == ctx.f32_id || lhs.ty == ctx.u32_id) {
        return Err("binary arithmetic only on scalar or vector types".into());
    }
    let bitwise = matches!(
        op,
        IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Shl | IrBinOp::Shr
    );
    if bitwise && (lhs.ty != ctx.u32_id || rhs.ty != ctx.u32_id) {
        return Err("& | ^ << >> need u32 operands".into());
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
            IrBinOp::BitAnd => ctx.spv.op_bitwise_and(ty, lhs.id, rhs.id),
            IrBinOp::BitOr => ctx.spv.op_bitwise_or(ty, lhs.id, rhs.id),
            IrBinOp::BitXor => ctx.spv.op_bitwise_xor(ty, lhs.id, rhs.id),
            IrBinOp::Shl => ctx.spv.op_shift_left_logical(ty, lhs.id, rhs.id),
            IrBinOp::Shr => ctx.spv.op_shift_right_logical(ty, lhs.id, rhs.id),
            _ => return Err("unsupported op; use + - * / & | ^ << >>".into()),
        }
    };
    Ok(GVal { id, ty: lhs.ty })
}

fn is_vec(ctx: &Ctx<'_>, t: TypeId) -> bool {
    t == ctx.vec2_id || t == ctx.vec3_id || t == ctx.vec4_id
}

fn lanes_of(ctx: &Ctx<'_>, t: TypeId) -> u32 {
    if t == ctx.vec2_id {
        2
    } else if t == ctx.vec3_id {
        3
    } else if t == ctx.vec4_id {
        4
    } else {
        0
    }
}

impl Ctx<'_> {
    fn vec_of(&self, lanes: u32) -> (TypeId, Id) {
        match lanes {
            2 => (self.vec2_id, self.t_vec2),
            3 => (self.vec3_id, self.t_vec3),
            _ => (self.vec4_id, self.t_vec4),
        }
    }
}

fn splat(ctx: &mut Ctx<'_>, v: GVal, to: TypeId) -> Id {
    let x = scalar_to_f32(ctx, v);
    let lanes = vec![x; lanes_of(ctx, to) as usize];
    let ty = ctx.spv_elem_ty(to);
    ctx.spv.op_composite_construct(ty, &lanes)
}

fn broadcast(ctx: &mut Ctx<'_>, v: GVal, to: TypeId) -> R<Id> {
    if v.ty == to {
        Ok(v.id)
    } else if is_vec(ctx, to) && lanes_of(ctx, v.ty) == 0 {
        Ok(splat(ctx, v, to))
    } else {
        coerce(ctx, v, to)
    }
}

fn widen_pair(ctx: &mut Ctx<'_>, a: GVal, b: GVal) -> R<(GVal, GVal)> {
    if a.ty == b.ty {
        return Ok((a, b));
    }
    if is_vec(ctx, a.ty) && !is_vec(ctx, b.ty) {
        let id = broadcast(ctx, b, a.ty)?;
        return Ok((a, GVal { id, ty: a.ty }));
    }
    if is_vec(ctx, b.ty) && !is_vec(ctx, a.ty) {
        let id = broadcast(ctx, a, b.ty)?;
        return Ok((GVal { id, ty: b.ty }, b));
    }
    if is_vec(ctx, a.ty) {
        return Err("vector widths differ".into());
    }
    Ok(unify_scalars(ctx, a, b))
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
        Err(format!("cannot coerce {:?} to {:?}", v.ty, target))
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

#[cfg(test)]
pub(crate) mod tests {
    use std::env::temp_dir;
    use std::fs::{remove_file, write};
    use std::process::{Command, id};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::lower_graphics;
    use crate::frontend::parser::{Shader, parse_zsl};

    fn lower(src: &str) -> Vec<u32> {
        match parse_zsl(src).expect("parse") {
            Shader::Graphics(m) => lower_graphics(&m).expect("lower"),
            Shader::Compute(_) => panic!("expected graphics"),
        }
    }

    /// Runs `spirv-val` over `words` when the tool is installed.
    ///
    /// A failing validation prints the reason.
    pub(crate) fn validate(words: &[u32]) {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = temp_dir().join(format!(
            "zsl_gfx_{}_{}.spv",
            id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        write(&path, bytes).expect("write spv");
        let out = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output();
        let _ = remove_file(&path);
        if let Ok(out) = out {
            assert!(
                out.status.success(),
                "spirv-val:\n{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    #[test]
    fn swizzles_spread_scalars_and_vector_push_fields() {
        validate(&lower(
            r#"
            push P { viewport: f32x2, tint: f32x4, slot: u32 }
            vertex vs(@location(0) rect: f32x4, p: P) -> (f32x4, f32x2) {
                let half = rect.zw * 0.5
                let reach = half + f32x2(1.0) + p.tint.zw * p.tint.x
                let px = rect.xy + half + reach
                let ndc = px / p.viewport * 2.0 - 1.0
                (f32x4(ndc, 0.0, 1.0), max(abs(ndc) - half + 1.0, 0.0))
            }
            "#,
        ));
    }

    #[test]
    fn colour_swizzles_casts_bits_and_else_if() {
        validate(&lower(
            r#"
            fragment fs(@location(0) color: f32x4, @location(1) shape: f32x2) -> f32x4 {
                let bits = u32(shape.x) & 3
                let a = f32(bits >> 1) + smoothstep(0.0, 1.0, shape.y)
                let d = 0.0
                if shape.y > 1.0 {
                    d = 1.0
                } else if shape.y > 0.5 {
                    d = step(0.5, shape.x)
                } else {
                    d = clamp(a, 0.0, 1.0)
                }
                f32x4(mix(color.rgb, f32x3(1.0), d), color.a * a)
            }
            "#,
        ));
    }

    #[test]
    fn quad_corners_from_the_vertex_index() {
        validate(&lower(
            r#"
            push P { viewport: f32x2 }
            vertex vs(@location(0) rect: f32x4, vi: vertex_index, ii: instance_index, p: P) -> (f32x4, u32) {
                // Two triangles: corners 0,1,2 and 1,3,2 of the unit square.
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                let px = rect.xy + corner * rect.zw
                (f32x4(px / p.viewport * 2.0 - 1.0, 0.0, 1.0), ii)
            }
            "#,
        ));
    }

    #[test]
    fn fragments_read_their_position_and_flat_integers() {
        validate(&lower(
            r#"
            fragment fs(@location(0) layer: u32, at: frag_coord) -> f32x4 {
                f32x4(at.xy * 0.01, f32(layer), 1.0)
            }
            "#,
        ));
    }

    #[test]
    fn rounded_box_discards_outside_its_edge() {
        validate(&lower(
            r#"
            fragment fs(
                @location(0) local: f32x2,
                @location(1) color: f32x4,
                @location(2) half: f32x2,
                @location(3) shape: f32x2,
            ) -> f32x4 {
                let r = clamp(shape.x, 0.0, min(half.x, half.y))
                let q = abs(local) - half + r
                let d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r
                if shape.y > 0.0 {
                    d = abs(d) - shape.y * 0.5
                }
                let alpha = clamp(0.5 - d, 0.0, 1.0)
                if alpha <= 0.0 {
                    discard
                }
                f32x4(color.rgb, color.a * alpha)
            }
            "#,
        ));
    }

    #[test]
    fn vertex_samples_name_their_level() {
        validate(&lower(
            r#"
            push P { slot: u32 }
            vertex vs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
                let a = tex[p.slot].sample(uv)
                let b = tex[p.slot + 1].sample_level(uv, 2.0)
                a + b
            }
            "#,
        ));
    }

    #[test]
    fn dual_source_text_writes_colour_and_coverage() {
        validate(&lower(
            r#"
            push P { viewport: f32x2, slot: u32 }
            fragment fs(@location(0) uv: f32x2, @location(1) color: f32x4, tex: textures, p: P)
                -> (@location(0) @blend_src(0) f32x4, @location(0) @blend_src(1) f32x4)
            {
                let coverage = tex[p.slot].sample(uv).rgb * color.a
                let alpha = max(coverage.r, max(coverage.g, coverage.b))
                if alpha <= 0.0 {
                    discard
                }
                (f32x4(color.rgb, 1.0), f32x4(coverage, alpha))
            }
            "#,
        ));
    }

    #[test]
    fn fragments_fill_several_attachments() {
        validate(&lower(
            r#"
            fragment fs(@location(0) c: f32x4) -> (f32x4, f32, @location(3) f32x2) {
                (c, c.a, c.xy)
            }
            "#,
        ));
    }

    #[test]
    fn dual_source_takes_one_pair_at_location_zero() {
        assert!(
            parse_zsl(
                "fragment fs(@location(0) c: f32x4) -> (f32x4, @blend_src(1) f32x4) { (c, c) }"
            )
            .is_err()
        );
        assert!(parse_zsl("fragment fs(@location(0) c: f32x4) -> (@location(1) f32x4, @location(1) f32x4) { (c, c) }").is_err());
    }

    #[test]
    fn constant_arrays_pick_by_vertex_index() {
        validate(&lower(
            r#"
            push P { angle: f32 }
            vertex vs(vi: vertex_index, p: P) -> (f32x4, f32x3) {
                let pos = [f32x2(-0.6, 0.6), f32x2(0.6, 0.6), f32x2(0.0, -0.6)]
                let col = [f32x3(1.0, 0.0, 0.0), f32x3(0.0, 1.0, 0.0), f32x3(0.0, 0.0, 1.0)]
                let c = cos(p.angle)
                let s = sin(p.angle)
                let q = pos[vi]
                (f32x4(c * q.x - s * q.y, s * q.x + c * q.y, 0.0, 1.0), col[vi])
            }
            "#,
        ));
    }

    #[test]
    fn stage_and_mutability_rules_are_enforced() {
        assert!(parse_zsl("vertex vs(vi: vertex_index) -> f32x4 { let a = [1.0, 2.0]\n a = 1.0\n f32x4(a[vi]) }").is_err());
        assert!(parse_zsl("vertex vs(@location(0) p: f32x4) -> f32x4 { discard\n p }").is_err());
        assert!(parse_zsl("fragment fs(vi: vertex_index) -> f32x4 { f32x4(1.0) }").is_err());
        assert!(parse_zsl("vertex vs(at: frag_coord) -> f32x4 { at }").is_err());
    }

    #[test]
    fn swizzle_past_the_vector_is_an_error() {
        let Shader::Graphics(m) =
            parse_zsl("fragment fs(@location(0) uv: f32x2) -> f32x4 { f32x4(uv.xyz, 1.0) }")
                .expect("parse")
        else {
            panic!("expected graphics");
        };
        assert!(lower_graphics(&m).is_err());
    }
}
