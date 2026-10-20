//! Lowers IR to SPIR-V for compute shaders. Names and structure are already checked; this pass
//! infers value types and emits SPIR-V.
//!
//! All `OpVariable`s are hoisted to the entry block (SPIR-V requires it), then
//! statements are lowered.

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
    pub const FMA: u32 = 50;
}

use std::collections::HashMap;

use crate::backend::spirv::builder::{Id, SpvBuilder, builtin, deco, op, sc};
use crate::frontend::diagnostic::Diagnostic;
use crate::ir::arena::TypeArena;
use crate::ir::binding::{IMAGE_BINDING, TEXTURE_BINDING, ssbo_binding_for};
use crate::ir::node::{
    BuiltinFn, IrBinOp, IrExpr, IrStmt, TexelOp, stmts_use_subgroup_collectives,
    stmts_use_subgroup_query, stmts_use_workgroup_reduce,
};
use crate::ir::typeck::frag_parts;
use crate::ir::types::TypeId;
use crate::ir::validate::check_uniformity;
use crate::ir::{EntryKind, Module, ParamKind};

mod coop;

use coop::Coop;

pub fn lower_compute(module: &Module) -> Result<Vec<u32>, Diagnostic> {
    check_uniformity(module)?;
    lower_compute_inner(module).map_err(Diagnostic::from)
}

fn lower_compute_inner(module: &Module) -> Result<Vec<u32>, String> {
    let entry = &module.entry;
    let EntryKind::Compute { local_size } = entry.kind;

    let mut spv = SpvBuilder::new();

    spv.capability_shader();
    spv.capability_runtime_descriptor_array();
    if has_atomic_add(&entry.body) {
        spv.enable_atomic_float32_add_ext();
    }
    let uses_subgroup_query = stmts_use_subgroup_query(&entry.body);
    let (
        uses_subgroup_vote,
        uses_subgroup_ballot,
        uses_subgroup_shuffle,
        uses_subgroup_shuffle_relative,
        uses_subgroup_arithmetic,
    ) = stmts_use_subgroup_collectives(&entry.body);
    if uses_subgroup_query
        || uses_subgroup_vote
        || uses_subgroup_ballot
        || uses_subgroup_shuffle
        || uses_subgroup_shuffle_relative
        || uses_subgroup_arithmetic
    {
        spv.capability_group_non_uniform();
    }
    if uses_subgroup_vote {
        spv.capability_group_non_uniform_vote();
    }
    if uses_subgroup_ballot {
        spv.capability_group_non_uniform_ballot();
    }
    if uses_subgroup_shuffle {
        spv.capability_group_non_uniform_shuffle();
    }
    if uses_subgroup_shuffle_relative {
        spv.capability_group_non_uniform_shuffle_relative();
    }
    if uses_subgroup_arithmetic {
        spv.capability_group_non_uniform_arithmetic();
    }
    let glsl_ext = spv.ext_inst_import_glsl();
    spv.memory_model_logical_glsl450();

    let mut types = module.types.clone();
    let u32_id = types.u32_id();
    let i32_id = types.i32_id();
    let f32_id = types.f32_id();
    let bool_id = types.bool_id();
    let bitset128_id = types.bitset128_id();

    let t_void = spv.type_void();
    let t_bool = spv.type_bool();
    let t_u32 = spv.type_int(32, false);
    let t_i32 = spv.type_int(32, true);
    let t_f32 = spv.type_float(32);
    let t_uvec3 = spv.type_vector(t_u32, 3);
    let t_uvec4 = spv.type_vector(t_u32, 4);

    let buf_params: Vec<(&str, TypeId)> = entry
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some((p.name.as_str(), *elem)),
            _ => None,
        })
        .collect();
    let scalar_params: Vec<(&str, TypeId)> = entry
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Scalar(ty) => Some((p.name.as_str(), *ty)),
            _ => None,
        })
        .collect();
    let tex_params: Vec<(&str, bool)> = entry
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Texture))
        .chain(
            entry
                .params
                .iter()
                .filter(|p| matches!(p.kind, ParamKind::Image)),
        )
        .map(|p| (p.name.as_str(), matches!(p.kind, ParamKind::Image)))
        .collect();

    let f16_id = types.f16_id();
    let bf16_id = types.bf16_id();
    let half_of = |elem: TypeId| {
        if elem == f16_id {
            Some(Half::F16)
        } else if elem == bf16_id {
            Some(Half::Bf16)
        } else {
            None
        }
    };
    let table_of = |elem: TypeId| {
        if half_of(elem).is_some() {
            u32_id
        } else {
            elem
        }
    };
    let mut ssbo_tables: HashMap<TypeId, SsboTable> = HashMap::new();
    for (_, elem) in &buf_params {
        let elem_ty = table_of(*elem);
        if ssbo_tables.contains_key(&elem_ty) {
            continue;
        }
        let t_elem = if elem_ty == f32_id {
            t_f32
        } else if elem_ty == u32_id {
            t_u32
        } else if elem_ty == i32_id {
            t_i32
        } else {
            return Err(format!(
                "ZSL: SPIR-V backend does not support buffer<{:?}>",
                types.get(elem_ty)
            ));
        };
        let binding = ssbo_binding_for(elem_ty, &mut types);

        let t_ra = spv.type_runtime_array(t_elem);
        spv.decorate(t_ra, deco::ARRAY_STRIDE, &[4]);
        let t_struct_buf = spv.type_struct(&[t_ra]);
        spv.decorate(t_struct_buf, deco::BLOCK, &[]);
        spv.member_decorate(t_struct_buf, 0, deco::OFFSET, &[0]);
        let t_ra_struct = spv.type_runtime_array(t_struct_buf);
        let t_ptr_ra_struct = spv.type_pointer(sc::STORAGE_BUFFER, t_ra_struct);
        let t_ptr_elem = spv.type_pointer(sc::STORAGE_BUFFER, t_elem);

        let var = spv.global_variable(t_ptr_ra_struct, sc::STORAGE_BUFFER);
        spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
        spv.decorate(var, deco::BINDING, &[binding]);

        ssbo_tables.insert(elem_ty, SsboTable { var, t_ptr_elem });
    }

    let t_vec2 = spv.type_vector(t_f32, 2);
    let images = if tex_params.is_empty() {
        None
    } else {
        spv.capability_image_query();
        if tex_params.iter().any(|(_, storage)| *storage) {
            spv.capability_storage_image_without_format();
        }
        let t_vec4 = spv.type_vector(t_f32, 4);
        let t_ivec2 = spv.type_vector(t_i32, 2);
        let t_image = spv.type_image_2d(t_f32);
        let t_sampled = spv.type_sampled_image(t_image);
        let t_storage = spv.type_storage_image_2d(t_f32);
        let mut table = |elem: Id, binding: u32| {
            let t_table = spv.type_runtime_array(elem);
            let t_ptr_table = spv.type_pointer(sc::UNIFORM_CONSTANT, t_table);
            let var = spv.global_variable(t_ptr_table, sc::UNIFORM_CONSTANT);
            spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
            spv.decorate(var, deco::BINDING, &[binding]);
            (var, spv.type_pointer(sc::UNIFORM_CONSTANT, elem))
        };
        let (sampled_var, t_ptr_sampled) = table(t_sampled, TEXTURE_BINDING);
        let (storage_var, t_ptr_storage) = table(t_storage, IMAGE_BINDING);
        Some(ImageTypes {
            t_vec2,
            t_vec4,
            t_ivec2,
            t_image,
            t_sampled,
            t_storage,
            sampled_var,
            t_ptr_sampled,
            storage_var,
            t_ptr_storage,
        })
    };

    let n_buf_params = buf_params.len() as u32;
    let n_slot_params = n_buf_params + tex_params.len() as u32;
    let pc_var = if n_slot_params != 0 || !scalar_params.is_empty() {
        let mut pc_members: Vec<Id> = (0..n_slot_params).map(|_| t_u32).collect();
        pc_members.extend(scalar_params.iter().map(|(_, ty)| {
            if *ty == u32_id {
                t_u32
            } else if *ty == i32_id {
                t_i32
            } else if *ty == f32_id {
                t_f32
            } else {
                t_u32
            }
        }));
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);
        for i in 0..pc_members.len() {
            spv.member_decorate(t_pc_struct, i as u32, deco::OFFSET, &[(i * 4) as u32]);
        }
        let t_ptr_pc = spv.type_pointer(sc::PUSH_CONSTANT, t_pc_struct);
        Some(spv.global_variable(t_ptr_pc, sc::PUSH_CONSTANT))
    } else {
        None
    };

    let t_ptr_in_uvec3 = spv.type_pointer(sc::INPUT, t_uvec3);
    let gid_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(gid_var, deco::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);
    let lid_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(lid_var, deco::BUILT_IN, &[builtin::LOCAL_INVOCATION_ID]);
    let group_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(group_var, deco::BUILT_IN, &[builtin::WORKGROUP_ID]);

    let subgroup_vars = if uses_subgroup_query {
        let t_ptr_in_u32 = spv.type_pointer(sc::INPUT, t_u32);
        let size_var = spv.global_variable(t_ptr_in_u32, sc::INPUT);
        spv.decorate(size_var, deco::BUILT_IN, &[builtin::SUBGROUP_SIZE]);
        let id_var = spv.global_variable(t_ptr_in_u32, sc::INPUT);
        spv.decorate(id_var, deco::BUILT_IN, &[builtin::SUBGROUP_ID]);
        let local_id_var = spv.global_variable(t_ptr_in_u32, sc::INPUT);
        spv.decorate(
            local_id_var,
            deco::BUILT_IN,
            &[builtin::SUBGROUP_LOCAL_INVOCATION_ID],
        );
        Some(SubgroupVars {
            size: size_var,
            id: id_var,
            local_id: local_id_var,
        })
    } else {
        None
    };

    let t_ptr_wg_f32 = spv.type_pointer(sc::WORKGROUP, t_f32);
    let mut shared = HashMap::new();
    for decl in &entry.shared {
        let len = spv.constant_u32(t_u32, decl.len);
        let array_ty = spv.type_array(t_f32, len);
        let ptr_ty = spv.type_pointer_global(sc::WORKGROUP, array_ty);
        let var = spv.global_variable(ptr_ty, sc::WORKGROUP);
        shared.insert(decl.name.clone(), var);
    }
    let uses_workgroup_reduce = stmts_use_workgroup_reduce(&entry.body);
    let workgroup_reduce_len = local_size[0] * local_size[1] * local_size[2];
    let workgroup_reduce_var = if uses_workgroup_reduce {
        if local_size[1] != 1 || local_size[2] != 1 {
            return Err(format!(
                "ZSL: workgroup_reduce_* requires a 1D workgroup, got local_size {local_size:?}"
            ));
        }
        if !workgroup_reduce_len.is_power_of_two() {
            return Err(format!(
                "ZSL: workgroup_reduce_* requires a power-of-two workgroup thread count, got {workgroup_reduce_len}"
            ));
        }
        let len_const = spv.constant_u32(t_u32, workgroup_reduce_len);
        let array_ty = spv.type_array(t_f32, len_const);
        let ptr_ty = spv.type_pointer_global(sc::WORKGROUP, array_ty);
        Some(spv.global_variable(ptr_ty, sc::WORKGROUP))
    } else {
        None
    };

    let t_fn = spv.type_function(t_void, &[]);
    let fn_id = spv.fresh_id();
    let mut interface_vars = vec![gid_var, lid_var, group_var];
    if let Some(sg) = &subgroup_vars {
        interface_vars.extend([sg.size, sg.id, sg.local_id]);
    }
    spv.entry_point_glcompute(fn_id, "main", &interface_vars);
    spv.execution_mode_local_size(fn_id, local_size[0], local_size[1], local_size[2]);

    let const_0_u32 = spv.constant_u32(t_u32, 0);

    let buf_param_map: HashMap<String, BufInfo> = buf_params
        .iter()
        .enumerate()
        .map(|(i, (name, elem))| {
            (
                name.to_string(),
                BufInfo {
                    pc_index: i as u32,
                    elem: *elem,
                    table: table_of(*elem),
                    half: half_of(*elem),
                },
            )
        })
        .collect();

    let tex_param_map: HashMap<String, TexInfo> = tex_params
        .iter()
        .enumerate()
        .map(|(i, (name, storage))| {
            (
                name.to_string(),
                TexInfo {
                    pc_index: n_buf_params + i as u32,
                    storage: *storage,
                },
            )
        })
        .collect();

    let scalar_param_map: HashMap<String, ScalarInfo> = scalar_params
        .iter()
        .enumerate()
        .map(|(i, (name, ty))| {
            (
                name.to_string(),
                ScalarInfo {
                    pc_index: i as u32,
                    ty: *ty,
                },
            )
        })
        .collect();

    spv.begin_function(t_void, fn_id, t_fn);
    spv.label();

    let mut ptr_func_u32 = Id(0);
    let mut ptr_func_i32 = Id(0);
    let mut ptr_func_f32 = Id(0);
    let mut ptr_func_bool = Id(0);
    let mut ptr_func_uvec4 = Id(0);

    let mut locals: HashMap<String, LocalVar> = HashMap::new();
    let mut coop = Coop::default();
    for (name, ty_id) in &entry.locals {
        let sty = *ty_id;
        if frag_parts(&types, sty).is_some() {
            let coop_ty = coop.type_of(&mut spv, &types, sty, t_u32, t_f32, t_i32)?;
            let ptr_ty = spv.type_pointer(sc::FUNCTION, coop_ty);
            let ptr = spv.op_variable(ptr_ty, sc::FUNCTION);
            locals.insert(
                name.clone(),
                LocalVar {
                    ptr,
                    ty: sty,
                    ptr_ty,
                },
            );
            continue;
        }
        let (ptr_ty_slot, spv_elem) = if sty == u32_id {
            (&mut ptr_func_u32, t_u32)
        } else if sty == i32_id {
            (&mut ptr_func_i32, t_i32)
        } else if sty == f32_id {
            (&mut ptr_func_f32, t_f32)
        } else if sty == bitset128_id {
            (&mut ptr_func_uvec4, t_uvec4)
        } else {
            (&mut ptr_func_bool, t_bool)
        };
        if *ptr_ty_slot == Id(0) {
            *ptr_ty_slot = spv.type_pointer(sc::FUNCTION, spv_elem);
        }
        let ptr = spv.op_variable(*ptr_ty_slot, sc::FUNCTION);
        locals.insert(
            name.clone(),
            LocalVar {
                ptr,
                ty: sty,
                ptr_ty: *ptr_ty_slot,
            },
        );
    }

    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_i32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    if pc_var.is_some() {
        if n_slot_params > 0 {
            t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
        }
        for (_, ty) in &scalar_params {
            if *ty == u32_id && t_ptr_pc_u32 == Id(0) {
                t_ptr_pc_u32 = spv.type_pointer(sc::PUSH_CONSTANT, t_u32);
            } else if *ty == i32_id && t_ptr_pc_i32 == Id(0) {
                t_ptr_pc_i32 = spv.type_pointer(sc::PUSH_CONSTANT, t_i32);
            } else if *ty == f32_id && t_ptr_pc_f32 == Id(0) {
                t_ptr_pc_f32 = spv.type_pointer(sc::PUSH_CONSTANT, t_f32);
            }
        }
    }

    let mut ctx = LowerCtx {
        spv: &mut spv,
        u32_id,
        i32_id,
        f32_id,
        bool_id,
        bitset128_id,
        t_bool,
        t_u32,
        t_i32,
        t_f32,
        t_uvec3,
        t_uvec4,
        t_vec2,
        ssbo_tables,
        n_slot_params,
        glsl_ext,
        buf_params: buf_param_map,
        tex_params: tex_param_map,
        images,
        scalar_params: scalar_param_map,
        pc_var,
        gid_var,
        lid_var,
        group_var,
        subgroup_vars,
        shared,
        workgroup_reduce_var,
        workgroup_reduce_len,
        t_ptr_wg_f32,
        local_size,
        const_0_u32,
        locals,
        coop,
        types: types.clone(),
        t_ptr_pc_u32,
        t_ptr_pc_i32,
        t_ptr_pc_f32,
    };

    lower_stmts(&mut ctx, &entry.body)?;

    ctx.spv.op_return();
    ctx.spv.end_function();

    Ok(spv.finish())
}

#[derive(Clone, Copy)]
struct BufInfo {
    pc_index: u32,
    elem: TypeId,
    /// SSBO table the buffer is read through (`u32` for halves).
    table: TypeId,
    half: Option<Half>,
}

/// A 16-bit float buffer element, stored packed two per `u32` word.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Half {
    F16,
    Bf16,
}

/// Bindless SSBO array for one element type. The pointee is the element's real SPIR-V type:
/// no bitcast.
#[derive(Clone, Copy)]
struct SsboTable {
    var: Id,
    t_ptr_elem: Id,
}

/// A `texture` or `image` param: where its slot sits in the push constants, and which table
/// it indexes.
#[derive(Clone, Copy)]
struct TexInfo {
    pc_index: u32,
    storage: bool,
}

/// The types and tables texel access needs, declared once when a kernel has any `texture` or
/// `image` param.
#[derive(Clone, Copy)]
struct ImageTypes {
    t_vec2: Id,
    t_vec4: Id,
    t_ivec2: Id,
    t_image: Id,
    t_sampled: Id,
    t_storage: Id,
    sampled_var: Id,
    t_ptr_sampled: Id,
    storage_var: Id,
    t_ptr_storage: Id,
}

#[derive(Clone, Copy)]
struct ScalarInfo {
    pc_index: u32,
    ty: TypeId,
}

/// Subgroup-query inputs, declared only when a kernel calls one.
#[derive(Clone, Copy)]
struct SubgroupVars {
    size: Id,
    id: Id,
    local_id: Id,
}

#[derive(Clone, Copy)]
struct Val {
    id: Id,
    ty: TypeId,
}

struct LocalVar {
    ptr: Id,
    ty: TypeId,
    #[allow(dead_code)]
    ptr_ty: Id,
}

struct LowerCtx<'a> {
    spv: &'a mut SpvBuilder,
    /// Canonical `TypeId`s, cached in `lower_compute_inner` so comparisons are plain id equality.
    u32_id: TypeId,
    i32_id: TypeId,
    f32_id: TypeId,
    bool_id: TypeId,
    bitset128_id: TypeId,
    t_bool: Id,
    t_u32: Id,
    t_i32: Id,
    t_f32: Id,
    t_uvec3: Id,
    t_uvec4: Id,
    /// `vec2<f32>`, declared only when an f16 buffer or a texture needs it.
    t_vec2: Id,
    /// One SSBO table per buffer element type, keyed by its canonical `TypeId`.
    ssbo_tables: HashMap<TypeId, SsboTable>,
    glsl_ext: Id,
    buf_params: HashMap<String, BufInfo>,
    tex_params: HashMap<String, TexInfo>,
    images: Option<ImageTypes>,
    scalar_params: HashMap<String, ScalarInfo>,
    /// Number of buffer, texture and image params: scalar PC indices are offset by this.
    n_slot_params: u32,
    pc_var: Option<Id>,
    gid_var: Id,
    lid_var: Id,
    group_var: Id,
    subgroup_vars: Option<SubgroupVars>,
    shared: HashMap<String, Id>,
    workgroup_reduce_var: Option<Id>,
    workgroup_reduce_len: u32,
    t_ptr_wg_f32: Id,
    /// Workgroup dimensions, for the linear local index that subgroups and fragment staging are
    /// derived from.
    local_size: [u32; 3],
    const_0_u32: Id,
    locals: HashMap<String, LocalVar>,
    coop: Coop,
    /// Type arena clone: fragment accumulator types are interned during lowering.
    types: TypeArena,
    t_ptr_pc_u32: Id,
    t_ptr_pc_i32: Id,
    t_ptr_pc_f32: Id,
}

impl LowerCtx<'_> {
    fn spv_ty(&self, tid: TypeId) -> Id {
        if tid == self.u32_id {
            self.t_u32
        } else if tid == self.i32_id {
            self.t_i32
        } else if tid == self.f32_id {
            self.t_f32
        } else if tid == self.bitset128_id {
            self.t_uvec4
        } else {
            self.t_bool
        }
    }
}

fn lower_stmts(ctx: &mut LowerCtx<'_>, stmts: &[IrStmt]) -> Result<(), String> {
    for stmt in stmts {
        lower_stmt(ctx, stmt)?;
    }
    Ok(())
}

fn has_atomic_add(stmts: &[IrStmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        IrStmt::AtomicAdd { .. } => true,
        IrStmt::If { then, else_, .. } => {
            has_atomic_add(then) || else_.as_deref().is_some_and(has_atomic_add)
        }
        IrStmt::For { body, .. } => has_atomic_add(body),
        _ => false,
    })
}

#[allow(clippy::expect_used)]
fn lower_stmt(ctx: &mut LowerCtx<'_>, stmt: &IrStmt) -> Result<(), String> {
    match stmt {
        IrStmt::FragStore {
            src,
            offset,
            ld,
            value,
        } => ctx.lower_frag_store(src, offset, ld, value),
        IrStmt::Let { name, init } => {
            let sty = ctx.locals.get(name).map(|l| l.ty).unwrap_or(ctx.u32_id);
            let ptr = ctx
                .locals
                .get(name)
                .ok_or_else(|| format!("ZSL: undeclared local `{name}`"))?
                .ptr;
            if frag_parts(&ctx.types, sty).is_some() {
                let id = ctx.frag_value_or_zero(init, sty)?;
                ctx.spv.op_store(ptr, id);
                return Ok(());
            }
            let val = lower_expr(ctx, init)?;
            let val_id = coerce(ctx, val, sty);
            ctx.spv.op_store(ptr, val_id);
            Ok(())
        }

        IrStmt::AssignLocal { name, value } => {
            let (ptr, ty) = ctx
                .locals
                .get(name)
                .map(|l| (l.ptr, l.ty))
                .ok_or_else(|| format!("ZSL: `{name}` is not a declared local variable"))?;
            if frag_parts(&ctx.types, ty).is_some() {
                let id = ctx.frag_value_or_zero(value, ty)?;
                ctx.spv.op_store(ptr, id);
                return Ok(());
            }
            let rhs = lower_expr(ctx, value)?;
            let rhs_id = coerce(ctx, rhs, ty);
            ctx.spv.op_store(ptr, rhs_id);
            Ok(())
        }

        IrStmt::AssignBuffer { buf, index, value } => {
            let info = *ctx
                .buf_params
                .get(buf)
                .ok_or_else(|| format!("ZSL: `{buf}` is not a buffer param"))?;
            let table = *ctx
                .ssbo_tables
                .get(&info.table)
                .expect("buffer element type has no SSBO table");
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| "ZSL: no push constant block".to_string())?;
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
            let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
            let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ctx.u32_id);
            if let Some(half) = info.half {
                let rhs = lower_expr(ctx, value)?;
                let rhs_id = coerce(ctx, rhs, ctx.f32_id);
                store_half(ctx, half, table, buf_idx, idx_id, rhs_id);
                return Ok(());
            }
            let c0 = ctx.const_0_u32;
            let ptr_elem =
                ctx.spv
                    .op_access_chain(table.t_ptr_elem, table.var, &[buf_idx, c0, idx_id]);
            let rhs = lower_expr(ctx, value)?;
            let rhs_id = coerce(ctx, rhs, info.elem);
            ctx.spv.op_store(ptr_elem, rhs_id);
            Ok(())
        }

        IrStmt::AtomicAdd { buf, index, value } => {
            let buf_pc_index = ctx
                .buf_params
                .get(buf)
                .ok_or_else(|| format!("ZSL: `{buf}` is not a buffer param"))?
                .pc_index;
            let table = *ctx
                .ssbo_tables
                .get(&ctx.f32_id)
                .expect("atomic_add requires an f32 SSBO table");
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| "ZSL: no push constant block".to_string())?;
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, buf_pc_index);
            let pc_chain = ctx
                .spv
                .op_access_chain(ctx.t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ctx.u32_id);
            let ptr_elem = ctx.spv.op_access_chain(
                table.t_ptr_elem,
                table.var,
                &[buf_idx, ctx.const_0_u32, idx_id],
            );
            let rhs = lower_expr(ctx, value)?;
            let rhs_id = coerce(ctx, rhs, ctx.f32_id);
            let scope = ctx.spv.constant_u32(ctx.t_u32, 1);
            let semantics = ctx.spv.constant_u32(ctx.t_u32, 0);
            ctx.spv
                .op_atomic_fadd_ext(ctx.t_f32, ptr_elem, scope, semantics, rhs_id);
            Ok(())
        }

        IrStmt::ImageStore { image, args } => {
            let [x, y, rgba @ ..] = args.as_slice() else {
                return Err("ZSL: image store needs x, y and four channels".into());
            };
            let (info, ty, handle) = load_texture(ctx, image)?;
            if !info.storage || rgba.len() != 4 {
                return Err(format!("ZSL: bad image store on `{image}`"));
            }
            let coord = texel_coord(ctx, &ty, x, y)?;
            let mut lanes = [Id(0); 4];
            for (lane, e) in lanes.iter_mut().zip(rgba) {
                let val = lower_expr(ctx, e)?;
                *lane = coerce(ctx, val, ctx.f32_id);
            }
            let texel = ctx.spv.op_composite_construct(ty.t_vec4, &lanes);
            ctx.spv.op_image_write(handle, coord, texel);
            Ok(())
        }

        IrStmt::AssignShared { name, index, value } => {
            let var = *ctx
                .shared
                .get(name)
                .ok_or_else(|| format!("ZSL: `{name}` is not a shared array"))?;
            let idx = lower_expr(ctx, index)?;
            let idx = coerce(ctx, idx, ctx.u32_id);
            let ptr = ctx.spv.op_access_chain(ctx.t_ptr_wg_f32, var, &[idx]);
            let rhs = lower_expr(ctx, value)?;
            let rhs = coerce(ctx, rhs, ctx.f32_id);
            ctx.spv.op_store(ptr, rhs);
            Ok(())
        }

        IrStmt::Discard => Err("ZSL: discard is only available in fragment shaders".into()),
        IrStmt::Barrier => {
            let scope = ctx.spv.constant_u32(ctx.t_u32, 2);
            let semantics = ctx.spv.constant_u32(ctx.t_u32, 0x108);
            ctx.spv.op_control_barrier(scope, scope, semantics);
            Ok(())
        }

        IrStmt::If { cond, then, else_ } => {
            let cond = lower_expr(ctx, cond)?;
            let cond_id = coerce(ctx, cond, ctx.bool_id);
            let true_label = ctx.spv.fresh_id();
            let merge_label = ctx.spv.fresh_id();

            if let Some(else_block) = else_ {
                let false_label = ctx.spv.fresh_id();
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, true_label, false_label);

                ctx.spv.label_with_id(true_label);
                lower_stmts(ctx, then)?;
                ctx.spv.op_branch(merge_label);

                ctx.spv.label_with_id(false_label);
                lower_stmts(ctx, else_block)?;
                ctx.spv.op_branch(merge_label);
            } else {
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, true_label, merge_label);

                ctx.spv.label_with_id(true_label);
                lower_stmts(ctx, then)?;
                ctx.spv.op_branch(merge_label);
            }

            ctx.spv.label_with_id(merge_label);
            Ok(())
        }

        IrStmt::For { var, lo, hi, body } => {
            let lo_val = lower_expr(ctx, lo)?;
            let lo_id = coerce(ctx, lo_val, ctx.u32_id);
            let hi_val = lower_expr(ctx, hi)?;
            let hi_id = coerce(ctx, hi_val, ctx.u32_id);

            let loop_ptr =
                ctx.locals.get(var).map(|l| l.ptr).ok_or_else(|| {
                    format!("ZSL: for-loop variable `{var}` not declared as a local")
                })?;

            ctx.spv.op_store(loop_ptr, lo_id);

            let header = ctx.spv.fresh_id();
            let body_lbl = ctx.spv.fresh_id();
            let cont = ctx.spv.fresh_id();
            let merge = ctx.spv.fresh_id();

            ctx.spv.op_branch(header);

            ctx.spv.label_with_id(header);
            let t_u32 = ctx.t_u32;
            let t_bool = ctx.t_bool;
            let i_val = ctx.spv.op_load(t_u32, loop_ptr);
            let cond = ctx.spv.op_ult(t_bool, i_val, hi_id);
            ctx.spv.op_loop_merge(merge, cont);
            ctx.spv.op_branch_conditional(cond, body_lbl, merge);

            ctx.spv.label_with_id(body_lbl);
            lower_stmts(ctx, body)?;
            ctx.spv.op_branch(cont);

            ctx.spv.label_with_id(cont);
            let i_old = ctx.spv.op_load(t_u32, loop_ptr);
            let const_1 = ctx.spv.constant_u32(t_u32, 1);
            let i_new = ctx.spv.op_iadd(t_u32, i_old, const_1);
            ctx.spv.op_store(loop_ptr, i_new);
            ctx.spv.op_branch(header);

            ctx.spv.label_with_id(merge);
            Ok(())
        }

        IrStmt::Eval(expr) => {
            lower_expr(ctx, expr)?;
            Ok(())
        }
    }
}

#[allow(clippy::expect_used)]
fn lower_expr(ctx: &mut LowerCtx<'_>, expr: &IrExpr) -> Result<Val, String> {
    match expr {
        IrExpr::FragLoad {
            ty,
            src,
            offset,
            ld,
        } => ctx.lower_frag_load(*ty, src, offset, ld),
        IrExpr::LitU32(v) => {
            let id = ctx.spv.constant_u32(ctx.t_u32, *v);
            Ok(Val { id, ty: ctx.u32_id })
        }

        IrExpr::LitF32(v) => {
            let id = ctx.spv.constant_f32(ctx.t_f32, *v);
            Ok(Val { id, ty: ctx.f32_id })
        }

        IrExpr::Local(name) => {
            let local = ctx
                .locals
                .get(name)
                .ok_or_else(|| format!("ZSL: unknown identifier `{name}`"))?;
            let (ty, ptr) = (local.ty, local.ptr);
            if frag_parts(&ctx.types, ty).is_some() {
                let coop_ty = ctx.frag_spv_ty(ty)?;
                let id = ctx.spv.op_load(coop_ty, ptr);
                return Ok(Val { id, ty });
            }
            let spv_ty = ctx.spv_ty(ty);
            let id = ctx.spv.op_load(spv_ty, ptr);
            Ok(Val { id, ty })
        }

        IrExpr::ScalarParam(name) => {
            let info = *ctx
                .scalar_params
                .get(name)
                .ok_or_else(|| format!("ZSL: unknown identifier `{name}`"))?;
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| "ZSL: no push constant block".to_string())?;
            let (pc_ptr_ty, ty, spv_scalar_ty) = if info.ty == ctx.u32_id {
                (ctx.t_ptr_pc_u32, ctx.u32_id, ctx.t_u32)
            } else if info.ty == ctx.i32_id {
                (ctx.t_ptr_pc_i32, ctx.i32_id, ctx.t_i32)
            } else {
                (ctx.t_ptr_pc_f32, ctx.f32_id, ctx.t_f32)
            };
            if pc_ptr_ty == Id(0) {
                return Err("ZSL: push-constant pointer type not allocated".to_string());
            }
            let actual_idx = ctx.n_slot_params + info.pc_index;
            let pc_idx = ctx.spv.constant_u32(ctx.t_u32, actual_idx);
            let chain = ctx.spv.op_access_chain(pc_ptr_ty, pc_var, &[pc_idx]);
            let id = ctx.spv.op_load(spv_scalar_ty, chain);
            Ok(Val { id, ty })
        }

        IrExpr::BufferLoad { buf, index } => {
            let info = *ctx
                .buf_params
                .get(buf)
                .ok_or_else(|| format!("ZSL: `{buf}` is not a buffer param"))?;
            let table = *ctx
                .ssbo_tables
                .get(&info.table)
                .expect("buffer element type has no SSBO table");
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| "ZSL: no push constant block".to_string())?;
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
            let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
            let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ctx.u32_id);
            if let Some(half) = info.half {
                let id = load_half(ctx, half, table, buf_idx, idx_id);
                return Ok(Val { id, ty: ctx.f32_id });
            }
            let c0 = ctx.const_0_u32;
            let ptr_elem =
                ctx.spv
                    .op_access_chain(table.t_ptr_elem, table.var, &[buf_idx, c0, idx_id]);
            let elem_spv_ty = ctx.spv_ty(info.elem);
            let id = ctx.spv.op_load(elem_spv_ty, ptr_elem);
            Ok(Val { id, ty: info.elem })
        }

        IrExpr::SharedLoad { name, index } => {
            let var = *ctx
                .shared
                .get(name)
                .ok_or_else(|| format!("ZSL: `{name}` is not a shared array"))?;
            let idx = lower_expr(ctx, index)?;
            let idx = coerce(ctx, idx, ctx.u32_id);
            let ptr = ctx.spv.op_access_chain(ctx.t_ptr_wg_f32, var, &[idx]);
            let id = ctx.spv.op_load(ctx.t_f32, ptr);
            Ok(Val { id, ty: ctx.f32_id })
        }

        IrExpr::GlobalId(component) => {
            let gid_var = ctx.gid_var;
            let t_uvec3 = ctx.t_uvec3;
            let gid = ctx.spv.op_load(t_uvec3, gid_var);
            let t_u32 = ctx.t_u32;
            let val = ctx.spv.op_composite_extract(t_u32, gid, &[*component]);
            Ok(Val {
                id: val,
                ty: ctx.u32_id,
            })
        }

        IrExpr::LocalId(component) | IrExpr::GroupId(component) => {
            let var = if matches!(expr, IrExpr::LocalId(_)) {
                ctx.lid_var
            } else {
                ctx.group_var
            };
            let id3 = ctx.spv.op_load(ctx.t_uvec3, var);
            let id = ctx.spv.op_composite_extract(ctx.t_u32, id3, &[*component]);
            Ok(Val { id, ty: ctx.u32_id })
        }

        IrExpr::Texel {
            tex,
            op,
            args,
            component,
        } => lower_texel(ctx, tex, *op, args, *component),

        IrExpr::TexSize { tex, axis } => {
            let (info, ty, handle) = load_texture(ctx, tex)?;
            let zero = ctx.spv.constant_u32(ctx.t_i32, 0);
            let size = if info.storage {
                ctx.spv.op_image_query_size(ty.t_ivec2, handle)
            } else {
                let image = ctx.spv.op_image(ty.t_image, handle);
                ctx.spv.op_image_query_size_lod(ty.t_ivec2, image, zero)
            };
            let n = ctx.spv.op_composite_extract(ctx.t_i32, size, &[*axis]);
            let id = ctx.spv.op_bitcast(ctx.t_u32, n);
            Ok(Val { id, ty: ctx.u32_id })
        }

        IrExpr::Builtin { func, args } => lower_builtin(ctx, *func, args),

        IrExpr::Neg(inner) => {
            let val = lower_expr(ctx, inner)?;
            let ty_id = ctx.spv_ty(val.ty);
            let id = if val.ty == ctx.f32_id {
                ctx.spv.op_fnegate(ty_id, val.id)
            } else {
                ctx.spv.op_snegate(ty_id, val.id)
            };
            Ok(Val { id, ty: val.ty })
        }

        IrExpr::Binary { op, lhs, rhs } => {
            let lhs = lower_expr(ctx, lhs)?;
            let rhs = lower_expr(ctx, rhs)?;
            lower_binary(ctx, *op, lhs, rhs)
        }

        IrExpr::Input(_)
        | IrExpr::FieldAccess { .. }
        | IrExpr::Swizzle { .. }
        | IrExpr::Sample { .. }
        | IrExpr::ArrayLit(_)
        | IrExpr::Index { .. }
        | IrExpr::Construct { .. } => {
            Err("ZSL: vector/graphics expression not supported in compute shaders".to_string())
        }
    }
}

/// Handle of a `texture` (sampled) or `image` (storage) param, via its push-constant slot.
fn load_texture(ctx: &mut LowerCtx<'_>, name: &str) -> Result<(TexInfo, ImageTypes, Id), String> {
    let info = *ctx
        .tex_params
        .get(name)
        .ok_or_else(|| format!("ZSL: `{name}` is not a texture or image param"))?;
    let ty = ctx
        .images
        .ok_or_else(|| "ZSL: no texture tables".to_string())?;
    let pc_var = ctx
        .pc_var
        .ok_or_else(|| "ZSL: no push constant block".to_string())?;
    let pc_field = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
    let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
    let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
    let slot = ctx.spv.op_load(ctx.t_u32, pc_chain);
    let handle = if info.storage {
        let ptr = ctx
            .spv
            .op_access_chain(ty.t_ptr_storage, ty.storage_var, &[slot]);
        ctx.spv.op_load(ty.t_storage, ptr)
    } else {
        let ptr = ctx
            .spv
            .op_access_chain(ty.t_ptr_sampled, ty.sampled_var, &[slot]);
        ctx.spv.op_load(ty.t_sampled, ptr)
    };
    Ok((info, ty, handle))
}

/// Two `u32` expressions as the `ivec2` texel coordinate images take.
fn texel_coord(
    ctx: &mut LowerCtx<'_>,
    ty: &ImageTypes,
    x: &IrExpr,
    y: &IrExpr,
) -> Result<Id, String> {
    let mut lanes = [Id(0); 2];
    for (lane, e) in lanes.iter_mut().zip([x, y]) {
        let v = lower_expr(ctx, e)?;
        let v = coerce(ctx, v, ctx.u32_id);
        *lane = ctx.spv.op_bitcast(ctx.t_i32, v);
    }
    Ok(ctx.spv.op_composite_construct(ty.t_ivec2, &lanes))
}

fn lower_texel(
    ctx: &mut LowerCtx<'_>,
    tex: &str,
    op: TexelOp,
    args: &[IrExpr],
    component: u32,
) -> Result<Val, String> {
    let (info, ty, handle) = load_texture(ctx, tex)?;
    let texel = match (op, args) {
        (TexelOp::Load, [x, y]) => {
            let coord = texel_coord(ctx, &ty, x, y)?;
            if info.storage {
                ctx.spv.op_image_read(ty.t_vec4, handle, coord)
            } else {
                let image = ctx.spv.op_image(ty.t_image, handle);
                let zero = ctx.spv.constant_u32(ctx.t_i32, 0);
                ctx.spv.op_image_fetch(ty.t_vec4, image, coord, zero)
            }
        }
        (TexelOp::Sample | TexelOp::SampleLevel, [u, v, lod @ ..]) if !info.storage => {
            let mut lanes = [Id(0); 2];
            for (lane, e) in lanes.iter_mut().zip([u, v]) {
                let val = lower_expr(ctx, e)?;
                *lane = coerce(ctx, val, ctx.f32_id);
            }
            let coord = ctx.spv.op_composite_construct(ty.t_vec2, &lanes);
            let lod = match lod {
                [lod] => {
                    let val = lower_expr(ctx, lod)?;
                    coerce(ctx, val, ctx.f32_id)
                }
                _ => ctx.spv.constant_f32(ctx.t_f32, 0.0),
            };
            ctx.spv
                .op_image_sample_explicit_lod(ty.t_vec4, handle, coord, lod)
        }
        _ => return Err(format!("ZSL: bad texel access on `{tex}`")),
    };
    let id = ctx.spv.op_composite_extract(ctx.t_f32, texel, &[component]);
    Ok(Val { id, ty: ctx.f32_id })
}

fn lower_binary(ctx: &mut LowerCtx<'_>, op: IrBinOp, lhs: Val, rhs: Val) -> Result<Val, String> {
    match op {
        IrBinOp::Add => binary_arith(ctx, lhs, rhs, SpvBuilder::op_fadd, SpvBuilder::op_iadd),
        IrBinOp::Sub => binary_arith(ctx, lhs, rhs, SpvBuilder::op_fsub, SpvBuilder::op_isub),
        IrBinOp::Mul => binary_arith(ctx, lhs, rhs, SpvBuilder::op_fmul, SpvBuilder::op_imul),
        IrBinOp::Div => {
            let (lhs, rhs) = unify(ctx, lhs, rhs);
            let ty = ctx.spv_ty(lhs.ty);
            let id = if lhs.ty == ctx.f32_id {
                ctx.spv.op_fdiv(ty, lhs.id, rhs.id)
            } else if lhs.ty == ctx.i32_id {
                ctx.spv.op_sdiv(ty, lhs.id, rhs.id)
            } else if lhs.ty == ctx.u32_id {
                ctx.spv.op_udiv(ty, lhs.id, rhs.id)
            } else {
                return Err("ZSL: division is not supported on bool".into());
            };
            Ok(Val { id, ty: lhs.ty })
        }
        IrBinOp::Rem => {
            let (lhs, rhs) = unify(ctx, lhs, rhs);
            let ty = ctx.spv_ty(lhs.ty);
            let id = if lhs.ty == ctx.f32_id {
                ctx.spv.op_frem(ty, lhs.id, rhs.id)
            } else if lhs.ty == ctx.i32_id {
                ctx.spv.op_srem(ty, lhs.id, rhs.id)
            } else if lhs.ty == ctx.u32_id {
                ctx.spv.op_umod(ty, lhs.id, rhs.id)
            } else {
                return Err("ZSL: remainder is not supported on bool".into());
            };
            Ok(Val { id, ty: lhs.ty })
        }
        IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne => {
            let (lhs, rhs) = unify(ctx, lhs, rhs);
            let bool_ty = ctx.t_bool;
            let id = if lhs.ty == ctx.f32_id {
                match op {
                    IrBinOp::Lt => ctx.spv.op_ford_lt(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Le => ctx.spv.op_ford_le(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Gt => ctx.spv.op_ford_gt(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ge => ctx.spv.op_ford_ge(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Eq => ctx.spv.op_ford_eq(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ne => ctx.spv.op_ford_ne(bool_ty, lhs.id, rhs.id),
                    _ => unreachable!(),
                }
            } else if lhs.ty == ctx.u32_id {
                match op {
                    IrBinOp::Lt => ctx.spv.op_ult(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Le => ctx.spv.op_ule(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Gt => ctx.spv.op_ugt(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ge => ctx.spv.op_uge(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Eq => ctx.spv.op_iequal(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ne => ctx.spv.op_inot_equal(bool_ty, lhs.id, rhs.id),
                    _ => unreachable!(),
                }
            } else if lhs.ty == ctx.i32_id {
                match op {
                    IrBinOp::Lt => ctx.spv.op_slt(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Le => ctx.spv.op_sle(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Gt => ctx.spv.op_sgt(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ge => ctx.spv.op_sge(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Eq => ctx.spv.op_iequal(bool_ty, lhs.id, rhs.id),
                    IrBinOp::Ne => ctx.spv.op_inot_equal(bool_ty, lhs.id, rhs.id),
                    _ => unreachable!(),
                }
            } else {
                return Err("ZSL: comparisons not supported on bool".into());
            };
            Ok(Val {
                id,
                ty: ctx.bool_id,
            })
        }
        IrBinOp::And | IrBinOp::Or => {
            if lhs.ty != ctx.bool_id || rhs.ty != ctx.bool_id {
                return Err("ZSL: `&&`/`||` require bool operands".into());
            }
            let bool_ty = ctx.t_bool;
            let id = match op {
                IrBinOp::And => ctx.spv.op_logical_and(bool_ty, lhs.id, rhs.id),
                IrBinOp::Or => ctx.spv.op_logical_or(bool_ty, lhs.id, rhs.id),
                _ => unreachable!(),
            };
            Ok(Val {
                id,
                ty: ctx.bool_id,
            })
        }
        IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Shl | IrBinOp::Shr => {
            if lhs.ty != ctx.u32_id || rhs.ty != ctx.u32_id {
                return Err("ZSL: `&`/`|`/`^`/`<<`/`>>` currently require u32 operands".into());
            }
            let ty = ctx.spv_ty(ctx.u32_id);
            let id = match op {
                IrBinOp::BitAnd => ctx.spv.op_bitwise_and(ty, lhs.id, rhs.id),
                IrBinOp::BitOr => ctx.spv.op_bitwise_or(ty, lhs.id, rhs.id),
                IrBinOp::BitXor => ctx.spv.op_bitwise_xor(ty, lhs.id, rhs.id),
                IrBinOp::Shl => ctx.spv.op_shift_left_logical(ty, lhs.id, rhs.id),
                IrBinOp::Shr => ctx.spv.op_shift_right_logical(ty, lhs.id, rhs.id),
                _ => unreachable!(),
            };
            Ok(Val { id, ty: ctx.u32_id })
        }
    }
}

#[allow(clippy::expect_used)]
fn lower_builtin(ctx: &mut LowerCtx<'_>, func: BuiltinFn, args: &[IrExpr]) -> Result<Val, String> {
    let name = func.name();
    match func {
        BuiltinFn::Mma => ctx.lower_mma(args),
        BuiltinFn::FragZeroAcc => Err(
            "SPIR-V backend: frag_zero_acc() must initialise a frag let or assignment, or be the acc of mma"
                .into(),
        ),
        BuiltinFn::FragLen
        | BuiltinFn::FragRow
        | BuiltinFn::FragCol
        | BuiltinFn::FragGet
        | BuiltinFn::FragSet => Err(format!(
            "SPIR-V backend: {name}() is not lowered: the cooperative matrix element mapping is opaque"
        )),
        BuiltinFn::U32 => {
            if args.len() != 1 {
                return Err("ZSL: u32() takes 1 arg".into());
            }
            let v = lower_expr(ctx, &args[0])?;
            let id = coerce(ctx, v, ctx.u32_id);
            Ok(Val { id, ty: ctx.u32_id })
        }
        BuiltinFn::F32 => {
            if args.len() != 1 {
                return Err("ZSL: f32() takes 1 arg".into());
            }
            let v = lower_expr(ctx, &args[0])?;
            let id = coerce(ctx, v, ctx.f32_id);
            Ok(Val { id, ty: ctx.f32_id })
        }
        BuiltinFn::Dot | BuiltinFn::Extend | BuiltinFn::Step | BuiltinFn::Smoothstep => {
            Err("ZSL: vector/graphics expression not supported in compute shaders".to_string())
        }
        BuiltinFn::IsNan | BuiltinFn::IsInf | BuiltinFn::IsFinite => {
            if args.len() != 1 {
                return Err(format!("ZSL: {name}() takes 1 arg"));
            }
            let value = lower_expr(ctx, &args[0])?;
            if value.ty != ctx.f32_id {
                return Err(format!("ZSL: {name}() requires f32"));
            }
            let id = match func {
                BuiltinFn::IsNan => ctx.spv.op_is_nan(ctx.t_bool, value.id),
                BuiltinFn::IsInf => ctx.spv.op_is_inf(ctx.t_bool, value.id),
                BuiltinFn::IsFinite => {
                    let is_nan = ctx.spv.op_is_nan(ctx.t_bool, value.id);
                    let is_inf = ctx.spv.op_is_inf(ctx.t_bool, value.id);
                    let non_finite = ctx.spv.op_logical_or(ctx.t_bool, is_nan, is_inf);
                    ctx.spv.op_logical_not(ctx.t_bool, non_finite)
                }
                _ => unreachable!(),
            };
            Ok(Val {
                id,
                ty: ctx.bool_id,
            })
        }
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
        | BuiltinFn::Fract => {
            if args.len() != 1 {
                return Err(format!("ZSL: {name}() takes 1 arg"));
            }
            let v = lower_expr(ctx, &args[0])?;
            if v.ty != ctx.f32_id {
                return Err(format!("ZSL: {name}() requires f32"));
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
                _ => unreachable!(),
            };
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(t_f32, glsl, opcode, &[v.id]);
            Ok(Val { id, ty: ctx.f32_id })
        }
        BuiltinFn::Min | BuiltinFn::Max | BuiltinFn::Pow => {
            if args.len() != 2 {
                return Err(format!("ZSL: {name}(a, b) takes 2 args"));
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            if a.ty != ctx.f32_id {
                return Err(format!("ZSL: {name}() requires f32"));
            }
            if b.ty != ctx.f32_id {
                return Err(format!("ZSL: {name}() requires f32"));
            }
            let opcode = match func {
                BuiltinFn::Min => glsl_op::F_MIN,
                BuiltinFn::Max => glsl_op::F_MAX,
                BuiltinFn::Pow => glsl_op::POW,
                _ => unreachable!(),
            };
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx.spv.op_ext_inst(t_f32, glsl, opcode, &[a.id, b.id]);
            Ok(Val { id, ty: ctx.f32_id })
        }
        BuiltinFn::Clamp => {
            if args.len() != 3 {
                return Err("ZSL: clamp(x, lo, hi) takes 3 args".into());
            }
            let x = lower_expr(ctx, &args[0])?;
            let lo = lower_expr(ctx, &args[1])?;
            let hi = lower_expr(ctx, &args[2])?;
            if x.ty != ctx.f32_id || lo.ty != ctx.f32_id || hi.ty != ctx.f32_id {
                return Err("ZSL: clamp() requires f32 args".into());
            }
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(t_f32, glsl, glsl_op::F_CLAMP, &[x.id, lo.id, hi.id]);
            Ok(Val { id, ty: ctx.f32_id })
        }
        BuiltinFn::Mix => {
            if args.len() != 3 {
                return Err("ZSL: mix(a, b, t) takes 3 args".into());
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            let t = lower_expr(ctx, &args[2])?;
            if a.ty != ctx.f32_id || b.ty != ctx.f32_id || t.ty != ctx.f32_id {
                return Err("ZSL: mix() requires f32 args".into());
            }
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(t_f32, glsl, glsl_op::F_MIX, &[a.id, b.id, t.id]);
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::Normalize | BuiltinFn::Length => {
            Err(format!("ZSL: {name}() is not supported in compute shaders"))
        }

        BuiltinFn::MulAddF32 => {
            if args.len() != 3 {
                return Err("ZSL: mul_add_f32(a, b, acc) takes 3 args".into());
            }
            let a = lower_expr(ctx, &args[0])?;
            let b = lower_expr(ctx, &args[1])?;
            let acc = lower_expr(ctx, &args[2])?;
            if a.ty != ctx.f32_id || b.ty != ctx.f32_id || acc.ty != ctx.f32_id {
                return Err("ZSL: mul_add_f32() requires f32 args".into());
            }
            let t_f32 = ctx.t_f32;
            let glsl = ctx.glsl_ext;
            let id = ctx
                .spv
                .op_ext_inst(t_f32, glsl, glsl_op::FMA, &[a.id, b.id, acc.id]);
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::SubgroupSize | BuiltinFn::SubgroupId | BuiltinFn::SubgroupLocalId => {
            let sg = ctx
                .subgroup_vars
                .expect("stmts_use_subgroup_query should have detected this call");
            let var = match func {
                BuiltinFn::SubgroupSize => sg.size,
                BuiltinFn::SubgroupId => sg.id,
                BuiltinFn::SubgroupLocalId => sg.local_id,
                _ => unreachable!(),
            };
            let id = ctx.spv.op_load(ctx.t_u32, var);
            Ok(Val { id, ty: ctx.u32_id })
        }

        BuiltinFn::SubgroupAny | BuiltinFn::SubgroupAll => {
            if args.len() != 1 {
                return Err(format!("ZSL: {}(predicate) takes 1 arg", func.name()));
            }
            let pred = lower_expr(ctx, &args[0])?;
            if pred.ty != ctx.bool_id {
                return Err(format!("ZSL: {}() requires a bool predicate", func.name()));
            }
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let id = if matches!(func, BuiltinFn::SubgroupAny) {
                ctx.spv.op_group_non_uniform_any(ctx.t_bool, scope, pred.id)
            } else {
                ctx.spv.op_group_non_uniform_all(ctx.t_bool, scope, pred.id)
            };
            Ok(Val {
                id,
                ty: ctx.bool_id,
            })
        }

        BuiltinFn::SubgroupBallot => {
            if args.len() != 1 {
                return Err("ZSL: subgroup_ballot(predicate) takes 1 arg".to_string());
            }
            let pred = lower_expr(ctx, &args[0])?;
            if pred.ty != ctx.bool_id {
                return Err("ZSL: subgroup_ballot() requires a bool predicate".to_string());
            }
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let id = ctx
                .spv
                .op_group_non_uniform_ballot(ctx.t_uvec4, scope, pred.id);
            Ok(Val {
                id,
                ty: ctx.bitset128_id,
            })
        }

        BuiltinFn::SubgroupBroadcast | BuiltinFn::SubgroupShuffle => {
            if args.len() != 2 {
                return Err(format!("ZSL: {}(value, lane) takes 2 args", func.name()));
            }
            let value = lower_expr(ctx, &args[0])?;
            let lane = lower_expr(ctx, &args[1])?;
            if value.ty != ctx.f32_id {
                return Err(format!("ZSL: {}() requires an f32 value", func.name()));
            }
            let lane = coerce(ctx, lane, ctx.u32_id);
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let id = if matches!(func, BuiltinFn::SubgroupBroadcast) {
                ctx.spv
                    .op_group_non_uniform_broadcast(ctx.t_f32, scope, value.id, lane)
            } else {
                ctx.spv
                    .op_group_non_uniform_shuffle(ctx.t_f32, scope, value.id, lane)
            };
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::SubgroupShuffleXor => {
            if args.len() != 2 {
                return Err("ZSL: subgroup_shuffle_xor(value, mask) takes 2 args".to_string());
            }
            let value = lower_expr(ctx, &args[0])?;
            let mask = lower_expr(ctx, &args[1])?;
            if value.ty != ctx.f32_id {
                return Err("ZSL: subgroup_shuffle_xor() requires an f32 value".to_string());
            }
            let mask = coerce(ctx, mask, ctx.u32_id);
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let id = ctx
                .spv
                .op_group_non_uniform_shuffle_xor(ctx.t_f32, scope, value.id, mask);
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::Dot4I8 | BuiltinFn::Dot4U8 | BuiltinFn::Dot4IU8 => {
            if args.len() != 3 {
                return Err(format!("ZSL: {name}(a, b, acc) takes 3 args"));
            }
            let (a_signed, b_signed) = match func {
                BuiltinFn::Dot4I8 => (true, true),
                BuiltinFn::Dot4U8 => (false, false),
                _ => (true, false),
            };
            let (res_zsl, res_spv) = if matches!(func, BuiltinFn::Dot4U8) {
                (ctx.u32_id, ctx.t_u32)
            } else {
                (ctx.i32_id, ctx.t_i32)
            };
            let a_zsl = if a_signed { ctx.i32_id } else { ctx.u32_id };
            let b_zsl = if b_signed { ctx.i32_id } else { ctx.u32_id };
            let a_val = lower_expr(ctx, &args[0])?;
            let a = coerce(ctx, a_val, a_zsl);
            let b_val = lower_expr(ctx, &args[1])?;
            let b = coerce(ctx, b_val, b_zsl);
            let acc_val = lower_expr(ctx, &args[2])?;
            let mut acc = coerce(ctx, acc_val, res_zsl);
            for lane in 0..4u32 {
                let x = dot4_lane(ctx, a, a_signed, lane, res_spv);
                let y = dot4_lane(ctx, b, b_signed, lane, res_spv);
                let prod = ctx.spv.op_imul(res_spv, x, y);
                acc = ctx.spv.op_iadd(res_spv, acc, prod);
            }
            Ok(Val {
                id: acc,
                ty: res_zsl,
            })
        }

        BuiltinFn::PkAddU16 | BuiltinFn::PkMulU16 | BuiltinFn::PkMadU16 => {
            let want = if func == BuiltinFn::PkMadU16 { 3 } else { 2 };
            if args.len() != want {
                return Err(format!("ZSL: {name}() takes {want} args"));
            }
            let mut words = Vec::with_capacity(want);
            for arg in args {
                let v = lower_expr(ctx, arg)?;
                words.push(coerce(ctx, v, ctx.u32_id));
            }
            let id = match func {
                BuiltinFn::PkAddU16 => pk_add_u16(ctx, words[0], words[1]),
                BuiltinFn::PkMulU16 => pk_mul_u16(ctx, words[0], words[1]),
                _ => {
                    let prod = pk_mul_u16(ctx, words[0], words[1]);
                    pk_add_u16(ctx, prod, words[2])
                }
            };
            Ok(Val { id, ty: ctx.u32_id })
        }

        BuiltinFn::PkAddF16 | BuiltinFn::PkMulF16 | BuiltinFn::PkFmaF16 => {
            let want = if func == BuiltinFn::PkFmaF16 { 3 } else { 2 };
            if args.len() != want {
                return Err(format!("ZSL: {name}() takes {want} args"));
            }
            let mut words = Vec::with_capacity(want);
            for arg in args {
                let v = lower_expr(ctx, arg)?;
                words.push(coerce(ctx, v, ctx.u32_id));
            }
            let t_f32 = ctx.t_f32;
            let (alo, ahi) = f16_lanes(ctx, words[0]);
            let (blo, bhi) = f16_lanes(ctx, words[1]);
            let (lo, hi) = match func {
                BuiltinFn::PkAddF16 => (
                    ctx.spv.op_fadd(t_f32, alo, blo),
                    ctx.spv.op_fadd(t_f32, ahi, bhi),
                ),
                BuiltinFn::PkMulF16 => (
                    ctx.spv.op_fmul(t_f32, alo, blo),
                    ctx.spv.op_fmul(t_f32, ahi, bhi),
                ),
                _ => {
                    let (clo, chi) = f16_lanes(ctx, words[2]);
                    (
                        fma_f16_lane(ctx, alo, blo, clo),
                        fma_f16_lane(ctx, ahi, bhi, chi),
                    )
                }
            };
            let id = f16_pack(ctx, lo, hi);
            Ok(Val { id, ty: ctx.u32_id })
        }

        BuiltinFn::Dot2F16 => {
            if args.len() != 3 {
                return Err("ZSL: dot2_f16(a, b, acc) takes 3 args".into());
            }
            let a_val = lower_expr(ctx, &args[0])?;
            let a = coerce(ctx, a_val, ctx.u32_id);
            let b_val = lower_expr(ctx, &args[1])?;
            let b = coerce(ctx, b_val, ctx.u32_id);
            let acc_val = lower_expr(ctx, &args[2])?;
            let acc = coerce(ctx, acc_val, ctx.f32_id);
            let t_f32 = ctx.t_f32;
            let (alo, ahi) = f16_lanes(ctx, a);
            let (blo, bhi) = f16_lanes(ctx, b);
            let p0 = ctx.spv.op_fmul(t_f32, alo, blo);
            let s0 = ctx.spv.op_fadd(t_f32, acc, p0);
            let p1 = ctx.spv.op_fmul(t_f32, ahi, bhi);
            let id = ctx.spv.op_fadd(t_f32, s0, p1);
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::Select => {
            if args.len() != 3 {
                return Err("ZSL: select(cond, a, b) takes 3 args".into());
            }
            let cond = lower_expr(ctx, &args[0])?;
            let a = lower_expr(ctx, &args[1])?;
            let b = lower_expr(ctx, &args[2])?;
            let (a, b) = unify(ctx, a, b);
            let ty = ctx.spv_ty(a.ty);
            let id = ctx.spv.op_select(ty, cond.id, a.id, b.id);
            Ok(Val { id, ty: a.ty })
        }
        BuiltinFn::FunnelShr => {
            if args.len() != 3 {
                return Err("ZSL: funnel_shr(hi, lo, s) takes 3 args".into());
            }
            let hi_val = lower_expr(ctx, &args[0])?;
            let hi = coerce(ctx, hi_val, ctx.u32_id);
            let lo_val = lower_expr(ctx, &args[1])?;
            let lo = coerce(ctx, lo_val, ctx.u32_id);
            let s_val = lower_expr(ctx, &args[2])?;
            let s = coerce(ctx, s_val, ctx.u32_id);
            let t_u32 = ctx.t_u32;
            let c1 = ctx.spv.constant_u32(t_u32, 1);
            let c31 = ctx.spv.constant_u32(t_u32, 31);
            let sh = ctx.spv.op_bitwise_and(t_u32, s, c31);
            let right = ctx.spv.op_shift_right_logical(t_u32, lo, sh);
            let inv = ctx.spv.op_isub(t_u32, c31, sh);
            let left = ctx.spv.op_shift_left_logical(t_u32, hi, inv);
            let left = ctx.spv.op_shift_left_logical(t_u32, left, c1);
            let id = ctx.spv.op_bitwise_or(t_u32, right, left);
            Ok(Val { id, ty: ctx.u32_id })
        }

        BuiltinFn::BytePerm => {
            if args.len() != 3 {
                return Err("ZSL: byte_perm(a, b, sel) takes 3 args".into());
            }
            let a_val = lower_expr(ctx, &args[0])?;
            let a = coerce(ctx, a_val, ctx.u32_id);
            let b_val = lower_expr(ctx, &args[1])?;
            let b = coerce(ctx, b_val, ctx.u32_id);
            let sel_val = lower_expr(ctx, &args[2])?;
            let sel = coerce(ctx, sel_val, ctx.u32_id);
            let t_u32 = ctx.t_u32;
            let t_bool = ctx.t_bool;
            let c3 = ctx.spv.constant_u32(t_u32, 3);
            let c4 = ctx.spv.constant_u32(t_u32, 4);
            let c7 = ctx.spv.constant_u32(t_u32, 7);
            let c_ff = ctx.spv.constant_u32(t_u32, 0xff);
            let mut out = ctx.spv.constant_u32(t_u32, 0);
            for i in 0..4u32 {
                let shift = ctx.spv.constant_u32(t_u32, 4 * i);
                let raw = ctx.spv.op_shift_right_logical(t_u32, sel, shift);
                let nib = ctx.spv.op_bitwise_and(t_u32, raw, c7);
                let from_a = ctx.spv.op_ult(t_bool, nib, c4);
                let word = ctx.spv.op_select(t_u32, from_a, a, b);
                let lane = ctx.spv.op_bitwise_and(t_u32, nib, c3);
                let bit_off = ctx.spv.op_shift_left_logical(t_u32, lane, c3);
                let picked = ctx.spv.op_shift_right_logical(t_u32, word, bit_off);
                let byte = ctx.spv.op_bitwise_and(t_u32, picked, c_ff);
                let dst = ctx.spv.constant_u32(t_u32, 8 * i);
                let placed = ctx.spv.op_shift_left_logical(t_u32, byte, dst);
                out = ctx.spv.op_bitwise_or(t_u32, out, placed);
            }
            Ok(Val {
                id: out,
                ty: ctx.u32_id,
            })
        }

        BuiltinFn::SubgroupShuffleUp | BuiltinFn::SubgroupShuffleDown => {
            if args.len() != 2 {
                return Err(format!("ZSL: {}(value, delta) takes 2 args", func.name()));
            }
            let value = lower_expr(ctx, &args[0])?;
            let delta = lower_expr(ctx, &args[1])?;
            if value.ty != ctx.f32_id {
                return Err(format!("ZSL: {}() requires an f32 value", func.name()));
            }
            let delta = coerce(ctx, delta, ctx.u32_id);
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let id = if matches!(func, BuiltinFn::SubgroupShuffleUp) {
                ctx.spv
                    .op_group_non_uniform_shuffle_up(ctx.t_f32, scope, value.id, delta)
            } else {
                ctx.spv
                    .op_group_non_uniform_shuffle_down(ctx.t_f32, scope, value.id, delta)
            };
            Ok(Val { id, ty: ctx.f32_id })
        }

        BuiltinFn::SubgroupReduceAdd
        | BuiltinFn::SubgroupReduceMin
        | BuiltinFn::SubgroupReduceMax
        | BuiltinFn::SubgroupReduceAnd
        | BuiltinFn::SubgroupReduceOr => {
            if args.len() != 1 {
                return Err(format!("ZSL: {}(value) takes 1 arg", func.name()));
            }
            let value = lower_expr(ctx, &args[0])?;
            let opcode = reduce_opcode(func, value.ty, ctx)?;
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let ty = ctx.spv_ty(value.ty);
            let id = ctx
                .spv
                .op_group_non_uniform_arith(opcode, ty, scope, 0, value.id);
            Ok(Val { id, ty: value.ty })
        }

        BuiltinFn::SubgroupScanInclusiveAdd | BuiltinFn::SubgroupScanExclusiveAdd => {
            if args.len() != 1 {
                return Err(format!("ZSL: {}(value) takes 1 arg", func.name()));
            }
            let value = lower_expr(ctx, &args[0])?;
            let opcode = add_opcode(value.ty, ctx)?;
            let scope = ctx.spv.constant_u32(ctx.t_u32, 3);
            let ty = ctx.spv_ty(value.ty);
            let group_op = if matches!(func, BuiltinFn::SubgroupScanInclusiveAdd) {
                1
            } else {
                2
            };
            let id = ctx
                .spv
                .op_group_non_uniform_arith(opcode, ty, scope, group_op, value.id);
            Ok(Val { id, ty: value.ty })
        }

        BuiltinFn::WorkgroupReduceAdd
        | BuiltinFn::WorkgroupReduceMin
        | BuiltinFn::WorkgroupReduceMax => {
            if args.len() != 1 {
                return Err(format!("ZSL: {}(value) takes 1 arg", func.name()));
            }
            let value = lower_expr(ctx, &args[0])?;
            if value.ty != ctx.f32_id {
                return Err(format!("ZSL: {}() requires an f32 value", func.name()));
            }
            let var = ctx
                .workgroup_reduce_var
                .expect("stmts_use_workgroup_reduce should have detected this call");
            let n = ctx.workgroup_reduce_len;

            let lid3 = ctx.spv.op_load(ctx.t_uvec3, ctx.lid_var);
            let lid_x = ctx.spv.op_composite_extract(ctx.t_u32, lid3, &[0]);

            let barrier_scope = ctx.spv.constant_u32(ctx.t_u32, 2);
            let barrier_semantics = ctx.spv.constant_u32(ctx.t_u32, 0x108);

            ctx.spv
                .op_control_barrier(barrier_scope, barrier_scope, barrier_semantics);

            let ptr = ctx.spv.op_access_chain(ctx.t_ptr_wg_f32, var, &[lid_x]);
            ctx.spv.op_store(ptr, value.id);

            ctx.spv
                .op_control_barrier(barrier_scope, barrier_scope, barrier_semantics);

            let mut s = n / 2;
            while s > 0 {
                let s_const = ctx.spv.constant_u32(ctx.t_u32, s);
                let cond = ctx.spv.op_ult(ctx.t_bool, lid_x, s_const);
                let true_label = ctx.spv.fresh_id();
                let merge_label = ctx.spv.fresh_id();
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv.op_branch_conditional(cond, true_label, merge_label);

                ctx.spv.label_with_id(true_label);
                let lid_plus_s = ctx.spv.op_iadd(ctx.t_u32, lid_x, s_const);
                let ptr_lo = ctx.spv.op_access_chain(ctx.t_ptr_wg_f32, var, &[lid_x]);
                let ptr_hi = ctx
                    .spv
                    .op_access_chain(ctx.t_ptr_wg_f32, var, &[lid_plus_s]);
                let v_lo = ctx.spv.op_load(ctx.t_f32, ptr_lo);
                let v_hi = ctx.spv.op_load(ctx.t_f32, ptr_hi);
                let combined = match func {
                    BuiltinFn::WorkgroupReduceAdd => ctx.spv.op_fadd(ctx.t_f32, v_lo, v_hi),
                    BuiltinFn::WorkgroupReduceMin => {
                        let glsl = ctx.glsl_ext;
                        ctx.spv
                            .op_ext_inst(ctx.t_f32, glsl, glsl_op::F_MIN, &[v_lo, v_hi])
                    }
                    BuiltinFn::WorkgroupReduceMax => {
                        let glsl = ctx.glsl_ext;
                        ctx.spv
                            .op_ext_inst(ctx.t_f32, glsl, glsl_op::F_MAX, &[v_lo, v_hi])
                    }
                    _ => unreachable!(),
                };
                ctx.spv.op_store(ptr_lo, combined);
                ctx.spv.op_branch(merge_label);

                ctx.spv.label_with_id(merge_label);
                ctx.spv
                    .op_control_barrier(barrier_scope, barrier_scope, barrier_semantics);

                s /= 2;
            }

            let zero_const = ctx.spv.constant_u32(ctx.t_u32, 0);
            let ptr_zero = ctx
                .spv
                .op_access_chain(ctx.t_ptr_wg_f32, var, &[zero_const]);
            let result = ctx.spv.op_load(ctx.t_f32, ptr_zero);
            Ok(Val {
                id: result,
                ty: ctx.f32_id,
            })
        }
    }
}

fn add_opcode(ty: TypeId, ctx: &LowerCtx<'_>) -> Result<u32, String> {
    if ty == ctx.f32_id {
        Ok(op::GROUP_NON_UNIFORM_FADD)
    } else if ty == ctx.u32_id || ty == ctx.i32_id {
        Ok(op::GROUP_NON_UNIFORM_IADD)
    } else {
        Err("ZSL: subgroup reduce/scan add requires f32, u32, or i32".to_string())
    }
}

fn reduce_opcode(func: BuiltinFn, ty: TypeId, ctx: &LowerCtx<'_>) -> Result<u32, String> {
    match func {
        BuiltinFn::SubgroupReduceAdd => add_opcode(ty, ctx),
        BuiltinFn::SubgroupReduceMin => {
            if ty == ctx.f32_id {
                Ok(op::GROUP_NON_UNIFORM_FMIN)
            } else if ty == ctx.u32_id {
                Ok(op::GROUP_NON_UNIFORM_UMIN)
            } else if ty == ctx.i32_id {
                Ok(op::GROUP_NON_UNIFORM_SMIN)
            } else {
                Err("ZSL: subgroup_reduce_min requires f32, u32, or i32".to_string())
            }
        }
        BuiltinFn::SubgroupReduceMax => {
            if ty == ctx.f32_id {
                Ok(op::GROUP_NON_UNIFORM_FMAX)
            } else if ty == ctx.u32_id {
                Ok(op::GROUP_NON_UNIFORM_UMAX)
            } else if ty == ctx.i32_id {
                Ok(op::GROUP_NON_UNIFORM_SMAX)
            } else {
                Err("ZSL: subgroup_reduce_max requires f32, u32, or i32".to_string())
            }
        }
        BuiltinFn::SubgroupReduceAnd => {
            if ty == ctx.u32_id {
                Ok(op::GROUP_NON_UNIFORM_BITWISE_AND)
            } else {
                Err("ZSL: subgroup_reduce_and requires u32".to_string())
            }
        }
        BuiltinFn::SubgroupReduceOr => {
            if ty == ctx.u32_id {
                Ok(op::GROUP_NON_UNIFORM_BITWISE_OR)
            } else {
                Err("ZSL: subgroup_reduce_or requires u32".to_string())
            }
        }
        _ => unreachable!(),
    }
}

const NO_CONTRACTION: u32 = 42;
const GLSL_UNPACK_HALF_2X16: u32 = 62;

/// `u32` word holding half element `idx`, and its bit offset (0 or 16).
fn half_word(ctx: &mut LowerCtx<'_>, table: SsboTable, buf_idx: Id, idx: Id) -> (Id, Id) {
    let t_u32 = ctx.t_u32;
    let c1 = ctx.spv.constant_u32(t_u32, 1);
    let c4 = ctx.spv.constant_u32(t_u32, 4);
    let word = ctx.spv.op_shift_right_logical(t_u32, idx, c1);
    let ptr = ctx.spv.op_access_chain(
        table.t_ptr_elem,
        table.var,
        &[buf_idx, ctx.const_0_u32, word],
    );
    let lane = ctx.spv.op_bitwise_and(t_u32, idx, c1);
    let shift = ctx.spv.op_shift_left_logical(t_u32, lane, c4);
    (ptr, shift)
}

fn load_half(ctx: &mut LowerCtx<'_>, half: Half, table: SsboTable, buf_idx: Id, idx: Id) -> Id {
    let (t_u32, t_f32) = (ctx.t_u32, ctx.t_f32);
    let (ptr, shift) = half_word(ctx, table, buf_idx, idx);
    let word = ctx.spv.op_load(t_u32, ptr);
    let shifted = ctx.spv.op_shift_right_logical(t_u32, word, shift);
    match half {
        Half::F16 => {
            let v2 =
                ctx.spv
                    .op_ext_inst(ctx.t_vec2, ctx.glsl_ext, GLSL_UNPACK_HALF_2X16, &[shifted]);
            ctx.spv.op_composite_extract(t_f32, v2, &[0])
        }
        Half::Bf16 => {
            let c16 = ctx.spv.constant_u32(t_u32, 16);
            let high = ctx.spv.op_shift_left_logical(t_u32, shifted, c16);
            ctx.spv.op_bitcast(t_f32, high)
        }
    }
}

/// Store `value` as f16 into half `idx`. Two atomics (clear, then set) so a neighbour's half
/// is not lost.
fn store_half(
    ctx: &mut LowerCtx<'_>,
    half: Half,
    table: SsboTable,
    buf_idx: Id,
    idx: Id,
    value: Id,
) {
    let t_u32 = ctx.t_u32;
    let c16 = ctx.spv.constant_u32(t_u32, 16);
    let bits = match half {
        Half::F16 => f32_to_f16_rne(ctx, value),
        Half::Bf16 => {
            let c1 = ctx.spv.constant_u32(t_u32, 1);
            let c7fff = ctx.spv.constant_u32(t_u32, 0x7fff);
            let raw = ctx.spv.op_bitcast(t_u32, value);
            let upper = ctx.spv.op_shift_right_logical(t_u32, raw, c16);
            let lsb = ctx.spv.op_bitwise_and(t_u32, upper, c1);
            let bias = ctx.spv.op_iadd(t_u32, c7fff, lsb);
            let rounded = ctx.spv.op_iadd(t_u32, raw, bias);
            let rounded = ctx.spv.op_shift_right_logical(t_u32, rounded, c16);
            let mag_mask = ctx.spv.constant_u32(t_u32, 0x7fff_ffff);
            let inf = ctx.spv.constant_u32(t_u32, 0x7f80_0000);
            let quiet_bit = ctx.spv.constant_u32(t_u32, 0x40);
            let mag = ctx.spv.op_bitwise_and(t_u32, raw, mag_mask);
            let is_nan = ctx.spv.op_ult(ctx.t_bool, inf, mag);
            let upper_quiet = ctx.spv.op_bitwise_or(t_u32, upper, quiet_bit);
            ctx.spv.op_select(t_u32, is_nan, upper_quiet, rounded)
        }
    };
    let (ptr, shift) = half_word(ctx, table, buf_idx, idx);
    let lane_mask = ctx.spv.constant_u32(t_u32, 0xffff);
    let all = ctx.spv.constant_u32(t_u32, u32::MAX);
    let own = ctx.spv.op_shift_left_logical(t_u32, lane_mask, shift);
    let keep = ctx.spv.op_bitwise_xor(t_u32, own, all);
    let placed = ctx.spv.op_shift_left_logical(t_u32, bits, shift);
    let scope = ctx.spv.constant_u32(t_u32, 1);
    let relaxed = ctx.const_0_u32;
    ctx.spv
        .op_atomic_int(op::ATOMIC_AND, t_u32, ptr, scope, relaxed, keep);
    ctx.spv
        .op_atomic_int(op::ATOMIC_OR, t_u32, ptr, scope, relaxed, placed);
}

fn dot4_lane(ctx: &mut LowerCtx<'_>, word: Id, signed: bool, lane: u32, want: Id) -> Id {
    let offset = ctx.spv.constant_u32(ctx.t_u32, 8 * lane);
    let count = ctx.spv.constant_u32(ctx.t_u32, 8);
    let base_ty = if signed { ctx.t_i32 } else { ctx.t_u32 };
    let field = if signed {
        ctx.spv.op_bitfield_sextract(base_ty, word, offset, count)
    } else {
        ctx.spv.op_bitfield_uextract(base_ty, word, offset, count)
    };
    if base_ty == want {
        field
    } else {
        ctx.spv.op_bitcast(want, field)
    }
}

/// Unpack the two f16 lanes of `word` to f32 (exact), lane 0 first.
fn f16_lanes(ctx: &mut LowerCtx<'_>, word: Id) -> (Id, Id) {
    let t_f32 = ctx.t_f32;
    let v2 = ctx
        .spv
        .op_ext_inst(ctx.t_vec2, ctx.glsl_ext, GLSL_UNPACK_HALF_2X16, &[word]);
    let lo = ctx.spv.op_composite_extract(t_f32, v2, &[0]);
    let hi = ctx.spv.op_composite_extract(t_f32, v2, &[1]);
    (lo, hi)
}

/// Round two f32 to f16 lanes (lane 0 low), nearest even. Integer ops only: PackHalf2x16 rounds
/// toward zero on RADV.
fn f16_pack(ctx: &mut LowerCtx<'_>, lo: Id, hi: Id) -> Id {
    let lo_bits = f32_to_f16_rne(ctx, lo);
    let hi_bits = f32_to_f16_rne(ctx, hi);
    let c16 = ctx.spv.constant_u32(ctx.t_u32, 16);
    let high = ctx.spv.op_shift_left_logical(ctx.t_u32, hi_bits, c16);
    ctx.spv.op_bitwise_or(ctx.t_u32, lo_bits, high)
}

/// f32 to f16 bits in a u32 low half, nearest even. Subnormals: add-2^23 trick, hardware RNE on
/// multiples of 2^-24.
fn f32_to_f16_rne(ctx: &mut LowerCtx<'_>, v: Id) -> Id {
    let (t_u32, t_f32, t_bool) = (ctx.t_u32, ctx.t_f32, ctx.t_bool);
    let k = |ctx: &mut LowerCtx<'_>, x: u32| ctx.spv.constant_u32(t_u32, x);
    let bits = ctx.spv.op_bitcast(t_u32, v);
    let sign_bit = k(ctx, 0x8000_0000);
    let sign_raw = ctx.spv.op_bitwise_and(t_u32, bits, sign_bit);
    let c16 = k(ctx, 16);
    let sign = ctx.spv.op_shift_right_logical(t_u32, sign_raw, c16);
    let mag_mask = k(ctx, 0x7fff_ffff);
    let abs = ctx.spv.op_bitwise_and(t_u32, bits, mag_mask);
    let rebias = k(ctx, 0x3800_0000);
    let c13 = k(ctx, 13);
    let c1 = k(ctx, 1);
    let c_fff = k(ctx, 0xfff);
    let shifted = ctx.spv.op_shift_right_logical(t_u32, abs, c13);
    let odd = ctx.spv.op_bitwise_and(t_u32, shifted, c1);
    let re = ctx.spv.op_isub(t_u32, abs, rebias);
    let re = ctx.spv.op_iadd(t_u32, re, c_fff);
    let re = ctx.spv.op_iadd(t_u32, re, odd);
    let normal = ctx.spv.op_shift_right_logical(t_u32, re, c13);
    let two24 = ctx.spv.constant_f32(t_f32, 16_777_216.0);
    let magic = ctx.spv.constant_u32(t_u32, 0x4b00_0000);
    let abs_f = ctx.spv.op_bitcast(t_f32, abs);
    let scaled = ctx.spv.op_fmul(t_f32, abs_f, two24);
    let magic_f = ctx.spv.op_bitcast(t_f32, magic);
    let rounded = ctx.spv.op_fadd(t_f32, scaled, magic_f);
    let rounded_bits = ctx.spv.op_bitcast(t_u32, rounded);
    let sub = ctx.spv.op_isub(t_u32, rounded_bits, magic);
    let min_normal = k(ctx, 0x3880_0000);
    let huge = k(ctx, 0x4780_0000);
    let inf = k(ctx, 0x7f80_0000);
    let is_sub = ctx.spv.op_ult(t_bool, abs, min_normal);
    let is_huge = ctx.spv.op_uge(t_bool, abs, huge);
    let is_nan = ctx.spv.op_ult(t_bool, inf, abs);
    let inf_bits = k(ctx, 0x7c00);
    let qnan = k(ctx, 0x7e00);
    let h = ctx.spv.op_select(t_u32, is_sub, sub, normal);
    let h = ctx.spv.op_select(t_u32, is_huge, inf_bits, h);
    let h = ctx.spv.op_select(t_u32, is_nan, qnan, h);
    ctx.spv.op_bitwise_or(t_u32, sign, h)
}

/// `a * b + c` for one f16 lane, as f32 (caller packs). Round-to-odd on the f32 sum via TwoSum:
/// nudge one ulp toward the error when nonzero and the last bit is even. Packing to f16 then rounds
/// once, correctly (24 >= 2*11+2).
fn fma_f16_lane(ctx: &mut LowerCtx<'_>, a: Id, b: Id, c: Id) -> Id {
    let (t_f32, t_u32, t_bool) = (ctx.t_f32, ctx.t_u32, ctx.t_bool);
    let zero = ctx.spv.constant_f32(t_f32, 0.0);
    let c0 = ctx.spv.constant_u32(t_u32, 0);
    let c1 = ctx.spv.constant_u32(t_u32, 1);
    let c2 = ctx.spv.constant_u32(t_u32, 2);
    let c31 = ctx.spv.constant_u32(t_u32, 31);
    let all = ctx.spv.constant_u32(t_u32, u32::MAX);
    let p = ctx.spv.op_fmul(t_f32, a, b);
    let s = ctx.spv.op_fadd(t_f32, p, c);
    let bp = ctx.spv.op_fsub(t_f32, s, p);
    let t = ctx.spv.op_fsub(t_f32, s, bp);
    let e1 = ctx.spv.op_fsub(t_f32, p, t);
    let e2 = ctx.spv.op_fsub(t_f32, c, bp);
    let err = ctx.spv.op_fadd(t_f32, e1, e2);
    for id in [p, s, bp, t, e1, e2, err] {
        ctx.spv.decorate(id, NO_CONTRACTION, &[]);
    }
    let bits = ctx.spv.op_bitcast(t_u32, s);
    let inexact = ctx.spv.op_ford_ne(t_bool, err, zero);
    let lsb = ctx.spv.op_bitwise_and(t_u32, bits, c1);
    let even = ctx.spv.op_iequal(t_bool, lsb, c0);
    let adjust = ctx.spv.op_logical_and(t_bool, inexact, even);
    let err_pos = ctx.spv.op_ford_gt(t_bool, err, zero);
    let sign = ctx.spv.op_shift_right_logical(t_u32, bits, c31);
    let s_neg = ctx.spv.op_iequal(t_bool, sign, c1);
    let pos_u = ctx.spv.op_select(t_u32, err_pos, c1, c0);
    let neg_u = ctx.spv.op_select(t_u32, s_neg, c1, c0);
    let flip = ctx.spv.op_bitwise_xor(t_u32, pos_u, neg_u);
    let twice = ctx.spv.op_imul(t_u32, flip, c2);
    let delta = ctx.spv.op_iadd(t_u32, twice, all);
    let stepped = ctx.spv.op_iadd(t_u32, bits, delta);
    let fixed = ctx.spv.op_select(t_u32, adjust, stepped, bits);
    ctx.spv.op_bitcast(t_f32, fixed)
}

fn pk_add_u16(ctx: &mut LowerCtx<'_>, a: Id, b: Id) -> Id {
    let t_u32 = ctx.t_u32;
    let low15 = ctx.spv.constant_u32(t_u32, 0x7fff_7fff);
    let top = ctx.spv.constant_u32(t_u32, 0x8000_8000);
    let a_low = ctx.spv.op_bitwise_and(t_u32, a, low15);
    let b_low = ctx.spv.op_bitwise_and(t_u32, b, low15);
    let sum = ctx.spv.op_iadd(t_u32, a_low, b_low);
    let ab = ctx.spv.op_bitwise_xor(t_u32, a, b);
    let top_bits = ctx.spv.op_bitwise_and(t_u32, ab, top);
    ctx.spv.op_bitwise_xor(t_u32, sum, top_bits)
}

fn pk_mul_u16(ctx: &mut LowerCtx<'_>, a: Id, b: Id) -> Id {
    let t_u32 = ctx.t_u32;
    let mask = ctx.spv.constant_u32(t_u32, 0xffff);
    let c16 = ctx.spv.constant_u32(t_u32, 16);
    let prod = ctx.spv.op_imul(t_u32, a, b);
    let lo = ctx.spv.op_bitwise_and(t_u32, prod, mask);
    let a_hi = ctx.spv.op_shift_right_logical(t_u32, a, c16);
    let b_hi = ctx.spv.op_shift_right_logical(t_u32, b, c16);
    let hi_prod = ctx.spv.op_imul(t_u32, a_hi, b_hi);
    let hi = ctx.spv.op_shift_left_logical(t_u32, hi_prod, c16);
    ctx.spv.op_bitwise_or(t_u32, lo, hi)
}

fn coerce(ctx: &mut LowerCtx<'_>, val: Val, target: TypeId) -> Id {
    if val.ty == target {
        return val.id;
    }
    if val.ty == ctx.u32_id && target == ctx.f32_id {
        let f32_ty = ctx.t_f32;
        ctx.spv.op_convert_u_to_f(f32_ty, val.id)
    } else if val.ty == ctx.f32_id && target == ctx.u32_id {
        let u32_ty = ctx.t_u32;
        ctx.spv.op_convert_f_to_u(u32_ty, val.id)
    } else if val.ty == ctx.i32_id && target == ctx.f32_id {
        let f32_ty = ctx.t_f32;
        ctx.spv.op_convert_s_to_f(f32_ty, val.id)
    } else if val.ty == ctx.f32_id && target == ctx.i32_id {
        let i32_ty = ctx.t_i32;
        ctx.spv.op_convert_f_to_s(i32_ty, val.id)
    } else if val.ty == ctx.u32_id && target == ctx.i32_id {
        ctx.spv.op_bitcast(ctx.t_i32, val.id)
    } else if val.ty == ctx.i32_id && target == ctx.u32_id {
        ctx.spv.op_bitcast(ctx.t_u32, val.id)
    } else {
        val.id
    }
}

fn unify(ctx: &mut LowerCtx<'_>, lhs: Val, rhs: Val) -> (Val, Val) {
    if lhs.ty == rhs.ty {
        return (lhs, rhs);
    }
    let target = if lhs.ty == ctx.f32_id || rhs.ty == ctx.f32_id {
        ctx.f32_id
    } else if lhs.ty == ctx.i32_id || rhs.ty == ctx.i32_id {
        ctx.i32_id
    } else {
        ctx.u32_id
    };
    (
        Val {
            id: coerce(ctx, lhs, target),
            ty: target,
        },
        Val {
            id: coerce(ctx, rhs, target),
            ty: target,
        },
    )
}

fn binary_arith(
    ctx: &mut LowerCtx<'_>,
    lhs: Val,
    rhs: Val,
    float_op: fn(&mut SpvBuilder, Id, Id, Id) -> Id,
    int_op: fn(&mut SpvBuilder, Id, Id, Id) -> Id,
) -> Result<Val, String> {
    let (lhs, rhs) = unify(ctx, lhs, rhs);
    let ty = ctx.spv_ty(lhs.ty);
    let id = if lhs.ty == ctx.f32_id {
        float_op(ctx.spv, ty, lhs.id, rhs.id)
    } else {
        int_op(ctx.spv, ty, lhs.id, rhs.id)
    };
    Ok(Val { id, ty: lhs.ty })
}

#[cfg(test)]
mod tests {
    use super::lower_compute;
    use crate::backend::spirv::graphics_ir::tests::validate;
    use crate::frontend::parser::parse_compute;

    #[test]
    fn texture_and_image_params_lower_to_valid_spirv() {
        let module = parse_compute(
            r#"
            push P { scale: f32 }
            @workgroup_size(8, 8)
            kernel k(src: texture, mid: image, dst: image, p: P, id: global_id) {
                let x = id.x
                let y = id.y
                if x < dst.width() && y < src.height() {
                    let u = (x + 0.5) / src.width()
                    let a = src.load(x, y).r + src.sample(u, 0.5).g + src.sample_level(u, 0.5, 1.0).b
                    dst.store(x, y, a * p.scale, mid.load(x, y).g, 0.0, 1.0)
                }
            }
        "#,
        )
        .expect("parse");
        validate(&lower_compute(&module).expect("lower"));
    }
}
