//! IR → SPIR-V lowering for compute shaders.
//!
//! Consumes a [`Module`] built by [`crate::ir::build`]. Identifiers are already
//! resolved (local / scalar param / buffer) and structural validity checked;
//! this pass does value-type inference and emits SPIR-V.
//!
//! Two-pass: all `OpVariable` declarations are hoisted to the entry block
//! (SPIR-V requires this), then statements are lowered.

/// GLSL.std.450 extended instruction opcodes used by ZSL compute builtins.
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

use crate::backend::spirv::builder::{Id, SpvBuilder, builtin, deco, sc};
use crate::frontend::diagnostic::Diagnostic;
use crate::ir::binding::ssbo_binding_for;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::TypeId;
use crate::ir::{EntryKind, Module, ParamKind};

// ── Public entry ─────────────────────────────────────────────────────────────

/// Lower a compute [`Module`] to SPIR-V words.
pub fn lower_compute(module: &Module) -> Result<Vec<u32>, Diagnostic> {
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
    let uses_subgroup_query = crate::ir::node::stmts_use_subgroup_query(&entry.body);
    let (
        uses_subgroup_vote,
        uses_subgroup_ballot,
        uses_subgroup_shuffle,
        uses_subgroup_shuffle_relative,
    ) = crate::ir::node::stmts_use_subgroup_collectives(&entry.body);
    if uses_subgroup_query
        || uses_subgroup_vote
        || uses_subgroup_ballot
        || uses_subgroup_shuffle
        || uses_subgroup_shuffle_relative
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
    let glsl_ext = spv.ext_inst_import_glsl();
    spv.memory_model_logical_glsl450();

    // ── Canonical scalar TypeIds ─────────────────────────────────────────────
    // A local, mutable clone of the module's type arena: reuses every TypeId
    // the parser already interned and adds the four canonical scalars if a
    // module happens not to reference one (e.g. a kernel with no i32 use).
    // This arena is never written back to `module` — it exists only so this
    // lowering pass has stable `TypeId`s to compare against.
    let mut types = module.types.clone();
    let u32_id = types.u32_id();
    let i32_id = types.i32_id();
    let f32_id = types.f32_id();
    let bool_id = types.bool_id();
    let bitset128_id = types.bitset128_id();

    // ── Core scalar types ────────────────────────────────────────────────────
    let t_void = spv.type_void();
    let t_bool = spv.type_bool();
    let t_u32 = spv.type_int(32, false);
    let t_i32 = spv.type_int(32, true);
    let t_f32 = spv.type_float(32);
    let t_uvec3 = spv.type_vector(t_u32, 3);
    let t_uvec4 = spv.type_vector(t_u32, 4);

    // ── Classify params (declaration order) ──────────────────────────────────
    // Buffer writability is already enforced in `ir::build`; the backend only
    // needs the names (for push-constant index assignment).
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

    // ── SSBO types — bindless model, one typed table per element type ───────
    // Generates one `layout(set=0,binding=N) buffer Buf{T data[];} g_bufs[];`
    // per distinct buffer-element type this kernel actually uses (typically
    // just one). `N` matches zengpu-vulkan's bindless descriptor set layout
    // (`SSBO_BINDING_F32`/`SSBO_BINDING_U32`/`SSBO_BINDING_I32` in
    // `zengpu-vulkan/src/device/mod.rs`: f32→0, u32→2, i32→3) — every STORAGE
    // buffer is registered into all three typed tables on the host side, so a
    // kernel just picks the table matching each buffer param's declared
    // element type; no bitcast is needed anywhere. Buffer params receive
    // auto-injected u32 indices in the push-constant block.
    let mut ssbo_tables: HashMap<TypeId, SsboTable> = HashMap::new();
    for (_, elem) in &buf_params {
        if ssbo_tables.contains_key(elem) {
            continue;
        }
        let elem_ty = *elem;
        let t_elem = if elem_ty == f32_id {
            t_f32
        } else if elem_ty == u32_id {
            t_u32
        } else if elem_ty == i32_id {
            t_i32
        } else {
            // bf16/f16 typed storage is not yet implemented for SPIR-V: real
            // 16-bit storage needs a device-capability-gated 4th bindless
            // table (Int16 capability + StorageBuffer16BitAccess extension),
            // which depends on device-profile integration `ir::capability`
            // deliberately deferred (see its module doc comment) — not a
            // language-level restriction, so this is a clean diagnostic, not
            // a compiler defect.
            return Err(format!(
                "ZSL: SPIR-V backend does not yet support buffer<{:?}> \
                 (only f32/u32/i32 typed storage exists today)",
                types.get(elem_ty)
            ));
        };
        let binding = ssbo_binding_for(elem_ty, f32_id, u32_id, i32_id);

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

    // ── Push-constant block — buffer indices (u32) then user scalars ─────────
    let n_buf_params = buf_params.len() as u32;
    let pc_var = if !buf_params.is_empty() || !scalar_params.is_empty() {
        let mut pc_members: Vec<Id> = (0..n_buf_params).map(|_| t_u32).collect();
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

    // ── GlobalInvocationID input ─────────────────────────────────────────────
    let t_ptr_in_uvec3 = spv.type_pointer(sc::INPUT, t_uvec3);
    let gid_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(gid_var, deco::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);
    let lid_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(lid_var, deco::BUILT_IN, &[builtin::LOCAL_INVOCATION_ID]);
    let group_var = spv.global_variable(t_ptr_in_uvec3, sc::INPUT);
    spv.decorate(group_var, deco::BUILT_IN, &[builtin::WORKGROUP_ID]);

    // ── Subgroup-query inputs (only if this kernel actually uses one) ────────
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

    // ── Workgroup-shared arrays ─────────────────────────────────────────────
    let t_ptr_wg_f32 = spv.type_pointer(sc::WORKGROUP, t_f32);
    let mut shared = HashMap::new();
    for decl in &entry.shared {
        let len = spv.constant_u32(t_u32, decl.len);
        let array_ty = spv.type_array(t_f32, len);
        let ptr_ty = spv.type_pointer_global(sc::WORKGROUP, array_ty);
        let var = spv.global_variable(ptr_ty, sc::WORKGROUP);
        shared.insert(decl.name.clone(), var);
    }

    // ── Function + entry-point declaration ───────────────────────────────────
    let t_fn = spv.type_function(t_void, &[]);
    let fn_id = spv.fresh_id();
    let mut interface_vars = vec![gid_var, lid_var, group_var];
    if let Some(sg) = &subgroup_vars {
        interface_vars.extend([sg.size, sg.id, sg.local_id]);
    }
    spv.entry_point_glcompute(fn_id, "main", &interface_vars);
    spv.execution_mode_local_size(fn_id, local_size[0], local_size[1], local_size[2]);

    // ── Constants ────────────────────────────────────────────────────────────
    let const_0_u32 = spv.constant_u32(t_u32, 0);

    // ── Build param maps ─────────────────────────────────────────────────────
    let buf_param_map: HashMap<String, BufInfo> = buf_params
        .iter()
        .enumerate()
        .map(|(i, (name, elem))| {
            (
                name.to_string(),
                BufInfo {
                    pc_index: i as u32,
                    elem: *elem,
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

    // ── Emit function, hoist all OpVariable to entry block ───────────────────
    spv.begin_function(t_void, fn_id, t_fn);
    spv.label(); // entry block

    let mut ptr_func_u32 = Id(0);
    let mut ptr_func_i32 = Id(0);
    let mut ptr_func_f32 = Id(0);
    let mut ptr_func_bool = Id(0);
    let mut ptr_func_uvec4 = Id(0);

    let mut locals: HashMap<String, LocalVar> = HashMap::new();
    for (name, ty_id) in &entry.locals {
        let sty = *ty_id;
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

    // Allocate push-constant pointer types before entering the body.
    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_i32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    if pc_var.is_some() {
        // Buffer indices are always u32 — allocate the u32 PC pointer when there are buf params.
        if n_buf_params > 0 {
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
        ssbo_tables,
        n_buf_params,
        glsl_ext,
        buf_params: buf_param_map,
        scalar_params: scalar_param_map,
        pc_var,
        gid_var,
        lid_var,
        group_var,
        subgroup_vars,
        shared,
        t_ptr_wg_f32,
        const_0_u32,
        locals,
        t_ptr_pc_u32,
        t_ptr_pc_i32,
        t_ptr_pc_f32,
    };

    lower_stmts(&mut ctx, &entry.body)?;

    ctx.spv.op_return();
    ctx.spv.end_function();

    Ok(spv.finish())
}

// ── Internal types ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct BufInfo {
    pc_index: u32,
    elem: TypeId,
}

/// A bindless SSBO array typed for one buffer element type, and the
/// storage-buffer-class pointer used to access its elements directly (no
/// bitcast — the pointer's pointee type is the element's real SPIR-V type).
#[derive(Clone, Copy)]
struct SsboTable {
    var: Id,
    t_ptr_elem: Id,
}

#[derive(Clone, Copy)]
struct ScalarInfo {
    pc_index: u32,
    ty: TypeId,
}

/// The three subgroup-query input variables, declared only when a kernel
/// actually calls `subgroup_id`/`subgroup_local_id`/`subgroup_size`.
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
    /// Canonical `TypeId`s for compute values, cached once at the
    /// top of lowering (see `lower_compute_inner`) so every comparison below
    /// is a cheap `TypeId` equality check rather than an arena lookup.
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
    /// One bindless SSBO table per distinct buffer-element type this kernel
    /// actually uses, keyed by the element's canonical `TypeId`. See the
    /// construction comment in `lower_compute_inner` for the binding scheme.
    ssbo_tables: HashMap<TypeId, SsboTable>,
    glsl_ext: Id,
    buf_params: HashMap<String, BufInfo>,
    scalar_params: HashMap<String, ScalarInfo>,
    /// Number of buffer params — scalar PC indices are offset by this.
    n_buf_params: u32,
    pc_var: Option<Id>,
    gid_var: Id,
    lid_var: Id,
    group_var: Id,
    subgroup_vars: Option<SubgroupVars>,
    shared: HashMap<String, Id>,
    t_ptr_wg_f32: Id,
    const_0_u32: Id,
    locals: HashMap<String, LocalVar>,
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

// ── Statement lowering ─────────────────────────────────────────────────────────

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


fn lower_stmt(ctx: &mut LowerCtx<'_>, stmt: &IrStmt) -> Result<(), String> {
    match stmt {
        IrStmt::Let { name, init } => {
            let sty = ctx.locals.get(name).map(|l| l.ty).unwrap_or(ctx.u32_id);
            let val = lower_expr(ctx, init)?;
            let ptr = ctx
                .locals
                .get(name)
                .ok_or_else(|| format!("ZSL: undeclared local `{name}`"))?
                .ptr;
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
                .get(&info.elem)
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
            let c0 = ctx.const_0_u32;
            let ptr_elem = ctx
                .spv
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
            // atomic_add is grammar-restricted to `buffer<f32>` (checked in
            // the parser), so this always uses the f32 table.
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
            // ScopeDevice = 1; MemorySemanticsRelaxed = 0.
            let scope = ctx.spv.constant_u32(ctx.t_u32, 1);
            let semantics = ctx.spv.constant_u32(ctx.t_u32, 0);
            ctx.spv
                .op_atomic_fadd_ext(ctx.t_f32, ptr_elem, scope, semantics, rhs_id);
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

        IrStmt::Barrier => {
            // ScopeWorkgroup = 2; AcquireRelease|WorkgroupMemory = 0x108.
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

            // Loop header: condition check + OpLoopMerge
            ctx.spv.label_with_id(header);
            let t_u32 = ctx.t_u32;
            let t_bool = ctx.t_bool;
            let i_val = ctx.spv.op_load(t_u32, loop_ptr);
            let cond = ctx.spv.op_ult(t_bool, i_val, hi_id);
            ctx.spv.op_loop_merge(merge, cont);
            ctx.spv.op_branch_conditional(cond, body_lbl, merge);

            // Loop body
            ctx.spv.label_with_id(body_lbl);
            lower_stmts(ctx, body)?;
            ctx.spv.op_branch(cont);

            // Continue block: increment counter
            ctx.spv.label_with_id(cont);
            let i_old = ctx.spv.op_load(t_u32, loop_ptr);
            let const_1 = ctx.spv.constant_u32(t_u32, 1);
            let i_new = ctx.spv.op_iadd(t_u32, i_old, const_1);
            ctx.spv.op_store(loop_ptr, i_new);
            ctx.spv.op_branch(header);

            // Merge block
            ctx.spv.label_with_id(merge);
            Ok(())
        }

        IrStmt::Eval(expr) => {
            lower_expr(ctx, expr)?;
            Ok(())
        }
    }
}

// ── Expression lowering ───────────────────────────────────────────────────────

fn lower_expr(ctx: &mut LowerCtx<'_>, expr: &IrExpr) -> Result<Val, String> {
    match expr {
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
            // Buffer index fields come first; scalar fields start at offset n_buf_params.
            let actual_idx = ctx.n_buf_params + info.pc_index;
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
                .get(&info.elem)
                .expect("buffer element type has no SSBO table");
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| "ZSL: no push constant block".to_string())?;
            // Load the buffer's slot index from the push-constant block.
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
            let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
            let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            // g_bufs[buf_idx].data[elem_idx]
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ctx.u32_id);
            let c0 = ctx.const_0_u32;
            let ptr_elem = ctx
                .spv
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

        // Graphics-only expression forms never appear in a compute module.
        IrExpr::Input(_) | IrExpr::FieldAccess { .. } | IrExpr::Construct { .. } => {
            Err("ZSL: vector/graphics expression not supported in compute shaders".to_string())
        }
    }
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
            Ok(Val { id, ty: ctx.bool_id })
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
            Ok(Val { id, ty: ctx.bool_id })
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

fn lower_builtin(ctx: &mut LowerCtx<'_>, func: BuiltinFn, args: &[IrExpr]) -> Result<Val, String> {
    let name = func.name();
    match func {
        BuiltinFn::U32 => {
            if args.len() != 1 {
                return Err("ZSL: u32() takes 1 arg".into());
            }
            let v = lower_expr(ctx, &args[0])?;
            let id = coerce(ctx, v, ctx.u32_id);
            Ok(Val { id, ty: ctx.u32_id })
        }
        BuiltinFn::Dot | BuiltinFn::Extend => {
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
                    // OpIsFinite is not accepted consistently by Vulkan shader
                    // drivers. The equivalent predicate uses core shader ops.
                    let is_nan = ctx.spv.op_is_nan(ctx.t_bool, value.id);
                    let is_inf = ctx.spv.op_is_inf(ctx.t_bool, value.id);
                    let non_finite = ctx.spv.op_logical_or(ctx.t_bool, is_nan, is_inf);
                    ctx.spv.op_logical_not(ctx.t_bool, non_finite)
                }
                _ => unreachable!(),
            };
            Ok(Val { id, ty: ctx.bool_id })
        }
        // Unary GLSL builtins: f32 → f32
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
        // Binary GLSL builtins: (f32, f32) → f32
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
        // clamp(x, lo, hi): (f32, f32, f32) → f32
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
        // mix(a, b, t): (f32, f32, f32) → f32
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

        // Vector-only builtins never appear in a compute module.
        BuiltinFn::Normalize | BuiltinFn::Length => {
            Err(format!("ZSL: {name}() is not supported in compute shaders"))
        }

        // mul_add_f32(a, b, acc): (f32, f32, f32) → f32, single-rounding FMA.
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

        // Subgroup queries: a plain OpLoad from the corresponding builtin
        // input variable. `stmts_use_subgroup_query` (called before this
        // kernel's SPIR-V header is built) guarantees `subgroup_vars` is
        // `Some` whenever one of these three builtins is actually reachable.
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

        // subgroup_any/subgroup_all(predicate: bool) -> bool
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
                ctx.spv
                    .op_group_non_uniform_any(ctx.t_bool, scope, pred.id)
            } else {
                ctx.spv
                    .op_group_non_uniform_all(ctx.t_bool, scope, pred.id)
            };
            Ok(Val {
                id,
                ty: ctx.bool_id,
            })
        }

        // subgroup_ballot(predicate: bool) -> bitset<128> (uvec4 in SPIR-V)
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

        // subgroup_broadcast/shuffle(value: f32, lane: u32) -> f32
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

        // subgroup_shuffle_xor(value: f32, mask: u32) -> f32
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
            let id =
                ctx.spv
                    .op_group_non_uniform_shuffle_xor(ctx.t_f32, scope, value.id, mask);
            Ok(Val { id, ty: ctx.f32_id })
        }

        // subgroup_shuffle_up/down(value: f32, delta: u32) -> f32
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
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
