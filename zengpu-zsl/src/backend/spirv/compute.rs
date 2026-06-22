//! ZSL → SPIR-V lowering for compute shaders.
//!
//! Supports: `#[compute]` entry points with `Buf<f32>`, `BufMut<f32>`, and
//! `u32`/`f32` push-constant params. Body: arithmetic, buffer indexing,
//! `global_id().x/y/z`, `if`/`else`, comparison operators (`< > <= >= == !=`),
//! logical operators (`&&`/`||`), and GLSL.std.450 math builtins (`abs`, `sign`,
//! `sqrt`, `floor`, `ceil`, `fract`, `min`, `max`, `pow`, `clamp`, `mix`).
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
    pub const POW: u32 = 26;
    pub const SQRT: u32 = 31;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const F_MIX: u32 = 46;
}

use std::collections::HashMap;

use proc_macro2::Span;
use syn::{
    BinOp, Block, Expr, ExprBinary, ExprCall, ExprField, ExprForLoop, ExprIndex, ExprLit,
    ExprPath, ExprRange, ExprUnary, Lit, Member, Pat, RangeLimits, Stmt, UnOp,
    spanned::Spanned,
};

use crate::frontend::ast::{ZslEntryPoint, ZslParam};
use crate::frontend::types::ZslType;
use crate::backend::spirv::builder::{Id, SpvBuilder, builtin, deco, sc};

// ── Public entry ─────────────────────────────────────────────────────────────

/// Lower a validated `#[compute]` entry point to SPIR-V words.
pub fn lower_compute(
    entry: &ZslEntryPoint,
    body: &Block,
    local_size: (u32, u32, u32),
) -> Result<Vec<u32>, (Span, String)> {
    let mut spv = SpvBuilder::new();

    spv.capability_shader();
    spv.capability_runtime_descriptor_array();
    let glsl_ext = spv.ext_inst_import_glsl();
    spv.memory_model_logical_glsl450();

    // ── Core scalar types ────────────────────────────────────────────────────
    let t_void = spv.type_void();
    let t_bool = spv.type_bool();
    let t_u32 = spv.type_int(32, false);
    let t_f32 = spv.type_float(32);
    let t_uvec3 = spv.type_vector(t_u32, 3);

    // ── Classify params ──────────────────────────────────────────────────────
    let buf_params: Vec<&ZslParam> = entry
        .params
        .iter()
        .filter(|p| matches!(p.ty, ZslType::Buf(_) | ZslType::BufMut(_)))
        .collect();
    let scalar_params: Vec<&ZslParam> = entry
        .params
        .iter()
        .filter(|p| matches!(p.ty, ZslType::U32 | ZslType::F32))
        .collect();

    // ── SSBO types — bindless model ──────────────────────────────────────────
    // Generates: layout(set=0,binding=0) buffer Buf{float data[];} g_bufs[];
    // Buffer params receive auto-injected u32 indices in the push-constant block.
    let t_ra_f32 = spv.type_runtime_array(t_f32);
    spv.decorate(t_ra_f32, deco::ARRAY_STRIDE, &[4]);
    let t_struct_buf_f32 = spv.type_struct(&[t_ra_f32]);
    spv.decorate(t_struct_buf_f32, deco::BLOCK, &[]);
    spv.member_decorate(t_struct_buf_f32, 0, deco::OFFSET, &[0]);
    let t_ra_struct = spv.type_runtime_array(t_struct_buf_f32);
    let t_ptr_ra_struct = spv.type_pointer(sc::STORAGE_BUFFER, t_ra_struct);
    let t_ptr_ssbo_f32 = spv.type_pointer(sc::STORAGE_BUFFER, t_f32);

    let g_bufs_var = spv.global_variable(t_ptr_ra_struct, sc::STORAGE_BUFFER);
    spv.decorate(g_bufs_var, deco::DESCRIPTOR_SET, &[0]);
    spv.decorate(g_bufs_var, deco::BINDING, &[0]);

    // ── Push-constant block — buffer indices (u32) then user scalars ─────────
    let n_buf_params = buf_params.len() as u32;
    let pc_var = if !buf_params.is_empty() || !scalar_params.is_empty() {
        let mut pc_members: Vec<Id> = (0..n_buf_params).map(|_| t_u32).collect();
        pc_members.extend(
            scalar_params
                .iter()
                .map(|p| if p.ty == ZslType::U32 { t_u32 } else { t_f32 }),
        );
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

    // ── Function + entry-point declaration ───────────────────────────────────
    let t_fn = spv.type_function(t_void, &[]);
    let fn_id = spv.fresh_id();
    spv.entry_point_glcompute(fn_id, "main", &[gid_var]);
    spv.execution_mode_local_size(fn_id, local_size.0, local_size.1, local_size.2);

    // ── Constants ────────────────────────────────────────────────────────────
    let const_0_u32 = spv.constant_u32(t_u32, 0);

    // ── Build param maps ─────────────────────────────────────────────────────
    let buf_param_map: HashMap<String, BufInfo> = buf_params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                p.ident.to_string(),
                BufInfo {
                    pc_index: i as u32,
                    writable: matches!(p.ty, ZslType::BufMut(_)),
                },
            )
        })
        .collect();

    let scalar_param_map: HashMap<String, ScalarInfo> = scalar_params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let ty = if p.ty == ZslType::U32 { t_u32 } else { t_f32 };
            (
                p.ident.to_string(),
                ScalarInfo {
                    pc_index: i as u32,
                    ty,
                },
            )
        })
        .collect();

    // ── Emit function, hoist all OpVariable to entry block ───────────────────
    spv.begin_function(t_void, fn_id, t_fn);
    spv.label(); // entry block

    // Pre-scan locals and declare all OpVariables at entry-block top (SPIR-V
    // requires all OpVariable in a function to be in the first block).
    let local_decls = collect_all_locals(body)?;

    let mut ptr_func_u32 = Id(0);
    let mut ptr_func_f32 = Id(0);
    let mut ptr_func_bool = Id(0);

    let mut locals: HashMap<String, LocalVar> = HashMap::new();
    for (name, sty) in &local_decls {
        let (ptr_ty_slot, spv_elem) = match sty {
            ScalarTy::U32 => (&mut ptr_func_u32, t_u32),
            ScalarTy::F32 => (&mut ptr_func_f32, t_f32),
            ScalarTy::Bool => (&mut ptr_func_bool, t_bool),
        };
        if *ptr_ty_slot == Id(0) {
            *ptr_ty_slot = spv.type_pointer(sc::FUNCTION, spv_elem);
        }
        let ptr = spv.op_variable(*ptr_ty_slot, sc::FUNCTION);
        locals.insert(
            name.clone(),
            LocalVar {
                ptr,
                ty: *sty,
                ptr_ty: *ptr_ty_slot,
            },
        );
    }

    // Allocate push-constant pointer types before entering the body.
    let mut t_ptr_pc_u32 = Id(0);
    let mut t_ptr_pc_f32 = Id(0);
    if pc_var.is_some() {
        // Buffer indices are always u32 — allocate the u32 PC pointer when there are buf params.
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
                _ => {}
            }
        }
    }

    let mut ctx = LowerCtx {
        spv: &mut spv,
        t_bool,
        t_u32,
        t_f32,
        t_uvec3,
        g_bufs_var,
        n_buf_params,
        t_ptr_ssbo_f32,
        glsl_ext,
        buf_params: buf_param_map,
        scalar_params: scalar_param_map,
        pc_var,
        gid_var,
        const_0_u32,
        locals,
        t_ptr_pc_u32,
        t_ptr_pc_f32,
    };

    lower_block(&mut ctx, body)?;

    ctx.spv.op_return();
    ctx.spv.end_function();

    Ok(spv.finish())
}

// ── Internal types ────────────────────────────────────────────────────────────

struct BufInfo {
    pc_index: u32,
    writable: bool,
}

#[derive(Clone, Copy)]
struct ScalarInfo {
    pc_index: u32,
    ty: Id,
}

#[derive(Clone, Copy)]
struct Val {
    id: Id,
    ty: ScalarTy,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarTy {
    U32,
    F32,
    Bool,
}

struct LocalVar {
    ptr: Id,
    ty: ScalarTy,
    #[allow(dead_code)]
    ptr_ty: Id,
}

struct LowerCtx<'a> {
    spv: &'a mut SpvBuilder,
    t_bool: Id,
    t_u32: Id,
    t_f32: Id,
    t_uvec3: Id,
    t_ptr_ssbo_f32: Id,
    /// Bindless SSBO array: `layout(set=0,binding=0) buffer Buf{float data[];} g_bufs[];`
    g_bufs_var: Id,
    glsl_ext: Id,
    buf_params: HashMap<String, BufInfo>,
    scalar_params: HashMap<String, ScalarInfo>,
    /// Number of buffer params — scalar PC indices are offset by this.
    n_buf_params: u32,
    pc_var: Option<Id>,
    gid_var: Id,
    const_0_u32: Id,
    locals: HashMap<String, LocalVar>,
    t_ptr_pc_u32: Id,
    t_ptr_pc_f32: Id,
}

impl LowerCtx<'_> {
    fn spv_ty(&self, sty: ScalarTy) -> Id {
        match sty {
            ScalarTy::U32 => self.t_u32,
            ScalarTy::F32 => self.t_f32,
            ScalarTy::Bool => self.t_bool,
        }
    }
}

// ── Local variable pre-scan ───────────────────────────────────────────────────

fn collect_all_locals(block: &Block) -> Result<Vec<(String, ScalarTy)>, (Span, String)> {
    let mut out = Vec::new();
    collect_block_locals(block, &mut out)?;
    Ok(out)
}

fn collect_block_locals(
    block: &Block,
    out: &mut Vec<(String, ScalarTy)>,
) -> Result<(), (Span, String)> {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                let ident = local_ident_str(local)?;
                let sty = match &local.pat {
                    syn::Pat::Type(pt) => zsl_scalar_ty(&pt.ty)?,
                    _ => ScalarTy::U32,
                };
                out.push((ident, sty));
            }
            Stmt::Expr(Expr::If(expr_if), _) => {
                collect_block_locals(&expr_if.then_branch, out)?;
                if let Some((_, else_expr)) = &expr_if.else_branch {
                    if let Expr::Block(eb) = else_expr.as_ref() {
                        collect_block_locals(&eb.block, out)?;
                    }
                }
            }
            Stmt::Expr(Expr::ForLoop(expr_for), _) => {
                // The loop variable itself is a u32 local (auto-declared by the for-loop).
                let loop_var = match expr_for.pat.as_ref() {
                    Pat::Ident(p) => p.ident.to_string(),
                    _ => {
                        return Err((
                            Span::call_site(),
                            "ZSL: for-loop pattern must be a simple identifier".into(),
                        ))
                    }
                };
                out.push((loop_var, ScalarTy::U32));
                collect_block_locals(&expr_for.body, out)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn local_ident_str(local: &syn::Local) -> Result<String, (Span, String)> {
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

// ── Block / statement lowering ────────────────────────────────────────────────

fn lower_block(ctx: &mut LowerCtx<'_>, block: &Block) -> Result<(), (Span, String)> {
    for stmt in &block.stmts {
        lower_stmt(ctx, stmt)?;
    }
    Ok(())
}

fn lower_stmt(ctx: &mut LowerCtx<'_>, stmt: &Stmt) -> Result<(), (Span, String)> {
    match stmt {
        // `let name: Type = init;` — variable already declared as OpVariable;
        // emit OpStore for the initializer.
        Stmt::Local(local) => {
            let ident = local_ident_str(local)?;
            if let Some(init) = &local.init {
                let sty = ctx
                    .locals
                    .get(&ident)
                    .map(|l| l.ty)
                    .unwrap_or(ScalarTy::U32);
                let val = lower_expr(ctx, &init.expr)?;
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
                let val_id = coerce(ctx, val, sty);
                ctx.spv.op_store(ptr, val_id);
            }
            Ok(())
        }

        Stmt::Expr(expr, _) => lower_expr_stmt(ctx, expr),

        other => Err((
            Span::call_site(),
            format!("ZSL: unsupported statement `{}`", quote::quote!(#other)),
        )),
    }
}

fn lower_expr_stmt(ctx: &mut LowerCtx<'_>, expr: &Expr) -> Result<(), (Span, String)> {
    match expr {
        // `buf[i] = value;` OR `local_var = value;`
        Expr::Assign(assign) => {
            match &*assign.left {
                Expr::Index(ExprIndex {
                    expr: base, index, ..
                }) => {
                    // Buffer element assignment: buf[i] = rhs
                    let name = expr_path_ident(base)?;
                    let (buf_pc_index, writable, g_bufs) = {
                        let info = ctx.buf_params.get(&name).ok_or_else(|| {
                            (base.span(), format!("ZSL: `{name}` is not a buffer param"))
                        })?;
                        (info.pc_index, info.writable, ctx.g_bufs_var)
                    };
                    if !writable {
                        return Err((
                            base.span(),
                            format!("`{name}` is `Buf<T>` (read-only); use `BufMut<T>` to write"),
                        ));
                    }
                    let pc_var = ctx
                        .pc_var
                        .ok_or_else(|| (base.span(), "ZSL: no push constant block".into()))?;
                    let pc_field = ctx.spv.constant_u32(ctx.t_u32, buf_pc_index);
                    let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
                    let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
                    let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
                    let idx_val = lower_expr(ctx, index)?;
                    let idx_id = coerce(ctx, idx_val, ScalarTy::U32);
                    let ptr_ssbo_f32 = ctx.t_ptr_ssbo_f32;
                    let c0 = ctx.const_0_u32;
                    let ptr_elem =
                        ctx.spv.op_access_chain(ptr_ssbo_f32, g_bufs, &[buf_idx, c0, idx_id]);
                    let rhs = lower_expr(ctx, &assign.right)?;
                    let rhs_id = coerce(ctx, rhs, ScalarTy::F32);
                    ctx.spv.op_store(ptr_elem, rhs_id);
                    Ok(())
                }
                Expr::Path(ExprPath { path, .. }) => {
                    // Local variable assignment: name = rhs
                    let ident = path
                        .get_ident()
                        .ok_or_else(|| {
                            (path.span(), "ZSL: assignment target must be a simple name".into())
                        })?
                        .to_string();
                    let (ptr, ty) = ctx
                        .locals
                        .get(&ident)
                        .map(|l| (l.ptr, l.ty))
                        .ok_or_else(|| {
                            (
                                path.span(),
                                format!("ZSL: `{ident}` is not a declared local variable"),
                            )
                        })?;
                    let rhs = lower_expr(ctx, &assign.right)?;
                    let rhs_id = coerce(ctx, rhs, ty);
                    ctx.spv.op_store(ptr, rhs_id);
                    Ok(())
                }
                other => Err((
                    other.span(),
                    "ZSL: assignment target must be `buf[i]` or a local variable name".into(),
                )),
            }
        }

        Expr::If(expr_if) => {
            let cond = lower_expr(ctx, &expr_if.cond)?;
            let cond_id = coerce(ctx, cond, ScalarTy::Bool);
            let true_label = ctx.spv.fresh_id();
            let merge_label = ctx.spv.fresh_id();

            if let Some((_, else_expr)) = &expr_if.else_branch {
                let false_label = ctx.spv.fresh_id();
                ctx.spv.op_selection_merge(merge_label);
                ctx.spv
                    .op_branch_conditional(cond_id, true_label, false_label);

                ctx.spv.label_with_id(true_label);
                lower_block(ctx, &expr_if.then_branch)?;
                ctx.spv.op_branch(merge_label);

                ctx.spv.label_with_id(false_label);
                match else_expr.as_ref() {
                    Expr::Block(eb) => lower_block(ctx, &eb.block)?,
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
                    .op_branch_conditional(cond_id, true_label, merge_label);

                ctx.spv.label_with_id(true_label);
                lower_block(ctx, &expr_if.then_branch)?;
                ctx.spv.op_branch(merge_label);
            }

            ctx.spv.label_with_id(merge_label);
            Ok(())
        }

        // `for i in lo..hi { ... }`
        Expr::ForLoop(ExprForLoop { pat, expr, body, .. }) => {
            let loop_var = match pat.as_ref() {
                Pat::Ident(p) => p.ident.to_string(),
                other => {
                    return Err((
                        other.span(),
                        "ZSL: for-loop pattern must be a simple identifier".into(),
                    ));
                }
            };
            let Expr::Range(ExprRange {
                start: Some(lo_expr),
                limits: RangeLimits::HalfOpen(_),
                end: Some(hi_expr),
                ..
            }) = expr.as_ref()
            else {
                return Err((
                    expr.span(),
                    "ZSL: for-loop range must be `lo..hi` (exclusive, both bounds required)".into(),
                ));
            };

            let lo_val = lower_expr(ctx, lo_expr)?;
            let lo_id = coerce(ctx, lo_val, ScalarTy::U32);
            let hi_val = lower_expr(ctx, hi_expr)?;
            let hi_id = coerce(ctx, hi_val, ScalarTy::U32);

            let loop_ptr = ctx
                .locals
                .get(&loop_var)
                .map(|l| l.ptr)
                .ok_or_else(|| {
                    (
                        pat.span(),
                        format!("ZSL: for-loop variable `{loop_var}` not declared as a local"),
                    )
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
            lower_block(ctx, body)?;
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

        _ => {
            lower_expr(ctx, expr)?;
            Ok(())
        }
    }
}

// ── Expression lowering ───────────────────────────────────────────────────────

fn lower_expr(ctx: &mut LowerCtx<'_>, expr: &Expr) -> Result<Val, (Span, String)> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit), ..
        }) => {
            let v: u32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of u32 range", lit)))?;
            let id = ctx.spv.constant_u32(ctx.t_u32, v);
            Ok(Val {
                id,
                ty: ScalarTy::U32,
            })
        }

        Expr::Lit(ExprLit {
            lit: Lit::Float(lit),
            ..
        }) => {
            let v: f32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of f32 range", lit)))?;
            let id = ctx.spv.constant_f32(ctx.t_f32, v);
            Ok(Val {
                id,
                ty: ScalarTy::F32,
            })
        }

        Expr::Path(ExprPath { path, .. }) => {
            let ident = path
                .get_ident()
                .ok_or_else(|| (path.span(), "ZSL: expected simple identifier".into()))?
                .to_string();

            if let Some(local) = ctx.locals.get(&ident) {
                let (ty, ptr) = (local.ty, local.ptr);
                let spv_ty = ctx.spv_ty(ty);
                let id = ctx.spv.op_load(spv_ty, ptr);
                return Ok(Val { id, ty });
            }

            if let Some(info) = ctx.scalar_params.get(&ident).cloned() {
                let pc_var = ctx
                    .pc_var
                    .ok_or_else(|| (path.span(), "ZSL: no push constant block".into()))?;
                let is_u32 = info.ty == ctx.t_u32;
                let pc_ptr_ty = if is_u32 {
                    ctx.t_ptr_pc_u32
                } else {
                    ctx.t_ptr_pc_f32
                };
                if pc_ptr_ty == Id(0) {
                    return Err((
                        path.span(),
                        "ZSL: push-constant pointer type not allocated".into(),
                    ));
                }
                // Buffer index fields come first; scalar fields start at offset n_buf_params.
                let actual_idx = ctx.n_buf_params + info.pc_index;
                let pc_idx = ctx.spv.constant_u32(ctx.t_u32, actual_idx);
                let chain = ctx.spv.op_access_chain(pc_ptr_ty, pc_var, &[pc_idx]);
                let id = ctx.spv.op_load(info.ty, chain);
                let ty = if is_u32 { ScalarTy::U32 } else { ScalarTy::F32 };
                return Ok(Val { id, ty });
            }

            Err((path.span(), format!("ZSL: unknown identifier `{ident}`")))
        }

        Expr::Index(ExprIndex {
            expr: base, index, ..
        }) => {
            let name = expr_path_ident(base)?;
            let buf_info = ctx
                .buf_params
                .get(&name)
                .ok_or_else(|| (base.span(), format!("ZSL: `{name}` is not a buffer param")))?;
            let (buf_pc_index, g_bufs) = (buf_info.pc_index, ctx.g_bufs_var);
            let pc_var = ctx
                .pc_var
                .ok_or_else(|| (base.span(), "ZSL: no push constant block".into()))?;
            // Load the buffer's slot index from the push-constant block.
            let pc_field = ctx.spv.constant_u32(ctx.t_u32, buf_pc_index);
            let t_ptr_pc_u32 = ctx.t_ptr_pc_u32;
            let pc_chain = ctx.spv.op_access_chain(t_ptr_pc_u32, pc_var, &[pc_field]);
            let buf_idx = ctx.spv.op_load(ctx.t_u32, pc_chain);
            // g_bufs[buf_idx].data[elem_idx]
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ScalarTy::U32);
            let ptr_ssbo_f32 = ctx.t_ptr_ssbo_f32;
            let c0 = ctx.const_0_u32;
            let ptr_elem =
                ctx.spv.op_access_chain(ptr_ssbo_f32, g_bufs, &[buf_idx, c0, idx_id]);
            let t_f32 = ctx.t_f32;
            let id = ctx.spv.op_load(t_f32, ptr_elem);
            Ok(Val {
                id,
                ty: ScalarTy::F32,
            })
        }

        // global_id().x / global_id().y / global_id().z
        Expr::Field(ExprField {
            base,
            member: Member::Named(field),
            ..
        }) => {
            if let Expr::Call(ExprCall { func, args, .. }) = base.as_ref() {
                if let Expr::Path(p) = func.as_ref() {
                    if p.path.is_ident("global_id") && args.is_empty() {
                        let component = match field.to_string().as_str() {
                            "x" => 0u32,
                            "y" => 1,
                            "z" => 2,
                            other => {
                                return Err((
                                    field.span(),
                                    format!(
                                        "ZSL: `global_id()` has no field `.{other}`; use .x, .y, or .z"
                                    ),
                                ));
                            }
                        };
                        let gid_var = ctx.gid_var;
                        let t_uvec3 = ctx.t_uvec3;
                        let gid = ctx.spv.op_load(t_uvec3, gid_var);
                        let t_u32 = ctx.t_u32;
                        let val = ctx.spv.op_composite_extract(t_u32, gid, &[component]);
                        return Ok(Val {
                            id: val,
                            ty: ScalarTy::U32,
                        });
                    }
                }
            }
            Err((base.span(), "ZSL: unsupported field access".into()))
        }

        Expr::Call(ExprCall { func, args, .. }) => {
            let Expr::Path(p) = &**func else {
                return Err((func.span(), "ZSL: expected a built-in function name".into()));
            };
            let Some(ident) = p.path.get_ident() else {
                return Err((func.span(), "ZSL: expected a simple function name".into()));
            };
            let name = ident.to_string();
            match name.as_str() {
                "global_id" => Err((
                    func.span(),
                    "ZSL: use `global_id().x`, `.y`, or `.z`".into(),
                )),
                // Unary GLSL builtins: f32 → f32
                "abs" | "sign" | "sqrt" | "floor" | "ceil" | "fract" => {
                    if args.len() != 1 {
                        return Err((func.span(), format!("ZSL: {name}() takes 1 arg")));
                    }
                    let v = lower_expr(ctx, &args[0])?;
                    if v.ty != ScalarTy::F32 {
                        return Err((args[0].span(), format!("ZSL: {name}() requires f32")));
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
                    let t_f32 = ctx.t_f32;
                    let glsl = ctx.glsl_ext;
                    let id = ctx.spv.op_ext_inst(t_f32, glsl, opcode, &[v.id]);
                    Ok(Val {
                        id,
                        ty: ScalarTy::F32,
                    })
                }
                // Binary GLSL builtins: (f32, f32) → f32
                "min" | "max" | "pow" => {
                    if args.len() != 2 {
                        return Err((func.span(), format!("ZSL: {name}(a, b) takes 2 args")));
                    }
                    let a = lower_expr(ctx, &args[0])?;
                    let b = lower_expr(ctx, &args[1])?;
                    if a.ty != ScalarTy::F32 {
                        return Err((args[0].span(), format!("ZSL: {name}() requires f32")));
                    }
                    if b.ty != ScalarTy::F32 {
                        return Err((args[1].span(), format!("ZSL: {name}() requires f32")));
                    }
                    let opcode = match name.as_str() {
                        "min" => glsl_op::F_MIN,
                        "max" => glsl_op::F_MAX,
                        "pow" => glsl_op::POW,
                        _ => unreachable!(),
                    };
                    let t_f32 = ctx.t_f32;
                    let glsl = ctx.glsl_ext;
                    let id = ctx.spv.op_ext_inst(t_f32, glsl, opcode, &[a.id, b.id]);
                    Ok(Val {
                        id,
                        ty: ScalarTy::F32,
                    })
                }
                // clamp(x, lo, hi): (f32, f32, f32) → f32
                "clamp" => {
                    if args.len() != 3 {
                        return Err((func.span(), "ZSL: clamp(x, lo, hi) takes 3 args".into()));
                    }
                    let x = lower_expr(ctx, &args[0])?;
                    let lo = lower_expr(ctx, &args[1])?;
                    let hi = lower_expr(ctx, &args[2])?;
                    if x.ty != ScalarTy::F32 || lo.ty != ScalarTy::F32 || hi.ty != ScalarTy::F32 {
                        return Err((func.span(), "ZSL: clamp() requires f32 args".into()));
                    }
                    let t_f32 = ctx.t_f32;
                    let glsl = ctx.glsl_ext;
                    let id =
                        ctx.spv
                            .op_ext_inst(t_f32, glsl, glsl_op::F_CLAMP, &[x.id, lo.id, hi.id]);
                    Ok(Val {
                        id,
                        ty: ScalarTy::F32,
                    })
                }
                // mix(a, b, t): (f32, f32, f32) → f32
                "mix" => {
                    if args.len() != 3 {
                        return Err((func.span(), "ZSL: mix(a, b, t) takes 3 args".into()));
                    }
                    let a = lower_expr(ctx, &args[0])?;
                    let b = lower_expr(ctx, &args[1])?;
                    let t = lower_expr(ctx, &args[2])?;
                    if a.ty != ScalarTy::F32 || b.ty != ScalarTy::F32 || t.ty != ScalarTy::F32 {
                        return Err((func.span(), "ZSL: mix() requires f32 args".into()));
                    }
                    let t_f32 = ctx.t_f32;
                    let glsl = ctx.glsl_ext;
                    let id = ctx
                        .spv
                        .op_ext_inst(t_f32, glsl, glsl_op::F_MIX, &[a.id, b.id, t.id]);
                    Ok(Val {
                        id,
                        ty: ScalarTy::F32,
                    })
                }
                other => Err((
                    ident.span(),
                    format!(
                        "ZSL: unknown function `{other}`; built-ins: \
                         abs, sign, sqrt, floor, ceil, fract, min, max, pow, clamp, mix; \
                         or use global_id().x/y/z"
                    ),
                )),
            }
        }

        Expr::Unary(ExprUnary { op, expr, .. }) => match op {
            UnOp::Neg(_) => {
                let val = lower_expr(ctx, expr)?;
                let ty_id = ctx.spv_ty(val.ty);
                let id = if val.ty == ScalarTy::F32 {
                    ctx.spv.op_fnegate(ty_id, val.id)
                } else {
                    ctx.spv.op_snegate(ty_id, val.id)
                };
                Ok(Val { id, ty: val.ty })
            }
            other => Err((other.span(), "ZSL: only unary `-` is supported".into())),
        },

        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let lhs = lower_expr(ctx, left)?;
            let rhs = lower_expr(ctx, right)?;

            match op {
                BinOp::Add(_) => {
                    binary_arith(ctx, lhs, rhs, SpvBuilder::op_fadd, SpvBuilder::op_iadd)
                }
                BinOp::Sub(_) => {
                    binary_arith(ctx, lhs, rhs, SpvBuilder::op_fsub, SpvBuilder::op_isub)
                }
                BinOp::Mul(_) => {
                    binary_arith(ctx, lhs, rhs, SpvBuilder::op_fmul, SpvBuilder::op_imul)
                }
                BinOp::Div(_) => {
                    binary_arith(ctx, lhs, rhs, SpvBuilder::op_fdiv, SpvBuilder::op_udiv)
                }
                BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
                | BinOp::Eq(_)
                | BinOp::Ne(_) => {
                    let (lhs, rhs) = unify(ctx, lhs, rhs);
                    let bool_ty = ctx.t_bool;
                    let id = match (op, lhs.ty) {
                        (BinOp::Lt(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_lt(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Le(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_le(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Gt(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_gt(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Ge(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_ge(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Eq(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_eq(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Ne(_), ScalarTy::F32) => {
                            ctx.spv.op_ford_ne(bool_ty, lhs.id, rhs.id)
                        }
                        (BinOp::Lt(_), ScalarTy::U32) => ctx.spv.op_ult(bool_ty, lhs.id, rhs.id),
                        (BinOp::Le(_), ScalarTy::U32) => ctx.spv.op_ule(bool_ty, lhs.id, rhs.id),
                        (BinOp::Gt(_), ScalarTy::U32) => ctx.spv.op_ugt(bool_ty, lhs.id, rhs.id),
                        (BinOp::Ge(_), ScalarTy::U32) => ctx.spv.op_uge(bool_ty, lhs.id, rhs.id),
                        (BinOp::Eq(_), ScalarTy::U32) => ctx.spv.op_iequal(bool_ty, lhs.id, rhs.id),
                        (BinOp::Ne(_), ScalarTy::U32) => {
                            ctx.spv.op_inot_equal(bool_ty, lhs.id, rhs.id)
                        }
                        _ => {
                            return Err((
                                op.span(),
                                "ZSL: comparisons not supported on bool".into(),
                            ));
                        }
                    };
                    Ok(Val {
                        id,
                        ty: ScalarTy::Bool,
                    })
                }
                BinOp::And(_) | BinOp::Or(_) => {
                    if lhs.ty != ScalarTy::Bool || rhs.ty != ScalarTy::Bool {
                        return Err((op.span(), "ZSL: `&&`/`||` require bool operands".into()));
                    }
                    let bool_ty = ctx.t_bool;
                    let id = match op {
                        BinOp::And(_) => ctx.spv.op_logical_and(bool_ty, lhs.id, rhs.id),
                        BinOp::Or(_) => ctx.spv.op_logical_or(bool_ty, lhs.id, rhs.id),
                        _ => unreachable!(),
                    };
                    Ok(Val {
                        id,
                        ty: ScalarTy::Bool,
                    })
                }
                other => Err((
                    other.span(),
                    "ZSL: unsupported op; use +, -, *, /, <, >, <=, >=, ==, !=, &&, ||".into(),
                )),
            }
        }

        Expr::Paren(ep) => lower_expr(ctx, &ep.expr),

        other => Err((
            other.span(),
            format!("ZSL: unsupported expression `{}`", quote::quote!(#other)),
        )),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn zsl_scalar_ty(ty: &syn::Type) -> Result<ScalarTy, (Span, String)> {
    let syn::Type::Path(tp) = ty else {
        return Err((
            Span::call_site(),
            "ZSL: expected scalar type (u32, f32, bool)".into(),
        ));
    };
    match tp
        .path
        .get_ident()
        .ok_or_else(|| (tp.path.span(), "ZSL: expected simple type name".into()))?
        .to_string()
        .as_str()
    {
        "u32" => Ok(ScalarTy::U32),
        "f32" => Ok(ScalarTy::F32),
        "bool" => Ok(ScalarTy::Bool),
        other => Err((
            tp.path.span(),
            format!("ZSL local var `{other}` not supported; use u32, f32, or bool"),
        )),
    }
}

fn expr_path_ident(expr: &Expr) -> Result<String, (Span, String)> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return Err((expr.span(), "ZSL: expected simple identifier".into()));
    };
    Ok(path
        .get_ident()
        .ok_or_else(|| (path.span(), "ZSL: expected simple identifier".into()))?
        .to_string())
}

fn coerce(ctx: &mut LowerCtx<'_>, val: Val, target: ScalarTy) -> Id {
    if val.ty == target {
        return val.id;
    }
    match (val.ty, target) {
        (ScalarTy::U32, ScalarTy::F32) => {
            let f32_ty = ctx.t_f32;
            ctx.spv.op_convert_u_to_f(f32_ty, val.id)
        }
        (ScalarTy::F32, ScalarTy::U32) => {
            let u32_ty = ctx.t_u32;
            ctx.spv.op_convert_f_to_u(u32_ty, val.id)
        }
        _ => val.id,
    }
}

fn unify(ctx: &mut LowerCtx<'_>, lhs: Val, rhs: Val) -> (Val, Val) {
    if lhs.ty == rhs.ty {
        return (lhs, rhs);
    }
    let target = if lhs.ty == ScalarTy::F32 || rhs.ty == ScalarTy::F32 {
        ScalarTy::F32
    } else {
        ScalarTy::U32
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
) -> Result<Val, (Span, String)> {
    let (lhs, rhs) = unify(ctx, lhs, rhs);
    let ty = ctx.spv_ty(lhs.ty);
    let id = if lhs.ty == ScalarTy::F32 {
        float_op(ctx.spv, ty, lhs.id, rhs.id)
    } else {
        int_op(ctx.spv, ty, lhs.id, rhs.id)
    };
    Ok(Val { id, ty: lhs.ty })
}
