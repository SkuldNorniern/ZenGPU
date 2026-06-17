//! ZSL → SPIR-V lowering for compute shaders.
//!
//! Scope: `#[compute]` entry points with `Buf<f32>`, `BufMut<f32>`, and
//! `u32`/`f32` push-constant params. Body supports: arithmetic, buffer
//! indexing, `gl_global_id_x()`, and `if`-without-else.
//!
//! Two-pass approach: all function-scope `OpVariable` declarations are hoisted
//! to the entry block (SPIR-V requires this), then the statements are lowered.

use std::collections::HashMap;

use proc_macro2::Span;
use syn::{
    BinOp, Block, Expr, ExprBinary, ExprCall, ExprField, ExprIndex, ExprLit, ExprPath, ExprUnary,
    Lit, Member, Stmt, UnOp, spanned::Spanned,
};

use crate::ast::{ZslEntryPoint, ZslParam};
use crate::spirv::{Id, SpvBuilder, builtin, deco, sc};
use crate::types::ZslType;

// ── Public entry ─────────────────────────────────────────────────────────────

/// Lower a validated `#[compute]` entry point to SPIR-V words.
pub fn lower_compute(
    entry: &ZslEntryPoint,
    body: &Block,
    local_size: (u32, u32, u32),
) -> Result<Vec<u32>, (Span, String)> {
    let mut spv = SpvBuilder::new();

    spv.capability_shader();
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

    // ── SSBO types (f32 element only in step 4) ──────────────────────────────
    let t_ra_f32 = spv.type_runtime_array(t_f32);
    spv.decorate(t_ra_f32, deco::ARRAY_STRIDE, &[4]);
    let t_struct_buf_f32 = spv.type_struct(&[t_ra_f32]);
    spv.decorate(t_struct_buf_f32, deco::BLOCK, &[]);
    spv.member_decorate(t_struct_buf_f32, 0, deco::OFFSET, &[0]);
    let t_ptr_ssbo_struct_f32 = spv.type_pointer(sc::STORAGE_BUFFER, t_struct_buf_f32);
    let t_ptr_ssbo_f32 = spv.type_pointer(sc::STORAGE_BUFFER, t_f32);

    // ── SSBO global variables ────────────────────────────────────────────────
    let mut buf_vars: Vec<Id> = Vec::new();
    for (binding, param) in buf_params.iter().enumerate() {
        let var = spv.global_variable(t_ptr_ssbo_struct_f32, sc::STORAGE_BUFFER);
        spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
        spv.decorate(var, deco::BINDING, &[binding as u32]);
        if matches!(param.ty, ZslType::Buf(_)) {
            spv.decorate(var, deco::NON_WRITABLE, &[]);
        }
        buf_vars.push(var);
    }

    // ── Push-constant block ──────────────────────────────────────────────────
    let pc_var = if !scalar_params.is_empty() {
        let pc_members: Vec<Id> = scalar_params
            .iter()
            .map(|p| if p.ty == ZslType::U32 { t_u32 } else { t_f32 })
            .collect();
        let t_pc_struct = spv.type_struct(&pc_members);
        spv.decorate(t_pc_struct, deco::BLOCK, &[]);
        for (i, _) in scalar_params.iter().enumerate() {
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
    spv.entry_point_glcompute(fn_id, &entry.ident.to_string(), &[gid_var]);
    spv.execution_mode_local_size(fn_id, local_size.0, local_size.1, local_size.2);

    // ── Constants ────────────────────────────────────────────────────────────
    let const_0_u32 = spv.constant_u32(t_u32, 0);

    // ── Build param maps ─────────────────────────────────────────────────────
    let buf_param_map: HashMap<String, BufInfo> = buf_params
        .iter()
        .zip(buf_vars.iter())
        .map(|(p, &var)| {
            (
                p.ident.to_string(),
                BufInfo {
                    var,
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
        t_ptr_ssbo_f32,
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
    var: Id,
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
    buf_params: HashMap<String, BufInfo>,
    scalar_params: HashMap<String, ScalarInfo>,
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
        // `buf[i] = value;`
        Expr::Assign(assign) => {
            let Expr::Index(ExprIndex {
                expr: base, index, ..
            }) = &*assign.left
            else {
                return Err((
                    assign.left.span(),
                    "ZSL: assignment target must be `buf[i]`".into(),
                ));
            };
            let name = expr_path_ident(base)?;
            let (var, writable) = {
                let info = ctx
                    .buf_params
                    .get(&name)
                    .ok_or_else(|| (base.span(), format!("ZSL: `{name}` is not a buffer param")))?;
                (info.var, info.writable)
            };
            if !writable {
                return Err((
                    base.span(),
                    format!("`{name}` is `Buf<T>` (read-only); use `BufMut<T>` to write"),
                ));
            }
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ScalarTy::U32);
            let ptr_ssbo_f32 = ctx.t_ptr_ssbo_f32;
            let c0 = ctx.const_0_u32;
            let ptr_elem = ctx.spv.op_access_chain(ptr_ssbo_f32, var, &[c0, idx_id]);
            let rhs = lower_expr(ctx, &assign.right)?;
            let rhs_id = coerce(ctx, rhs, ScalarTy::F32);
            ctx.spv.op_store(ptr_elem, rhs_id);
            Ok(())
        }

        // `if cond { block }` without else
        Expr::If(expr_if) => {
            if expr_if.else_branch.is_some() {
                return Err((
                    expr_if.if_token.span,
                    "ZSL: `if-else` not yet supported (step 4 handles `if` only)".into(),
                ));
            }
            let cond = lower_expr(ctx, &expr_if.cond)?;
            let cond_id = coerce(ctx, cond, ScalarTy::Bool);

            // Pre-allocate both target IDs before emitting branch instructions.
            let true_label = ctx.spv.fresh_id();
            let merge_label = ctx.spv.fresh_id();

            ctx.spv.op_selection_merge(merge_label);
            ctx.spv
                .op_branch_conditional(cond_id, true_label, merge_label);

            ctx.spv.label_with_id(true_label);
            lower_block(ctx, &expr_if.then_branch)?;
            ctx.spv.op_branch(merge_label);

            ctx.spv.label_with_id(merge_label);
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
                let pc_idx = ctx.spv.constant_u32(ctx.t_u32, info.pc_index);
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
            let var = ctx
                .buf_params
                .get(&name)
                .ok_or_else(|| (base.span(), format!("ZSL: `{name}` is not a buffer param")))?
                .var;
            let idx_val = lower_expr(ctx, index)?;
            let idx_id = coerce(ctx, idx_val, ScalarTy::U32);
            let ptr_ssbo_f32 = ctx.t_ptr_ssbo_f32;
            let c0 = ctx.const_0_u32;
            let ptr_elem = ctx.spv.op_access_chain(ptr_ssbo_f32, var, &[c0, idx_id]);
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

        Expr::Call(ExprCall { func, .. }) => Err((
            func.span(),
            "ZSL: unknown function call; use `global_id().x/y/z` for compute built-ins".into(),
        )),

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
                BinOp::Lt(_) => {
                    let (lhs, rhs) = unify(ctx, lhs, rhs);
                    let bool_ty = ctx.t_bool;
                    let id = if lhs.ty == ScalarTy::F32 {
                        ctx.spv.op_slt(bool_ty, lhs.id, rhs.id)
                    } else {
                        ctx.spv.op_ult(bool_ty, lhs.id, rhs.id)
                    };
                    Ok(Val {
                        id,
                        ty: ScalarTy::Bool,
                    })
                }
                other => Err((
                    other.span(),
                    "ZSL: unsupported op (step 4: +, -, *, /, <)".into(),
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
