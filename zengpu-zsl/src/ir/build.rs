//! Front-end AST → [`Module`] IR for compute entry points.
//!
//! This is the single tree-walk over the parsed `syn` body. It resolves
//! identifiers (local / scalar param / buffer), checks structural validity
//! (buffer existence + writability, known builtins), and emits backend-neutral
//! IR nodes. Value-type inference is deferred to the backends.

use std::collections::{HashMap, HashSet};

use proc_macro2::Span;
use syn::{
    BinOp, Block, Expr, ExprBinary, ExprCall, ExprField, ExprForLoop, ExprIndex, ExprLit, ExprPath,
    ExprRange, ExprUnary, Lit, Member, Pat, RangeLimits, Stmt, UnOp, spanned::Spanned,
};

use crate::frontend::ast::ZslEntryPoint;
use crate::frontend::types::ZslType;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::{Entry, EntryKind, Module, Mutability, Param, ParamKind, ScalarTy};

/// Symbol tables used while resolving identifiers in the body.
struct Scope {
    /// Buffer params → writable flag.
    buffers: HashMap<String, bool>,
    scalar_params: HashSet<String>,
    locals: HashSet<String>,
}

/// Build the compute IR for a parsed entry point.
pub fn build_compute(
    entry: &ZslEntryPoint,
    body: &Block,
    local_size: (u32, u32, u32),
) -> Result<Module, (Span, String)> {
    // ── Classify params (declaration order preserved) ────────────────────────
    let mut params: Vec<Param> = Vec::new();
    let mut buffers: HashMap<String, bool> = HashMap::new();
    let mut scalar_params: HashSet<String> = HashSet::new();
    for p in &entry.params {
        let name = p.ident.to_string();
        match p.ty {
            ZslType::Buf(elem) | ZslType::BufMut(elem) => {
                let writable = matches!(p.ty, ZslType::BufMut(_));
                buffers.insert(name.clone(), writable);
                params.push(Param {
                    name,
                    kind: ParamKind::Buffer {
                        elem,
                        mutability: if writable {
                            Mutability::ReadWrite
                        } else {
                            Mutability::Read
                        },
                    },
                });
            }
            ZslType::U32 => {
                scalar_params.insert(name.clone());
                params.push(Param {
                    name,
                    kind: ParamKind::Scalar(ScalarTy::U32),
                });
            }
            ZslType::F32 => {
                scalar_params.insert(name.clone());
                params.push(Param {
                    name,
                    kind: ParamKind::Scalar(ScalarTy::F32),
                });
            }
            // Other param types are not valid for compute entry points; they are
            // simply not classified (matching the prior lowering, which filtered
            // to buffers + u32/f32 scalars).
            _ => {}
        }
    }

    // ── Pre-scan locals (declaration order) ──────────────────────────────────
    let mut locals_vec: Vec<(String, ScalarTy)> = Vec::new();
    collect_block_locals(body, &mut locals_vec)?;
    let locals: HashSet<String> = locals_vec.iter().map(|(n, _)| n.clone()).collect();

    let scope = Scope {
        buffers,
        scalar_params,
        locals,
    };

    let body_ir = build_stmts(&scope, body)?;

    Ok(Module {
        entry: Entry {
            kind: EntryKind::Compute {
                local_size: [local_size.0, local_size.1, local_size.2],
            },
            params,
            locals: locals_vec,
            body: body_ir,
        },
    })
}

// ── Local pre-scan (mirrors the prior collect_all_locals traversal) ──────────

fn collect_block_locals(
    block: &Block,
    out: &mut Vec<(String, ScalarTy)>,
) -> Result<(), (Span, String)> {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                let ident = local_ident_str(local)?;
                let sty = match &local.pat {
                    syn::Pat::Type(pt) => scalar_ty_from_syn(&pt.ty)?,
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
                let loop_var = match expr_for.pat.as_ref() {
                    Pat::Ident(p) => p.ident.to_string(),
                    _ => {
                        return Err((
                            Span::call_site(),
                            "ZSL: for-loop pattern must be a simple identifier".into(),
                        ));
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
            p => Err((p.span(), "ZSL: let binding must be a simple identifier".into())),
        },
        syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
        p => Err((p.span(), "ZSL: let binding must be a simple identifier".into())),
    }
}

// ── Statement building ───────────────────────────────────────────────────────

fn build_stmts(scope: &Scope, block: &Block) -> Result<Vec<IrStmt>, (Span, String)> {
    block.stmts.iter().map(|s| build_stmt(scope, s)).collect()
}

fn build_stmt(scope: &Scope, stmt: &Stmt) -> Result<IrStmt, (Span, String)> {
    match stmt {
        Stmt::Local(local) => {
            let name = local_ident_str(local)?;
            let init = match &local.init {
                Some(init) => build_expr(scope, &init.expr)?,
                // A bare `let x;` had no initializer in the prior lowering (it
                // simply declared the variable). Represent as `x = x`-free: keep
                // the declaration only by lowering to an Eval of a 0 — but the
                // prior code emitted nothing, so model "no init" as an explicit
                // marker. In practice every ZSL `let` has an initializer.
                None => {
                    return Err((
                        local.span(),
                        "ZSL: `let` binding requires an initializer".into(),
                    ));
                }
            };
            Ok(IrStmt::Let { name, init })
        }
        Stmt::Expr(expr, _) => build_effect(scope, expr),
        other => Err((
            Span::call_site(),
            format!("ZSL: unsupported statement `{}`", quote::quote!(#other)),
        )),
    }
}

fn build_effect(scope: &Scope, expr: &Expr) -> Result<IrStmt, (Span, String)> {
    match expr {
        Expr::Assign(assign) => match &*assign.left {
            Expr::Index(ExprIndex {
                expr: base, index, ..
            }) => {
                let buf = expr_path_ident(base)?;
                match scope.buffers.get(&buf) {
                    None => Err((base.span(), format!("ZSL: `{buf}` is not a buffer param"))),
                    Some(false) => Err((
                        base.span(),
                        format!("`{buf}` is `Buf<T>` (read-only); use `BufMut<T>` to write"),
                    )),
                    Some(true) => Ok(IrStmt::AssignBuffer {
                        buf,
                        index: build_expr(scope, index)?,
                        value: build_expr(scope, &assign.right)?,
                    }),
                }
            }
            Expr::Path(ExprPath { path, .. }) => {
                let name = path
                    .get_ident()
                    .ok_or_else(|| {
                        (path.span(), "ZSL: assignment target must be a simple name".into())
                    })?
                    .to_string();
                if !scope.locals.contains(&name) {
                    return Err((
                        path.span(),
                        format!("ZSL: `{name}` is not a declared local variable"),
                    ));
                }
                Ok(IrStmt::AssignLocal {
                    name,
                    value: build_expr(scope, &assign.right)?,
                })
            }
            other => Err((
                other.span(),
                "ZSL: assignment target must be `buf[i]` or a local variable name".into(),
            )),
        },
        Expr::If(expr_if) => {
            let cond = build_expr(scope, &expr_if.cond)?;
            let then = build_stmts(scope, &expr_if.then_branch)?;
            let else_ = match &expr_if.else_branch {
                Some((_, else_expr)) => match else_expr.as_ref() {
                    Expr::Block(eb) => Some(build_stmts(scope, &eb.block)?),
                    other => {
                        return Err((other.span(), "ZSL: else branch must be a block `{ ... }`".into()));
                    }
                },
                None => None,
            };
            Ok(IrStmt::If { cond, then, else_ })
        }
        Expr::ForLoop(ExprForLoop { pat, expr, body, .. }) => {
            let var = match pat.as_ref() {
                Pat::Ident(p) => p.ident.to_string(),
                other => {
                    return Err((other.span(), "ZSL: for-loop pattern must be a simple identifier".into()));
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
            Ok(IrStmt::For {
                var,
                lo: build_expr(scope, lo_expr)?,
                hi: build_expr(scope, hi_expr)?,
                body: build_stmts(scope, body)?,
            })
        }
        other => Ok(IrStmt::Eval(build_expr(scope, other)?)),
    }
}

// ── Expression building ──────────────────────────────────────────────────────

fn build_expr(scope: &Scope, expr: &Expr) -> Result<IrExpr, (Span, String)> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit), ..
        }) => {
            let v: u32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of u32 range", lit)))?;
            Ok(IrExpr::LitU32(v))
        }
        Expr::Lit(ExprLit {
            lit: Lit::Float(lit),
            ..
        }) => {
            let v: f32 = lit
                .base10_parse()
                .map_err(|_| (lit.span(), format!("ZSL: `{}` out of f32 range", lit)))?;
            Ok(IrExpr::LitF32(v))
        }
        Expr::Path(ExprPath { path, .. }) => {
            let ident = path
                .get_ident()
                .ok_or_else(|| (path.span(), "ZSL: expected simple identifier".into()))?
                .to_string();
            // Locals shadow scalar params (matches the prior resolution order).
            if scope.locals.contains(&ident) {
                Ok(IrExpr::Local(ident))
            } else if scope.scalar_params.contains(&ident) {
                Ok(IrExpr::ScalarParam(ident))
            } else {
                Err((path.span(), format!("ZSL: unknown identifier `{ident}`")))
            }
        }
        Expr::Index(ExprIndex {
            expr: base, index, ..
        }) => {
            let buf = expr_path_ident(base)?;
            if !scope.buffers.contains_key(&buf) {
                return Err((base.span(), format!("ZSL: `{buf}` is not a buffer param")));
            }
            Ok(IrExpr::BufferLoad {
                buf,
                index: Box::new(build_expr(scope, index)?),
            })
        }
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
                        return Ok(IrExpr::GlobalId(component));
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
            if name == "global_id" {
                return Err((func.span(), "ZSL: use `global_id().x`, `.y`, or `.z`".into()));
            }
            let func = builtin_from_name(&name).ok_or_else(|| {
                (
                    ident.span(),
                    format!(
                        "ZSL: unknown function `{name}`; built-ins: \
                         abs, sign, sqrt, floor, ceil, fract, min, max, pow, clamp, mix; \
                         or use global_id().x/y/z"
                    ),
                )
            })?;
            let args = args
                .iter()
                .map(|a| build_expr(scope, a))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrExpr::Builtin { func, args })
        }
        Expr::Unary(ExprUnary { op, expr, .. }) => match op {
            UnOp::Neg(_) => Ok(IrExpr::Neg(Box::new(build_expr(scope, expr)?))),
            other => Err((other.span(), "ZSL: only unary `-` is supported".into())),
        },
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let ir_op = binop_from_syn(op)?;
            Ok(IrExpr::Binary {
                op: ir_op,
                lhs: Box::new(build_expr(scope, left)?),
                rhs: Box::new(build_expr(scope, right)?),
            })
        }
        Expr::Paren(ep) => build_expr(scope, &ep.expr),
        other => Err((
            other.span(),
            format!("ZSL: unsupported expression `{}`", quote::quote!(#other)),
        )),
    }
}

// ── Small helpers ────────────────────────────────────────────────────────────

fn builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        _ => return None,
    })
}

fn binop_from_syn(op: &BinOp) -> Result<IrBinOp, (Span, String)> {
    Ok(match op {
        BinOp::Add(_) => IrBinOp::Add,
        BinOp::Sub(_) => IrBinOp::Sub,
        BinOp::Mul(_) => IrBinOp::Mul,
        BinOp::Div(_) => IrBinOp::Div,
        BinOp::Lt(_) => IrBinOp::Lt,
        BinOp::Le(_) => IrBinOp::Le,
        BinOp::Gt(_) => IrBinOp::Gt,
        BinOp::Ge(_) => IrBinOp::Ge,
        BinOp::Eq(_) => IrBinOp::Eq,
        BinOp::Ne(_) => IrBinOp::Ne,
        BinOp::And(_) => IrBinOp::And,
        BinOp::Or(_) => IrBinOp::Or,
        other => {
            return Err((
                other.span(),
                "ZSL: unsupported op; use +, -, *, /, <, >, <=, >=, ==, !=, &&, ||".into(),
            ));
        }
    })
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

fn scalar_ty_from_syn(ty: &syn::Type) -> Result<ScalarTy, (Span, String)> {
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
