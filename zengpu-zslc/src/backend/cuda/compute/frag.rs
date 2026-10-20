//! CUDA lowering of `frag<…>` onto `nvcuda::wmma` (16x16x16, compute_70+).
//!
//! Element mapping is opaque, so only load / mma / store lower. `frag_len`, `frag_row`, `frag_col`,
//! `frag_get` and `frag_set` are compile errors.

use std::fmt::Write;

use super::{Ctx, emit_expr};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrExpr};
use crate::ir::typeck::frag_parts;
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Type, TypeId};

pub(super) const FRAG_PRELUDE: &str = r#"#include <cuda_fp16.h>
#include <mma.h>
// wmma cannot convert on load, so `zsl_load` reads the staged `__half` tile.
__device__ __forceinline__ void zsl_stage_tile(__half* dst, const float* src, unsigned int off, unsigned int ld, unsigned int lane) {
    for (unsigned int i = 0; i < 8u; ++i) {
        unsigned int e = lane + 32u * i;
        dst[e] = __float2half(src[off + (e >> 4) * ld + (e & 15u)]);
    }
}
"#;

const LAYOUT_ROW: &str = "nvcuda::wmma::mem_row_major";

/// The `wmma::fragment` C++ type of a fragment local, or `None` for other types.
pub(super) fn cuda_frag_type(types: &TypeArena, ty: TypeId) -> Option<&'static str> {
    let (elem, usage) = frag_parts(types, ty)?;
    Some(match (types.get(elem), usage) {
        (
            Type::Float {
                format: FloatFormat::F16,
            },
            FragUse::A,
        ) => {
            "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>"
        }
        (
            Type::Float {
                format: FloatFormat::F16,
            },
            FragUse::B,
        ) => {
            "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major>"
        }
        (
            Type::Int {
                width: IntWidth::W8,
                ..
            },
            FragUse::A,
        ) => {
            "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major>"
        }
        (
            Type::Int {
                width: IntWidth::W8,
                ..
            },
            FragUse::B,
        ) => {
            "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major>"
        }
        (
            Type::Float {
                format: FloatFormat::F32,
            },
            FragUse::Acc,
        ) => "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>",
        (
            Type::Int {
                width: IntWidth::W32,
                ..
            },
            FragUse::Acc,
        ) => "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int>",
        _ => return None,
    })
}

fn unsupported(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Unsupported,
        format!("CUDA backend: {}", msg.into()),
    )
}

/// Fragment ops used as values are errors: fragments come only from `let` or assignment.
pub(super) fn expr_error(func: &str) -> Diagnostic {
    unsupported(format!(
        "{func} is a statement on CUDA: bind it with `let` or assign it to a frag local"
    ))
}

pub(super) fn is_frag_statement(e: &IrExpr) -> bool {
    matches!(
        e,
        IrExpr::FragLoad { .. }
            | IrExpr::Builtin {
                func: BuiltinFn::FragZeroAcc | BuiltinFn::Mma,
                ..
            }
    )
}

fn load_args(
    ctx: &Ctx<'_>,
    src: &FragSrc,
    usage: FragUse,
    offset: &str,
    ld: &str,
) -> Result<String, Diagnostic> {
    let name = match src {
        FragSrc::Buffer(n) | FragSrc::Shared(n) => n,
    };
    let elem = match src {
        FragSrc::Buffer(n) => ctx.type_ctx.buffers.get(n).copied(),
        FragSrc::Shared(_) => None,
    };
    let elem_ty = elem.map(|t| ctx.types.borrow().get(t).clone());
    let ptr = match (usage, src, elem_ty) {
        (
            FragUse::A | FragUse::B,
            FragSrc::Buffer(_),
            Some(Type::Float {
                format: FloatFormat::F16,
            }),
        ) => {
            format!("(const __half*){name}")
        }
        (
            FragUse::A | FragUse::B,
            FragSrc::Buffer(_),
            Some(Type::Int {
                width: IntWidth::W32,
                ..
            }),
        ) => {
            format!("(const signed char*){name}")
        }
        (
            FragUse::Acc,
            FragSrc::Buffer(_),
            Some(Type::Float {
                format: FloatFormat::F32,
            }),
        ) => {
            format!("(const float*){name}")
        }
        (
            FragUse::Acc,
            FragSrc::Buffer(_),
            Some(Type::Int {
                width: IntWidth::W32,
                ..
            }),
        ) => {
            format!("(const int*){name}")
        }
        (FragUse::Acc, FragSrc::Shared(_), _) => name.to_string(),
        _ => {
            return Err(unsupported(format!(
                "frag load from `{name}`: CUDA loads f16 or packed-i8 buffers into a/b, and f32 or i32 buffers or shared f32 into acc"
            )));
        }
    };
    let layout = if usage == FragUse::Acc {
        format!(", {LAYOUT_ROW}")
    } else {
        String::new()
    };
    Ok(format!("({ptr} + ({offset})), {ld}{layout}"))
}

/// A and B load from shared f32 through a per-warp `__half` tile, because `load_matrix_sync` reads
/// the fragment's element type.
fn load_stmt(
    ctx: &Ctx<'_>,
    name: &str,
    usage: FragUse,
    src: &FragSrc,
    off: &str,
    ldv: &str,
) -> Result<String, Diagnostic> {
    if let FragSrc::Shared(tile) = src
        && matches!(usage, FragUse::A | FragUse::B)
    {
        let warps = ctx.block_size.div_ceil(32);
        let [lx, ly, _] = ctx.local_size;
        return Ok(format!(
            "{{ __shared__ __half zsl_stage[{warps}][256]; \
             const unsigned int zsl_lin = threadIdx.x + {lx}u * (threadIdx.y + {ly}u * threadIdx.z); \
             zsl_stage_tile(zsl_stage[zsl_lin >> 5], {tile}, {off}, {ldv}, zsl_lin & 31u); \
             __syncwarp(); \
             nvcuda::wmma::load_matrix_sync({name}, zsl_stage[zsl_lin >> 5], 16); \
             __syncwarp(); }}"
        ));
    }
    let args = load_args(ctx, src, usage, off, ldv)?;
    Ok(format!("nvcuda::wmma::load_matrix_sync({name}, {args});"))
}

pub(super) fn let_stmt(
    ctx: &Ctx<'_>,
    ty: &str,
    name: &str,
    init: &IrExpr,
) -> Result<String, Diagnostic> {
    let mut out = format!("{ty} {name};");
    match init {
        IrExpr::FragLoad {
            ty: fty,
            src,
            offset,
            ld,
        } => {
            let usage = usage_from(ctx, *fty)?;
            let off = emit_expr(ctx, offset)?;
            let ldv = emit_expr(ctx, ld)?;
            let _ = write!(out, " {}", load_stmt(ctx, name, usage, src, &off, &ldv)?);
        }
        _ => out.push_str(&mma_or_zero(ctx, name, init)?),
    }
    Ok(out)
}

pub(super) fn assign_stmt(ctx: &Ctx<'_>, name: &str, value: &IrExpr) -> Result<String, Diagnostic> {
    match value {
        IrExpr::FragLoad {
            ty: fty,
            src,
            offset,
            ld,
        } => {
            let usage = usage_from(ctx, *fty)?;
            let off = emit_expr(ctx, offset)?;
            let ldv = emit_expr(ctx, ld)?;
            load_stmt(ctx, name, usage, src, &off, &ldv)
        }
        _ => mma_or_zero(ctx, name, value),
    }
}

fn mma_or_zero(ctx: &Ctx<'_>, name: &str, value: &IrExpr) -> Result<String, Diagnostic> {
    match value {
        IrExpr::Builtin {
            func: BuiltinFn::FragZeroAcc,
            ..
        } => Ok(format!("nvcuda::wmma::fill_fragment({name}, 0);")),
        IrExpr::Builtin {
            func: BuiltinFn::Mma,
            args,
        } => {
            let a = emit_expr(ctx, &args[0])?;
            let b = emit_expr(ctx, &args[1])?;
            let zero_acc = matches!(
                args[2],
                IrExpr::Builtin {
                    func: BuiltinFn::FragZeroAcc,
                    ..
                }
            );
            if zero_acc {
                Ok(format!(
                    "nvcuda::wmma::fill_fragment({name}, 0); nvcuda::wmma::mma_sync({name}, {a}, {b}, {name});"
                ))
            } else {
                let c = emit_expr(ctx, &args[2])?;
                Ok(format!("nvcuda::wmma::mma_sync({name}, {a}, {b}, {c});"))
            }
        }
        other => Err(unsupported(format!(
            "frag local `{name}` must be set by frag_load, mma or frag_zero_acc, not {other:?}"
        ))),
    }
}

pub(super) fn store_stmt(
    ctx: &Ctx<'_>,
    src: &FragSrc,
    offset: &IrExpr,
    ld: &IrExpr,
    value: &IrExpr,
) -> Result<String, Diagnostic> {
    let name = match src {
        FragSrc::Buffer(n) | FragSrc::Shared(n) => n,
    };
    let off = emit_expr(ctx, offset)?;
    let ldv = emit_expr(ctx, ld)?;
    let val = emit_expr(ctx, value)?;
    let int_acc = matches!(
        src,
        FragSrc::Buffer(n) if ctx.type_ctx.buffers.get(n).is_some_and(|t| matches!(
            ctx.types.borrow().get(*t),
            Type::Int { width: IntWidth::W32, .. }
        ))
    );
    let ptr_ty = if int_acc { "int" } else { "float" };
    Ok(format!(
        "nvcuda::wmma::store_matrix_sync(({ptr_ty}*){name} + ({off}), {val}, {ldv}, {LAYOUT_ROW});"
    ))
}

fn usage_from(ctx: &Ctx<'_>, ty: TypeId) -> Result<FragUse, Diagnostic> {
    let types = ctx.types.borrow();
    frag_parts(&types, ty)
        .map(|(_, usage)| usage)
        .ok_or_else(|| unsupported("frag load must produce a frag type"))
}
