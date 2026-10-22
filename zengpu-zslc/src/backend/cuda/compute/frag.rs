//! CUDA lowering of `frag<…>` onto `nvcuda::wmma` (16x16x16, compute_70+).
//!
//! wmma's element mapping is opaque. `frag_row`/`frag_col` find an element by loading a fragment
//! of the same type from a tile of row (or column) indices; `frag_get` reads `x[i]`; `frag_set`
//! goes through a per-warp tile so copies of an element held by several lanes stay equal.

use std::fmt::Write;

use super::{Ctx, emit_expr};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrExpr};
use crate::ir::typeck::frag_parts;
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Type, TypeId};

pub(super) const FRAG_PRELUDE: &str = r#"#include <cuda_fp16.h>
#include <mma.h>
template <typename F> struct zsl_frag_col_major { static const bool value = false; };
template <typename U, int M, int N, int K, typename T>
struct zsl_frag_col_major<nvcuda::wmma::fragment<U, M, N, K, T, nvcuda::wmma::col_major> > {
    static const bool value = true;
};
template <typename U, int M, int N, int K, typename T, typename L>
__device__ __forceinline__ void zsl_frag_fill(nvcuda::wmma::fragment<U, M, N, K, T, L>& f, const T* p) {
    nvcuda::wmma::load_matrix_sync(f, p, 16);
}
template <int M, int N, int K, typename T>
__device__ __forceinline__ void zsl_frag_fill(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T, void>& f, const T* p) {
    nvcuda::wmma::load_matrix_sync(f, p, 16, nvcuda::wmma::mem_row_major);
}
// `d` gets each element's row (or column): it is loaded from a tile of indices. `mem` is the
// warp's first 1 KiB of scratch.
template <typename U, int M, int N, int K, typename T, typename L>
__device__ __forceinline__ void zsl_frag_index(nvcuda::wmma::fragment<U, M, N, K, T, L>& d, bool row, unsigned char* mem, unsigned int lane) {
    T* tile = reinterpret_cast<T*>(mem);
    const bool col_major = zsl_frag_col_major<nvcuda::wmma::fragment<U, M, N, K, T, L> >::value;
    __syncwarp();
    for (unsigned int e = lane; e < 256u; e += 32u) {
        unsigned int r = col_major ? (e & 15u) : (e >> 4);
        unsigned int c = col_major ? (e >> 4) : (e & 15u);
        tile[e] = (T)(float)(row ? r : c);
    }
    __syncwarp();
    zsl_frag_fill(d, tile);
    __syncwarp();
}
template <typename U, int M, int N, int K, typename T, typename L>
__device__ __forceinline__ unsigned int zsl_frag_pos(const nvcuda::wmma::fragment<U, M, N, K, T, L>&, unsigned int i, bool row, unsigned char* mem, unsigned int lane) {
    nvcuda::wmma::fragment<U, M, N, K, T, L> d;
    zsl_frag_index(d, row, mem, lane);
    return (unsigned int)(float)d.x[i];
}
// Every lane writes its elements to the warp's tile, the caller's element changes, and every lane
// reloads, so duplicated elements stay equal. `mem` is the warp's 2 KiB of scratch.
template <typename U, int M, int N, int K, typename T, typename L, typename V>
__device__ __forceinline__ void zsl_frag_set(nvcuda::wmma::fragment<U, M, N, K, T, L>& f, unsigned int i, V v, unsigned char* mem, unsigned int lane) {
    nvcuda::wmma::fragment<U, M, N, K, T, L> dr, dc;
    zsl_frag_index(dr, true, mem, lane);
    zsl_frag_index(dc, false, mem, lane);
    T* tile = reinterpret_cast<T*>(mem + 1024);
    for (int k = 0; k < f.num_elements; ++k) {
        tile[(unsigned int)(float)dr.x[k] * 16u + (unsigned int)(float)dc.x[k]] = f.x[k];
    }
    __syncwarp();
    tile[(unsigned int)(float)dr.x[i] * 16u + (unsigned int)(float)dc.x[i]] = (T)v;
    __syncwarp();
    for (int k = 0; k < f.num_elements; ++k) {
        f.x[k] = tile[(unsigned int)(float)dr.x[k] * 16u + (unsigned int)(float)dc.x[k]];
    }
    __syncwarp();
}
// wmma cannot convert on load, so `zsl_load` reads the staged `__half` tile.
__device__ __forceinline__ void zsl_stage_tile(__half* dst, const float* src, unsigned int off, unsigned int ld, unsigned int lane) {
    for (unsigned int i = 0; i < 8u; ++i) {
        unsigned int e = lane + 32u * i;
        dst[e] = __float2half(src[off + (e >> 4) * ld + (e & 15u)]);
    }
}
// f32 a and b: wmma has no exact f32 product (tf32 is not f32), so they hold 8 floats per lane in
// the fixed layout below. Lane l holds A[l%16][8*(l/16)+i] (kA), B at [8*(l/16)+i][l%16].
template <bool kA> struct zsl_f32_ab { float x[8]; static const int num_elements = 8; };
typedef zsl_f32_ab<true> zsl_f32_a;
typedef zsl_f32_ab<false> zsl_f32_b;
template <bool kA> __device__ __forceinline__ unsigned int zsl_f32_row(unsigned int lane, unsigned int i) {
    return kA ? (lane & 15u) : (8u * (lane >> 4) + i);
}
template <bool kA> __device__ __forceinline__ unsigned int zsl_f32_col(unsigned int lane, unsigned int i) {
    return kA ? (8u * (lane >> 4) + i) : (lane & 15u);
}
// `off`, `ld` in elements. `T`: B from N x K row-major, element (k, n) at n * ld + k.
template <bool kA, bool T>
__device__ __forceinline__ zsl_f32_ab<kA> zsl_load_f32_ab(const float* p, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_f32_ab<kA> v;
    for (unsigned int i = 0; i < 8u; ++i) {
        unsigned int r = zsl_f32_row<kA>(lane, i);
        unsigned int c = zsl_f32_col<kA>(lane, i);
        if (T) { unsigned int t = r; r = c; c = t; }
        v.x[i] = p[off + r * ld + c];
    }
    return v;
}
// The products are staged in the warp's 2 KiB scratch (`mem`): a at [row][k], b at [k][col]. Each
// lane takes its acc elements as f32 FMA chains over k, so the result is exact f32.
template <typename U, int M, int N, int K, typename T, typename L>
__device__ __forceinline__ void zsl_mma_f32(nvcuda::wmma::fragment<U, M, N, K, T, L>& d, const zsl_f32_ab<true>& a, const zsl_f32_ab<false>& b, const nvcuda::wmma::fragment<U, M, N, K, T, L>& c, unsigned char* mem, unsigned int lane) {
    // d's element positions come from the wmma mapping, read before the scratch is reused.
    nvcuda::wmma::fragment<U, M, N, K, T, L> dr, dc;
    zsl_frag_index(dr, true, mem, lane);
    zsl_frag_index(dc, false, mem, lane);
    float* sa = reinterpret_cast<float*>(mem);
    float* sb = sa + 256;
    for (unsigned int i = 0; i < 8u; ++i) {
        sa[zsl_f32_row<true>(lane, i) * 16u + zsl_f32_col<true>(lane, i)] = a.x[i];
        sb[zsl_f32_row<false>(lane, i) * 16u + zsl_f32_col<false>(lane, i)] = b.x[i];
    }
    __syncwarp();
    for (int k = 0; k < d.num_elements; ++k) {
        unsigned int r = (unsigned int)(float)dr.x[k];
        unsigned int col = (unsigned int)(float)dc.x[k];
        float acc = (float)c.x[k];
        for (unsigned int kk = 0; kk < 16u; ++kk) acc = fmaf(sa[r * 16u + kk], sb[kk * 16u + col], acc);
        d.x[k] = (T)acc;
    }
    __syncwarp();
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
            Type::Float {
                format: FloatFormat::F32,
            },
            FragUse::A,
        ) => "zsl_f32_a",
        (
            Type::Float {
                format: FloatFormat::F32,
            },
            FragUse::B,
        ) => "zsl_f32_b",
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
                func: BuiltinFn::FragZeroAcc | BuiltinFn::Mma | BuiltinFn::FragSet,
                ..
            }
    )
}

/// Kernel-scope scratch for the element builtins: 2 KiB per warp, and this thread's lane.
pub(super) fn element_prologue(local_size: [u32; 3], block_size: u32) -> String {
    let warps = block_size.div_ceil(32);
    let [lx, ly, _] = local_size;
    format!(
        "    __shared__ __align__(16) unsigned char zsl_frag_mem[{warps}][2048];\n\
         \x20   const unsigned int zsl_frag_lin = threadIdx.x + {lx}u * (threadIdx.y + {ly}u * threadIdx.z);\n\
         \x20   unsigned char* zsl_frag_scratch = zsl_frag_mem[zsl_frag_lin >> 5];\n\
         \x20   const unsigned int zsl_frag_lane = zsl_frag_lin & 31u;\n"
    )
}

/// `frag_len`, `frag_row`, `frag_col` and `frag_get` as expressions.
pub(super) fn element_expr(
    ctx: &Ctx<'_>,
    func: BuiltinFn,
    args: &[IrExpr],
) -> Result<String, Diagnostic> {
    let IrExpr::Local(f) = &args[0] else {
        return Err(unsupported(format!(
            "{}: the first argument must be a frag local",
            func.name()
        )));
    };
    ctx.frag_elements.set(true);
    if func == BuiltinFn::FragLen {
        return Ok(format!("((unsigned int){f}.num_elements)"));
    }
    let i = emit_expr(ctx, &args[1])?;
    let f32_ab = f32_ab_usage(ctx, f);
    Ok(match func {
        BuiltinFn::FragRow | BuiltinFn::FragCol if f32_ab.is_some() => {
            let k_a = f32_ab == Some(FragUse::A);
            let axis = if func == BuiltinFn::FragRow {
                "row"
            } else {
                "col"
            };
            format!("zsl_f32_{axis}<{k_a}>(zsl_frag_lane, {i})")
        }
        BuiltinFn::FragRow | BuiltinFn::FragCol => format!(
            "zsl_frag_pos({f}, {i}, {}, zsl_frag_scratch, zsl_frag_lane)",
            func == BuiltinFn::FragRow
        ),
        _ if frag_is_int(ctx, f) => format!("((int)({f}.x[{i}]))"),
        _ => format!("((float)({f}.x[{i}]))"),
    })
}

/// The usage of a fragment local that is an f32 a or b, which has the fixed lane layout.
fn f32_ab_usage(ctx: &Ctx<'_>, local: &str) -> Option<FragUse> {
    let types = ctx.types.borrow();
    let (elem, usage) = ctx
        .type_ctx
        .locals
        .get(local)
        .and_then(|t| frag_parts(&types, *t))?;
    let f32 = Type::Float {
        format: FloatFormat::F32,
    };
    (types.get(elem) == &f32 && usage != FragUse::Acc).then_some(usage)
}

fn frag_is_int(ctx: &Ctx<'_>, local: &str) -> bool {
    let types = ctx.types.borrow();
    ctx.type_ctx
        .locals
        .get(local)
        .and_then(|t| frag_parts(&types, *t))
        .is_some_and(|(e, _)| matches!(types.get(e), Type::Int { .. }))
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
        FragSrc::Shared(n) => ctx.type_ctx.shared.get(n).copied(),
    };
    let elem_ty = elem.map(|t| ctx.types.borrow().get(t).clone());
    let ptr = match (usage, src, elem_ty) {
        (
            FragUse::A | FragUse::B,
            _,
            Some(Type::Float {
                format: FloatFormat::F16,
            }),
        ) => {
            format!("(const __half*){name}")
        }
        (
            FragUse::A | FragUse::B,
            FragSrc::Shared(_),
            Some(Type::Int {
                width: IntWidth::W8,
                ..
            }),
        ) => {
            format!("(const signed char*){name}")
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
                "frag load from `{name}`: CUDA loads f16 shared or buffers and packed-i8 buffers into a/b, i8 shared into a/b, and f32 or i32 buffers or shared arrays into acc"
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

fn is_shared_f32(ctx: &Ctx<'_>, name: &str) -> bool {
    ctx.type_ctx.shared.get(name).is_some_and(|elem| {
        ctx.types.borrow().get(*elem)
            == &Type::Float {
                format: FloatFormat::F32,
            }
    })
}

/// A/B from f32 shared go through a per-warp `__half` tile: `load_matrix_sync` reads the fragment's
/// element type. f16 and i8 load directly.
fn load_stmt(
    ctx: &Ctx<'_>,
    name: &str,
    usage: FragUse,
    src: &FragSrc,
    off: &str,
    ldv: &str,
    trans: bool,
) -> Result<String, Diagnostic> {
    if f32_ab_usage(ctx, name).is_some() {
        ctx.frag_elements.set(true);
        let src_name = match src {
            FragSrc::Buffer(n) | FragSrc::Shared(n) => n,
        };
        let k_a = usage == FragUse::A;
        return Ok(format!(
            "{name} = zsl_load_f32_ab<{k_a}, {trans}>((const float*){src_name}, {off}, {ldv}, zsl_frag_lane);"
        ));
    }
    if let FragSrc::Shared(tile) = src
        && matches!(usage, FragUse::A | FragUse::B)
        && is_shared_f32(ctx, tile)
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
            trans,
        } => {
            let usage = usage_from(ctx, *fty)?;
            let off = emit_expr(ctx, offset)?;
            let ldv = emit_expr(ctx, ld)?;
            if *trans {
                // A col_major fragment reads the (staged or direct) row-major block transposed.
                out = format!("{} {name};", ty.replace("row_major>", "col_major>"));
            }
            let _ = write!(
                out,
                " {}",
                load_stmt(ctx, name, usage, src, &off, &ldv, *trans)?
            );
        }
        _ => out.push_str(&mma_or_zero(ctx, name, init)?),
    }
    Ok(out)
}

pub(super) fn assign_stmt(ctx: &Ctx<'_>, name: &str, value: &IrExpr) -> Result<String, Diagnostic> {
    match value {
        IrExpr::FragLoad { trans: true, .. } => Err(unsupported(
            "frag_load_b_t must initialize a let local on CUDA (its wmma layout is part of the declaration)",
        )),
        IrExpr::FragLoad {
            ty: fty,
            src,
            offset,
            ld,
            trans: false,
        } => {
            let usage = usage_from(ctx, *fty)?;
            let off = emit_expr(ctx, offset)?;
            let ldv = emit_expr(ctx, ld)?;
            load_stmt(ctx, name, usage, src, &off, &ldv, false)
        }
        _ => mma_or_zero(ctx, name, value),
    }
}

fn mma_or_zero(ctx: &Ctx<'_>, name: &str, value: &IrExpr) -> Result<String, Diagnostic> {
    match value {
        IrExpr::Builtin {
            func: BuiltinFn::FragSet,
            args,
        } => {
            let IrExpr::Local(src) = &args[0] else {
                return Err(unsupported(
                    "frag_set: the first argument must be a frag local",
                ));
            };
            ctx.frag_elements.set(true);
            let copy = if src == name {
                String::new()
            } else {
                format!("{name} = {src}; ")
            };
            let i = emit_expr(ctx, &args[1])?;
            let x = emit_expr(ctx, &args[2])?;
            if f32_ab_usage(ctx, name).is_some() {
                // Each element is held by one lane, so the lane's own element is the write.
                return Ok(format!("{copy}{name}.x[{i}] = (float)({x});"));
            }
            Ok(format!(
                "{copy}zsl_frag_set({name}, {i}, {x}, zsl_frag_scratch, zsl_frag_lane);"
            ))
        }
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
            let f32_ab = matches!(&args[0], IrExpr::Local(an) if f32_ab_usage(ctx, an).is_some());
            if f32_ab {
                ctx.frag_elements.set(true);
                let (zero, c) = if zero_acc {
                    (
                        format!("nvcuda::wmma::fill_fragment({name}, 0); "),
                        name.to_string(),
                    )
                } else {
                    (String::new(), emit_expr(ctx, &args[2])?)
                };
                return Ok(format!(
                    "{zero}zsl_mma_f32({name}, {a}, {b}, {c}, zsl_frag_scratch, zsl_frag_lane);"
                ));
            }
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
    let elem = match src {
        FragSrc::Buffer(n) => ctx.type_ctx.buffers.get(n),
        FragSrc::Shared(n) => ctx.type_ctx.shared.get(n),
    };
    let int_acc = elem.is_some_and(|t| {
        matches!(
            ctx.types.borrow().get(*t),
            Type::Int {
                width: IntWidth::W32,
                ..
            }
        )
    });
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
