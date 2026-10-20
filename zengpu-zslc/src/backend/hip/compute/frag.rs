//! HIP lowering of `frag<…>` and `mma` for gfx12 wave32 WMMA. A fragment is a per-lane C vector:
//! `zsl_h8` / `zsl_i2` for a and b, `zsl_f8` / `zsl_i8v` for acc.
//!
//! One layout mapping in [`FRAG_PRELUDE`] drives loads, stores, `frag_row`/`frag_col` and the
//! WMMA intrinsic.

use std::fmt::Write;

use super::{Ctx, expr_type_id};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrExpr};
use crate::ir::typeck::frag_parts;
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Signedness, Type, TypeId};

pub(super) const FRAG_PRELUDE: &str = r#"typedef _Float16 zsl_h8 __attribute__((ext_vector_type(8)));
typedef float zsl_f8 __attribute__((ext_vector_type(8)));
typedef int zsl_i2 __attribute__((ext_vector_type(2)));
typedef int zsl_i8v __attribute__((ext_vector_type(8)));
// gfx12 wave32: lane l holds A[l%16][8*(l/16)+i] (kA), B and acc at [8*(l/16)+i][l%16]. `off`, `ld`
// in elements; packed i8 in bytes. `T`: B from N x K row-major, element (k, n) at n * ld + k.
template <bool kA, bool T = false>
__device__ __forceinline__ unsigned int zsl_frag_at(unsigned int lane, unsigned int i, unsigned int off, unsigned int ld) {
    unsigned int r = kA ? (lane & 15u) : (8u * (lane >> 4) + i);
    unsigned int c = kA ? (8u * (lane >> 4) + i) : (lane & 15u);
    if (T) { unsigned int t = r; r = c; c = t; }
    return off + r * ld + c;
}
template <bool kA, bool T = false>
__device__ __forceinline__ zsl_h8 zsl_load_h8(const unsigned short* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_h8 v;
    for (unsigned int i = 0; i < 8; ++i) v[i] = __builtin_bit_cast(_Float16, b[zsl_frag_at<kA, T>(lane, i, off, ld)]);
    return v;
}
template <bool kA, bool T = false>
__device__ __forceinline__ zsl_h8 zsl_load_h8f(const float* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_h8 v;
    for (unsigned int i = 0; i < 8; ++i) v[i] = (_Float16)b[zsl_frag_at<kA, T>(lane, i, off, ld)];
    return v;
}
template <bool kA, bool T = false>
__device__ __forceinline__ zsl_i2 zsl_load_i2s(const signed char* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_i2 v = {0, 0};
    for (unsigned int i = 0; i < 8; ++i) {
        unsigned int byte = (unsigned int)(unsigned char)b[zsl_frag_at<kA, T>(lane, i, off, ld)];
        v[i >> 2] |= (int)(byte << (8u * (i & 3u)));
    }
    return v;
}
template <bool kA, bool T = false>
__device__ __forceinline__ zsl_i2 zsl_load_i2(const unsigned int* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_i2 v = {0, 0};
    for (unsigned int i = 0; i < 8; ++i) {
        unsigned int at = zsl_frag_at<kA, T>(lane, i, off, ld);
        unsigned int byte = (b[at >> 2] >> (8u * (at & 3u))) & 0xffu;
        v[i >> 2] |= (int)(byte << (8u * (i & 3u)));
    }
    return v;
}
__device__ __forceinline__ zsl_f8 zsl_load_f8(const float* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_f8 v;
    for (unsigned int i = 0; i < 8; ++i) v[i] = b[zsl_frag_at<false>(lane, i, off, ld)];
    return v;
}
__device__ __forceinline__ zsl_i8v zsl_load_i8v(const int* b, unsigned int off, unsigned int ld, unsigned int lane) {
    zsl_i8v v;
    for (unsigned int i = 0; i < 8; ++i) v[i] = b[zsl_frag_at<false>(lane, i, off, ld)];
    return v;
}
__device__ __forceinline__ void zsl_store_f8(float* b, unsigned int off, unsigned int ld, unsigned int lane, zsl_f8 v) {
    for (unsigned int i = 0; i < 8; ++i) b[zsl_frag_at<false>(lane, i, off, ld)] = v[i];
}
__device__ __forceinline__ void zsl_store_i8v(int* b, unsigned int off, unsigned int ld, unsigned int lane, zsl_i8v v) {
    for (unsigned int i = 0; i < 8; ++i) b[zsl_frag_at<false>(lane, i, off, ld)] = v[i];
}
__device__ __forceinline__ zsl_f8 zsl_mma_f16(zsl_h8 a, zsl_h8 b, zsl_f8 c) {
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
#error "ZSL mma: only gfx12 wave32 WMMA is lowered for HIP"
#endif
}
__device__ __forceinline__ zsl_i8v zsl_mma_i8(zsl_i2 a, zsl_i2 b, zsl_i8v c) {
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
#error "ZSL mma: only gfx12 wave32 WMMA is lowered for HIP"
#endif
}
__device__ __forceinline__ float zsl_frag_get(zsl_h8 v, unsigned int i) { return (float)v[i]; }
__device__ __forceinline__ float zsl_frag_get(zsl_f8 v, unsigned int i) { return v[i]; }
__device__ __forceinline__ int zsl_frag_get(zsl_i2 v, unsigned int i) {
    return (int)(signed char)((unsigned int)v[i >> 2] >> (8u * (i & 3u)));
}
__device__ __forceinline__ int zsl_frag_get(zsl_i8v v, unsigned int i) { return v[i]; }
__device__ __forceinline__ zsl_h8 zsl_frag_set(zsl_h8 v, unsigned int i, float x) { v[i] = (_Float16)x; return v; }
__device__ __forceinline__ zsl_f8 zsl_frag_set(zsl_f8 v, unsigned int i, float x) { v[i] = x; return v; }
__device__ __forceinline__ zsl_i2 zsl_frag_set(zsl_i2 v, unsigned int i, int x) {
    unsigned int sh = 8u * (i & 3u);
    unsigned int w = (unsigned int)v[i >> 2];
    v[i >> 2] = (int)((w & ~(0xffu << sh)) | (((unsigned int)x & 0xffu) << sh));
    return v;
}
__device__ __forceinline__ zsl_i8v zsl_frag_set(zsl_i8v v, unsigned int i, int x) { v[i] = x; return v; }
"#;

pub(super) fn hip_frag_type(types: &TypeArena, ty: TypeId) -> Option<&'static str> {
    let (elem, usage) = frag_parts(types, ty)?;
    Some(match (types.get(elem), usage) {
        (
            Type::Float {
                format: FloatFormat::F16,
            },
            FragUse::A | FragUse::B,
        ) => "zsl_h8",
        (
            Type::Int {
                width: IntWidth::W8,
                ..
            },
            FragUse::A | FragUse::B,
        ) => "zsl_i2",
        (
            Type::Float {
                format: FloatFormat::F32,
            },
            FragUse::Acc,
        ) => "zsl_f8",
        (
            Type::Int {
                width: IntWidth::W32,
                ..
            },
            FragUse::Acc,
        ) => "zsl_i8v",
        _ => return None,
    })
}

fn unsupported(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Unsupported,
        format!("HIP backend: {}", msg.into()),
    )
}

fn frag_shape(ctx: &Ctx<'_>, ty: TypeId) -> Result<(TypeId, FragUse), Diagnostic> {
    let types = ctx.types.borrow();
    frag_parts(&types, ty).ok_or_else(|| unsupported("expected a frag value"))
}

/// A 32-bit integer element: a packed-i8 source word for a and b.
fn is_word(ctx: &Ctx<'_>, t: TypeId) -> bool {
    matches!(
        ctx.types.borrow().get(t),
        Type::Int {
            width: IntWidth::W32,
            ..
        }
    )
}

fn kind_is(ctx: &Ctx<'_>, t: TypeId, want: &Type) -> bool {
    ctx.types.borrow().get(t) == want
}

fn f16_type() -> Type {
    Type::Float {
        format: FloatFormat::F16,
    }
}

fn i32_type() -> Type {
    Type::Int {
        width: IntWidth::W32,
        signedness: Signedness::Signed,
    }
}

fn source_elem(ctx: &Ctx<'_>, src: &FragSrc) -> Result<TypeId, Diagnostic> {
    match src {
        FragSrc::Buffer(name) => ctx
            .type_ctx
            .buffers
            .get(name)
            .copied()
            .ok_or_else(|| unsupported(format!("`{name}` is not a buffer"))),
        FragSrc::Shared(name) => ctx
            .type_ctx
            .shared
            .get(name)
            .copied()
            .ok_or_else(|| unsupported(format!("`{name}` is not a shared array"))),
    }
}

fn src_name(src: &FragSrc) -> &str {
    match src {
        FragSrc::Buffer(n) | FragSrc::Shared(n) => n,
    }
}

pub(super) fn load_expr(
    ctx: &Ctx<'_>,
    ty: TypeId,
    src: &FragSrc,
    offset: &IrExpr,
    ld: &IrExpr,
    trans: bool,
) -> Result<String, Diagnostic> {
    ctx.frag_used.set(true);
    let (_, usage) = frag_shape(ctx, ty)?;
    let name = src_name(src);
    let elem = source_elem(ctx, src)?;
    let off = super::emit_expr(ctx, offset)?;
    let ld = super::emit_expr(ctx, ld)?;
    let args = format!("{off}, {ld}, __lane_id())");
    let k_a = usage == FragUse::A;
    let t = trans;
    let shared = matches!(src, FragSrc::Shared(_));
    let f16 = f16_type();
    let i32 = i32_type();
    let f32 = Type::Float {
        format: FloatFormat::F32,
    };
    let i8 = Type::Int {
        width: IntWidth::W8,
        signedness: Signedness::Signed,
    };
    Ok(match usage {
        FragUse::A | FragUse::B if shared && kind_is(ctx, elem, &f32) => {
            format!("zsl_load_h8f<{k_a}, {t}>({name}, {args}")
        }
        FragUse::A | FragUse::B if kind_is(ctx, elem, &f16) => {
            let p = if shared {
                "(const unsigned short*)"
            } else {
                ""
            };
            format!("zsl_load_h8<{k_a}, {t}>({p}{name}, {args}")
        }
        FragUse::A | FragUse::B if shared && kind_is(ctx, elem, &i8) => {
            format!("zsl_load_i2s<{k_a}, {t}>((const signed char*){name}, {args}")
        }
        FragUse::A | FragUse::B if is_word(ctx, elem) => {
            format!("zsl_load_i2<{k_a}, {t}>((const unsigned int*){name}, {args}")
        }
        FragUse::Acc if kind_is(ctx, elem, &i32) => {
            format!("zsl_load_i8v((const int*){name}, {args}")
        }
        FragUse::Acc if kind_is(ctx, elem, &f32) => {
            format!("zsl_load_f8({name}, {args}")
        }
        _ => {
            return Err(unsupported(format!(
                "frag load from `{name}`: source element does not fit this use"
            )));
        }
    })
}

pub(super) fn store_stmt(
    ctx: &Ctx<'_>,
    src: &FragSrc,
    offset: &IrExpr,
    ld: &IrExpr,
    value: &IrExpr,
) -> Result<String, Diagnostic> {
    ctx.frag_used.set(true);
    let name = src_name(src);
    let elem = source_elem(ctx, src)?;
    let off = super::emit_expr(ctx, offset)?;
    let ld = super::emit_expr(ctx, ld)?;
    let val = super::emit_expr(ctx, value)?;
    let mut out = String::new();
    if kind_is(ctx, elem, &i32_type()) {
        let _ = write!(
            out,
            "zsl_store_i8v((int*){name}, {off}, {ld}, __lane_id(), {val});"
        );
    } else {
        let _ = write!(
            out,
            "zsl_store_f8({name}, {off}, {ld}, __lane_id(), {val});"
        );
    }
    Ok(out)
}

/// `frag_zero_acc()` has no C value: `Let` and `AssignLocal` emit an empty initializer, `mma` an
/// empty accumulator.
pub(super) fn builtin_expr(
    ctx: &Ctx<'_>,
    func: BuiltinFn,
    args: &[IrExpr],
) -> Result<String, Diagnostic> {
    ctx.frag_used.set(true);
    let mut args_s = Vec::with_capacity(args.len());
    for a in args {
        args_s.push(match a {
            IrExpr::Builtin {
                func: BuiltinFn::FragZeroAcc,
                ..
            } => String::new(),
            _ => super::emit_expr(ctx, a)?,
        });
    }
    let args_s = &args_s;
    Ok(match func {
        BuiltinFn::Mma => {
            let (a_elem, _) = frag_shape(ctx, expr_type_id(ctx, &args[0]))?;
            let zero_acc = matches!(
                args[2],
                IrExpr::Builtin {
                    func: BuiltinFn::FragZeroAcc,
                    ..
                }
            );
            if kind_is(ctx, a_elem, &f16_type()) {
                let acc = if zero_acc {
                    "zsl_f8{}".to_string()
                } else {
                    args_s[2].clone()
                };
                format!("zsl_mma_f16({}, {}, {acc})", args_s[0], args_s[1])
            } else if kind_is(
                ctx,
                a_elem,
                &Type::Int {
                    width: IntWidth::W8,
                    signedness: Signedness::Signed,
                },
            ) {
                let acc = if zero_acc {
                    "zsl_i8v{}".to_string()
                } else {
                    args_s[2].clone()
                };
                format!("zsl_mma_i8({}, {}, {acc})", args_s[0], args_s[1])
            } else {
                return Err(unsupported("mma operands must be f16 or i8"));
            }
        }
        BuiltinFn::FragLen => "8u".to_string(),
        BuiltinFn::FragRow | BuiltinFn::FragCol => {
            let (_, usage) = frag_shape(ctx, expr_type_id(ctx, &args[0]))?;
            let i = &args_s[1];
            let row_is_lane_lo = (usage == FragUse::A) == (func == BuiltinFn::FragRow);
            if row_is_lane_lo {
                "(__lane_id() & 15u)".to_string()
            } else {
                format!("(8u * (__lane_id() >> 4) + ({i}))")
            }
        }
        BuiltinFn::FragGet => format!("zsl_frag_get({}, {})", args_s[0], args_s[1]),
        BuiltinFn::FragSet => format!("zsl_frag_set({}, {}, {})", args_s[0], args_s[1], args_s[2]),
        other => {
            return Err(unsupported(format!(
                "{} is not a fragment builtin",
                other.name()
            )));
        }
    })
}
