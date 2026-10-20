use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::cuda::{CudaShader, ENTRY};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt, stmts_use_workgroup_reduce};
use crate::ir::typeck::{ComputeTypeCtx, type_of_compute_expr};
use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type, TypeId};
use crate::ir::{EntryKind, Module, Mutability, ParamKind};

mod frag;

pub fn lower_compute(module: &Module) -> Result<CudaShader, Diagnostic> {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;
    if e.params
        .iter()
        .any(|p| matches!(p.kind, ParamKind::Texture | ParamKind::Image))
    {
        return Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            "CUDA has no texture or image params",
        ));
    }

    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    for (n, t) in &e.locals {
        let cty = match frag::cuda_frag_type(&module.types, *t) {
            Some(cty) => cty,
            None => cuda_scalar(module.types.get(*t))?,
        };
        locals.insert(n.as_str(), cty);
    }

    let buffers: Vec<(&str, Mutability, TypeId)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, mutability } => Some((p.name.as_str(), *mutability, *elem)),
            _ => None,
        })
        .collect();

    let scalars: Vec<(&str, TypeId)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Scalar(t) => Some((p.name.as_str(), *t)),
            _ => None,
        })
        .collect();

    let bf16_id = {
        let mut t = module.types.clone();
        t.bf16_id()
    };
    let bf16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == bf16_id)
        .map(|(name, _, _)| *name)
        .collect();

    let f16_id = {
        let mut t = module.types.clone();
        t.f16_id()
    };
    let f16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == f16_id)
        .map(|(name, _, _)| *name)
        .collect();
    let uses_workgroup_reduce = stmts_use_workgroup_reduce(&e.body);
    let block_size = local_size[0] * local_size[1] * local_size[2];
    if uses_workgroup_reduce && (local_size[1] != 1 || local_size[2] != 1) {
        return Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!(
                "ZSL: workgroup_reduce_* requires a 1D workgroup, got local_size {local_size:?}"
            ),
        ));
    }
    if uses_workgroup_reduce && !block_size.is_power_of_two() {
        return Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!(
                "ZSL: workgroup_reduce_* requires a power-of-two workgroup thread count, got {block_size}"
            ),
        ));
    }

    let mut s = String::new();

    if !f16_buffers.is_empty() {
        s.push_str("#include <cuda_fp16.h>\n");
    }
    if !bf16_buffers.is_empty() {
        s.push_str(
            "__device__ __forceinline__ unsigned short f32_to_bf16(float x) {\n\
             \x20   unsigned int bits = __float_as_uint(x);\n\
             \x20   unsigned int bias = 0x7fffu + ((bits >> 16) & 1u);\n\
             \x20   if ((bits & 0x7fffffffu) > 0x7f800000u) {\n\
             \x20       return (unsigned short)((bits >> 16) | 0x40u);\n\
             \x20   }\n\
             \x20   return (unsigned short)((bits + bias) >> 16);\n\
             }\n\
             __device__ __forceinline__ float bf16_to_f32(unsigned short x) {\n\
             \x20   return __uint_as_float(((unsigned int)x) << 16);\n\
             }\n",
        );
    }

    s.push_str(
        "__device__ __forceinline__ int zsl_dot4_i8(int a, int b, int acc) {\n\
         \x20   return __dp4a(a, b, acc);\n\
         }\n\
         __device__ __forceinline__ unsigned int zsl_dot4_u8(unsigned int a, unsigned int b, unsigned int acc) {\n\
         \x20   return __dp4a(a, b, acc);\n\
         }\n\
         __device__ __forceinline__ int zsl_dot4_iu8(int a, unsigned int b, int acc) {\n\
         \x20   unsigned int s = (unsigned int)acc;\n\
         \x20   for (int i = 0; i < 4; ++i) s += (unsigned int)(((signed char)(a >> (8 * i))) * (int)((b >> (8 * i)) & 0xffu));\n\
         \x20   return (int)s;\n\
         }\n",
    );

    s.push_str(r#"
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned int zsl_pk_add_u16(unsigned int a, unsigned int b) {
    return ((a & 0x7fff7fffu) + (b & 0x7fff7fffu)) ^ ((a ^ b) & 0x80008000u);
}
__device__ __forceinline__ unsigned int zsl_pk_mul_u16(unsigned int a, unsigned int b) {
    return ((a * b) & 0xffffu) | (((a >> 16) * (b >> 16)) << 16);
}
__device__ __forceinline__ unsigned int zsl_pk_mad_u16(unsigned int a, unsigned int b, unsigned int c) {
    return zsl_pk_add_u16(zsl_pk_mul_u16(a, b), c);
}
__device__ __forceinline__ __half2 zsl_h2(unsigned int x) {
    return *reinterpret_cast<__half2 *>(&x);
}
__device__ __forceinline__ unsigned int zsl_u32(__half2 h) {
    return *reinterpret_cast<unsigned int *>(&h);
}
__device__ __forceinline__ unsigned int zsl_pk_add_f16(unsigned int a, unsigned int b) {
    return zsl_u32(__hadd2(zsl_h2(a), zsl_h2(b)));
}
__device__ __forceinline__ unsigned int zsl_pk_mul_f16(unsigned int a, unsigned int b) {
    return zsl_u32(__hmul2(zsl_h2(a), zsl_h2(b)));
}
__device__ __forceinline__ unsigned int zsl_pk_fma_f16(unsigned int a, unsigned int b, unsigned int c) {
    return zsl_u32(__hfma2(zsl_h2(a), zsl_h2(b), zsl_h2(c)));
}
__device__ __forceinline__ float zsl_dot2_f16(unsigned int a, unsigned int b, float acc) {
    float2 x = __half22float2(zsl_h2(a));
    float2 y = __half22float2(zsl_h2(b));
    acc = fmaf(x.x, y.x, acc);
    return fmaf(x.y, y.y, acc);
}
__device__ __forceinline__ unsigned int zsl_funnel_shr(unsigned int hi, unsigned int lo, unsigned int s) {
    return __funnelshift_r(lo, hi, s);
}
__device__ __forceinline__ unsigned int zsl_byte_perm(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int r = 0;
    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int k = (sel >> (4 * i)) & 7u;
        unsigned int w = k < 4u ? a : b;
        r |= ((w >> (8 * (k & 3u))) & 0xffu) << (8 * i);
    }
    return r;
}

// device functions, not statement expressions: NVRTC rejects GNU statement expressions.
struct ZslAddOp {
    template <typename T> __device__ T operator()(T a, T b) const { return a + b; }
};
struct ZslAndOp {
    template <typename T> __device__ T operator()(T a, T b) const { return a & b; }
};
struct ZslOrOp {
    template <typename T> __device__ T operator()(T a, T b) const { return a | b; }
};
struct ZslMinOp {
    template <typename T> __device__ T operator()(T a, T b) const { return a < b ? a : b; }
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
};
struct ZslMaxOp {
    template <typename T> __device__ T operator()(T a, T b) const { return a > b ? a : b; }
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
template <typename T, typename Op> __device__ __forceinline__ T zsl_sg_xor_reduce(T v, Op op) {
    for (unsigned int off = warpSize / 2; off > 0; off /= 2) {
        v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}
template <typename T> __device__ __forceinline__ T zsl_sg_scan_incl_add(T v) {
    for (unsigned int off = 1; off < warpSize; off *= 2) {
        T o = __shfl_up_sync(0xffffffffu, v, off);
        if ((threadIdx.x % warpSize) >= off) v = v + o;
    }
    return v;
}
template <typename T> __device__ __forceinline__ T zsl_sg_scan_excl_add(T v) {
    T incl = zsl_sg_scan_incl_add(v);
    T excl = __shfl_up_sync(0xffffffffu, incl, 1u);
    if ((threadIdx.x % warpSize) == 0) excl = T(0);
    return excl;
}
template <typename Op> __device__ __forceinline__ float zsl_wg_reduce(float *scratch, unsigned int n, float v, Op op) {
    __syncthreads();
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) scratch[threadIdx.x] = op(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    return scratch[0];
}
"#);

    let threads = local_size[0] * local_size[1] * local_size[2];
    let kernel_start = s.len();
    let _ = writeln!(s, "extern \"C\" __global__ __launch_bounds__({threads})");
    let _ = writeln!(s, "void {ENTRY}(");

    for (i, (name, mutability, elem)) in buffers.iter().enumerate() {
        let comma = if i + 1 < buffers.len() + scalars.len() {
            ","
        } else {
            ""
        };
        match mutability {
            Mutability::Read => {
                let ty = cuda_buffer_elem(module.types.get(*elem))?;
                let _ = writeln!(s, "    const {ty}* __restrict__ {name}{comma}");
            }
            Mutability::ReadWrite => {
                let ty = cuda_buffer_elem(module.types.get(*elem))?;
                let _ = writeln!(s, "    {ty}* __restrict__ {name}{comma}");
            }
        }
    }

    for (i, (name, ty)) in scalars.iter().enumerate() {
        let comma = if i + 1 < scalars.len() { "," } else { "" };
        let cty = cuda_scalar(module.types.get(*ty))?;
        let _ = writeln!(s, "    {cty} {name}{comma}");
    }

    s.push_str(") {\n");

    for shared in &e.shared {
        let _ = writeln!(s, "    __shared__ float {}[{}];", shared.name, shared.len);
    }
    if uses_workgroup_reduce {
        let _ = writeln!(s, "    __shared__ float __zsl_wg_reduce_f32[{block_size}];");
    }

    let _ = writeln!(
        s,
        "    unsigned int zsl_gx = blockIdx.x * blockDim.x + threadIdx.x;"
    );
    let _ = writeln!(
        s,
        "    unsigned int zsl_gy = blockIdx.y * blockDim.y + threadIdx.y;"
    );
    let _ = writeln!(
        s,
        "    unsigned int zsl_gz = blockIdx.z * blockDim.z + threadIdx.z;"
    );

    let ctx = Ctx {
        locals,
        bf16_buffers,
        f16_buffers,
        type_ctx: ComputeTypeCtx {
            locals: e.locals.iter().cloned().collect(),
            scalars: scalars
                .iter()
                .map(|(name, ty)| ((*name).to_string(), *ty))
                .collect(),
            buffers: buffers
                .iter()
                .map(|(name, _, ty)| ((*name).to_string(), *ty))
                .collect(),
        },
        types: RefCell::new(module.types.clone()),
        block_size,
    };
    for st in &e.body {
        emit_stmt(&mut s, &ctx, st, 1)?;
    }

    s.push_str("}\n");
    if ctx.locals.values().any(|t| t.starts_with("nvcuda::")) {
        s.insert_str(kernel_start, frag::FRAG_PRELUDE);
    }

    Ok(CudaShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_scalars: !scalars.is_empty(),
        local_size,
    })
}

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
    bf16_buffers: HashSet<&'a str>,
    f16_buffers: HashSet<&'a str>,
    type_ctx: ComputeTypeCtx,
    types: RefCell<TypeArena>,
    block_size: u32,
}

fn expr_type_id(ctx: &Ctx<'_>, expr: &IrExpr) -> TypeId {
    let mut types = ctx.types.borrow_mut();
    type_of_compute_expr(expr, &ctx.type_ctx, &mut types)
}

fn expr_is_f32(ctx: &Ctx<'_>, expr: &IrExpr) -> bool {
    let ty = expr_type_id(ctx, expr);
    ty == ctx.types.borrow_mut().f32_id()
}

/// Mixed `i32`/`u32` operands unify to `i32`, as in SPIR-V. Bitwise and `&&`/`||` excluded.
fn is_mixed_sign_checked_op(op: IrBinOp) -> bool {
    matches!(
        op,
        IrBinOp::Add
            | IrBinOp::Sub
            | IrBinOp::Mul
            | IrBinOp::Div
            | IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
    )
}

/// Builtins SPIR-V rejects unless all operands are `f32`. HIP and CUDA must reject the same set.
fn builtin_requires_f32(func: BuiltinFn) -> bool {
    matches!(
        func,
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
            | BuiltinFn::IsNan
            | BuiltinFn::IsInf
            | BuiltinFn::IsFinite
            | BuiltinFn::Min
            | BuiltinFn::Max
            | BuiltinFn::Pow
            | BuiltinFn::Clamp
            | BuiltinFn::Mix
            | BuiltinFn::MulAddF32
    )
}

fn is_frag_builtin(f: BuiltinFn) -> bool {
    matches!(
        f,
        BuiltinFn::FragZeroAcc
            | BuiltinFn::Mma
            | BuiltinFn::FragLen
            | BuiltinFn::FragRow
            | BuiltinFn::FragCol
            | BuiltinFn::FragGet
            | BuiltinFn::FragSet
    )
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

/// CUDA spelling of a scalar [`Type`]. Non-scalar is an `Unsupported` diagnostic.
fn cuda_scalar(ty: &Type) -> Result<&'static str, Diagnostic> {
    match ty {
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => Ok("unsigned int"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => Ok("int"),
        Type::Float {
            format: FloatFormat::F32,
        } => Ok("float"),
        Type::Bool => Ok("bool"),
        other => Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!("CUDA backend: unsupported scalar type {other:?}"),
        )),
    }
}

/// CUDA spelling of a buffer element [`Type`]. `bf16` is `unsigned short`, bit-exact with the HIP
/// backend: round-to-nearest-even on store, zero-extend on load.
fn cuda_buffer_elem(ty: &Type) -> Result<&'static str, Diagnostic> {
    match ty {
        Type::Float {
            format: FloatFormat::F32,
        } => Ok("float"),
        Type::Float {
            format: FloatFormat::Bf16,
        } => Ok("unsigned short"),
        Type::Float {
            format: FloatFormat::F16,
        } => Ok("__half"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => Ok("unsigned int"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => Ok("int"),
        other => Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!("CUDA backend: unsupported compute buffer element {other:?}"),
        )),
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) -> Result<(), Diagnostic> {
    match stmt {
        IrStmt::Let { name, init } => {
            let ty = ctx
                .locals
                .get(name.as_str())
                .copied()
                .unwrap_or("unsigned int");
            indent(s, depth);
            if frag::is_frag_statement(init) {
                let _ = writeln!(s, "{}", frag::let_stmt(ctx, ty, name, init)?);
            } else {
                let _ = writeln!(s, "{} {} = {};", ty, name, emit_expr(ctx, init)?);
            }
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            if frag::is_frag_statement(value) {
                let _ = writeln!(s, "{}", frag::assign_stmt(ctx, name, value)?);
            } else {
                let _ = writeln!(s, "{} = {};", name, emit_expr(ctx, value)?);
            }
        }
        IrStmt::FragStore {
            src,
            offset,
            ld,
            value,
        } => {
            indent(s, depth);
            let _ = writeln!(s, "{}", frag::store_stmt(ctx, src, offset, ld, value)?);
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            indent(s, depth);
            let idx = emit_expr(ctx, index)?;
            let val = emit_expr(ctx, value)?;
            let val = if ctx.bf16_buffers.contains(buf.as_str()) {
                format!("f32_to_bf16({val})")
            } else if ctx.f16_buffers.contains(buf.as_str()) {
                format!("__float2half({val})")
            } else {
                val
            };
            let _ = writeln!(s, "{buf}[{idx}] = {val};");
        }
        IrStmt::AtomicAdd { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "atomicAdd(&{}[(unsigned int)({})], {});",
                buf,
                emit_expr(ctx, index)?,
                emit_expr(ctx, value)?
            );
        }
        IrStmt::AssignShared { name, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "{}[{}] = {};",
                name,
                emit_expr(ctx, index)?,
                emit_expr(ctx, value)?
            );
        }
        IrStmt::Discard => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "discard is fragment-only and never reaches a compute backend",
            ));
        }
        IrStmt::ImageStore { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "kernels with image params are rejected before emission",
            ));
        }
        IrStmt::Barrier => {
            indent(s, depth);
            s.push_str("__syncthreads();\n");
        }
        IrStmt::If { cond, then, else_ } => {
            indent(s, depth);
            let _ = writeln!(s, "if ({}) {{", emit_expr(ctx, cond)?);
            for st in then {
                emit_stmt(s, ctx, st, depth + 1)?;
            }
            if let Some(els) = else_ {
                indent(s, depth);
                s.push_str("} else {\n");
                for st in els {
                    emit_stmt(s, ctx, st, depth + 1)?;
                }
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::For { var, lo, hi, body } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "for (unsigned int {var} = {}; {var} < {}; {var}++) {{",
                emit_expr(ctx, lo)?,
                emit_expr(ctx, hi)?,
            );
            for st in body {
                emit_stmt(s, ctx, st, depth + 1)?;
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::Eval(e) => {
            indent(s, depth);
            let _ = writeln!(s, "{};", emit_expr(ctx, e)?);
        }
    }
    Ok(())
}

fn emit_expr(ctx: &Ctx<'_>, e: &IrExpr) -> Result<String, Diagnostic> {
    Ok(match e {
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => {
            if v.is_nan() {
                "__builtin_nanf(\"\")".to_string()
            } else if *v == f32::INFINITY {
                "__builtin_inff()".to_string()
            } else if *v == f32::NEG_INFINITY {
                "(-__builtin_inff())".to_string()
            } else if v.fract() == 0.0 {
                format!("{v}.0f")
            } else {
                format!("{v}f")
            }
        }
        IrExpr::Local(n) => n.clone(),
        IrExpr::ScalarParam(n) => n.clone(),
        IrExpr::BufferLoad { buf, index } => {
            let inner = format!("{}[{}]", buf, emit_expr(ctx, index)?);
            if ctx.bf16_buffers.contains(buf.as_str()) {
                format!("bf16_to_f32({inner})")
            } else if ctx.f16_buffers.contains(buf.as_str()) {
                format!("__half2float({inner})")
            } else {
                inner
            }
        }
        IrExpr::SharedLoad { name, index } => format!("{}[{}]", name, emit_expr(ctx, index)?),
        IrExpr::GlobalId(0) => "zsl_gx".into(),
        IrExpr::GlobalId(1) => "zsl_gy".into(),
        IrExpr::GlobalId(2) => "zsl_gz".into(),
        IrExpr::GlobalId(n) => format!("/* bad GlobalId({n}) */ 0u"),
        IrExpr::LocalId(c) => format!("threadIdx.{}", ["x", "y", "z"][*c as usize]),
        IrExpr::GroupId(c) => format!("blockIdx.{}", ["x", "y", "z"][*c as usize]),
        IrExpr::Input(n) => n.clone(),
        IrExpr::FieldAccess { base, component } => {
            let field = ["x", "y", "z", "w"]
                .get(*component as usize)
                .copied()
                .unwrap_or("x");
            format!("{}.{}", emit_expr(ctx, base)?, field)
        }
        IrExpr::Swizzle { .. }
        | IrExpr::Sample { .. }
        | IrExpr::ArrayLit(_)
        | IrExpr::Index { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "swizzles, arrays and texture samples are graphics-only and never reach a compute backend",
            ));
        }
        IrExpr::Texel { .. } | IrExpr::TexSize { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "kernels with texture params are rejected before emission",
            ));
        }
        IrExpr::Construct { dim, args, .. } => {
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            format!("make_float{}({})", dim, args_s.join(", "))
        }
        IrExpr::FragLoad { .. } => return Err(frag::expr_error("frag_load")),
        IrExpr::Builtin {
            func: func @ (BuiltinFn::FragZeroAcc | BuiltinFn::Mma),
            ..
        } => return Err(frag::expr_error(func.name())),
        IrExpr::Builtin { func, .. } if is_frag_builtin(*func) => {
            return Err(Diagnostic::new(
                DiagnosticCode::Unsupported,
                format!(
                    "CUDA backend: {} has no lowering (wmma fragment element mapping is opaque)",
                    func.name()
                ),
            ));
        }
        IrExpr::Builtin { func, args } => {
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            if builtin_requires_f32(*func) && args.iter().any(|a| !expr_is_f32(ctx, a)) {
                return Err(Diagnostic::new(
                    DiagnosticCode::Unsupported,
                    format!("ZSL: {}() requires f32 operands", func.name()),
                ));
            }
            match func {
                BuiltinFn::FragZeroAcc
                | BuiltinFn::Mma
                | BuiltinFn::FragLen
                | BuiltinFn::FragRow
                | BuiltinFn::FragCol
                | BuiltinFn::FragGet
                | BuiltinFn::FragSet => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Internal,
                        "fragment builtin reached the scalar builtin path",
                    ));
                }
                BuiltinFn::Dot => format!("dot({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Extend => format!("make_float4({}, {})", args_s[0], args_s[1]),
                BuiltinFn::U32 => format!("((unsigned int)({}))", args_s[0]),
                BuiltinFn::F32 => format!("((float)({}))", args_s[0]),
                BuiltinFn::Step | BuiltinFn::Smoothstep => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Internal,
                        "step and smoothstep are graphics-only",
                    ));
                }
                BuiltinFn::Abs => format!("fabsf({})", args_s[0]),
                BuiltinFn::Sign => format!("(({0} > 0.0f) - ({0} < 0.0f))", args_s[0]),
                BuiltinFn::Exp => format!("expf({})", args_s[0]),
                BuiltinFn::Tanh => format!("tanhf({})", args_s[0]),
                BuiltinFn::Sin => format!("sinf({})", args_s[0]),
                BuiltinFn::Cos => format!("cosf({})", args_s[0]),
                BuiltinFn::Tan => format!("tanf({})", args_s[0]),
                BuiltinFn::IsNan => format!("isnan({})", args_s[0]),
                BuiltinFn::IsInf => format!("isinf({})", args_s[0]),
                BuiltinFn::IsFinite => format!("isfinite({})", args_s[0]),
                BuiltinFn::Log => format!("logf({})", args_s[0]),
                BuiltinFn::Sqrt => format!("sqrtf({})", args_s[0]),
                BuiltinFn::Floor => format!("floorf({})", args_s[0]),
                BuiltinFn::Ceil => format!("ceilf({})", args_s[0]),
                BuiltinFn::Fract => format!("({0} - floorf({0}))", args_s[0]),
                BuiltinFn::Pow => format!("powf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Min => format!("fminf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Max => format!("fmaxf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Clamp => {
                    format!("fminf(fmaxf({}, {}), {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::Mix => format!(
                    "({0} + ({2}) * ({1} - {0}))",
                    args_s[0], args_s[1], args_s[2]
                ),
                BuiltinFn::Normalize => {
                    format!(
                        "(({0} > 0.0f) ? 1.0f : ({0} < 0.0f) ? -1.0f : 0.0f)",
                        args_s[0]
                    )
                }
                BuiltinFn::Length => format!("sqrtf(dot({0}, {0}))", args_s[0]),
                BuiltinFn::MulAddF32 => {
                    format!("fmaf({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::Dot4I8 => {
                    format!("zsl_dot4_i8({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::Dot4U8 => {
                    format!("zsl_dot4_u8({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::Dot4IU8 => {
                    format!("zsl_dot4_iu8({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::PkAddU16 => format!("zsl_pk_add_u16({}, {})", args_s[0], args_s[1]),
                BuiltinFn::PkMulU16 => format!("zsl_pk_mul_u16({}, {})", args_s[0], args_s[1]),
                BuiltinFn::PkMadU16 => {
                    format!(
                        "zsl_pk_mad_u16({}, {}, {})",
                        args_s[0], args_s[1], args_s[2]
                    )
                }
                BuiltinFn::PkAddF16 => format!("zsl_pk_add_f16({}, {})", args_s[0], args_s[1]),
                BuiltinFn::PkMulF16 => format!("zsl_pk_mul_f16({}, {})", args_s[0], args_s[1]),
                BuiltinFn::PkFmaF16 => {
                    format!(
                        "zsl_pk_fma_f16({}, {}, {})",
                        args_s[0], args_s[1], args_s[2]
                    )
                }
                BuiltinFn::Dot2F16 => {
                    format!("zsl_dot2_f16({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::FunnelShr => {
                    format!(
                        "zsl_funnel_shr({}, {}, {})",
                        args_s[0], args_s[1], args_s[2]
                    )
                }
                BuiltinFn::BytePerm => {
                    format!("zsl_byte_perm({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::SubgroupAny => {
                    format!("((bool)__any_sync(0xffffffffu, (int)({})))", args_s[0])
                }
                BuiltinFn::SubgroupAll => {
                    format!("((bool)__all_sync(0xffffffffu, (int)({})))", args_s[0])
                }
                BuiltinFn::SubgroupBallot => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Unsupported,
                        "CUDA backend: subgroup_ballot() is not yet supported \
                         (needs a 128-bit-capable value representation)",
                    ));
                }
                BuiltinFn::SubgroupBroadcast | BuiltinFn::SubgroupShuffle => {
                    format!(
                        "__shfl_sync(0xffffffffu, {}, (int)({}))",
                        args_s[0], args_s[1]
                    )
                }
                BuiltinFn::SubgroupShuffleXor => {
                    format!(
                        "__shfl_xor_sync(0xffffffffu, {}, (int)({}))",
                        args_s[0], args_s[1]
                    )
                }
                BuiltinFn::SubgroupShuffleUp => {
                    format!(
                        "__shfl_up_sync(0xffffffffu, {}, (unsigned int)({}))",
                        args_s[0], args_s[1]
                    )
                }
                BuiltinFn::SubgroupShuffleDown => {
                    format!(
                        "__shfl_down_sync(0xffffffffu, {}, (unsigned int)({}))",
                        args_s[0], args_s[1]
                    )
                }
                BuiltinFn::SubgroupReduceAdd
                | BuiltinFn::SubgroupReduceMin
                | BuiltinFn::SubgroupReduceMax
                | BuiltinFn::SubgroupReduceAnd
                | BuiltinFn::SubgroupReduceOr => {
                    let op = match func {
                        BuiltinFn::SubgroupReduceAdd => "ZslAddOp",
                        BuiltinFn::SubgroupReduceMin => "ZslMinOp",
                        BuiltinFn::SubgroupReduceMax => "ZslMaxOp",
                        BuiltinFn::SubgroupReduceAnd => "ZslAndOp",
                        BuiltinFn::SubgroupReduceOr => "ZslOrOp",
                        _ => unreachable!(),
                    };
                    format!("zsl_sg_xor_reduce({}, {op}())", args_s[0])
                }
                BuiltinFn::SubgroupScanInclusiveAdd => {
                    format!("zsl_sg_scan_incl_add({})", args_s[0])
                }
                BuiltinFn::SubgroupScanExclusiveAdd => {
                    format!("zsl_sg_scan_excl_add({})", args_s[0])
                }
                BuiltinFn::WorkgroupReduceAdd
                | BuiltinFn::WorkgroupReduceMin
                | BuiltinFn::WorkgroupReduceMax => {
                    let op = match func {
                        BuiltinFn::WorkgroupReduceAdd => "ZslAddOp",
                        BuiltinFn::WorkgroupReduceMin => "ZslMinOp",
                        BuiltinFn::WorkgroupReduceMax => "ZslMaxOp",
                        _ => unreachable!(),
                    };
                    format!(
                        "zsl_wg_reduce(__zsl_wg_reduce_f32, {}u, {}, {op}())",
                        ctx.block_size, args_s[0]
                    )
                }
                BuiltinFn::SubgroupSize => "((unsigned int)warpSize)".to_string(),
                BuiltinFn::SubgroupLocalId => "(threadIdx.x % warpSize)".to_string(),
                BuiltinFn::SubgroupId => "(threadIdx.x / warpSize)".to_string(),
            }
        }
        IrExpr::Neg(inner) => format!("(-{})", emit_expr(ctx, inner)?),
        IrExpr::Binary { op, lhs, rhs } => {
            let op_s = match op {
                IrBinOp::Add => "+",
                IrBinOp::Sub => "-",
                IrBinOp::Mul => "*",
                IrBinOp::Div => "/",
                IrBinOp::Lt => "<",
                IrBinOp::Le => "<=",
                IrBinOp::Gt => ">",
                IrBinOp::Ge => ">=",
                IrBinOp::Eq => "==",
                IrBinOp::Ne => "!=",
                IrBinOp::And => "&&",
                IrBinOp::Or => "||",
                IrBinOp::BitAnd => "&",
                IrBinOp::BitOr => "|",
                IrBinOp::BitXor => "^",
                IrBinOp::Shl => "<<",
                IrBinOp::Shr => ">>",
            };
            let mut l = emit_expr(ctx, lhs)?;
            let mut r = emit_expr(ctx, rhs)?;
            if is_mixed_sign_checked_op(*op) {
                let (lt, rt) = (expr_type_id(ctx, lhs), expr_type_id(ctx, rhs));
                let (i32_id, u32_id) = {
                    let mut types = ctx.types.borrow_mut();
                    (types.i32_id(), types.u32_id())
                };
                if lt == i32_id && rt == u32_id {
                    r = format!("((int)({r}))");
                } else if lt == u32_id && rt == i32_id {
                    l = format!("((int)({l}))");
                }
            }
            format!("({l} {op_s} {r})")
        }
    })
}
