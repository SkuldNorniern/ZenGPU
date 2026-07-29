use std::collections::HashMap;
use std::fmt::Write;

use crate::backend::hip::{ENTRY, HipShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type, TypeId};
use crate::ir::{EntryKind, Module, Mutability, ParamKind};

pub fn lower_compute(module: &Module) -> Result<HipShader, Diagnostic> {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    for (n, t) in &e.locals {
        locals.insert(n.as_str(), hip_scalar(module.types.get(*t))?);
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

    let bf16_id = { let mut t = module.types.clone(); t.bf16_id() };
    let bf16_buffers: std::collections::HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == bf16_id)
        .map(|(name, _, _)| *name)
        .collect();

    let f16_id = { let mut t = module.types.clone(); t.f16_id() };
    let f16_buffers: std::collections::HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == f16_id)
        .map(|(name, _, _)| *name)
        .collect();

    let mut s = String::new();

    if !f16_buffers.is_empty() {
        s.push_str("#include <hip/hip_fp16.h>\n");
    }
    if !bf16_buffers.is_empty() {
        s.push_str(
            "__device__ __forceinline__ unsigned short f32_to_bf16(float x) {\n\
             \x20   unsigned int bits = __float_as_uint(x);\n\
             \x20   unsigned int bias = 0x7fffu + ((bits >> 16) & 1u);\n\
             \x20   return (unsigned short)((bits + bias) >> 16);\n\
             }\n\
             __device__ __forceinline__ float bf16_to_f32(unsigned short x) {\n\
             \x20   return __uint_as_float(((unsigned int)x) << 16);\n\
             }\n",
        );
    }

    let threads = local_size[0] * local_size[1] * local_size[2];
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
                let ty = hip_buffer_elem(module.types.get(*elem))?;
                let _ = writeln!(s, "    const {ty}* __restrict__ {name}{comma}");
            }
            Mutability::ReadWrite => {
                let ty = hip_buffer_elem(module.types.get(*elem))?;
                let _ = writeln!(s, "    {ty}* __restrict__ {name}{comma}");
            }
        }
    }

    for (i, (name, ty)) in scalars.iter().enumerate() {
        let comma = if i + 1 < scalars.len() { "," } else { "" };
        let cty = hip_scalar(module.types.get(*ty))?;
        let _ = writeln!(s, "    {cty} {name}{comma}");
    }

    s.push_str(") {\n");

    for shared in &e.shared {
        let _ = writeln!(s, "    __shared__ float {}[{}];", shared.name, shared.len);
    }

    let _ = writeln!(
        s,
        "    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;"
    );
    if local_size[1] > 1 || local_size[2] > 1 {
        let _ = writeln!(
            s,
            "    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;"
        );
        let _ = writeln!(
            s,
            "    unsigned int gz = blockIdx.z * blockDim.z + threadIdx.z;"
        );
    }

    let ctx = Ctx {
        locals,
        bf16_buffers,
        f16_buffers,
    };
    for st in &e.body {
        emit_stmt(&mut s, &ctx, st, 1)?;
    }

    s.push_str("}\n");

    Ok(HipShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_scalars: !scalars.is_empty(),
        local_size,
    })
}

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
    bf16_buffers: std::collections::HashSet<&'a str>,
    f16_buffers: std::collections::HashSet<&'a str>,
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

/// HIP spelling of a scalar [`Type`]. Non-scalar is an `Unsupported` diagnostic.
fn hip_scalar(ty: &Type) -> Result<&'static str, Diagnostic> {
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
            format!("HIP backend: unsupported scalar type {other:?}"),
        )),
    }
}

/// HIP spelling of a buffer element [`Type`]. `bf16` is `unsigned short`, bit-exact with
/// `laminax-runtime`: round-to-nearest-even on store, zero-extend on load.
fn hip_buffer_elem(ty: &Type) -> Result<&'static str, Diagnostic> {
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
            format!("HIP backend: unsupported compute buffer element {other:?}"),
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
            let _ = writeln!(s, "{} {} = {};", ty, name, emit_expr(ctx, init)?);
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{} = {};", name, emit_expr(ctx, value)?);
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
            if v.fract() == 0.0 {
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
        IrExpr::GlobalId(0) => "gx".into(),
        IrExpr::GlobalId(1) => "gy".into(),
        IrExpr::GlobalId(2) => "gz".into(),
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
        IrExpr::Construct { dim, args, .. } => {
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            format!("float{}({})", dim, args_s.join(", "))
        }
        IrExpr::Builtin { func, args } => {
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            match func {
                BuiltinFn::Dot => format!("dot({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Extend => format!("float4({}, {})", args_s[0], args_s[1]),
                BuiltinFn::U32 => format!("((unsigned int)({}))", args_s[0]),
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
                BuiltinFn::Normalize => format!("__hip_normalize_stub({})", args_s[0]),
                BuiltinFn::Length => format!("sqrtf(dot({0}, {0}))", args_s[0]),
                BuiltinFn::MulAddF32 => {
                    format!("fmaf({}, {}, {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::SubgroupSize => "((unsigned int)warpSize)".to_string(),
                BuiltinFn::SubgroupLocalId => "__lane_id()".to_string(),
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
                IrBinOp::BitAnd
                | IrBinOp::BitOr
                | IrBinOp::BitXor
                | IrBinOp::Shl
                | IrBinOp::Shr => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Unsupported,
                        "HIP backend: bitwise operators (&, |, ^, <<, >>) are not yet supported; \
                         this ZSL source is SPIR-V-only today",
                    ));
                }
            };
            format!("({} {} {})", emit_expr(ctx, lhs)?, op_s, emit_expr(ctx, rhs)?)
        }
    })
}
