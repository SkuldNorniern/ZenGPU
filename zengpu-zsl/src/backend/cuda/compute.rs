use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write;

use crate::backend::cuda::{CudaShader, ENTRY};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::typeck::{type_of_compute_expr, ComputeTypeCtx};
use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type, TypeId};
use crate::ir::{EntryKind, Module, Mutability, ParamKind};

pub fn lower_compute(module: &Module) -> Result<CudaShader, Diagnostic> {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    // Locals carry their pre-resolved C-type spelling so `emit_stmt` needs no
    // arena access. Every local is a scalar today; a non-scalar `TypeId` here
    // surfaces as a clean `Diagnostic` rather than reaching codegen.
    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    for (n, t) in &e.locals {
        locals.insert(n.as_str(), cuda_scalar(module.types.get(*t))?);
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

    // Buffers whose element is `bf16`: every load/store on these goes through
    // `bf16_to_f32`/`f32_to_bf16` — storage is `unsigned short`, every value
    // that reaches an expression is `f32` (there is no bf16 arithmetic today,
    // per the design doc's storage/arithmetic capability split).
    let bf16_id = { let mut t = module.types.clone(); t.bf16_id() };
    let bf16_buffers: std::collections::HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == bf16_id)
        .map(|(name, _, _)| *name)
        .collect();

    // Buffers whose element is `f16`: storage is the real `__half` type
    // (`<cuda_fp16.h>`), converted via the compiler/hardware's own
    // `__float2half`/`__half2float` — no hand-rolled bit algorithm needed,
    // unlike bf16 (which has no dedicated hardware conversion path and must
    // match `laminax-runtime`'s existing raw-bit algorithm bit-for-bit).
    let f16_id = { let mut t = module.types.clone(); t.f16_id() };
    let f16_buffers: std::collections::HashSet<&str> = buffers
        .iter()
        .filter(|(_, _, elem)| *elem == f16_id)
        .map(|(name, _, _)| *name)
        .collect();
    let uses_workgroup_reduce = crate::ir::node::stmts_use_workgroup_reduce(&e.body);
    let block_size = local_size[0] * local_size[1] * local_size[2];
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
        let _ = writeln!(
            s,
            "    __shared__ float __zsl_wg_reduce_f32[{block_size}];"
        );
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

    Ok(CudaShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_scalars: !scalars.is_empty(),
        local_size,
    })
}

struct Ctx<'a> {
    /// Local name → its resolved C-type spelling.
    locals: HashMap<&'a str, &'static str>,
    /// Buffer names whose element is `bf16` — every load/store on these is
    /// wrapped in `bf16_to_f32`/`f32_to_bf16`.
    bf16_buffers: std::collections::HashSet<&'a str>,
    /// Buffer names whose element is `f16` — every load/store on these is
    /// wrapped in `__half2float`/`__float2half`.
    f16_buffers: std::collections::HashSet<&'a str>,
    /// Resolved expression symbols used only to select type-specific subgroup
    /// reduction spellings and the exclusive-scan identity literal.
    type_ctx: ComputeTypeCtx,
    types: RefCell<TypeArena>,
    /// Total workgroup thread count used to unroll workgroup reductions.
    block_size: u32,
}

fn expr_is_f32(ctx: &Ctx<'_>, expr: &IrExpr) -> bool {
    let mut types = ctx.types.borrow_mut();
    let ty = type_of_compute_expr(expr, &ctx.type_ctx, &mut types);
    ty == types.f32_id()
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

/// The CUDA C++ spelling of a scalar [`Type`]. Non-scalar types are a
/// compiler defect for a compute value and surface as an `Unsupported`
/// diagnostic.
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

/// The CUDA C++ spelling of a `device buffer<T>` element [`Type`]. `bf16` is
/// stored as `unsigned short` (matching `laminax-runtime`'s raw-HIP bf16
/// kernels and this crate's HIP backend bit-for-bit — same round-to-nearest-
/// even truncation on store, same zero-extend-and-reinterpret on load).
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
            format!("make_float{}({})", dim, args_s.join(", "))
        }
        IrExpr::Builtin { func, args } => {
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            match func {
                BuiltinFn::Dot => format!("dot({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Extend => format!("make_float4({}, {})", args_s[0], args_s[1]),
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
                BuiltinFn::Normalize => {
                    // CUDA has no built-in normalize for scalar floats; the sign of a
                    // non-zero float is ±1, zero stays zero.
                    format!(
                        "(({0} > 0.0f) ? 1.0f : ({0} < 0.0f) ? -1.0f : 0.0f)",
                        args_s[0]
                    )
                }
                BuiltinFn::Length => format!("sqrtf(dot({0}, {0}))", args_s[0]),
                BuiltinFn::MulAddF32 => {
                    format!("fmaf({}, {}, {})", args_s[0], args_s[1], args_s[2])
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
                    let combine = match func {
                        BuiltinFn::SubgroupReduceAdd => "v + o",
                        BuiltinFn::SubgroupReduceMin if expr_is_f32(ctx, &args[0]) => "fminf(v, o)",
                        BuiltinFn::SubgroupReduceMin => "(v < o ? v : o)",
                        BuiltinFn::SubgroupReduceMax if expr_is_f32(ctx, &args[0]) => "fmaxf(v, o)",
                        BuiltinFn::SubgroupReduceMax => "(v > o ? v : o)",
                        BuiltinFn::SubgroupReduceAnd => "(v & o)",
                        BuiltinFn::SubgroupReduceOr => "(v | o)",
                        _ => unreachable!(),
                    };
                    format!(
                        "({{ auto v = {}; for (unsigned int off = warpSize / 2; off > 0; off /= 2) {{ auto o = __shfl_xor_sync(0xffffffffu, v, off); v = {combine}; }} v; }})",
                        args_s[0]
                    )
                }
                BuiltinFn::SubgroupScanInclusiveAdd => format!(
                    "({{ auto v = {}; for (unsigned int off = 1; off < warpSize; off *= 2) {{ auto o = __shfl_up_sync(0xffffffffu, v, off); if ((threadIdx.x % warpSize) >= off) v = v + o; }} v; }})",
                    args_s[0]
                ),
                BuiltinFn::SubgroupScanExclusiveAdd => {
                    let zero = if expr_is_f32(ctx, &args[0]) {
                        "0.0f"
                    } else {
                        "0u"
                    };
                    format!(
                        "({{ auto v = {}; for (unsigned int off = 1; off < warpSize; off *= 2) {{ auto o = __shfl_up_sync(0xffffffffu, v, off); if ((threadIdx.x % warpSize) >= off) v = v + o; }} auto incl = v; auto excl = __shfl_up_sync(0xffffffffu, incl, 1u); if ((threadIdx.x % warpSize) == 0) excl = {zero}; excl; }})",
                        args_s[0]
                    )
                }
                BuiltinFn::WorkgroupReduceAdd
                | BuiltinFn::WorkgroupReduceMin
                | BuiltinFn::WorkgroupReduceMax => {
                    // Assumes a 1D workgroup (local_size[1] == local_size[2] ==
                    // 1, true of every kernel in this codebase today) — a
                    // multi-dimensional workgroup would need
                    // threadIdx.x + threadIdx.y*sizeX + threadIdx.z*sizeX*sizeY.
                    let combine = match func {
                        BuiltinFn::WorkgroupReduceAdd => "v + o",
                        BuiltinFn::WorkgroupReduceMin => "fminf(v, o)",
                        BuiltinFn::WorkgroupReduceMax => "fmaxf(v, o)",
                        _ => unreachable!(),
                    };
                    let mut steps = String::new();
                    let mut s = ctx.block_size / 2;
                    while s > 0 {
                        steps.push_str(&format!(
                            "if (threadIdx.x < {s}u) {{ auto o = __zsl_wg_reduce_f32[threadIdx.x + {s}u]; auto v = __zsl_wg_reduce_f32[threadIdx.x]; __zsl_wg_reduce_f32[threadIdx.x] = {combine}; }} __syncthreads(); "
                        ));
                        s /= 2;
                    }
                    format!(
                        "({{ __zsl_wg_reduce_f32[threadIdx.x] = {}; __syncthreads(); {steps} __zsl_wg_reduce_f32[0]; }})",
                        args_s[0]
                    )
                }
                BuiltinFn::SubgroupSize => "((unsigned int)warpSize)".to_string(),
                // No real NVIDIA hardware in this repo to verify an inline-PTX
                // `%laneid` read against, so lane/subgroup index are derived
                // arithmetically from `threadIdx.x` instead — correct for any
                // 1D dispatch (true of every kernel in this codebase today;
                // see the identical assumption in the HIP backend).
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
                // Not yet supported on this backend; see the SPIR-V-only note
                // on `IrBinOp`. Previously emitted a deliberately invalid
                // marker comment so a downstream C++ compile would fail
                // loudly; now fails ZSL compilation itself instead.
                IrBinOp::BitAnd
                | IrBinOp::BitOr
                | IrBinOp::BitXor
                | IrBinOp::Shl
                | IrBinOp::Shr => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Unsupported,
                        "CUDA backend: bitwise operators (&, |, ^, <<, >>) are not yet supported; \
                         this ZSL source is SPIR-V-only today",
                    ));
                }
            };
            format!("({} {} {})", emit_expr(ctx, lhs)?, op_s, emit_expr(ctx, rhs)?)
        }
    })
}
