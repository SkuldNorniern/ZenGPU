
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type, TypeId};
use crate::ir::{EntryKind, Module, ParamKind};

pub fn lower_compute(module: &Module) -> Result<MslShader, Diagnostic> {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    for (n, t) in &e.locals {
        locals.insert(n.as_str(), msl_scalar(module.types.get(*t))?);
    }
    let buffers: Vec<(&str, TypeId)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some((p.name.as_str(), *elem)),
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
    let bf16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, elem)| *elem == bf16_id)
        .map(|(name, _)| *name)
        .collect();

    let f16_id = { let mut t = module.types.clone(); t.f16_id() };
    let f16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, elem)| *elem == f16_id)
        .map(|(name, _)| *name)
        .collect();

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    if !bf16_buffers.is_empty() {
        s.push_str(
            "inline ushort f32_to_bf16(float x) {\n\
             \x20   uint bits = as_type<uint>(x);\n\
             \x20   uint bias = 0x7fffu + ((bits >> 16) & 1u);\n\
             \x20   return (ushort)((bits + bias) >> 16);\n\
             }\n\
             inline float bf16_to_f32(ushort x) {\n\
             \x20   return as_type<float>(uint(x) << 16);\n\
             }\n\n",
        );
    }
    let mut atomic_buffers = HashSet::new();
    collect_atomic_buffers(&e.body, &mut atomic_buffers);

    let has_push = !scalars.is_empty();
    if has_push {
        s.push_str("struct Push {\n");
        for (n, t) in &scalars {
            let _ = writeln!(s, "    {} {};", msl_scalar(module.types.get(*t))?, n);
        }
        s.push_str("};\n\n");
    }

    let _ = writeln!(s, "kernel void {ENTRY}(");
    let mut slot = 0u32;
    for (b, buffer_elem) in &buffers {
        let elem = if atomic_buffers.contains(b) {
            "atomic_float"
        } else {
            msl_buffer_elem(module.types.get(*buffer_elem))?
        };
        let _ = writeln!(s, "    device {elem}* {b} [[buffer({slot})]],");
        slot += 1;
    }
    if has_push {
        let _ = writeln!(s, "    constant Push& pc [[buffer({slot})]],");
    }
    s.push_str("    uint3 gid [[thread_position_in_grid]],\n");
    s.push_str("    uint3 lid [[thread_position_in_threadgroup]],\n");
    s.push_str("    uint3 group [[threadgroup_position_in_grid]]");
    let uses_subgroup_query = crate::ir::node::stmts_use_subgroup_query(&e.body);
    if uses_subgroup_query {
        s.push_str(",\n    uint simd_size [[threads_per_simdgroup]],\n");
        s.push_str("    uint simd_group [[simdgroup_index_in_threadgroup]],\n");
        s.push_str("    uint simd_lane [[thread_index_in_simdgroup]]\n) {\n");
    } else {
        s.push_str("\n) {\n");
    }

    for shared in &e.shared {
        let _ = writeln!(s, "    threadgroup float {}[{}];", shared.name, shared.len);
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

    Ok(MslShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_push,
        local_size,
    })
}

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
    bf16_buffers: HashSet<&'a str>,
    f16_buffers: HashSet<&'a str>,
}

fn collect_atomic_buffers<'a>(stmts: &'a [IrStmt], buffers: &mut HashSet<&'a str>) {
    for stmt in stmts {
        match stmt {
            IrStmt::AtomicAdd { buf, .. } => {
                buffers.insert(buf);
            }
            IrStmt::If { then, else_, .. } => {
                collect_atomic_buffers(then, buffers);
                if let Some(else_) = else_ {
                    collect_atomic_buffers(else_, buffers);
                }
            }
            IrStmt::For { body, .. } => collect_atomic_buffers(body, buffers),
            _ => {}
        }
    }
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) -> Result<(), Diagnostic> {
    match stmt {
        IrStmt::Let { name, init } => {
            let ty = ctx
                .locals
                .get(name.as_str())
                .copied()
                .unwrap_or("uint");
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
                format!("half({val})")
            } else {
                val
            };
            let _ = writeln!(s, "{buf}[{idx}] = {val};");
        }
        IrStmt::AtomicAdd { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "atomic_fetch_add_explicit(&{}[uint({})], {}, memory_order_relaxed);",
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
            s.push_str("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        }
        IrStmt::If { cond, then, else_ } => {
            indent(s, depth);
            let _ = writeln!(s, "if ({}) {{", emit_expr(ctx, cond)?);
            for st in then {
                emit_stmt(s, ctx, st, depth + 1)?;
            }
            indent(s, depth);
            s.push('}');
            if let Some(else_block) = else_ {
                s.push_str(" else {\n");
                for st in else_block {
                    emit_stmt(s, ctx, st, depth + 1)?;
                }
                indent(s, depth);
                s.push('}');
            }
            s.push('\n');
        }
        IrStmt::For { var, lo, hi, body } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "for (uint {var} = {}; {var} < {}; {var}++) {{",
                emit_expr(ctx, lo)?,
                emit_expr(ctx, hi)?
            );
            for st in body {
                emit_stmt(s, ctx, st, depth + 1)?;
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::Eval(expr) => {
            indent(s, depth);
            let _ = writeln!(s, "{};", emit_expr(ctx, expr)?);
        }
    }
    Ok(())
}

fn emit_expr(ctx: &Ctx<'_>, expr: &IrExpr) -> Result<String, Diagnostic> {
    Ok(match expr {
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => msl_float(*v),
        IrExpr::Local(n) => n.clone(),
        IrExpr::ScalarParam(n) => format!("pc.{n}"),
        IrExpr::BufferLoad { buf, index } => {
            let inner = format!("{buf}[{}]", emit_expr(ctx, index)?);
            if ctx.bf16_buffers.contains(buf.as_str()) {
                format!("bf16_to_f32({inner})")
            } else if ctx.f16_buffers.contains(buf.as_str()) {
                format!("float({inner})")
            } else {
                inner
            }
        }
        IrExpr::SharedLoad { name, index } => format!("{name}[{}]", emit_expr(ctx, index)?),
        IrExpr::GlobalId(c) => format!("gid.{}", component(*c)),
        IrExpr::LocalId(c) => format!("lid.{}", component(*c)),
        IrExpr::GroupId(c) => format!("group.{}", component(*c)),
        IrExpr::Builtin { func, args } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(ctx, arg)?);
            }
            match func {
                BuiltinFn::U32 => format!("uint({})", a[0]),
                BuiltinFn::MulAddF32 => format!("fma({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::SubgroupAny => format!("simd_any({})", a[0]),
                BuiltinFn::SubgroupAll => format!("simd_all({})", a[0]),
                BuiltinFn::SubgroupBallot => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Unsupported,
                        "MSL backend: subgroup_ballot() is not yet supported \
                         (needs a 128-bit-capable value representation)",
                    ));
                }
                BuiltinFn::SubgroupBroadcast => format!("simd_broadcast({}, {})", a[0], a[1]),
                BuiltinFn::SubgroupShuffle => format!("simd_shuffle({}, {})", a[0], a[1]),
                BuiltinFn::SubgroupShuffleXor => format!("simd_shuffle_xor({}, {})", a[0], a[1]),
                BuiltinFn::SubgroupShuffleUp => format!("simd_shuffle_up({}, {})", a[0], a[1]),
                BuiltinFn::SubgroupShuffleDown => {
                    format!("simd_shuffle_down({}, {})", a[0], a[1])
                }
                BuiltinFn::SubgroupReduceAdd => format!("simd_sum({})", a[0]),
                BuiltinFn::SubgroupReduceMin => format!("simd_min({})", a[0]),
                BuiltinFn::SubgroupReduceMax => format!("simd_max({})", a[0]),
                BuiltinFn::SubgroupReduceAnd => format!("simd_and({})", a[0]),
                BuiltinFn::SubgroupReduceOr => format!("simd_or({})", a[0]),
                BuiltinFn::SubgroupScanInclusiveAdd => {
                    format!("simd_prefix_inclusive_sum({})", a[0])
                }
                BuiltinFn::SubgroupScanExclusiveAdd => {
                    format!("simd_prefix_exclusive_sum({})", a[0])
                }
                BuiltinFn::SubgroupSize => "simd_size".to_string(),
                BuiltinFn::SubgroupId => "simd_group".to_string(),
                BuiltinFn::SubgroupLocalId => "simd_lane".to_string(),
                BuiltinFn::Dot | BuiltinFn::Extend => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Internal,
                        "MSL compute backend: graphics-only IR node in a compute module",
                    ));
                }
                _ => format!("{}({})", func.name(), a.join(", ")),
            }
        }
        IrExpr::Neg(e) => format!("(-{})", emit_expr(ctx, e)?),
        IrExpr::Binary { op, lhs, rhs } => {
            format!(
                "({} {} {})",
                emit_expr(ctx, lhs)?,
                binop(*op)?,
                emit_expr(ctx, rhs)?
            )
        }
        IrExpr::Input(_) | IrExpr::FieldAccess { .. } | IrExpr::Construct { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL compute backend: graphics-only IR node in a compute module",
            ));
        }
    })
}

fn component(c: u32) -> &'static str {
    match c {
        0 => "x",
        1 => "y",
        _ => "z",
    }
}

fn binop(op: IrBinOp) -> Result<&'static str, Diagnostic> {
    Ok(match op {
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
        IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Shl | IrBinOp::Shr => {
            return Err(Diagnostic::new(
                DiagnosticCode::Unsupported,
                "MSL backend: bitwise operators (&, |, ^, <<, >>) are not yet supported; \
                 this ZSL source is SPIR-V-only today",
            ));
        }
    })
}

/// the MSL spelling of a scalar [`Type`]. non-scalar is a compiler defect,
/// reported as `Unsupported`.
fn msl_scalar(t: &Type) -> Result<&'static str, Diagnostic> {
    match t {
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => Ok("uint"),
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
            format!("MSL backend: unsupported scalar type {other:?}"),
        )),
    }
}

/// MSL spelling of a buffer element [`Type`]. `bf16` is `ushort`, bit-exact with the HIP/CUDA
/// backends: round-to-nearest-even on store, zero-extend on load.
fn msl_buffer_elem(t: &Type) -> Result<&'static str, Diagnostic> {
    match t {
        Type::Float {
            format: FloatFormat::F32,
        } => Ok("float"),
        Type::Float {
            format: FloatFormat::Bf16,
        } => Ok("ushort"),
        Type::Float {
            format: FloatFormat::F16,
        } => Ok("half"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => Ok("uint"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => Ok("int"),
        other => Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!("MSL backend: unsupported compute buffer element {other:?}"),
        )),
    }
}

/// Formats an `f32` as an MSL float literal, with a decimal point.
///
/// `BuiltinFn::name()` already returns MSL spellings, so no separate name map is needed.
fn msl_float(v: f32) -> String {
    format!("{v:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::parse_compute;

    fn msl(src: &str) -> MslShader {
        lower_compute(&parse_compute(src).expect("parse")).expect("lower")
    }

    /// Returns `false` when `xcrun metal` is unavailable.
    fn metal_compiles(source: &str) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let uniq = N.fetch_add(1, Ordering::Relaxed);
        let src = dir.join(format!("zsl_msl_{}_{}.metal", std::process::id(), uniq));
        if std::fs::write(&src, source).is_err() {
            return None;
        }
        let air = src.with_extension("air");
        let out = std::process::Command::new("xcrun")
            .args(["metal", "-c"])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .output();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&air);
        match out {
            Ok(o) => Some(if o.status.success() {
                true
            } else {
                panic!(
                    "MSL failed to compile:\n{}",
                    String::from_utf8_lossy(&o.stderr)
                )
            }),
            Err(_) => None,
        }
    }

    #[test]
    fn saxpy_emits_expected_msl() {
        let m = msl(r#"
            push P { n: u32, alpha: f32 }
            @workgroup_size(256)
            kernel saxpy(x: device buffer<f32>, y: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    y[i] = y[i] + p.alpha * x[i]
                }
            }
        "#);
        assert_eq!(m.local_size, [256, 1, 1]);
        assert_eq!(m.buffer_count, 2);
        assert!(m.has_push);
        assert!(m.source.contains("kernel void zsl_main"));
        assert!(m.source.contains("device float* x [[buffer(0)]]"));
        assert!(m.source.contains("device float* y [[buffer(1)]]"));
        assert!(m.source.contains("constant Push& pc [[buffer(2)]]"));
        assert!(m.source.contains("uint3 gid [[thread_position_in_grid]]"));
        assert!(m.source.contains("y[i] = (y[i] + (pc.alpha * x[i]))"));
    }

    #[test]
    fn saxpy_compiles_with_metal_toolchain() {
        let m = msl(r#"
            push P { n: u32, alpha: f32 }
            @workgroup_size(256)
            kernel saxpy(x: device buffer<f32>, y: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n { y[i] = y[i] + p.alpha * x[i] }
            }
        "#);
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn gemm_with_loop_and_builtins_compiles() {
        let m = msl(r#"
            push P { m: u32, n: u32, k: u32, alpha: f32 }
            @workgroup_size(16, 16)
            kernel gemm(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: P, id: global_id) {
                let row = id.y
                let col = id.x
                if row < p.m && col < p.n {
                    let sum: f32 = 0.0
                    for i in 0..p.k {
                        sum = sum + a[row * p.k + i] * b[i * p.n + col]
                    }
                    c[row * p.n + col] = p.alpha * max(sum, 0.0)
                }
            }
        "#);
        assert!(m.source.contains("for (uint i = 0u; i < pc.k; i++)"));
        assert!(m.source.contains("float sum = 0.0;"));
        assert!(m.source.contains("max("));
        let _ = metal_compiles(&m.source);
    }
}
