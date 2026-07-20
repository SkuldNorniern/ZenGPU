//! IR → MSL lowering for compute kernels.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::{BufElem, EntryKind, Module, ParamKind, ScalarTy};

/// Lower a compute [`Module`] to MSL.
pub fn lower_compute(module: &Module) -> MslShader {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    let locals: HashMap<&str, ScalarTy> = e.locals.iter().map(|(n, t)| (n.as_str(), *t)).collect();
    let buffers: Vec<(&str, BufElem)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some((p.name.as_str(), *elem)),
            _ => None,
        })
        .collect();
    let scalars: Vec<(&str, ScalarTy)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Scalar(t) => Some((p.name.as_str(), *t)),
            _ => None,
        })
        .collect();

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    let mut atomic_buffers = HashSet::new();
    collect_atomic_buffers(&e.body, &mut atomic_buffers);

    let has_push = !scalars.is_empty();
    if has_push {
        s.push_str("struct Push {\n");
        for (n, t) in &scalars {
            let _ = writeln!(s, "    {} {};", msl_scalar(*t), n);
        }
        s.push_str("};\n\n");
    }

    let _ = writeln!(s, "kernel void {ENTRY}(");
    let mut slot = 0u32;
    for (b, buffer_elem) in &buffers {
        let elem = if atomic_buffers.contains(b) {
            "atomic_float"
        } else {
            msl_buffer_elem(*buffer_elem)
        };
        let _ = writeln!(s, "    device {elem}* {b} [[buffer({slot})]],");
        slot += 1;
    }
    if has_push {
        let _ = writeln!(s, "    constant Push& pc [[buffer({slot})]],");
    }
    s.push_str("    uint3 gid [[thread_position_in_grid]],\n");
    s.push_str("    uint3 lid [[thread_position_in_threadgroup]],\n");
    s.push_str("    uint3 group [[threadgroup_position_in_grid]]\n) {\n");

    for shared in &e.shared {
        let _ = writeln!(s, "    threadgroup float {}[{}];", shared.name, shared.len);
    }

    let ctx = Ctx { locals };
    for st in &e.body {
        emit_stmt(&mut s, &ctx, st, 1);
    }
    s.push_str("}\n");

    MslShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_push,
        local_size,
    }
}

struct Ctx<'a> {
    locals: HashMap<&'a str, ScalarTy>,
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

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) {
    match stmt {
        IrStmt::Let { name, init } => {
            let ty = ctx
                .locals
                .get(name.as_str())
                .copied()
                .unwrap_or(ScalarTy::U32);
            indent(s, depth);
            let _ = writeln!(s, "{} {} = {};", msl_scalar(ty), name, emit_expr(init));
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{} = {};", name, emit_expr(value));
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{}[{}] = {};", buf, emit_expr(index), emit_expr(value));
        }
        IrStmt::AtomicAdd { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "atomic_fetch_add_explicit(&{}[uint({})], {}, memory_order_relaxed);",
                buf,
                emit_expr(index),
                emit_expr(value)
            );
        }
        IrStmt::AssignShared { name, index, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{}[{}] = {};", name, emit_expr(index), emit_expr(value));
        }
        IrStmt::Barrier => {
            indent(s, depth);
            s.push_str("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        }
        IrStmt::If { cond, then, else_ } => {
            indent(s, depth);
            let _ = writeln!(s, "if ({}) {{", emit_expr(cond));
            for st in then {
                emit_stmt(s, ctx, st, depth + 1);
            }
            indent(s, depth);
            s.push('}');
            if let Some(else_block) = else_ {
                s.push_str(" else {\n");
                for st in else_block {
                    emit_stmt(s, ctx, st, depth + 1);
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
                emit_expr(lo),
                emit_expr(hi)
            );
            for st in body {
                emit_stmt(s, ctx, st, depth + 1);
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::Eval(expr) => {
            indent(s, depth);
            let _ = writeln!(s, "{};", emit_expr(expr));
        }
    }
}

fn emit_expr(expr: &IrExpr) -> String {
    match expr {
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => msl_float(*v),
        IrExpr::Local(n) => n.clone(),
        IrExpr::ScalarParam(n) => format!("pc.{n}"),
        IrExpr::BufferLoad { buf, index } => format!("{buf}[{}]", emit_expr(index)),
        IrExpr::SharedLoad { name, index } => format!("{name}[{}]", emit_expr(index)),
        IrExpr::GlobalId(c) => format!("gid.{}", component(*c)),
        IrExpr::LocalId(c) => format!("lid.{}", component(*c)),
        IrExpr::GroupId(c) => format!("group.{}", component(*c)),
        IrExpr::Builtin { func, args } => {
            let a: Vec<String> = args.iter().map(emit_expr).collect();
            match func {
                BuiltinFn::U32 => format!("uint({})", a[0]),
                _ => format!("{}({})", func.name(), a.join(", ")),
            }
        }
        IrExpr::Neg(e) => format!("(-{})", emit_expr(e)),
        IrExpr::Binary { op, lhs, rhs } => {
            format!("({} {} {})", emit_expr(lhs), binop(*op), emit_expr(rhs))
        }
        // Graphics-only nodes never appear in a compute module.
        IrExpr::Input(_)
        | IrExpr::FieldAccess { .. }
        | IrExpr::VecConstruct { .. }
        | IrExpr::Extend { .. }
        | IrExpr::Dot { .. } => "/* unsupported */".to_string(),
    }
}

fn component(c: u32) -> &'static str {
    match c {
        0 => "x",
        1 => "y",
        _ => "z",
    }
}

fn binop(op: IrBinOp) -> &'static str {
    match op {
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
        // Not yet supported on this backend; see the SPIR-V-only note on
        // `IrBinOp`. Emits a deliberately invalid marker so any attempt to
        // actually compile this source fails loudly instead of silently
        // producing wrong MSL code.
        IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Shl | IrBinOp::Shr => {
            "/* UNSUPPORTED_BITWISE_OP_ON_MSL_BACKEND */"
        }
    }
}

fn msl_scalar(t: ScalarTy) -> &'static str {
    match t {
        ScalarTy::U32 => "uint",
        ScalarTy::I32 => "int",
        ScalarTy::F32 => "float",
        ScalarTy::Bool => "bool",
    }
}

fn msl_buffer_elem(elem: BufElem) -> &'static str {
    match elem {
        BufElem::F32 => "float",
        BufElem::U32 => "uint",
        BufElem::I32 => "int",
        _ => unreachable!("unsupported compute buffer element"),
    }
}

/// Format an `f32` as an MSL float literal (always with a decimal point).
/// `BuiltinFn::name()` already returns MSL-compatible spellings (abs/sqrt/min/
/// clamp/mix/normalize/length/…), so no separate MSL name map is needed.
fn msl_float(v: f32) -> String {
    format!("{v:?}") // Debug guarantees a `.0` for integral values.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::parse_compute;

    fn msl(src: &str) -> MslShader {
        lower_compute(&parse_compute(src).expect("parse"))
    }

    /// Compile MSL with the real Metal toolchain if present; skip otherwise.
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
            Err(_) => None, // xcrun not available
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
        // Skips silently when the Metal toolchain isn't installed.
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
