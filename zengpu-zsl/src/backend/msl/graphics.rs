//! IR → MSL lowering for vertex and fragment shaders.
//!
//! ABI for the Metal graphics pipeline to bind against:
//! - vertex `@location(N)` inputs arrive via `[[stage_in]]` with `[[attribute(N)]]`;
//! - fragment `@location(N)` inputs arrive via `[[stage_in]]` with `[[user(locnN)]]`
//!   (matching the vertex output varyings);
//! - the push block, if any, binds at `[[buffer(0)]]`; storage buffers follow at
//!   `[[buffer(1..)]]`.

use std::collections::HashMap;
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::GraphicsModule;
use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type};

/// Buffer index for the push-constant block in graphics shaders. Kept high so it
/// never overlaps vertex geometry buffers bound for `[[stage_in]]`. The Metal
/// HAL backend binds the push constant to this same index.
pub const PUSH_BUFFER_INDEX: u32 = 16;

/// Lower a graphics [`GraphicsModule`] to MSL.
pub fn lower_graphics(module: &GraphicsModule) -> Result<MslShader, Diagnostic> {
    let e = &module.entry;
    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    for (n, tid) in &e.locals {
        locals.insert(n.as_str(), msl_ty_of(module.types.get(*tid))?);
    }
    let ctx = Ctx { locals };

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let has_push = !e.scalar_params.is_empty();
    if has_push {
        s.push_str("struct Push {\n");
        for sp in &e.scalar_params {
            let _ = writeln!(s, "    {} {};", msl_ty_of(module.types.get(sp.ty))?, sp.name);
        }
        s.push_str("};\n\n");
    }

    // Stage-in inputs. Vertex inputs are vertex-buffer attributes; fragment
    // inputs are interpolated varyings.
    let in_qualifier = if e.is_fragment {
        "user(locn"
    } else {
        "attribute("
    };
    if !e.inputs.is_empty() {
        s.push_str("struct StageIn {\n");
        for inp in &e.inputs {
            let _ = writeln!(
                s,
                "    {} {} [[{}{})]];",
                msl_ty_of(module.types.get(inp.ty))?,
                inp.name,
                in_qualifier,
                inp.location,
            );
        }
        s.push_str("};\n\n");
    }

    // The push constant binds at a high, fixed buffer index so it never
    // collides with vertex geometry buffers (which occupy `[[buffer(0..)]]` via
    // the `[[stage_in]]` vertex descriptor). The Metal backend binds to match.
    let push_arg = format!("constant Push& pc [[buffer({PUSH_BUFFER_INDEX})]]");

    if e.is_fragment {
        // fragment <ret> zsl_main(StageIn in [[stage_in]], ...)
        let _ = writeln!(s, "fragment float4 {ENTRY}(");
        let mut args: Vec<String> = Vec::new();
        if !e.inputs.is_empty() {
            args.push("    StageIn in [[stage_in]]".to_string());
        }
        if has_push {
            args.push(format!("    {push_arg}"));
        }
        s.push_str(&args.join(",\n"));
        s.push_str("\n) {\n");
        for st in &e.body {
            emit_stmt(&mut s, &ctx, st, 1)?;
        }
        let _ = writeln!(s, "    return {};", emit_expr(&e.ret[0])?);
        s.push_str("}\n");
    } else {
        // VsOut struct: position + varyings.
        s.push_str("struct VsOut {\n");
        s.push_str("    float4 position [[position]];\n");
        for (i, vty) in e.varyings.iter().enumerate() {
            let _ = writeln!(
                s,
                "    {} vary{i} [[user(locn{i})]];",
                msl_ty_of(module.types.get(*vty))?
            );
        }
        s.push_str("};\n\n");

        let _ = writeln!(s, "vertex VsOut {ENTRY}(");
        let mut args: Vec<String> = Vec::new();
        if !e.inputs.is_empty() {
            args.push("    StageIn in [[stage_in]]".to_string());
        }
        if has_push {
            args.push(format!("    {push_arg}"));
        }
        s.push_str(&args.join(",\n"));
        s.push_str("\n) {\n");
        for st in &e.body {
            emit_stmt(&mut s, &ctx, st, 1)?;
        }
        s.push_str("    VsOut out;\n");
        let _ = writeln!(s, "    out.position = {};", emit_expr(&e.ret[0])?);
        for i in 0..e.varyings.len() {
            let _ = writeln!(s, "    out.vary{i} = {};", emit_expr(&e.ret[i + 1])?);
        }
        s.push_str("    return out;\n}\n");
    }

    Ok(MslShader {
        source: s,
        entry: ENTRY,
        buffer_count: e.buf_params.len() as u32,
        has_push,
        local_size: [1, 1, 1],
    })
}

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) -> Result<(), Diagnostic> {
    match stmt {
        IrStmt::Let { name, init } => {
            let ty = ctx.locals.get(name.as_str()).copied().unwrap_or("float");
            indent(s, depth);
            let _ = writeln!(s, "{ty} {} = {};", name, emit_expr(init)?);
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{} = {};", name, emit_expr(value)?);
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{}[{}] = {};", buf, emit_expr(index)?, emit_expr(value)?);
        }
        IrStmt::If { cond, then, else_ } => {
            indent(s, depth);
            let _ = writeln!(s, "if ({}) {{", emit_expr(cond)?);
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
                emit_expr(lo)?,
                emit_expr(hi)?
            );
            for st in body {
                emit_stmt(s, ctx, st, depth + 1)?;
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        // These are compute-only statements; this crate's own parser never
        // constructs them inside a graphics module. Hitting this is a
        // compiler defect, not a target limitation.
        IrStmt::AssignShared { .. } | IrStmt::Barrier => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL graphics backend: workgroup operations are unavailable in graphics shaders",
            ));
        }
        IrStmt::AtomicAdd { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL graphics backend: atomic_add is unavailable in graphics shaders",
            ));
        }
        IrStmt::Eval(expr) => {
            indent(s, depth);
            let _ = writeln!(s, "{};", emit_expr(expr)?);
        }
    }
    Ok(())
}

fn emit_expr(expr: &IrExpr) -> Result<String, Diagnostic> {
    Ok(match expr {
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => format!("{v:?}"),
        IrExpr::Local(n) => n.clone(),
        IrExpr::Input(n) => format!("in.{n}"),
        IrExpr::ScalarParam(n) => format!("pc.{n}"),
        IrExpr::BufferLoad { buf, index } => format!("{buf}[{}]", emit_expr(index)?),
        IrExpr::FieldAccess { base, component } => {
            format!("({}).{}", emit_expr(base)?, component_char(*component))
        }
        IrExpr::Construct { dim, args, .. } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(arg)?);
            }
            format!("float{dim}({})", a.join(", "))
        }
        IrExpr::Builtin { func, args } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(arg)?);
            }
            match func {
                BuiltinFn::Dot => format!("dot({}, {})", a[0], a[1]),
                BuiltinFn::Extend => format!("float4({}, {})", a[0], a[1]),
                _ => format!("{}({})", func.name(), a.join(", ")),
            }
        }
        IrExpr::Neg(e) => format!("(-{})", emit_expr(e)?),
        IrExpr::Binary { op, lhs, rhs } => {
            format!("({} {} {})", emit_expr(lhs)?, binop(*op)?, emit_expr(rhs)?)
        }
        // These are compute-only IR nodes; this crate's own parser never
        // constructs them inside a graphics module. Hitting this is a
        // compiler defect, not a target limitation.
        IrExpr::GlobalId(_) | IrExpr::LocalId(_) | IrExpr::GroupId(_) | IrExpr::SharedLoad { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL graphics backend: compute-only expression in a graphics module",
            ));
        }
    })
}

fn component_char(c: u32) -> char {
    match c {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        _ => 'w',
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
        // Not yet supported on this backend; see the SPIR-V-only note on
        // `IrBinOp`. Previously emitted a deliberately invalid marker
        // comment; now fails ZSL compilation itself instead.
        IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Shl | IrBinOp::Shr => {
            return Err(Diagnostic::new(
                DiagnosticCode::Unsupported,
                "MSL backend: bitwise operators (&, |, ^, <<, >>) are not yet supported; \
                 this ZSL source is SPIR-V-only today",
            ));
        }
    })
}

/// The MSL spelling of a unified [`Type`] — used for `GfxScalar.ty`/
/// `GfxInput.ty`/`GraphicsEntry.varyings`/`GraphicsEntry.locals`, all `TypeId`.
fn msl_ty_of(t: &Type) -> Result<&'static str, Diagnostic> {
    match t {
        Type::Float {
            format: FloatFormat::F32,
        } => Ok("float"),
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => Ok("uint"),
        Type::Vector { lanes, .. } => match lanes {
            2 => Ok("float2"),
            3 => Ok("float3"),
            4 => Ok("float4"),
            other => Err(Diagnostic::new(
                DiagnosticCode::Unsupported,
                format!("MSL backend: unsupported vector width {other}"),
            )),
        },
        Type::Matrix {
            rows: 4, columns: 4, ..
        } => Ok("float4x4"),
        other => Err(Diagnostic::new(
            DiagnosticCode::Unsupported,
            format!("MSL backend: unsupported graphics value type {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::{Shader, parse_zsl};

    fn graphics(src: &str) -> MslShader {
        match parse_zsl(src).expect("parse") {
            Shader::Graphics(m) => lower_graphics(&m).expect("lower"),
            Shader::Compute(_) => panic!("expected graphics"),
        }
    }

    fn metal_compiles(source: &str) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let uniq = N.fetch_add(1, Ordering::Relaxed);
        let p = dir.join(format!("zsl_gfx_{}_{}.metal", std::process::id(), uniq));
        if std::fs::write(&p, source).is_err() {
            return None;
        }
        let air = p.with_extension("air");
        let out = std::process::Command::new("xcrun")
            .args(["metal", "-c"])
            .arg(&p)
            .arg("-o")
            .arg(&air)
            .output();
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&air);
        match out {
            Ok(o) if o.status.success() => Some(true),
            Ok(o) => panic!(
                "MSL failed:\n--- stderr ---\n{}\n--- stdout ---\n{}\n--- source ---\n{source}",
                String::from_utf8_lossy(&o.stderr),
                String::from_utf8_lossy(&o.stdout)
            ),
            Err(_) => None,
        }
    }

    #[test]
    fn vertex_mvp_with_varying() {
        let m = graphics(
            r#"
            push P { mvp: mat4x4<f32> }
            vertex vs(@location(0) pos: f32x3, @location(1) col: f32x3, p: P) -> (f32x4, f32x3) {
                (p.mvp * pos.extend(1.0), col)
            }
        "#,
        );
        assert!(m.source.contains("struct Push {"));
        assert!(m.source.contains("float4x4 mvp;"));
        assert!(m.source.contains("float3 pos [[attribute(0)]]"));
        assert!(m.source.contains("float4 position [[position]]"));
        assert!(m.source.contains("float3 vary0 [[user(locn0)]]"));
        assert!(m.source.contains("vertex VsOut zsl_main"));
        assert!(
            m.source
                .contains("out.position = (pc.mvp * float4(in.pos, 1.0));")
        );
        assert!(m.source.contains("out.vary0 = in.col;"));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn fragment_color() {
        let m = graphics(
            r#"
            fragment fs(@location(0) v_color: f32x3) -> f32x4 {
                v_color.extend(1.0)
            }
        "#,
        );
        assert!(m.source.contains("float3 v_color [[user(locn0)]]"));
        assert!(m.source.contains("fragment float4 zsl_main"));
        assert!(m.source.contains("return float4(in.v_color, 1.0);"));
        let _ = metal_compiles(&m.source);
    }
}
