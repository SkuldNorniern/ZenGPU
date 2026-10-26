//! IR to MSL lowering for vertex and fragment shaders. ABI for the Metal graphics pipeline:
//! - vertex `@location(N)` inputs arrive via `[[stage_in]]` with `[[attribute(N)]]`;
//! - fragment `@location(N)` inputs arrive via `[[stage_in]]` with `[[user(locnN)]]` (matching the
//!   vertex output varyings);
//! - the push block, if any, binds at `[[buffer(0)]]`; storage buffers follow at `[[buffer(1..)]]`;
//! - the texture table binds at [`TEXTURE_BUFFER_INDEX`] as an argument buffer of `{ tex, smp }`
//!   texture and sampler pairs, one per bindless slot.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::{ArrayLen, FloatFormat, IntWidth, Signedness, Type};
use crate::ir::{GfxBuiltin, GfxSlot, GraphicsModule};

/// Push-constant buffer index for graphics shaders. High so it never overlaps vertex buffers; the
/// Metal HAL binds to this same index.
pub const PUSH_BUFFER_INDEX: u32 = 16;

pub const TEXTURE_BUFFER_INDEX: u32 = 17;

pub fn lower_graphics(module: &GraphicsModule) -> Result<MslShader, Diagnostic> {
    let e = &module.entry;
    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    let mut arrays: HashMap<&str, (&'static str, u32)> = HashMap::new();
    for (n, tid) in &e.locals {
        match module.types.get(*tid) {
            Type::Array {
                element,
                length: ArrayLen::Const(len),
            } => {
                arrays.insert(n.as_str(), (msl_ty_of(module.types.get(*element))?, *len));
            }
            ty => {
                locals.insert(n.as_str(), msl_ty_of(ty)?);
            }
        }
    }
    let builtins = e
        .inputs
        .iter()
        .filter(|i| matches!(i.slot, GfxSlot::Builtin(_)))
        .map(|i| i.name.as_str())
        .collect();
    let ctx = Ctx {
        locals,
        arrays,
        builtins,
        is_fragment: e.is_fragment,
    };
    let stage_in = e
        .inputs
        .iter()
        .any(|i| matches!(i.slot, GfxSlot::Location(_)));
    let builtin_args: Vec<String> = e
        .inputs
        .iter()
        .filter_map(|i| match i.slot {
            GfxSlot::Builtin(b) => Some((b, &i.name)),
            GfxSlot::Location(_) => None,
        })
        .map(|(b, name)| match b {
            GfxBuiltin::VertexIndex => format!("    uint {name} [[vertex_id]]"),
            GfxBuiltin::InstanceIndex => format!("    uint {name} [[instance_id]]"),
            GfxBuiltin::FragCoord => format!("    float4 {name} [[position]]"),
        })
        .collect();

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let has_push = !e.scalar_params.is_empty();
    if has_push {
        s.push_str("struct Push {\n");
        for sp in &e.scalar_params {
            let _ = writeln!(
                s,
                "    {} {};",
                msl_ty_of(module.types.get(sp.ty))?,
                sp.name
            );
        }
        s.push_str("};\n\n");
    }

    let in_qualifier = if e.is_fragment {
        "user(locn"
    } else {
        "attribute("
    };
    if stage_in {
        s.push_str("struct StageIn {\n");
        for inp in &e.inputs {
            let GfxSlot::Location(location) = inp.slot else {
                continue;
            };
            let _ = writeln!(
                s,
                "    {} {} [[{}{})]];",
                msl_ty_of(module.types.get(inp.ty))?,
                inp.name,
                in_qualifier,
                location,
            );
        }
        s.push_str("};\n\n");
    }

    let push_arg = format!("constant Push& pc [[buffer({PUSH_BUFFER_INDEX})]]");
    if e.textures.is_some() {
        texture_abi(&mut s);
    }
    let texture_arg = e
        .textures
        .as_ref()
        .map(|_| format!("    ZSL_TEXTURE_ARGS({TEXTURE_BUFFER_INDEX})"));

    if e.is_fragment {
        let single = match &e.outputs[..] {
            [only] if only.location == 0 => Some(msl_ty_of(module.types.get(only.ty))?),
            _ => None,
        };
        if single.is_none() {
            s.push_str("struct FsOut {\n");
            for (i, o) in e.outputs.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "    {} out{i} [[color({}), index({})]];",
                    msl_ty_of(module.types.get(o.ty))?,
                    o.location,
                    o.blend_src
                );
            }
            s.push_str("};\n\n");
        }
        let _ = writeln!(s, "fragment {} {ENTRY}(", single.unwrap_or("FsOut"));
        let mut args: Vec<String> = Vec::new();
        if stage_in {
            args.push("    StageIn in [[stage_in]]".to_string());
        }
        args.extend(builtin_args.iter().cloned());
        if has_push {
            args.push(format!("    {push_arg}"));
        }
        args.extend(texture_arg.iter().cloned());
        s.push_str(&args.join(",\n"));
        s.push_str("\n) {\n");
        for st in &e.body {
            emit_stmt(&mut s, &ctx, st, 1)?;
        }
        if single.is_some() {
            let _ = writeln!(s, "    return {};", emit_expr(&ctx, &e.ret[0])?);
        } else {
            s.push_str("    FsOut out;\n");
            for (i, ret) in e.ret.iter().enumerate() {
                let _ = writeln!(s, "    out.out{i} = {};", emit_expr(&ctx, ret)?);
            }
            s.push_str("    return out;\n");
        }
        s.push_str("}\n");
    } else {
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
        if stage_in {
            args.push("    StageIn in [[stage_in]]".to_string());
        }
        args.extend(builtin_args.iter().cloned());
        if has_push {
            args.push(format!("    {push_arg}"));
        }
        args.extend(texture_arg.iter().cloned());
        s.push_str(&args.join(",\n"));
        s.push_str("\n) {\n");
        for st in &e.body {
            emit_stmt(&mut s, &ctx, st, 1)?;
        }
        s.push_str("    VsOut out;\n");
        let _ = writeln!(s, "    out.position = {};", emit_expr(&ctx, &e.ret[0])?);
        s.push_str("    out.position.y = -out.position.y;\n");
        for i in 0..e.varyings.len() {
            let _ = writeln!(s, "    out.vary{i} = {};", emit_expr(&ctx, &e.ret[i + 1])?);
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

/// Tier 1 cannot index an array of argument-buffer structs. Bind a small per-draw
/// resource set instead, retaining the public table indices through an ID lookup.
fn texture_abi(s: &mut String) {
    s.push_str("#ifdef ZSL_METAL_TIER1\n");
    s.push_str("struct ZslTextureTable {\n    array<texture2d<float>, 16> tex [[id(0)]];\n    array<sampler, 16> smp [[id(16)]];\n};\n");
    for (suffix, lod) in [("", false), ("_lod", true)] {
        let extra = if lod { ", float lod" } else { "" };
        let _ = writeln!(
            s,
            "inline float4 zsl_sample{suffix}(constant ZslTextureTable& t, constant uint* ids, uint slot, float2 uv{extra}) {{"
        );
        // Constant resource indices also work on devices without resource-array indexing.
        for i in 0..16 {
            let level = if lod { ", level(lod)" } else { "" };
            let _ = writeln!(
                s,
                "    if (ids[{i}] == slot) return t.tex[{i}].sample(t.smp[{i}], uv{level});"
            );
        }
        s.push_str("    return float4(0.0);\n}\n");
    }
    s.push_str("#define ZSL_TEXTURE_ARGS(B) constant ZslTextureTable& zsl_textures [[buffer(B)]], constant uint* zsl_texture_ids [[buffer(18)]]\n");
    s.push_str("#define ZSL_SAMPLE(I, UV) zsl_sample(zsl_textures, zsl_texture_ids, (I), (UV))\n");
    s.push_str("#define ZSL_SAMPLE_LOD(I, UV, L) zsl_sample_lod(zsl_textures, zsl_texture_ids, (I), (UV), (L))\n");
    s.push_str("#else\nstruct ZslTexture {\n    texture2d<float> tex;\n    sampler smp;\n};\n");
    s.push_str("#define ZSL_TEXTURE_ARGS(B) const device ZslTexture* zsl_textures [[buffer(B)]]\n");
    s.push_str("#define ZSL_SAMPLE(I, UV) zsl_textures[I].tex.sample(zsl_textures[I].smp, (UV))\n");
    s.push_str("#define ZSL_SAMPLE_LOD(I, UV, L) zsl_textures[I].tex.sample(zsl_textures[I].smp, (UV), level(L))\n#endif\n\n");
}

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
    arrays: HashMap<&'a str, (&'static str, u32)>,
    builtins: HashSet<&'a str>,
    is_fragment: bool,
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) -> Result<(), Diagnostic> {
    match stmt {
        IrStmt::FragStore { .. } => return Err(super::compute::frag_unsupported("frag_store")),
        IrStmt::Let { name, init } if ctx.arrays.contains_key(name.as_str()) => {
            let (ty, len) = ctx.arrays[name.as_str()];
            indent(s, depth);
            let _ = writeln!(s, "{ty} {name}[{len}] = {};", emit_expr(ctx, init)?);
        }
        IrStmt::Let { name, init } => {
            let ty = ctx.locals.get(name.as_str()).copied().unwrap_or("float");
            indent(s, depth);
            let _ = writeln!(s, "{ty} {} = {};", name, emit_expr(ctx, init)?);
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{} = {};", name, emit_expr(ctx, value)?);
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "{}[{}] = {};",
                buf,
                emit_expr(ctx, index)?,
                emit_expr(ctx, value)?
            );
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
        IrStmt::AssignShared { .. } | IrStmt::Barrier | IrStmt::SharedBarrier => {
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
        IrStmt::ImageStore { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL graphics backend: image stores are unavailable in graphics shaders",
            ));
        }
        IrStmt::Discard => {
            indent(s, depth);
            s.push_str("discard_fragment();\n");
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
        IrExpr::FragLoad { .. } => return Err(super::compute::frag_unsupported("frag_load")),
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => super::compute::msl_float(*v),
        IrExpr::Local(n) => n.clone(),
        IrExpr::Input(n) if ctx.builtins.contains(n.as_str()) => n.clone(),
        IrExpr::Input(n) => format!("in.{n}"),
        IrExpr::ScalarParam(n) => format!("pc.{n}"),
        IrExpr::BufferLoad { buf, index } => format!("{buf}[{}]", emit_expr(ctx, index)?),
        IrExpr::FieldAccess { base, component } => {
            format!("({}).{}", emit_expr(ctx, base)?, component_char(*component))
        }
        IrExpr::ArrayLit(elems) => {
            let mut a = Vec::with_capacity(elems.len());
            for el in elems {
                a.push(emit_expr(ctx, el)?);
            }
            format!("{{{}}}", a.join(", "))
        }
        IrExpr::Index { array, index } => format!("{array}[{}]", emit_expr(ctx, index)?),
        IrExpr::Texel { .. } | IrExpr::TexSize { .. } => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "MSL graphics backend: texture and image params are compute-only",
            ));
        }
        IrExpr::Sample { index, uv, lod } => {
            let index = emit_expr(ctx, index)?;
            let uv = emit_expr(ctx, uv)?;
            match lod {
                Some(lod) => format!("ZSL_SAMPLE_LOD({index}, {uv}, {})", emit_expr(ctx, lod)?),
                None if ctx.is_fragment => format!("ZSL_SAMPLE({index}, {uv})"),
                None => format!("ZSL_SAMPLE_LOD({index}, {uv}, 0.0)"),
            }
        }

        IrExpr::Swizzle { base, lanes } => {
            let picked: String = lanes.iter().map(|&l| component_char(l)).collect();
            format!("({}).{picked}", emit_expr(ctx, base)?)
        }
        IrExpr::Construct { dim, args, .. } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(ctx, arg)?);
            }
            format!("float{dim}({})", a.join(", "))
        }
        IrExpr::Builtin { func, args } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(ctx, arg)?);
            }
            match func {
                BuiltinFn::Dot => format!("dot({}, {})", a[0], a[1]),
                BuiltinFn::Extend => format!("float4({}, {})", a[0], a[1]),
                BuiltinFn::U32 => format!("uint({})", a[0]),
                BuiltinFn::FragZeroAcc
                | BuiltinFn::Mma
                | BuiltinFn::FragLen
                | BuiltinFn::FragRow
                | BuiltinFn::FragCol
                | BuiltinFn::FragGet
                | BuiltinFn::FragSet => return Err(super::compute::frag_unsupported(func.name())),
                BuiltinFn::F32 => format!("float({})", a[0]),
                BuiltinFn::RoundEven => format!("rint({})", a[0]),
                BuiltinFn::FastExp => format!("fast::exp({})", a[0]),
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
        IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::SharedLoad { .. } => {
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
        IrBinOp::Rem => {
            return Err(Diagnostic::new(
                DiagnosticCode::Unsupported,
                "MSL graphics: `%` is compute-only".to_string(),
            ));
        }
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
    })
}

/// MSL spelling of a [`Type`], for the graphics `TypeId` fields.
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
            rows: 4,
            columns: 4,
            ..
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
    use std::env::temp_dir;
    use std::fs::{remove_file, write};
    use std::process::{Command, id};

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
        let dir = temp_dir();
        let uniq = N.fetch_add(1, Ordering::Relaxed);
        let p = dir.join(format!("zsl_gfx_{}_{}.metal", id(), uniq));
        if write(&p, source).is_err() {
            return None;
        }
        let air = p.with_extension("air");
        let out = Command::new("xcrun")
            .args(["metal", "-c"])
            .arg(&p)
            .arg("-o")
            .arg(&air)
            .output();
        let _ = remove_file(&p);
        let _ = remove_file(&air);
        match out {
            Ok(o) if o.status.success() => Some(true),
            Ok(o) if String::from_utf8_lossy(&o.stderr).contains("missing Metal Toolchain") => None,
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
        assert!(m.source.contains("out.position.y = -out.position.y;"));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn builtin_inputs_are_entry_arguments() {
        let m = graphics(
            r#"
            vertex vs(@location(0) rect: f32x4, vi: vertex_index) -> f32x4 {
                f32x4(rect.xy + f32x2(f32(vi & 1), 0.0), 0.0, 1.0)
            }
        "#,
        );
        assert!(m.source.contains("uint vi [[vertex_id]]"));
        assert!(m.source.contains("(in.rect).xy"));
        assert!(m.source.contains("float(((vi & 1u)))") || m.source.contains("float((vi & 1u))"));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn discard_drops_the_fragment() {
        let m = graphics(
            r#"
            fragment fs(@location(0) c: f32x4) -> f32x4 {
                if c.a <= 0.0 { discard }
                c
            }
        "#,
        );
        assert!(m.source.contains("discard_fragment();"));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn samples_through_the_texture_argument_buffer() {
        let m = graphics(
            r#"
            push P { slot: u32 }
            fragment fs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
                tex[p.slot].sample(uv)
            }
        "#,
        );
        assert!(m.source.contains("struct ZslTexture {"));
        assert!(m.source.contains("ZSL_TEXTURE_ARGS(17)"));
        assert!(m.source.contains("ZSL_SAMPLE(pc.slot, in.uv)"));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn dual_source_outputs_share_a_colour() {
        let m = graphics(
            r#"
            fragment fs(@location(0) c: f32x4)
                -> (@location(0) @blend_src(0) f32x4, @location(0) @blend_src(1) f32x4)
            {
                (c, c * 0.5)
            }
        "#,
        );
        assert!(m.source.contains("float4 out0 [[color(0), index(0)]];"));
        assert!(m.source.contains("float4 out1 [[color(0), index(1)]];"));
        assert!(m.source.contains("fragment FsOut zsl_main("));
        let _ = metal_compiles(&m.source);
    }

    #[test]
    fn arrays_are_local_initialised_arrays() {
        let m = graphics(
            r#"
            vertex vs(vi: vertex_index) -> f32x4 {
                let pos = [f32x2(-1.0, 1.0), f32x2(1.0, 1.0), f32x2(0.0, -1.0)]
                f32x4(pos[vi], 0.0, 1.0)
            }
        "#,
        );
        assert!(m.source.contains("float2 pos[3] = {float2("));
        assert!(m.source.contains("float4(pos[vi], 0.0, 1.0)"));
        let _ = metal_compiles(&m.source);
    }
}
