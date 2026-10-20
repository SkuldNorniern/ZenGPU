
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{
    BuiltinFn, FragSrc, IrBinOp, IrExpr, IrStmt, TexelOp, stmts_use_subgroup_query,
    stmts_use_workgroup_reduce,
};
use crate::ir::typeck::frag_parts;
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Signedness, Type, TypeId};
use crate::ir::validate::check_uniformity;
use crate::ir::{EntryKind, Module, ParamKind};

pub fn lower_compute(module: &Module) -> Result<MslShader, Diagnostic> {
    check_uniformity(module)?;
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    let mut locals: HashMap<&str, &'static str> = HashMap::new();
    let mut frags: HashMap<&str, FragTiles> = HashMap::new();
    for (n, t) in &e.locals {
        if frag_parts(&module.types, *t).is_some() {
            frags.insert(n.as_str(), frag_tiles(&module.types, *t)?);
        } else {
            locals.insert(n.as_str(), msl_scalar(module.types.get(*t))?);
        }
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
    let textures: Vec<&str> = e
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Texture))
        .map(|p| p.name.as_str())
        .collect();
    let images: Vec<&str> = e
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Image))
        .map(|p| p.name.as_str())
        .collect();

    let bf16_id = {
        let mut t = module.types.clone();
        t.bf16_id()
    };
    let bf16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, elem)| *elem == bf16_id)
        .map(|(name, _)| *name)
        .collect();

    let f16_id = {
        let mut t = module.types.clone();
        t.f16_id()
    };
    let f16_buffers: HashSet<&str> = buffers
        .iter()
        .filter(|(_, elem)| *elem == f16_id)
        .map(|(name, _)| *name)
        .collect();
    let mut buffer_elems: HashMap<&str, &'static str> = HashMap::new();
    for (name, elem) in &buffers {
        buffer_elems.insert(name, msl_buffer_elem(module.types.get(*elem))?);
    }
    let uses_workgroup_reduce = stmts_use_workgroup_reduce(&e.body);
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
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    s.push_str(
        "template <int OP> inline float zsl_wg_reduce(threadgroup float* scratch, uint lid, uint n, float x) {\n\
         \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   scratch[lid] = x;\n\
         \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   for (uint s = n / 2; s > 0; s /= 2) {\n\
         \x20       if (lid < s) {\n\
         \x20           float o = scratch[lid + s];\n\
         \x20           float v = scratch[lid];\n\
         \x20           scratch[lid] = OP == 0 ? v + o : OP == 1 ? min(v, o) : max(v, o);\n\
         \x20       }\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   }\n\
         \x20   return scratch[0];\n\
         }\n",
    );
    s.push_str(
        "inline uint zsl_dot4(uint a, bool sa, uint b, bool sb, uint acc) {\n\
         \x20   for (uint i = 0; i < 4; ++i) {\n\
         \x20       uint ua = (a >> (8 * i)) & 0xffu;\n\
         \x20       uint ub = (b >> (8 * i)) & 0xffu;\n\
         \x20       int x = sa ? int(char(ua)) : int(ua);\n\
         \x20       int y = sb ? int(char(ub)) : int(ub);\n\
         \x20       acc += uint(x * y);\n\
         \x20   }\n\
         \x20   return acc;\n\
         }\n\n",
    );
    s.push_str(
        r#"
inline uint zsl_pk_add_u16(uint a, uint b) {
    return as_type<uint>(as_type<ushort2>(a) + as_type<ushort2>(b));
}
inline uint zsl_pk_mul_u16(uint a, uint b) {
    return as_type<uint>(as_type<ushort2>(a) * as_type<ushort2>(b));
}
inline uint zsl_pk_mad_u16(uint a, uint b, uint c) {
    return zsl_pk_add_u16(zsl_pk_mul_u16(a, b), c);
}
inline uint zsl_pk_add_f16(uint a, uint b) {
    return as_type<uint>(as_type<half2>(a) + as_type<half2>(b));
}
inline uint zsl_pk_mul_f16(uint a, uint b) {
    return as_type<uint>(as_type<half2>(a) * as_type<half2>(b));
}
inline uint zsl_pk_fma_f16(uint a, uint b, uint c) {
    return as_type<uint>(fma(as_type<half2>(a), as_type<half2>(b), as_type<half2>(c)));
}
inline float zsl_dot2_f16(uint a, uint b, float acc) {
    half2 x = as_type<half2>(a);
    half2 y = as_type<half2>(b);
    acc = fma(float(x.x), float(y.x), acc);
    return fma(float(x.y), float(y.y), acc);
}
inline uint zsl_funnel_shr(uint hi, uint lo, uint s) {
    return uint(((ulong(hi) << 32) | ulong(lo)) >> (s & 31u));
}
inline uint zsl_byte_perm(uint a, uint b, uint sel) {
    uint r = 0;
    for (uint i = 0; i < 4; ++i) {
        uint k = (sel >> (4 * i)) & 7u;
        uint w = k < 4u ? a : b;
        r |= ((w >> (8 * (k & 3u))) & 0xffu) << (8 * i);
    }
    return r;
}
"#,
    );

    if !bf16_buffers.is_empty() {
        s.push_str(
            "inline ushort f32_to_bf16(float x) {\n\
             \x20   uint bits = as_type<uint>(x);\n\
             \x20   uint bias = 0x7fffu + ((bits >> 16) & 1u);\n\
             \x20   if ((bits & 0x7fffffffu) > 0x7f800000u) {\n\
             \x20       return (ushort)((bits >> 16) | 0x40u);\n\
             \x20   }\n\
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
    for (i, t) in textures.iter().enumerate() {
        let _ = writeln!(s, "    texture2d<float> {t} [[texture({i})]],");
        let _ = writeln!(s, "    sampler {t}_smp [[sampler({i})]],");
    }
    for (i, t) in images.iter().enumerate() {
        let _ = writeln!(
            s,
            "    texture2d<float, access::read_write> {t} [[texture({})]],",
            textures.len() + i
        );
    }
    s.push_str("    uint3 gid [[thread_position_in_grid]],\n");
    s.push_str("    uint3 lid [[thread_position_in_threadgroup]],\n");
    s.push_str("    uint3 group [[threadgroup_position_in_grid]]");
    let uses_subgroup_query = stmts_use_subgroup_query(&e.body);
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
    if uses_workgroup_reduce {
        let _ = writeln!(
            s,
            "    threadgroup float __zsl_wg_reduce_f32[{block_size}];"
        );
    }

    let ctx = Ctx {
        locals,
        frags,
        buffer_elems,
        bf16_buffers,
        f16_buffers,
        block_size,
        has_images: !images.is_empty(),
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

/// Four 8x8 `simdgroup_matrix` tiles, indexed `tile = 2 * row + col`.
#[derive(Clone, Copy)]
struct FragTiles {
    usage: FragUse,
    /// The tile element: `half` for a/b, `float` for an accumulator.
    elem: &'static str,
}

const TILES: usize = 4;

struct Ctx<'a> {
    locals: HashMap<&'a str, &'static str>,
    frags: HashMap<&'a str, FragTiles>,
    buffer_elems: HashMap<&'a str, &'static str>,
    bf16_buffers: HashSet<&'a str>,
    f16_buffers: HashSet<&'a str>,
    block_size: u32,
    /// Kernel writes images: barriers must order texture memory too.
    has_images: bool,
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

fn emit_lines(s: &mut String, depth: usize, lines: &[String]) {
    for line in lines {
        indent(s, depth);
        s.push_str(line);
        s.push('\n');
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) -> Result<(), Diagnostic> {
    match stmt {
        IrStmt::FragStore {
            src,
            offset,
            ld,
            value,
        } => {
            let lines = frag_store_lines(ctx, src, offset, ld, value)?;
            emit_lines(s, depth, &lines);
        }
        IrStmt::Let { name, init } => {
            if let Some(frag) = ctx.frags.get(name.as_str()).copied() {
                let mut lines = vec![format!(
                    "simdgroup_matrix<{}, 8, 8> {name}[{TILES}];",
                    frag.elem
                )];
                lines.extend(frag_fill_lines(ctx, name, frag, init)?);
                emit_lines(s, depth, &lines);
            } else {
                let ty = ctx.locals.get(name.as_str()).copied().unwrap_or("uint");
                indent(s, depth);
                let _ = writeln!(s, "{} {} = {};", ty, name, emit_expr(ctx, init)?);
            }
        }
        IrStmt::AssignLocal { name, value } => {
            if let Some(frag) = ctx.frags.get(name.as_str()).copied() {
                let lines = frag_fill_lines(ctx, name, frag, value)?;
                emit_lines(s, depth, &lines);
            } else {
                indent(s, depth);
                let _ = writeln!(s, "{} = {};", name, emit_expr(ctx, value)?);
            }
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
        IrStmt::ImageStore { image, args } => {
            let a = args
                .iter()
                .map(|a| emit_expr(ctx, a))
                .collect::<Result<Vec<_>, _>>()?;
            let [x, y, r, g, b, alpha] = a.as_slice() else {
                return Err(Diagnostic::new(
                    DiagnosticCode::Internal,
                    "image store without x, y and four channels",
                ));
            };
            indent(s, depth);
            let _ = writeln!(
                s,
                "{image}.write(float4({r}, {g}, {b}, {alpha}), uint2({x}, {y}));"
            );
        }
        IrStmt::Discard => {
            return Err(Diagnostic::new(
                DiagnosticCode::Internal,
                "discard is fragment-only and never reaches a compute backend",
            ));
        }
        IrStmt::Barrier => {
            indent(s, depth);
            if ctx.has_images {
                s.push_str(
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_texture);\n",
                );
            } else {
                s.push_str("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
            }
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
        IrExpr::FragLoad { .. } => {
            return Err(frag_rejected(
                "frag_load",
                "a fragment load must be the whole initializer of a `let` or an assignment",
            ));
        }
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
                BuiltinFn::FragZeroAcc | BuiltinFn::Mma => {
                    return Err(frag_rejected(
                        func.name(),
                        "a fragment op must be the whole initializer of a `let` or an assignment",
                    ));
                }
                BuiltinFn::FragLen
                | BuiltinFn::FragRow
                | BuiltinFn::FragCol
                | BuiltinFn::FragGet
                | BuiltinFn::FragSet => {
                    return Err(frag_rejected(
                        func.name(),
                        "simdgroup_matrix elements have no per-lane fragment mapping; use HIP",
                    ));
                }
                BuiltinFn::F32 => format!("float({})", a[0]),
                BuiltinFn::Step => format!("step({}, {})", a[0], a[1]),
                BuiltinFn::Smoothstep => format!("smoothstep({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::MulAddF32 => format!("fma({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::Dot4I8 => {
                    format!("int(zsl_dot4({}, true, {}, true, {}))", a[0], a[1], a[2])
                }
                BuiltinFn::Dot4U8 => {
                    format!("zsl_dot4({}, false, {}, false, {})", a[0], a[1], a[2])
                }
                BuiltinFn::Dot4IU8 => {
                    format!("int(zsl_dot4({}, true, {}, false, {}))", a[0], a[1], a[2])
                }
                BuiltinFn::PkAddU16 => format!("zsl_pk_add_u16({}, {})", a[0], a[1]),
                BuiltinFn::PkMulU16 => format!("zsl_pk_mul_u16({}, {})", a[0], a[1]),
                BuiltinFn::PkMadU16 => format!("zsl_pk_mad_u16({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::PkAddF16 => format!("zsl_pk_add_f16({}, {})", a[0], a[1]),
                BuiltinFn::PkMulF16 => format!("zsl_pk_mul_f16({}, {})", a[0], a[1]),
                BuiltinFn::PkFmaF16 => format!("zsl_pk_fma_f16({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::Dot2F16 => format!("zsl_dot2_f16({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::FunnelShr => format!("zsl_funnel_shr({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::BytePerm => format!("zsl_byte_perm({}, {}, {})", a[0], a[1], a[2]),
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
                BuiltinFn::WorkgroupReduceAdd
                | BuiltinFn::WorkgroupReduceMin
                | BuiltinFn::WorkgroupReduceMax => {
                    let op = match func {
                        BuiltinFn::WorkgroupReduceAdd => 0,
                        BuiltinFn::WorkgroupReduceMin => 1,
                        BuiltinFn::WorkgroupReduceMax => 2,
                        _ => unreachable!(),
                    };
                    format!(
                        "zsl_wg_reduce<{op}>(__zsl_wg_reduce_f32, lid.x, {}u, {})",
                        ctx.block_size, a[0]
                    )
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
        IrExpr::Texel {
            tex,
            op,
            args,
            component: c,
        } => {
            let a = args
                .iter()
                .map(|a| emit_expr(ctx, a))
                .collect::<Result<Vec<_>, _>>()?;
            let read = match (op, a.as_slice()) {
                (TexelOp::Load, [x, y]) => format!("{tex}.read(uint2({x}, {y}))"),
                (TexelOp::Sample, [u, v]) => {
                    format!("{tex}.sample({tex}_smp, float2({u}, {v}), level(0.0))")
                }
                (TexelOp::SampleLevel, [u, v, lod]) => {
                    format!("{tex}.sample({tex}_smp, float2({u}, {v}), level({lod}))")
                }
                _ => {
                    return Err(Diagnostic::new(
                        DiagnosticCode::Internal,
                        "texel read with the wrong argument count",
                    ));
                }
            };
            format!("{read}.{}", ["r", "g", "b", "a"][*c as usize & 3])
        }
        IrExpr::TexSize { tex, axis } => {
            let side = if *axis == 0 { "width" } else { "height" };
            format!("{tex}.get_{side}()")
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
        IrExpr::Input(_)
        | IrExpr::FieldAccess { .. }
        | IrExpr::Swizzle { .. }
        | IrExpr::Sample { .. }
        | IrExpr::ArrayLit(_)
        | IrExpr::Index { .. }
        | IrExpr::Construct { .. } => {
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
        IrBinOp::BitAnd => "&",
        IrBinOp::BitOr => "|",
        IrBinOp::BitXor => "^",
        IrBinOp::Shl => "<<",
        IrBinOp::Shr => ">>",
    })
}

/// Non-scalar types are a compiler defect here, reported as `Unsupported`. Not yet: graphics
/// entries are not lowered for MSL.
pub(super) fn frag_unsupported(what: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Unsupported,
        format!("MSL backend: {what} is not lowered yet (simdgroup_matrix is 8x8)"),
    )
}

fn frag_rejected(what: &str, why: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Unsupported,
        format!("MSL backend: {what} is not supported: {why}"),
    )
}

/// MSL `simdgroup_matrix` is only `half` or `float`: packed i8 and i32 fragments are rejected.
fn frag_tiles(types: &TypeArena, ty: TypeId) -> Result<FragTiles, Diagnostic> {
    let (elem, usage) =
        frag_parts(types, ty).ok_or_else(|| frag_rejected("fragment", "not a fragment type"))?;
    let elem = match (types.get(elem), usage) {
        (
            Type::Float {
                format: FloatFormat::F16,
            },
            FragUse::A | FragUse::B,
        ) => "half",
        (
            Type::Float {
                format: FloatFormat::F32,
            },
            FragUse::Acc,
        ) => "float",
        (
            Type::Int {
                width: IntWidth::W8,
                ..
            },
            _,
        ) => {
            return Err(frag_rejected(
                "packed i8 fragment",
                "simdgroup_matrix has no integer element type",
            ));
        }
        (
            Type::Int {
                width: IntWidth::W32,
                ..
            },
            _,
        ) => {
            return Err(frag_rejected(
                "i32 fragment",
                "simdgroup_matrix has no integer element type",
            ));
        }
        _ => {
            return Err(frag_rejected(
                "fragment",
                "a/b fragments are f16 and accumulators are f32 on MSL",
            ));
        }
    };
    Ok(FragTiles { usage, elem })
}

fn tile_origin(t: usize) -> (usize, usize) {
    ((t / 2) * 8, (t % 2) * 8)
}

fn frag_fill_lines(
    ctx: &Ctx<'_>,
    name: &str,
    frag: FragTiles,
    value: &IrExpr,
) -> Result<Vec<String>, Diagnostic> {
    match value {
        IrExpr::FragLoad {
            src, offset, ld, ..
        } => {
            let off = emit_expr(ctx, offset)?;
            let ld = emit_expr(ctx, ld)?;
            let base = match src {
                FragSrc::Buffer(buf) => {
                    let elem = ctx.buffer_elems.get(buf.as_str()).copied().unwrap_or("");
                    if elem != frag.elem {
                        return Err(frag_rejected(
                            &format!("frag load from `{buf}`"),
                            "a/b load f16 buffers, accumulator loads f32 buffers",
                        ));
                    }
                    format!("({buf} + ({off}))")
                }
                FragSrc::Shared(tile) => {
                    if frag.elem != "float" {
                        return Err(frag_rejected(
                            &format!("frag load from shared `{tile}`"),
                            "only accumulators load from shared arrays on MSL (a/b would need an f16 stage)",
                        ));
                    }
                    format!("({tile} + ({off}))")
                }
            };
            Ok((0..TILES)
                .map(|t| {
                    let (r, c) = tile_origin(t);
                    format!("simdgroup_load({name}[{t}], {base} + {r}u * ({ld}) + {c}u, ({ld}));")
                })
                .collect())
        }
        IrExpr::Builtin {
            func: BuiltinFn::Mma,
            args,
        } => mma_lines(ctx, name, frag, args),
        IrExpr::Builtin {
            func: BuiltinFn::FragZeroAcc,
            ..
        } => Ok((0..TILES)
            .map(|t| format!("{name}[{t}] = simdgroup_matrix<{}, 8, 8>(0);", frag.elem))
            .collect()),
        IrExpr::Local(src) => {
            let src_frag = ctx.frags.get(src.as_str()).copied().ok_or_else(|| {
                frag_rejected(
                    &format!("`{src}`"),
                    "a fragment value must be a fragment local",
                )
            })?;
            if src_frag.elem != frag.elem || src_frag.usage != frag.usage {
                return Err(frag_rejected(
                    &format!("assignment from `{src}`"),
                    "fragment types must match",
                ));
            }
            Ok((0..TILES)
                .map(|t| format!("{name}[{t}] = {src}[{t}];"))
                .collect())
        }
        _ => Err(frag_rejected(
            "fragment value",
            "expected a fragment load, mma, frag_zero_acc or a fragment local",
        )),
    }
}

/// `mma` as eight `simdgroup_multiply_accumulate` calls. `c` seeds `d`.
fn mma_lines(
    ctx: &Ctx<'_>,
    name: &str,
    frag: FragTiles,
    args: &[IrExpr],
) -> Result<Vec<String>, Diagnostic> {
    let (IrExpr::Local(a), IrExpr::Local(b), c) = (&args[0], &args[1], &args[2]) else {
        return Err(frag_rejected("mma", "operands must be fragment locals"));
    };
    let a_frag = ctx
        .frags
        .get(a.as_str())
        .copied()
        .filter(|f| f.usage == FragUse::A)
        .ok_or_else(|| frag_rejected("mma", "the first operand must be an a fragment"))?;
    let b_frag = ctx
        .frags
        .get(b.as_str())
        .copied()
        .filter(|f| f.usage == FragUse::B)
        .ok_or_else(|| frag_rejected("mma", "the second operand must be a b fragment"))?;
    if frag.usage != FragUse::Acc || a_frag.elem != "half" || b_frag.elem != "half" {
        return Err(frag_rejected(
            "mma",
            "mma takes f16 a and b and yields an f32 accumulator",
        ));
    }
    let mut lines = match c {
        IrExpr::Local(cn) => {
            let c_frag =
                ctx.frags.get(cn.as_str()).copied().ok_or_else(|| {
                    frag_rejected("mma", "the accumulator must be a fragment local")
                })?;
            if c_frag.usage != FragUse::Acc || c_frag.elem != "float" {
                return Err(frag_rejected(
                    "mma",
                    "the accumulator must be an f32 acc fragment",
                ));
            }
            (0..TILES)
                .map(|t| format!("{name}[{t}] = {cn}[{t}];"))
                .collect::<Vec<_>>()
        }
        IrExpr::Builtin {
            func: BuiltinFn::FragZeroAcc,
            ..
        } => (0..TILES)
            .map(|t| format!("{name}[{t}] = simdgroup_matrix<float, 8, 8>(0);"))
            .collect(),
        _ => {
            return Err(frag_rejected(
                "mma",
                "the accumulator must be a fragment local or frag_zero_acc()",
            ));
        }
    };
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                lines.push(format!(
                    "simdgroup_multiply_accumulate({name}[{}], {a}[{}], {b}[{}], {name}[{}]);",
                    2 * i + j,
                    2 * i + k,
                    2 * k + j,
                    2 * i + j,
                ));
            }
        }
    }
    Ok(lines)
}

fn frag_store_lines(
    ctx: &Ctx<'_>,
    src: &FragSrc,
    offset: &IrExpr,
    ld: &IrExpr,
    value: &IrExpr,
) -> Result<Vec<String>, Diagnostic> {
    let IrExpr::Local(acc) = value else {
        return Err(frag_rejected(
            "frag_store",
            "the value must be a fragment local",
        ));
    };
    let is_f32_acc = ctx
        .frags
        .get(acc.as_str())
        .is_some_and(|f| f.usage == FragUse::Acc && f.elem == "float");
    if !is_f32_acc {
        return Err(frag_rejected(
            "frag_store",
            "only f32 accumulators can be stored",
        ));
    }
    let off = emit_expr(ctx, offset)?;
    let ld = emit_expr(ctx, ld)?;
    let base = match src {
        FragSrc::Buffer(buf) => {
            if ctx.buffer_elems.get(buf.as_str()).copied() != Some("float") {
                return Err(frag_rejected(
                    &format!("frag_store to `{buf}`"),
                    "accumulators are stored to f32 buffers",
                ));
            }
            format!("({buf} + ({off}))")
        }
        FragSrc::Shared(tile) => format!("({tile} + ({off}))"),
    };
    Ok((0..TILES)
        .map(|t| {
            let (r, c) = tile_origin(t);
            format!("simdgroup_store({acc}[{t}], {base} + {r}u * ({ld}) + {c}u, ({ld}));")
        })
        .collect())
}

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
        Type::Frag { .. } => Err(frag_unsupported("frag local")),
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

/// f32 as an MSL literal, always with a decimal point. Non-finite values use the Metal math macros:
/// `{:?}` gives `NaN`/`inf`, which MSL lacks.
pub(super) fn msl_float(v: f32) -> String {
    if v.is_nan() {
        "NAN".to_string()
    } else if v == f32::INFINITY {
        "INFINITY".to_string()
    } else if v == f32::NEG_INFINITY {
        "(-INFINITY)".to_string()
    } else {
        format!("{v:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use std::fs::{remove_file, write};
    use std::process::{Command, id};

    use crate::frontend::parser::parse_compute;

    fn msl(src: &str) -> MslShader {
        lower_compute(&parse_compute(src).expect("parse")).expect("lower")
    }

    /// Returns `false` when `xcrun metal` is unavailable.
    fn metal_compiles(source: &str) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = temp_dir();
        let uniq = N.fetch_add(1, Ordering::Relaxed);
        let src = dir.join(format!("zsl_msl_{}_{}.metal", id(), uniq));
        if write(&src, source).is_err() {
            return None;
        }
        let air = src.with_extension("air");
        let out = Command::new("xcrun")
            .args(["metal", "-c"])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .output();
        let _ = remove_file(&src);
        let _ = remove_file(&air);
        match out {
            Ok(o) if String::from_utf8_lossy(&o.stderr).contains("missing Metal Toolchain") => None,
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
