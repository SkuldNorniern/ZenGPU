use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::backend::msl::{ENTRY, MslShader};
use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::ir::arena::TypeArena;
use crate::ir::node::{
    BuiltinFn, FragSrc, IrBinOp, IrExpr, IrStmt, TexelOp, stmts_use_subgroup_query,
    stmts_use_workgroup_reduce,
};
use crate::ir::typeck::{ComputeTypeCtx, frag_parts, type_of_compute_expr, u32_vec_lanes};
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Signedness, Type, TypeId};
use crate::ir::validate::check_uniformity;
use crate::ir::{EntryKind, Module, ParamKind};
use std::cell::RefCell;

pub fn lower_compute(module: &Module) -> Result<MslShader, Diagnostic> {
    check_uniformity(module)?;
    let e = &module.entry;
    let EntryKind::Compute {
        local_size,
        occupancy,
    } = e.kind;

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
    let mut shared_elems: HashMap<&str, &'static str> = HashMap::new();
    for decl in &e.shared {
        shared_elems.insert(
            decl.name.as_str(),
            msl_buffer_elem(module.types.get(decl.elem))?,
        );
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
        "template <int OP, typename T> inline T zsl_wg_reduce(threadgroup float* scratch_f, uint lid, uint n, T x) {\n\
         \x20   threadgroup T* scratch = (threadgroup T*)scratch_f;\n\
         \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   scratch[lid] = x;\n\
         \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   for (uint s = n / 2; s > 0; s /= 2) {\n\
         \x20       if (lid < s) {\n\
         \x20           T o = scratch[lid + s];\n\
         \x20           T v = scratch[lid];\n\
         \x20           scratch[lid] = OP == 0 ? v + o : OP == 1 ? min(v, o) : max(v, o);\n\
         \x20       }\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   }\n\
         \x20   return scratch[0];\n\
         }\n",
    );
    s.push_str(
        "// A 16x16 fragment is four 8x8 simdgroup matrices (row-major tiles). Each lane holds\n\
         // two adjacent elements of each tile; element i of a lane is in tile i / 2.\n\
         inline uint zsl_frag_row(uint lane, uint i) {\n\
         \x20   return (i / 4u) * 8u + ((lane / 4u) & 4u) + (lane / 2u) % 4u;\n\
         }\n\
         inline uint zsl_frag_col(uint lane, uint i) {\n\
         \x20   return ((i / 2u) % 2u) * 8u + ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u + i % 2u;\n\
         }\n\
         template <typename T> inline float zsl_frag_get(thread simdgroup_matrix<T, 8, 8> (&f)[4], uint i) {\n\
         \x20   return float(f[i / 2u].thread_elements()[i % 2u]);\n\
         }\n\
         // Lane n is bit n % 32 of component n / 32.\n\
         inline uint4 zsl_ballot(bool p) {\n\
         \x20   ulong v = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(p)));\n\
         \x20   return uint4(uint(v), uint(v >> 32), 0u, 0u);\n\
         }\n\
         inline float zsl_f16_lo(uint x) { return float(as_type<half>(ushort(x & 0xffffu))); }\n\
         inline float zsl_f16_hi(uint x) { return float(as_type<half>(ushort(x >> 16))); }\n\
         inline uint zsl_pack_f16x2(float lo, float hi) {\n\
         \x20   return uint(as_type<ushort>(half(lo))) | (uint(as_type<ushort>(half(hi))) << 16);\n\
         }\n\
         inline float zsl_bitcast_f32(uint x) { return as_type<float>(x); }\n\
         inline uint zsl_bitcast_u32(float x) { return as_type<uint>(x); }\n\
         inline uint zsl_bitcast_u32(int x) { return uint(x); }\n\
         inline uint zsl_bitcast_u32(uint x) { return x; }\n\
         inline int zsl_bitcast_i32(uint x) { return int(x); }\n\
         inline uint zsl_byte(uint x, uint i) { return (x >> (8u * (i & 3u))) & 0xffu; }\n\
         inline int zsl_sbyte(uint x, uint i) { return int(char((x >> (8u * (i & 3u))) & 0xffu)); }\n",
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
// Rust `as` from f32: NaN -> 0, saturate at the i32 bounds, truncate.
inline int zsl_f32_to_i32(float x) {
    if (x != x) return 0;
    if (x >= 2147483648.0f) return 2147483647;
    if (x <= -2147483648.0f) return (-2147483647 - 1);
    return int(x);
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

    let shared_bf16 = shared_elems.values().any(|t| *t == "ushort");
    if !bf16_buffers.is_empty() || shared_bf16 {
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

    let _ = writeln!(
        s,
        "[[max_total_threads_per_threadgroup({})]] kernel void {ENTRY}(",
        occupancy.launch_bounds(local_size)
    );
    let mut slot = 0u32;
    for (b, buffer_elem) in &buffers {
        let elem = if atomic_buffers.contains(b) {
            match msl_buffer_elem(module.types.get(*buffer_elem))? {
                "uint" => "atomic_uint",
                "int" => "atomic_int",
                _ => "atomic_float",
            }
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
    s.push_str("    uint3 __zsl_gid [[thread_position_in_grid]],\n");
    s.push_str("    uint3 __zsl_lid [[thread_position_in_threadgroup]],\n");
    s.push_str("    uint3 __zsl_group [[threadgroup_position_in_grid]]");
    // Fragments loaded or stored element-wise index by lane.
    let uses_subgroup_query = stmts_use_subgroup_query(&e.body) || !frags.is_empty();
    if uses_subgroup_query {
        s.push_str(",\n    uint __zsl_simd_size [[threads_per_simdgroup]],\n");
        s.push_str("    uint __zsl_simd_group [[simdgroup_index_in_threadgroup]],\n");
        s.push_str("    uint __zsl_simd_lane [[thread_index_in_simdgroup]]\n) {\n");
    } else {
        s.push_str("\n) {\n");
    }

    let aligned = e.shared.iter().any(|d| d.view_of.is_some());
    for shared in &e.shared {
        let elem = shared_elems[shared.name.as_str()];
        match &shared.view_of {
            None => {
                let attr = if aligned {
                    " __attribute__((aligned(16)))"
                } else {
                    ""
                };
                let _ = writeln!(
                    s,
                    "    threadgroup {elem} {}[{}]{attr};",
                    shared.name, shared.len
                );
            }
            Some(base) => {
                let _ = writeln!(
                    s,
                    "    threadgroup {elem}* {} = (threadgroup {elem}*){base};",
                    shared.name
                );
            }
        }
    }
    if uses_workgroup_reduce {
        let _ = writeln!(
            s,
            "    threadgroup float __zsl_wg_reduce_f32[{block_size}];"
        );
    }

    let type_ctx = ComputeTypeCtx {
        shared: e.shared.iter().map(|d| (d.name.clone(), d.elem)).collect(),
        locals: e.locals.iter().cloned().collect(),
        scalars: scalars
            .iter()
            .map(|(name, ty)| ((*name).to_string(), *ty))
            .collect(),
        buffers: buffers
            .iter()
            .map(|(name, ty)| ((*name).to_string(), *ty))
            .collect(),
    };
    let ctx = Ctx {
        types: RefCell::new(module.types.clone()),
        type_ctx,
        locals,
        frags,
        buffer_elems,
        shared_elems,
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
    /// `half` (f16 a/b) and `float` (f32 a/b and accumulator) are `simdgroup_matrix` tiles. `char`
    /// (i8 a/b) is held as `half` tiles, exact for every i8. `int` (i32 accumulator) is `int2` per
    /// tile in the lane layout of `zsl_frag_row`/`zsl_frag_col`, since simdgroup matrices have no
    /// integer type.
    elem: &'static str,
}

impl FragTiles {
    /// Elements are moved one at a time through the lane layout, not `simdgroup_load`/`store`.
    fn by_element(self) -> bool {
        matches!(self.elem, "char" | "int")
    }

    /// Scalar type of one element in storage.
    fn storage(self) -> &'static str {
        match self.elem {
            "char" | "half" => "half",
            other => other,
        }
    }

    fn declaration(self, name: &str) -> String {
        if self.elem == "int" {
            format!("int2 {name}[{TILES}];")
        } else {
            format!(
                "simdgroup_matrix<{}, 8, 8> {name}[{TILES}];",
                self.storage()
            )
        }
    }
}

const TILES: usize = 4;

struct Ctx<'a> {
    type_ctx: ComputeTypeCtx,
    types: RefCell<TypeArena>,
    locals: HashMap<&'a str, &'static str>,
    frags: HashMap<&'a str, FragTiles>,
    buffer_elems: HashMap<&'a str, &'static str>,
    shared_elems: HashMap<&'a str, &'static str>,
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

/// Longest constant-trip loop that is fully unrolled.
const UNROLL_MAX: u32 = 8;

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
                let mut lines = vec![frag.declaration(name)];
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
            } else if ctx.buffer_elems.get(buf.as_str()) == Some(&"char") {
                format!("char({val})")
            } else if ctx.buffer_elems.get(buf.as_str()) == Some(&"uchar") {
                format!("uchar({val})")
            } else {
                val
            };
            let _ = writeln!(s, "{buf}[{idx}] = {val};");
        }
        IrStmt::AtomicAdd { buf, index, value } => {
            indent(s, depth);
            // The value converts to the buffer's element type.
            let cty = ctx
                .type_ctx
                .buffers
                .get(buf.as_str())
                .map_or("float", |&t| scalar_c_type(ctx, t));
            let _ = writeln!(
                s,
                "atomic_fetch_add_explicit(&{}[uint({})], {cty}({}), memory_order_relaxed);",
                buf,
                emit_expr(ctx, index)?,
                emit_expr(ctx, value)?
            );
        }
        IrStmt::AssignShared { name, index, value } => {
            indent(s, depth);
            let val = emit_expr(ctx, value)?;
            let val = match ctx.shared_elems.get(name.as_str()).copied() {
                Some("ushort") => format!("f32_to_bf16({val})"),
                Some("half") => format!("half({val})"),
                Some("char") => format!("char({val})"),
                Some("uchar") => format!("uchar({val})"),
                _ => val,
            };
            let _ = writeln!(s, "{}[{}] = {};", name, emit_expr(ctx, index)?, val);
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
        IrStmt::SharedBarrier => {
            indent(s, depth);
            s.push_str("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        }
        IrStmt::Barrier => {
            indent(s, depth);
            if ctx.has_images {
                s.push_str(
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device | mem_flags::mem_texture);\n",
                );
            } else {
                s.push_str(
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n",
                );
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
            // Short constant loops unroll fully: Metal's compiler otherwise keeps them as loops,
            // which costs simdgroup-matrix code its register allocation.
            if let (IrExpr::LitU32(lo), IrExpr::LitU32(hi)) = (lo, hi)
                && hi.saturating_sub(*lo) <= UNROLL_MAX
            {
                indent(s, depth);
                s.push_str("_Pragma(\"clang loop unroll(full)\")\n");
            }
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
            } else if ctx.buffer_elems.get(buf.as_str()) == Some(&"char") {
                format!("int({inner})")
            } else if ctx.buffer_elems.get(buf.as_str()) == Some(&"uchar") {
                format!("uint({inner})")
            } else {
                inner
            }
        }
        IrExpr::SharedLoad { name, index } => {
            let inner = format!("{name}[{}]", emit_expr(ctx, index)?);
            match ctx.shared_elems.get(name.as_str()).copied() {
                Some("ushort") => format!("bf16_to_f32({inner})"),
                Some("half") => format!("float({inner})"),
                Some("char") => format!("int({inner})"),
                Some("uchar") => format!("uint({inner})"),
                _ => inner,
            }
        }
        IrExpr::GlobalId(c) => format!("__zsl_gid.{}", component(*c)),
        IrExpr::LocalId(c) => format!("__zsl_lid.{}", component(*c)),
        IrExpr::GroupId(c) => format!("__zsl_group.{}", component(*c)),
        IrExpr::Builtin { func, args } => {
            let mut a = Vec::with_capacity(args.len());
            for arg in args {
                a.push(emit_expr(ctx, arg)?);
            }
            match func {
                // Overloads need one operand type: convert to the call's type.
                BuiltinFn::Sign if scalar_c_type(ctx, expr_type(ctx, expr)) != "float" => {
                    if scalar_c_type(ctx, expr_type(ctx, expr)) == "int" {
                        format!("(int({0} > 0) - int({0} < 0))", a[0])
                    } else {
                        format!("uint({} != 0u)", a[0])
                    }
                }
                BuiltinFn::Abs | BuiltinFn::Min | BuiltinFn::Max | BuiltinFn::Clamp => {
                    let ty = expr_type(ctx, expr);
                    let cty = scalar_c_type(ctx, ty);
                    let conv: Vec<String> = args
                        .iter()
                        .zip(&a)
                        .map(|(arg, s)| {
                            if expr_type(ctx, arg) == ty {
                                s.clone()
                            } else {
                                format!("{cty}({s})")
                            }
                        })
                        .collect();
                    format!("{}({})", func.name(), conv.join(", "))
                }
                BuiltinFn::U32 => format!("uint({})", a[0]),
                BuiltinFn::FragZeroAcc | BuiltinFn::Mma => {
                    return Err(frag_rejected(
                        func.name(),
                        "a fragment op must be the whole initializer of a `let` or an assignment",
                    ));
                }
                // 256 elements over a 32-lane simdgroup.
                BuiltinFn::FragLen => "8u".to_string(),
                BuiltinFn::FragRow => format!("zsl_frag_row(__zsl_simd_lane, {})", a[1]),
                BuiltinFn::FragCol => format!("zsl_frag_col(__zsl_simd_lane, {})", a[1]),
                BuiltinFn::FragGet => {
                    let f = frag_local(ctx, &args[0], "frag_get")?;
                    match ctx.frags.get(f).map(|t| t.elem) {
                        Some("int") => format!("{}[({}) / 2u][({}) % 2u]", a[0], a[1], a[1]),
                        Some("char") => format!("int(zsl_frag_get({}, {}))", a[0], a[1]),
                        _ => format!("zsl_frag_get({}, {})", a[0], a[1]),
                    }
                }
                BuiltinFn::FragSet => {
                    return Err(frag_rejected(
                        "frag_set",
                        "a fragment op must be the whole initializer of a `let` or an assignment",
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
                BuiltinFn::Select => format!("select({}, {}, {})", a[2], a[1], a[0]),
                BuiltinFn::F16Round => format!("float(half({}))", a[0]),
                BuiltinFn::I32 => {
                    let f32_id = ctx.types.borrow_mut().f32_id();
                    if expr_type(ctx, &args[0]) == f32_id {
                        format!("zsl_f32_to_i32({})", a[0])
                    } else {
                        format!("int({})", a[0])
                    }
                }
                BuiltinFn::RoundEven => format!("rint({})", a[0]),
                BuiltinFn::FastExp => format!("fast::exp({})", a[0]),
                BuiltinFn::BytePerm => format!("zsl_byte_perm({}, {}, {})", a[0], a[1], a[2]),
                BuiltinFn::F16Lo => format!("zsl_f16_lo({})", a[0]),
                BuiltinFn::F16Hi => format!("zsl_f16_hi({})", a[0]),
                BuiltinFn::PackF16x2 => format!("zsl_pack_f16x2({}, {})", a[0], a[1]),
                BuiltinFn::BitcastF32 => format!("zsl_bitcast_f32({})", a[0]),
                BuiltinFn::BitcastU32 => format!("zsl_bitcast_u32({})", a[0]),
                BuiltinFn::BitcastI32 => format!("zsl_bitcast_i32({})", a[0]),
                BuiltinFn::Byte => format!("zsl_byte({}, {})", a[0], a[1]),
                BuiltinFn::SByte => format!("zsl_sbyte({}, {})", a[0], a[1]),
                BuiltinFn::SubgroupAny => format!("simd_any({})", a[0]),
                BuiltinFn::SubgroupAll => format!("simd_all({})", a[0]),
                BuiltinFn::SubgroupBallot => format!("zsl_ballot({})", a[0]),
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
                    let cty = scalar_c_type(ctx, expr_type(ctx, expr));
                    format!(
                        "zsl_wg_reduce<{op}, {cty}>(__zsl_wg_reduce_f32, __zsl_lid.x, {}u, {cty}({}))",
                        ctx.block_size, a[0]
                    )
                }
                BuiltinFn::SubgroupSize => "__zsl_simd_size".to_string(),
                BuiltinFn::SubgroupId => "__zsl_simd_group".to_string(),
                BuiltinFn::SubgroupLocalId => "__zsl_simd_lane".to_string(),
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
        IrExpr::Binary {
            op: IrBinOp::Rem,
            lhs,
            rhs,
        } => {
            let (l, r) = (emit_expr(ctx, lhs)?, emit_expr(ctx, rhs)?);
            let (lt, rt) = (expr_type(ctx, lhs), expr_type(ctx, rhs));
            let (f32_id, i32_id, u32_id) = {
                let mut types = ctx.types.borrow_mut();
                (types.f32_id(), types.i32_id(), types.u32_id())
            };
            if lt == f32_id {
                format!("fmod({l}, {r})")
            } else if lt == i32_id && rt == u32_id {
                format!("({l} % ((int)({r})))")
            } else if lt == u32_id && rt == i32_id {
                format!("(((int)({l})) % {r})")
            } else {
                format!("({l} % {r})")
            }
        }
        IrExpr::Binary { op, lhs, rhs } => {
            let mut l = emit_expr(ctx, lhs)?;
            let mut r = emit_expr(ctx, rhs)?;
            // C++ makes `int op uint` unsigned. ZSL makes it i32, as SPIR-V does.
            if !matches!(
                op,
                IrBinOp::BitAnd
                    | IrBinOp::BitOr
                    | IrBinOp::BitXor
                    | IrBinOp::Shl
                    | IrBinOp::Shr
                    | IrBinOp::And
                    | IrBinOp::Or
            ) {
                let (lt, rt) = (expr_type(ctx, lhs), expr_type(ctx, rhs));
                let (i32_id, u32_id) = {
                    let mut types = ctx.types.borrow_mut();
                    (types.i32_id(), types.u32_id())
                };
                if lt == i32_id && rt == u32_id {
                    r = format!("int({r})");
                } else if lt == u32_id && rt == i32_id {
                    l = format!("int({l})");
                }
            }
            format!("({l} {} {r})", binop(*op)?)
        }
        IrExpr::FieldAccess { base, component: c } => {
            format!("{}.{}", emit_expr(ctx, base)?, component(*c))
        }
        IrExpr::Construct { ty, dim, args } => {
            let is_u32_vec = ty.is_some_and(|t| u32_vec_lanes(&ctx.types.borrow(), t).is_some());
            if !is_u32_vec {
                return Err(Diagnostic::new(
                    DiagnosticCode::Internal,
                    "MSL compute backend: only u32x2 and u32x4 constructors reach compute",
                ));
            }
            let mut args_s = Vec::with_capacity(args.len());
            for a in args {
                args_s.push(emit_expr(ctx, a)?);
            }
            format!("uint{}({})", dim, args_s.join(", "))
        }
        IrExpr::Input(_)
        | IrExpr::Swizzle { .. }
        | IrExpr::Sample { .. }
        | IrExpr::ArrayLit(_)
        | IrExpr::Index { .. } => {
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
        2 => "z",
        _ => "w",
    }
}

/// MSL name of a scalar `f32`, `i32` or `u32` type.
fn scalar_c_type(ctx: &Ctx<'_>, ty: TypeId) -> &'static str {
    let mut types = ctx.types.borrow_mut();
    if ty == types.i32_id() {
        "int"
    } else if ty == types.u32_id() {
        "uint"
    } else {
        "float"
    }
}

fn expr_type(ctx: &Ctx<'_>, expr: &IrExpr) -> TypeId {
    let mut types = ctx.types.borrow_mut();
    type_of_compute_expr(expr, &ctx.type_ctx, &mut types)
}

fn binop(op: IrBinOp) -> Result<&'static str, Diagnostic> {
    Ok(match op {
        IrBinOp::Add => "+",
        IrBinOp::Sub => "-",
        IrBinOp::Mul => "*",
        IrBinOp::Div => "/",
        IrBinOp::Rem => "%",
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

/// The fragment local `e` names, for the element builtins.
fn frag_local<'e>(ctx: &Ctx<'_>, e: &'e IrExpr, what: &str) -> Result<&'e str, Diagnostic> {
    match e {
        IrExpr::Local(n) if ctx.frags.contains_key(n.as_str()) => Ok(n),
        _ => Err(frag_rejected(
            what,
            "the first argument must be a fragment local",
        )),
    }
}

fn frag_rejected(what: &str, why: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Unsupported,
        format!("MSL backend: {what} is not supported: {why}"),
    )
}

/// f16 and i8 a/b fragments, f32 and i32 accumulators.
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
            FragUse::A | FragUse::B | FragUse::Acc,
        ) => "float",
        (
            Type::Int {
                width: IntWidth::W8,
                ..
            },
            FragUse::A | FragUse::B,
        ) => "char",
        (
            Type::Int {
                width: IntWidth::W32,
                ..
            },
            FragUse::Acc,
        ) => "int",
        _ => {
            return Err(frag_rejected(
                "fragment",
                "a/b fragments are f16, f32 or i8 and accumulators are f32 or i32 on MSL",
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
            src,
            offset,
            ld,
            trans,
            ..
        } => {
            let off = emit_expr(ctx, offset)?;
            let ld = emit_expr(ctx, ld)?;
            // f32 sources convert per element: simdgroup_load cannot narrow.
            let f32_source = frag.elem == "half"
                && match src {
                    FragSrc::Buffer(n) => ctx.buffer_elems.get(n.as_str()) == Some(&"float"),
                    FragSrc::Shared(n) => ctx.shared_elems.get(n.as_str()) == Some(&"float"),
                };
            if frag.by_element() || f32_source {
                return frag_load_elements(ctx, name, frag, src, &off, &ld, *trans);
            }
            if *trans && !matches!(frag.elem, "half" | "float") {
                return Err(frag_rejected(
                    "frag_load_b_t",
                    "the transposed B load takes f16 or f32 operands",
                ));
            }
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
                    let elem = ctx.shared_elems.get(tile.as_str()).copied();
                    let fits = if frag.elem == "half" {
                        elem == Some("half")
                    } else {
                        frag.elem == "float" && elem == Some("float")
                    };
                    if !fits {
                        return Err(frag_rejected(
                            &format!("frag load from shared `{tile}`"),
                            "a/b load f16 or f32 shared arrays, and accumulators load f32 shared arrays",
                        ));
                    }
                    format!("({tile} + ({off}))")
                }
            };
            Ok((0..TILES)
                .map(|t| {
                    let (r, c) = tile_origin(t);
                    if *trans {
                        format!(
                            "simdgroup_load({name}[{t}], {base} + {c}u * ({ld}) + {r}u, ({ld}), ulong2(0, 0), true);"
                        )
                    } else {
                        format!("simdgroup_load({name}[{t}], {base} + {r}u * ({ld}) + {c}u, ({ld}));")
                    }
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
            .map(|t| {
                if frag.elem == "int" {
                    format!("{name}[{t}] = int2(0);")
                } else {
                    format!(
                        "{name}[{t}] = simdgroup_matrix<{}, 8, 8>(0);",
                        frag.storage()
                    )
                }
            })
            .collect()),
        IrExpr::Builtin {
            func: BuiltinFn::FragSet,
            args,
        } => {
            frag_local(ctx, &args[0], "frag_set")?;
            // Copies the source fragment unless it is `name` itself.
            let mut lines = frag_fill_lines(ctx, name, frag, &args[0])?;
            let i = emit_expr(ctx, &args[1])?;
            let x = emit_expr(ctx, &args[2])?;
            let slot = if frag.elem == "int" {
                format!("{name}[({i}) / 2u][({i}) % 2u]")
            } else {
                format!("{name}[({i}) / 2u].thread_elements()[({i}) % 2u]")
            };
            lines.push(format!("{slot} = {}({x});", frag.storage()));
            Ok(lines)
        }
        IrExpr::Local(src) if src == name => Ok(Vec::new()),
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

/// i8 and i32 fragments, and f16 fragments from f32 arrays, one element at a time through the
/// lane layout. `frag_load_b_t` reads element `(r, c)` at `c * ld + r`.
fn frag_load_elements(
    ctx: &Ctx<'_>,
    name: &str,
    frag: FragTiles,
    src: &FragSrc,
    off: &str,
    ld: &str,
    trans: bool,
) -> Result<Vec<String>, Diagnostic> {
    let (name_src, elem) = match src {
        FragSrc::Buffer(n) => (n, ctx.buffer_elems.get(n.as_str()).copied()),
        FragSrc::Shared(n) => (n, ctx.shared_elems.get(n.as_str()).copied()),
    };
    let base = match (frag.elem, src, elem) {
        (e, _, Some(b)) if e == b => name_src.clone(),
        ("half", _, Some("float")) => name_src.clone(),
        // Packed i8 in a 32-bit buffer: the offset counts bytes.
        ("char", FragSrc::Buffer(_), Some("uint" | "int")) => {
            format!("((device const char*){name_src})")
        }
        _ => {
            return Err(frag_rejected(
                &format!("frag load from `{name_src}`"),
                "i8 fragments load i8 arrays or packed 32-bit buffers, and i32 accumulators load i32 arrays",
            ));
        }
    };
    Ok((0..TILES * 2)
        .map(|i| {
            let (r, c) = (
                format!("zsl_frag_row(__zsl_simd_lane, {i}u)"),
                format!("zsl_frag_col(__zsl_simd_lane, {i}u)"),
            );
            let at = if trans {
                format!("({off}) + {c} * ({ld}) + {r}")
            } else {
                format!("({off}) + {r} * ({ld}) + {c}")
            };
            let (t, j) = (i / 2, i % 2);
            if frag.elem == "int" {
                format!("{name}[{t}][{j}] = {base}[{at}];")
            } else {
                format!("{name}[{t}].thread_elements()[{j}] = half({base}[{at}]);")
            }
        })
        .collect())
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
    let int_mma = a_frag.elem == "char" && b_frag.elem == "char" && frag.elem == "int";
    let float_mma = a_frag.elem == b_frag.elem
        && matches!(a_frag.elem, "half" | "float")
        && frag.elem == "float";
    if frag.usage != FragUse::Acc || !(int_mma || float_mma) {
        return Err(frag_rejected(
            "mma",
            "mma takes f16 or f32 a and b into an f32 accumulator, or i8 a and b into an i32 one",
        ));
    }
    let mut lines = match c {
        IrExpr::Local(cn) => {
            let c_frag =
                ctx.frags.get(cn.as_str()).copied().ok_or_else(|| {
                    frag_rejected("mma", "the accumulator must be a fragment local")
                })?;
            if c_frag.usage != FragUse::Acc || c_frag.elem != frag.elem {
                return Err(frag_rejected(
                    "mma",
                    "the accumulator must have the result's type",
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
            .map(|t| {
                if int_mma {
                    format!("{name}[{t}] = int2(0);")
                } else {
                    format!("{name}[{t}] = simdgroup_matrix<float, 8, 8>(0);")
                }
            })
            .collect(),
        _ => {
            return Err(frag_rejected(
                "mma",
                "the accumulator must be a fragment local or frag_zero_acc()",
            ));
        }
    };
    if int_mma {
        // Each 8x8x16 partial sum is at most 16 * 128 * 128, exact in f32, then added as i32.
        for i in 0..2 {
            for j in 0..2 {
                lines.push(format!(
                    "{{ simdgroup_matrix<float, 8, 8> p = simdgroup_matrix<float, 8, 8>(0); \
                     simdgroup_multiply_accumulate(p, {a}[{}], {b}[{}], p); \
                     simdgroup_multiply_accumulate(p, {a}[{}], {b}[{}], p); \
                     {name}[{}] += int2(int(p.thread_elements()[0]), int(p.thread_elements()[1])); }}",
                    2 * i,
                    j,
                    2 * i + 1,
                    2 + j,
                    2 * i + j,
                ));
            }
        }
        return Ok(lines);
    }
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
    let acc_frag = ctx
        .frags
        .get(acc.as_str())
        .copied()
        .filter(|f| f.usage == FragUse::Acc)
        .ok_or_else(|| frag_rejected("frag_store", "only accumulators can be stored"))?;
    let off = emit_expr(ctx, offset)?;
    let ld = emit_expr(ctx, ld)?;
    if acc_frag.elem == "int" {
        let (dst, elem) = match src {
            FragSrc::Buffer(n) => (n, ctx.buffer_elems.get(n.as_str()).copied()),
            FragSrc::Shared(n) => (n, ctx.shared_elems.get(n.as_str()).copied()),
        };
        if elem != Some("int") {
            return Err(frag_rejected(
                &format!("frag_store to `{dst}`"),
                "i32 accumulators are stored to i32 buffers or shared arrays",
            ));
        }
        return Ok((0..TILES * 2)
            .map(|i| {
                format!(
                    "{dst}[({off}) + zsl_frag_row(__zsl_simd_lane, {i}u) * ({ld}) + zsl_frag_col(__zsl_simd_lane, {i}u)] = {acc}[{}][{}];",
                    i / 2,
                    i % 2
                )
            })
            .collect());
    }
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
        FragSrc::Shared(tile) => {
            if ctx.shared_elems.get(tile.as_str()).copied() != Some("float") {
                return Err(frag_rejected(
                    &format!("frag_store to shared `{tile}`"),
                    "accumulators are stored to f32 shared arrays",
                ));
            }
            format!("({tile} + ({off}))")
        }
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
        Type::Vector { lanes: 2, .. } => Ok("uint2"),
        Type::Vector { lanes: 4, .. } => Ok("uint4"),
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
        Type::Int {
            width: IntWidth::W8,
            signedness: Signedness::Signed,
        } => Ok("char"),
        Type::Int {
            width: IntWidth::W8,
            signedness: Signedness::Unsigned,
        } => Ok("uchar"),
        Type::Float {
            format: FloatFormat::F32,
        } => Ok("float"),
        Type::Float {
            format: FloatFormat::Bf16,
        } => Ok("ushort"),
        Type::Float {
            format: FloatFormat::F16,
        } => Ok("half"),
        Type::Vector { lanes: 2, .. } => Ok("uint2"),
        Type::Vector { lanes: 4, .. } => Ok("uint4"),
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
        assert!(
            m.source
                .contains("uint3 __zsl_gid [[thread_position_in_grid]]")
        );
        assert!(m.source.contains("y[i] = (y[i] + (pc.alpha * x[i]))"));
    }

    /// Kernel names that match the stage inputs must not shadow them.
    #[test]
    fn user_names_do_not_shadow_stage_inputs() {
        let m = msl(r#"
            @workgroup_size(64)
            kernel shadow(out: device mut buffer<u32>, id: global_id) {
                let gid = id.x
                let lid = local_id().x
                let group = gid / 64u
                let simd_lane = subgroup_local_id()
                out[gid] = lid + group + simd_lane
            }
        "#);
        assert!(m.source.contains("uint lid = __zsl_lid.x"), "{}", m.source);
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
