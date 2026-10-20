//! Lowers a tile kernel to one ZSL compute kernel. One workgroup per output tile. Shapes,
//! chosen by `crate::schedule`:
//! - `Naive` (O0): scalar per output element, reads globals. The reference.
//! - `Staged`: A and B blocks per k step go to shared arrays, then scalar FMAs.
//! - `Mma`: staged, with `mma` on 16x16 fragments loaded from shared.

use std::fmt::Write as _;

use crate::TileError;
use crate::ir::{Affine, Block, DType, Op, Stmt, Tile, TileKernel};

/// Naive workgroup size for tiles of at least this many elements.
const MAX_WORKGROUP: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Naive,
    Staged,
    Mma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plan {
    pub mode: Mode,
    pub workgroup: usize,
    /// Waves per workgroup (`Mma` only; 1 otherwise).
    pub waves: usize,
    /// Lanes per wave the code assumes (`Mma` only; 0 otherwise).
    pub wave_size: usize,
    /// Staging for `Mma` only.
    ///
    /// Without native shared arrays (Vulkan), f16 stages as f32, and B stages K x
    /// N for `frag_load_b`.
    pub native_shared: bool,
    /// `Mma` only: `wave_rows * wave_cols = waves`. Stages per operand: 1, or 2 for
    /// double buffering.
    pub wave_rows: usize,
    pub wave_cols: usize,
    pub stages: usize,
    /// `Mma` only: full interior blocks are staged with 16-byte global loads through `u32x4` alias
    /// params (see [`vector_stage`]).
    pub vector_loads: bool,
    /// `Mma` only: k groups (grid `z`), each writing a partial tile to the workspace. 1 = no split;
    /// above 1, [`reduce_zsl`] runs second.
    pub split_k: usize,
}

impl Plan {
    /// Naive form. Needs no target.
    pub(crate) fn naive(kernel: &TileKernel) -> Self {
        let tile_elems = kernel.tile[0] * kernel.tile[1];
        let workgroup = MAX_WORKGROUP.min(tile_elems.div_ceil(32) * 32);
        Plan {
            mode: Mode::Naive,
            workgroup,
            waves: 1,
            wave_size: 0,
            native_shared: false,
            wave_rows: 1,
            wave_cols: 1,
            stages: 1,
            vector_loads: false,
            split_k: 1,
        }
    }
}

/// A k loop the staged lowerings can tile: loads plus one `dot` into a zeroed accumulator, then an
/// elementwise epilogue and stores.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Parts {
    pub extent: usize,
    pub step: usize,
    /// Operand blocks of the dot: `a` is `[BM, BK]`, `b` is `[BN, BK]`.
    pub a: Block,
    pub b: Block,
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub dot_out: Tile,
    pub loop_at: usize,
}

pub(crate) fn analyze(kernel: &TileKernel) -> Result<Parts, String> {
    let loops: Vec<usize> = kernel
        .body
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, Stmt::Loop { .. }))
        .map(|(i, _)| i)
        .collect();
    let [loop_at] = loops[..] else {
        return Err("staged lowering needs exactly one k loop".to_string());
    };
    for s in &kernel.body[..loop_at] {
        if !matches!(s, Stmt::Def(t, Op::Zeros) if kernel.values[t.0].def == Op::Zeros) {
            return Err("staged lowering needs only zeros before the k loop".to_string());
        }
    }
    for s in &kernel.body[loop_at + 1..] {
        if matches!(s, Stmt::Loop { .. } | Stmt::Def(_, Op::Dot { .. })) {
            return Err("staged lowering needs one dot, inside the k loop".to_string());
        }
    }
    let Stmt::Loop { extent, step, body } = &kernel.body[loop_at] else {
        return Err("staged lowering needs a k loop".to_string());
    };
    let (extent, step) = (*extent, *step);
    let mut dot = None;
    for s in body {
        match s {
            Stmt::Def(_, Op::Load(_)) => {}
            Stmt::Def(t, Op::Dot { a, b, acc }) if dot.is_none() => {
                dot = Some((*t, *a, *b, *acc));
            }
            _ => return Err("the k loop body must be loads and one dot".to_string()),
        }
    }
    let Some((dot_out, a, b, acc)) = dot else {
        return Err("the k loop has no dot".to_string());
    };
    if kernel.values[acc.0].def != Op::Zeros {
        return Err("the dot accumulator must be zeros".to_string());
    }
    let block_of = |t: Tile| match kernel.values[t.0].def {
        Op::Load(block) => Some(block),
        _ => None,
    };
    let (Some(ab), Some(bb)) = (block_of(a), block_of(b)) else {
        return Err("dot operands must be loads".to_string());
    };
    let tile = kernel.tile;
    if ab.rows != Affine::grid_m() || ab.cols != Affine::k() || ab.shape != [tile[0], step] {
        return Err("dot lhs must be the [BM, BK] block at (grid_m, k)".to_string());
    }
    if bb.rows != Affine::grid_n() || bb.cols != Affine::k() || bb.shape != [tile[1], step] {
        return Err("dot rhs must be the [BN, BK] block at (grid_n, k)".to_string());
    }
    let a_dtype = kernel.tensors[ab.tensor.0].dtype;
    let b_dtype = kernel.tensors[bb.tensor.0].dtype;
    Ok(Parts {
        extent,
        step,
        a: ab,
        b: bb,
        a_dtype,
        b_dtype,
        dot_out,
        loop_at,
    })
}

/// Tensors with `u32x4` alias params, in param order. [`crate::TileProgram`] binds each alias to
/// its tensor's host buffer.
pub(crate) fn alias_tensors(kernel: &TileKernel, plan: &Plan) -> Vec<usize> {
    match plan.mode {
        Mode::Mma => analyze(kernel)
            .map(|parts| vector_tensors(kernel, &parts, plan))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

pub(crate) fn zsl_source(kernel: &TileKernel, plan: &Plan) -> Result<String, TileError> {
    match plan.mode {
        Mode::Naive => naive_source(kernel, plan),
        Mode::Staged | Mode::Mma => {
            let parts = analyze(kernel).map_err(lower_err)?;
            staged_source(kernel, plan, &parts)
        }
    }
}

/// Buffer element type. i8 is one byte per element.
fn buffer_elem(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
    }
}

/// Local type of a tile element. i8 loads sign-extend to i32.
fn scalar(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::F32 => "f32",
        DType::I32 | DType::I8 => "i32",
        DType::U32 => "u32",
    }
}

fn zero(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::F32 => "0.0",
        _ => "0",
    }
}

fn affine(a: Affine, kt: &str) -> String {
    let mut terms = Vec::new();
    if a.c0 != 0 || (a.c_m == 0 && a.c_n == 0 && a.c_k == 0) {
        terms.push(a.c0.to_string());
    }
    for (c, name) in [(a.c_m, "gm"), (a.c_n, "gn"), (a.c_k, kt)] {
        match c {
            0 => {}
            1 => terms.push(name.to_string()),
            c => terms.push(format!("{c} * {name}")),
        }
    }
    format!("({})", terms.join(" + "))
}

/// `[1, cols]` values are the same in every row: the row is always 0.
fn row_arg(shape0: usize, r: &str) -> String {
    if shape0 == 1 {
        "0".to_string()
    } else {
        r.to_string()
    }
}

/// Elements of `dtype` in one 16-byte vector.
fn vec_lanes(dtype: DType) -> usize {
    match dtype {
        DType::F16 => 8,
        DType::I8 => 16,
        DType::F32 | DType::I32 | DType::U32 => 4,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageForm {
    Rows,
    /// Row-major. Full blocks: 16 bytes per step when [`vector_stage`] allows it, else scalar.
    Vector,
    /// Column-major (the layout `frag_load_b` reads).
    Transposed,
}

/// Full blocks use 16-byte loads only if tensor and block rows are whole vectors (each row
/// 16-byte aligned).
fn vector_stage(kernel: &TileKernel, block: Block, cols: usize) -> bool {
    let decl = &kernel.tensors[block.tensor.0];
    let lanes = vec_lanes(decl.dtype);
    decl.shape[1].is_multiple_of(lanes) && cols.is_multiple_of(lanes)
}

fn alias_name(kernel: &TileKernel, t: usize) -> String {
    format!("b_{}_v", kernel.tensors[t].name)
}

fn vector_tensors(kernel: &TileKernel, parts: &Parts, plan: &Plan) -> Vec<usize> {
    let mut out = Vec::new();
    if !plan.vector_loads {
        return out;
    }
    for block in [parts.a, parts.b] {
        if vector_stage(kernel, block, parts.step) && !out.contains(&block.tensor.0) {
            out.push(block.tensor.0);
        }
    }
    out
}

/// Params: one per tensor, then `u32x4` aliases, then the workspace (split-K passes).
fn header(
    kernel: &TileKernel,
    workgroup: usize,
    aliases: &[usize],
    workspace: bool,
    src: &mut String,
) {
    let mut params: Vec<String> = kernel
        .tensors
        .iter()
        .map(|t| {
            let mutability = if t.is_output { "mut " } else { "" };
            format!(
                "b_{}: device {mutability}buffer<{}>",
                t.name,
                buffer_elem(t.dtype)
            )
        })
        .collect();
    params.extend(
        aliases
            .iter()
            .map(|&t| format!("{}: device buffer<u32x4>", alias_name(kernel, t))),
    );
    if workspace {
        params.push("ws: device mut buffer<f32>".to_string());
    }
    let _ = writeln!(src, "@workgroup_size({workgroup})");
    let _ = writeln!(src, "kernel {}({}) {{", kernel.name, params.join(", "));
    let _ = writeln!(src, "let gn = group_id().x");
    let _ = writeln!(src, "let gm = group_id().y");
}

fn naive_source(kernel: &TileKernel, plan: &Plan) -> Result<String, TileError> {
    let tile_elems = kernel.tile[0] * kernel.tile[1];
    let workgroup = plan.workgroup;
    let iters = tile_elems.div_ceil(workgroup);

    let mut g = Gen {
        kernel,
        values: vec![None; kernel.values.len()],
        fresh: 0,
        kt: "kt".to_string(),
    };
    let mut body = String::new();
    for stmt in &kernel.body {
        g.stmt(&mut body, stmt)?;
    }

    let mut src = String::new();
    header(kernel, workgroup, &[], false, &mut src);
    let _ = writeln!(src, "for it in 0..{iters} {{");
    let _ = writeln!(src, "let e = local_id().x + it * {workgroup}");
    let _ = writeln!(src, "if e < {tile_elems} {{");
    let _ = writeln!(src, "let tr = e / {}", kernel.tile[1]);
    let _ = writeln!(src, "let tc = e % {}", kernel.tile[1]);
    src.push_str(&body);
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    Ok(src)
}

/// The k loop stages A and B to shared memory (masked, zero outside the tensor), then runs a
/// barrier, compute, and barrier. The epilogue runs the kernel's ops per output element.
fn staged_source(kernel: &TileKernel, plan: &Plan, parts: &Parts) -> Result<String, TileError> {
    if plan.mode == Mode::Mma {
        return mma_source(kernel, plan, parts);
    }
    let [bm, bn] = kernel.tile;
    let bk = parts.step;
    let wg = plan.workgroup;
    let elems = bm * bn;
    let per_thread = elems.div_ceil(wg);
    let ksteps = parts.extent.div_ceil(bk);

    let mut g = Gen::new(kernel);
    let mut src = String::new();
    header(kernel, wg, &[], false, &mut src);
    let _ = writeln!(src, "shared sa: array<f32, {}>", bm * bk);
    let _ = writeln!(src, "shared sb: array<f32, {}>", bk * bn);
    let _ = writeln!(src, "let lid = local_id().x");
    for p in 0..per_thread {
        let _ = writeln!(src, "let acc{p}: f32 = 0.0");
    }

    let _ = writeln!(src, "for kt in 0..{ksteps} {{");
    g.stage(&mut src, "sa", parts.a, [bm, bk], StageForm::Rows, wg);
    g.stage(&mut src, "sb", parts.b, [bn, bk], StageForm::Rows, wg);
    let _ = writeln!(src, "barrier()");
    for p in 0..per_thread {
        let _ = writeln!(src, "let e{p} = lid + {}", p * wg);
        let _ = writeln!(src, "if e{p} < {elems} {{");
        let _ = writeln!(src, "let tr = e{p} / {bn}");
        let _ = writeln!(src, "let tc = e{p} % {bn}");
        let _ = writeln!(src, "for kk in 0..{bk} {{");
        let _ = writeln!(
            src,
            "acc{p} = acc{p} + sa[tr * {bk} + kk] * sb[tc * {bk} + kk]"
        );
        let _ = writeln!(src, "}}");
        let _ = writeln!(src, "}}");
    }
    let _ = writeln!(src, "barrier()");
    let _ = writeln!(src, "}}");

    let loop_at = parts.loop_at;
    for p in 0..per_thread {
        let _ = writeln!(src, "let e{p} = lid + {}", p * wg);
        let _ = writeln!(src, "if e{p} < {elems} {{");
        let _ = writeln!(src, "let tr = e{p} / {bn}");
        let _ = writeln!(src, "let tc = e{p} % {bn}");
        g.values[parts.dot_out.0] = Some(format!("acc{p}"));
        for stmt in &kernel.body[loop_at + 1..] {
            g.stmt(&mut src, stmt)?;
        }
        let _ = writeln!(src, "}}");
    }
    let _ = writeln!(src, "}}");
    Ok(src)
}

/// Matrix lowering. Each wave owns a `WM x WN` grid of 16x16 accumulators: A fragments are reused
/// across `WN`, B across `WM`. Two stages: compute one while loading the next. The epilogue goes
/// through a shared `BM x (WN_waves * 16)` slice per fragment column.
///
/// `split_k > 1`: grid gets a z dimension. Group `z` reduces k blocks from `z * ksplit` and writes
/// its partial tile to `ws[(z * M + row) * N + col]`, no epilogue. [`reduce_source`] sums them.
fn mma_source(kernel: &TileKernel, plan: &Plan, parts: &Parts) -> Result<String, TileError> {
    let [bm, bn] = kernel.tile;
    let bk = parts.step;
    let wg = plan.workgroup;
    let split = plan.split_k;
    let ksteps = parts.extent.div_ceil(bk);
    let ksplit = ksteps.div_ceil(split);
    let (out_m, out_n) = output_shape(kernel);
    let (nwr, nwc) = (plan.wave_rows, plan.wave_cols);
    let wm = bm / 16 / nwr;
    let wn = bn / 16 / nwc;
    let wave = plan.wave_size;
    let native = plan.native_shared;
    let (ea, eb) = if native {
        (parts.a_dtype, parts.b_dtype)
    } else {
        let staged = |d: DType| if d == DType::F16 { DType::F32 } else { d };
        (staged(parts.a_dtype), staged(parts.b_dtype))
    };
    let acc = kernel.values[parts.dot_out.0].dtype;
    let (fa, fb, fc) = (
        buffer_elem(parts.a_dtype),
        buffer_elem(parts.b_dtype),
        buffer_elem(acc),
    );
    let scw = nwc * 16;
    let slice = bm * scw;
    let per_slice = slice.div_ceil(wg);
    let double = plan.stages == 2;
    let bufs: Vec<(String, String)> = if double {
        vec![("sa0".into(), "sb0".into()), ("sa1".into(), "sb1".into())]
    } else {
        vec![("sa".into(), "sb".into())]
    };

    let mut g = Gen::new(kernel);
    let mut src = String::new();
    let aliases = vector_tensors(kernel, parts, plan);
    header(kernel, wg, &aliases, split > 1, &mut src);
    if split > 1 {
        let _ = writeln!(src, "let gz = group_id().z");
    }
    let (view_a, view_b) = (
        plan.vector_loads && vector_stage(kernel, parts.a, parts.step),
        plan.vector_loads && vector_stage(kernel, parts.b, parts.step),
    );
    for (sa, sb) in &bufs {
        let _ = writeln!(src, "shared {sa}: array<{}, {}>", buffer_elem(ea), bm * bk);
        let _ = writeln!(src, "shared {sb}: array<{}, {}>", buffer_elem(eb), bk * bn);
        if view_a {
            let _ = writeln!(src, "view {sa}v: array<u32x4> = {sa}");
        }
        if view_b {
            let _ = writeln!(src, "view {sb}v: array<u32x4> = {sb}");
        }
    }
    let _ = writeln!(src, "shared sc: array<{fc}, {slice}>");
    let _ = writeln!(src, "let lid = local_id().x");
    let _ = writeln!(src, "let wv = lid / {wave}");
    let _ = writeln!(src, "let wr = wv / {nwc}");
    let _ = writeln!(src, "let wc = wv % {nwc}");
    for i in 0..wm {
        for j in 0..wn {
            let _ = writeln!(
                src,
                "let c{i}_{j}: frag<{fc}, 16, 16, acc> = frag_zero_acc()"
            );
        }
    }

    let kidx = |x: &str| {
        if split > 1 {
            format!("(gz * {ksplit} + {x})")
        } else {
            x.to_string()
        }
    };
    let stage = |g: &mut Gen, src: &mut String, (sa, sb): &(String, String), kt: &str| {
        g.kt = kidx(kt);
        let form = if plan.vector_loads {
            StageForm::Vector
        } else {
            StageForm::Rows
        };
        g.stage(src, sa, parts.a, [bm, bk], form, wg);
        let form_b = match (native, plan.vector_loads) {
            (false, _) => StageForm::Transposed,
            (true, true) => StageForm::Vector,
            (true, false) => StageForm::Rows,
        };
        g.stage(src, sb, parts.b, [bn, bk], form_b, wg);
    };
    let compute = |src: &mut String, (sa, sb): &(String, String)| {
        let _ = writeln!(src, "for kq in 0..{} {{", bk / 16);
        for i in 0..wm {
            let _ = writeln!(
                src,
                "let fa{i}: frag<{fa}, 16, 16, a> = frag_load_a({sa}, (wr * {wm} + {i}) * {} + kq * 16, {bk})",
                16 * bk
            );
        }
        for j in 0..wn {
            let col = format!("(wc * {wn} + {j}) * 16");
            let load = if native {
                format!("frag_load_b_t({sb}, {col} * {bk} + kq * 16, {bk})")
            } else {
                format!("frag_load_b({sb}, kq * {} + {col}, {bn})", 16 * bn)
            };
            let _ = writeln!(src, "let fb{j}: frag<{fb}, 16, 16, b> = {load}");
        }
        for i in 0..wm {
            for j in 0..wn {
                let _ = writeln!(src, "c{i}_{j} = mma(fa{i}, fb{j}, c{i}_{j})");
            }
        }
        let _ = writeln!(src, "}}");
    };

    if !double {
        let _ = writeln!(src, "for kt in 0..{ksplit} {{");
        stage(&mut g, &mut src, &bufs[0], "kt");
        let _ = writeln!(src, "barrier()");
        compute(&mut src, &bufs[0]);
        let _ = writeln!(src, "barrier()");
        let _ = writeln!(src, "}}");
    } else {
        let pairs = ksplit / 2;
        stage(&mut g, &mut src, &bufs[0], "0");
        let _ = writeln!(src, "barrier()");
        let _ = writeln!(src, "for kp in 0..{pairs} {{");
        let _ = writeln!(src, "let kn: u32 = kp * 2 + 1");
        let _ = writeln!(src, "let kn2: u32 = kp * 2 + 2");
        stage(&mut g, &mut src, &bufs[1], "kn");
        compute(&mut src, &bufs[0]);
        let _ = writeln!(src, "barrier()");
        stage(&mut g, &mut src, &bufs[0], "kn2");
        compute(&mut src, &bufs[1]);
        let _ = writeln!(src, "barrier()");
        let _ = writeln!(src, "}}");
        if ksplit % 2 == 1 {
            compute(&mut src, &bufs[0]);
        }
    }

    let loop_at = parts.loop_at;
    for j in 0..wn {
        for i in 0..wm {
            let _ = writeln!(
                src,
                "frag_store(sc, (wr * {wm} + {i}) * {} + wc * 16, {scw}, c{i}_{j})",
                16 * scw
            );
        }
        let _ = writeln!(src, "barrier()");
        let _ = writeln!(src, "for it in 0..{per_slice} {{");
        let _ = writeln!(src, "let e = lid + it * {wg}");
        let _ = writeln!(src, "if e < {slice} {{");
        let _ = writeln!(src, "let tr = e / {scw}");
        let _ = writeln!(src, "let ts = e % {scw}");
        let _ = writeln!(src, "let tc = ((ts / 16) * {wn} + {j}) * 16 + ts % 16");
        if split > 1 {
            let _ = writeln!(src, "let row = gm * {bm} + tr");
            let _ = writeln!(src, "let col = gn * {bn} + tc");
            let _ = writeln!(src, "if row < {out_m} && col < {out_n} {{");
            let _ = writeln!(
                src,
                "ws[(gz * {out_m} + row) * {out_n} + col] = sc[tr * {scw} + ts]"
            );
            let _ = writeln!(src, "}}");
        } else {
            g.values[parts.dot_out.0] = Some(format!("sc[tr * {scw} + ts]"));
            for stmt in &kernel.body[loop_at + 1..] {
                g.stmt(&mut src, stmt)?;
            }
        }
        let _ = writeln!(src, "}}");
        let _ = writeln!(src, "}}");
        let _ = writeln!(src, "barrier()");
    }
    let _ = writeln!(src, "}}");
    Ok(src)
}

/// Split-K second pass: sums the `split_k` partials in group order, then the epilogue. One
/// workgroup per output tile.
fn reduce_source(kernel: &TileKernel, plan: &Plan, parts: &Parts) -> Result<String, TileError> {
    let [bm, bn] = kernel.tile;
    let wg = plan.workgroup;
    let elems = bm * bn;
    let per = elems.div_ceil(wg);
    let (out_m, out_n) = output_shape(kernel);
    let split = plan.split_k;

    let mut g = Gen::new(kernel);
    g.values[parts.dot_out.0] = Some("rsum".to_string());
    let mut src = String::new();
    header(kernel, wg, &[], true, &mut src);
    let _ = writeln!(src, "let lid = local_id().x");
    let _ = writeln!(src, "for it in 0..{per} {{");
    let _ = writeln!(src, "let e = lid + it * {wg}");
    let _ = writeln!(src, "if e < {elems} {{");
    let _ = writeln!(src, "let tr = e / {bn}");
    let _ = writeln!(src, "let tc = e % {bn}");
    let _ = writeln!(src, "let row = gm * {bm} + tr");
    let _ = writeln!(src, "let col = gn * {bn} + tc");
    let _ = writeln!(src, "if row < {out_m} && col < {out_n} {{");
    let _ = writeln!(src, "let rsum: f32 = 0.0");
    let _ = writeln!(src, "for s in 0..{split} {{");
    let _ = writeln!(src, "rsum = rsum + ws[(s * {out_m} + row) * {out_n} + col]");
    let _ = writeln!(src, "}}");
    for stmt in &kernel.body[parts.loop_at + 1..] {
        g.stmt(&mut src, stmt)?;
    }
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    Ok(src)
}

fn output_shape(kernel: &TileKernel) -> (usize, usize) {
    let shape = kernel
        .tensors
        .iter()
        .find(|t| t.is_output)
        .map(|t| t.shape)
        .unwrap_or([0, 0]);
    (shape[0], shape[1])
}

/// Reduction pass source for a split-K lowering, or `None` when unsplit.
pub(crate) fn reduce_zsl(kernel: &TileKernel, plan: &Plan) -> Result<Option<String>, TileError> {
    if plan.split_k <= 1 {
        return Ok(None);
    }
    let parts = analyze(kernel).map_err(lower_err)?;
    reduce_source(kernel, plan, &parts).map(Some)
}

fn lower_err(msg: String) -> TileError {
    TileError::Lower(msg)
}

struct Gen<'a> {
    kernel: &'a TileKernel,
    /// A local or expression per materialized value. Loads have none: they are read on demand.
    values: Vec<Option<String>>,
    fresh: usize,
    kt: String,
}

impl<'a> Gen<'a> {
    fn new(kernel: &'a TileKernel) -> Self {
        Gen {
            kernel,
            values: vec![None; kernel.values.len()],
            fresh: 0,
            kt: "kt".to_string(),
        }
    }

    fn name(&mut self, prefix: &str) -> String {
        self.fresh += 1;
        format!("{prefix}{}", self.fresh)
    }

    fn stmt(&mut self, out: &mut String, stmt: &Stmt) -> Result<(), TileError> {
        match stmt {
            Stmt::Def(t, op) => self.def(out, *t, *op),
            Stmt::Store { block, value } => self.store(out, block, *value),
            Stmt::Loop { extent, step, body } => {
                let _ = writeln!(out, "for kt in 0..{} {{", extent.div_ceil(*step));
                for s in body {
                    self.stmt(out, s)?;
                }
                let _ = writeln!(out, "}}");
                Ok(())
            }
        }
    }

    fn def(&mut self, out: &mut String, t: Tile, op: Op) -> Result<(), TileError> {
        let dtype = self.kernel.values[t.0].dtype;
        match op {
            Op::Load(_) => {}
            Op::Zeros => {
                let v = format!("v{}", t.0);
                let _ = writeln!(out, "let {v}: {} = {}", scalar(dtype), zero(dtype));
                self.values[t.0] = Some(v);
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                let ea = self.elem_row(out, a, "tr", "tc")?;
                let eb = self.elem_row(out, b, "tr", "tc")?;
                let sym = if matches!(op, Op::Add(..)) { "+" } else { "*" };
                let v = format!("v{}", t.0);
                let _ = writeln!(out, "let {v}: {} = ({ea} {sym} {eb})", scalar(dtype));
                self.values[t.0] = Some(v);
            }
            Op::Cast(a, to) => {
                let src_dtype = self.kernel.values[a.0].dtype;
                let ea = self.elem_row(out, a, "tr", "tc")?;
                let v = format!("v{}", t.0);
                let expr = match (src_dtype, to) {
                    (DType::F32, DType::F16) => format!("f16_round({ea})"),
                    (DType::I32 | DType::U32, DType::F32) => format!("f32({ea})"),
                    (DType::F32, DType::I32) => format!("i32({ea})"),
                    _ => ea,
                };
                let _ = writeln!(out, "let {v}: {} = {expr}", scalar(to));
                self.values[t.0] = Some(v);
            }
            Op::Dot { a, b, acc } => {
                let acc_var = self.values[acc.0]
                    .clone()
                    .ok_or_else(|| lower_err("dot accumulator has no local".to_string()))?;
                let k_len = self.kernel.values[a.0].shape[1];
                let idx = self.name("kk");
                let _ = writeln!(out, "for {idx} in 0..{k_len} {{");
                let ea = self.elem(out, a, "tr", &idx)?;
                let eb = self.elem(out, b, "tc", &idx)?;
                let _ = writeln!(out, "{acc_var} = {acc_var} + {ea} * {eb}");
                let _ = writeln!(out, "}}");
                self.values[t.0] = Some(acc_var);
            }
        }
        Ok(())
    }

    fn elem_row(
        &mut self,
        out: &mut String,
        t: Tile,
        r: &str,
        c: &str,
    ) -> Result<String, TileError> {
        let shape0 = self.kernel.values[t.0].shape[0];
        let r = row_arg(shape0, r);
        self.elem(out, t, &r, c)
    }

    /// `r` and `c` are block-local ZSL u32 expressions. May emit statements into `out`.
    fn elem(&mut self, out: &mut String, t: Tile, r: &str, c: &str) -> Result<String, TileError> {
        let value = &self.kernel.values[t.0];
        match value.def {
            Op::Load(block) => Ok(self.load(out, block, r, c)),
            _ => self.values[t.0]
                .clone()
                .ok_or_else(|| lower_err("value used before it is defined".to_string())),
        }
    }

    /// Masked element read from an input block: out of range reads 0. Returns its local.
    fn load(&mut self, out: &mut String, block: Block, r: &str, c: &str) -> String {
        let decl = &self.kernel.tensors[block.tensor.0];
        let v = self.name("ld");
        let (ty, zero) = (scalar(decl.dtype), zero(decl.dtype));
        let _ = writeln!(out, "let {v}: {ty} = {zero}");
        let gr = format!(
            "({} * {} + {r})",
            affine(block.rows, &self.kt),
            block.shape[0]
        );
        let gc = format!(
            "({} * {} + {c})",
            affine(block.cols, &self.kt),
            block.shape[1]
        );
        let (rows, cols) = (decl.shape[0], decl.shape[1]);
        let _ = writeln!(out, "if {gr} < {rows} && {gc} < {cols} {{");
        let _ = writeln!(out, "{v} = b_{}[{gr} * {cols} + {gc}]", decl.name);
        let _ = writeln!(out, "}}");
        v
    }

    /// Copies the `[rows, cols]` block into shared `dst`, with the whole workgroup cooperating.
    /// `form` is the element layout ([`StageForm`]). Full blocks are copied unmasked, partial
    /// blocks masked.
    fn stage(
        &mut self,
        out: &mut String,
        dst: &str,
        block: Block,
        [rows, cols]: [usize; 2],
        form: StageForm,
        workgroup: usize,
    ) {
        let total = rows * cols;
        let per = total.div_ceil(workgroup);
        let transposed = form == StageForm::Transposed;
        let index = if transposed {
            format!("c * {rows} + r")
        } else {
            format!("r * {cols} + c")
        };
        let decl = &self.kernel.tensors[block.tensor.0];
        let (tr, tc) = (decl.shape[0], decl.shape[1]);
        let br = format!("({} * {rows})", affine(block.rows, &self.kt));
        let bc = format!("({} * {cols})", affine(block.cols, &self.kt));
        let exact = total % workgroup == 0;
        let (ty, zero) = (scalar(decl.dtype), zero(decl.dtype));
        let _ = writeln!(out, "if {br} + {rows} <= {tr} && {bc} + {cols} <= {tc} {{");
        if form == StageForm::Vector && vector_stage(self.kernel, block, cols) {
            let lanes = vec_lanes(decl.dtype);
            let cpr = cols / lanes;
            let chunks = rows * cpr;
            let per_v = chunks.div_ceil(workgroup);
            let exact_v = chunks.is_multiple_of(workgroup);
            let alias = alias_name(self.kernel, block.tensor.0);
            let acol = affine(block.cols, &self.kt);
            let _ = writeln!(out, "for it in 0..{per_v} {{");
            let _ = writeln!(out, "let e = lid + it * {workgroup}");
            if !exact_v {
                let _ = writeln!(out, "if e < {chunks} {{");
            }
            let _ = writeln!(out, "let r = e / {cpr}");
            let _ = writeln!(out, "let c8 = e % {cpr}");
            let _ = writeln!(
                out,
                "let vec16: u32x4 = {alias}[({br} + r) * {} + {acol} * {cpr} + c8]",
                tc / lanes
            );
            let _ = writeln!(out, "{dst}v[e] = vec16");
            if !exact_v {
                let _ = writeln!(out, "}}");
            }
            let _ = writeln!(out, "}}");
        } else {
            let _ = writeln!(out, "for it in 0..{per} {{");
            let _ = writeln!(out, "let e = lid + it * {workgroup}");
            if !exact {
                let _ = writeln!(out, "if e < {total} {{");
            }
            let _ = writeln!(out, "let r = e / {cols}");
            let _ = writeln!(out, "let c = e % {cols}");
            let v = self.name("ld");
            let _ = writeln!(out, "let {v}: {ty} = {zero}");
            let _ = writeln!(out, "{v} = b_{}[({br} + r) * {tc} + {bc} + c]", decl.name);
            let _ = writeln!(out, "{dst}[{index}] = {v}");
            if !exact {
                let _ = writeln!(out, "}}");
            }
            let _ = writeln!(out, "}}");
        }
        let _ = writeln!(out, "}} else {{");
        let _ = writeln!(out, "for it in 0..{per} {{");
        let _ = writeln!(out, "let e = lid + it * {workgroup}");
        let _ = writeln!(out, "if e < {total} {{");
        let _ = writeln!(out, "let r = e / {cols}");
        let _ = writeln!(out, "let c = e % {cols}");
        let v = self.load(out, block, "r", "c");
        let _ = writeln!(out, "{dst}[{index}] = {v}");
        let _ = writeln!(out, "}}");
        let _ = writeln!(out, "}}");
        let _ = writeln!(out, "}}");
    }

    fn store(&mut self, out: &mut String, block: &Block, value: Tile) -> Result<(), TileError> {
        let (tr, tc) = ("tr", "tc");
        let v = self.elem(out, value, tr, tc)?;
        let decl = &self.kernel.tensors[block.tensor.0];
        let gr = format!(
            "({} * {} + {tr})",
            affine(block.rows, &self.kt),
            block.shape[0]
        );
        let gc = format!(
            "({} * {} + {tc})",
            affine(block.cols, &self.kt),
            block.shape[1]
        );
        let (rows, cols) = (decl.shape[0], decl.shape[1]);
        let _ = writeln!(out, "if {gr} < {rows} && {gc} < {cols} {{");
        let _ = writeln!(out, "b_{}[{gr} * {cols} + {gc}] = {v}", decl.name);
        let _ = writeln!(out, "}}");
        Ok(())
    }
}
