//! Schedules change speed only: every resolved schedule computes the same values. Unsupported
//! requests fall back, with reasons in `crate::CompiledTile::notes`. Fallback order: `Mma`, then
//! `Staged` scalar, then `Naive` (the O0 reference).

use zengpu_hal::{BackendPreference, ComputeCaps};

use crate::ir::{DType, TileKernel};
use crate::lower::{self, Mode, Parts, Plan};

/// Which unit computes `dot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Matrix {
    /// Matrix unit when the kernel and target allow it, else scalar.
    #[default]
    Auto,
    /// Matrix unit (wmma, mma, cooperative matrix or simdgroup matrix). Scalar fallback.
    Mma,
    /// Scalar FMAs.
    Scalar,
}

/// How to run a kernel. `default` is tuned, `o0` is the naive reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schedule {
    pub matrix: Matrix,
    /// Stage A and B through shared memory. Without it the naive form runs and `matrix: Mma`
    /// cannot apply.
    pub stage_lds: bool,
    /// Block shape and k step. 0 keeps the kernel's own; differing values are noted and ignored.
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    /// Waves per workgroup for the staged forms. 0 means 4.
    pub waves: u32,
    /// matrix: waves across M, rest across N. 0 picks the split with fewest fragments per
    /// wave edge.
    pub wave_rows: u32,
    /// Matrix form: stages per operand, 1 or 2. 0 means 1, since double buffering is slower on
    /// gfx1200 (sync copy).
    pub stages: u32,
    /// Matrix form: 16-byte loads for full interior blocks if rows are 16-byte aligned, else
    /// scalar. Vulkan needs an explicit workgroup layout.
    pub vector_loads: bool,
    /// Matrix form: k split into N groups, each writing a partial tile. A second pass sums them in
    /// group order (deterministic). 0 or 1 means no split. f16 x f16 -> f32 only.
    pub split_k: u32,
    /// Matrix form only: `@waves_per_eu(min, max)` trades occupancy for fewer spills. `None` leaves
    /// the choice to the compiler.
    pub waves_per_eu: Option<(u32, u32)>,
}

impl Schedule {
    /// The O0 reference: no staging, scalar math, one workgroup per tile.
    pub const fn o0() -> Self {
        Schedule {
            matrix: Matrix::Scalar,
            stage_lds: false,
            block_m: 0,
            block_n: 0,
            block_k: 0,
            waves: 1,
            wave_rows: 0,
            stages: 0,
            vector_loads: false,
            split_k: 1,
            waves_per_eu: None,
        }
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule {
            matrix: Matrix::Auto,
            stage_lds: true,
            block_m: 0,
            block_n: 0,
            block_k: 0,
            waves: 4,
            wave_rows: 0,
            stages: 0,
            vector_loads: true,
            split_k: 1,
            waves_per_eu: None,
        }
    }
}

/// The HAL does not report these, so the caller states them (`hip_gfx12` for RDNA4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub backend: BackendPreference,
    /// Lanes per wave the mma code assumes; the workgroup is a multiple of it.
    pub wave_size: u32,
    pub lds_bytes: u32,
    /// Matrix ops from f16 x f16 to f32.
    pub has_mma_f16: bool,
    /// Matrix ops from f32 x f32 to f32.
    pub has_mma_f32: bool,
    /// Matrix ops from i8 x i8 to i32.
    pub has_mma_i8: bool,
}

impl Target {
    /// HIP on gfx12: wave32, 64 KiB LDS, wmma for f16 and i8.
    pub const fn hip_gfx12() -> Self {
        Target {
            backend: BackendPreference::Hip,
            wave_size: 32,
            lds_bytes: 65536,
            has_mma_f16: true,
            has_mma_f32: false,
            has_mma_i8: true,
        }
    }

    /// Subgroup 32 if the device runs it (the matrix form needs it), else the default. Shared
    /// memory and 16x16x16 f16, f32 and i8 shapes at that subgroup.
    pub fn from_caps(backend: BackendPreference, caps: &ComputeCaps) -> Self {
        let (lo, hi) = caps.subgroup_sizes;
        let wave = if (lo..=hi).contains(&32) {
            32
        } else {
            caps.default_subgroup
        };
        let has = |a: zengpu_hal::DType, acc: zengpu_hal::DType| {
            caps.mma.iter().any(|s| {
                (s.m, s.n, s.k) == (16, 16, 16)
                    && s.a == a
                    && s.b == a
                    && s.acc == acc
                    && s.subgroup == wave
            })
        };
        Target {
            backend,
            wave_size: wave,
            lds_bytes: caps.shared_bytes,
            has_mma_f16: has(zengpu_hal::DType::F16, zengpu_hal::DType::F32),
            has_mma_f32: has(zengpu_hal::DType::F32, zengpu_hal::DType::F32),
            has_mma_i8: has(zengpu_hal::DType::I8, zengpu_hal::DType::I32),
        }
    }

    /// No matrix unit. Used by text-only backends (CUDA) and `compile`.
    pub const fn scalar_only(backend: BackendPreference) -> Self {
        Target {
            backend,
            wave_size: 32,
            lds_bytes: 32768,
            has_mma_f16: false,
            has_mma_f32: false,
            has_mma_i8: false,
        }
    }
}

pub(crate) fn resolve(
    kernel: &TileKernel,
    schedule: &Schedule,
    target: &Target,
) -> (Plan, Vec<String>) {
    let naive = Plan::naive(kernel);
    if schedule.matrix == Matrix::Scalar && !schedule.stage_lds {
        return (naive, Vec::new());
    }
    let mut notes = Vec::new();
    let parts = match lower::analyze(kernel) {
        Ok(parts) => parts,
        Err(why) => {
            notes.push(format!("unstaged: {why}"));
            return (naive, notes);
        }
    };
    check_blocks(kernel, &parts, schedule, &mut notes);
    if !schedule.stage_lds {
        notes.push("unstaged: stage_lds is off".to_string());
        return (naive, notes);
    }

    let waves = if schedule.waves == 0 {
        4
    } else {
        schedule.waves as usize
    };
    if schedule.matrix != Matrix::Scalar {
        match mma_plan(kernel, &parts, target, schedule, &mut notes) {
            Ok(plan) => return (plan, notes),
            Err(why) => notes.push(format!("mma: {why}")),
        }
    }
    match staged_plan(kernel, &parts, target, waves) {
        Ok(plan) => (plan, notes),
        Err(why) => {
            notes.push(format!("staged: {why}"));
            (naive, notes)
        }
    }
}

fn check_blocks(kernel: &TileKernel, parts: &Parts, s: &Schedule, notes: &mut Vec<String>) {
    let [bm, bn] = kernel.tile;
    let differs = (s.block_m != 0 && s.block_m != bm)
        || (s.block_n != 0 && s.block_n != bn)
        || (s.block_k != 0 && s.block_k != parts.step);
    if differs {
        notes.push(format!(
            "schedule blocks {}x{}x{} differ from the kernel's {bm}x{bn}x{}; using the kernel's",
            s.block_m, s.block_n, s.block_k, parts.step
        ));
    }
}

fn staged_plan(
    kernel: &TileKernel,
    parts: &Parts,
    target: &Target,
    waves: usize,
) -> Result<Plan, String> {
    let (a, b) = (parts.a_dtype, parts.b_dtype);
    if !matches!(a, DType::F16 | DType::F32) || !matches!(b, DType::F16 | DType::F32) {
        return Err(format!(
            "{a} x {b} operands are not staged (shared arrays hold f32)"
        ));
    }
    let [bm, bn] = kernel.tile;
    let need = (bm * parts.step + parts.step * bn) * 4;
    let lds = target.lds_bytes as usize;
    if need > lds {
        return Err(format!("blocks need {need} bytes of LDS, target has {lds}"));
    }
    let wave = target.wave_size as usize;
    Ok(Plan {
        mode: Mode::Staged,
        workgroup: waves * wave,
        waves: 1,
        wave_size: wave,
        native_shared: false,
        wave_rows: 1,
        wave_cols: 1,
        stages: 1,
        vector_loads: false,
        split_k: 1,
        waves_per_eu: None,
        clamp_rows: false,
        pad: 0,
        alias_epilogue: false,
    })
}

fn mma_plan(
    kernel: &TileKernel,
    parts: &Parts,
    target: &Target,
    schedule: &Schedule,
    notes: &mut Vec<String>,
) -> Result<Plan, String> {
    let acc = kernel.values[parts.dot_out.0].dtype;
    let elem = (parts.a_dtype, parts.b_dtype, acc);
    let (has, want) = match elem {
        (DType::F16, DType::F16, DType::F32) => (target.has_mma_f16, "f16"),
        (DType::F32, DType::F32, DType::F32) => (target.has_mma_f32, "f32"),
        (DType::I8, DType::I8, DType::I32) => (target.has_mma_i8, "i8"),
        _ => {
            return Err(format!(
                "dot is {} x {} -> {acc}; the matrix unit runs f16 x f16 -> f32, f32 x f32 -> f32 and i8 x i8 -> i32",
                parts.a_dtype, parts.b_dtype
            ));
        }
    };
    if !has {
        return Err(format!("target has no {want} matrix unit"));
    }
    let [bm, bn] = kernel.tile;
    let bk = parts.step;
    for (what, v) in [("BM", bm), ("BN", bn), ("BK", bk)] {
        if v % 16 != 0 {
            return Err(format!("{what} {v} is not a multiple of 16"));
        }
    }
    let wave = target.wave_size as usize;
    let want_waves = if schedule.waves == 0 {
        4
    } else {
        schedule.waves as usize
    };
    let (waves, nwr, nwc) = wave_grid(bm / 16, bn / 16, want_waves, schedule.wave_rows, notes);
    let split_k = match schedule.split_k.max(1) as usize {
        1 => 1,
        n if elem == (DType::F16, DType::F16, DType::F32) => n,
        n => {
            notes.push(format!(
                "split_k {n} needs f16 x f16 -> f32; running unsplit"
            ));
            1
        }
    };
    let operand_bytes = |d: DType| match d {
        DType::I8 => 1,
        DType::F16 => 2,
        _ => 4,
    };
    let stage_bytes =
        (bm * bk * operand_bytes(parts.a_dtype) + bk * bn * operand_bytes(parts.b_dtype)) as u32;
    let lds = target.lds_bytes;
    // Metal: measured on M1, a branch-free k loop, padded rows at bk >= 32 and an epilogue in A's
    // staging memory are faster. Not yet measured elsewhere.
    let metal = target.backend == BackendPreference::Metal;
    let alias = metal && split_k == 1;
    let f16 = parts.a_dtype == DType::F16 && parts.b_dtype == DType::F16;
    let need = |stages: u32, pad: usize| -> u32 {
        let pad_bytes = ((bm + bn) * pad * 2) as u32;
        if alias {
            // The epilogue slice lives in A's (first) staging array, grown if it is smaller.
            let a_bytes = (bm * (bk + pad) * operand_bytes(parts.a_dtype)) as u32;
            let slice = (nwr * 16 * nwc * 16 * 4) as u32;
            stages * (stage_bytes + pad_bytes) + slice.saturating_sub(a_bytes)
        } else {
            stages * (stage_bytes + pad_bytes) + (bm * nwc * 16 * 4) as u32
        }
    };
    let stages = match schedule.stages {
        0 | 1 => 1,
        2 => 2,
        n => return Err(format!("{n} stages are not supported")),
    };
    // Padding is dropped when it does not fit.
    let pad = if metal && bk >= 32 && f16 && need(stages, 8) <= lds {
        8
    } else {
        0
    };
    if need(stages, pad) > lds {
        return Err(format!(
            "{stages} stage(s) need {} bytes of LDS, target has {lds}",
            need(stages, pad)
        ));
    }
    if waves * wave > 1024 {
        return Err(format!("{waves} waves of {wave} lanes exceed 1024 threads"));
    }
    Ok(Plan {
        mode: Mode::Mma,
        workgroup: waves * wave,
        waves,
        wave_size: wave,
        native_shared: true,
        wave_rows: nwr,
        wave_cols: nwc,
        stages: stages as usize,
        vector_loads: schedule.vector_loads,
        split_k,
        waves_per_eu: schedule.waves_per_eu,
        clamp_rows: metal && split_k == 1 && stages == 1,
        pad,
        alias_epilogue: alias,
    })
}

/// Splits `waves` into `rows x cols` so each wave's fragment block fits. The wanted row split is
/// used if it fits, else the one with the smallest perimeter. No split: one wave.
pub(crate) fn wave_grid(
    fm: usize,
    fnn: usize,
    waves: usize,
    want_rows: u32,
    notes: &mut Vec<String>,
) -> (usize, usize, usize) {
    if want_rows != 0 {
        let r = want_rows as usize;
        if fits_for(waves, r, fm, fnn) {
            return (waves, r, waves / r);
        }
        notes.push(format!(
            "wave rows {r} do not split {waves} waves over {fm}x{fnn} fragments; choosing the split"
        ));
    }
    let best = (1..=waves)
        .filter(|&r| fits_for(waves, r, fm, fnn))
        .min_by_key(|&r| fm / r + fnn / (waves / r));
    if let Some(r) = best {
        return (waves, r, waves / r);
    }
    notes.push(format!(
        "{waves} waves do not split {fm}x{fnn} fragments; using 1 wave"
    ));
    (1, 1, 1)
}

fn fits_for(waves: usize, rows: usize, fm: usize, fnn: usize) -> bool {
    waves.is_multiple_of(rows) && fm.is_multiple_of(rows) && fnn.is_multiple_of(waves / rows)
}
