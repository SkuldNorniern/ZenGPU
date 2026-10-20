//! Schedules change speed only: every resolved schedule computes the same values. Unsupported
//! requests fall back, with reasons in `crate::CompiledTile::notes`. Fallback order: `Mma`, then
//! `Staged` scalar, then `Naive` (the O0 reference).

use zengpu_hal::BackendPreference;

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
            has_mma_i8: true,
        }
    }

    /// No matrix unit. Used by text-only backends (CUDA, Metal) and `compile`.
    pub const fn scalar_only(backend: BackendPreference) -> Self {
        Target {
            backend,
            wave_size: 32,
            lds_bytes: 32768,
            has_mma_f16: false,
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
        match mma_plan(kernel, &parts, target, waves, &mut notes) {
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
    })
}

fn mma_plan(
    kernel: &TileKernel,
    parts: &Parts,
    target: &Target,
    waves: usize,
    notes: &mut Vec<String>,
) -> Result<Plan, String> {
    if !target.has_mma_f16 {
        return Err("target has no f16 matrix unit".to_string());
    }
    let acc = kernel.values[parts.dot_out.0].dtype;
    if (parts.a_dtype, parts.b_dtype, acc) != (DType::F16, DType::F16, DType::F32) {
        return Err(format!(
            "dot is {} x {} -> {acc}; the matrix unit runs f16 x f16 -> f32",
            parts.a_dtype, parts.b_dtype
        ));
    }
    let [bm, bn] = kernel.tile;
    let bk = parts.step;
    for (what, v) in [("BM", bm), ("BN", bn), ("BK", bk)] {
        if v % 16 != 0 {
            return Err(format!("{what} {v} is not a multiple of 16"));
        }
    }
    let frags = (bm / 16) * (bn / 16);
    let waves = if frags % waves == 0 {
        waves
    } else {
        notes.push(format!(
            "{waves} waves do not divide {frags} fragments; using 1 wave"
        ));
        1
    };
    let wave = target.wave_size as usize;
    let need = (bm * bk + bk * bn + bm * bn) * 4;
    let lds = target.lds_bytes as usize;
    if need > lds {
        return Err(format!("tile needs {need} bytes of LDS, target has {lds}"));
    }
    if waves * wave > 1024 {
        return Err(format!("{waves} waves of {wave} lanes exceed 1024 threads"));
    }
    Ok(Plan {
        mode: Mode::Mma,
        workgroup: waves * wave,
        waves,
        wave_size: wave,
    })
}
