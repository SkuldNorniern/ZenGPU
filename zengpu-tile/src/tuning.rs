//! Tile and schedule choice for GEMM-shaped kernels. Only Metal is tuned, on an M1; the other
//! targets get the crate defaults, clamped to the problem.

use zengpu_hal::BackendPreference;

use crate::ir::DType;
use crate::schedule::{Matrix, Schedule, Target, wave_grid};

/// A tile, k step and schedule. Build the kernel with `TileKernel::builder(name, tile)` and
/// `loop_k(k, bk, ..)`, then compile it with `compile_with(&kernel, &schedule, &target)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Suggestion {
    /// Output block `[bm, bn]`, a multiple of 16 on each side.
    pub tile: [usize; 2],
    /// k step.
    pub bk: usize,
    pub schedule: Schedule,
}

/// A measured configuration, before it is checked against the problem and the target.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    tile: [usize; 2],
    bk: usize,
    waves: u32,
    /// Wave rows, 0 for auto.
    rows: u32,
}

/// Suggests a tile, k step and schedule for `C[m, n] = A[m, k] * B[n, k]^T` with operands of
/// dtype `input` on `target`. The tile never exceeds the problem rounded up to 16, and the
/// suggestion always fits the target's LDS and thread limits.
///
/// Metal f16 uses the simdgroup matrix form. Measured on M1 (`bench_f16_gemm`): a 128x64 tile
/// with bk 32 when the grid has at least 256 such tiles (`wide`), else a 64x64 tile with two
/// wave rows. The later candidates are smaller tiles for limited LDS. Smaller k, and any shape
/// outside the bench, is an extrapolation.
///
/// Metal i8 with an i8 matrix unit uses 32x32, bk 16, one wave (best measured for 2048^3).
/// Metal f32 with an f32 matrix unit uses 64x64, bk 32, four waves, or 64x32 when LDS is short.
/// Everything else gets the crate defaults: a 64x64 tile, bk 16 and `Schedule::default()`.
/// That includes i8 where the target has no i8 matrix unit.
/// Candidates that do not fit `target.lds_bytes` are skipped, so a small LDS only shrinks the
/// tile.
pub fn suggest(target: &Target, input: DType, m: usize, n: usize, k: usize) -> Suggestion {
    for c in candidates(target, input, m, n, k) {
        if let Some(s) = fit(target, input, c, m, n, k) {
            return s;
        }
    }
    Suggestion {
        tile: [clamp(64, m), clamp(64, n)],
        bk: clamp(16, k),
        schedule: Schedule::default(),
    }
}

fn candidates(target: &Target, input: DType, m: usize, n: usize, k: usize) -> Vec<Candidate> {
    if target.backend != BackendPreference::Metal {
        return Vec::new();
    }
    let cand = |tile, bk, waves, rows| Candidate {
        tile,
        bk,
        waves,
        rows,
    };
    if input == DType::I8 && target.has_mma_i8 {
        // Measured on M1 at 2048^3: 32x32 with one wave (11.0 ms) beat 64x32 (11.7) and 64x64
        // (12.2); i8 fragments load element-wise, so smaller tiles win.
        return vec![cand([32, 32], 16, 1, 0)];
    }
    if input == DType::F32 && target.has_mma_f32 {
        // Measured on M1 at 2048^3: 64x64 bk 32 with four waves, then 64x32 with two.
        return vec![cand([64, 64], 32, 4, 0), cand([64, 32], 32, 2, 0)];
    }
    if input != DType::F16 || !target.has_mma_f16 {
        return Vec::new();
    }
    let bk = if k >= 64 { 32 } else { 16 };
    let wide = m.div_ceil(128) * n.div_ceil(64) >= 256;
    let mut out = Vec::new();
    if wide {
        out.push(cand([128, 64], bk, 4, 0));
    }
    out.push(cand([64, 64], bk, 4, 2));
    out.push(cand([64, 32], 16, 2, 0));
    out.push(cand([32, 32], 16, 1, 0));
    out
}

/// Clamps the candidate to the problem and checks it against the target. `None` if it does not
/// fit.
fn fit(
    target: &Target,
    input: DType,
    c: Candidate,
    m: usize,
    n: usize,
    k: usize,
) -> Option<Suggestion> {
    let [bm, bn] = [clamp(c.tile[0], m), clamp(c.tile[1], n)];
    let bk = clamp(c.bk, k);
    // Same wave split as mma_plan, which falls back to fewer waves when the fragments do not
    // split. The notes it would add are not needed here.
    let (fm, fnn) = (bm / 16, bn / 16);
    let (waves, nwr, nwc) = wave_grid(fm, fnn, c.waves as usize, c.rows, &mut Vec::new());

    // Single stage, as mma_plan lays out on Metal. The epilogue slice reuses A's staging
    // memory, so only the part of it that A does not cover adds to the stage. The A tile is
    // taken without padding, which only makes the bound larger.
    // Bytes per operand element.
    let elem = match input {
        DType::I8 => 1,
        DType::F16 => 2,
        _ => 4,
    };
    let stage = (bm * bk + bk * bn) * elem;
    let slice = nwr * 16 * nwc * 16 * 4;
    if stage + slice.saturating_sub(bm * bk * elem) > target.lds_bytes as usize {
        return None;
    }
    if waves * target.wave_size as usize > 1024 {
        return None;
    }

    Some(Suggestion {
        tile: [bm, bn],
        bk,
        schedule: Schedule {
            matrix: Matrix::Mma,
            waves: waves as u32,
            wave_rows: if c.rows as usize == nwr { c.rows } else { 0 },
            ..Schedule::default()
        },
    })
}

/// `tile` capped at `extent` rounded up to 16, and at least 16.
fn clamp(tile: usize, extent: usize) -> usize {
    tile.min(extent.div_ceil(16) * 16).max(16)
}
