//! Register-blocked f32 GEMM in ZSL, for backends that take ZSL output.
//!
//! Each 16×16 workgroup computes a 128×128 tile of C. Per 8-deep K slice it
//! stages A (transposed, `at[k * 128 + m]`) and B (`bt[k * 128 + n]`) in
//! shared memory, four elements per thread, zero-filling past the matrix
//! edges; then every thread accumulates an 8×8 block of C in registers with a
//! 16-element stride, so shared reads are conflict-free and C stores
//! coalesce. ZSL has no local arrays, so the 64 accumulators are the scalars
//! `cI_J` (row I, column J) and the loads and stores are unrolled.
//!
//! RX 9060 XT under RADV: 5.45 TFLOP/s at 4096³, 4.9 at 2048³, against
//! 0.56 for the naive kernel. Apple M1 under Metal, in Low Power Mode:
//! 0.52 TFLOP/s at 2048³–4096³, against 0.15.

use zengpu_spirv::{ZslShader, zsl};

/// Output tile edge per workgroup.
pub(crate) const TILE: u32 = 128;

pub(crate) const GEMM_TILED_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, alpha: f32 }
    @workgroup_size(16, 16)
    kernel cs_gemm_tiled(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: P) {
        shared at: array<f32, 1024>
        shared bt: array<f32, 1024>
        let tx = local_id().x
        let ty = local_id().y
        let tid = ty * 16 + tx
        let bm = group_id().y * 128
        let bn = group_id().x * 128
        let c0_0: f32 = 0.0
        let c0_1: f32 = 0.0
        let c0_2: f32 = 0.0
        let c0_3: f32 = 0.0
        let c0_4: f32 = 0.0
        let c0_5: f32 = 0.0
        let c0_6: f32 = 0.0
        let c0_7: f32 = 0.0
        let c1_0: f32 = 0.0
        let c1_1: f32 = 0.0
        let c1_2: f32 = 0.0
        let c1_3: f32 = 0.0
        let c1_4: f32 = 0.0
        let c1_5: f32 = 0.0
        let c1_6: f32 = 0.0
        let c1_7: f32 = 0.0
        let c2_0: f32 = 0.0
        let c2_1: f32 = 0.0
        let c2_2: f32 = 0.0
        let c2_3: f32 = 0.0
        let c2_4: f32 = 0.0
        let c2_5: f32 = 0.0
        let c2_6: f32 = 0.0
        let c2_7: f32 = 0.0
        let c3_0: f32 = 0.0
        let c3_1: f32 = 0.0
        let c3_2: f32 = 0.0
        let c3_3: f32 = 0.0
        let c3_4: f32 = 0.0
        let c3_5: f32 = 0.0
        let c3_6: f32 = 0.0
        let c3_7: f32 = 0.0
        let c4_0: f32 = 0.0
        let c4_1: f32 = 0.0
        let c4_2: f32 = 0.0
        let c4_3: f32 = 0.0
        let c4_4: f32 = 0.0
        let c4_5: f32 = 0.0
        let c4_6: f32 = 0.0
        let c4_7: f32 = 0.0
        let c5_0: f32 = 0.0
        let c5_1: f32 = 0.0
        let c5_2: f32 = 0.0
        let c5_3: f32 = 0.0
        let c5_4: f32 = 0.0
        let c5_5: f32 = 0.0
        let c5_6: f32 = 0.0
        let c5_7: f32 = 0.0
        let c6_0: f32 = 0.0
        let c6_1: f32 = 0.0
        let c6_2: f32 = 0.0
        let c6_3: f32 = 0.0
        let c6_4: f32 = 0.0
        let c6_5: f32 = 0.0
        let c6_6: f32 = 0.0
        let c6_7: f32 = 0.0
        let c7_0: f32 = 0.0
        let c7_1: f32 = 0.0
        let c7_2: f32 = 0.0
        let c7_3: f32 = 0.0
        let c7_4: f32 = 0.0
        let c7_5: f32 = 0.0
        let c7_6: f32 = 0.0
        let c7_7: f32 = 0.0
        let tiles = (p.k + 7) / 8
        for t in 0..tiles {
            let k0 = t * 8
            for s in 0..4 {
                let idx = tid + s * 256
                let ar = idx / 8
                let ak = idx - ar * 8
                if bm + ar < p.m && k0 + ak < p.k {
                    at[ak * 128 + ar] = a[(bm + ar) * p.k + k0 + ak]
                } else {
                    at[ak * 128 + ar] = 0.0
                }
                let bk = idx / 128
                let bc = idx - bk * 128
                if k0 + bk < p.k && bn + bc < p.n {
                    bt[bk * 128 + bc] = b[(k0 + bk) * p.n + bn + bc]
                } else {
                    bt[bk * 128 + bc] = 0.0
                }
            }
            barrier()
            for q in 0..8 {
                let a0 = at[q * 128 + ty + 0]
                let a1 = at[q * 128 + ty + 16]
                let a2 = at[q * 128 + ty + 32]
                let a3 = at[q * 128 + ty + 48]
                let a4 = at[q * 128 + ty + 64]
                let a5 = at[q * 128 + ty + 80]
                let a6 = at[q * 128 + ty + 96]
                let a7 = at[q * 128 + ty + 112]
                let b0 = bt[q * 128 + tx + 0]
                let b1 = bt[q * 128 + tx + 16]
                let b2 = bt[q * 128 + tx + 32]
                let b3 = bt[q * 128 + tx + 48]
                let b4 = bt[q * 128 + tx + 64]
                let b5 = bt[q * 128 + tx + 80]
                let b6 = bt[q * 128 + tx + 96]
                let b7 = bt[q * 128 + tx + 112]
                c0_0 = c0_0 + a0 * b0
                c0_1 = c0_1 + a0 * b1
                c0_2 = c0_2 + a0 * b2
                c0_3 = c0_3 + a0 * b3
                c0_4 = c0_4 + a0 * b4
                c0_5 = c0_5 + a0 * b5
                c0_6 = c0_6 + a0 * b6
                c0_7 = c0_7 + a0 * b7
                c1_0 = c1_0 + a1 * b0
                c1_1 = c1_1 + a1 * b1
                c1_2 = c1_2 + a1 * b2
                c1_3 = c1_3 + a1 * b3
                c1_4 = c1_4 + a1 * b4
                c1_5 = c1_5 + a1 * b5
                c1_6 = c1_6 + a1 * b6
                c1_7 = c1_7 + a1 * b7
                c2_0 = c2_0 + a2 * b0
                c2_1 = c2_1 + a2 * b1
                c2_2 = c2_2 + a2 * b2
                c2_3 = c2_3 + a2 * b3
                c2_4 = c2_4 + a2 * b4
                c2_5 = c2_5 + a2 * b5
                c2_6 = c2_6 + a2 * b6
                c2_7 = c2_7 + a2 * b7
                c3_0 = c3_0 + a3 * b0
                c3_1 = c3_1 + a3 * b1
                c3_2 = c3_2 + a3 * b2
                c3_3 = c3_3 + a3 * b3
                c3_4 = c3_4 + a3 * b4
                c3_5 = c3_5 + a3 * b5
                c3_6 = c3_6 + a3 * b6
                c3_7 = c3_7 + a3 * b7
                c4_0 = c4_0 + a4 * b0
                c4_1 = c4_1 + a4 * b1
                c4_2 = c4_2 + a4 * b2
                c4_3 = c4_3 + a4 * b3
                c4_4 = c4_4 + a4 * b4
                c4_5 = c4_5 + a4 * b5
                c4_6 = c4_6 + a4 * b6
                c4_7 = c4_7 + a4 * b7
                c5_0 = c5_0 + a5 * b0
                c5_1 = c5_1 + a5 * b1
                c5_2 = c5_2 + a5 * b2
                c5_3 = c5_3 + a5 * b3
                c5_4 = c5_4 + a5 * b4
                c5_5 = c5_5 + a5 * b5
                c5_6 = c5_6 + a5 * b6
                c5_7 = c5_7 + a5 * b7
                c6_0 = c6_0 + a6 * b0
                c6_1 = c6_1 + a6 * b1
                c6_2 = c6_2 + a6 * b2
                c6_3 = c6_3 + a6 * b3
                c6_4 = c6_4 + a6 * b4
                c6_5 = c6_5 + a6 * b5
                c6_6 = c6_6 + a6 * b6
                c6_7 = c6_7 + a6 * b7
                c7_0 = c7_0 + a7 * b0
                c7_1 = c7_1 + a7 * b1
                c7_2 = c7_2 + a7 * b2
                c7_3 = c7_3 + a7 * b3
                c7_4 = c7_4 + a7 * b4
                c7_5 = c7_5 + a7 * b5
                c7_6 = c7_6 + a7 * b6
                c7_7 = c7_7 + a7 * b7
            }
            barrier()
        }
        let r0 = bm + ty + 0
        if r0 < p.m {
            if bn + tx + 0 < p.n {
                c[r0 * p.n + bn + tx + 0] = p.alpha * c0_0
            }
            if bn + tx + 16 < p.n {
                c[r0 * p.n + bn + tx + 16] = p.alpha * c0_1
            }
            if bn + tx + 32 < p.n {
                c[r0 * p.n + bn + tx + 32] = p.alpha * c0_2
            }
            if bn + tx + 48 < p.n {
                c[r0 * p.n + bn + tx + 48] = p.alpha * c0_3
            }
            if bn + tx + 64 < p.n {
                c[r0 * p.n + bn + tx + 64] = p.alpha * c0_4
            }
            if bn + tx + 80 < p.n {
                c[r0 * p.n + bn + tx + 80] = p.alpha * c0_5
            }
            if bn + tx + 96 < p.n {
                c[r0 * p.n + bn + tx + 96] = p.alpha * c0_6
            }
            if bn + tx + 112 < p.n {
                c[r0 * p.n + bn + tx + 112] = p.alpha * c0_7
            }
        }
        let r1 = bm + ty + 16
        if r1 < p.m {
            if bn + tx + 0 < p.n {
                c[r1 * p.n + bn + tx + 0] = p.alpha * c1_0
            }
            if bn + tx + 16 < p.n {
                c[r1 * p.n + bn + tx + 16] = p.alpha * c1_1
            }
            if bn + tx + 32 < p.n {
                c[r1 * p.n + bn + tx + 32] = p.alpha * c1_2
            }
            if bn + tx + 48 < p.n {
                c[r1 * p.n + bn + tx + 48] = p.alpha * c1_3
            }
            if bn + tx + 64 < p.n {
                c[r1 * p.n + bn + tx + 64] = p.alpha * c1_4
            }
            if bn + tx + 80 < p.n {
                c[r1 * p.n + bn + tx + 80] = p.alpha * c1_5
            }
            if bn + tx + 96 < p.n {
                c[r1 * p.n + bn + tx + 96] = p.alpha * c1_6
            }
            if bn + tx + 112 < p.n {
                c[r1 * p.n + bn + tx + 112] = p.alpha * c1_7
            }
        }
        let r2 = bm + ty + 32
        if r2 < p.m {
            if bn + tx + 0 < p.n {
                c[r2 * p.n + bn + tx + 0] = p.alpha * c2_0
            }
            if bn + tx + 16 < p.n {
                c[r2 * p.n + bn + tx + 16] = p.alpha * c2_1
            }
            if bn + tx + 32 < p.n {
                c[r2 * p.n + bn + tx + 32] = p.alpha * c2_2
            }
            if bn + tx + 48 < p.n {
                c[r2 * p.n + bn + tx + 48] = p.alpha * c2_3
            }
            if bn + tx + 64 < p.n {
                c[r2 * p.n + bn + tx + 64] = p.alpha * c2_4
            }
            if bn + tx + 80 < p.n {
                c[r2 * p.n + bn + tx + 80] = p.alpha * c2_5
            }
            if bn + tx + 96 < p.n {
                c[r2 * p.n + bn + tx + 96] = p.alpha * c2_6
            }
            if bn + tx + 112 < p.n {
                c[r2 * p.n + bn + tx + 112] = p.alpha * c2_7
            }
        }
        let r3 = bm + ty + 48
        if r3 < p.m {
            if bn + tx + 0 < p.n {
                c[r3 * p.n + bn + tx + 0] = p.alpha * c3_0
            }
            if bn + tx + 16 < p.n {
                c[r3 * p.n + bn + tx + 16] = p.alpha * c3_1
            }
            if bn + tx + 32 < p.n {
                c[r3 * p.n + bn + tx + 32] = p.alpha * c3_2
            }
            if bn + tx + 48 < p.n {
                c[r3 * p.n + bn + tx + 48] = p.alpha * c3_3
            }
            if bn + tx + 64 < p.n {
                c[r3 * p.n + bn + tx + 64] = p.alpha * c3_4
            }
            if bn + tx + 80 < p.n {
                c[r3 * p.n + bn + tx + 80] = p.alpha * c3_5
            }
            if bn + tx + 96 < p.n {
                c[r3 * p.n + bn + tx + 96] = p.alpha * c3_6
            }
            if bn + tx + 112 < p.n {
                c[r3 * p.n + bn + tx + 112] = p.alpha * c3_7
            }
        }
        let r4 = bm + ty + 64
        if r4 < p.m {
            if bn + tx + 0 < p.n {
                c[r4 * p.n + bn + tx + 0] = p.alpha * c4_0
            }
            if bn + tx + 16 < p.n {
                c[r4 * p.n + bn + tx + 16] = p.alpha * c4_1
            }
            if bn + tx + 32 < p.n {
                c[r4 * p.n + bn + tx + 32] = p.alpha * c4_2
            }
            if bn + tx + 48 < p.n {
                c[r4 * p.n + bn + tx + 48] = p.alpha * c4_3
            }
            if bn + tx + 64 < p.n {
                c[r4 * p.n + bn + tx + 64] = p.alpha * c4_4
            }
            if bn + tx + 80 < p.n {
                c[r4 * p.n + bn + tx + 80] = p.alpha * c4_5
            }
            if bn + tx + 96 < p.n {
                c[r4 * p.n + bn + tx + 96] = p.alpha * c4_6
            }
            if bn + tx + 112 < p.n {
                c[r4 * p.n + bn + tx + 112] = p.alpha * c4_7
            }
        }
        let r5 = bm + ty + 80
        if r5 < p.m {
            if bn + tx + 0 < p.n {
                c[r5 * p.n + bn + tx + 0] = p.alpha * c5_0
            }
            if bn + tx + 16 < p.n {
                c[r5 * p.n + bn + tx + 16] = p.alpha * c5_1
            }
            if bn + tx + 32 < p.n {
                c[r5 * p.n + bn + tx + 32] = p.alpha * c5_2
            }
            if bn + tx + 48 < p.n {
                c[r5 * p.n + bn + tx + 48] = p.alpha * c5_3
            }
            if bn + tx + 64 < p.n {
                c[r5 * p.n + bn + tx + 64] = p.alpha * c5_4
            }
            if bn + tx + 80 < p.n {
                c[r5 * p.n + bn + tx + 80] = p.alpha * c5_5
            }
            if bn + tx + 96 < p.n {
                c[r5 * p.n + bn + tx + 96] = p.alpha * c5_6
            }
            if bn + tx + 112 < p.n {
                c[r5 * p.n + bn + tx + 112] = p.alpha * c5_7
            }
        }
        let r6 = bm + ty + 96
        if r6 < p.m {
            if bn + tx + 0 < p.n {
                c[r6 * p.n + bn + tx + 0] = p.alpha * c6_0
            }
            if bn + tx + 16 < p.n {
                c[r6 * p.n + bn + tx + 16] = p.alpha * c6_1
            }
            if bn + tx + 32 < p.n {
                c[r6 * p.n + bn + tx + 32] = p.alpha * c6_2
            }
            if bn + tx + 48 < p.n {
                c[r6 * p.n + bn + tx + 48] = p.alpha * c6_3
            }
            if bn + tx + 64 < p.n {
                c[r6 * p.n + bn + tx + 64] = p.alpha * c6_4
            }
            if bn + tx + 80 < p.n {
                c[r6 * p.n + bn + tx + 80] = p.alpha * c6_5
            }
            if bn + tx + 96 < p.n {
                c[r6 * p.n + bn + tx + 96] = p.alpha * c6_6
            }
            if bn + tx + 112 < p.n {
                c[r6 * p.n + bn + tx + 112] = p.alpha * c6_7
            }
        }
        let r7 = bm + ty + 112
        if r7 < p.m {
            if bn + tx + 0 < p.n {
                c[r7 * p.n + bn + tx + 0] = p.alpha * c7_0
            }
            if bn + tx + 16 < p.n {
                c[r7 * p.n + bn + tx + 16] = p.alpha * c7_1
            }
            if bn + tx + 32 < p.n {
                c[r7 * p.n + bn + tx + 32] = p.alpha * c7_2
            }
            if bn + tx + 48 < p.n {
                c[r7 * p.n + bn + tx + 48] = p.alpha * c7_3
            }
            if bn + tx + 64 < p.n {
                c[r7 * p.n + bn + tx + 64] = p.alpha * c7_4
            }
            if bn + tx + 80 < p.n {
                c[r7 * p.n + bn + tx + 80] = p.alpha * c7_5
            }
            if bn + tx + 96 < p.n {
                c[r7 * p.n + bn + tx + 96] = p.alpha * c7_6
            }
            if bn + tx + 112 < p.n {
                c[r7 * p.n + bn + tx + 112] = p.alpha * c7_7
            }
        }
    }
);
