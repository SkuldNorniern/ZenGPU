//! HIP GEMM on RDNA4 matrix cores (WMMA) for half-precision A and B.
//!
//! `C[M,N] = alpha * A[M,K] @ B[K,N]`, row-major, A and B both f16 or both
//! bf16, accumulated and stored in f32. On an RX 9060 XT (ROCm 7.2) this
//! matches or beats rocBLAS `gemm_ex` (f16 in, f32 out): 67 TFLOP/s at
//! 4096³ (rocBLAS 68), 67 at 1024×11008×4096 (rocBLAS 59), and 1.2–1.3× at
//! 17–64 rows.
//!
//! The kernels only exist when hipRTC compiles for a gfx12 target
//! (`__GFX12__`); elsewhere the module is empty and pipeline creation fails,
//! which [`crate::GemmKernel`] takes as "no WMMA here".

/// HIP source. Each kernel is a [`Variant`]: waves of 32 lanes laid out
/// `WAVES_M`×`WAVES_N`, each owning `TM`×`TN` WMMA 16×16 output tiles, K in
/// steps of 64.
///
/// A goes through LDS (one tile per step, shared by the block's waves). B
/// skips LDS: `global_load_tr` loads each 16×16 B fragment from memory
/// already transposed into the WMMA operand layout. The next step's A and B
/// are loaded into registers while the current step computes, and the
/// barriers wait only for LDS, so those loads stay in flight.
///
/// Interior blocks run without bounds checks. Edge blocks clamp their
/// addresses to valid memory and skip out-of-range stores; past K, A reads
/// as zero, so the clamped B rows add nothing. Needs `N % 8 == 0` and
/// `K % 8 == 0`.
pub(crate) const SOURCE: &str = r#"
#if defined(__GFX12__)
typedef short wm_s8 __attribute__((ext_vector_type(8)));
typedef _Float16 wm_h8 __attribute__((ext_vector_type(8)));
typedef float wm_f8 __attribute__((ext_vector_type(8)));
typedef wm_s8 __attribute__((address_space(1))) wm_global_s8;

#define WM_BK 64
#define WM_LDS (WM_BK + 8)

// LDS-only barrier: global loads issued for the next step stay in flight.
#define WM_LDS_BARRIER() \
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory")

template <bool BF16>
__device__ __forceinline__ wm_f8 wm_mma(wm_s8 a, wm_s8 b, wm_f8 c) {
    if constexpr (BF16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            __builtin_bit_cast(wm_h8, a), __builtin_bit_cast(wm_h8, b), c);
    }
}

// EDGE: the block's tile crosses M or N. KTAIL: K is not a multiple of 64.
template <bool BF16, int WAVES_M, int WAVES_N, int TM, int TN, bool EDGE, bool KTAIL>
__device__ __forceinline__ void wm_body(
        short* As, const short* __restrict__ A, const short* __restrict__ B,
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
        unsigned int row0, unsigned int col0, float alpha) {
    constexpr unsigned int THREADS = WAVES_M * WAVES_N * 32;
    constexpr unsigned int AV_PER_ROW = WM_BK / 8;
    constexpr unsigned int A_ROW_STEP = THREADS / AV_PER_ROW;
    constexpr int A_VECS = TM * 16 * WAVES_M * WM_BK / 8 / THREADS;
    constexpr int KS = WM_BK / 16;
    const unsigned int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const unsigned int wm = wave / WAVES_N, wn = wave % WAVES_N;

    const unsigned int a_c = (t % AV_PER_ROW) * 8;
    const short* pa[A_VECS];
    _Pragma("unroll") for (int v = 0; v < A_VECS; v++) {
        const unsigned int r = row0 + t / AV_PER_ROW + v * A_ROW_STEP;
        pa[v] = A + (size_t)(EDGE ? min(r, M - 1) : r) * K + a_c;
    }
    // Lane l of a B fragment reads row (l/16)*8 + l%8 at column ((l%16)/8)*8;
    // the transposing load hands lane l column l%16, k (l/16)*8 .. +8.
    const unsigned int b_row = (lane >> 4) * 8 + (lane & 7);
    unsigned int b_col[TN];
    _Pragma("unroll") for (int j = 0; j < TN; j++) {
        const unsigned int c = col0 + (wn * TN + j) * 16 + ((lane & 15) >> 3) * 8;
        b_col[j] = EDGE ? min(c, N - 8) : c;
    }

    wm_s8 ra[A_VECS];
    wm_s8 bcur[KS][TN], bnext[KS][TN];
    auto load_a = [&](unsigned int k0) {
        if (!KTAIL || k0 + WM_BK <= K) {
            _Pragma("unroll") for (int v = 0; v < A_VECS; v++) ra[v] = *(const wm_s8*)(pa[v] + k0);
        } else {
            _Pragma("unroll") for (int v = 0; v < A_VECS; v++)
                ra[v] = k0 + a_c < K ? *(const wm_s8*)(pa[v] + k0) : wm_s8{};
        }
    };
    auto load_b = [&](wm_s8 (&dst)[KS][TN], unsigned int k0) {
        const bool tail = KTAIL && k0 + WM_BK > K;
        _Pragma("unroll") for (int kk = 0; kk < KS; kk++) {
            const unsigned int k = k0 + kk * 16 + b_row;
            const size_t row = (size_t)(tail ? min(k, K - 1) : k) * N;
            _Pragma("unroll") for (int j = 0; j < TN; j++)
                dst[kk][j] = __builtin_amdgcn_global_load_tr_b128_v8i16(
                    (wm_global_s8*)(B + row + b_col[j]));
        }
    };

    wm_f8 acc[TM][TN];
    _Pragma("unroll") for (int i = 0; i < TM; i++)
        _Pragma("unroll") for (int j = 0; j < TN; j++) acc[i][j] = wm_f8{};
    load_a(0);
    load_b(bcur, 0);
    for (unsigned int k0 = 0; k0 < K; k0 += WM_BK) {
        _Pragma("unroll") for (int v = 0; v < A_VECS; v++) {
            const unsigned int idx = t + v * THREADS;
            *(wm_s8*)(As + (idx / AV_PER_ROW) * WM_LDS + (idx % AV_PER_ROW) * 8) = ra[v];
        }
        WM_LDS_BARRIER();
        const bool more = k0 + WM_BK < K;
        if (more) {
            load_a(k0 + WM_BK);
            load_b(bnext, k0 + WM_BK);
        }
        _Pragma("unroll") for (int kk = 0; kk < KS; kk++) {
            wm_s8 af[TM];
            _Pragma("unroll") for (int i = 0; i < TM; i++)
                af[i] = *(const wm_s8*)(As + ((wm * TM + i) * 16 + (lane & 15)) * WM_LDS
                                        + kk * 16 + (lane >> 4) * 8);
            // Issue every fragment read before the first WMMA, so LDS
            // latency is paid once per step rather than once per row.
            __builtin_amdgcn_sched_barrier(0);
            _Pragma("unroll") for (int i = 0; i < TM; i++)
                _Pragma("unroll") for (int j = 0; j < TN; j++)
                    acc[i][j] = wm_mma<BF16>(af[i], bcur[kk][j], acc[i][j]);
        }
        WM_LDS_BARRIER();
        if (more) {
            _Pragma("unroll") for (int kk = 0; kk < KS; kk++)
                _Pragma("unroll") for (int j = 0; j < TN; j++) bcur[kk][j] = bnext[kk][j];
        }
    }
    // Accumulator layout: lane holds column lane%16, rows (lane/16)*8 + e.
    _Pragma("unroll") for (int i = 0; i < TM; i++)
        _Pragma("unroll") for (int j = 0; j < TN; j++) {
            const unsigned int col = col0 + (wn * TN + j) * 16 + (lane & 15);
            _Pragma("unroll") for (int e = 0; e < 8; e++) {
                const unsigned int row = row0 + (wm * TM + i) * 16 + (lane >> 4) * 8 + e;
                if (!EDGE || (row < M && col < N)) C[(size_t)row * N + col] = alpha * acc[i][j][e];
            }
        }
}

// `group` M-tiles sweep each B column panel together, so B is read about
// once from memory rather than once per row of M-tiles.
template <bool BF16, int WAVES_M, int WAVES_N, int TM, int TN>
__device__ __forceinline__ void wm_gemm(
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) {
    constexpr unsigned int BM = WAVES_M * TM * 16, BN = WAVES_N * TN * 16;
    __shared__ short As[BM * WM_LDS];
    const unsigned int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
    const unsigned int pid = blockIdx.x, per_group = group * tiles_n;
    const unsigned int first_m = (pid / per_group) * group;
    const unsigned int gsize = min(tiles_m - first_m, group);
    const unsigned int row0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned int col0 = ((pid % per_group) / gsize) * BN;
    const bool inner = row0 + BM <= M && col0 + BN <= N;
    if (inner && K % WM_BK == 0)
        wm_body<BF16, WAVES_M, WAVES_N, TM, TN, false, false>(As, A, B, C, M, N, K, row0, col0, alpha);
    else if (inner)
        wm_body<BF16, WAVES_M, WAVES_N, TM, TN, false, true>(As, A, B, C, M, N, K, row0, col0, alpha);
    else
        wm_body<BF16, WAVES_M, WAVES_N, TM, TN, true, true>(As, A, B, C, M, N, K, row0, col0, alpha);
}

#define WM_ENTRY(NAME, BF16, WAVES_M, WAVES_N, TM, TN)                                    \
extern "C" __global__ __launch_bounds__(WAVES_M * WAVES_N * 32) void NAME(               \
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,  \
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) { \
    wm_gemm<BF16, WAVES_M, WAVES_N, TM, TN>(A, B, C, M, N, K, group, alpha);              \
}
WM_ENTRY(hgemm_wmma_128_f16, false, 2, 4, 4, 2)
WM_ENTRY(hgemm_wmma_128_bf16, true, 2, 4, 4, 2)
WM_ENTRY(hgemm_wmma_64_f16, false, 1, 2, 4, 2)
WM_ENTRY(hgemm_wmma_64_bf16, true, 1, 2, 4, 2)
WM_ENTRY(hgemm_wmma_32_f16, false, 1, 4, 2, 2)
WM_ENTRY(hgemm_wmma_32_bf16, true, 1, 4, 2, 2)
#endif
"#;

/// One kernel shape: output tile `bm`×`bn`, `threads` per workgroup, and
/// its speed on full tiles relative to the 128-row kernel (4096³ and
/// 512×4096×4096 on an RX 9060 XT), used to weigh padding waste.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Variant {
    pub(crate) name: &'static str,
    pub(crate) bm: u32,
    pub(crate) bn: u32,
    pub(crate) threads: u32,
    speed: f32,
}

/// Kernel shapes, in the order [`crate::GemmKernel`] builds them.
pub(crate) const VARIANTS: [Variant; 3] = [
    Variant {
        name: "hgemm_wmma_128",
        bm: 128,
        bn: 128,
        threads: 256,
        speed: 1.0,
    },
    Variant {
        name: "hgemm_wmma_64",
        bm: 64,
        bn: 64,
        threads: 64,
        speed: 0.93,
    },
    Variant {
        name: "hgemm_wmma_32",
        bm: 32,
        bn: 64,
        threads: 128,
        speed: 0.88,
    },
];

/// Whether the WMMA kernels take this shape: their vector loads need rows
/// of A and B that are whole 16-byte units.
pub(crate) fn fits(n: u32, k: u32) -> bool {
    n.is_multiple_of(8) && k.is_multiple_of(8)
}

/// Index into [`VARIANTS`] for `m` rows. Up to 64 rows the 32-row kernel
/// wins outright (B streaming dominates; 1.2–1.3× rocBLAS). Above that,
/// the kernel with the least padded-row work per unit of speed.
pub(crate) fn pick(m: u32) -> usize {
    if m <= 64 {
        return 2;
    }
    let cost = |v: &Variant| m.div_ceil(v.bm) as f32 * v.bm as f32 / v.speed;
    (0..VARIANTS.len())
        .min_by(|&a, &b| cost(&VARIANTS[a]).total_cmp(&cost(&VARIANTS[b])))
        .unwrap_or(0)
}

/// M-tiles that sweep each B panel together. 4 was fastest or within 2% at
/// every measured shape (1, 2, 8 and 16 lose up to 45% on some).
pub(crate) const GROUP: u32 = 4;

/// Workgroups for an `m`×`n` output with `v`.
pub(crate) fn grid(v: &Variant, m: u32, n: u32) -> [u32; 3] {
    [m.div_ceil(v.bm) * n.div_ceil(v.bn), 1, 1]
}

#[cfg(test)]
mod tests {
    use super::{VARIANTS, fits, grid, pick};

    #[test]
    fn few_rows_take_the_small_tile() {
        for m in [9, 17, 32, 64] {
            assert_eq!(VARIANTS[pick(m)].bm, 32, "m={m}");
        }
    }

    #[test]
    fn padding_waste_picks_the_tile() {
        // 96 rows fill three 32-row tiles exactly.
        assert_eq!(VARIANTS[pick(96)].bm, 32);
        assert_eq!(VARIANTS[pick(128)].bm, 128);
        // 192 = three 64-row tiles, against two 128-row ones.
        assert_eq!(VARIANTS[pick(192)].bm, 64);
        assert_eq!(VARIANTS[pick(333)].bm, 128);
        assert_eq!(VARIANTS[pick(4096)].bm, 128);
    }

    #[test]
    fn unaligned_rows_do_not_fit() {
        assert!(fits(4096, 4096));
        assert!(!fits(4100, 4096));
        assert!(!fits(4096, 4095));
    }

    #[test]
    fn grid_covers_every_tile() {
        assert_eq!(grid(&VARIANTS[0], 1, 1), [1, 1, 1]);
        assert_eq!(grid(&VARIANTS[0], 129, 256), [4, 1, 1]);
        assert_eq!(grid(&VARIANTS[2], 33, 65), [4, 1, 1]);
    }
}
