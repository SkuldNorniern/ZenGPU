//! HIP GEMM on RDNA4 matrix cores (WMMA), A and B both f16 or both bf16, f32 accumulate and out.
//!
//! Kernels exist only when hipRTC targets gfx12 (`__GFX12__`); elsewhere pipeline creation fails,
//! which [`crate::GemmKernel`] takes as "no WMMA here".

/// HIP source. Each kernel is a [`Variant`]: 32-lane waves in `WAVES_M`×`WAVES_N`, each owning
/// `TM`×`TN` WMMA 16×16 tiles, K in steps of 64.
///
/// A goes through LDS. B skips LDS: `global_load_tr` loads each fragment already transposed. The
/// next step's loads go to registers during compute, and barriers wait only on LDS, so those loads
/// stay in flight.
///
/// Interior blocks have no bounds checks. Edge blocks clamp addresses and skip out-of-range stores.
/// Past K, A reads as zero, so clamped B rows add nothing. Needs `N % 8 == 0` and `K % 8 == 0`.
pub(crate) const SOURCE: &str = r#"
// gfx12 WMMA builtins exist only in wave32 code.
#if defined(__GFX12__) && (!defined(__AMDGCN_WAVEFRONT_SIZE__) || __AMDGCN_WAVEFRONT_SIZE__ == 32)
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
    // Lane l of a B fragment reads row (l/16)*8 + l%8 at column ((l%16)/8)*8; the transposing load
    // hands lane l column l%16, k (l/16)*8 .. +8.
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
            // Issue all fragment reads before the first WMMA: LDS latency is paid once per step.
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

// 128-row kernel: B fetched once per block. Each wave loads its share of the step's B fragments
// with global_load_tr into LDS in fragment order (16 B per lane, conflict-free), then reads the
// ones it needs. This halves L2 traffic for B against per-wave loads, which bounds the kernel. K
// steps of 32 keep the 64×64 per-wave accumulators in registers.
#define WM_BK_S 32
#define WM_LDS_S (WM_BK_S + 8)
template <bool BF16, int WAVES_M, int WAVES_N, int TM, int TN, bool EDGE, bool KTAIL>
__device__ __forceinline__ void wm_body_shared_b(
        short* As, short* Bs, const short* __restrict__ A, const short* __restrict__ B,
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
        unsigned int row0, unsigned int col0, float alpha) {
    constexpr unsigned int THREADS = WAVES_M * WAVES_N * 32;
    constexpr unsigned int AV_PER_ROW = WM_BK_S / 8;
    constexpr unsigned int A_ROW_STEP = THREADS / AV_PER_ROW;
    constexpr int A_VECS = TM * 16 * WAVES_M * WM_BK_S / 8 / THREADS;
    constexpr int KS = WM_BK_S / 16;
    constexpr int FRAG_COLS = WAVES_N * TN;
    constexpr int FRAGS_PER_WAVE = KS * FRAG_COLS / (WAVES_M * WAVES_N);
    const unsigned int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const unsigned int wm = wave / WAVES_N, wn = wave % WAVES_N;

    const unsigned int a_c = (t % AV_PER_ROW) * 8;
    const short* pa[A_VECS];
    _Pragma("unroll") for (int v = 0; v < A_VECS; v++) {
        const unsigned int r = row0 + t / AV_PER_ROW + v * A_ROW_STEP;
        pa[v] = A + (size_t)(EDGE ? min(r, M - 1) : r) * K + a_c;
    }
    // Fragment f of a step is (k slice f / FRAG_COLS, column f % FRAG_COLS); this wave loads f =
    // wave * FRAGS_PER_WAVE + q.
    const unsigned int b_row = (lane >> 4) * 8 + (lane & 7);
    unsigned int b_col[FRAGS_PER_WAVE], b_kk[FRAGS_PER_WAVE];
    _Pragma("unroll") for (int q = 0; q < FRAGS_PER_WAVE; q++) {
        const unsigned int f = wave * FRAGS_PER_WAVE + q;
        b_kk[q] = f / FRAG_COLS;
        const unsigned int c = col0 + (f % FRAG_COLS) * 16 + ((lane & 15) >> 3) * 8;
        b_col[q] = EDGE ? min(c, N - 8) : c;
    }

    wm_s8 ra[A_VECS], rb[FRAGS_PER_WAVE];
    auto load = [&](unsigned int k0) {
        if (!KTAIL || k0 + WM_BK_S <= K) {
            _Pragma("unroll") for (int v = 0; v < A_VECS; v++) ra[v] = *(const wm_s8*)(pa[v] + k0);
        } else {
            _Pragma("unroll") for (int v = 0; v < A_VECS; v++)
                ra[v] = k0 + a_c < K ? *(const wm_s8*)(pa[v] + k0) : wm_s8{};
        }
        const bool tail = KTAIL && k0 + WM_BK_S > K;
        _Pragma("unroll") for (int q = 0; q < FRAGS_PER_WAVE; q++) {
            const unsigned int k = k0 + b_kk[q] * 16 + b_row;
            const size_t row = (size_t)(tail ? min(k, K - 1) : k) * N;
            rb[q] = __builtin_amdgcn_global_load_tr_b128_v8i16((wm_global_s8*)(B + row + b_col[q]));
        }
    };

    wm_f8 acc[TM][TN];
    _Pragma("unroll") for (int i = 0; i < TM; i++)
        _Pragma("unroll") for (int j = 0; j < TN; j++) acc[i][j] = wm_f8{};
    load(0);
    for (unsigned int k0 = 0; k0 < K; k0 += WM_BK_S) {
        _Pragma("unroll") for (int v = 0; v < A_VECS; v++) {
            const unsigned int idx = t + v * THREADS;
            *(wm_s8*)(As + (idx / AV_PER_ROW) * WM_LDS_S + (idx % AV_PER_ROW) * 8) = ra[v];
        }
        _Pragma("unroll") for (int q = 0; q < FRAGS_PER_WAVE; q++)
            *(wm_s8*)(Bs + ((wave * FRAGS_PER_WAVE + q) * 32 + lane) * 8) = rb[q];
        WM_LDS_BARRIER();
        if (k0 + WM_BK_S < K) load(k0 + WM_BK_S);
        _Pragma("unroll") for (int kk = 0; kk < KS; kk++) {
            wm_s8 af[TM], bf[TN];
            _Pragma("unroll") for (int i = 0; i < TM; i++)
                af[i] = *(const wm_s8*)(As + ((wm * TM + i) * 16 + (lane & 15)) * WM_LDS_S
                                        + kk * 16 + (lane >> 4) * 8);
            _Pragma("unroll") for (int j = 0; j < TN; j++)
                bf[j] = *(const wm_s8*)(Bs + ((kk * FRAG_COLS + wn * TN + j) * 32 + lane) * 8);
            __builtin_amdgcn_sched_barrier(0);
            _Pragma("unroll") for (int i = 0; i < TM; i++)
                _Pragma("unroll") for (int j = 0; j < TN; j++)
                    acc[i][j] = wm_mma<BF16>(af[i], bf[j], acc[i][j]);
        }
        WM_LDS_BARRIER();
    }
    _Pragma("unroll") for (int i = 0; i < TM; i++)
        _Pragma("unroll") for (int j = 0; j < TN; j++) {
            const unsigned int col = col0 + (wn * TN + j) * 16 + (lane & 15);
            _Pragma("unroll") for (int e = 0; e < 8; e++) {
                const unsigned int row = row0 + (wm * TM + i) * 16 + (lane >> 4) * 8 + e;
                if (!EDGE || (row < M && col < N)) C[(size_t)row * N + col] = alpha * acc[i][j][e];
            }
        }
}

// `group` M-tiles sweep each B column panel together, so B is read about once per panel. SHARED_B
// picks wm_body_shared_b over wm_body.
template <bool BF16, int WAVES_M, int WAVES_N, int TM, int TN, bool SHARED_B>
__device__ __forceinline__ void wm_gemm(
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) {
    constexpr unsigned int BM = WAVES_M * TM * 16, BN = WAVES_N * TN * 16;
    constexpr unsigned int BK = SHARED_B ? WM_BK_S : WM_BK;
    __shared__ short As[BM * (SHARED_B ? WM_LDS_S : WM_LDS)];
    __shared__ short Bs[SHARED_B ? (WM_BK_S / 16) * (BN / 16) * 256 : 1];
    const unsigned int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
    const unsigned int pid = blockIdx.x, per_group = group * tiles_n;
    const unsigned int first_m = (pid / per_group) * group;
    const unsigned int gsize = min(tiles_m - first_m, group);
    const unsigned int row0 = (first_m + (pid % per_group) % gsize) * BM;
    const unsigned int col0 = ((pid % per_group) / gsize) * BN;
    const bool inner = row0 + BM <= M && col0 + BN <= N;
    if constexpr (SHARED_B) {
        if (inner && K % BK == 0)
            wm_body_shared_b<BF16, WAVES_M, WAVES_N, TM, TN, false, false>(As, Bs, A, B, C, M, N, K, row0, col0, alpha);
        else if (inner)
            wm_body_shared_b<BF16, WAVES_M, WAVES_N, TM, TN, false, true>(As, Bs, A, B, C, M, N, K, row0, col0, alpha);
        else
            wm_body_shared_b<BF16, WAVES_M, WAVES_N, TM, TN, true, true>(As, Bs, A, B, C, M, N, K, row0, col0, alpha);
    } else {
        if (inner && K % BK == 0)
            wm_body<BF16, WAVES_M, WAVES_N, TM, TN, false, false>(As, A, B, C, M, N, K, row0, col0, alpha);
        else if (inner)
            wm_body<BF16, WAVES_M, WAVES_N, TM, TN, false, true>(As, A, B, C, M, N, K, row0, col0, alpha);
        else
            wm_body<BF16, WAVES_M, WAVES_N, TM, TN, true, true>(As, A, B, C, M, N, K, row0, col0, alpha);
    }
}

#define WM_ENTRY(NAME, BF16, WAVES_M, WAVES_N, TM, TN, SHARED_B)                          \
extern "C" __global__ __launch_bounds__(WAVES_M * WAVES_N * 32) void NAME(               \
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,  \
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) { \
    wm_gemm<BF16, WAVES_M, WAVES_N, TM, TN, SHARED_B>(A, B, C, M, N, K, group, alpha);    \
}
WM_ENTRY(hgemm_wmma_128_f16, false, 2, 2, 4, 4, true)
WM_ENTRY(hgemm_wmma_128_bf16, true, 2, 2, 4, 4, true)
WM_ENTRY(hgemm_wmma_64_f16, false, 1, 2, 4, 2, false)
WM_ENTRY(hgemm_wmma_64_bf16, true, 1, 2, 4, 2, false)
WM_ENTRY(hgemm_wmma_32_f16, false, 1, 4, 2, 2, false)
WM_ENTRY(hgemm_wmma_32_bf16, true, 1, 4, 2, 2, false)
#endif
"#;

/// Describes one kernel shape: output tile `bm`×`bn`, `threads` per workgroup, and its full-tile
/// speed relative to the 128-row kernel (used to weigh padding waste).
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
        threads: 128,
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
        bn: 128,
        threads: 128,
        speed: 0.88,
    },
];

/// A shape uses the WMMA kernels only if rows of A and B are whole 16-byte units.
pub(crate) fn fits(n: u32, k: u32) -> bool {
    n.is_multiple_of(8) && k.is_multiple_of(8)
}

/// Returns the index into [`VARIANTS`] for `m` rows. Up to 64 rows the 32-row kernel wins (B
/// streaming dominates). Above that, the kernel with the least padded-row work per unit of speed.
pub(crate) fn pick(m: u32) -> usize {
    if m <= 64 {
        return 2;
    }
    let cost = |v: &Variant| m.div_ceil(v.bm) as f32 * v.bm as f32 / v.speed;
    (0..VARIANTS.len())
        .min_by(|&a, &b| cost(&VARIANTS[a]).total_cmp(&cost(&VARIANTS[b])))
        .unwrap_or(0)
}

/// M-tiles per B panel sweep. 4 is fastest or within 2% at every measured shape (1, 2, 8, 16 lose
/// up to 45%).
pub(crate) const GROUP: u32 = 4;

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
        assert_eq!(VARIANTS[pick(96)].bm, 32);
        assert_eq!(VARIANTS[pick(128)].bm, 128);
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
    fn variants_match_their_kernels() {
        let layouts = [(2, 2, 4, 4), (1, 2, 4, 2), (1, 4, 2, 2)];
        for (v, (wm, wn, tm, tn)) in VARIANTS.iter().zip(layouts) {
            assert_eq!(v.bm, wm * tm * 16, "{}", v.name);
            assert_eq!(v.bn, wn * tn * 16, "{}", v.name);
            assert_eq!(v.threads, wm * wn * 32, "{}", v.name);
        }
    }

    #[test]
    fn grid_covers_every_tile() {
        assert_eq!(grid(&VARIANTS[0], 1, 1), [1, 1, 1]);
        assert_eq!(grid(&VARIANTS[0], 129, 256), [4, 1, 1]);
        assert_eq!(grid(&VARIANTS[2], 33, 129), [4, 1, 1]);
    }
}
