//! HIP GEMM on RDNA4 matrix cores (WMMA) for half-precision A and B.
//!
//! `C[M,N] = alpha * A[M,K] @ B[K,N]`, row-major, A and B both f16 or both
//! bf16, accumulated and stored in f32. On an RX 9060 XT this runs at
//! 40–50 TFLOP/s for large shapes, against ~10 TFLOP/s for the f32 FMA
//! kernels in [`crate::gemm_hip`].
//!
//! The kernels only exist when hipRTC compiles for a gfx12 target
//! (`__GFX12__`); elsewhere the module is empty and pipeline creation fails,
//! which [`crate::GemmKernel`] takes as "no WMMA here".

/// HIP source. 256 threads (8 wave32) per 128×128 output tile, 64×32 per
/// wave as 4×2 WMMA 16×16×16 tiles, K in steps of 32 through LDS. B is
/// transposed in registers so every LDS fragment read is one 16-byte load.
/// The next K step's tiles are loaded into registers while the current one
/// runs. Needs `K % 8 == 0` and `N % 8 == 0`.
pub(crate) const SOURCE: &str = r#"
#if defined(__GFX12__)
typedef short wm_s8 __attribute__((ext_vector_type(8)));
typedef short wm_s2 __attribute__((ext_vector_type(2)));
typedef _Float16 wm_h8 __attribute__((ext_vector_type(8)));
typedef float wm_f8 __attribute__((ext_vector_type(8)));

#define WM_BM 128
#define WM_BN 128
#define WM_BK 32
#define WM_LDS (WM_BK + 8)

template <bool BF16>
__device__ __forceinline__ wm_f8 wmma(wm_s8 a, wm_s8 b, wm_f8 c) {
    if constexpr (BF16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            __builtin_bit_cast(wm_h8, a), __builtin_bit_cast(wm_h8, b), c);
    }
}

// `group` M-tiles sweep each B column panel together; > 1 when B is bigger
// than the GPU's cache, so B is read about once rather than once per M-tile.
template <bool BF16>
__device__ __forceinline__ void hgemm_wmma(
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) {
    __shared__ short As[WM_BM * WM_LDS];
    __shared__ short Bt[WM_BN * WM_LDS];
    const unsigned int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const unsigned int wm = wave >> 2, wn = wave & 3;
    const unsigned int tiles_m = (M + WM_BM - 1) / WM_BM, tiles_n = (N + WM_BN - 1) / WM_BN;
    const unsigned int pid = blockIdx.x, per_group = group * tiles_n;
    const unsigned int first_m = (pid / per_group) * group;
    const unsigned int gsize = min(tiles_m - first_m, group);
    const unsigned int row0 = (first_m + (pid % per_group) % gsize) * WM_BM;
    const unsigned int col0 = ((pid % per_group) / gsize) * WM_BN;

    // A: two 8-wide k vectors per thread. B: 8 k-rows of 2 columns.
    const unsigned int a_r = t >> 2, a_c = (t & 3) * 8;
    const unsigned int b_n = (t & 63) * 2, b_k = (t >> 6) * 8;
    wm_s8 ra[2];
    wm_s2 rb[8];
    auto load = [&](unsigned int k0) {
        _Pragma("unroll") for (int v = 0; v < 2; v++) {
            const unsigned int gr = row0 + a_r + v * 64, gk = k0 + a_c;
            ra[v] = (gr < M && gk < K) ? *(const wm_s8*)(A + (size_t)gr * K + gk) : wm_s8{};
        }
        const unsigned int gc = col0 + b_n;
        _Pragma("unroll") for (int j = 0; j < 8; j++) {
            const unsigned int gk = k0 + b_k + j;
            rb[j] = (gk < K && gc < N) ? *(const wm_s2*)(B + (size_t)gk * N + gc) : wm_s2{};
        }
    };

    wm_f8 acc[4][2];
    _Pragma("unroll") for (int i = 0; i < 4; i++)
        _Pragma("unroll") for (int j = 0; j < 2; j++) acc[i][j] = wm_f8{};
    load(0);
    for (unsigned int k0 = 0; k0 < K; k0 += WM_BK) {
        _Pragma("unroll") for (int v = 0; v < 2; v++)
            *(wm_s8*)(As + (a_r + v * 64) * WM_LDS + a_c) = ra[v];
        wm_s8 c0, c1;
        _Pragma("unroll") for (int j = 0; j < 8; j++) { c0[j] = rb[j][0]; c1[j] = rb[j][1]; }
        *(wm_s8*)(Bt + b_n * WM_LDS + b_k) = c0;
        *(wm_s8*)(Bt + (b_n + 1) * WM_LDS + b_k) = c1;
        __syncthreads();
        if (k0 + WM_BK < K) load(k0 + WM_BK);
        _Pragma("unroll") for (int kk = 0; kk < WM_BK; kk += 16) {
            wm_s8 af[4], bf[2];
            _Pragma("unroll") for (int i = 0; i < 4; i++)
                af[i] = *(const wm_s8*)(As + (wm * 64 + i * 16 + (lane & 15)) * WM_LDS
                                         + kk + (lane >> 4) * 8);
            _Pragma("unroll") for (int j = 0; j < 2; j++)
                bf[j] = *(const wm_s8*)(Bt + (wn * 32 + j * 16 + (lane & 15)) * WM_LDS
                                         + kk + (lane >> 4) * 8);
            _Pragma("unroll") for (int i = 0; i < 4; i++)
                _Pragma("unroll") for (int j = 0; j < 2; j++)
                    acc[i][j] = wmma<BF16>(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }
    // gfx12 accumulator layout: lane holds column lane%16, rows (lane/16)*8 + e.
    _Pragma("unroll") for (int i = 0; i < 4; i++)
        _Pragma("unroll") for (int j = 0; j < 2; j++) {
            const unsigned int col = col0 + wn * 32 + j * 16 + (lane & 15);
            _Pragma("unroll") for (int e = 0; e < 8; e++) {
                const unsigned int row = row0 + wm * 64 + i * 16 + (lane >> 4) * 8 + e;
                if (row < M && col < N) C[(size_t)row * N + col] = alpha * acc[i][j][e];
            }
        }
}

extern "C" __global__ __launch_bounds__(256) void hgemm_wmma_f16(
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) {
    hgemm_wmma<false>(A, B, C, M, N, K, group, alpha);
}

extern "C" __global__ __launch_bounds__(256) void hgemm_wmma_bf16(
        const short* __restrict__ A, const short* __restrict__ B, float* __restrict__ C,
        unsigned int M, unsigned int N, unsigned int K, unsigned int group, float alpha) {
    hgemm_wmma<true>(A, B, C, M, N, K, group, alpha);
}
#endif
"#;

/// Output tile edge of the WMMA kernels.
pub(crate) const TILE: u32 = 128;

/// Whether the WMMA kernels take this shape: their vector loads need rows
/// of A and B that are whole 16-byte and 4-byte units.
pub(crate) fn fits(n: u32, k: u32) -> bool {
    n.is_multiple_of(8) && k.is_multiple_of(8)
}

/// M-tiles that sweep each B panel together: 1 (row-major tile order) while
/// B fits in a 32 MiB last-level cache, 8 once it does not. Measured on an
/// RX 9060 XT: 4096³ is fastest at 1, 1024×11008×4096 at 8.
pub(crate) fn group(n: u32, k: u32) -> u32 {
    if u64::from(n) * u64::from(k) * 2 > 32 << 20 {
        8
    } else {
        1
    }
}

/// Workgroups for an `m`×`n` output.
pub(crate) fn grid(m: u32, n: u32) -> [u32; 3] {
    [m.div_ceil(TILE) * n.div_ceil(TILE), 1, 1]
}

#[cfg(test)]
mod tests {
    use super::{fits, grid, group};

    #[test]
    fn big_weights_group_tiles() {
        assert_eq!(group(4096, 4096), 1);
        assert_eq!(group(11008, 4096), 8);
    }

    #[test]
    fn unaligned_rows_do_not_fit() {
        assert!(fits(4096, 4096));
        assert!(!fits(4100, 4096));
        assert!(!fits(4096, 4095));
        assert!(!fits(1001, 64));
    }

    #[test]
    fn grid_covers_every_tile() {
        assert_eq!(grid(1, 1), [1, 1, 1]);
        assert_eq!(grid(129, 256), [4, 1, 1]);
    }
}
