//! CUDA GEMM on tensor cores (`mma.sync.m16n8k16`) for f16 or bf16 A and B, f32 accumulate
//! and store.
//!
//! `C[M,N] = alpha * A[M,K] @ B[K,N]`, row-major. Measured on an RTX 4090 (sm_89) through
//! `GemmKernel::sgemm`, 10 sgemms per replay, against cuBLAS `GemmEx` with the same types in the
//! same session:
//!
//! | shape            | this f16 / bf16 | cuBLAS f16 / bf16 |
//! |------------------|-----------------|-------------------|
//! | 1×4096×4096      | 1.9-2.4 / 2.4-2.6 TF | 0.28 / 0.29 TF |
//! | 16×4096×4096     | 33-35 / 26-32 TF     | 1.8 / 2.8 TF   |
//! | 1024³            | 63-93 / 76-93 TF     | 97 / 56 TF     |
//! | 2048³            | 133-143 / 136-144 TF | 149 / 152 TF   |
//! | 4096³            | 151-154 / 151-167 TF | 171 / 142 TF   |
//! | 8192³            | 148-152 / 145-158 TF | 156 / 156 TF   |
//!
//! f32 (m > 16, N and K multiples of 4) uses `sgemm_simt128`, a hand-written SIMT kernel: 128x128
//! tiles, 8x8 outputs per thread, float4 loads, double-buffered register prefetch. Against the
//! tiled ZSL kernel ([`crate::gemm_tiled`], 26.6-29.8 TF at 1024³ and up):
//!
//! | shape            | f32 SIMT (this) | f32 ZSL (tiled) |
//! |------------------|-----------------|------------------|
//! | 1024³            | 41-44 TF        | 27 TF            |
//! | 2048³            | 38-47 TF        | 30 TF            |
//! | 4096³            | 40-42 TF        | 27 TF            |
//! | 8192³            | 41 TF           | 27 TF            |
//!
//! Error at 512x512x4096 (inputs in [-1, 1), max error / max |C| vs f64) is 2.7e-6 for both. 3xTF32
//! (Ootomo & Yokota) ran at 17-22 TF from 1024³ up, so it is not used. Accumulating all three mma
//! products in one register set grew error with K (4.5e-5). Adding each k step's products to the
//! f32 accumulator with a plain add gave 1.0e-6, which points at the tensor-core accumulator's
//! rounding. Run-to-run variation is about ±10% for f32.

/// One block-tile kernel per entry, each with an `_f16` and a `_bf16` entry point (see [`SOURCE`]).
pub(crate) struct Variant {
    pub(crate) name: &'static str,
    pub(crate) bm: u32,
    pub(crate) bn: u32,
    pub(crate) threads: u32,
}

/// Tensor-core kernels, indexed by [`pick`]: 128x128 for large outputs, 64x64 when 128x128 would
/// leave SMs idle, and the few-row split-K kernel.
pub(crate) const VARIANTS: [Variant; 3] = [
    Variant {
        name: "sgemm_tc128",
        bm: 128,
        bn: 128,
        threads: 256,
    },
    Variant {
        name: "sgemm_tc64",
        bm: 64,
        bn: 64,
        threads: 128,
    },
    Variant {
        name: "sgemm_tc16",
        bm: 16,
        bn: 64,
        threads: 128,
    },
];

pub(crate) const SKINNY: usize = 2;

/// Up to this many rows, [`pick`] uses the few-row split-K kernel.
pub(crate) const SKINNY_ROWS: u32 = 16;

/// Minimum number of 128x128 tiles before the large tile is used: two per SM (the RTX 4090
/// has 128 SMs).
const FULL_TILES: u64 = 256;

/// Requires A and B both f16 or both bf16, with N and K multiples of 8.
///
/// Rows must be whole 16-byte chunks for `cp.async`.
pub(crate) fn fits(n: u32, k: u32) -> bool {
    n.is_multiple_of(8) && k.is_multiple_of(8)
}

/// The f32 SIMT kernel requires N and K multiples of 4, so rows are whole 16-byte units for its
/// float4 loads.
pub(crate) fn simt_fits(n: u32, k: u32) -> bool {
    n.is_multiple_of(4) && k.is_multiple_of(4)
}

/// Picks split-K for at most [`SKINNY_ROWS`] rows, 128×128 with at least [`FULL_TILES`] tiles,
/// otherwise 64×64.
pub(crate) fn pick(m: u32, n: u32) -> usize {
    if m <= SKINNY_ROWS {
        return SKINNY;
    }
    let tiles = u64::from(m.div_ceil(128)) * u64::from(n.div_ceil(128));
    if tiles >= FULL_TILES { 0 } else { 1 }
}

/// Returns `(splits, kchunk)`.
///
/// `kchunk` is a multiple of 32 and at least 64. Splits grow until about 512 blocks.
pub(crate) fn split(n: u32, k: u32) -> (u32, u32) {
    let col_blocks = n.div_ceil(64).max(1);
    let want = (k / 64).clamp(1, (512 / col_blocks).max(1));
    let kchunk = (k.div_ceil(want).div_ceil(32) * 32).max(32);
    (k.div_ceil(kchunk).max(1), kchunk)
}

pub(crate) fn grid(variant: usize, m: u32, n: u32, splits: u32) -> [u32; 3] {
    let v = &VARIANTS[variant];
    [n.div_ceil(v.bn), m.div_ceil(v.bm), splits]
}

/// NVRTC source. Params: A, B, C, `M`, `N`, `K`, `kchunk` and `alpha`.
///
/// `blockIdx.z` picks the K range and the output slice `C + z*M*N`.
pub(crate) const SOURCE: &str = r#"
// Tensor-core GEMM over raw 16-bit lanes (f16 or bf16), f32 accumulate and store.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define TC_STAGES 3
#define TC_BK 32

__device__ __forceinline__ unsigned tc_smem(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void tc_cp16(unsigned dst, const void* src, bool valid) {
    const unsigned sz = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void tc_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tc_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void tc_ldx4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void tc_ldx4_t(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
template <bool BF16>
__device__ __forceinline__ void tc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if constexpr (BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}

// BK = 32, TC_STAGES-deep cp.async pipeline. A tile row-major [BM][32]: 16-byte chunk c of row r at
// c ^ ((r>>1)&3). B tile [32][BN]: chunk c of row k at c ^ (k&7). blockIdx.z: K range (split-K),
// partial C at C + z*M*N.
template <bool BF16, int BM, int BN, int WM, int WN>
__device__ __forceinline__ void tc_gemm(const unsigned short* __restrict__ A,
                                        const unsigned short* __restrict__ B,
                                        float* __restrict__ C, unsigned M, unsigned N,
                                        unsigned K, unsigned kchunk, float alpha) {
    constexpr int WARPS_N = BN / WN;
    constexpr int THREADS = (BM / WM) * WARPS_N * 32;
    constexpr int MI = WM / 16, NI = WN / 8;
    constexpr int A_STAGE = BM * TC_BK, B_STAGE = TC_BK * BN;
    constexpr int A_CHUNKS = (BM * (TC_BK / 8) + THREADS - 1) / THREADS;
    constexpr int B_CHUNKS = (TC_BK * (BN / 8) + THREADS - 1) / THREADS;
    __shared__ __align__(128) unsigned short smem[TC_STAGES * (A_STAGE + B_STAGE)];

    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned wm = warp / WARPS_N, wn = warp % WARPS_N;
    const unsigned bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const unsigned kb = blockIdx.z * kchunk;
    const unsigned ke = min(kb + kchunk, K);
    const unsigned nk = ke > kb ? (ke - kb + TC_BK - 1) / TC_BK : 0;

    auto load_stage = [&](unsigned stage, unsigned k0) {
        unsigned short* a_s = smem + stage * A_STAGE;
        unsigned short* b_s = smem + TC_STAGES * A_STAGE + stage * B_STAGE;
#pragma unroll
        for (int s = 0; s < A_CHUNKS; ++s) {
            const unsigned idx = tid + s * THREADS;
            if (idx >= BM * 4) break;
            const unsigned r = idx >> 2, c = idx & 3;
            const unsigned gr = bm + r, gk = k0 + c * 8;
            const bool valid = idx < BM * 4 && gr < M && gk < ke;
            tc_cp16(tc_smem(a_s + r * TC_BK + ((c ^ ((r >> 1) & 3)) << 3)),
                    valid ? A + (size_t)gr * K + gk : A, valid);
        }
#pragma unroll
        for (int s = 0; s < B_CHUNKS; ++s) {
            const unsigned idx = tid + s * THREADS;
            if (idx >= TC_BK * (BN / 8)) break;
            const unsigned r = idx / (BN / 8), c = idx % (BN / 8);
            const unsigned gk = k0 + r, gn = bn + c * 8;
            const bool valid = idx < TC_BK * (BN / 8) && gk < ke && gn < N;
            tc_cp16(tc_smem(b_s + r * BN + ((c ^ (r & 7)) << 3)),
                    valid ? B + (size_t)gk * N + gn : B, valid);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < TC_STAGES - 1; ++s) {
        if (s < nk) load_stage(s, kb + s * TC_BK);
        tc_commit();
    }
    for (unsigned kt = 0; kt < nk; ++kt) {
        tc_wait<TC_STAGES - 2>();
        __syncthreads();
        const unsigned nxt = kt + TC_STAGES - 1;
        if (nxt < nk) load_stage(nxt % TC_STAGES, kb + nxt * TC_BK);
        tc_commit();

        const unsigned stage = kt % TC_STAGES;
        const unsigned short* a_s = smem + stage * A_STAGE;
        const unsigned short* b_s = smem + TC_STAGES * A_STAGE + stage * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < TC_BK / 16; ++kk) {
            unsigned af[MI][4];
            unsigned bf[NI][2];
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const unsigned row = wm * WM + i * 16 + (lane & 15);
                const unsigned chunk = kk * 2 + (lane >> 4);
                tc_ldx4(af[i], tc_smem(a_s + row * TC_BK + ((chunk ^ ((row >> 1) & 3)) << 3)));
            }
#pragma unroll
            for (int j = 0; j < NI / 2; ++j) {
                const unsigned k = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const unsigned n = wn * WN + j * 16 + (lane >> 4) * 8;
                unsigned r[4];
                tc_ldx4_t(r, tc_smem(b_s + k * BN + (((n >> 3) ^ (k & 7)) << 3)));
                bf[2 * j][0] = r[0];
                bf[2 * j][1] = r[1];
                bf[2 * j + 1][0] = r[2];
                bf[2 * j + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) tc_mma<BF16>(acc[i][j], af[i], bf[j][0], bf[j][1]);
        }
    }
    tc_wait<0>();

    float* out = C + (size_t)blockIdx.z * M * N;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const unsigned row = bm + wm * WM + i * 16 + (lane >> 2);
            const unsigned col = bn + wn * WN + j * 8 + (lane & 3) * 2;
            if (row < M && col < N)
                *reinterpret_cast<float2*>(out + (size_t)row * N + col) =
                    make_float2(alpha * acc[i][j][0], alpha * acc[i][j][1]);
            if (row + 8 < M && col < N)
                *reinterpret_cast<float2*>(out + (size_t)(row + 8) * N + col) =
                    make_float2(alpha * acc[i][j][2], alpha * acc[i][j][3]);
        }
}

#define TC_ENTRY(NAME, BF, BM, BN, WM, WN)                                               \
    extern "C" __global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) NAME(        \
            const unsigned short* A, const unsigned short* B, float* C, unsigned M,      \
            unsigned N, unsigned K, unsigned kchunk, float alpha) {                      \
        tc_gemm<BF, BM, BN, WM, WN>(A, B, C, M, N, K, kchunk, alpha);                    \
    }
TC_ENTRY(sgemm_tc128_f16, false, 128, 128, 64, 32)
TC_ENTRY(sgemm_tc128_bf16, true, 128, 128, 64, 32)
TC_ENTRY(sgemm_tc64_f16, false, 64, 64, 32, 32)
TC_ENTRY(sgemm_tc64_bf16, true, 64, 64, 32, 32)
TC_ENTRY(sgemm_tc16_f16, false, 16, 64, 16, 16)
TC_ENTRY(sgemm_tc16_bf16, true, 16, 64, 16, 16)

// f32 SIMT: 128×128 block tile, BK = 8, 8×8 outputs per thread. A is staged as [k][m], so each k
// step reads A and B fragments as float4 pairs. Loads prefetch one tile ahead into a
// double-buffered tile. One __syncthreads per k tile.
extern "C" __global__ void __launch_bounds__(256) sgemm_simt128(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        unsigned M, unsigned N, unsigned K, float alpha) {
    __shared__ __align__(16) float As[2][8][128 + 4];
    __shared__ __align__(16) float Bs[2][8][128];
    const unsigned tid = threadIdx.x;
    const unsigned tx = tid & 15, ty = tid >> 4;
    const unsigned bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    const unsigned ar = tid >> 1, ac = (tid & 1) * 4;
    const unsigned br = tid >> 5, bc = (tid & 31) * 4;
    const bool a_row = bm + ar < M, b_col = bn + bc < N;
    const float* a_src = A + (size_t)(bm + ar) * K + ac;
    const float* b_src = B + (size_t)br * N + bn + bc;

    float4 ra, rb;
    auto fetch = [&](unsigned k0) {
        ra = a_row && k0 + ac < K ? *reinterpret_cast<const float4*>(a_src + k0)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        rb = b_col && k0 + br < K ? *reinterpret_cast<const float4*>(b_src + (size_t)k0 * N)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    auto store = [&](unsigned buf) {
        As[buf][ac][ar] = ra.x;
        As[buf][ac + 1][ar] = ra.y;
        As[buf][ac + 2][ar] = ra.z;
        As[buf][ac + 3][ar] = ra.w;
        *reinterpret_cast<float4*>(&Bs[buf][br][bc]) = rb;
    };

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const unsigned nk = (K + 7) / 8;
    fetch(0);
    store(0);
    __syncthreads();
    for (unsigned kt = 0; kt < nk; ++kt) {
        const unsigned cur = kt & 1;
        if (kt + 1 < nk) fetch((kt + 1) * 8);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][k][64 + tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        if (kt + 1 < nk) store(cur ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned row = bm + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (row >= M) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned col = bn + h * 64 + tx * 4;
            if (col < N)
                *reinterpret_cast<float4*>(C + (size_t)row * N + col) =
                    make_float4(alpha * acc[i][h * 4], alpha * acc[i][h * 4 + 1],
                                alpha * acc[i][h * 4 + 2], alpha * acc[i][h * 4 + 3]);
        }
    }
}

extern "C" __global__ void sgemm_tc_reduce(const float* ws, float* out, unsigned mn, unsigned split) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    float s = 0.0f;
    for (unsigned z = 0; z < split; ++z) s += ws[(size_t)z * mn + i];
    out[i] = s;
}
#endif
"#;
