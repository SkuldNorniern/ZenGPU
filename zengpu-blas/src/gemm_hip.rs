//! Hand-written f32 GEMM kernels for AMD GPUs (HIP), and the choice between them per shape.
//!
//! - Large outputs: a 128×128 tile per 256-thread workgroup, 8×8 register block per thread.
//! - Small outputs: split K across workgroups (partial products, then a deterministic reduction).
//!    Below ~384², a 64×64 tile with a 4×4 register block.
//! - M ≤ 32 (LLM decode) is bandwidth-bound. Each thread streams four B columns, loads one 16-byte
//!    chunk per row of K, keeps every output row in registers, and splits K for parallelism.
//!
//! All kernels zero-fill past the edges and mask stores, so any M, N, K works. B may be f32, f16 or
//! bf16 (LLM weights); A, C and arithmetic are f32.

pub(crate) const SOURCE: &str = r#"
// Half as raw bits: hipRTC has no hip_fp16.h. Conversion goes through clang's _Float16 on every
// AMDGPU target.
struct f16 { unsigned short bits; };
struct bf16 { unsigned short bits; };

__device__ __forceinline__ float f16_bits(unsigned int bits) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)bits);
}
__device__ __forceinline__ float bf16_bits(unsigned int bits) {
    return __uint_as_float(bits << 16);
}
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(f16 v) { return f16_bits(v.bits); }
__device__ __forceinline__ float to_f32(bf16 v) { return bf16_bits(v.bits); }
__device__ __forceinline__ float half_bits_to_f32(unsigned int bits, const f16*) {
    return f16_bits(bits);
}
__device__ __forceinline__ float half_bits_to_f32(unsigned int bits, const bf16*) {
    return bf16_bits(bits);
}

// Four consecutive B elements as f32; one aligned `vec` load when N % 4 == 0.
__device__ __forceinline__ float4 load4(const float* row, unsigned int col,
                                        unsigned int N, bool vec) {
    if (vec) return *reinterpret_cast<const float4*>(row + col);
    float4 b;
    b.x = row[col];
    b.y = col + 1 < N ? row[col + 1] : 0.0f;
    b.z = col + 2 < N ? row[col + 2] : 0.0f;
    b.w = col + 3 < N ? row[col + 3] : 0.0f;
    return b;
}
template <typename H>
__device__ __forceinline__ float4 load4(const H* row, unsigned int col,
                                        unsigned int N, bool vec) {
    float4 b;
    if (vec) {
        uint2 w = *reinterpret_cast<const uint2*>(row + col);
        b.x = half_bits_to_f32(w.x & 0xffffu, row);
        b.y = half_bits_to_f32(w.x >> 16, row);
        b.z = half_bits_to_f32(w.y & 0xffffu, row);
        b.w = half_bits_to_f32(w.y >> 16, row);
        return b;
    }
    b.x = to_f32(row[col]);
    b.y = col + 1 < N ? to_f32(row[col + 1]) : 0.0f;
    b.z = col + 2 < N ? to_f32(row[col + 2]) : 0.0f;
    b.w = col + 3 < N ? to_f32(row[col + 3]) : 0.0f;
    return b;
}

template <int BM, int BN, int BK, int TM, int TN, typename BT>
__device__ __forceinline__ void tile_gemm(
        const float* __restrict__ A, const BT* __restrict__ B,
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
        unsigned int kchunk, float alpha) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const unsigned int bm = blockIdx.y * BM;
    const unsigned int bn = blockIdx.x * BN;
    const unsigned int kb = blockIdx.z * kchunk;
    const unsigned int ke = min(kb + kchunk, K);
    float acc[TM][TN] = {};
    for (unsigned int k0 = kb; k0 < ke; k0 += BK) {
        #pragma unroll
        for (int s = 0; s < (BM * BK) / 256; ++s) {
            int idx = tid + s * 256;
            int r = idx / BK, c = idx % BK;
            unsigned int gm = bm + r, gk = k0 + c;
            As[c][r] = (gm < M && gk < ke) ? A[(size_t)gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < (BN * BK) / 256; ++s) {
            int idx = tid + s * 256;
            int r = idx / BN, c = idx % BN;
            unsigned int gk = k0 + r, gn = bn + c;
            Bs[r][c] = (gk < ke && gn < N) ? to_f32(B[(size_t)gk * N + gn]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty + i * 16];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + j * 16];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    float* out = C + (size_t)blockIdx.z * M * N;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        unsigned int gm = bm + ty + i * 16;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                unsigned int gn = bn + tx + j * 16;
                if (gn < N) out[(size_t)gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

#define TILE(NAME, BT, BM, BN, BK, TM, TN)                                     \
extern "C" __global__ __launch_bounds__(256)                                   \
void NAME(const float* __restrict__ A, const BT* __restrict__ B,               \
          float* __restrict__ C, unsigned int M, unsigned int N,              \
          unsigned int K, unsigned int kchunk, float alpha) {                  \
    tile_gemm<BM, BN, BK, TM, TN, BT>(A, B, C, M, N, K, kchunk, alpha);        \
}
TILE(sgemm_t128, float, 128, 128, 8, 8, 8)
TILE(sgemm_t128_f16, f16, 128, 128, 8, 8, 8)
TILE(sgemm_t128_bf16, bf16, 128, 128, 8, 8, 8)
TILE(sgemm_t64, float, 64, 64, 16, 4, 4)
TILE(sgemm_t64_f16, f16, 64, 64, 16, 4, 4)
TILE(sgemm_t64_bf16, bf16, 64, 64, 16, 4, 4)

#define SKINNY_THREADS 128
#define SKINNY_KC_MAX 128

// Rows 0..M of C (M <= MT). This thread owns columns col..col+3; K range [k0, k0 + kc).
template <int MT, typename BT>
__device__ __forceinline__ void skinny_gemm(
        const float* __restrict__ A, const BT* __restrict__ B,
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
        unsigned int kc, float alpha) {
    __shared__ float As[MT * SKINNY_KC_MAX];
    const unsigned int col = (blockIdx.x * SKINNY_THREADS + threadIdx.x) * 4;
    const unsigned int k0 = blockIdx.y * kc;
    const unsigned int k1 = min(k0 + kc, K);
    for (unsigned int i = threadIdx.x; i < MT * kc; i += SKINNY_THREADS) {
        unsigned int m = i / kc, kk = i % kc;
        As[i] = (m < M && k0 + kk < K) ? A[(size_t)m * K + k0 + kk] : 0.0f;
    }
    __syncthreads();
    if (col >= N) return;
    float acc[MT][4] = {};
    const bool vec = (N % 4 == 0);
    for (unsigned int k = k0; k < k1; ++k) {
        const float4 b = load4(B + (size_t)k * N, col, N, vec);
        const unsigned int kk = k - k0;
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            float a = As[m * kc + kk];
            acc[m][0] += a * b.x;
            acc[m][1] += a * b.y;
            acc[m][2] += a * b.z;
            acc[m][3] += a * b.w;
        }
    }
    float* out = C + (size_t)blockIdx.y * M * N;
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < (int)M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (col + j < N) out[(size_t)m * N + col + j] = alpha * acc[m][j];
        }
    }
}

#define SKINNY(NAME, MT, BT)                                                   \
extern "C" __global__ __launch_bounds__(SKINNY_THREADS)                        \
void NAME(const float* __restrict__ A, const BT* __restrict__ B,               \
          float* __restrict__ C, unsigned int M, unsigned int N,              \
          unsigned int K, unsigned int kc, float alpha) {                      \
    skinny_gemm<MT, BT>(A, B, C, M, N, K, kc, alpha);                          \
}
#define SKINNY_ALL(MT)                                                         \
    SKINNY(sgemm_skinny_##MT, MT, float)                                       \
    SKINNY(sgemm_skinny_##MT##_f16, MT, f16)                                \
    SKINNY(sgemm_skinny_##MT##_bf16, MT, bf16)
SKINNY_ALL(1)
SKINNY_ALL(4)
SKINNY_ALL(8)
SKINNY_ALL(16)
SKINNY_ALL(32)

extern "C" __global__ __launch_bounds__(256)
void sgemm_reduce(const float* __restrict__ P, float* __restrict__ C,
                  unsigned int MN, unsigned int split) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < MN) {
        float s = 0.0f;
        for (unsigned int p = 0; p < split; ++p) s += P[(size_t)p * MN + i];
        C[i] = s;
    }
}
"#;

pub(crate) const B_TYPES: [(zengpu_hal::DType, &str); 3] = [
    (zengpu_hal::DType::F32, ""),
    (zengpu_hal::DType::F16, "_f16"),
    (zengpu_hal::DType::Bf16, "_bf16"),
];

pub(crate) const SKINNY_ROWS: [u32; 5] = [1, 4, 8, 16, 32];

/// Threads per skinny workgroup, four columns each.
pub(crate) const SKINNY_THREADS: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Kernel `sgemm_skinny_<rows>`, with K in chunks of `kc`.
    Skinny { rows: u32, kc: u32, split: u32 },
    /// Kernel `sgemm_t128` or `sgemm_t64`, picked by `tile`, with `split` K chunks of `kchunk`.
    Tile { tile: u32, kchunk: u32, split: u32 },
}

impl Plan {
    pub(crate) fn split(self) -> u32 {
        match self {
            Self::Skinny { split, .. } | Self::Tile { split, .. } => split,
        }
    }

    pub(crate) fn grid(self, m: u32, n: u32) -> [u32; 3] {
        match self {
            Self::Skinny { split, .. } => [n.div_ceil(4 * SKINNY_THREADS), split, 1],
            Self::Tile { tile, split, .. } => [n.div_ceil(tile), m.div_ceil(tile), split],
        }
    }
}

fn chunk(k: u32, split: u32, align: u32) -> u32 {
    k.div_ceil(split).div_ceil(align) * align
}

/// Picks the kernel for `C[m,n] = A[m,k] B[k,n]` on a GPU with `cus` compute units (0 if unknown).
///
/// Thresholds were measured on RX 9060 XT.
pub(crate) fn plan(m: u32, n: u32, k: u32, cus: u32) -> Plan {
    let cus = if cus == 0 { 32 } else { cus };
    let k = k.max(1);
    if m <= 32 {
        let rows = SKINNY_ROWS.into_iter().find(|&r| r >= m).unwrap_or(32);
        let kc: u32 = if rows <= 8 { 128 } else { 64 };
        let kc = kc.min(k);
        return Plan::Skinny {
            rows,
            kc,
            split: k.div_ceil(kc),
        };
    }
    let tiles128 = m.div_ceil(128) * n.div_ceil(128);
    if m > 64 && tiles128 >= 16 {
        let want = if tiles128 >= 2 * cus {
            1
        } else {
            (8 * cus).div_ceil(tiles128)
        };
        let split = want.clamp(1, 8).min((k / 64).max(1));
        let kchunk = chunk(k, split, 8);
        return Plan::Tile {
            tile: 128,
            kchunk,
            split: k.div_ceil(kchunk),
        };
    }
    let tiles64 = m.div_ceil(64) * n.div_ceil(64);
    let factor = if m <= 64 { 4 } else { 2 };
    let want = (factor * cus).div_ceil(tiles64);
    let split = want.clamp(1, 8).min((k / 192).max(1));
    let kchunk = chunk(k, split, 16);
    Plan::Tile {
        tile: 64,
        kchunk,
        split: k.div_ceil(kchunk),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_shapes_use_the_skinny_kernel() {
        assert_eq!(
            plan(1, 4096, 4096, 32),
            Plan::Skinny {
                rows: 1,
                kc: 128,
                split: 32
            }
        );
        assert_eq!(
            plan(13, 4096, 11008, 32),
            Plan::Skinny {
                rows: 16,
                kc: 64,
                split: 172
            }
        );
        assert_eq!(plan(2, 64, 40, 32).split(), 1);
    }

    #[test]
    fn large_outputs_use_the_big_tile_unsplit() {
        assert_eq!(
            plan(2048, 2048, 2048, 32),
            Plan::Tile {
                tile: 128,
                kchunk: 2048,
                split: 1
            }
        );
    }

    #[test]
    fn small_outputs_split_k() {
        assert_eq!(
            plan(512, 512, 512, 32),
            Plan::Tile {
                tile: 128,
                kchunk: 64,
                split: 8
            }
        );
        let Plan::Tile { tile, .. } = plan(384, 384, 384, 32) else {
            panic!("tile plan");
        };
        assert_eq!(tile, 64);
        for (m, n, k) in [(512, 512, 517), (777, 300, 1000), (100, 100, 64)] {
            let Plan::Tile { kchunk, split, .. } = plan(m, n, k, 32) else {
                panic!("tile plan");
            };
            assert!(
                kchunk * split >= k && kchunk * (split - 1) < k,
                "{m} {n} {k}"
            );
        }
    }

    #[test]
    fn grids_cover_the_output() {
        let p = plan(1, 4100, 4096, 32);
        assert_eq!(p.grid(1, 4100), [9, 32, 1]);
        let p = plan(1000, 1000, 1000, 32);
        let Plan::Tile { tile, split, .. } = p else {
            panic!("tile plan");
        };
        assert_eq!(
            p.grid(1000, 1000),
            [1000u32.div_ceil(tile), 1000u32.div_ceil(tile), split]
        );
    }
}
