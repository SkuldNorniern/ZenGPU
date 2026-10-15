//! BLAS Level 3: matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    BackendPreference, Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle,
    Result, Scalar, ShaderDesc, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::gemm_tiled::{GEMM_TILED_ZSL, TILE as TILED_GEMM_TILE};

/// Computes `C[m,n] = alpha * sum_k A[m,k] * B[k,n]`.
///
/// Naive (untiled) f32 GEMM with a 16×16 workgroup.
const GEMM_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, alpha: f32 }
    @workgroup_size(16, 16)
    kernel cs_gemm(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: P, id: global_id) {
        let row = id.y
        let col = id.x
        if row < p.m && col < p.n {
            let sum: f32 = 0.0
            for i in 0..p.k {
                sum = sum + a[row * p.k + i] * b[i * p.n + col]
            }
            c[row * p.n + col] = p.alpha * sum
        }
    }
);

/// Compiled GEMM pipeline. Create once per device, reuse across dispatches.
pub struct GemmKernel {
    shader: ShaderHandle,
    pub pipeline: PipelineHandle,
    /// Output tile per workgroup: 16 for the naive kernel, 128 for the register-blocked ones.
    tile: u32,
}

/// Register-blocked f32 GEMM for AMD GPUs (HIP).
///
/// Each 256-thread block computes a 128×128 tile of C from 8-deep slices of A and B in LDS. Each
/// thread accumulates an 8×8 sub-block in registers with a 16-element stride, so LDS reads are
/// conflict-free. A is staged transposed (`As[k][m]`), padded by 4 against bank conflicts on the
/// transposed store. Edges zero-fill on load and mask on store.
///
/// LDS use is 8.4 KiB, so it fits every gfx9-gfx12 part.
const GEMM_HIP_TILED: &str = r#"
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
extern "C" __global__ __launch_bounds__(256)
void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                 float* __restrict__ C, unsigned int M, unsigned int N,
                 unsigned int K, float alpha) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    float acc[TM][TN] = {};
    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int s = 0; s < (BM * BK) / 256; ++s) {
            int idx = tid + s * 256;
            int r = idx / BK, c = idx % BK;
            unsigned int gm = bm + r, gk = k0 + c;
            As[c][r] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < (BN * BK) / 256; ++s) {
            int idx = tid + s * 256;
            int r = idx / BN, c = idx % BN;
            unsigned int gk = k0 + r, gn = bn + c;
            Bs[r][c] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;
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
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        unsigned int gm = bm + ty + i * 16;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                unsigned int gn = bn + tx + j * 16;
                if (gn < N) C[(size_t)gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}
"#;

impl GemmKernel {
    /// Compiles the GEMM pipeline. HIP uses the hand-written kernel, Vulkan uses register-blocked
    /// ZSL, and other backends use naive ZSL.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        if device.backend() == BackendPreference::Vulkan {
            let (shader, entry) = GEMM_TILED_ZSL.create(device)?;
            let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [16, 16, 1],
            });
            return match pipeline {
                Ok(pipeline) => Ok(Self {
                    shader,
                    pipeline,
                    tile: TILED_GEMM_TILE,
                }),
                Err(e) => {
                    device.destroy_shader(shader);
                    Err(e)
                }
            };
        }
        if device.backend() == BackendPreference::Hip {
            let shader = device.create_shader(ShaderDesc::hip(GEMM_HIP_TILED))?;
            let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm_tiled",
                block: [256, 1, 1],
            });
            return match pipeline {
                Ok(pipeline) => Ok(Self {
                    shader,
                    pipeline,
                    tile: 128,
                }),
                Err(e) => {
                    device.destroy_shader(shader);
                    Err(e)
                }
            };
        }
        let (shader, entry) = GEMM_ZSL.create(device)?;
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
            block: [16, 16, 1],
        })?;
        Ok(Self {
            shader,
            pipeline,
            tile: 16,
        })
    }

    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`, row-major f32. `C` is allocated from `pool`.
    pub fn sgemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(GpuError::Dispatch(format!(
                "sgemm: only f32 is supported, got {:?} and {:?}",
                a.dtype, b.dtype
            )));
        }
        let (&[m, k], &[k2, n]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(GpuError::Dispatch(format!(
                "sgemm: expected 2D arrays, got shapes {:?} and {:?}",
                a.shape, b.shape
            )));
        };
        if k != k2 {
            return Err(GpuError::Dispatch(format!(
                "sgemm: inner dimensions mismatch ({k} vs {k2})"
            )));
        }

        let c = pool.alloc(vec![m, n], DType::F32)?;
        device.dispatch(
            self.pipeline,
            Bindings {
                buffers: &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                scalars: &[
                    Scalar::U32(m),
                    Scalar::U32(n),
                    Scalar::U32(k),
                    Scalar::F32(alpha),
                ],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(self.tile), m.div_ceil(self.tile), 1],
        )?;
        Ok(c)
    }

    /// Computes `C = A @ B`. This is [`Self::sgemm`] with `alpha = 1.0`.
    pub fn gemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        self.sgemm(device, pool, 1.0, a, b)
    }
}
