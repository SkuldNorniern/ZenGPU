//! BLAS Level 3: matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

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
}

impl GemmKernel {
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let (shader, entry) = GEMM_ZSL.create(device)?;
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
            block: [16, 16, 1],
        })?;
        Ok(Self { shader, pipeline })
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
            [n.div_ceil(16), m.div_ceil(16), 1],
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
