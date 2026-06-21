//! BLAS Level 3 — matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderDesc, ShaderHandle,
};
use zengpu_spirv::zengpu_spirv;

/// `C[m,n] += alpha * sum_k A[m,k] * B[k,n]`
///
/// Naive (untiled) f32 GEMM. 16×16 local workgroup.
const GEMM_SPV: &[u32] = zengpu_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint  a_idx;
        uint  b_idx;
        uint  c_idx;
        uint  m;
        uint  n;
        uint  k;
        float alpha;
    } pc;

    layout(local_size_x = 16, local_size_y = 16) in;

    void main() {
        uint row = gl_GlobalInvocationID.y;
        uint col = gl_GlobalInvocationID.x;
        if (row < pc.m && col < pc.n) {
            float sum = 0.0;
            for (uint i = 0; i < pc.k; i++) {
                sum += g_bufs[pc.a_idx].data[row * pc.k + i]
                     * g_bufs[pc.b_idx].data[i  * pc.n + col];
            }
            g_bufs[pc.c_idx].data[row * pc.n + col] = pc.alpha * sum;
        }
    }
    "#,
    comp,
    vulkan1_2
);

fn spv_bytes(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

/// Compiled GEMM pipeline. Create once per device, reuse across dispatches.
///
/// This implements the Level-3 BLAS `sgemm` operation.
pub struct GemmKernel {
    shader: ShaderHandle,
    pub pipeline: PipelineHandle,
}

impl GemmKernel {
    /// Compile the GEMM pipeline on `device`. For the CPU oracle, also
    /// register a matching kernel via `CpuDevice::register_kernel`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let shader = device.create_shader(ShaderDesc {
            spirv: spv_bytes(GEMM_SPV),
        })?;
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "main",
        })?;
        Ok(Self { shader, pipeline })
    }

    /// Destroy the pipeline and shader created by [`Self::new`].
    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`  (row-major, f32).
    ///
    /// `a.shape = [m, k]`, `b.shape = [k, n]`. `C` is allocated from `pool`
    /// and returned as a new `[m, n]` `DeviceArray`. `alpha = 1.0` gives
    /// the simple `C = A @ B` form.
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
                buffers:  &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                scalars:  &[
                    Scalar::U32(m),
                    Scalar::U32(n),
                    Scalar::U32(k),
                    Scalar::F32(alpha),
                ],
                textures: &[],
            },
            [n.div_ceil(16), m.div_ceil(16), 1],
        )?;
        Ok(c)
    }

    /// `C = A @ B` — shorthand for [`Self::sgemm`] with `alpha = 1.0`.
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
