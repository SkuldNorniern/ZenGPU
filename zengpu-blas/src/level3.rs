//! BLAS Level 3 — matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

/// `C[m,n] = alpha * sum_k A[m,k] * B[k,n]`
///
/// Naive (untiled) f32 GEMM. 16×16 local workgroup.
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

/// Naive f32 GEMM in PTX assembly (sm_70+), for non-Vulkan backends (CUDA/HIP).
///
/// Kernel signature (C equivalent):
/// `__global__ void cs_gemm(float* a, float* b, float* c, u32 m, u32 n, u32 k, float alpha)`
///
/// Each thread computes one output element. Block: 16×16.
const GEMM_PTX: &[u8] = b"\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry main(\n\
    .param .u64 pa, .param .u64 pb, .param .u64 pc,\n\
    .param .u32 pm, .param .u32 pn, .param .u32 pk,\n\
    .param .f32 palpha\n\
)\n\
{\n\
    .reg .pred  %p<2>;\n\
    .reg .u32   %r<12>;\n\
    .reg .u64   %rd<8>;\n\
    .reg .f32   %f<6>;\n\
\n\
    ld.param.u64 %rd0, [pa];\n\
    ld.param.u64 %rd1, [pb];\n\
    ld.param.u64 %rd2, [pc];\n\
    ld.param.u32 %r0,  [pm];\n\
    ld.param.u32 %r1,  [pn];\n\
    ld.param.u32 %r2,  [pk];\n\
    ld.param.f32 %f0,  [palpha];\n\
\n\
    mov.u32 %r3, %tid.y; mov.u32 %r4, %ntid.y; mov.u32 %r5, %ctaid.y;\n\
    mad.lo.u32 %r6, %r5, %r4, %r3;\n\
    mov.u32 %r3, %tid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %ctaid.x;\n\
    mad.lo.u32 %r7, %r5, %r4, %r3;\n\
\n\
    setp.ge.u32 %p0, %r6, %r0;\n\
    setp.ge.u32 %p1, %r7, %r1;\n\
    or.pred     %p0, %p0, %p1;\n\
    @%p0 bra done;\n\
\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u32 %r8, 0;\n\
loop:\n\
    setp.ge.u32 %p0, %r8, %r2;\n\
    @%p0 bra exit_loop;\n\
    mul.lo.u32 %r9, %r6, %r2; add.u32 %r9, %r9, %r8;\n\
    cvt.u64.u32 %rd3, %r9; shl.b64 %rd3, %rd3, 2; add.u64 %rd3, %rd0, %rd3;\n\
    ld.global.f32 %f2, [%rd3];\n\
    mul.lo.u32 %r9, %r8, %r1; add.u32 %r9, %r9, %r7;\n\
    cvt.u64.u32 %rd4, %r9; shl.b64 %rd4, %rd4, 2; add.u64 %rd4, %rd1, %rd4;\n\
    ld.global.f32 %f3, [%rd4];\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    add.u32 %r8, %r8, 1;\n\
    bra loop;\n\
exit_loop:\n\
    mul.f32 %f1, %f0, %f1;\n\
    mul.lo.u32 %r9, %r6, %r1; add.u32 %r9, %r9, %r7;\n\
    cvt.u64.u32 %rd5, %r9; shl.b64 %rd5, %rd5, 2; add.u64 %rd5, %rd2, %rd5;\n\
    st.global.f32 [%rd5], %f1;\n\
done:\n\
    ret;\n\
}\n\0";

fn create_gemm_shader(device: &dyn GpuDevice) -> Result<ShaderHandle> {
    device
        .create_shader(GEMM_ZSL.spirv_desc())
        .or_else(|_| device.create_shader(zengpu_hal::ShaderDesc::ptx(GEMM_PTX)))
}

/// Compiled GEMM pipeline. Create once per device, reuse across dispatches.
///
/// This implements the Level-3 BLAS `sgemm` operation.
pub struct GemmKernel {
    shader: ShaderHandle,
    pub pipeline: PipelineHandle,
}

impl GemmKernel {
    /// Compile the GEMM pipeline on `device`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let shader = create_gemm_shader(device)?;
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "main",
            block: [16, 16, 1],
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
