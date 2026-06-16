//! ZenGPU BLAS bridge.
//!
//! Vendor libraries (cuBLAS, rocBLAS, oneMKL, MPS) are not used. [`GemmKernel::gemm`] is a portable
//! single-dispatch f32 GEMM over [`DeviceArray`]s.

use inline_spirv::inline_spirv;
use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderDesc, ShaderHandle,
};

/// Computes `C[m,n] = sum_k A[m,k] * B[k,n]` in untiled f32.
const GEMM_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint a_idx;
        uint b_idx;
        uint c_idx;
        uint m;
        uint n;
        uint k;
    } pc;

    layout(local_size_x = 16, local_size_y = 16) in;

    void main() {
        uint row = gl_GlobalInvocationID.y;
        uint col = gl_GlobalInvocationID.x;
        if (row < pc.m && col < pc.n) {
            float sum = 0.0;
            for (uint i = 0; i < pc.k; i++) {
                sum += g_bufs[pc.a_idx].data[row * pc.k + i]
                     * g_bufs[pc.b_idx].data[i * pc.n + col];
            }
            g_bufs[pc.c_idx].data[row * pc.n + col] = sum;
        }
    }
    "#,
    comp,
    vulkan1_2
);

fn spv_bytes(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

/// Create once per device, reuse across dispatches.
pub struct GemmKernel {
    shader: ShaderHandle,
    pub pipeline: PipelineHandle,
}

impl GemmKernel {
    /// CPU oracle. Also registers a kernel for [`Self::pipeline`] with
    /// `CpuDevice::register_kernel`.
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

    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }

    /// Computes `C = A @ B` for 2D row-major f32, where `a.shape = [m, k]`, `b.shape = [k, n]` and
    /// `c.shape = [m, n]`.
    ///
    /// Allocates `c` from `pool`. This is one dispatch, with no chaining.
    pub fn gemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(GpuError::Dispatch(format!(
                "gemm: only DType::F32 is supported, got {:?} and {:?}",
                a.dtype, b.dtype
            )));
        }
        let (&[m, k], &[k2, n]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(GpuError::Dispatch(format!(
                "gemm: expected 2D arrays, got shapes {:?} and {:?}",
                a.shape, b.shape
            )));
        };
        if k != k2 {
            return Err(GpuError::Dispatch(format!(
                "gemm: inner dimensions mismatch ({k} vs {k2}) for shapes {:?} and {:?}",
                a.shape, b.shape
            )));
        }

        let c = pool.alloc(vec![m, n], DType::F32)?;
        device.dispatch(
            self.pipeline,
            Bindings {
                buffers: &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                scalars: &[Scalar::U32(m), Scalar::U32(n), Scalar::U32(k)],
                textures: &[],
            },
            [n.div_ceil(16), m.div_ceil(16), 1],
        )?;
        Ok(c)
    }
}
