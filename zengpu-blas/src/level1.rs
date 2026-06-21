//! BLAS Level 1 — vector operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderDesc, ShaderHandle,
};
use zengpu_spirv::zengpu_spirv;

/// `y[i] += alpha * x[i]`  (BLAS SAXPY)
const AXPY_SPV: &[u32] = zengpu_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint  x_idx;
        uint  y_idx;
        uint  n;
        float alpha;
    } pc;

    layout(local_size_x = 256) in;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.n) {
            g_bufs[pc.y_idx].data[i] +=
                pc.alpha * g_bufs[pc.x_idx].data[i];
        }
    }
    "#,
    comp,
    vulkan1_2
);

/// `x[i] *= alpha`  (BLAS SSCAL)
const SCAL_SPV: &[u32] = zengpu_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint  x_idx;
        uint  n;
        float alpha;
    } pc;

    layout(local_size_x = 256) in;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.n) {
            g_bufs[pc.x_idx].data[i] *= pc.alpha;
        }
    }
    "#,
    comp,
    vulkan1_2
);

fn spv_bytes(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn check_f32_shape(a: &DeviceArray, b: &DeviceArray, op: &str) -> Result<u32> {
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(GpuError::Dispatch(format!(
            "{op}: only f32 supported, got {:?} and {:?}",
            a.dtype, b.dtype
        )));
    }
    if a.shape != b.shape {
        return Err(GpuError::Dispatch(format!(
            "{op}: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    Ok(a.len())
}

/// Compiled pipelines for BLAS Level-1 operations. Create once per device.
pub struct Level1Kernels {
    axpy_shader:    ShaderHandle,
    pub axpy_pipeline: PipelineHandle,
    scal_shader:    ShaderHandle,
    pub scal_pipeline: PipelineHandle,
}

impl Level1Kernels {
    /// Compile SAXPY and SSCAL pipelines on `device`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let axpy_shader = device.create_shader(ShaderDesc {
            spirv: spv_bytes(AXPY_SPV),
        })?;
        let axpy_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: axpy_shader,
            entry:  "main",
        })?;
        let scal_shader = device.create_shader(ShaderDesc {
            spirv: spv_bytes(SCAL_SPV),
        })?;
        let scal_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: scal_shader,
            entry:  "main",
        })?;
        Ok(Self {
            axpy_shader,
            axpy_pipeline,
            scal_shader,
            scal_pipeline,
        })
    }

    /// Destroy the pipelines and shaders created by [`Self::new`].
    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.axpy_pipeline);
        device.destroy_shader(self.axpy_shader);
        device.destroy_pipeline(self.scal_pipeline);
        device.destroy_shader(self.scal_shader);
    }

    /// `y[i] += alpha * x[i]`  (BLAS SAXPY).
    ///
    /// `x` and `y` must have the same shape and be `f32`. `y` is updated
    /// in place; pool is not used (no allocation).
    pub fn saxpy(
        &self,
        device: &dyn GpuDevice,
        _pool: &BufferPool,
        alpha: f32,
        x: &DeviceArray,
        y: &DeviceArray,
    ) -> Result<()> {
        let n = check_f32_shape(x, y, "saxpy")?;
        device.dispatch(
            self.axpy_pipeline,
            Bindings {
                buffers:  &[x.buffer.index(), y.buffer.index()],
                scalars:  &[Scalar::U32(n), Scalar::F32(alpha)],
                textures: &[],
            },
            [n.div_ceil(256), 1, 1],
        )
    }

    /// `x[i] *= alpha`  (BLAS SSCAL).
    ///
    /// `x` is updated in place; pool is not used (no allocation).
    pub fn sscal(
        &self,
        device: &dyn GpuDevice,
        _pool: &BufferPool,
        alpha: f32,
        x: &DeviceArray,
    ) -> Result<()> {
        if x.dtype != DType::F32 {
            return Err(GpuError::Dispatch(format!(
                "sscal: only f32 supported, got {:?}",
                x.dtype
            )));
        }
        let n = x.len();
        device.dispatch(
            self.scal_pipeline,
            Bindings {
                buffers:  &[x.buffer.index()],
                scalars:  &[Scalar::U32(n), Scalar::F32(alpha)],
                textures: &[],
            },
            [n.div_ceil(256), 1, 1],
        )
    }
}
