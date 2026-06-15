//! Single-dispatch elementwise ops over [`DeviceArray`]s. No chaining, scheduling or fusion.
//!
//! f32 only; other dtypes fail with [`GpuError::Dispatch`].

use inline_spirv::inline_spirv;
use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderDesc, ShaderHandle,
};

use crate::{BufferPool, DeviceArray};

/// `out[i] = a[i] + b[i]` (matches `ZenGPU/examples/vec_add.rs`).
const ADD_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint a_idx;
        uint b_idx;
        uint out_idx;
        uint len;
    } pc;

    layout(local_size_x = 256) in;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.len) {
            g_bufs[pc.out_idx].data[i] =
                g_bufs[pc.a_idx].data[i] + g_bufs[pc.b_idx].data[i];
        }
    }
    "#,
    comp, vulkan1_2
);

const RELU_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint in_idx;
        uint out_idx;
        uint len;
    } pc;

    layout(local_size_x = 256) in;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.len) {
            g_bufs[pc.out_idx].data[i] = max(g_bufs[pc.in_idx].data[i], 0.0);
        }
    }
    "#,
    comp, vulkan1_2
);

fn spv_bytes(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn check_f32(dtype: DType, op: &str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(GpuError::Dispatch(format!(
            "{op}: only DType::F32 is supported in this version, got {dtype:?}"
        )));
    }
    Ok(())
}

/// Compiled pipelines for the elementwise ops. Create once per device.
pub struct ElementwiseKernels {
    add_shader: ShaderHandle,
    pub add_pipeline: PipelineHandle,
    relu_shader: ShaderHandle,
    pub relu_pipeline: PipelineHandle,
}

impl ElementwiseKernels {
    /// Create pipelines for [`Self::add`] and [`Self::relu`] on `device`. CPU device:
    /// also register kernels for [`Self::add_pipeline`] and [`Self::relu_pipeline`]
    /// with `CpuDevice::register_kernel`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let add_shader = device.create_shader(ShaderDesc { spirv: spv_bytes(ADD_SPV) })?;
        let add_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: add_shader,
            entry: "main",
        })?;
        let relu_shader = device.create_shader(ShaderDesc { spirv: spv_bytes(RELU_SPV) })?;
        let relu_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: relu_shader,
            entry: "main",
        })?;
        Ok(Self {
            add_shader,
            add_pipeline,
            relu_shader,
            relu_pipeline,
        })
    }

    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.add_pipeline);
        device.destroy_shader(self.add_shader);
        device.destroy_pipeline(self.relu_pipeline);
        device.destroy_shader(self.relu_shader);
    }

    /// `a` and `b` must share shape and be `DType::F32`. `out` is allocated from `pool`.
    pub fn add(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        check_f32(a.dtype, "elementwise::add")?;
        check_f32(b.dtype, "elementwise::add")?;
        if a.shape != b.shape {
            return Err(GpuError::Dispatch(format!(
                "elementwise::add: shape mismatch {:?} vs {:?}",
                a.shape, b.shape
            )));
        }

        let out = pool.alloc(a.shape.clone(), DType::F32)?;
        let n = a.len();
        device.dispatch(
            self.add_pipeline,
            Bindings {
                buffers: &[a.buffer.index(), b.buffer.index(), out.buffer.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
            },
            [n.div_ceil(256), 1, 1],
        )?;
        Ok(out)
    }

    /// `a` must be `DType::F32`. `out` is allocated from `pool`.
    pub fn relu(&self, device: &dyn GpuDevice, pool: &BufferPool, a: &DeviceArray) -> Result<DeviceArray> {
        check_f32(a.dtype, "elementwise::relu")?;

        let out = pool.alloc(a.shape.clone(), DType::F32)?;
        let n = a.len();
        device.dispatch(
            self.relu_pipeline,
            Bindings {
                buffers: &[a.buffer.index(), out.buffer.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
            },
            [n.div_ceil(256), 1, 1],
        )?;
        Ok(out)
    }
}
