//! Single-dispatch elementwise ops over [`DeviceArray`]s.
//!
//! Each op below is exactly one GPU dispatch — no chaining, scheduling, or
//! fusion; graph-level optimization belongs to the calling compiler. f32 only
//! for now; other dtypes are
//! rejected with [`GpuError::Dispatch`].

use zengpu_hal::{
    Bindings, ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar,
    ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::{BufferPool, DeviceArray};

/// `out[i] = a[i] + b[i]` (matches `ZenGPU/examples/vec_add.rs`).
const ADD_ZSL: ZslShader = zsl!(
    push P { len: u32 }
    @workgroup_size(256)
    kernel add(a: device buffer<f32>, b: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            out[i] = a[i] + b[i]
        }
    }
);

/// `out[i] = max(a[i], 0)`.
const RELU_ZSL: ZslShader = zsl!(
    push P { len: u32 }
    @workgroup_size(256)
    kernel relu(inp: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            out[i] = max(inp[i], 0.0)
        }
    }
);

fn check_f32(dtype: DType, op: &str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(GpuError::Dispatch(format!(
            "{op}: only DType::F32 is supported in this version, got {dtype:?}"
        )));
    }
    Ok(())
}

/// Compiled pipelines for the elementwise ops in this module. Create once per
/// device, reuse across dispatches.
pub struct ElementwiseKernels {
    add_shader: ShaderHandle,
    pub add_pipeline: PipelineHandle,
    relu_shader: ShaderHandle,
    pub relu_pipeline: PipelineHandle,
}

impl ElementwiseKernels {
    /// Compile and create pipelines for [`Self::add`] and [`Self::relu`] on
    /// `device`. For the CPU oracle, the caller must additionally register
    /// matching kernels for [`Self::add_pipeline`]/[`Self::relu_pipeline`]
    /// via `CpuDevice::register_kernel`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        let add_shader = device.create_shader(ADD_ZSL.spirv_desc())?;
        let add_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: add_shader,
            entry: "main",
            block: [256, 1, 1],
        })?;
        let relu_shader = device.create_shader(RELU_ZSL.spirv_desc())?;
        let relu_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: relu_shader,
            entry: "main",
            block: [256, 1, 1],
        })?;
        Ok(Self {
            add_shader,
            add_pipeline,
            relu_shader,
            relu_pipeline,
        })
    }

    /// Destroy the pipelines and shaders created by [`Self::new`].
    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.add_pipeline);
        device.destroy_shader(self.add_shader);
        device.destroy_pipeline(self.relu_pipeline);
        device.destroy_shader(self.relu_shader);
    }

    /// `out[i] = a[i] + b[i]`, element-by-element. `a` and `b` must have the
    /// same shape and be `DType::F32`. Allocates `out` from `pool`.
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

    /// `out[i] = max(a[i], 0)`. `a` must be `DType::F32`. Allocates `out`
    /// from `pool`.
    pub fn relu(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
    ) -> Result<DeviceArray> {
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
