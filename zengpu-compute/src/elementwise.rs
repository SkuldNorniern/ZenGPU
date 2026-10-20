//! Single-dispatch elementwise ops over [`DeviceArray`]s. No chaining, scheduling or fusion.
//!
//! f32 only; other dtypes fail with [`GpuError::Dispatch`].

use zengpu_hal::{
    ComputePipelineDesc, DType, GpuDevice, GpuError, PipelineHandle, Result, Scalar, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::{BufferPool, DeviceArray, OpList};

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
        let (add_shader, add_entry) = ADD_ZSL.create(device)?;
        let add_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: add_shader,
            entry: add_entry,
            block: [256, 1, 1],
        })?;
        let (relu_shader, relu_entry) = RELU_ZSL.create(device)?;
        let relu_pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader: relu_shader,
            entry: relu_entry,
            block: [256, 1, 1],
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
        let mut ops = OpList::new(pool);
        let out = self.record_add(&mut ops, a, b)?;
        finish(pool, out, ops.run(device))
    }

    /// Record into `ops`. `out` is allocated now, written when the list runs.
    pub fn record_add(
        &self,
        ops: &mut OpList<'_>,
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
        let n = a.validate_contiguous(ops.pool().device())?;
        b.validate_contiguous(ops.pool().device())?;
        let out = ops.pool().alloc(a.shape.clone(), DType::F32)?;
        ops.require_buffer(a.buffer, a.size_bytes());
        ops.require_buffer(b.buffer, b.size_bytes());
        ops.require_buffer(out.buffer, out.size_bytes());
        ops.dispatch(
            self.add_pipeline,
            &[a.buffer.index(), b.buffer.index(), out.buffer.index()],
            &[Scalar::U32(n)],
            [n.div_ceil(256), 1, 1],
        );
        Ok(out)
    }

    /// `a` must be `DType::F32`. `out` is allocated from `pool`.
    pub fn relu(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
    ) -> Result<DeviceArray> {
        let mut ops = OpList::new(pool);
        let out = self.record_relu(&mut ops, a)?;
        finish(pool, out, ops.run(device))
    }

    /// Record into `ops`. `out` is allocated now, written when the list runs.
    pub fn record_relu(&self, ops: &mut OpList<'_>, a: &DeviceArray) -> Result<DeviceArray> {
        check_f32(a.dtype, "elementwise::relu")?;
        let n = a.validate_contiguous(ops.pool().device())?;
        let out = ops.pool().alloc(a.shape.clone(), DType::F32)?;
        ops.require_buffer(a.buffer, a.size_bytes());
        ops.require_buffer(out.buffer, out.size_bytes());
        ops.dispatch(
            self.relu_pipeline,
            &[a.buffer.index(), out.buffer.index()],
            &[Scalar::U32(n)],
            [n.div_ceil(256), 1, 1],
        );
        Ok(out)
    }
}

/// `out` is kept only if its dispatch went through. Otherwise it returns to `pool`.
fn finish(pool: &BufferPool, out: DeviceArray, dispatched: Result<()>) -> Result<DeviceArray> {
    match dispatched {
        Ok(()) => Ok(out),
        Err(e) => {
            pool.free(out);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};
    use zengpu_hip::HipInstance;

    use super::*;

    fn hip_device() -> Option<Arc<dyn GpuDevice>> {
        let inst = HipInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
    }

    fn upload(pool: &BufferPool, values: &[f32]) -> DeviceArray {
        let array = pool.alloc(vec![values.len() as u32], DType::F32).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
        array
    }

    #[test]
    fn hip_add_then_relu_matches_cpu() {
        let Some(device) = hip_device() else { return };
        let pool = BufferPool::new(device.clone());
        let kernels = ElementwiseKernels::new(&*device).unwrap();

        let n = 1000;
        let a: Vec<f32> = (0..n).map(|i| i as f32 - 500.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
        let da = upload(&pool, &a);
        let db = upload(&pool, &b);

        let sum = kernels.add(&*device, &pool, &da, &db).unwrap();
        let relu = kernels.relu(&*device, &pool, &sum).unwrap();
        let raw = device.read_buffer(relu.buffer, 0, relu.size_bytes()).unwrap();
        let got: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        for i in 0..n {
            assert_eq!(got[i], (a[i] + b[i]).max(0.0), "at {i}");
        }

        for array in [da, db, sum, relu] {
            pool.free(array);
        }
        kernels.destroy(&*device);
    }
}
