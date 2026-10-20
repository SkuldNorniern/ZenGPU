//! Metal batched ops run in submission order. Skips without Metal.

#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use zengpu_hal::{
    AdapterRequest, Bindings, BufferCopyOp, BufferDesc, BufferUsage, ComputeOp,
    ComputePipelineDesc, DeviceRequest, DispatchOp, GpuDevice, GpuInstance, MemoryUsage, Scalar,
    SubmissionStatus,
};
use zengpu_metal::MetalInstance;
use zengpu_spirv::{ZslShader, zsl};

const INC: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel inc(data: device mut buffer<u32>, p: P, id: global_id) {
        if id.x < p.n {
            data[id.x] = data[id.x] + 1
        }
    }
);

fn device() -> Option<Arc<dyn GpuDevice>> {
    let inst = MetalInstance::new();
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

/// Each dispatch reads the one before; the final copy sees them all.
#[test]
fn batched_ops_run_in_order() {
    let Some(dev) = device() else { return };
    let (shader, entry) = INC.create(&*dev).unwrap();
    let pipeline = dev
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let n = 1000u32;
    let desc = BufferDesc {
        size: u64::from(n) * 4,
        usage: BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::GpuOnly,
    };
    let (data, copy) = (
        dev.create_buffer(desc).unwrap(),
        dev.create_buffer(desc).unwrap(),
    );
    dev.write_buffer(data, 0, &vec![0; n as usize * 4]).unwrap();
    let scalars = [Scalar::U32(n)];
    let bindings = Bindings {
        buffers: &[data.index()],
        textures: &[],
        images: &[],
        scalars: &scalars,
    };
    let dispatch = DispatchOp {
        pipeline,
        bindings,
        grid: [n.div_ceil(64), 1, 1],
    };
    dev.dispatch_batch(&[dispatch; 100]).unwrap();
    let mut ops = vec![ComputeOp::Dispatch(dispatch); 50];
    ops.push(ComputeOp::CopyBuffer(BufferCopyOp {
        src: data,
        src_offset: 0,
        dst: copy,
        dst_offset: 0,
        len: desc.size,
    }));
    let other = dev.create_buffer(desc).unwrap();
    let sub = dev.submit_compute_ops(7, &ops).unwrap();
    assert_eq!(sub.cycle_id(), 7);
    dev.write_buffer(other, 0, &[1; 4]).unwrap();
    if dev.read_buffer(copy, 0, 4).is_ok() {
        assert_eq!(sub.poll().unwrap(), SubmissionStatus::Complete);
    }
    sub.wait(Duration::from_secs(10)).unwrap();
    let words = |b| {
        dev.read_buffer(b, 0, desc.size)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect::<Vec<_>>()
    };
    assert!(words(data).iter().all(|&v| v == 150));
    assert!(words(copy).iter().all(|&v| v == 150));
    dev.destroy_buffer(data);
    dev.destroy_buffer(copy);
    dev.destroy_buffer(other);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader(shader);
}
