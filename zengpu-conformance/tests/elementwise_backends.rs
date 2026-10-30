//! Cross-backend array tests live here to keep published crate dependencies acyclic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};
use zengpu_hip::HipInstance;
use zengpu_metal::MetalInstance;

use zengpu_compute::{BufferPool, DeviceArray, ElementwiseKernels};
use zengpu_hal::{DType, GpuDevice};

fn hip_device() -> Option<Arc<dyn GpuDevice>> {
    let inst = HipInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

fn metal_device() -> Option<Arc<dyn GpuDevice>> {
    let inst = MetalInstance::new();
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
    add_then_relu_matches_cpu(device);
}

#[test]
fn metal_add_then_relu_matches_cpu() {
    let Some(device) = metal_device() else { return };
    add_then_relu_matches_cpu(device);
}

fn add_then_relu_matches_cpu(device: Arc<dyn GpuDevice>) {
    let pool = BufferPool::new(device.clone());
    let kernels = ElementwiseKernels::new(&*device).unwrap();

    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 - 500.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
    let da = upload(&pool, &a);
    let db = upload(&pool, &b);

    let sum = kernels.add(&*device, &pool, &da, &db).unwrap();
    let relu = kernels.relu(&*device, &pool, &sum).unwrap();
    let raw = device
        .read_buffer(relu.buffer, 0, relu.size_bytes())
        .unwrap();
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
