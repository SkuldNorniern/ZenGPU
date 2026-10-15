//! zengpu-compute and zengpu-blas on the HIP backend, checked against plain
//! Rust. Skips when ROCm or an AMD GPU is unavailable.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use zengpu_blas::Blas;
use zengpu_compute::elementwise::ElementwiseKernels;
use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{AdapterRequest, BackendPreference, DType, DeviceRequest, GpuDevice, GpuInstance};
use zengpu_hip::HipInstance;

fn hip_device() -> Option<Arc<dyn GpuDevice>> {
    let inst = HipInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

fn upload(pool: &BufferPool, shape: Vec<u32>, values: &[f32]) -> DeviceArray {
    let array = pool.alloc(shape, DType::F32).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
    array
}

fn download(device: &dyn GpuDevice, array: &DeviceArray) -> Vec<f32> {
    device
        .read_buffer(array.buffer, 0, array.size_bytes())
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * b.abs().max(1.0)
}

#[test]
fn reports_hip_backend() {
    let Some(device) = hip_device() else { return };
    assert_eq!(device.backend(), BackendPreference::Hip);
}

#[test]
fn elementwise_add_relu() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let kernels = ElementwiseKernels::new(&*device).unwrap();

    // Not a multiple of the 256-thread block, to exercise the bounds check.
    let n = 1000u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 - 500.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
    let da = upload(&pool, vec![n], &a);
    let db = upload(&pool, vec![n], &b);

    let sum = kernels.add(&*device, &pool, &da, &db).unwrap();
    let relu = kernels.relu(&*device, &pool, &sum).unwrap();
    let got = download(&*device, &relu);
    for i in 0..n as usize {
        assert!(close(got[i], (a[i] + b[i]).max(0.0)), "i={i}: {}", got[i]);
    }

    for array in [da, db, sum, relu] {
        pool.free(array);
    }
    kernels.destroy(&*device);
}

#[test]
fn blas_saxpy_sscal_sgemm() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();

    let n = 777u32;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 - i as f32).collect();
    let dx = upload(&pool, vec![n], &x);
    let dy = upload(&pool, vec![n], &y);
    blas.saxpy(&*device, &pool, 2.0, &dx, &dy).unwrap();
    blas.sscal(&*device, &pool, -0.5, &dy).unwrap();
    let got = download(&*device, &dy);
    for i in 0..n as usize {
        assert!(close(got[i], (y[i] + 2.0 * x[i]) * -0.5), "i={i}");
    }

    let (m, k, nn) = (67u32, 45u32, 53u32);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * nn).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let da = upload(&pool, vec![m, k], &a);
    let db = upload(&pool, vec![k, nn], &b);
    let dc = blas.sgemm(&*device, &pool, 1.5, &da, &db).unwrap();
    let c = download(&*device, &dc);
    for row in 0..m as usize {
        for col in 0..nn as usize {
            let want: f32 = 1.5
                * (0..k as usize)
                    .map(|i| a[row * k as usize + i] * b[i * nn as usize + col])
                    .sum::<f32>();
            let got = c[row * nn as usize + col];
            assert!(close(got, want), "C[{row}][{col}] = {got}, want {want}");
        }
    }

    for array in [dx, dy, da, db, dc] {
        pool.free(array);
    }
    blas.destroy(&*device);
}
