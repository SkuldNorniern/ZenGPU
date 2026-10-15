//! zengpu-compute and zengpu-blas on the HIP backend, checked against plain
//! Rust. Skips when ROCm or an AMD GPU is unavailable.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

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

    // Larger than one 128×128 tile, no dimension a multiple of the tile or
    // of the 8-deep K slice.
    let (m, k, nn) = (301u32, 133u32, 259u32);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * nn)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let da = upload(&pool, vec![m, k], &a);
    let db = upload(&pool, vec![k, nn], &b);
    let dc = blas.sgemm(&*device, &pool, -2.0, &da, &db).unwrap();
    let c = download(&*device, &dc);
    for row in (0..m as usize).step_by(7) {
        for col in 0..nn as usize {
            let want: f32 = -2.0
                * (0..k as usize)
                    .map(|i| a[row * k as usize + i] * b[i * nn as usize + col])
                    .sum::<f32>();
            let got = c[row * nn as usize + col];
            assert!(close(got, want), "C[{row}][{col}] = {got}, want {want}");
        }
    }
    let last = c[(m * nn - 1) as usize];
    let want: f32 = -2.0
        * (0..k as usize)
            .map(|i| a[(m as usize - 1) * k as usize + i] * b[i * nn as usize + nn as usize - 1])
            .sum::<f32>();
    assert!(close(last, want), "corner {last} vs {want}");
    for array in [da, db, dc] {
        pool.free(array);
    }
    blas.destroy(&*device);
}

#[test]
fn blas_sgemm_throughput() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();
    for size in [512u32, 1024, 2048, 4096] {
        let a: Vec<f32> = (0..size * size).map(|i| (i % 17) as f32 * 0.01).collect();
        let da = upload(&pool, vec![size, size], &a);
        let db = upload(&pool, vec![size, size], &a);
        let warm = blas.sgemm(&*device, &pool, 1.0, &da, &db).unwrap();
        pool.free(warm);
        let reps = if size >= 2048 { 3 } else { 10 };
        let t = Instant::now();
        for _ in 0..reps {
            let c = blas.sgemm(&*device, &pool, 1.0, &da, &db).unwrap();
            pool.free(c);
        }
        let secs = t.elapsed().as_secs_f64() / reps as f64;
        let gflops = 2.0 * f64::from(size).powi(3) / secs / 1e9;
        println!("hip: blas sgemm {size}³  {gflops:.0} GFLOP/s");
        pool.free(da);
        pool.free(db);
    }
    blas.destroy(&*device);
}

#[test]
fn memory_bound_kernel_bandwidth() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let kernels = ElementwiseKernels::new(&*device).unwrap();
    let blas = Blas::new(&*device).unwrap();
    let n = 64u32 << 20; // 64 Mi floats = 256 MiB per array
    let a = pool.alloc(vec![n], DType::F32).unwrap();
    let b = pool.alloc(vec![n], DType::F32).unwrap();
    let gb = |arrays: f64, secs: f64| arrays * f64::from(n) * 4.0 / secs / 1e9;

    let out = kernels.add(&*device, &pool, &a, &b).unwrap();
    pool.free(out);
    let t = Instant::now();
    for _ in 0..5 {
        let out = kernels.add(&*device, &pool, &a, &b).unwrap();
        pool.free(out);
    }
    println!(
        "hip: elementwise add  {:.0} GB/s",
        gb(3.0, t.elapsed().as_secs_f64() / 5.0)
    );

    blas.saxpy(&*device, &pool, 2.0, &a, &b).unwrap();
    let t = Instant::now();
    for _ in 0..5 {
        blas.saxpy(&*device, &pool, 2.0, &a, &b).unwrap();
    }
    println!(
        "hip: saxpy            {:.0} GB/s",
        gb(3.0, t.elapsed().as_secs_f64() / 5.0)
    );

    pool.free(a);
    pool.free(b);
    kernels.destroy(&*device);
    blas.destroy(&*device);
}
