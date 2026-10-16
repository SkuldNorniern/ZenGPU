//! zengpu-compute and zengpu-blas on the HIP backend, checked against plain
//! Rust. Skips when ROCm or an AMD GPU is unavailable.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use zengpu_blas::Blas;
use zengpu_compute::elementwise::ElementwiseKernels;
use zengpu_compute::{BufferPool, DeviceArray, OpList};
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

/// Uploads little-endian 16-bit elements (F16 or Bf16 bit patterns).
fn upload_half(pool: &BufferPool, shape: Vec<u32>, dtype: DType, bits: &[u16]) -> DeviceArray {
    let array = pool.alloc(shape, dtype).unwrap();
    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
    array
}

/// Round-to-nearest-even of the top 16 bits of an f32.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// IEEE binary16 encode with round-to-nearest-even, including subnormals.
/// Overflow saturates to infinity.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // Inf or NaN; the finite test inputs never reach this.
        let payload = if man != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | payload;
    }
    // Biased half exponent: (exp - 127) + 15.
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: the result is the 24-bit significand shifted right by
        // (14 - e), rounded to nearest even.
        let full = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let q = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let q = if rem > half || (rem == half && (q & 1) == 1) {
            q + 1
        } else {
            q
        };
        return sign | q as u16;
    }
    let m = man >> 13;
    let rem = man & 0x1fff;
    let h = ((e as u32) << 10) | m;
    // A carry out of the mantissa bumps the exponent, and at the top it
    // lands exactly on infinity.
    let h = if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        h + 1
    } else {
        h
    };
    sign | h as u16
}

/// Exact binary16 decode, including subnormals.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 if man == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Best per-call time of three rounds of ten `sgemm` calls, after one warm-up.
fn best_sgemm_time(
    device: &dyn GpuDevice,
    pool: &BufferPool,
    blas: &Blas,
    a: &DeviceArray,
    b: &DeviceArray,
) -> Duration {
    let warm = blas.sgemm(device, pool, 1.0, a, b).unwrap();
    pool.free(warm);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..10 {
                let c = blas.sgemm(device, pool, 1.0, a, b).unwrap();
                pool.free(c);
            }
            t.elapsed() / 10
        })
        .min()
        .unwrap()
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
fn blas_sgemm_every_plan() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();

    // Each shape lands on a different branch of the planner: skinny rows
    // 1/4/8/16/32, the 64 tile with and without split-K, and the 128 tile
    // with and without split-K.
    let cases: [(u32, u32, u32, f32); 14] = [
        (1, 4096, 4096, 1.0),
        (3, 1001, 257, -0.5),
        (5, 64, 40, 2.0),
        (13, 4096, 1100, 1.0),
        (31, 777, 300, 1.5),
        (32, 128, 2048, 1.0),
        (33, 200, 129, 1.0),
        (64, 4096, 512, 1.0),
        (100, 100, 64, 3.0),
        (128, 4096, 1024, 1.0),
        (300, 300, 1000, -1.0),
        (384, 384, 384, 1.0),
        (512, 512, 517, 1.0),
        (777, 300, 1000, 0.25),
    ];
    for (m, n, k, alpha) in cases {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let da = upload(&pool, vec![m, k], &a);
        let db = upload(&pool, vec![k, n], &b);
        let dc = blas.sgemm(&*device, &pool, alpha, &da, &db).unwrap();
        let c = download(&*device, &dc);

        let row_step = if mu <= 64 { 1 } else { 7 };
        let col_step = if nu <= 1024 { 1 } else { 13 };
        for row in (0..mu).step_by(row_step) {
            for col in (0..nu).step_by(col_step) {
                let sum: f64 = (0..ku)
                    .map(|i| f64::from(a[row * ku + i]) * f64::from(b[i * nu + col]))
                    .sum();
                let want = alpha * sum as f32;
                let got = c[row * nu + col];
                assert!(
                    close(got, want),
                    "m={m} n={n} k={k} row={row} col={col} got={got} want={want}"
                );
            }
        }

        for array in [da, db, dc] {
            pool.free(array);
        }
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

/// Width of the decode-step chain: x is [1, K], W1 and W2 are [K, K].
const DECODE_K: u32 = 4096;

fn decode_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..DECODE_K)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let w1: Vec<f32> = (0..DECODE_K * DECODE_K)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
        .collect();
    let w2: Vec<f32> = (0..DECODE_K * DECODE_K)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.001)
        .collect();
    (x, w1, w2)
}

/// z = 2.5 * (relu(x @ W1) @ W2), computed in f64.
fn decode_reference(x: &[f32], w1: &[f32], w2: &[f32]) -> Vec<f64> {
    let n = DECODE_K as usize;
    let h: Vec<f64> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| f64::from(x[i]) * f64::from(w1[i * n + j]))
                .sum()
        })
        .collect();
    let r: Vec<f64> = h.iter().map(|v| v.max(0.0)).collect();
    (0..n)
        .map(|c| {
            let y: f64 = (0..n).map(|k| r[k] * f64::from(w2[k * n + c])).sum();
            2.5 * y
        })
        .collect()
}

#[test]
fn op_list_chains_and_replays() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();
    let kernels = ElementwiseKernels::new(&*device).unwrap();

    let (x, w1, w2) = decode_inputs();
    let dx = upload(&pool, vec![1, DECODE_K], &x);
    let dw1 = upload(&pool, vec![DECODE_K, DECODE_K], &w1);
    let dw2 = upload(&pool, vec![DECODE_K, DECODE_K], &w2);

    let mut ops = OpList::new(&pool);
    let h = blas.gemm.record_sgemm(&mut ops, 1.0, &dx, &dw1).unwrap();
    let r = kernels.record_relu(&mut ops, &h).unwrap();
    let y = blas.gemm.record_sgemm(&mut ops, 1.0, &r, &dw2).unwrap();
    let z = kernels.record_add(&mut ops, &y, &y).unwrap();
    blas.level1.record_saxpy(&mut ops, 0.5, &y, &z).unwrap();

    ops.run(&*device).unwrap();
    let first = download(&*device, &z);
    let want = decode_reference(&x, &w1, &w2);
    assert_eq!(first.len(), want.len());
    for col in (0..first.len()).step_by(31) {
        assert!(
            close(first[col], want[col] as f32),
            "z[{col}] = {}, want {}",
            first[col],
            want[col]
        );
    }

    // Replay: add rewrites z from y and saxpy adds onto it, so every run
    // must reproduce the first result exactly.
    for _ in 0..2 {
        ops.run(&*device).unwrap();
    }
    let replay = download(&*device, &z);
    assert_eq!(replay, first);

    ops.clear();
    for array in [dx, dw1, dw2, h, r, y, z] {
        pool.free(array);
    }
    blas.destroy(&*device);
    kernels.destroy(&*device);
}

#[test]
fn op_list_beats_per_op_waits() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();
    let kernels = ElementwiseKernels::new(&*device).unwrap();

    let (x, w1, w2) = decode_inputs();
    let dx = upload(&pool, vec![1, DECODE_K], &x);
    let dw1 = upload(&pool, vec![DECODE_K, DECODE_K], &w1);
    let dw2 = upload(&pool, vec![DECODE_K, DECODE_K], &w2);
    const ITERS: u32 = 20;

    // One-shot calls: every op waits for the GPU before the next is issued.
    let t = Instant::now();
    let mut per_op_z: Option<DeviceArray> = None;
    for _ in 0..ITERS {
        let h = blas.sgemm(&*device, &pool, 1.0, &dx, &dw1).unwrap();
        let r = kernels.relu(&*device, &pool, &h).unwrap();
        pool.free(h);
        let y = blas.sgemm(&*device, &pool, 1.0, &r, &dw2).unwrap();
        pool.free(r);
        let z = kernels.add(&*device, &pool, &y, &y).unwrap();
        blas.saxpy(&*device, &pool, 0.5, &y, &z).unwrap();
        pool.free(y);
        if let Some(old) = per_op_z.replace(z) {
            pool.free(old);
        }
    }
    let per_op = t.elapsed() / ITERS;
    let per_op_z = per_op_z.unwrap();

    // The same chain recorded once and replayed: one GPU wait per run.
    let mut ops = OpList::new(&pool);
    let h = blas.gemm.record_sgemm(&mut ops, 1.0, &dx, &dw1).unwrap();
    let r = kernels.record_relu(&mut ops, &h).unwrap();
    let y = blas.gemm.record_sgemm(&mut ops, 1.0, &r, &dw2).unwrap();
    let z = kernels.record_add(&mut ops, &y, &y).unwrap();
    blas.level1.record_saxpy(&mut ops, 0.5, &y, &z).unwrap();

    let t = Instant::now();
    for _ in 0..ITERS {
        ops.run(&*device).unwrap();
    }
    let op_list = t.elapsed() / ITERS;
    println!("hip: decode chain  per-op {per_op:?}  op-list {op_list:?}");

    let got_per_op = download(&*device, &per_op_z);
    let got_list = download(&*device, &z);
    for (col, (a, b)) in got_per_op.iter().zip(&got_list).enumerate() {
        assert!(close(*a, *b), "col {col}: per-op {a} vs op-list {b}");
    }

    ops.clear();
    for array in [dx, dw1, dw2, h, r, y, z, per_op_z] {
        pool.free(array);
    }
    blas.destroy(&*device);
    kernels.destroy(&*device);
}

#[test]
fn half_conversions_round_trip() {
    let cases: [(f32, u16); 6] = [
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (0.5, 0x3800),
        (65504.0, 0x7bff),
        (6.1035156e-5, 0x0400),
        (5.9604645e-8, 0x0001),
    ];
    for (x, bits) in cases {
        assert_eq!(f32_to_f16_bits(x), bits, "encode {x}");
        assert_eq!(f16_bits_to_f32(bits), x, "decode {bits:#06x}");
    }
    // Ties go to the even mantissa: 1 + 2^-11 is halfway between 1.0 and
    // 1 + 2^-10, so it rounds down; 1 + 3 * 2^-11 rounds up to 1 + 2 * 2^-10.
    assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
    assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);

    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(-2.0)), -2.0);
}

#[test]
fn blas_sgemm_half_weights() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();

    let cases: [(u32, u32, u32, f32); 7] = [
        (1, 4096, 4096, 1.0),
        (3, 1001, 257, -0.5),
        (16, 4096, 1100, 1.0),
        (33, 200, 129, 1.0),
        (128, 4096, 1024, 1.0),
        (300, 300, 1000, -1.0),
        (512, 512, 517, 1.0),
    ];
    type HalfFormat = (DType, fn(f32) -> u16, fn(u16) -> f32);
    let formats: [HalfFormat; 2] = [
        (DType::F16, f32_to_f16_bits, f16_bits_to_f32),
        (DType::Bf16, f32_to_bf16_bits, bf16_bits_to_f32),
    ];
    for (dtype, encode, decode) in formats {
        for (m, n, k, alpha) in cases {
            let (mu, nu, ku) = (m as usize, n as usize, k as usize);
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();
            // The reference uses the decoded half values, so only the
            // accumulation error is left to compare.
            let bits: Vec<u16> = b.iter().map(|&v| encode(v)).collect();
            let decoded: Vec<f32> = bits.iter().map(|&h| decode(h)).collect();
            let da = upload(&pool, vec![m, k], &a);
            let db = upload_half(&pool, vec![k, n], dtype, &bits);
            let dc = blas.sgemm(&*device, &pool, alpha, &da, &db).unwrap();
            let c = download(&*device, &dc);

            let row_step = if mu <= 64 { 1 } else { 7 };
            let col_step = if nu <= 1024 { 1 } else { 13 };
            for row in (0..mu).step_by(row_step) {
                for col in (0..nu).step_by(col_step) {
                    let sum: f64 = (0..ku)
                        .map(|i| f64::from(a[row * ku + i]) * f64::from(decoded[i * nu + col]))
                        .sum();
                    let want = (f64::from(alpha) * sum) as f32;
                    let got = c[row * nu + col];
                    assert!(
                        close(got, want),
                        "{dtype:?} m={m} n={n} k={k} row={row} col={col} got={got} want={want}"
                    );
                }
            }

            for array in [da, db, dc] {
                pool.free(array);
            }
        }
    }
    blas.destroy(&*device);
}

#[test]
fn blas_sgemm_half_weights_speed() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let blas = Blas::new(&*device).unwrap();

    for (m, n, k) in [
        (1u32, 4096u32, 4096u32),
        (16, 4096, 4096),
        (128, 4096, 4096),
        (2048, 2048, 2048),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let bits_f16: Vec<u16> = b.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let bits_bf16: Vec<u16> = b.iter().map(|&v| f32_to_bf16_bits(v)).collect();

        let da = upload(&pool, vec![m, k], &a);
        let db_f32 = upload(&pool, vec![k, n], &b);
        let db_f16 = upload_half(&pool, vec![k, n], DType::F16, &bits_f16);
        let db_bf16 = upload_half(&pool, vec![k, n], DType::Bf16, &bits_bf16);

        let t_f32 = best_sgemm_time(&*device, &pool, &blas, &da, &db_f32);
        let t_f16 = best_sgemm_time(&*device, &pool, &blas, &da, &db_f16);
        let t_bf16 = best_sgemm_time(&*device, &pool, &blas, &da, &db_bf16);
        println!(
            "hip: half-weight sgemm M={m} N={n} K={k}  f32 {t_f32:?}  f16 {t_f16:?}  bf16 {t_bf16:?}"
        );

        for array in [da, db_f32, db_f16, db_bf16] {
            pool.free(array);
        }
    }
    blas.destroy(&*device);
}

const HALF_IO_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel half_io(xh: device buffer<f16>, xb: device buffer<bf16>, yh: device mut buffer<f16>, yb: device mut buffer<bf16>, sum: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            yh[i] = xh[i] * 2.0
            yb[i] = xb[i] * 2.0
            sum[i] = xh[i] + xb[i]
        }
    }
);

#[test]
fn zsl_half_buffers_load_and_store() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let (shader, entry) = HALF_IO_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    // Odd length: last element sits alone in its 32-bit word.
    let n = 1001u32;
    // Exact in both formats (bf16 keeps 8 significant bits).
    let x: Vec<f32> = (0..n).map(|i| ((i % 256) as f32 - 128.0) * 0.25).collect();
    let xh_bits: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let xb_bits: Vec<u16> = x.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let xh = upload_half(&pool, vec![n], DType::F16, &xh_bits);
    let xb = upload_half(&pool, vec![n], DType::Bf16, &xb_bits);
    let yh = pool.alloc(vec![n], DType::F16).unwrap();
    let yb = pool.alloc(vec![n], DType::Bf16).unwrap();
    let sum = pool.alloc(vec![n], DType::F32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            xh.buffer.index(),
            xb.buffer.index(),
            yh.buffer.index(),
            yb.buffer.index(),
            sum.buffer.index(),
        ],
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let halves = |a: &DeviceArray| -> Vec<u16> {
        device
            .read_buffer(a.buffer, 0, a.size_bytes())
            .unwrap()
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let (out_h, out_b, out_sum) = (halves(&yh), halves(&yb), download(&*device, &sum));
    for i in 0..n as usize {
        assert_eq!(f16_bits_to_f32(out_h[i]), 2.0 * x[i], "f16 at {i}");
        assert_eq!(bf16_bits_to_f32(out_b[i]), 2.0 * x[i], "bf16 at {i}");
        assert_eq!(out_sum[i], 2.0 * x[i], "sum at {i}");
    }
    drop(ops);
    for a in [xh, xb, yh, yb, sum] {
        pool.free(a);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}
