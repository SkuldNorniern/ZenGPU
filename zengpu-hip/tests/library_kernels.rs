//! Checks zengpu-compute library kernels on HIP against plain Rust.
//!
//! Skips without ROCm or an AMD GPU.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * b.abs().max(1.0)
}

/// Uploads f16 or bf16 bit patterns as little-endian bytes.
fn upload_half(pool: &BufferPool, shape: Vec<u32>, dtype: DType, bits: &[u16]) -> DeviceArray {
    let array = pool.alloc(shape, dtype).unwrap();
    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
    array
}

/// Converts f32 to bf16, rounding to nearest even on the top 16 bits.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Converts f32 to binary16 with nearest even rounding. Subnormals are kept and overflow goes
/// to inf.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        let payload = if man != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | payload;
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
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
    let h = if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        h + 1
    } else {
        h
    };
    sign | h as u16
}

/// Converts binary16 to f32 exactly, subnormals included.
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
fn memory_bound_kernel_bandwidth() {
    let Some(device) = hip_device() else { return };
    let pool = BufferPool::new(device.clone());
    let kernels = ElementwiseKernels::new(&*device).unwrap();
    let n = 64u32 << 20;
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

    pool.free(a);
    pool.free(b);
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
    assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
    assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);

    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(-2.0)), -2.0);
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
    let n = 1001u32;
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
            .as_chunks::<2>()
            .0
            .iter()
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

const NIBBLES_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel nibbles(packed: device buffer<u32>, out: device mut buffer<u32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            let w = packed[i >> 3]
            out[i] = ((w >> ((i & 7) << 2)) & 15) | ((i ^ 1) << 8)
        }
    }
);

#[test]
fn zsl_bitwise_ops_unpack_nibbles() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = NIBBLES_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 1000u32;
    let words: Vec<u32> = (0..n.div_ceil(8))
        .map(|i| i.wrapping_mul(0x9e37_79b9))
        .collect();
    let packed = pool.alloc(vec![words.len() as u32], DType::U32).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device.write_buffer(packed.buffer, 0, &bytes).unwrap();
    let out = pool.alloc(vec![n], DType::U32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[packed.buffer.index(), out.buffer.index()],
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let got: Vec<u32> = device
        .read_buffer(out.buffer, 0, out.size_bytes())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    for i in 0..n {
        let want = ((words[(i >> 3) as usize] >> ((i & 7) << 2)) & 15) | ((i ^ 1) << 8);
        assert_eq!(got[i as usize], want, "at {i}");
    }
    drop(ops);
    pool.free(packed);
    pool.free(out);
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}
