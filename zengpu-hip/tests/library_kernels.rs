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
    kernel half_io(xh: device buffer<f16>, xb: device buffer<bf16>, yh: device mut buffer<f16>, yb: device mut buffer<bf16>, sum: device mut buffer<f32>, xf: device buffer<f32>, zh: device mut buffer<f16>, zb: device mut buffer<bf16>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            yh[i] = xh[i] * 2.0
            yb[i] = xb[i] * 2.0
            sum[i] = xh[i] + xb[i]
            zh[i] = xf[i]
            zb[i] = xf[i]
        }
    }
);

/// Rounds `x` to nearest even at `man` fraction bits. Assumes exp >= `min_exp` and a finite result
/// <= `max`. `x` must not be NaN.
fn round_ref(x: f32, man: i32, min_exp: i32, max: f64) -> f64 {
    let v = f64::from(x);
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let e = (v.abs().log2().floor() as i32).max(min_exp);
    let ulp = 2f64.powi(e - man);
    let q = (v / ulp).round_ties_even() * ulp;
    if q.abs() > max {
        f64::INFINITY.copysign(v)
    } else {
        q
    }
}

fn store_inputs(n: u32) -> Vec<f32> {
    let edge = [
        65504.0,
        65519.0,
        65520.0,
        1.0e30,
        -1.0e30,
        2f32.powi(-24),
        2f32.powi(-25),
        3.0 * 2f32.powi(-25),
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2f32.powi(-14) * (1.0 - 2f32.powi(-11)),
        3.0e38,
        f32::MAX,
        1.0e-40,
        -2.0 * 2f32.powi(-25),
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc0_0000),
    ];
    (0..n)
        .map(|i| {
            if let Some(&e) = edge.get(i as usize) {
                return e;
            }
            let v = if i % 2 == 0 {
                (1.0 + ((i % 1024) as f32 + 0.5) * 2f32.powi(-10)) * 2f32.powi((i % 40) as i32 - 20)
            } else {
                (1.0 + ((i % 128) as f32 + 0.5) * 2f32.powi(-7)) * 2f32.powi((i % 40) as i32 - 20)
            };
            if i % 3 == 0 { -v } else { v }
        })
        .collect()
}

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
    let store_in = store_inputs(n);
    let xf = upload(&pool, vec![n], &store_in);
    let zh = pool.alloc(vec![n], DType::F16).unwrap();
    let zb = pool.alloc(vec![n], DType::Bf16).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            xh.buffer.index(),
            xb.buffer.index(),
            yh.buffer.index(),
            yb.buffer.index(),
            sum.buffer.index(),
            xf.buffer.index(),
            zh.buffer.index(),
            zb.buffer.index(),
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
    let (store_h, store_b) = (halves(&zh), halves(&zb));
    for i in 0..n as usize {
        assert_eq!(f16_bits_to_f32(out_h[i]), 2.0 * x[i], "f16 at {i}");
        assert_eq!(bf16_bits_to_f32(out_b[i]), 2.0 * x[i], "bf16 at {i}");
        assert_eq!(out_sum[i], 2.0 * x[i], "sum at {i}");
        let v = store_in[i];
        let got_h = f16_bits_to_f32(store_h[i]);
        let got_b = bf16_bits_to_f32(store_b[i]);
        if v.is_nan() {
            assert!(got_h.is_nan(), "f16 store of NaN at {i}");
            assert!(got_b.is_nan(), "bf16 store of NaN at {i}");
            continue;
        }
        let want_h = round_ref(v, 10, -14, 65504.0);
        let want_b = round_ref(v, 7, -126, f64::from(f32::from_bits(0x7f7f_0000)));
        assert_eq!(
            f64::from(got_h).to_bits(),
            want_h.to_bits(),
            "f16 store of {v:e} at {i}"
        );
        assert_eq!(
            f64::from(got_b).to_bits(),
            want_b.to_bits(),
            "bf16 store of {v:e} at {i}"
        );
    }
    drop(ops);
    for a in [xh, xb, yh, yb, sum, xf, zh, zb] {
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

const TRIG_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { len: u32 }
    @workgroup_size(64)
    kernel trig(src: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            let x = src[i]
            out[i] = sin(x) + cos(x) + tan(x)
        }
    }
);

#[test]
fn zsl_trigonometry_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = TRIG_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 128u32;
    let input: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 / n as f32).collect();
    let src = upload(&pool, vec![n], &input);
    let out = pool.alloc(vec![n], DType::F32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[src.buffer.index(), out.buffer.index()],
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    for (i, x) in input.iter().enumerate() {
        let want = x.sin() + x.cos() + x.tan();
        assert!(
            (got[i] - want).abs() < 2e-5,
            "out[{i}]: got {} want {want}",
            got[i]
        );
    }
    drop(ops);
    pool.free(src);
    pool.free(out);
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

const TYPED_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { len: u32 }
    @workgroup_size(64)
    kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            out_u[i] = src_u[i]
            out_i[i] = src_i[i]
        }
    }
);

#[test]
fn zsl_integer_buffers_round_trip() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = TYPED_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 128u32;
    let input_u: Vec<u32> = (0..n).map(|v| v.wrapping_mul(2_654_435_761)).collect();
    let input_i: Vec<i32> = (0..n as i32).map(|v| v - 64).collect();
    let src_u = pool.alloc(vec![n], DType::U32).unwrap();
    let src_i = pool.alloc(vec![n], DType::I32).unwrap();
    let out_u = pool.alloc(vec![n], DType::U32).unwrap();
    let out_i = pool.alloc(vec![n], DType::I32).unwrap();
    let bytes_u: Vec<u8> = input_u.iter().flat_map(|w| w.to_le_bytes()).collect();
    let bytes_i: Vec<u8> = input_i.iter().flat_map(|w| w.to_le_bytes()).collect();
    device.write_buffer(src_u.buffer, 0, &bytes_u).unwrap();
    device.write_buffer(src_i.buffer, 0, &bytes_i).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            src_u.buffer.index(),
            src_i.buffer.index(),
            out_u.buffer.index(),
            out_i.buffer.index(),
        ],
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |a: &DeviceArray| device.read_buffer(a.buffer, 0, a.size_bytes()).unwrap();
    assert_eq!(read(&out_u), bytes_u);
    assert_eq!(read(&out_i), bytes_i);
    drop(ops);
    for a in [src_u, src_i, out_u, out_i] {
        pool.free(a);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

const FINITE_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { len: u32 }
    @workgroup_size(64)
    kernel classify(src: device buffer<f32>, out: device mut buffer<u32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            let x = src[i]
            if isfinite(x) {
                out[i] = 1
            } else {
                if isnan(x) {
                    out[i] = 2
                } else {
                    if isinf(x) { out[i] = 3 } else { out[i] = 4 }
                }
            }
        }
    }
);

#[test]
fn zsl_finite_classification_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = FINITE_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let input = [0.0f32, -12.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let n = input.len() as u32;
    let src = upload(&pool, vec![n], &input);
    let out = pool.alloc(vec![n], DType::U32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[src.buffer.index(), out.buffer.index()],
        &[zengpu_hal::Scalar::U32(n)],
        [1, 1, 1],
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
    assert_eq!(got, [1, 1, 2, 3, 3]);
    drop(ops);
    pool.free(src);
    pool.free(out);
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

const DOT4_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel dot4(x: device buffer<i32>, y: device buffer<u32>, k: device buffer<i32>, z: device buffer<u32>, out_i: device mut buffer<i32>, out_iu: device mut buffer<i32>, out_u: device mut buffer<u32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            out_i[i] = dot4_i8(x[i], x[i ^ 1], k[i])
            out_iu[i] = dot4_iu8(x[i], y[i], k[i])
            out_u[i] = dot4_u8(y[i], y[i ^ 1], z[i])
        }
    }
);

#[test]
fn zsl_dot4_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = DOT4_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 1000u32;
    const EDGE: [u8; 4] = [0x80, 0x7f, 0xff, 0x00];
    let word = |i: u32| {
        u32::from_le_bytes([
            EDGE[(i % 4) as usize],
            EDGE[((i / 4 + 1) % 4) as usize],
            EDGE[((i / 2 + 2) % 4) as usize],
            (i.wrapping_mul(0x9e37_79b9) >> 24) as u8,
        ])
    };
    let xs: Vec<u32> = (0..n).map(word).collect();
    let ys: Vec<u32> = (0..n).map(|i| word(i + 7)).collect();
    let ks: Vec<u32> = (0..n).map(|i| i32::MAX as u32 - i % 1000).collect();
    let zs: Vec<u32> = (0..n).map(|i| u32::MAX - i % 1000).collect();
    let mut bufs = Vec::new();
    for data in [&xs, &ys, &ks, &zs] {
        let b = pool.alloc(vec![n], DType::U32).unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        device.write_buffer(b.buffer, 0, &bytes).unwrap();
        bufs.push(b);
    }
    let outs: Vec<_> = (0..3)
        .map(|_| pool.alloc(vec![n], DType::U32).unwrap())
        .collect();
    let mut ops = OpList::new(&pool);
    let slots: Vec<u32> = bufs
        .iter()
        .chain(outs.iter())
        .map(|b| b.buffer.index())
        .collect();
    ops.dispatch(
        pipeline,
        &slots,
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |b: &DeviceArray| -> Vec<u32> {
        device
            .read_buffer(b.buffer, 0, b.size_bytes())
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (got_i, got_iu, got_u) = (read(&outs[0]), read(&outs[1]), read(&outs[2]));
    let s8 = |w: u32, l: u32| ((w >> (8 * l)) as u8) as i8 as i32;
    let u8l = |w: u32, l: u32| ((w >> (8 * l)) & 0xff) as i32;
    for i in 0..n as usize {
        let (x, x2, y, y2) = (xs[i], xs[i ^ 1], ys[i], ys[i ^ 1]);
        let (k, z) = (ks[i] as i32, zs[i]);
        let mut want_i = k;
        let mut want_iu = k;
        let mut want_u = z;
        for l in 0..4 {
            want_i = want_i.wrapping_add(s8(x, l).wrapping_mul(s8(x2, l)));
            want_iu = want_iu.wrapping_add(s8(x, l).wrapping_mul(u8l(y, l)));
            want_u = want_u.wrapping_add((u8l(y, l) * u8l(y2, l)) as u32);
        }
        assert_eq!(got_i[i] as i32, want_i, "dot4_i8 at {i}");
        assert_eq!(got_iu[i] as i32, want_iu, "dot4_iu8 at {i}");
        assert_eq!(got_u[i], want_u, "dot4_u8 at {i}");
    }
    drop(ops);
    for b in bufs.into_iter().chain(outs) {
        pool.free(b);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// Reference for f16 operands: products and sums exact, rounded once at the end.
mod packed_ref {
    #[derive(Clone, Copy)]
    enum Fin {
        Zero(bool),
        Num(bool, i128, i32),
        Inf(bool),
        Nan,
    }

    fn classify(h: u16) -> Fin {
        let neg = h >> 15 == 1;
        let exp = (h >> 10) & 0x1f;
        let frac = i128::from(h & 0x3ff);
        match exp {
            0x1f if frac == 0 => Fin::Inf(neg),
            0x1f => Fin::Nan,
            0 if frac == 0 => Fin::Zero(neg),
            0 => Fin::Num(neg, frac, -24),
            _ => Fin::Num(neg, frac | 0x400, i32::from(exp) - 25),
        }
    }

    fn sign(f: Fin) -> bool {
        match f {
            Fin::Zero(s) | Fin::Inf(s) | Fin::Num(s, ..) => s,
            Fin::Nan => false,
        }
    }

    fn mul(a: Fin, b: Fin) -> Fin {
        let s = sign(a) ^ sign(b);
        match (a, b) {
            (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
            (Fin::Inf(_), Fin::Zero(_)) | (Fin::Zero(_), Fin::Inf(_)) => Fin::Nan,
            (Fin::Inf(_), _) | (_, Fin::Inf(_)) => Fin::Inf(s),
            (Fin::Zero(_), _) | (_, Fin::Zero(_)) => Fin::Zero(s),
            (Fin::Num(_, m1, e1), Fin::Num(_, m2, e2)) => Fin::Num(s, m1 * m2, e1 + e2),
        }
    }

    fn add(a: Fin, b: Fin) -> Fin {
        match (a, b) {
            (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
            (Fin::Inf(x), Fin::Inf(y)) => {
                if x == y {
                    Fin::Inf(x)
                } else {
                    Fin::Nan
                }
            }
            (Fin::Inf(x), _) | (_, Fin::Inf(x)) => Fin::Inf(x),
            (Fin::Zero(x), Fin::Zero(y)) => Fin::Zero(x && y),
            (Fin::Zero(_), o) | (o, Fin::Zero(_)) => o,
            (Fin::Num(sa, ma, ea), Fin::Num(sb, mb, eb)) => {
                let e = ea.min(eb);
                let va = (if sa { -ma } else { ma }) << (ea - e);
                let vb = (if sb { -mb } else { mb }) << (eb - e);
                let s = va + vb;
                if s == 0 {
                    Fin::Zero(false)
                } else {
                    Fin::Num(s < 0, s.abs(), e)
                }
            }
        }
    }

    /// Rounds `m * 2^e` (m > 0) to nearest even.
    fn round(neg: bool, m: u128, e: i32) -> u16 {
        let bitlen = (128 - m.leading_zeros()) as i32;
        let drop = (bitlen - 11).max(-24 - e);
        let (mut q, mut qe) = if drop <= 0 {
            (m << (-drop) as u32, e + drop)
        } else if drop >= 127 {
            (0, -24)
        } else {
            let d = drop as u32;
            let mut q = m >> d;
            let rem = m & ((1u128 << d) - 1);
            let half = 1u128 << (d - 1);
            if rem > half || (rem == half && q & 1 == 1) {
                q += 1;
            }
            (q, e + drop)
        };
        let sign = u16::from(neg) << 15;
        if q == 0 {
            return sign;
        }
        while q >= 2048 {
            q >>= 1;
            qe += 1;
        }
        if q >= 1024 {
            let t = qe + 10;
            if t > 15 {
                return sign | 0x7c00;
            }
            sign | (((t + 15) as u16) << 10) | (q as u16 - 1024)
        } else {
            sign | q as u16
        }
    }

    fn bits(f: Fin) -> u16 {
        match f {
            Fin::Nan => 0x7e00,
            Fin::Inf(s) => (u16::from(s) << 15) | 0x7c00,
            Fin::Zero(s) => u16::from(s) << 15,
            Fin::Num(s, m, e) => round(s, m as u128, e),
        }
    }

    pub fn add16(a: u16, b: u16) -> u16 {
        bits(add(classify(a), classify(b)))
    }

    pub fn mul16(a: u16, b: u16) -> u16 {
        bits(mul(classify(a), classify(b)))
    }

    pub fn fma16(a: u16, b: u16, c: u16) -> u16 {
        bits(add(mul(classify(a), classify(b)), classify(c)))
    }

    fn to_f32(h: u16) -> f32 {
        match classify(h) {
            Fin::Nan => f32::NAN,
            Fin::Inf(s) => {
                if s {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            }
            Fin::Zero(s) => {
                if s {
                    -0.0
                } else {
                    0.0
                }
            }
            Fin::Num(s, m, e) => {
                let v = m as f32 * 2f32.powi(e);
                if s { -v } else { v }
            }
        }
    }

    /// Reference dot2 in f32: exact products, then two adds.
    pub fn dot2(a: u32, b: u32, acc: f32) -> f32 {
        let p0 = to_f32(a as u16) * to_f32(b as u16);
        let p1 = to_f32((a >> 16) as u16) * to_f32((b >> 16) as u16);
        (acc + p0) + p1
    }

    pub fn lanes(a: u32, b: u32, f: impl Fn(u16, u16) -> u16) -> u32 {
        u32::from(f(a as u16, b as u16)) | (u32::from(f((a >> 16) as u16, (b >> 16) as u16)) << 16)
    }

    pub fn lanes3(a: u32, b: u32, c: u32, f: impl Fn(u16, u16, u16) -> u16) -> u32 {
        u32::from(f(a as u16, b as u16, c as u16))
            | (u32::from(f((a >> 16) as u16, (b >> 16) as u16, (c >> 16) as u16)) << 16)
    }

    /// Compares per f16 lane, bit-exact except that any NaN matches any NaN.
    pub fn same_f16x2(got: u32, want: u32) -> bool {
        let lane_ok = |g: u16, w: u16| {
            let nan = |h: u16| (h & 0x7fff) > 0x7c00;
            (nan(g) && nan(w)) || g == w
        };
        lane_ok(got as u16, want as u16) && lane_ok((got >> 16) as u16, (want >> 16) as u16)
    }

    pub fn pk_add_u16(a: u32, b: u32) -> u32 {
        lanes(a, b, |x, y| x.wrapping_add(y))
    }

    pub fn pk_mul_u16(a: u32, b: u32) -> u32 {
        lanes(a, b, |x, y| x.wrapping_mul(y))
    }

    pub fn pk_mad_u16(a: u32, b: u32, c: u32) -> u32 {
        lanes3(a, b, c, |x, y, z| x.wrapping_mul(y).wrapping_add(z))
    }

    pub fn funnel_shr(hi: u32, lo: u32, s: u32) -> u32 {
        ((u64::from(hi) << 32 | u64::from(lo)) >> (s & 31)) as u32
    }

    /// Reference for CUDA `__byte_perm` with the 3-bit selector and no sign mode.
    pub fn byte_perm(a: u32, b: u32, sel: u32) -> u32 {
        let src = u64::from(a) | (u64::from(b) << 32);
        (0..4).fold(0u32, |out, i| {
            let k = (sel >> (4 * i)) & 7;
            let byte = ((src >> (8 * k)) & 0xff) as u32;
            out | (byte << (8 * i))
        })
    }
}

const PACKED_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel packed(x: device buffer<u32>, y: device buffer<u32>, z: device buffer<u32>, f: device buffer<f32>, o1: device mut buffer<u32>, o2: device mut buffer<u32>, o3: device mut buffer<u32>, o4: device mut buffer<u32>, o5: device mut buffer<u32>, o6: device mut buffer<u32>, o7: device mut buffer<u32>, o8: device mut buffer<u32>, of: device mut buffer<f32>, o9: device mut buffer<u32>, o10: device mut buffer<u32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            o1[i] = pk_add_u16(x[i], y[i])
            o2[i] = pk_mul_u16(x[i], y[i])
            o3[i] = pk_mad_u16(x[i], y[i], z[i])
            o4[i] = pk_add_f16(x[i], y[i])
            o5[i] = pk_mul_f16(x[i], y[i])
            o6[i] = pk_fma_f16(x[i], y[i], z[i])
            of[i] = dot2_f16(x[i], y[i], f[i])
            o7[i] = funnel_shr(x[i], y[i], z[i])
            o8[i] = byte_perm(x[i], y[i], z[i])
            o9[i] = pk_mul_f16(x[i], 0x3c003c00u32)
            o10[i] = pk_add_u16(x[i], 0x3c003c00u32) + (-2147483648) + 0b1010 + 0o17 + 0xffffffffu32
        }
    }
);

#[test]
fn zsl_packed_ops_match_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = PACKED_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 1024u32;
    const F16_EDGE: [u16; 16] = [
        0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff,
        0x3c00, 0xbc00, 0x3555, 0x4000, 0x0200,
    ];
    let half = |s: u32| -> u16 {
        if s.is_multiple_of(3) {
            (s.wrapping_mul(0x9e37_79b9) >> 16) as u16
        } else {
            F16_EDGE[(s % 16) as usize]
        }
    };
    let lanes = |s: u32| u32::from(half(s)) | (u32::from(half(s.wrapping_add(7919))) << 16);
    let mut xs: Vec<u32> = (0..n).map(lanes).collect();
    let mut ys: Vec<u32> = (0..n).map(|i| lanes(i + 4096)).collect();
    let mut zs: Vec<u32> = (0..n)
        .map(|i| match i % 5 {
            0 => 0,
            1 => 31,
            2 => 32,
            3 => 0x7654_3210,
            _ => lanes(i + 8192),
        })
        .collect();
    xs[0] = (xs[0] & 0xffff_0000) | 0x3c01;
    ys[0] = (ys[0] & 0xffff_0000) | 0x3bff;
    zs[0] = (zs[0] & 0xffff_0000) | 0x0009;
    let fs: Vec<f32> = (0..n)
        .map(|i| match i % 4 {
            0 => 0.0,
            1 => -0.0,
            2 => i as f32 * 0.37 - 100.0,
            _ => f32::from_bits(lanes(i + 123)),
        })
        .collect();
    let mut inputs = Vec::new();
    for data in [&xs, &ys, &zs] {
        let b = pool.alloc(vec![n], DType::U32).unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        device.write_buffer(b.buffer, 0, &bytes).unwrap();
        inputs.push(b);
    }
    let b = pool.alloc(vec![n], DType::F32).unwrap();
    let bytes: Vec<u8> = fs.iter().flat_map(|w| w.to_le_bytes()).collect();
    device.write_buffer(b.buffer, 0, &bytes).unwrap();
    inputs.push(b);
    let outs: Vec<_> = (0..11)
        .map(|k| {
            let ty = if k == 8 { DType::F32 } else { DType::U32 };
            pool.alloc(vec![n], ty).unwrap()
        })
        .collect();
    let mut ops = OpList::new(&pool);
    let slots: Vec<u32> = inputs
        .iter()
        .chain(outs.iter())
        .map(|b| b.buffer.index())
        .collect();
    ops.dispatch(
        pipeline,
        &slots,
        &[zengpu_hal::Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |b: &DeviceArray| -> Vec<u32> {
        device
            .read_buffer(b.buffer, 0, b.size_bytes())
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let got: Vec<Vec<u32>> = outs.iter().map(read).collect();
    use packed_ref as r;
    for i in 0..n as usize {
        let (x, y, z) = (xs[i], ys[i], zs[i]);
        assert!(
            r::same_f16x2(got[9][i], r::lanes(x, 0x3c00_3c00, r::mul16)),
            "pk_mul_f16 with a hex literal at {i}: x={x:08x}"
        );
        assert_eq!(
            got[10][i],
            r::pk_add_u16(x, 0x3c00_3c00)
                .wrapping_add(0x8000_0000)
                .wrapping_add(10)
                .wrapping_add(15)
                .wrapping_add(0xffff_ffff),
            "literal expression at {i}"
        );
        let want = [
            r::pk_add_u16(x, y),
            r::pk_mul_u16(x, y),
            r::pk_mad_u16(x, y, z),
            r::lanes(x, y, r::add16),
            r::lanes(x, y, r::mul16),
            r::lanes3(x, y, z, r::fma16),
            r::funnel_shr(x, y, z),
            r::byte_perm(x, y, z),
        ];
        for (k, w) in want.iter().enumerate() {
            let f16_out = (3..=5).contains(&k);
            let ok = if f16_out {
                r::same_f16x2(got[k][i], *w)
            } else {
                got[k][i] == *w
            };
            assert!(
                ok,
                "output {k} at {i}: got {:08x} want {w:08x} x={x:08x} y={y:08x} z={z:08x}",
                got[k][i]
            );
        }
        let want_f = r::dot2(x, y, fs[i]);
        let got_f = f32::from_bits(got[8][i]);
        let p0 = r::dot2(x & 0xffff, y & 0xffff, 0.0);
        let p1 = r::dot2(x >> 16, y >> 16, 0.0);
        let tol = fs[i].abs().max(p0.abs()).max(p1.abs()) * 1e-4;
        assert!(
            (want_f.is_nan() && got_f.is_nan())
                || want_f.to_bits() == got_f.to_bits()
                || (got_f - want_f).abs() <= tol,
            "dot2_f16 at {i}: got {got_f:?} want {want_f:?} x={x:08x} y={y:08x} acc={:?}",
            fs[i]
        );
    }
    drop(ops);
    for b in inputs.into_iter().chain(outs) {
        pool.free(b);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// gfx12 WMMA: f16 and packed-i8 `C = A*B + C`. SPIR-V has no `frag_row`/`frag_get`/`frag_set`: the
/// lane probe is in `FRAG_PROBE`.
const FRAG_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel frag_gemm(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, out: device mut buffer<f32>, ba: device buffer<u32>, bb: device buffer<u32>, ci: device buffer<i32>, oi: device mut buffer<i32>) {
        let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        frag_store(out, 0, 16, d)
        let ia: frag<i8, 16, 16, a> = frag_load_a(ba, 0, 16)
        let ib: frag<i8, 16, 16, b> = frag_load_b(bb, 0, 16)
        let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
        let id2 = mma(ia, ib, iacc)
        frag_store(oi, 0, 16, id2)
    }
);

/// Needs `frag_get`/`frag_set`, which SPIR-V cannot express, so HIP only.
const FRAG_GET_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel frag_get_bump(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, out: device mut buffer<f32>) {
        let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        let d2 = frag_set(d, 0, frag_get(d, 0) + 1.0)
        frag_store(out, 0, 16, d2)
    }
);

/// Fragment A/B loads from f32 shared memory, staged through f16: loads cannot convert.
const FRAG_SHARED_AB_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel frag_shared_ab(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, out: device mut buffer<f32>) {
        shared sa: array<f32, 256>
        shared sb: array<f32, 256>
        let lane = local_id().x
        for i in 0..8 {
            sa[lane + i * 32] = a[lane + i * 32]
            sb[lane + i * 32] = b[lane + i * 32]
        }
        barrier()
        let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(sb, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        frag_store(out, 0, 16, d)
    }
);

#[test]
fn zsl_frag_ab_from_shared_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = FRAG_SHARED_AB_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let a = upload(&pool, vec![256], &fa);
    let b = upload(&pool, vec![256], &fb);
    let c = upload(&pool, vec![256], &fc);
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            c.buffer.index(),
            out.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[k * 16 + n];
            }
            assert!(
                close(got[m * 16 + n], s),
                "({m}, {n}): {} vs {s}",
                got[m * 16 + n]
            );
        }
    }
    drop(ops);
    for buf in [a, b, c, out] {
        pool.free(buf);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

const TYPED_SHARED_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel typed_shared(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, ia: device buffer<i8>, ib: device buffer<i8>, ci: device buffer<i32>, wide: device buffer<i32>, ui: device buffer<u8>, e: device buffer<f32>, out: device mut buffer<f32>, oi: device mut buffer<i32>, rh: device mut buffer<f32>, rq: device mut buffer<i32>, ru: device mut buffer<u32>) {
        shared sa: array<f16, 256>
        shared sb: array<f16, 256>
        shared qa: array<i8, 256>
        shared qb: array<i8, 256>
        shared qw: array<i8, 256>
        shared us: array<u8, 256>
        shared he: array<f16, 256>
        let lane = local_id().x
        for i in 0..8 {
            sa[lane + i * 32] = a[lane + i * 32]
            sb[lane + i * 32] = b[lane + i * 32]
            qa[lane + i * 32] = ia[lane + i * 32]
            qb[lane + i * 32] = ib[lane + i * 32]
            qw[lane + i * 32] = wide[lane + i * 32]
            us[lane + i * 32] = ui[lane + i * 32]
            he[lane + i * 32] = e[lane + i * 32]
        }
        barrier()
        let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(sb, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        frag_store(out, 0, 16, d)
        let ga: frag<i8, 16, 16, a> = frag_load_a(qa, 0, 16)
        let gb: frag<i8, 16, 16, b> = frag_load_b(qb, 0, 16)
        let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
        let id2 = mma(ga, gb, iacc)
        frag_store(oi, 0, 16, id2)
        for i in 0..8 {
            rh[lane + i * 32] = he[lane + i * 32]
            rq[lane + i * 32] = qw[lane + i * 32]
            ru[lane + i * 32] = us[lane + i * 32]
        }
    }
);

#[test]
fn zsl_typed_shared_round_trips_and_feeds_frags() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = TYPED_SHARED_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let ia: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
    let ib: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
    let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
    let wide: Vec<i32> = (0..256i32).map(|i| i * 37 - 4000).collect();
    let ui: Vec<u8> = (0..256u32).map(|i| (i * 53 % 256) as u8).collect();
    let edge: Vec<f32> = (0..256)
        .map(|i| {
            [
                0.1,
                65504.0,
                65520.0,
                1.0 + 2f32.powi(-11),
                1.0 + 3.0 * 2f32.powi(-11),
                1e-5,
                -2.5e-7,
                123.456,
                2.75,
                -0.0,
            ][i % 10]
        })
        .collect();
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let i8s = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
    let make = |dtype: DType, bytes: &[u8]| {
        let arr = pool.alloc(vec![256], dtype).unwrap();
        device.write_buffer(arr.buffer, 0, bytes).unwrap();
        arr
    };
    let a = make(DType::F32, &f32s(&fa));
    let b = make(DType::F32, &f32s(&fb));
    let c = make(DType::F32, &f32s(&fc));
    let ga = make(DType::I8, &i8s(&ia));
    let gb = make(DType::I8, &i8s(&ib));
    let gci = make(DType::I32, &i32s(&ci));
    let gw = make(DType::I32, &i32s(&wide));
    let gu = make(DType::U8, &ui);
    let ge = make(DType::F32, &f32s(&edge));
    let out = pool.alloc(vec![256], DType::F32).unwrap();
    let oi = pool.alloc(vec![256], DType::I32).unwrap();
    let rh = pool.alloc(vec![256], DType::F32).unwrap();
    let rq = pool.alloc(vec![256], DType::I32).unwrap();
    let ru = pool.alloc(vec![256], DType::U32).unwrap();
    let bufs = [a, b, c, ga, gb, gci, gw, gu, ge, out, oi, rh, rq, ru];
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &bufs.iter().map(|x| x.buffer.index()).collect::<Vec<_>>(),
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |x: &DeviceArray| device.read_buffer(x.buffer, 0, x.size_bytes()).unwrap();
    let floats = |bytes: Vec<u8>| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let ints = |bytes: Vec<u8>| -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (got_out, got_oi) = (floats(read(&bufs[9])), ints(read(&bufs[10])));
    let (got_rh, got_rq) = (floats(read(&bufs[11])), ints(read(&bufs[12])));
    let got_ru: Vec<u32> = read(&bufs[13])
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            let mut si = ci[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[k * 16 + n];
                si = si.wrapping_add(i32::from(ia[m * 16 + k]) * i32::from(ib[k * 16 + n]));
            }
            let at = m * 16 + n;
            assert!(
                close(got_out[at], s),
                "f16 mma from shared at ({m}, {n}): {} vs {s}",
                got_out[at]
            );
            assert_eq!(got_oi[at], si, "i8 mma from shared at ({m}, {n})");
        }
    }
    for i in 0..256 {
        let want = f16_bits_to_f32(f32_to_f16_bits(edge[i]));
        assert_eq!(
            got_rh[i].to_bits(),
            want.to_bits(),
            "f16 round trip of {}",
            edge[i]
        );
        assert_eq!(
            got_rq[i],
            i32::from(wide[i] as i8),
            "i8 narrowing of {}",
            wide[i]
        );
        assert_eq!(got_ru[i], u32::from(ui[i]), "u8 round trip");
    }
    drop(ops);
    for x in bufs {
        pool.free(x);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// Loads B from an N x K row-major source with `frag_load_b_t`, with no transpose stage.
const FRAG_BT_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel frag_bt(a: device buffer<f16>, b: device buffer<f16>, ai: device buffer<u32>, bi: device buffer<u32>, c: device buffer<f32>, ci: device buffer<i32>, out: device mut buffer<f32>, oi: device mut buffer<i32>) {
        let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b_t(b, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        frag_store(out, 0, 16, mma(fa, fb, acc))
        let ia: frag<i8, 16, 16, a> = frag_load_a(ai, 0, 16)
        let ib: frag<i8, 16, 16, b> = frag_load_b_t(bi, 0, 16)
        let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
        frag_store(oi, 0, 16, mma(ia, ib, iacc))
    }
);

#[test]
fn zsl_frag_load_b_t_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = FRAG_BT_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let ia: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
    let ib: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
    let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
    let bits_a: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let bits_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let pack = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
    let a = upload_half(&pool, vec![256], DType::F16, &bits_a);
    let b = upload_half(&pool, vec![256], DType::F16, &bits_b);
    let ai = pool.alloc(vec![64], DType::U32).unwrap();
    let bi = pool.alloc(vec![64], DType::U32).unwrap();
    device.write_buffer(ai.buffer, 0, &pack(&ia)).unwrap();
    device.write_buffer(bi.buffer, 0, &pack(&ib)).unwrap();
    let c = upload(&pool, vec![256], &fc);
    let ci_buf = pool.alloc(vec![256], DType::I32).unwrap();
    let ci_bytes: Vec<u8> = ci.iter().flat_map(|x| x.to_le_bytes()).collect();
    device.write_buffer(ci_buf.buffer, 0, &ci_bytes).unwrap();
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let oi = pool.alloc(vec![256], DType::I32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            ai.buffer.index(),
            bi.buffer.index(),
            c.buffer.index(),
            ci_buf.buffer.index(),
            out.buffer.index(),
            oi.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    let got_i: Vec<i32> = device
        .read_buffer(oi.buffer, 0, 1024)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            let mut si = ci[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[n * 16 + k];
                si = si.wrapping_add(i32::from(ia[m * 16 + k]) * i32::from(ib[n * 16 + k]));
            }
            assert!(
                close(got[m * 16 + n], s),
                "f16 ({m}, {n}): {} vs {s}",
                got[m * 16 + n]
            );
            assert_eq!(got_i[m * 16 + n], si, "i8 ({m}, {n})");
        }
    }
    drop(ops);
    for buf in [a, b, c, out] {
        pool.free(buf);
    }
    for buf in [ai, bi, ci_buf, oi] {
        pool.free(buf);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

const FEATURES_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel scalar_features(ua: device buffer<u32>, ia: device buffer<i32>, fa: device buffer<f32>, sb: device buffer<i8>, ub: device buffer<u8>, ou: device mut buffer<u32>, oi: device mut buffer<i32>, of: device mut buffer<f32>, ob: device mut buffer<i8>, oub: device mut buffer<u8>, oh: device mut buffer<i32>, oh2: device mut buffer<i32>, ofr: device mut buffer<f32>, osel: device mut buffer<i32>, id: global_id) {
        let i = id.x
        ou[i] = ua[i] % 7u
        oi[i] = ia[i] % 5
        of[i] = fa[i] % 2.5
        ob[i] = sb[i] + 1
        oub[i] = ub[i] + 1
        oh[i] = i32(fa[i] * 3.0)
        oh2[i] = i32(ua[i])
        ofr[i] = f16_round(fa[i])
        osel[i] = select(ia[i] < 0, ia[i] * 2, ia[i])
    }
);

#[test]
fn zsl_scalar_features_match_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = FEATURES_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 256usize;
    let ua: Vec<u32> = (0..n as u32).map(|i| i * 37 + 5).collect();
    let ia: Vec<i32> = (0..n as i32).map(|i| i * 13 - 1500).collect();
    let fa: Vec<f32> = (0..n)
        .map(|i| match i % 16 {
            0 => 3.0e9,
            1 => -3.0e9,
            _ => i as f32 * 0.37 - 40.0,
        })
        .collect();
    let sb: Vec<i8> = (0..n as i32).map(|i| (i * 7 - 500) as i8).collect();
    let ub: Vec<u8> = (0..n).map(|i| (i * 11 % 256) as u8).collect();
    let u32_buf = |vals: &[u32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i32_buf = |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f32_buf = |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let sb_bytes: Vec<u8> = sb.iter().map(|&v| v as u8).collect();
    let mk = |bytes: &[u8]| {
        let buf = pool
            .alloc(vec![(bytes.len() as u32).div_ceil(4)], DType::U32)
            .unwrap();
        device.write_buffer(buf.buffer, 0, bytes).unwrap();
        buf
    };
    let bufs = [
        mk(&u32_buf(&ua)),
        mk(&i32_buf(&ia)),
        mk(&f32_buf(&fa)),
        mk(&sb_bytes),
        mk(&ub),
    ];
    let outs: Vec<_> = [1024usize, 1024, 1024, 256, 256, 1024, 1024, 1024, 1024]
        .iter()
        .map(|&size| {
            pool.alloc(vec![(size as u32).div_ceil(4)], DType::U32)
                .unwrap()
        })
        .collect();
    let mut ops = OpList::new(&pool);
    let slots: Vec<_> = bufs
        .iter()
        .chain(outs.iter())
        .map(|b| b.buffer.index())
        .collect();
    ops.dispatch(pipeline, &slots, &[], [8, 1, 1]);
    ops.run(&*device).unwrap();
    let read = |b: &DeviceArray, len: u64| device.read_buffer(b.buffer, 0, len).unwrap();
    let got_u32 = |b: &DeviceArray| -> Vec<u32> {
        read(b, 1024)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let got_i32 = |b: &DeviceArray| -> Vec<i32> {
        read(b, 1024)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let got_f32 = |b: &DeviceArray| -> Vec<f32> {
        read(b, 1024)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (ou, oi, of) = (got_u32(&outs[0]), got_i32(&outs[1]), got_f32(&outs[2]));
    let ob = read(&outs[3], 256);
    let oub = read(&outs[4], 256);
    let (oh, oh2, ofr, osel) = (
        got_i32(&outs[5]),
        got_i32(&outs[6]),
        got_f32(&outs[7]),
        got_i32(&outs[8]),
    );
    for i in 0..n {
        assert_eq!(ou[i], ua[i] % 7, "u32 %% at {i}");
        assert_eq!(oi[i], ia[i] % 5, "i32 %% at {i}");
        assert!(
            of[i] == fa[i] % 2.5,
            "f32 % at {i}: {} vs {}",
            of[i],
            fa[i] % 2.5
        );
        assert_eq!(ob[i] as i8, (i32::from(sb[i]) + 1) as i8, "i8 store at {i}");
        assert_eq!(oub[i], ((u32::from(ub[i])) + 1) as u8, "u8 store at {i}");
        assert_eq!(oh[i], (fa[i] * 3.0) as i32, "i32(f32) at {i}");
        assert_eq!(oh2[i], ua[i] as i32, "i32(u32) at {i}");
        let want_round = f16_bits_to_f32(f32_to_f16_bits(fa[i]));
        assert!(
            ofr[i] == want_round || (ofr[i].is_nan() && want_round.is_nan()),
            "f16_round at {i}: {} vs {want_round}",
            ofr[i]
        );
        let want_sel = if ia[i] < 0 { ia[i] * 2 } else { ia[i] };
        assert_eq!(osel[i], want_sel, "select at {i}");
    }
    drop(ops);
    for b in bufs.into_iter().chain(outs) {
        pool.free(b);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// Lane probe, HIP only: SPIR-V's cooperative matrix mapping is opaque. Lowered with `zengpu-zslc`,
/// not `zsl!`.
const FRAG_PROBE: &str = r#"
    @workgroup_size(32)
    kernel frag_probe(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, pr: device mut buffer<u32>, pc: device mut buffer<u32>, pv: device mut buffer<f32>, out2: device mut buffer<f32>, id: global_id) {
        let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        let lane = local_id().x
        for i in 0..8 {
            pr[lane * 8 + i] = frag_row(d, i)
            pc[lane * 8 + i] = frag_col(d, i)
            pv[lane * 8 + i] = frag_get(d, i)
        }
        let d2 = frag_set(d, 0, frag_get(d, 0) + 1.0)
        frag_store(out2, 0, 16, d2)
    }
"#;

#[test]
fn zsl_frag_mma_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = FRAG_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let ba8: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
    let bb8: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
    let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
    let words = |v: &[i8]| -> Vec<u8> {
        v.chunks_exact(4)
            .flat_map(|c| {
                u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8]).to_le_bytes()
            })
            .collect()
    };
    let halves: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let halves_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let a = upload_half(&pool, vec![256], DType::F16, &halves);
    let b = upload_half(&pool, vec![256], DType::F16, &halves_b);
    let c = upload(&pool, vec![256], &fc);
    let ba = pool.alloc(vec![64], DType::U32).unwrap();
    let bb = pool.alloc(vec![64], DType::U32).unwrap();
    device.write_buffer(ba.buffer, 0, &words(&ba8)).unwrap();
    device.write_buffer(bb.buffer, 0, &words(&bb8)).unwrap();
    let ci_buf = pool.alloc(vec![256], DType::I32).unwrap();
    let ci_bytes: Vec<u8> = ci.iter().flat_map(|v| v.to_le_bytes()).collect();
    device.write_buffer(ci_buf.buffer, 0, &ci_bytes).unwrap();
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let oi = pool.alloc(vec![256], DType::I32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            c.buffer.index(),
            out.buffer.index(),
            ba.buffer.index(),
            bb.buffer.index(),
            ci_buf.buffer.index(),
            oi.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got_f = download(&*device, &out);
    let got_i: Vec<i32> = device
        .read_buffer(oi.buffer, 0, oi.size_bytes())
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            let mut si = ci[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[k * 16 + n];
                si = si.wrapping_add(i32::from(ba8[m * 16 + k]) * i32::from(bb8[k * 16 + n]));
            }
            let at = m * 16 + n;
            assert!(
                close(got_f[at], s),
                "f16 mma at ({m}, {n}): {} vs {s}",
                got_f[at]
            );
            assert_eq!(got_i[at], si, "i8 mma at ({m}, {n})");
        }
    }
    drop(ops);
    for buf in [a, b, c, out, oi] {
        pool.free(buf);
    }
    for buf in [ba, bb, ci_buf] {
        pool.free(buf);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

#[test]
fn zsl_frag_get_is_hip_only_and_runs_on_hip() {
    assert!(FRAG_GET_ZSL.spv.is_err(), "frag_get has no SPIR-V lowering");
    let err = FRAG_GET_ZSL.spirv_desc().unwrap_err().to_string();
    assert!(err.contains("no spirv artifact"), "{err}");
    assert!(FRAG_GET_ZSL.msl.is_err(), "frag_get has no MSL lowering");

    let Some(device) = hip_device() else { return };
    let (shader, entry) = FRAG_GET_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let halves: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let halves_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let a = upload_half(&pool, vec![256], DType::F16, &halves);
    let b = upload_half(&pool, vec![256], DType::F16, &halves_b);
    let c = upload(&pool, vec![256], &fc);
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            c.buffer.index(),
            out.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    let mut bumped = 0;
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[k * 16 + n];
            }
            let diff = got[m * 16 + n] - s;
            if close(diff, 1.0) {
                bumped += 1;
            } else {
                assert!(close(diff, 0.0), "({m}, {n}): diff {diff}");
            }
        }
    }
    assert_eq!(bumped, 32, "one frag_set bump per lane");
    drop(ops);
    for buf in [a, b, c, out] {
        pool.free(buf);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

#[test]
fn zsl_frag_probe_maps_lanes() {
    use zengpu_zslc::{backend::hip::lower_compute, frontend::parser::parse_compute};
    let Some(device) = hip_device() else { return };
    let module = parse_compute(FRAG_PROBE).unwrap();
    let hip = lower_compute(&module).unwrap();
    let shader = device
        .create_shader(zengpu_hal::ShaderDesc::hip(&hip.source))
        .unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry: hip.entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let halves: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let halves_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let a = upload_half(&pool, vec![256], DType::F16, &halves);
    let b = upload_half(&pool, vec![256], DType::F16, &halves_b);
    let c = upload(&pool, vec![256], &fc);
    let pr = pool.alloc(vec![256], DType::U32).unwrap();
    let pc = pool.alloc(vec![256], DType::U32).unwrap();
    let pv = upload(&pool, vec![256], &[0.0; 256]);
    let out2 = upload(&pool, vec![256], &[0.0; 256]);
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            c.buffer.index(),
            pr.buffer.index(),
            pc.buffer.index(),
            pv.buffer.index(),
            out2.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let read_u32 = |arr: &DeviceArray| -> Vec<u32> {
        device
            .read_buffer(arr.buffer, 0, arr.size_bytes())
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (got_r, got_c) = (read_u32(&pr), read_u32(&pc));
    let got_v = download(&*device, &pv);
    let got_out2 = download(&*device, &out2);

    let mut want = vec![0f32; 256];
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[k * 16 + n];
            }
            want[m * 16 + n] = s;
        }
    }
    let mut seen = [false; 256];
    for i in 0..256 {
        let (r, c) = (got_r[i] as usize, got_c[i] as usize);
        assert!(r < 16 && c < 16, "frag_row/frag_col out of range at {i}");
        assert!(!seen[r * 16 + c], "frag position ({r}, {c}) mapped twice");
        seen[r * 16 + c] = true;
        assert!(close(got_v[i], want[r * 16 + c]), "frag_get at ({r}, {c})");
    }
    for r in 0..16 {
        for c in 0..16 {
            let bump = if r == 0 || r == 8 { 1.0 } else { 0.0 };
            assert!(
                close(got_out2[r * 16 + c], want[r * 16 + c] + bump),
                "frag_set store at ({r}, {c})"
            );
        }
    }
    drop(ops);
    for buf in [a, b, c, pv, out2] {
        pool.free(buf);
    }
    for buf in [pr, pc] {
        pool.free(buf);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// Vector copy shader over `u32x4` elements, with a scalar tail past the last full vector.
const VEC_COPY_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    push P { n: u32, tail: u32 }
    @workgroup_size(64)
    kernel vec_copy(src: device buffer<u32x4>, out: device mut buffer<u32x4>, words: device buffer<u32>, out_words: device mut buffer<u32>, p: P, id: global_id) {
        if id.x < p.n {
            let v = src[id.x]
            out[id.x] = u32x4(v.w, v.z, v.y + v.x, v.x)
        }
        if id.x < p.tail {
            out_words[p.n * 4 + id.x] = words[p.n * 4 + id.x]
        }
    }
);

#[test]
fn zsl_u32x4_copy_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = VEC_COPY_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let (n, tail) = (5u32, 3u32);
    let words: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let src_words: Vec<u32> = (0..20u32)
        .map(|i| i.wrapping_mul(40_503) ^ 0x9e37)
        .collect();
    let as_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let src = pool.alloc(vec![20], DType::U32).unwrap();
    device
        .write_buffer(src.buffer, 0, &as_bytes(&src_words))
        .unwrap();
    let wbuf = pool.alloc(vec![32], DType::U32).unwrap();
    device
        .write_buffer(wbuf.buffer, 0, &as_bytes(&words))
        .unwrap();
    let out = pool.alloc(vec![20], DType::U32).unwrap();
    device
        .write_buffer(out.buffer, 0, &as_bytes(&[0u32; 20]))
        .unwrap();
    let out_words = pool.alloc(vec![32], DType::U32).unwrap();
    device
        .write_buffer(out_words.buffer, 0, &as_bytes(&[0u32; 32]))
        .unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            src.buffer.index(),
            out.buffer.index(),
            wbuf.buffer.index(),
            out_words.buffer.index(),
        ],
        &[zengpu_hal::Scalar::U32(n), zengpu_hal::Scalar::U32(tail)],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |b: &DeviceArray, count: usize| -> Vec<u32> {
        device
            .read_buffer(b.buffer, 0, (count * 4) as u64)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let got = read(&out, 20);
    for i in 0..n as usize {
        let v = &src_words[i * 4..i * 4 + 4];
        let want = [v[3], v[2], v[1].wrapping_add(v[0]), v[0]];
        assert_eq!(&got[i * 4..i * 4 + 4], &want, "vector {i}");
    }
    let got_words = read(&out_words, 32);
    let tail_start = (n * 4) as usize;
    for i in 0..tail as usize {
        assert_eq!(
            got_words[tail_start + i],
            words[tail_start + i],
            "tail word {i}"
        );
    }
    assert!(got_words[..tail_start].iter().all(|&w| w == 0));
}

/// Expected f16 decode results: exact, with NaN staying NaN.
fn f16_decode(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    if e == 0 {
        sign * m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

/// Inputs for `pack_f16x2`: subnormal, inf and tie edges.
const BIT_HALVES: [u16; 16] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x3c00,
    0xbc00, 0x3555, 0x0200, 0x7d00,
];
const BIT_F32: [f32; 16] = [
    0.0,
    -0.0,
    1.000_488_3,
    1.001_464_8,
    65520.0,
    2.980_232_2e-8,
    8.940_697e-8,
    1.0,
    -2.5,
    65504.0,
    f32::INFINITY,
    f32::NAN,
    1.0e-8,
    -1.0e-8,
    0.333_333_34,
    3.0e-5,
];
/// Expected `pack_f16x2` bits for `BIT_F32`. 0x7e00 matches any NaN.
const BIT_F32_F16: [u16; 16] = [
    0x0000, 0x8000, 0x3c00, 0x3c02, 0x7c00, 0x0000, 0x0002, 0x3c00, 0xc100, 0x7bff, 0x7c00, 0x7e00,
    0x0000, 0x8000, 0x3555, 0x01f7,
];

const BITS_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel bits(x: device buffer<u32>, y: device buffer<f32>, z: device buffer<f32>, lo: device mut buffer<f32>, hi: device mut buffer<f32>, packed: device mut buffer<u32>, bf: device mut buffer<f32>, bu: device mut buffer<u32>, bi: device mut buffer<i32>, sb: device mut buffer<i32>, ub: device mut buffer<u32>, id: global_id) {
        lo[id.x] = f16_lo(x[id.x])
        hi[id.x] = f16_hi(x[id.x])
        packed[id.x] = pack_f16x2(y[id.x], z[id.x])
        bf[id.x] = bitcast_f32(x[id.x])
        bu[id.x] = bitcast_u32(y[id.x])
        bi[id.x] = bitcast_i32(x[id.x])
        sb[id.x] = sbyte(x[id.x], id.x)
        ub[id.x] = byte(x[id.x], id.x)
    }
);

#[test]
fn zsl_bit_builtins_match_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = BITS_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let n = 256usize;
    let xs: Vec<u32> = (0..n)
        .map(|i| {
            let lo = BIT_HALVES[i % 16];
            let hi = BIT_HALVES[(i * 7 + 3) % 16];
            u32::from(lo) | (u32::from(hi) << 16)
        })
        .collect();
    let ys: Vec<f32> = (0..n).map(|i| BIT_F32[i % 16]).collect();
    let zs: Vec<f32> = (0..n).map(|i| BIT_F32[(i * 5 + 1) % 16]).collect();
    let words = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let x = pool.alloc(vec![n as u32], DType::U32).unwrap();
    device.write_buffer(x.buffer, 0, &words(&xs)).unwrap();
    let y = upload(&pool, vec![n as u32], &ys);
    let z = upload(&pool, vec![n as u32], &zs);
    let lo = upload(&pool, vec![n as u32], &vec![0.0; n]);
    let hi = upload(&pool, vec![n as u32], &vec![0.0; n]);
    let packed = pool.alloc(vec![n as u32], DType::U32).unwrap();
    let bf = upload(&pool, vec![n as u32], &vec![0.0; n]);
    let bu = pool.alloc(vec![n as u32], DType::U32).unwrap();
    let bi = pool.alloc(vec![n as u32], DType::I32).unwrap();
    let sb = pool.alloc(vec![n as u32], DType::I32).unwrap();
    let ub = pool.alloc(vec![n as u32], DType::U32).unwrap();
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            x.buffer.index(),
            y.buffer.index(),
            z.buffer.index(),
            lo.buffer.index(),
            hi.buffer.index(),
            packed.buffer.index(),
            bf.buffer.index(),
            bu.buffer.index(),
            bi.buffer.index(),
            sb.buffer.index(),
            ub.buffer.index(),
        ],
        &[],
        [(n / 32) as u32, 1, 1],
    );
    ops.run(&*device).unwrap();
    let read = |arr: &DeviceArray| device.read_buffer(arr.buffer, 0, arr.size_bytes()).unwrap();
    let as_f32 = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let as_u32 = |b: Vec<u8>| -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (got_lo, got_hi) = (as_f32(read(&lo)), as_f32(read(&hi)));
    let got_packed = as_u32(read(&packed));
    let got_bf = as_f32(read(&bf));
    let got_bu = as_u32(read(&bu));
    let got_bi = as_u32(read(&bi));
    let got_sb = as_u32(read(&sb));
    let got_ub = as_u32(read(&ub));
    let same_f32 = |got: f32, want: f32| {
        if want.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == want.to_bits()
        }
    };
    let same_f16 = |got: u16, want: u16| {
        if want == 0x7e00 {
            got & 0x7fff > 0x7c00
        } else {
            got == want
        }
    };
    for i in 0..n {
        let x_bits = xs[i];
        let (lo_h, hi_h) = ((x_bits & 0xffff) as u16, (x_bits >> 16) as u16);
        assert!(
            same_f32(got_lo[i], f16_decode(lo_h)),
            "f16_lo {lo_h:#06x}: {}",
            got_lo[i]
        );
        assert!(
            same_f32(got_hi[i], f16_decode(hi_h)),
            "f16_hi {hi_h:#06x}: {}",
            got_hi[i]
        );
        let k = i % 16;
        let k2 = (i * 5 + 1) % 16;
        let (g_lo, g_hi) = (
            (got_packed[i] & 0xffff) as u16,
            (got_packed[i] >> 16) as u16,
        );
        assert!(
            same_f16(g_lo, BIT_F32_F16[k]),
            "pack lo {}: {g_lo:#06x}",
            BIT_F32[k]
        );
        assert!(
            same_f16(g_hi, BIT_F32_F16[k2]),
            "pack hi {}: {g_hi:#06x}",
            BIT_F32[k2]
        );
        assert_eq!(got_bf[i].to_bits(), x_bits, "bitcast_f32 at {i}");
        assert_eq!(got_bu[i], ys[i].to_bits(), "bitcast_u32 at {i}");
        assert_eq!(got_bi[i], x_bits, "bitcast_i32 at {i}");
        let lane = (i % 4) as u32;
        let byte = (x_bits >> (8 * lane)) & 0xff;
        assert_eq!(got_ub[i], byte, "byte at {i}");
        assert_eq!(got_sb[i], (byte as u8 as i8 as i32) as u32, "sbyte at {i}");
    }
    drop(ops);
    for arr in [x, y, z, lo, hi, bf] {
        pool.free(arr);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// Stores `u32x4`, runs a barrier, then loads f16 fragments and runs one mma.
const VIEW_MMA_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel view_mma(a: device buffer<u32x4>, b: device buffer<u32x4>, c: device buffer<f32>, out: device mut buffer<f32>) {
        shared sa: array<f16, 256>
        shared sb: array<f16, 256>
        view sa4: array<u32x4> = sa
        view sb4: array<u32x4> = sb
        let lane = local_id().x
        sa4[lane] = a[lane]
        sb4[lane] = b[lane]
        barrier()
        let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b_t(sb, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let done: frag<f32, 16, 16, acc> = mma(fa, fb, acc)
        frag_store(out, 0, 16, done)
    }
);

#[test]
fn zsl_shared_view_mma_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = VIEW_MMA_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let fa: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let fb: Vec<f32> = (0..256)
        .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
        .collect();
    let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
    let bits_a: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let bits_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let a = upload_half(&pool, vec![256], DType::F16, &bits_a);
    let b = upload_half(&pool, vec![256], DType::F16, &bits_b);
    let c = upload(&pool, vec![256], &fc);
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            a.buffer.index(),
            b.buffer.index(),
            c.buffer.index(),
            out.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    for m in 0..16 {
        for n in 0..16 {
            let mut s = fc[m * 16 + n];
            for k in 0..16 {
                s += fa[m * 16 + k] * fb[n * 16 + k];
            }
            assert!(
                close(got[m * 16 + n], s),
                "({m}, {n}): {} vs {s}",
                got[m * 16 + n]
            );
        }
    }
    drop(ops);
    for arr in [a, b, c, out] {
        pool.free(arr);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}

/// One buffer, two params: `x` written as u32x4, `xh` read as f16 after a barrier.
const ALIAS_BUF_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
    @workgroup_size(32)
    kernel alias_buf(src: device buffer<u32x4>, x: device mut buffer<u32x4>, xh: device buffer<f16>, out: device mut buffer<f32>, id: global_id) {
        x[id.x] = src[id.x]
        barrier()
        for j in 0..8 {
            out[id.x * 8 + j] = xh[id.x * 8 + j]
        }
    }
);

#[test]
fn zsl_one_buffer_two_element_types_matches_cpu() {
    let Some(device) = hip_device() else { return };
    let (shader, entry) = ALIAS_BUF_ZSL.create(&*device).unwrap();
    let pipeline = device
        .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
            shader,
            entry,
            block: [32, 1, 1],
        })
        .unwrap();
    let pool = BufferPool::new(device.clone());
    let vals: Vec<f32> = (0..256)
        .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
        .collect();
    let bits: Vec<u16> = vals.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let src = upload_half(&pool, vec![256], DType::F16, &bits);
    let x = upload_half(&pool, vec![256], DType::F16, &[0; 256]);
    let out = upload(&pool, vec![256], &[0.0; 256]);
    let mut ops = OpList::new(&pool);
    ops.dispatch(
        pipeline,
        &[
            src.buffer.index(),
            x.buffer.index(),
            x.buffer.index(),
            out.buffer.index(),
        ],
        &[],
        [1, 1, 1],
    );
    ops.run(&*device).unwrap();
    let got = download(&*device, &out);
    for (h, (&g, &v)) in got.iter().zip(&vals).enumerate() {
        assert_eq!(g, v, "half {h}");
    }
    drop(ops);
    for arr in [src, x, out] {
        pool.free(arr);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}
