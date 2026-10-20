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
    kernel packed(x: device buffer<u32>, y: device buffer<u32>, z: device buffer<u32>, f: device buffer<f32>, o1: device mut buffer<u32>, o2: device mut buffer<u32>, o3: device mut buffer<u32>, o4: device mut buffer<u32>, o5: device mut buffer<u32>, o6: device mut buffer<u32>, o7: device mut buffer<u32>, o8: device mut buffer<u32>, of: device mut buffer<f32>, p: P, id: global_id) {
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
        if s % 3 == 0 {
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
    let outs: Vec<_> = (0..9)
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
            "dot2_f16 at {i}: got {got_f:?} want {want_f:?} x={x:08x} y={y:08x} acc={:?}", fs[i]
        );
    }
    drop(ops);
    for b in inputs.into_iter().chain(outs) {
        pool.free(b);
    }
    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
}
