//! Conformance harness. Each `run_*_suite` runs ops on any [`GpuDevice`]. Run on the CPU oracle
//! first, then on each GPU backend.
//!
//! `compare_*` runs the same op on two devices and requires byte-identical results. One of them is
//! the CPU oracle.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::mem;
use std::slice;

use zengpu_hal::{
    Bindings, BufferDesc, BufferUsage, GpuDevice, GpuError, MemoryUsage, PipelineHandle, Scalar,
    UsageError,
};

fn rw_desc(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
        memory: MemoryUsage::Upload,
    }
}

/// Runs the full buffer suite on `dev`.
///
/// # Panics
///
/// Panics on failure. `label` names the backend in the message.
pub fn run_buffer_suite(label: &str, dev: &dyn GpuDevice) {
    buffer_roundtrip(label, dev);
    buffer_partial_read(label, dev);
    buffer_read_into(label, dev);
    buffer_explicit_init(label, dev);
    buffer_missing_readback_usage(label, dev);
    buffer_stale_after_destroy(label, dev);
    buffer_out_of_bounds_write(label, dev);
    buffer_multiple_allocs(label, dev);
    buffer_copy_ranges(label, dev);
    buffer_copy_validates_usage(label, dev);
}

fn buffer_roundtrip(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(8)).unwrap();
    dev.write_buffer(h, 0, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let out = dev.read_buffer(h, 0, 8).unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8], "[{label}] buffer_roundtrip");
    dev.destroy_buffer(h);
}

fn buffer_partial_read(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(8)).unwrap();
    dev.write_buffer(h, 0, &[10, 20, 30, 40, 50, 60, 70, 80])
        .unwrap();
    assert_eq!(
        dev.read_buffer(h, 2, 4).unwrap(),
        [30, 40, 50, 60],
        "[{label}] buffer_partial_read"
    );
    assert_eq!(
        dev.read_buffer(h, 7, 1).unwrap(),
        [80],
        "[{label}] buffer_partial_read tail"
    );
    dev.destroy_buffer(h);
}

fn buffer_explicit_init(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(16)).unwrap();
    dev.write_buffer(h, 0, &[0xA5; 16]).unwrap();
    dev.write_buffer(h, 0, &[0; 16]).unwrap();
    let out = dev.read_buffer(h, 0, 16).unwrap();
    assert_eq!(out, vec![0u8; 16], "[{label}] buffer_explicit_init");
    dev.destroy_buffer(h);
}

fn buffer_missing_readback_usage(label: &str, dev: &dyn GpuDevice) {
    let h = dev
        .create_buffer(BufferDesc {
            size: 4,
            usage: BufferUsage::STORAGE,
            memory: MemoryUsage::Upload,
        })
        .unwrap();
    let err = dev.read_buffer(h, 0, 4).unwrap_err();
    assert!(
        matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "READBACK",
                ..
            })
        ),
        "[{label}] missing-readback: expected MissingUsage, got {err}"
    );
    dev.destroy_buffer(h);
}

fn buffer_stale_after_destroy(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(4)).unwrap();
    dev.destroy_buffer(h);
    let err = dev.read_buffer(h, 0, 4).unwrap_err();
    assert!(
        matches!(err, GpuError::InvalidUsage(UsageError::StaleHandle { .. })),
        "[{label}] stale-after-destroy: expected StaleHandle, got {err}"
    );
}

fn buffer_out_of_bounds_write(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(4)).unwrap();
    let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
    assert!(
        matches!(err, GpuError::InvalidUsage(UsageError::BindingMismatch(_))),
        "[{label}] out-of-bounds-write: expected BindingMismatch, got {err}"
    );
    dev.destroy_buffer(h);
}

fn buffer_multiple_allocs(label: &str, dev: &dyn GpuDevice) {
    let handles: Vec<_> = (0..8u64)
        .map(|i| {
            let h = dev.create_buffer(rw_desc(4)).unwrap();
            dev.write_buffer(h, 0, &[i as u8; 4]).unwrap();
            h
        })
        .collect();
    for (i, h) in handles.iter().enumerate() {
        assert_eq!(
            dev.read_buffer(*h, 0, 4).unwrap(),
            vec![i as u8; 4],
            "[{label}] buffer_multiple_allocs slot {i}"
        );
    }
    for h in handles {
        dev.destroy_buffer(h);
    }
}

fn buffer_read_into(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(8)).unwrap();
    dev.write_buffer(h, 0, &[10, 20, 30, 40, 50, 60, 70, 80])
        .unwrap();
    let mut caller_owned = [0xAA; 6];
    dev.read_buffer_into(h, 2, &mut caller_owned[1..5]).unwrap();
    assert_eq!(
        caller_owned,
        [0xAA, 30, 40, 50, 60, 0xAA],
        "[{label}] allocation-free caller-owned readback"
    );
    let err = dev
        .read_buffer_into(h, 7, &mut caller_owned[..2])
        .unwrap_err();
    assert!(
        matches!(err, GpuError::InvalidUsage(UsageError::BindingMismatch(_))),
        "[{label}] read_buffer_into bounds: got {err}"
    );
    dev.destroy_buffer(h);
}

fn buffer_copy_ranges(label: &str, dev: &dyn GpuDevice) {
    let src = dev.create_buffer(rw_desc(16)).unwrap();
    let dst = dev.create_buffer(rw_desc(16)).unwrap();
    dev.write_buffer(src, 0, &(0u8..16).collect::<Vec<_>>())
        .unwrap();
    dev.write_buffer(dst, 0, &[0xA5; 16]).unwrap();
    dev.copy_buffer(src, 3, dst, 7, 6).unwrap();
    assert_eq!(
        dev.read_buffer(dst, 0, 16).unwrap(),
        [
            0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 3, 4, 5, 6, 7, 8, 0xA5, 0xA5, 0xA5
        ],
        "[{label}] buffer_copy_ranges"
    );
    dev.destroy_buffer(src);
    dev.destroy_buffer(dst);
}

fn buffer_copy_validates_usage(label: &str, dev: &dyn GpuDevice) {
    let src = dev
        .create_buffer(BufferDesc {
            size: 4,
            usage: BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        })
        .unwrap();
    let dst = dev.create_buffer(rw_desc(4)).unwrap();
    let err = dev.copy_buffer(src, 0, dst, 0, 4).unwrap_err();
    assert!(
        matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "TRANSFER_SRC",
                ..
            })
        ),
        "[{label}] copy usage: expected MissingUsage, got {err}"
    );
    dev.destroy_buffer(src);
    dev.destroy_buffer(dst);
}

/// Writes then reads the same data on `a` and `b`. Results must be byte-identical. `a` is the
/// CPU oracle.
///
/// # Panics
///
/// Panics if the results differ.
pub fn compare_buffer_write(label_a: &str, a: &dyn GpuDevice, label_b: &str, b: &dyn GpuDevice) {
    let patterns: &[&[u8]] = &[
        &[0xDE, 0xAD, 0xBE, 0xEF],
        &[0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA],
        &[0u8; 64],
        &[255u8; 64],
    ];
    for (i, data) in patterns.iter().enumerate() {
        let ha = a.create_buffer(rw_desc(data.len() as u64)).unwrap();
        let hb = b.create_buffer(rw_desc(data.len() as u64)).unwrap();
        a.write_buffer(ha, 0, data).unwrap();
        b.write_buffer(hb, 0, data).unwrap();
        let ra = a.read_buffer(ha, 0, data.len() as u64).unwrap();
        let rb = b.read_buffer(hb, 0, data.len() as u64).unwrap();
        assert_eq!(
            ra, rb,
            "compare_buffer_write pattern {i}: [{label_a}] != [{label_b}]"
        );
        a.destroy_buffer(ha);
        b.destroy_buffer(hb);
    }
}

/// Runs the full buffer suite on `a` and `b`, then compares every write/read pattern across them.
///
/// # Panics
///
/// Panics on failure or on any mismatch between `a` and `b`.
pub fn compare_full(label_a: &str, a: &dyn GpuDevice, label_b: &str, b: &dyn GpuDevice) {
    run_buffer_suite(label_a, a);
    run_buffer_suite(label_b, b);
    compare_buffer_write(label_a, a, label_b, b);
}

fn compute_desc(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::Upload,
    }
}

pub fn as_bytes_f32(s: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(s.as_ptr() as *const u8, mem::size_of_val(s)) }
}

pub fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Runs one single-dispatch compute op on `dev`. Creates a buffer per input (written with its data)
/// and per output size, dispatches `pipeline` with `Bindings.buffers = [inputs..., outputs...]` and
/// `scalars`, and reads back the outputs. Destroys its buffers. The caller destroys `pipeline` and
/// its shader.
pub fn run_dispatch(
    dev: &dyn GpuDevice,
    pipeline: PipelineHandle,
    inputs: &[&[u8]],
    output_sizes: &[u64],
    scalars: &[Scalar],
    grid: [u32; 3],
) -> Vec<Vec<u8>> {
    let in_handles: Vec<_> = inputs
        .iter()
        .map(|data| {
            let h = dev.create_buffer(compute_desc(data.len() as u64)).unwrap();
            dev.write_buffer(h, 0, data).unwrap();
            h
        })
        .collect();
    let out_handles: Vec<_> = output_sizes
        .iter()
        .map(|&size| dev.create_buffer(compute_desc(size)).unwrap())
        .collect();

    let buf_indices: Vec<u32> = in_handles
        .iter()
        .chain(out_handles.iter())
        .map(|h| h.index())
        .collect();
    dev.dispatch(
        pipeline,
        Bindings {
            buffers: &buf_indices,
            scalars,
            textures: &[],
            images: &[],
        },
        grid,
    )
    .unwrap();

    let outputs: Vec<Vec<u8>> = out_handles
        .iter()
        .zip(output_sizes)
        .map(|(&h, &size)| dev.read_buffer(h, 0, size).unwrap())
        .collect();

    for h in in_handles.into_iter().chain(out_handles) {
        dev.destroy_buffer(h);
    }
    outputs
}

/// Runs vec_add (`out[i] = a[i] + b[i]`) on `a` and `b`. The CPU oracle needs a registered
/// vector-add kernel.
///
/// # Panics
///
/// Panics if either result differs from the expected sum.
pub fn compare_vec_add(
    label_a: &str,
    a: &dyn GpuDevice,
    pipeline_a: PipelineHandle,
    label_b: &str,
    b: &dyn GpuDevice,
    pipeline_b: PipelineHandle,
) {
    const N: u32 = 1024;
    let size = (N as u64) * 4;
    let a_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..N).map(|i| 100.0 * i as f32).collect();
    let inputs: [&[u8]; 2] = [as_bytes_f32(&a_data), as_bytes_f32(&b_data)];
    let scalars = [Scalar::U32(N)];
    let grid = [N.div_ceil(256), 1, 1];

    let out_a = from_bytes_f32(&run_dispatch(a, pipeline_a, &inputs, &[size], &scalars, grid)[0]);
    let out_b = from_bytes_f32(&run_dispatch(b, pipeline_b, &inputs, &[size], &scalars, grid)[0]);
    for i in 0..N as usize {
        let expected = i as f32 + 100.0 * i as f32;
        assert!(
            (out_a[i] - expected).abs() < 1e-4,
            "[{label_a}] vec_add[{i}] = {}, expected {expected}",
            out_a[i]
        );
        assert!(
            (out_a[i] - out_b[i]).abs() < 1e-4,
            "compare_vec_add[{i}]: [{label_a}]={} != [{label_b}]={}",
            out_a[i],
            out_b[i]
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod cpu_vs_vulkan_tests {
    //! CPU oracle vs Vulkan. Skips without a Vulkan driver.

    use std::mem::size_of_val;
    use std::slice::from_raw_parts;

    use crate::{compare_full, compare_vec_add, run_buffer_suite, run_dispatch};
    use zengpu_cpu::CpuDevice;
    use zengpu_hal::{
        AdapterRequest, ComputePipelineDesc, DeviceRequest, GpuDevice, GpuInstance, Scalar,
        ShaderDesc,
    };
    use zengpu_spirv::{ZslShader, zsl};
    use zengpu_vulkan::VulkanInstance;

    const VEC_ADD_SHADER: ZslShader = zsl!(
        push P { len: u32 }
        @workgroup_size(256)
        kernel add(a: device buffer<f32>, b: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
            let i = id.x
            if i < p.len {
                out[i] = a[i] + b[i]
            }
        }
    );

    fn vec_add_spv_bytes() -> &'static [u8] {
        unsafe {
            from_raw_parts(
                VEC_ADD_SHADER.spv.as_ptr() as *const u8,
                size_of_val(VEC_ADD_SHADER.spv),
            )
        }
    }

    fn register_vec_add_kernel(dev: &CpuDevice, pipeline: zengpu_hal::PipelineHandle) {
        dev.register_kernel(
            pipeline,
            Box::new(|ctx| {
                let a: Vec<f32> = ctx.buffers[0]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let b: Vec<f32> = ctx.buffers[1]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let out: Vec<u8> = a
                    .iter()
                    .zip(&b)
                    .flat_map(|(x, y)| (x + y).to_le_bytes())
                    .collect();
                ctx.buffers[2][..out.len()].copy_from_slice(&out);
            }),
        );
    }

    fn cpu_device() -> Box<dyn zengpu_hal::GpuDevice> {
        Box::new(CpuDevice::new())
    }

    fn vulkan_device() -> Option<Box<dyn zengpu_hal::GpuDevice>> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    #[test]
    fn cpu_buffer_suite() {
        run_buffer_suite("cpu", &*cpu_device());
    }

    #[test]
    fn vulkan_buffer_suite() {
        let Some(dev) = vulkan_device() else { return };
        run_buffer_suite("vulkan", &*dev);
    }

    #[test]
    fn cpu_vs_vulkan() {
        let Some(vk) = vulkan_device() else { return };
        compare_full("cpu", &*cpu_device(), "vulkan", &*vk);
    }

    #[test]
    fn cpu_vs_vulkan_vec_add() {
        let Some(vk) = vulkan_device() else { return };

        let cpu = CpuDevice::new();
        let cpu_shader = cpu
            .create_shader(ShaderDesc::spirv(vec_add_spv_bytes()))
            .unwrap();
        let cpu_pipeline = cpu
            .create_compute_pipeline(ComputePipelineDesc {
                shader: cpu_shader,
                entry: "main",
                block: [256, 1, 1],
            })
            .unwrap();
        register_vec_add_kernel(&cpu, cpu_pipeline);

        let vk_shader = vk
            .create_shader(ShaderDesc::spirv(vec_add_spv_bytes()))
            .unwrap();
        let vk_pipeline = vk
            .create_compute_pipeline(ComputePipelineDesc {
                shader: vk_shader,
                entry: "main",
                block: [256, 1, 1],
            })
            .unwrap();

        compare_vec_add("cpu", &cpu, cpu_pipeline, "vulkan", &*vk, vk_pipeline);

        cpu.destroy_pipeline(cpu_pipeline);
        cpu.destroy_shader(cpu_shader);
        vk.destroy_pipeline(vk_pipeline);
        vk.destroy_shader(vk_shader);
    }

    const MIX64_C1_HI: u32 = 3210233709;
    const MIX64_C1_LO: u32 = 484763065;
    const MIX64_C2_HI: u32 = 2496678331;
    const MIX64_C2_LO: u32 = 321982955;

    const MIX64_SHADER: ZslShader = zsl!(
        push P { n: u32 }

        fn mulhi_u32(u: u32, v: u32, out result: u32) {
            let u0 = u & 65535
            let u1 = u >> 16
            let v0 = v & 65535
            let v1 = v >> 16

            let t = u0 * v0
            let k = t >> 16

            t = u1 * v0 + k
            let w1 = t & 65535
            let w2 = t >> 16

            t = u0 * v1 + w1
            k = t >> 16

            result = u1 * v1 + w2 + k
        }

        fn mul64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32, out r_hi: u32, out r_lo: u32) {
            let cross: u32 = 0
            mulhi_u32(a_lo, b_lo, cross)
            r_lo = a_lo * b_lo
            r_hi = a_hi * b_lo + a_lo * b_hi + cross
        }

        fn shr64(hi: u32, lo: u32, n: u32, out r_hi: u32, out r_lo: u32) {
            r_lo = (lo >> n) | (hi << (32 - n))
            r_hi = hi >> n
        }

        fn mix64(hi: u32, lo: u32, out out_hi: u32, out out_lo: u32) {
            let s1_hi: u32 = 0
            let s1_lo: u32 = 0
            shr64(hi, lo, 30, s1_hi, s1_lo)
            let v1_hi = hi ^ s1_hi
            let v1_lo = lo ^ s1_lo

            let m1_hi: u32 = 0
            let m1_lo: u32 = 0
            mul64(v1_hi, v1_lo, 3210233709, 484763065, m1_hi, m1_lo)

            let s2_hi: u32 = 0
            let s2_lo: u32 = 0
            shr64(m1_hi, m1_lo, 27, s2_hi, s2_lo)
            let v2_hi = m1_hi ^ s2_hi
            let v2_lo = m1_lo ^ s2_lo

            let m2_hi: u32 = 0
            let m2_lo: u32 = 0
            mul64(v2_hi, v2_lo, 2496678331, 321982955, m2_hi, m2_lo)

            let s3_hi: u32 = 0
            let s3_lo: u32 = 0
            shr64(m2_hi, m2_lo, 31, s3_hi, s3_lo)
            out_hi = m2_hi ^ s3_hi
            out_lo = m2_lo ^ s3_lo
        }

        @workgroup_size(64)
        kernel hash(
            in_hi: device buffer<u32>, in_lo: device buffer<u32>,
            out_hi: device mut buffer<u32>, out_lo: device mut buffer<u32>,
            p: P, id: global_id,
        ) {
            let i = id.x
            if i < p.n {
                let h: u32 = 0
                let l: u32 = 0
                mix64(in_hi[i], in_lo[i], h, l)
                out_hi[i] = h
                out_lo[i] = l
            }
        }
    );

    /// Plain Rust `xytron_mppi::core::noise::mix64`, copied (no dependency) as the
    /// independent reference.
    fn mix64_reference(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        debug_assert_eq!((0xbf58_476d_1ce4_e5b9u64 >> 32) as u32, MIX64_C1_HI);
        debug_assert_eq!(0xbf58_476d_1ce4_e5b9u64 as u32, MIX64_C1_LO);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        debug_assert_eq!((0x94d0_49bb_1331_11ebu64 >> 32) as u32, MIX64_C2_HI);
        debug_assert_eq!(0x94d0_49bb_1331_11ebu64 as u32, MIX64_C2_LO);
        value ^ (value >> 31)
    }

    fn mix64_spv_bytes() -> &'static [u8] {
        unsafe {
            from_raw_parts(
                MIX64_SHADER.spv.as_ptr() as *const u8,
                size_of_val(MIX64_SHADER.spv),
            )
        }
    }

    #[test]
    fn mix64_matches_rust_reference_on_vulkan() {
        let Some(vk) = vulkan_device() else { return };

        let shader = vk
            .create_shader(ShaderDesc::spirv(mix64_spv_bytes()))
            .unwrap();
        let pipeline = vk
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [64, 1, 1],
            })
            .unwrap();

        let inputs: Vec<u64> = vec![
            0,
            1,
            u64::MAX,
            0x9e37_79b9_7f4a_7c15,
            0x1234_5678_9abc_def0,
            0xffff_ffff_0000_0000,
            0x0000_0000_ffff_ffff,
            0x8000_0000_0000_0001,
            123456789,
            u64::MAX - 1,
        ];
        let in_hi: Vec<u32> = inputs.iter().map(|v| (v >> 32) as u32).collect();
        let in_lo: Vec<u32> = inputs.iter().map(|v| *v as u32).collect();
        let n = inputs.len() as u32;

        let as_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let in_hi_bytes = as_bytes(&in_hi);
        let in_lo_bytes = as_bytes(&in_lo);

        let outputs = run_dispatch(
            &*vk,
            pipeline,
            &[&in_hi_bytes, &in_lo_bytes],
            &[(n as u64) * 4, (n as u64) * 4],
            &[Scalar::U32(n)],
            [n.div_ceil(64), 1, 1],
        );
        let out_hi: Vec<u32> = outputs[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let out_lo: Vec<u32> = outputs[1]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();

        for (i, &value) in inputs.iter().enumerate() {
            let expected = mix64_reference(value);
            let actual = ((out_hi[i] as u64) << 32) | out_lo[i] as u64;
            assert_eq!(
                actual, expected,
                "mix64({value:#x}): GPU={actual:#x}, expected={expected:#x}"
            );
        }

        vk.destroy_pipeline(pipeline);
        vk.destroy_shader(shader);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod devarray_elementwise_tests {
    //! Cross-backend conformance for elementwise ops (device-array add and ReLU). Skips
    //! without Vulkan.

    use std::sync::Arc;

    use crate::{as_bytes_f32, from_bytes_f32};
    use zengpu_compute::BufferPool;
    use zengpu_compute::elementwise::ElementwiseKernels;
    use zengpu_cpu::{CpuDevice, CpuKernelCtx};
    use zengpu_hal::{AdapterRequest, DType, DeviceRequest, GpuDevice, GpuInstance, Scalar};
    use zengpu_vulkan::VulkanInstance;

    fn vulkan_device() -> Option<Arc<dyn GpuDevice>> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
    }

    fn read_f32(buf: &[u8], i: usize) -> f32 {
        f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
    }

    fn write_f32(buf: &mut [u8], i: usize, v: f32) {
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn register_cpu_kernels(cpu: &CpuDevice, kernels: &ElementwiseKernels) {
        cpu.register_kernel(
            kernels.add_pipeline,
            Box::new(|ctx: &mut CpuKernelCtx| {
                let len = match ctx.scalars.first() {
                    Some(&Scalar::U32(n)) => n as usize,
                    _ => return,
                };
                let a: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[0], i)).collect();
                let b: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[1], i)).collect();
                for i in 0..len {
                    write_f32(&mut ctx.buffers[2], i, a[i] + b[i]);
                }
            }),
        );
        cpu.register_kernel(
            kernels.relu_pipeline,
            Box::new(|ctx: &mut CpuKernelCtx| {
                let len = match ctx.scalars.first() {
                    Some(&Scalar::U32(n)) => n as usize,
                    _ => return,
                };
                for i in 0..len {
                    let v = read_f32(&ctx.buffers[0], i);
                    write_f32(&mut ctx.buffers[1], i, v.max(0.0));
                }
            }),
        );
    }

    #[test]
    fn devarray_add_relu_cpu_vs_vulkan() {
        let Some(vk) = vulkan_device() else { return };
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());

        let cpu_kernels = ElementwiseKernels::new(&*cpu).unwrap();
        register_cpu_kernels(
            cpu.as_any().downcast_ref::<CpuDevice>().unwrap(),
            &cpu_kernels,
        );
        let vk_kernels = ElementwiseKernels::new(&*vk).unwrap();

        let cpu_pool = BufferPool::new(cpu.clone());
        let vk_pool = BufferPool::new(vk.clone());

        const N: u32 = 256;
        let shape = vec![N];
        let a_data: Vec<f32> = (0..N).map(|i| i as f32 - 128.0).collect();
        let b_data: Vec<f32> = (0..N).map(|i| 100.0 - i as f32).collect();

        let mut results: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (label, dev, pool, kernels) in [
            ("cpu", &cpu, &cpu_pool, &cpu_kernels),
            ("vulkan", &vk, &vk_pool, &vk_kernels),
        ] {
            let a = pool.alloc(shape.clone(), DType::F32).unwrap();
            let b = pool.alloc(shape.clone(), DType::F32).unwrap();
            dev.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))
                .unwrap();
            dev.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))
                .unwrap();

            let sum = kernels.add(&**dev, pool, &a, &b).unwrap();
            let relu_out = kernels.relu(&**dev, pool, &sum).unwrap();

            let sum_bytes = dev.read_buffer(sum.buffer, 0, sum.size_bytes()).unwrap();
            let relu_bytes = dev
                .read_buffer(relu_out.buffer, 0, relu_out.size_bytes())
                .unwrap();
            let sum_out = from_bytes_f32(&sum_bytes);
            let relu_result = from_bytes_f32(&relu_bytes);

            for i in 0..N as usize {
                let expected_sum = a_data[i] + b_data[i];
                assert!(
                    (sum_out[i] - expected_sum).abs() < 1e-4,
                    "[{label}] add[{i}] = {}, expected {expected_sum}",
                    sum_out[i]
                );
                let expected_relu = expected_sum.max(0.0);
                assert!(
                    (relu_result[i] - expected_relu).abs() < 1e-4,
                    "[{label}] relu[{i}] = {}, expected {expected_relu}",
                    relu_result[i]
                );
            }

            pool.free(a);
            pool.free(b);
            pool.free(sum);
            pool.free(relu_out);
            results.push((sum_out, relu_result));
        }

        let (cpu_sum, cpu_relu) = &results[0];
        let (vk_sum, vk_relu) = &results[1];
        for i in 0..N as usize {
            assert!(
                (cpu_sum[i] - vk_sum[i]).abs() < 1e-4,
                "add[{i}]: cpu={} != vulkan={}",
                cpu_sum[i],
                vk_sum[i]
            );
            assert!(
                (cpu_relu[i] - vk_relu[i]).abs() < 1e-4,
                "relu[{i}]: cpu={} != vulkan={}",
                cpu_relu[i],
                vk_relu[i]
            );
        }

        cpu_kernels.destroy(&*cpu);
        vk_kernels.destroy(&*vk);
    }

    /// A failed dispatch returns its output. No CPU kernel is registered, so each op fails and the
    /// buffer comes back.
    #[test]
    fn failed_ops_give_their_output_back() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let kernels = ElementwiseKernels::new(&*cpu).unwrap();
        let pool = BufferPool::new(cpu.clone());

        let input = pool.alloc(vec![64], DType::F32).unwrap();
        let spare = pool.alloc(vec![64], DType::F32).unwrap();
        let pooled = spare.buffer;
        pool.free(spare);
        for _ in 0..10 {
            assert!(kernels.relu(&*cpu, &pool, &input).is_err());
            assert!(kernels.add(&*cpu, &pool, &input, &input).is_err());
        }
        let again = pool.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(
            again.buffer, pooled,
            "the pooled buffer was not kept by a failure"
        );

        pool.free(again);
        pool.free(input);
        kernels.destroy(&*cpu);
    }

    #[test]
    fn freeing_a_cloned_array_twice_cannot_alias_later_allocations() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let pool = BufferPool::new(cpu.clone());
        let array = pool.alloc(vec![64], DType::F32).unwrap();
        pool.free(array.clone());
        pool.free(array);
        let first = pool.alloc(vec![64], DType::F32).unwrap();
        let second = pool.alloc(vec![64], DType::F32).unwrap();
        assert_ne!(first.buffer, second.buffer);
        cpu.write_buffer(first.buffer, 0, &[1; 256]).unwrap();
        cpu.write_buffer(second.buffer, 0, &[2; 256]).unwrap();
        assert_eq!(cpu.read_buffer(first.buffer, 0, 256).unwrap(), [1; 256]);
        pool.free(first);
        pool.free(second);
    }

    #[test]
    fn an_old_array_alias_cannot_free_a_reused_allocation() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let pool = BufferPool::new(cpu);
        let array = pool.alloc(vec![64], DType::F32).unwrap();
        let old = array.clone();
        pool.free(array);
        let current = pool.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(old.buffer, current.buffer, "the buffer was reused");
        pool.free(old);
        let other = pool.alloc(vec![64], DType::F32).unwrap();
        assert_ne!(current.buffer, other.buffer);
        pool.free(current);
        pool.free(other);
    }

    #[test]
    fn pools_reject_foreign_and_manually_reconstructed_arrays() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let owner = BufferPool::new(cpu.clone());
        let other = BufferPool::new(cpu);
        let array = owner.alloc(vec![64], DType::F32).unwrap();
        assert!(other.try_free(array.clone()).is_err());
        let reconstructed =
            zengpu_compute::DeviceArray::new(array.buffer, array.shape.clone(), array.dtype);
        assert!(owner.try_free(reconstructed).is_err());
        let handle = array.buffer;
        owner.try_free(array).unwrap();
        let reused = owner.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(
            reused.buffer, handle,
            "failed returns did not lose the allocation"
        );
        owner.try_free(reused).unwrap();
    }

    #[test]
    fn typed_array_ops_reject_foreign_stale_and_undersized_buffers() {
        let local: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let foreign: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let kernels = ElementwiseKernels::new(&*local).unwrap();
        let pool = BufferPool::new(local.clone());
        let foreign_pool = BufferPool::new(foreign);
        let wrong_device = foreign_pool.alloc(vec![64], DType::F32).unwrap();
        let mut ops = zengpu_compute::OpList::new(&pool);
        assert!(kernels.record_relu(&mut ops, &wrong_device).is_err());
        assert!(ops.is_empty(), "invalid inputs must not record a dispatch");
        let stale = pool.alloc(vec![64], DType::F32).unwrap();
        local.destroy_buffer(stale.buffer);
        let replacement = pool.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(stale.buffer.index(), replacement.buffer.index());
        assert!(kernels.record_relu(&mut ops, &stale).is_err());
        let mut too_small = replacement.clone();
        too_small.shape = vec![128];
        assert!(
            kernels
                .record_add(&mut ops, &too_small, &too_small)
                .is_err()
        );
        let mut strided = replacement.clone();
        strided.stride = vec![2];
        assert!(kernels.record_relu(&mut ops, &strided).is_err());
        pool.free(replacement);
        foreign_pool.free(wrong_device);
        kernels.destroy(&*local);
    }

    #[test]
    fn recorded_arrays_are_checked_again_before_dispatch() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let kernels = ElementwiseKernels::new(&*cpu).unwrap();
        register_cpu_kernels(cpu.as_any().downcast_ref::<CpuDevice>().unwrap(), &kernels);
        let pool = BufferPool::new(cpu.clone());
        let input = pool.alloc(vec![64], DType::F32).unwrap();
        let mut ops = zengpu_compute::OpList::new(&pool);
        let output = kernels.record_relu(&mut ops, &input).unwrap();
        cpu.destroy_buffer(input.buffer);
        let replacement = pool.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(replacement.buffer.index(), input.buffer.index());
        assert!(
            ops.run(&*cpu).is_err(),
            "slot reuse must not hide a stale generation"
        );
        assert!(ops.submit(&*cpu, 7).is_err());
        ops.clear();
        pool.free(replacement);
        pool.free(output);
        kernels.destroy(&*cpu);
    }

    #[test]
    fn an_operation_list_cannot_run_on_another_device_with_matching_slots() {
        let first: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let second: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let pool = BufferPool::new(first);
        let other = BufferPool::new(second.clone());
        let kernels = ElementwiseKernels::new(&*second).unwrap();
        register_cpu_kernels(
            second.as_any().downcast_ref::<CpuDevice>().unwrap(),
            &kernels,
        );
        let input = pool.alloc(vec![64], DType::F32).unwrap();
        let canary = other.alloc(vec![64], DType::F32).unwrap();
        let target = other.alloc(vec![64], DType::F32).unwrap();
        second.write_buffer(target.buffer, 0, &[9; 256]).unwrap();
        let mut ops = zengpu_compute::OpList::new(&pool);
        let output = kernels.record_relu(&mut ops, &input).unwrap();
        assert_eq!(input.buffer.index(), canary.buffer.index());
        assert_eq!(output.buffer.index(), target.buffer.index());
        assert!(ops.run(&*second).is_err());
        assert!(ops.submit(&*second, 1).is_err());
        assert_eq!(second.read_buffer(target.buffer, 0, 256).unwrap(), [9; 256]);
        pool.free(input);
        pool.free(output);
        other.free(canary);
        other.free(target);
        kernels.destroy(&*second);
    }

    #[test]
    fn failed_recording_rolls_back_its_dispatches_and_scratch() {
        let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());
        let pool = BufferPool::new(cpu.clone());
        let kernels = ElementwiseKernels::new(&*cpu).unwrap();
        let spare = pool.alloc(vec![64], DType::F32).unwrap();
        let handle = spare.buffer;
        pool.free(spare);
        let mut ops = zengpu_compute::OpList::new(&pool);
        let failed: zengpu_hal::Result<()> = ops.transaction(|ops| {
            let slot = ops.scratch(vec![64], DType::F32)?;
            ops.dispatch(
                kernels.relu_pipeline,
                &[slot, slot],
                &[Scalar::U32(64)],
                [1, 1, 1],
            );
            Err(zengpu_hal::GpuError::OutOfMemory(
                zengpu_hal::MemoryUsage::GpuOnly,
            ))
        });
        assert!(failed.is_err());
        assert!(ops.is_empty());
        let reused = pool.alloc(vec![64], DType::F32).unwrap();
        assert_eq!(reused.buffer, handle);
        cpu.destroy_buffer(reused.buffer);
        assert!(
            ops.run(&*cpu).is_ok(),
            "failed buffer requirements were also removed"
        );
        kernels.destroy(&*cpu);
    }
}

#[cfg(test)]
mod half_buffer_tests {
    //! ZSL `buffer<f16>`/`buffer<bf16>` on Vulkan (two halves per u32 word). Skips without Vulkan.

    use zengpu_hal::{
        AdapterRequest, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, DeviceRequest,
        GpuInstance, MemoryUsage, Scalar,
    };
    use zengpu_spirv::{ZslShader, zsl};
    use zengpu_vulkan::VulkanInstance;

    const HALF_IO_ZSL: ZslShader = zsl!(
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

    /// Binary16 bits of `x`, for values exactly representable as normal halves.
    fn f16_bits(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as u16 - 112;
        sign | (exp << 10) | ((b & 0x007f_ffff) >> 13) as u16
    }

    fn f16_value(h: u16) -> f32 {
        if h & 0x7fff == 0 {
            return 0.0;
        }
        let sign = u32::from(h & 0x8000) << 16;
        let exp = u32::from((h >> 10) & 0x1f) + 112;
        f32::from_bits(sign | (exp << 23) | (u32::from(h & 0x3ff) << 13))
    }

    #[test]
    fn vulkan_half_buffers_load_and_store() {
        let Some(vk) = VulkanInstance::new().ok().and_then(|inst| {
            inst.request_adapter(AdapterRequest::default())?
                .open(DeviceRequest::default())
                .ok()
        }) else {
            return;
        };
        let (shader, entry) = HALF_IO_ZSL.create(&*vk).unwrap();
        let pipeline = vk
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .unwrap();
        let n = 1001u32;
        let x: Vec<f32> = (0..n).map(|i| ((i % 256) as f32 - 128.0) * 0.25).collect();
        let xh: Vec<u8> = x.iter().flat_map(|&v| f16_bits(v).to_le_bytes()).collect();
        let xb: Vec<u8> = x
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let buffer = |size: u64| {
            vk.create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap()
        };
        let half_bytes = u64::from(n) * 2;
        let bufs = [
            buffer(half_bytes),
            buffer(half_bytes),
            buffer(half_bytes),
            buffer(half_bytes),
            buffer(u64::from(n) * 4),
        ];
        vk.write_buffer(bufs[0], 0, &xh).unwrap();
        vk.write_buffer(bufs[1], 0, &xb).unwrap();
        let slots: Vec<u32> = bufs.iter().map(|b| b.index()).collect();
        vk.dispatch(
            pipeline,
            Bindings {
                buffers: &slots,
                scalars: &[Scalar::U32(n)],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(64), 1, 1],
        )
        .unwrap();
        let halves = |i: usize| -> Vec<u16> {
            vk.read_buffer(bufs[i], 0, half_bytes)
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let (yh, yb) = (halves(2), halves(3));
        let sum = crate::from_bytes_f32(&vk.read_buffer(bufs[4], 0, u64::from(n) * 4).unwrap());
        for i in 0..n as usize {
            let want = 2.0 * x[i];
            assert_eq!(f16_value(yh[i]), want, "f16 at {i}");
            assert_eq!(f32::from_bits(u32::from(yb[i]) << 16), want, "bf16 at {i}");
            assert_eq!(sum[i], want, "sum at {i}");
        }
        for b in bufs {
            vk.destroy_buffer(b);
        }
        vk.destroy_pipeline(pipeline);
        vk.destroy_shader(shader);
    }
}
