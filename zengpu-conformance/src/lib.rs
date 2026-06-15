//! ZenGPU conformance harness.
//!
//! Each `run_*_suite` function runs a set of ops on any [`GpuDevice`]. Run them on the CPU oracle
//! first, then on each GPU backend.
//!
//! `compare_*` runs one op on two devices and asserts byte-identical results. One side is always
//! the CPU oracle.

use zengpu_hal::{
    Bindings, BufferDesc, BufferUsage, GpuDevice, GpuError, MemoryUsage, PipelineHandle, Scalar,
    UsageError,
};

fn rw_desc(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
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
    buffer_zero_init(label, dev);
    buffer_missing_readback_usage(label, dev);
    buffer_stale_after_destroy(label, dev);
    buffer_out_of_bounds_write(label, dev);
    buffer_multiple_allocs(label, dev);
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
    dev.write_buffer(h, 0, &[10, 20, 30, 40, 50, 60, 70, 80]).unwrap();
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

fn buffer_zero_init(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(16)).unwrap();
    let out = dev.read_buffer(h, 0, 16).unwrap();
    assert_eq!(out, vec![0u8; 16], "[{label}] buffer_zero_init");
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
            GpuError::InvalidUsage(UsageError::MissingUsage { needed: "READBACK", .. })
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
        matches!(
            err,
            GpuError::InvalidUsage(UsageError::StaleHandle { .. })
        ),
        "[{label}] stale-after-destroy: expected StaleHandle, got {err}"
    );
}

fn buffer_out_of_bounds_write(label: &str, dev: &dyn GpuDevice) {
    let h = dev.create_buffer(rw_desc(4)).unwrap();
    let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
    assert!(
        matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ),
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

/// Write, then read, on `a` and `b` with the same data. Asserts byte-identical results. `a` is the
/// CPU oracle.
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
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
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

/// Runs vec_add (`out[i] = a[i] + b[i]`) on `a` and `b`, asserts both match the expected sum. Each
/// device needs a vec_add pipeline. The CPU oracle also needs a registered kernel.
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
