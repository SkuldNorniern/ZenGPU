//! ZenGPU conformance harness.
//!
//! Each `run_*_suite` function exercises a set of operations on any
//! [`GpuDevice`].  Call them on the CPU oracle to prove the tests are
//! correct, then on every GPU backend to prove the backend is correct.
//!
//! The cross-backend `compare_*` functions run the same operation on two
//! devices and assert byte-identical results — the CPU oracle is always one
//! of the two.

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

// ── Per-device suite ──────────────────────────────────────────────────────────

/// Run the full buffer conformance suite on `dev`.  Panics on any failure,
/// printing `label` in the message so failures identify the backend.
pub fn run_buffer_suite(label: &str, dev: &dyn GpuDevice) {
    buffer_roundtrip(label, dev);
    buffer_partial_read(label, dev);
    buffer_zero_init(label, dev);
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

fn buffer_copy_ranges(label: &str, dev: &dyn GpuDevice) {
    let src = dev.create_buffer(rw_desc(16)).unwrap();
    let dst = dev.create_buffer(rw_desc(16)).unwrap();
    dev.write_buffer(src, 0, &(0u8..16).collect::<Vec<_>>())
        .unwrap();
    dev.copy_buffer(src, 3, dst, 7, 6).unwrap();
    assert_eq!(
        dev.read_buffer(dst, 0, 16).unwrap(),
        [0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 0, 0, 0],
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

// ── Cross-backend comparison ──────────────────────────────────────────────────

/// Run a write→read cycle on both `a` and `b` with the same data and assert
/// byte-identical results. `a` should be the CPU oracle.
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

/// Run the full buffer suite on both `a` and `b`, then compare results across
/// backends for every write→read pattern.
pub fn compare_full(label_a: &str, a: &dyn GpuDevice, label_b: &str, b: &dyn GpuDevice) {
    run_buffer_suite(label_a, a);
    run_buffer_suite(label_b, b);
    compare_buffer_write(label_a, a, label_b, b);
}

// ── Compute ─────────────────────────────────────────────────────────────────

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
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run a single-dispatch compute op on `dev`: create one buffer per `inputs`
/// slice (written with that data) and one buffer per `output_sizes` entry,
/// dispatch `pipeline` with `Bindings.buffers = [input indices..., output
/// indices...]` and `scalars`, then read back and return the output buffers'
/// bytes. All buffers are destroyed before returning; `pipeline`/its shader
/// are left for the caller to destroy.
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

/// Run vec_add (`out[i] = a[i] + b[i]`) on both `a` and `b` and assert the
/// results agree and match the expected sum. Each device must already have a
/// pipeline and, for the CPU oracle, a registered kernel implementing vector
/// addition.
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
