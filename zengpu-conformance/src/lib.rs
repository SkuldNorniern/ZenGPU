//! ZenGPU conformance harness (plan D7 / M1.5).
//!
//! Each `run_*_suite` function exercises a set of operations on any
//! [`GpuDevice`].  Call them on the CPU oracle to prove the tests are
//! correct, then on every GPU backend to prove the backend is correct.
//!
//! The cross-backend `compare_*` functions run the same operation on two
//! devices and assert byte-identical results — the CPU oracle is always one
//! of the two (plan §18).

use zengpu_hal::{BufferDesc, BufferUsage, GpuDevice, GpuError, MemoryUsage, UsageError};

fn rw_desc(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
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

// ── Cross-backend comparison ──────────────────────────────────────────────────

/// Run a write→read cycle on both `a` and `b` with the same data and assert
/// byte-identical results.  `a` should be the CPU oracle (plan §18).
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
