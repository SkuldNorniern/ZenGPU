//! Timestamp query pools on Metal's counter sample buffers.
//!
//! Apple GPUs sample counters only where an encoder starts or ends, not between draws or
//! dispatches. So a timestamp outside a render pass is taken at the start of an empty blit
//! encoder put there for it, and one written inside a pass is taken when that pass ends.

use std::slice::from_raw_parts;
use std::thread::sleep;
use std::time::{Duration, Instant};

use metal::{
    BlitPassDescriptor, CommandBuffer, CommandBufferRef, CounterSampleBuffer,
    CounterSampleBufferDescriptor, CounterSet, DeviceRef, MTLCommandBufferStatus,
    MTLCounterSamplingPoint, MTLResourceOptions, MTLStorageMode, NSRange,
};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};

/// No sample at this point of the encoder (`MTLCounterDontSample`).
const DONT_SAMPLE: u64 = u64::MAX;

/// One timestamp pool: the samples, and the command buffer that last wrote each.
pub(crate) struct MetalQueryPool {
    pub(crate) samples: CounterSampleBuffer,
    pub(crate) writers: Vec<Option<CommandBuffer>>,
}

/// The timestamp counter set, when the GPU samples it at encoder boundaries.
pub(crate) fn timestamp_set(device: &DeviceRef) -> Option<CounterSet> {
    if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    device
        .counter_sets()
        .into_iter()
        .find(|set| set.name().eq_ignore_ascii_case("timestamp"))
}

pub(crate) fn new_pool(device: &DeviceRef, count: u32) -> Result<MetalQueryPool, String> {
    let set = timestamp_set(device).ok_or("this gpu samples no timestamps")?;
    let desc = CounterSampleBufferDescriptor::new();
    desc.set_counter_set(&set);
    desc.set_sample_count(u64::from(count));
    desc.set_storage_mode(MTLStorageMode::Shared);
    let samples = device.new_counter_sample_buffer_with_descriptor(&desc)?;
    Ok(MetalQueryPool {
        samples,
        writers: vec![None; count as usize],
    })
}

/// Takes timestamp `index` at the start of a blit encoder in `cmd`: after the work encoded before
/// it. The encoder fills four scratch bytes: Metal drops an empty encoder, sample and all.
pub(crate) fn sample(
    device: &DeviceRef,
    cmd: &CommandBufferRef,
    samples: &CounterSampleBuffer,
    index: u32,
) {
    let pass = BlitPassDescriptor::new();
    let Some(attachment) = pass.sample_buffer_attachments().object_at(0) else {
        return;
    };
    attachment.set_sample_buffer(samples);
    attachment.set_start_of_encoder_sample_index(u64::from(index));
    attachment.set_end_of_encoder_sample_index(DONT_SAMPLE);
    let scratch = device.new_buffer(4, MTLResourceOptions::StorageModePrivate);
    let enc = cmd.blit_command_encoder_with_descriptor(pass);
    enc.fill_buffer(&scratch, NSRange::new(0, 4), 0);
    enc.end_encoding();
}

/// Every sample in `pool` in GPU ticks, or `None` while any was not written by finished work.
pub(crate) fn read(pool: &MetalQueryPool) -> Option<Vec<u64>> {
    let done = pool.writers.iter().all(|w| {
        w.as_ref()
            .is_some_and(|cmd| cmd.status() == MTLCommandBufferStatus::Completed)
    });
    if !done {
        return None;
    }
    let count = pool.writers.len() as u64;
    let range = NSRange::new(0, count);
    // SAFETY: resolveCounterRange: returns an autoreleased NSData of `count` u64 timestamps, or
    // nil; the bytes are copied out before the pool drains.
    autoreleasepool(|| unsafe {
        let data: *mut Object = msg_send![&*pool.samples, resolveCounterRange: range];
        if data.is_null() {
            return None;
        }
        let len: usize = msg_send![data, length];
        let bytes: *const u8 = msg_send![data, bytes];
        if bytes.is_null() || len < count as usize * 8 {
            return None;
        }
        let raw = from_raw_parts(bytes, count as usize * 8);
        Some(
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_ne_bytes(*c))
                .collect(),
        )
    })
}

/// Nanoseconds a GPU timestamp tick lasts, from the GPU clock against the CPU's over a few ms.
pub(crate) fn period_ns(device: &DeviceRef) -> f32 {
    let (mut cpu0, mut gpu0, mut cpu1, mut gpu1) = (0, 0, 0, 0);
    device.sample_timestamps(&mut cpu0, &mut gpu0);
    let start = Instant::now();
    sleep(Duration::from_millis(3));
    device.sample_timestamps(&mut cpu1, &mut gpu1);
    let elapsed = start.elapsed().as_nanos() as f64;
    let ticks = gpu1.saturating_sub(gpu0);
    if ticks == 0 {
        return 1.0;
    }
    (elapsed / ticks as f64) as f32
}
