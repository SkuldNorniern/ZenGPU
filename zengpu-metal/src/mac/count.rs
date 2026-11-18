//! Count driven indirect draws. Metal has no draw count read on the GPU, so a small kernel copies
//! the draw arguments into a scratch buffer first and zeroes the instance count of every draw at
//! or past the count; the pass then issues `max_draws` plain indirect draws from the scratch.
//!
//! The kernel runs in a command buffer committed just before the list's own, so the count is read
//! when the list starts on the GPU. A draw earlier in the same list writing the count is not seen.

use std::sync::OnceLock;

use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineState, DeviceRef, MTLResourceOptions,
    MTLSize,
};

/// Draw arguments are `u32` words: four for a draw, five for an indexed draw, as on Vulkan.
pub(crate) const DRAW_WORDS: u32 = 4;
pub(crate) const INDEXED_WORDS: u32 = 5;

const CLAMP_MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct Clamp { uint offset; uint stride; uint words; uint count_at; uint max_draws; };
kernel void zengpu_clamp_draws(device const uint* src [[buffer(0)]],
                               device const uint* count [[buffer(1)]],
                               device uint* dst [[buffer(2)]],
                               constant Clamp& c [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= c.max_draws) return;
    uint n = min(count[c.count_at], c.max_draws);
    uint from = (c.offset + i * c.stride) / 4;
    for (uint w = 0; w < c.words; w++) dst[i * c.words + w] = src[from + w];
    if (i >= n) dst[i * c.words + 1] = 0;
}
";

/// What one count driven draw reads, checked against the buffers' sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Clamp {
    pub(crate) offset: u64,
    pub(crate) stride: u64,
    pub(crate) words: u32,
    pub(crate) count_offset: u64,
    pub(crate) max_draws: u32,
}

impl Clamp {
    /// Why a draw would read past a buffer or off a word, if it would.
    pub(crate) fn check(&self, args_size: u64, count_size: u64) -> Result<(), String> {
        let record = u64::from(self.words) * 4;
        let aligned = |v: u64| v.is_multiple_of(4);
        if !aligned(self.offset) || !aligned(self.stride) || !aligned(self.count_offset) {
            return Err("offsets and stride must be 4 aligned".into());
        }
        if self.stride < record {
            return Err(format!(
                "stride {} is under one record of {record} bytes",
                self.stride
            ));
        }
        if self
            .count_offset
            .checked_add(4)
            .is_none_or(|end| end > count_size)
        {
            return Err(format!(
                "count at {} is past its {count_size} bytes",
                self.count_offset
            ));
        }
        let last = u64::from(self.max_draws.saturating_sub(1)) * self.stride + self.offset + record;
        if self.max_draws > 0 && last > args_size {
            return Err(format!(
                "{} draws need {last} bytes of {args_size}",
                self.max_draws
            ));
        }
        Ok(())
    }

    /// Encodes the copy into `cmd` and returns the scratch the draws read, `words` apart.
    pub(crate) fn encode(
        &self,
        device: &DeviceRef,
        pipeline: &ComputePipelineState,
        cmd: &CommandBufferRef,
        args: &Buffer,
        count: &Buffer,
    ) -> Buffer {
        let size = u64::from(self.max_draws) * u64::from(self.words) * 4;
        let scratch = device.new_buffer(size.max(4), MTLResourceOptions::StorageModePrivate);
        let params: [u32; 5] = [
            u32::try_from(self.offset).unwrap_or(u32::MAX),
            u32::try_from(self.stride).unwrap_or(u32::MAX),
            self.words,
            u32::try_from(self.count_offset / 4).unwrap_or(u32::MAX),
            self.max_draws,
        ];
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(args), 0);
        enc.set_buffer(1, Some(count), 0);
        enc.set_buffer(2, Some(&scratch), 0);
        enc.set_bytes(3, size_of_val(&params) as u64, params.as_ptr().cast());
        let groups = u64::from(self.max_draws.div_ceil(64));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        scratch
    }
}

/// The clamp kernel, built once per device.
pub(crate) fn pipeline(
    cell: &OnceLock<Option<ComputePipelineState>>,
    device: &DeviceRef,
) -> Option<ComputePipelineState> {
    cell.get_or_init(|| {
        let library = device
            .new_library_with_source(CLAMP_MSL, &CompileOptions::new())
            .map_err(|e| log::error!("[zengpu-metal] draw count kernel: {e}"))
            .ok()?;
        let function = library.get_function("zengpu_clamp_draws", None).ok()?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| log::error!("[zengpu-metal] draw count kernel: {e}"))
            .ok()
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clamp(offset: u64, stride: u64, count_offset: u64, max_draws: u32) -> Clamp {
        Clamp {
            offset,
            stride,
            words: DRAW_WORDS,
            count_offset,
            max_draws,
        }
    }

    #[test]
    fn draws_inside_both_buffers_pass() {
        assert_eq!(clamp(0, 16, 0, 4).check(64, 4), Ok(()));
        assert_eq!(clamp(16, 32, 4, 2).check(16 + 32 + 16, 8), Ok(()));
        assert_eq!(
            clamp(0, 16, 0, 0).check(0, 4),
            Ok(()),
            "no draws read no arguments"
        );
    }

    #[test]
    fn reading_past_or_off_a_word_is_refused() {
        assert!(
            clamp(0, 16, 0, 5).check(64, 4).is_err(),
            "a fifth draw is past the end"
        );
        assert!(
            clamp(0, 16, 4, 1).check(64, 4).is_err(),
            "count past its buffer"
        );
        assert!(clamp(2, 16, 0, 1).check(64, 4).is_err(), "unaligned offset");
        assert!(
            clamp(0, 12, 0, 1).check(64, 4).is_err(),
            "stride under a record"
        );
        let indexed = Clamp {
            words: INDEXED_WORDS,
            ..clamp(0, 16, 0, 1)
        };
        assert!(
            indexed.check(64, 4).is_err(),
            "an indexed record is 20 bytes"
        );
    }
}
