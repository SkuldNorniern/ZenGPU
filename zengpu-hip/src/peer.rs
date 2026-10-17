//! Cross-GPU copies when the devices cannot reach each other directly.
//!
//! Without PCIe peer-to-peer a copy goes through host memory: device to host
//! on the source GPU's link, host to device on the destination's. Done one
//! after the other that is ~3.1 GB/s on two RX 9060 XTs; the two links are
//! independent, so chunks are pipelined through two pinned slots (the source
//! fills one while the destination drains the other), reaching ~5.9 GB/s.

use std::ptr::null_mut;

use zengpu_hal::Result;

use crate::ffi::*;
use crate::{HIP_EVENT_DISABLE_TIMING, HIP_SUCCESS, HipBuffer, HipCore};
use crate::{HIP_MEMCPY_D2H, HIP_MEMCPY_H2D};

/// Bytes per pinned slot.
pub(crate) const CHUNK: usize = 4 << 20;

/// `hipHostMallocPortable`: pinned for every device, not just the current one.
const HIP_HOST_MALLOC_PORTABLE: u32 = 1;

/// Two pinned host slots, kept by the destination device.
pub(crate) struct PeerStaging {
    slots: [*mut u8; 2],
}

// SAFETY: the slots are owned here and used only under the owning device's
// peer_staging mutex.
unsafe impl Send for PeerStaging {}

impl PeerStaging {
    pub(crate) fn new() -> Option<Self> {
        let mut slots = [null_mut::<u8>(); 2];
        for i in 0..2 {
            let mut ptr = null_mut();
            if unsafe { hipHostMalloc(&mut ptr, CHUNK, HIP_HOST_MALLOC_PORTABLE) } != HIP_SUCCESS {
                for slot in &slots[..i] {
                    unsafe { hipHostFree(slot.cast()) };
                }
                return None;
            }
            slots[i] = ptr.cast();
        }
        Some(Self { slots })
    }

    /// Copy `bytes` from `src_buf` on `src` to `dst_buf` on `dst`, pipelined
    /// across both devices' streams. Both streams must be idle on entry;
    /// both are idle again on return, whatever the outcome.
    pub(crate) fn copy(
        &self,
        dst: &HipCore,
        dst_buf: HipBuffer,
        src: &HipCore,
        src_buf: HipBuffer,
        bytes: usize,
    ) -> Result<()> {
        // An event must be recorded on a stream of the device it was made on.
        let mut d2h = [null_mut(); 2];
        let mut h2d = [null_mut(); 2];
        let made = (|| {
            src.activate()?;
            for event in &mut d2h {
                unsafe {
                    src.check(
                        hipEventCreateWithFlags(event, HIP_EVENT_DISABLE_TIMING),
                        "hipEventCreateWithFlags",
                    )?
                };
            }
            dst.activate()?;
            for event in &mut h2d {
                unsafe {
                    dst.check(
                        hipEventCreateWithFlags(event, HIP_EVENT_DISABLE_TIMING),
                        "hipEventCreateWithFlags",
                    )?
                };
            }
            Ok(())
        })();

        let copied = made.and_then(|()| {
            for (i, start) in (0..bytes).step_by(CHUNK).enumerate() {
                let slot = i % 2;
                let len = CHUNK.min(bytes - start);
                let host = self.slots[slot];
                src.activate()?;
                unsafe {
                    if i >= 2 {
                        // The destination has finished reading this slot.
                        src.check(
                            hipStreamWaitEvent(src.stream, h2d[slot], 0),
                            "hipStreamWaitEvent",
                        )?;
                    }
                    src.check(
                        hipMemcpyAsync(
                            host.cast(),
                            (src_buf.ptr as *const u8).add(start).cast(),
                            len,
                            src_buf.copy_kind(HIP_MEMCPY_D2H),
                            src.stream,
                        ),
                        "hipMemcpyAsync peer D→H",
                    )?;
                    src.check(hipEventRecord(d2h[slot], src.stream), "hipEventRecord")?;
                }
                dst.activate()?;
                unsafe {
                    dst.check(
                        hipStreamWaitEvent(dst.stream, d2h[slot], 0),
                        "hipStreamWaitEvent",
                    )?;
                    dst.check(
                        hipMemcpyAsync(
                            (dst_buf.ptr as *mut u8).add(start).cast(),
                            host.cast(),
                            len,
                            dst_buf.copy_kind(HIP_MEMCPY_H2D),
                            dst.stream,
                        ),
                        "hipMemcpyAsync peer H→D",
                    )?;
                    dst.check(hipEventRecord(h2d[slot], dst.stream), "hipEventRecord")?;
                }
            }
            Ok(())
        });

        // Drain both streams before the events and slots are touched again.
        let drained = dst.synchronize().and(src.synchronize());
        for event in d2h.into_iter().chain(h2d) {
            if !event.is_null() {
                unsafe { hipEventDestroy(event) };
            }
        }
        copied.and(drained)
    }
}

impl Drop for PeerStaging {
    fn drop(&mut self) {
        for slot in self.slots {
            unsafe { hipHostFree(slot.cast()) };
        }
    }
}
