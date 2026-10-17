//! Pinned host blocks that stage async transfers.
//!
//! A queued copy needs DMA-able memory the caller's slice does not keep borrowed, so each transfer
//! leases a pinned block until its event completes. `hipHostMalloc` costs more than small copies,
//! so blocks are reused.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::slice;
use std::sync::{Arc, Mutex};

use zengpu_hal::{GpuError, Result};

use crate::ffi::{hipHostFree, hipHostMalloc};
use crate::{HIP_ERROR_OUT_OF_MEMORY, check};

/// Smallest block size. Smaller transfers share this size class.
const MIN_BLOCK: usize = 64 << 10;
/// Bytes of free blocks kept for reuse. Anything beyond goes back to HIP.
const KEEP_BYTES: usize = 256 << 20;

struct Block {
    ptr: *mut u8,
    cap: usize,
}

#[derive(Default)]
pub(crate) struct PinnedPool(Mutex<Vec<Block>>);

// SAFETY: blocks are plain pinned host allocations, owned here or by one lease.
unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}

fn block_size(len: usize) -> usize {
    len.max(MIN_BLOCK).next_power_of_two()
}

impl PinnedPool {
    /// Leases a block of at least `len` bytes. The device must be current.
    pub(crate) fn lease(self: &Arc<Self>, len: usize) -> Result<PinnedLease> {
        let cap = block_size(len);
        let reused = {
            let mut free = self.0.lock().map_err(|_| GpuError::DeviceLost)?;
            free.iter()
                .position(|b| b.cap == cap)
                .map(|i| free.swap_remove(i))
        };
        let block = match reused {
            Some(block) => block,
            None => {
                let mut ptr: *mut c_void = null_mut();
                let mut rc = unsafe { hipHostMalloc(&mut ptr, cap, 0) };
                if rc == HIP_ERROR_OUT_OF_MEMORY {
                    self.trim(0);
                    rc = unsafe { hipHostMalloc(&mut ptr, cap, 0) };
                }
                check(rc, "hipHostMalloc (transfer staging)")?;
                Block {
                    ptr: ptr.cast(),
                    cap,
                }
            }
        };
        Ok(PinnedLease {
            block: Some(block),
            pool: Arc::clone(self),
        })
    }

    /// Frees cached blocks until at most `keep` bytes remain.
    fn trim(&self, keep: usize) {
        let Ok(mut free) = self.0.lock() else { return };
        let mut total: usize = free.iter().map(|b| b.cap).sum();
        while total > keep {
            let Some(block) = free.pop() else { break };
            total -= block.cap;
            unsafe { hipHostFree(block.ptr.cast()) };
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        self.trim(0);
    }
}

/// Returns the block to the pool on drop.
pub(crate) struct PinnedLease {
    block: Option<Block>,
    pool: Arc<PinnedPool>,
}

// SAFETY: the lease is the block's only owner.
unsafe impl Send for PinnedLease {}
unsafe impl Sync for PinnedLease {}

impl PinnedLease {
    pub(crate) fn ptr(&self) -> *mut u8 {
        self.block.as_ref().map_or(null_mut(), |b| b.ptr)
    }

    /// Returns the bytes of this lease. The transfer using these bytes must have completed.
    pub(crate) fn bytes(&self, len: usize) -> &[u8] {
        let block = self.block.as_ref().expect("lease holds a block");
        assert!(len <= block.cap);
        // SAFETY: the block holds `cap` bytes and nothing else writes it.
        unsafe { slice::from_raw_parts(block.ptr, len) }
    }
}

impl Drop for PinnedLease {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        let Ok(mut free) = self.pool.0.lock() else {
            unsafe { hipHostFree(block.ptr.cast()) };
            return;
        };
        let total: usize = free.iter().map(|b| b.cap).sum();
        if total + block.cap <= KEEP_BYTES {
            free.push(block);
        } else {
            drop(free);
            unsafe { hipHostFree(block.ptr.cast()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes() {
        assert_eq!(block_size(1), MIN_BLOCK);
        assert_eq!(block_size(MIN_BLOCK), MIN_BLOCK);
        assert_eq!(block_size(MIN_BLOCK + 1), 2 * MIN_BLOCK);
        assert_eq!(block_size(5 << 20), 8 << 20);
    }
}
