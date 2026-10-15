//! Device memory beyond plain `hipMalloc`/`hipFree`.
//!
//! Two costs of HIP's allocator are worked around here:
//!
//! - `hipFree` synchronizes the whole device, so freeing any buffer while a
//!   kernel runs (on any stream, from any device handle) blocks the caller
//!   until that kernel ends. Frees are handed to one background thread, so
//!   only it waits.
//! - `hipMalloc` + `hipFree` of 2 MiB and up cost 160–2300 µs. Each device
//!   keeps freed blocks (4 KiB and up, within a byte budget) and hands them
//!   back to later allocations of about the same size, so most frees never
//!   reach HIP at all. Reuse stays on the device's own stream, which orders it
//!   after every earlier use.
//!
//! HIP's stream-ordered pool (`hipMallocAsync`) would give both, but on
//! ROCm 7.2 it deadlocked inside the runtime when a pool allocation had to
//! grow while kernels from other device handles ran on the same GPU.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::take;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread::{Builder, sleep};
use std::time::{Duration, Instant};

use crate::ffi::{hipFree, hipSetDevice};

/// Smallest block worth caching; smaller ones go straight back to HIP.
pub(crate) const CACHE_MIN_BYTES: usize = 4 << 10;

/// Freed large blocks of one device, by size.
pub(crate) struct BlockCache {
    blocks: BTreeMap<usize, Vec<usize>>,
    bytes: usize,
    limit: usize,
}

impl BlockCache {
    /// A cache holding at most `limit` bytes.
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            blocks: BTreeMap::new(),
            bytes: 0,
            limit,
        }
    }

    /// The smallest cached block of at least `size` bytes and at most a
    /// quarter more, with its size.
    pub(crate) fn take(&mut self, size: usize) -> Option<(*mut c_void, usize)> {
        let max = size.saturating_add(size / 4);
        let (&block, ptrs) = self.blocks.range_mut(size..=max).next()?;
        let ptr = ptrs.pop()?;
        if ptrs.is_empty() {
            self.blocks.remove(&block);
        }
        self.bytes -= block;
        Some((ptr as *mut c_void, block))
    }

    /// Keep a freed block; `false` when the cache is full and the caller
    /// must free it.
    pub(crate) fn put(&mut self, ptr: *mut c_void, block: usize) -> bool {
        if block < CACHE_MIN_BYTES || self.bytes + block > self.limit {
            return false;
        }
        self.blocks.entry(block).or_default().push(ptr as usize);
        self.bytes += block;
        true
    }

    /// Empty the cache, returning every block.
    pub(crate) fn drain(&mut self) -> Vec<*mut c_void> {
        self.bytes = 0;
        take(&mut self.blocks)
            .into_values()
            .flatten()
            .map(|ptr| ptr as *mut c_void)
            .collect()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Frees queued to the background thread and not yet done.
static PENDING: AtomicUsize = AtomicUsize::new(0);

/// Where frees are sent: (device ordinal, pointer).
type FreeQueue = Mutex<Sender<(i32, usize)>>;

fn reaper() -> Option<&'static FreeQueue> {
    static REAPER: OnceLock<Option<FreeQueue>> = OnceLock::new();
    REAPER
        .get_or_init(|| {
            let (tx, rx) = channel::<(i32, usize)>();
            Builder::new()
                .name("zengpu-hip-free".into())
                .spawn(move || {
                    for (ordinal, ptr) in rx {
                        unsafe {
                            hipSetDevice(ordinal);
                            hipFree(ptr as *mut c_void);
                        }
                        PENDING.fetch_sub(1, Ordering::AcqRel);
                    }
                })
                .ok()
                .map(|_| Mutex::new(tx))
        })
        .as_ref()
}

/// Free `ptr` on device `ordinal` without waiting for the GPU. Falls back to
/// an inline `hipFree` if the background thread cannot be had.
pub(crate) fn free_later(ordinal: i32, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    PENDING.fetch_add(1, Ordering::AcqRel);
    let sent = reaper()
        .and_then(|tx| tx.lock().ok())
        .is_some_and(|tx| tx.send((ordinal, ptr as usize)).is_ok());
    if !sent {
        PENDING.fetch_sub(1, Ordering::AcqRel);
        unsafe {
            hipSetDevice(ordinal);
            hipFree(ptr);
        }
    }
}

/// Wait up to `timeout` for queued frees to finish, so their memory can be
/// allocated again. Returns whether none are left.
pub(crate) fn wait_for_frees(timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while PENDING.load(Ordering::Acquire) != 0 {
        if Instant::now() >= deadline {
            return false;
        }
        sleep(Duration::from_micros(200));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(n: usize) -> *mut c_void {
        n as *mut c_void
    }

    #[test]
    fn cache_reuses_close_sizes_only() {
        let mut cache = BlockCache::new(64 << 20);
        assert!(!cache.put(p(1), 1 << 10), "below the caching floor");
        assert!(cache.put(p(0x1000), 8 << 20));
        assert!(cache.put(p(0x2000), 4 << 20));
        assert_eq!(cache.bytes(), 12 << 20);

        // 4 MiB fits 4 MiB; 3.5 MiB may take 4 MiB (within a quarter).
        assert_eq!(cache.take(4 << 20), Some((p(0x2000), 4 << 20)));
        // 5 MiB would waste more than a quarter of 8 MiB: no match.
        assert_eq!(cache.take(5 << 20), None);
        assert_eq!(cache.take(7 << 20), Some((p(0x1000), 8 << 20)));
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn cache_respects_its_limit_and_drains() {
        let mut cache = BlockCache::new(10 << 20);
        assert!(cache.put(p(1), 8 << 20));
        assert!(!cache.put(p(2), 4 << 20), "over the limit");
        assert!(cache.put(p(3), 2 << 20));
        let mut drained = cache.drain();
        drained.sort();
        assert_eq!(drained, vec![p(1), p(3)]);
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.take(2 << 20), None);
    }
}
