//! Pooled buffer allocation for [`crate::DeviceArray`].
//!
//! Size-classed free list, so repeated elementwise/BLAS ops skip the device allocator.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result, UsageError,
};

use crate::DeviceArray;

/// Pooled [`DeviceArray`] allocator. Freed buffers return to a free list keyed by size class.
pub struct BufferPool {
    device: Arc<dyn GpuDevice>,
    state: Mutex<PoolState>,
}

/// Buffer capacities tracked by a pool, not driver allocation sizes. Live buffers
/// are never evicted. Cached buffers are rebuildable and subject to the limit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferPoolStats {
    pub live_bytes: u64,
    pub live_allocations: usize,
    pub cached_bytes: u64,
    pub cached_buffers: usize,
    pub cache_limit: u64,
}

#[derive(Default)]
struct PoolState {
    free: HashMap<u64, Vec<BufferHandle>>,
    live: HashMap<u64, (BufferHandle, u64)>,
    live_bytes: u64,
    cached_bytes: u64,
    cached_buffers: usize,
    cache_limit: u64,
}

impl PoolState {
    fn trim(&mut self, keep: u64) -> Vec<(BufferHandle, u64)> {
        let mut removed = Vec::new();
        while self.cached_bytes > keep {
            let Some(size) = self.free.keys().copied().max() else {
                break;
            };
            let list = self.free.get_mut(&size).expect("size class exists");
            if let Some(buffer) = list.pop() {
                self.cached_bytes -= size;
                self.cached_buffers -= 1;
                removed.push((buffer, size));
            }
            if list.is_empty() {
                self.free.remove(&size);
            }
        }
        removed
    }
}

impl BufferPool {
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        Self::with_cache_limit(device, u64::MAX)
    }

    /// Creates a pool with a byte limit on unused buffers. Zero disables buffer
    /// retention; live allocations are unaffected. `new` retains its existing
    /// unlimited-cache behaviour for callers that manage pressure themselves.
    pub fn with_cache_limit(device: Arc<dyn GpuDevice>, max_cached_bytes: u64) -> Self {
        Self {
            device,
            state: Mutex::new(PoolState {
                cache_limit: max_cached_bytes,
                ..Default::default()
            }),
        }
    }

    /// Reads capacity accounting without a device query or GPU synchronization.
    pub fn stats(&self) -> BufferPoolStats {
        let state = self.state.lock().unwrap();
        BufferPoolStats {
            live_bytes: state.live_bytes,
            live_allocations: state.live.len(),
            cached_bytes: state.cached_bytes,
            cached_buffers: state.cached_buffers,
            cache_limit: state.cache_limit,
        }
    }

    /// Changes the retained-byte limit and releases excess unused buffers,
    /// largest size classes first. Returns the buffer capacities released.
    pub fn set_cache_limit(&self, max_cached_bytes: u64) -> u64 {
        let removed = {
            let mut state = self.state.lock().unwrap();
            state.cache_limit = max_cached_bytes;
            state.trim(max_cached_bytes)
        };
        self.destroy_unused(removed)
    }

    /// Releases unused buffers until at most `keep` bytes remain. Does not change
    /// the configured limit. Destruction happens outside the pool lock; backend
    /// resource lifetime rules still apply to work already in flight.
    pub fn trim_unused(&self, keep: u64) -> u64 {
        let removed = self.state.lock().unwrap().trim(keep);
        self.destroy_unused(removed)
    }

    fn destroy_unused(&self, removed: Vec<(BufferHandle, u64)>) -> u64 {
        let mut bytes = 0u64;
        for (buffer, size) in removed {
            bytes = bytes.saturating_add(size);
            self.device.destroy_buffer(buffer);
        }
        bytes
    }

    pub fn device(&self) -> &dyn GpuDevice {
        &*self.device
    }

    /// Allocate a [`DeviceArray`], reusing a free buffer of the same size class. Contents are
    /// unspecified, even on reuse.
    pub fn alloc(&self, shape: Vec<u32>, dtype: DType) -> Result<DeviceArray> {
        let len = checked_len(&shape)?;
        let requested = len as u64 * dtype.size_bytes() as u64;
        let allocation_size = size_class(requested);
        static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);
        #[allow(deprecated)]
        let identity = NEXT_ALLOCATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| {
                UsageError::BindingMismatch("pool allocation identities exhausted".into())
            })?;

        let buffer = {
            let mut state = self.state.lock().unwrap();
            let reused = state.free.get_mut(&allocation_size).and_then(Vec::pop);
            if reused.is_some() {
                state.cached_bytes -= allocation_size;
                state.cached_buffers -= 1;
            }
            if state.free.get(&allocation_size).is_some_and(Vec::is_empty) {
                state.free.remove(&allocation_size);
            }
            reused
        };
        let buffer = match buffer {
            Some(b) => b,
            None => self.device.create_buffer(BufferDesc {
                size: allocation_size,
                usage: BufferUsage::STORAGE
                    | BufferUsage::READBACK
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::GpuOnly,
            })?,
        };

        {
            let mut state = self.state.lock().unwrap();
            state.live.insert(identity, (buffer, allocation_size));
            state.live_bytes = state.live_bytes.saturating_add(allocation_size);
        }
        let mut array = DeviceArray::with_allocation_size(buffer, shape, dtype, allocation_size);
        array.pool_allocation = Some(identity);
        Ok(array)
    }

    /// Return `array`'s buffer to its size class while the cache limit allows it;
    /// excess buffers are destroyed. Returning one clone
    /// ends the allocation for every alias. Duplicate, stale and foreign returns are ignored; see
    /// [`Self::try_free`] to diagnose them.
    pub fn free(&self, array: DeviceArray) {
        let _ = self.try_free(array);
    }

    /// Frees only a live allocation from this pool. The checkout identity stops an old clone from
    /// freeing a reused buffer.
    ///
    /// # Errors
    ///
    /// Returns an error for a duplicate, stale, or foreign array.
    pub fn try_free(&self, array: DeviceArray) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let identity = array.pool_allocation.ok_or_else(|| {
            UsageError::BindingMismatch("array was not allocated by this pool".into())
        })?;
        let Some(&(buffer, size)) = state
            .live
            .get(&identity)
            .filter(|(buffer, _)| *buffer == array.buffer)
        else {
            return Err(UsageError::BindingMismatch(
                "array allocation is not live in this pool".into(),
            )
            .into());
        };
        state.live.remove(&identity);
        state.live_bytes -= size;
        let keep = state
            .cached_bytes
            .checked_add(size)
            .filter(|&total| total <= state.cache_limit);
        if let Some(total) = keep {
            state.free.entry(size).or_default().push(buffer);
            state.cached_bytes = total;
            state.cached_buffers += 1;
        }
        drop(state);
        if keep.is_none() {
            self.device.destroy_buffer(buffer);
        }
        Ok(())
    }
}

pub(crate) fn checked_len(shape: &[u32]) -> Result<u32> {
    let multiply = |len: u32, &dim: &u32| -> Result<u32> {
        len.checked_mul(dim).ok_or_else(|| {
            UsageError::BindingMismatch("array shape exceeds u32 element or stride capacity".into())
                .into()
        })
    };
    let len = shape.iter().try_fold(1u32, multiply)?;
    shape.iter().rev().try_fold(1u32, multiply)?;
    Ok(len)
}

/// Buffer size for `size` bytes: powers of two up to 1 MiB, then steps of 1/8 of the power. Waste
/// stays under 12.5% for large arrays.
fn size_class(size: u64) -> u64 {
    const MIN_CLASS: u64 = 256;
    const FINE_FROM: u64 = 1 << 20;
    let pow2 = size.max(1).next_power_of_two().max(MIN_CLASS);
    if pow2 <= FINE_FROM {
        return pow2;
    }
    let step = pow2 / 16;
    size.div_ceil(step) * step
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        for buffers in state.free.values_mut() {
            for buffer in buffers.drain(..) {
                self.device.destroy_buffer(buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_len, size_class};

    #[test]
    fn rejects_overflowing_shapes_and_strides() {
        assert!(checked_len(&[65536, 65536]).is_err());
        assert!(checked_len(&[0, 65536, 65536]).is_err());
        assert!(checked_len(&[65536, 65536, 0]).is_err());
        assert_eq!(checked_len(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_len(&[2, 0, 4]).unwrap(), 0);
        assert_eq!(checked_len(&[]).unwrap(), 1);
    }

    #[test]
    fn size_classes_are_power_of_two_with_minimum_and_bounded_waste() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(255), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(4097), 8192);
        assert_eq!(size_class(1 << 20), 1 << 20);

        let weight = 4096 * 11008 * 4;
        assert_eq!(size_class(weight), 176 << 20);
        assert_eq!(size_class((1 << 20) + 1), (1 << 20) + (1 << 17));
        for size in [(1u64 << 20) + 1, 3 << 20, 100_000_000, 180_355_072, 5 << 30] {
            let class = size_class(size);
            assert!(class >= size, "{size}");
            assert!(class - size < size / 8, "{size} -> {class}");
            assert_eq!(size_class(class), class, "{size}");
        }
    }
}
