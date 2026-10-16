//! Pooled buffer allocation for [`crate::DeviceArray`].
//!
//! A size-classed free list absorbs allocation churn from repeated
//! elementwise/BLAS ops without round-tripping through the device allocator on
//! every call.

// A poisoned pool mutex is an internal invariant failure and is meant to panic loudly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result, UsageError,
};

use crate::DeviceArray;

/// Allocates [`DeviceArray`]s backed by pooled buffers on `device`. Freed
/// arrays return their buffer to a free list, keyed by power-of-two byte size,
/// for reuse by a later [`BufferPool::alloc`] that fits in the same class.
pub struct BufferPool {
    device: Arc<dyn GpuDevice>,
    free: Mutex<HashMap<u64, Vec<BufferHandle>>>,
}

impl BufferPool {
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        Self {
            device,
            free: Mutex::new(HashMap::new()),
        }
    }

    pub fn device(&self) -> &dyn GpuDevice {
        &*self.device
    }

    /// Allocate a [`DeviceArray`] of `shape`/`dtype`, reusing a pooled buffer
    /// from the matching size class if one is free.
    /// Contents are unspecified, including when reusing a freed buffer.
    pub fn alloc(&self, shape: Vec<u32>, dtype: DType) -> Result<DeviceArray> {
        let len = checked_len(&shape)?;
        let requested = len as u64 * dtype.size_bytes() as u64;
        let allocation_size = size_class(requested);

        let buffer = {
            let mut free = self.free.lock().unwrap();
            free.get_mut(&allocation_size).and_then(Vec::pop)
        };
        let buffer = match buffer {
            Some(b) => b,
            // Device-local: kernels read arrays at VRAM speed, and every
            // backend moves host data in and out (Vulkan through staging).
            None => self.device.create_buffer(BufferDesc {
                size: allocation_size,
                usage: BufferUsage::STORAGE
                    | BufferUsage::READBACK
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::GpuOnly,
            })?,
        };

        Ok(DeviceArray::with_allocation_size(
            buffer,
            shape,
            dtype,
            allocation_size,
        ))
    }

    /// Return `array`'s buffer to the pool for reuse by a later [`Self::alloc`]
    /// in the same size class. Does not destroy the buffer.
    pub fn free(&self, array: DeviceArray) {
        let size = array.allocation_size_bytes();
        self.free
            .lock()
            .unwrap()
            .entry(size)
            .or_default()
            .push(array.buffer);
    }
}

// Check suffix products too: even a zero-length array needs representable strides.
fn checked_len(shape: &[u32]) -> Result<u32> {
    let multiply = |len: u32, &dim: &u32| -> Result<u32> {
        len.checked_mul(dim).ok_or_else(|| {
            UsageError::BindingMismatch("array shape exceeds u32 element or stride capacity".into())
                .into()
        })
    };
    // DeviceArray's element count uses forward products; strides use suffixes.
    let len = shape.iter().try_fold(1u32, multiply)?;
    shape.iter().rev().try_fold(1u32, multiply)?;
    Ok(len)
}

/// The buffer size an array of `size` bytes gets: powers of two up to 1 MiB,
/// then steps of an eighth of the power of two below it, so a large array
/// (an LLM weight, say 180 MB) wastes under 12.5% of its size instead of up
/// to half its buffer.
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
        let mut free = self.free.lock().unwrap();
        for buffers in free.values_mut() {
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
    fn size_classes_are_power_of_two_with_minimum() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(255), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(4097), 8192);
        assert_eq!(size_class(1 << 20), 1 << 20);
    }

    #[test]
    fn large_size_classes_waste_at_most_an_eighth() {
        // 4096 x 11008 f32 weight: 180 MB lands in 176 MiB, not 256 MiB.
        let weight = 4096 * 11008 * 4;
        assert_eq!(size_class(weight), 176 << 20);
        assert_eq!(size_class((1 << 20) + 1), (1 << 20) + (1 << 17));
        for size in [(1u64 << 20) + 1, 3 << 20, 100_000_000, 180_355_072, 5 << 30] {
            let class = size_class(size);
            assert!(class >= size, "{size}");
            assert!(class - size < size / 8, "{size} -> {class}");
            // A class is its own class, so freed buffers are found again.
            assert_eq!(size_class(class), class, "{size}");
        }
    }
}
