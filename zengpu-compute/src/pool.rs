//! Pooled buffer allocation for [`crate::DeviceArray`].
//!
//! A size-classed free list absorbs allocation churn from repeated
//! elementwise/BLAS ops without round-tripping through the device allocator on
//! every call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu_hal::{BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result};

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
    pub fn alloc(&self, shape: Vec<u32>, dtype: DType) -> Result<DeviceArray> {
        let len: u32 = shape.iter().product();
        let requested = len as u64 * dtype.size_bytes() as u64;
        let allocation_size = size_class(requested);

        let buffer = {
            let mut free = self.free.lock().unwrap();
            free.get_mut(&allocation_size).and_then(Vec::pop)
        };
        let buffer = match buffer {
            Some(b) => b,
            None => self.device.create_buffer(BufferDesc {
                size: allocation_size,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::Upload,
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

fn size_class(size: u64) -> u64 {
    const MIN_CLASS: u64 = 256;
    size.max(1).next_power_of_two().max(MIN_CLASS)
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
    use super::size_class;

    #[test]
    fn size_classes_are_power_of_two_with_minimum() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(255), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(4097), 8192);
    }
}
