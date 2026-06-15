//! Pooled buffer allocation for [`crate::DeviceArray`]. A simple size-classed
//! free list absorbs the allocation churn of
//! repeated elementwise/BLAS ops without round-tripping through the device
//! allocator on every call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu_hal::{BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result};

use crate::DeviceArray;

/// Allocates [`DeviceArray`]s backed by pooled buffers on `device`. Freed
/// arrays return their buffer to a free list, keyed by exact byte size, for
/// reuse by a later [`BufferPool::alloc`] of the same size.
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
    /// of the same byte size if one is free.
    pub fn alloc(&self, shape: Vec<u32>, dtype: DType) -> Result<DeviceArray> {
        let len: u32 = shape.iter().product();
        let size = len as u64 * dtype.size_bytes() as u64;

        let buffer = {
            let mut free = self.free.lock().unwrap();
            free.get_mut(&size).and_then(Vec::pop)
        };
        let buffer = match buffer {
            Some(b) => b,
            None => self.device.create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::Upload,
            })?,
        };

        Ok(DeviceArray::new(buffer, shape, dtype))
    }

    /// Return `array`'s buffer to the pool for reuse by a later [`Self::alloc`]
    /// of the same byte size. Does not destroy the buffer.
    pub fn free(&self, array: DeviceArray) {
        let size = array.size_bytes();
        self.free.lock().unwrap().entry(size).or_default().push(array.buffer);
    }
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
