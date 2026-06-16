//! Pooled buffers for [`crate::DeviceArray`]. Size-classed free list, so repeated elementwise/BLAS
//! ops skip the device allocator.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu_hal::{BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result};

use crate::DeviceArray;

/// Free list keyed by exact byte size. Freed arrays return their buffer. `alloc` reuses only an
/// exact size match.
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

    /// Does not destroy the buffer.
    pub fn free(&self, array: DeviceArray) {
        let size = array.size_bytes();
        self.free
            .lock()
            .unwrap()
            .entry(size)
            .or_default()
            .push(array.buffer);
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
