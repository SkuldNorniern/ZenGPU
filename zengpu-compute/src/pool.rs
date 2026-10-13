//! Pooled buffer allocation for [`crate::DeviceArray`].
//!
//! Size-classed free list, so repeated elementwise/BLAS ops skip the device allocator.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, DType, GpuDevice, MemoryUsage, Result, UsageError,
};

use crate::DeviceArray;

/// Pooled [`DeviceArray`] allocator. Freed buffers return to a free list keyed by size class.
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

    /// Allocate a [`DeviceArray`], reusing a free buffer of the same size class. Contents are
    /// unspecified, even on reuse.
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

    /// Returns `array`'s buffer to the pool for reuse in its size class. The buffer is
    /// not destroyed.
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

fn checked_len(shape: &[u32]) -> Result<u32> {
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
    }
}
