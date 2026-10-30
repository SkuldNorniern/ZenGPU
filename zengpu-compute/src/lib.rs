//! Compute runtime: [`DeviceArray`] and pooled allocation. No op-graph, scheduler or fusion; that
//! lives in Laminax.

mod pool;

pub mod elementwise;
pub mod ops;

pub use elementwise::ElementwiseKernels;
pub use ops::OpList;
pub use pool::{BufferPool, BufferPoolStats};

use pool::checked_len;
use zengpu_hal::{BufferHandle, DType, GpuDevice, Result, UsageError};

/// Resident buffer plus shape/stride/dtype. No autograd or graph state (Laminax owns it).
#[derive(Debug, Clone)]
pub struct DeviceArray {
    pub buffer: BufferHandle,
    pub shape: Vec<u32>,
    pub stride: Vec<u32>,
    pub dtype: DType,
    allocation_size: u64,
    pub(crate) pool_allocation: Option<u64>,
}

impl DeviceArray {
    /// Creates an array with contiguous row-major strides derived from `shape`.
    pub fn new(buffer: BufferHandle, shape: Vec<u32>, dtype: DType) -> Self {
        let allocation_size = array_size_bytes(&shape, dtype);
        Self::with_allocation_size(buffer, shape, dtype, allocation_size)
    }

    pub(crate) fn with_allocation_size(
        buffer: BufferHandle,
        shape: Vec<u32>,
        dtype: DType,
        allocation_size: u64,
    ) -> Self {
        let stride = contiguous_strides(&shape);
        Self {
            buffer,
            shape,
            stride,
            dtype,
            allocation_size,
            pool_allocation: None,
        }
    }

    /// Checks the array against its device buffer before a built-in kernel records it.
    ///
    /// Rejects stale or foreign handles, overflowing shapes, unsupported strides, and metadata
    /// beyond the buffer capacity.
    ///
    /// # Errors
    ///
    /// Returns an error for each rejected case. On success, returns the element count.
    pub fn validate_contiguous(&self, device: &dyn GpuDevice) -> Result<u32> {
        let len = checked_len(&self.shape)?;
        if self.stride != contiguous_strides(&self.shape) {
            return Err(UsageError::BindingMismatch(
                "kernel requires contiguous array strides".into(),
            )
            .into());
        }
        let bytes = u64::from(len) * self.dtype.size_bytes() as u64;
        let capacity = device.buffer_size(self.buffer)?;
        if bytes > capacity {
            return Err(UsageError::BindingMismatch(format!(
                "array needs {bytes} bytes, buffer has {capacity}"
            ))
            .into());
        }
        Ok(len)
    }

    pub fn len(&self) -> u32 {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> u64 {
        array_size_bytes(&self.shape, self.dtype)
    }

    /// Buffer capacity in bytes. Can exceed [`Self::size_bytes`] when pool-allocated by size class.
    pub fn allocation_size_bytes(&self) -> u64 {
        self.allocation_size
    }
}

fn array_size_bytes(shape: &[u32], dtype: DType) -> u64 {
    shape.iter().product::<u32>() as u64 * dtype.size_bytes() as u64
}

fn contiguous_strides(shape: &[u32]) -> Vec<u32> {
    let mut stride = vec![1u32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert_eq!(contiguous_strides(&[]), Vec::<u32>::new());
    }
}
