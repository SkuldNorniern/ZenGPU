//! ZenGPU compute runtime.
//!
//! ZenGPU is a pure execution runtime: this crate provides only the thin
//! [`DeviceArray`] (a resident buffer + shape/stride/dtype) and pooled
//! allocation. There is no op-graph, scheduler, or fusion here — that is
//! Laminax's job.

mod pool;

pub mod elementwise;

pub use elementwise::ElementwiseKernels;
pub use pool::BufferPool;

use zengpu_hal::{BufferHandle, DType};

/// ZenGPU's entire "tensor" surface: a resident buffer plus the
/// dimension metadata BLAS/elementwise kernels need. Carries no autograd, op
/// identity, or graph membership — Laminax owns all of that.
#[derive(Debug, Clone)]
pub struct DeviceArray {
    pub buffer: BufferHandle,
    pub shape: Vec<u32>,
    pub stride: Vec<u32>,
    pub dtype: DType,
}

impl DeviceArray {
    /// A new array description over `buffer`, with row-major contiguous
    /// strides computed from `shape`.
    pub fn new(buffer: BufferHandle, shape: Vec<u32>, dtype: DType) -> Self {
        let stride = contiguous_strides(&shape);
        Self {
            buffer,
            shape,
            stride,
            dtype,
        }
    }

    /// Total element count (product of `shape`).
    pub fn len(&self) -> u32 {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total size in bytes (`len() * dtype.size_bytes()`).
    pub fn size_bytes(&self) -> u64 {
        self.len() as u64 * self.dtype.size_bytes() as u64
    }
}

/// Row-major contiguous strides for `shape`.
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
    use zengpu_hal::{SlotMap, marker};

    fn dummy_handle() -> BufferHandle {
        let mut map = SlotMap::<marker::Buffer, ()>::new();
        map.insert(())
    }

    #[test]
    fn contiguous_strides_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert_eq!(contiguous_strides(&[]), Vec::<u32>::new());
    }

    #[test]
    fn device_array_len_and_size() {
        let arr = DeviceArray::new(dummy_handle(), vec![2, 3, 4], DType::F32);
        assert_eq!(arr.len(), 24);
        assert_eq!(arr.size_bytes(), 24 * 4);
        assert_eq!(arr.stride, vec![12, 4, 1]);
    }
}
