//! ZenGPU compute runtime.
//!
//! Thin layer: [`DeviceArray`] (resident buffer + shape/stride/dtype) and pooled allocation. No
//! op-graph, scheduler or fusion.

mod pool;

pub mod elementwise;

pub use pool::BufferPool;

use zengpu_hal::{BufferHandle, DType};

/// A resident buffer plus shape/stride/dtype. No autograd, op identity or graph membership.
#[derive(Debug, Clone)]
pub struct DeviceArray {
    pub buffer: BufferHandle,
    pub shape: Vec<u32>,
    pub stride: Vec<u32>,
    pub dtype: DType,
}

impl DeviceArray {
    /// Creates an array with contiguous row-major strides derived from `shape`.
    pub fn new(buffer: BufferHandle, shape: Vec<u32>, dtype: DType) -> Self {
        let stride = contiguous_strides(&shape);
        Self {
            buffer,
            shape,
            stride,
            dtype,
        }
    }

    pub fn len(&self) -> u32 {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> u64 {
        self.len() as u64 * self.dtype.size_bytes() as u64
    }
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
