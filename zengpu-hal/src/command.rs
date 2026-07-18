//! Command-recording value types, consumed by backends when recording a dispatch or draw.

use crate::handle::{BufferHandle, PipelineHandle};

/// Scalar arguments, kept push-constant sized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Resources are referenced by slot index ([`crate::Handle::index`]), not by a per-pipeline
/// descriptor slot. That keeps the compiler ABI stable as kernels gain arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bindings<'a> {
    pub buffers: &'a [u32],
    /// Bindless indices into the texture table.
    pub textures: &'a [u32],
    pub scalars: &'a [Scalar],
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchOp<'a> {
    pub pipeline: PipelineHandle,
    pub bindings: Bindings<'a>,
    pub grid: [u32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct BufferCopyOp {
    pub src: BufferHandle,
    pub src_offset: u64,
    pub dst: BufferHandle,
    pub dst_offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum ComputeOp<'a> {
    CopyBuffer(BufferCopyOp),
    Dispatch(DispatchOp<'a>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_default_is_empty() {
        let b = Bindings::default();
        assert!(b.buffers.is_empty());
        assert!(b.textures.is_empty());
        assert!(b.scalars.is_empty());
    }

    #[test]
    fn bindings_carry_indices_and_scalars() {
        let b = Bindings {
            buffers: &[3, 7],
            textures: &[],
            scalars: &[Scalar::U32(256)],
        };
        assert_eq!(b.buffers, &[3, 7]);
        assert_eq!(b.scalars, &[Scalar::U32(256)]);
    }
}
