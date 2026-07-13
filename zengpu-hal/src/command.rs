//! Command-recording value types — bindless bindings and inline scalars
//! Backends consume these when recording a dispatch or draw.

use crate::handle::PipelineHandle;

/// An inline scalar argument passed to a pipeline (push-constant-sized).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Bindless bindings for a dispatch or draw.
///
/// Resources are referenced by their slot index ([`crate::Handle::index`]), not
/// by a per-pipeline descriptor slot — which is what keeps the compiler ABI
/// stable as kernels gain arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bindings<'a> {
    /// Bindless indices into the storage-buffer table.
    pub buffers: &'a [u32],
    /// Bindless indices into the texture table.
    pub textures: &'a [u32],
    /// Inline scalar arguments.
    pub scalars: &'a [Scalar],
}

/// One dispatch within a [`crate::GpuDevice::dispatch_batch`] call.
#[derive(Debug, Clone, Copy)]
pub struct DispatchOp<'a> {
    pub pipeline: PipelineHandle,
    pub bindings: Bindings<'a>,
    pub grid: [u32; 3],
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
