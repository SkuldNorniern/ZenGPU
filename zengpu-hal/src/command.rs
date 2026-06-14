//! Command value types: bindless bindings and inline scalars.

/// Scalar arguments, kept push-constant sized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Bindless bindings for a dispatch or draw.
///
/// Resources are referenced by slot index ([`crate::Handle::index`]), not a per-pipeline descriptor
/// slot. This keeps the compiler ABI stable as kernels gain arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bindings<'a> {
    pub buffers: &'a [u32],
    /// Bindless indices into the texture table.
    pub textures: &'a [u32],
    pub scalars: &'a [Scalar],
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
