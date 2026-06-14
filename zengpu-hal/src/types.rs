//! Core value types: backend selection, memory/usage classes, feature flags, formats, dtypes. No
//! backend types appear here.

/// Which backend to use. `Auto` selects the best available native backend. WebGPU is
/// never a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendPreference {
    #[default]
    Auto,
    Vulkan,
    /// CPU reference backend. Conformance oracle, not a product fallback.
    Cpu,
}

/// Used by `Auto` selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PowerPreference {
    LowPower,
    #[default]
    HighPerformance,
}

/// Memory residency intent. The allocator maps this onto backend memory types/heaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryUsage {
    /// Device-local, no host visibility (resident arrays, render targets).
    GpuOnly,
    /// Host-visible and write-combined. Staging for `GpuOnly`.
    Upload,
    /// Host-visible, cached, for GPU to CPU readback.
    Readback,
    /// Host-visible device-local where available. For small frequent writes.
    CpuToGpu,
    /// Per-frame scratch: sub-allocated and recycled.
    Transient,
    Persistent,
    /// `GpuOnly` memory, reused by size class.
    Pooled,
}

/// Buffer usage flags, a `u32` bitset that composes with `|`. The validation layer rejects bindings
/// the buffer was not created for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BufferUsage(u32);

impl BufferUsage {
    pub const STORAGE: Self = Self(1 << 0);
    pub const UNIFORM: Self = Self(1 << 1);
    pub const VERTEX: Self = Self(1 << 2);
    pub const INDEX: Self = Self(1 << 3);
    pub const INDIRECT: Self = Self(1 << 4);
    pub const TRANSFER_SRC: Self = Self(1 << 5);
    pub const TRANSFER_DST: Self = Self(1 << 6);
    /// Mappable for readback.
    pub const READBACK: Self = Self(1 << 7);

    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Device feature flags requested at creation, a `u32` bitset that composes with `|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Features(u32);

impl Features {
    pub const COMPUTE: Self = Self(1 << 0);
    pub const GRAPHICS: Self = Self(1 << 1);
    /// Bindless / descriptor indexing. Required by the binding model.
    pub const DESCRIPTOR_INDEXING: Self = Self(1 << 2);
    pub const TIMESTAMPS: Self = Self(1 << 3);
    pub const FLOAT16: Self = Self(1 << 4);
    pub const BFLOAT16: Self = Self(1 << 5);
    pub const INT8_DOT: Self = Self(1 << 6);
    pub const TENSOR_CORES: Self = Self(1 << 7);
    pub const ASYNC_COMPUTE: Self = Self(1 << 8);
    pub const MULTI_QUEUE: Self = Self(1 << 9);
    pub const UNIFIED_MEMORY: Self = Self(1 << 10);

    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for Features {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for Features {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PresentMode {
    /// Vsync. Always supported.
    #[default]
    Fifo,
    /// Low-latency, no tearing where supported.
    Mailbox,
    /// No vsync; may tear.
    Immediate,
}

/// Texture and surface pixel formats (a subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R32Float,
    Depth32Float,
    Depth24PlusStencil8,
}

impl Format {
    pub const fn is_depth_stencil(self) -> bool {
        matches!(self, Format::Depth32Float | Format::Depth24PlusStencil8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitflags_compose_and_query() {
        let usage = BufferUsage::STORAGE | BufferUsage::READBACK;
        assert!(usage.contains(BufferUsage::STORAGE));
        assert!(usage.contains(BufferUsage::READBACK));
        assert!(!usage.contains(BufferUsage::VERTEX));
        assert!(!usage.is_empty());
        assert!(BufferUsage::empty().is_empty());
    }

    #[test]
    fn features_compose() {
        let mut f = Features::COMPUTE | Features::DESCRIPTOR_INDEXING;
        assert!(f.contains(Features::COMPUTE));
        assert!(!f.contains(Features::GRAPHICS));
        f |= Features::GRAPHICS;
        assert!(f.contains(Features::GRAPHICS));
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::F64.size_bytes(), 8);
    }

    #[test]
    fn depth_formats_flagged() {
        assert!(Format::Depth32Float.is_depth_stencil());
        assert!(!Format::Rgba8Unorm.is_depth_stencil());
    }
}
