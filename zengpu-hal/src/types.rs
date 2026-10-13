//! Core value types shared across ZenGPU. No backend or consumer types.

use core::ops::{BitOr, BitOrAssign};

/// `Auto` picks the best native backend. WebGPU is not a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendPreference {
    #[default]
    Auto,
    Vulkan,
    /// Used as a conformance oracle, not as a product fallback.
    Cpu,
    /// Compute only.
    Cuda,
    /// Graphics and compute.
    Metal,
    /// Compute only.
    Hip,
    /// Graphics and compute.
    Dx12,
}

/// Used by `Auto` selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PowerPreference {
    LowPower,
    #[default]
    HighPerformance,
}

/// Residency intent. The allocator maps it to backend memory types and heaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryUsage {
    /// Device-local. Host-visible too on shared CPU/GPU memory.
    GpuOnly,
    /// Host-visible and write-combined. Staging for `GpuOnly`.
    Upload,
    /// Host-visible and cached, for GPU-to-CPU readback.
    Readback,
    /// Host-visible device-local where available. For small frequent writes.
    CpuToGpu,
    /// Per-frame scratch: sub-allocated and recycled.
    Transient,
    Persistent,
    /// `GpuOnly` memory, reused by size class.
    Pooled,
}

/// Bitflags over `u32`, combined with `|`.
///
/// Validation rejects a binding that needs a usage the buffer was not created with.
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

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Bitflags over `u32`, combined with `|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Features(u32);

impl Features {
    pub const COMPUTE: Self = Self(1 << 0);
    pub const GRAPHICS: Self = Self(1 << 1);
    /// Required by the binding model.
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
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for Features {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for Features {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Bitflags over `u32`, combined with `|`. Validated like [`BufferUsage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureUsage(u32);

impl TextureUsage {
    pub const SAMPLED: Self = Self(1 << 0);
    pub const STORAGE: Self = Self(1 << 1);
    pub const RENDER_TARGET: Self = Self(1 << 2);
    pub const DEPTH_STENCIL: Self = Self(1 << 3);
    pub const TRANSFER_SRC: Self = Self(1 << 4);
    pub const TRANSFER_DST: Self = Self(1 << 5);

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

impl BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for TextureUsage {
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

/// Ways a format can be used on a device, as reported by
/// [`GpuDevice::format_caps`](crate::GpuDevice::format_caps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    /// Half float. Works for compositing.
    Rgba16Float,
    Rgba32Float,
    Rgba16Unorm,
    R8Unorm,
    R16Float,
    R16Unorm,
    R32Float,
    Depth32Float,
    Depth24PlusStencil8,
}

impl Format {
    pub const fn is_depth_stencil(self) -> bool {
        matches!(self, Format::Depth32Float | Format::Depth24PlusStencil8)
    }

    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Format::R8Unorm => 1,
            Format::R16Float | Format::R16Unorm => 2,
            Format::Rgba8Unorm
            | Format::Rgba8UnormSrgb
            | Format::Bgra8Unorm
            | Format::Bgra8UnormSrgb
            | Format::R32Float
            | Format::Depth32Float
            | Format::Depth24PlusStencil8 => 4,
            Format::Rgba16Float | Format::Rgba16Unorm => 8,
            Format::Rgba32Float => 16,
        }
    }

    pub const ALL: [Format; 13] = [
        Format::Rgba8Unorm,
        Format::Rgba8UnormSrgb,
        Format::Bgra8Unorm,
        Format::Bgra8UnormSrgb,
        Format::Rgba16Float,
        Format::Rgba32Float,
        Format::Rgba16Unorm,
        Format::R8Unorm,
        Format::R16Float,
        Format::R16Unorm,
        Format::R32Float,
        Format::Depth32Float,
        Format::Depth24PlusStencil8,
    ];
}

/// Axis-aligned pixel rectangle, for scissor and damage regions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
    /// Covers `(0, 0)..(width, height)`, depth `0.0..1.0`.
    pub const fn new(width: f32, height: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
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
        assert_eq!(
            f.difference(Features::COMPUTE),
            Features::GRAPHICS | Features::DESCRIPTOR_INDEXING
        );
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
