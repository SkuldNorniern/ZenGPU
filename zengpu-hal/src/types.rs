//! Core value types shared across ZenGPU. No backend or consumer types.

use core::ops::{BitOr, BitOrAssign};

/// `Auto` picks the best native backend. WebGPU is not a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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
    /// Reserved for the unimplemented legacy OpenGL backend.
    OpenGl,
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

/// Width of each index in an index buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexFormat {
    U16,
    #[default]
    U32,
}

impl IndexFormat {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
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
    pub const UNIFIED_MEMORY: Self = Self(1 << 5);
    /// Indirect draws whose count the gpu reads from a buffer, so culling on the gpu decides how
    /// many draws run.
    pub const DRAW_INDIRECT_COUNT: Self = Self(1 << 6);
    /// Many draws from one indirect call, each with its own first instance. Without it a
    /// backend issues them one at a time.
    pub const MULTI_DRAW_INDIRECT: Self = Self(1 << 7);
    /// [`Reduction::Min`](crate::Reduction::Min) and [`Reduction::Max`](crate::Reduction::Max) samplers.
    pub const SAMPLER_MINMAX: Self = Self(1 << 8);

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
#[non_exhaustive]
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
    /// Packed unsigned floats 11/11/10. No alpha, no negatives.
    R11G11B10Float,
    Rg16Float,
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc5RgUnorm,
    Bc6hRgbUfloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Astc4x4Unorm,
    Astc4x4UnormSrgb,
}

impl Format {
    pub const fn is_depth_stencil(self) -> bool {
        matches!(self, Format::Depth32Float | Format::Depth24PlusStencil8)
    }

    pub const fn is_compressed(self) -> bool {
        self.block_size().0 > 1
    }

    /// Texels per block, `(width, height)`. `(1, 1)` for plain formats.
    pub const fn block_size(self) -> (u32, u32) {
        match self {
            Format::Bc1RgbaUnorm
            | Format::Bc1RgbaUnormSrgb
            | Format::Bc3RgbaUnorm
            | Format::Bc3RgbaUnormSrgb
            | Format::Bc4RUnorm
            | Format::Bc5RgUnorm
            | Format::Bc6hRgbUfloat
            | Format::Bc7RgbaUnorm
            | Format::Bc7RgbaUnormSrgb
            | Format::Astc4x4Unorm
            | Format::Astc4x4UnormSrgb => (4, 4),
            _ => (1, 1),
        }
    }

    /// Bytes per block. One texel for plain formats.
    pub const fn block_bytes(self) -> u32 {
        match self {
            Format::R8Unorm => 1,
            Format::R16Float | Format::R16Unorm => 2,
            Format::Rgba8Unorm
            | Format::Rgba8UnormSrgb
            | Format::Bgra8Unorm
            | Format::Bgra8UnormSrgb
            | Format::R32Float
            | Format::Depth32Float
            | Format::Depth24PlusStencil8
            | Format::R11G11B10Float
            | Format::Rg16Float => 4,
            Format::Rgba16Float | Format::Rgba16Unorm => 8,
            Format::Rgba32Float => 16,
            Format::Bc1RgbaUnorm | Format::Bc1RgbaUnormSrgb | Format::Bc4RUnorm => 8,
            Format::Bc3RgbaUnorm
            | Format::Bc3RgbaUnormSrgb
            | Format::Bc5RgUnorm
            | Format::Bc6hRgbUfloat
            | Format::Bc7RgbaUnorm
            | Format::Bc7RgbaUnormSrgb
            | Format::Astc4x4Unorm
            | Format::Astc4x4UnormSrgb => 16,
        }
    }

    /// Bytes per texel of a plain format. Zero for block formats, which are sized with
    /// `row_bytes` and `block_rows`.
    pub const fn bytes_per_texel(self) -> u32 {
        if self.is_compressed() {
            0
        } else {
            self.block_bytes()
        }
    }

    /// Packed bytes of one row of `width` texels, rounded up to whole blocks.
    pub const fn row_bytes(self, width: u32) -> u64 {
        let (bw, _) = self.block_size();
        width.div_ceil(bw) as u64 * self.block_bytes() as u64
    }

    /// Block rows that cover `height` texels.
    pub const fn block_rows(self, height: u32) -> u32 {
        height.div_ceil(self.block_size().1)
    }

    pub const ALL: &'static [Format] = &[
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
        Format::R11G11B10Float,
        Format::Rg16Float,
        Format::Bc1RgbaUnorm,
        Format::Bc1RgbaUnormSrgb,
        Format::Bc3RgbaUnorm,
        Format::Bc3RgbaUnormSrgb,
        Format::Bc4RUnorm,
        Format::Bc5RgUnorm,
        Format::Bc6hRgbUfloat,
        Format::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb,
        Format::Astc4x4Unorm,
        Format::Astc4x4UnormSrgb,
    ];
}

#[cfg(test)]
mod format_tests {
    use super::Format;

    #[test]
    fn plain_formats_are_one_texel_blocks() {
        for &f in Format::ALL.iter().filter(|f| !f.is_compressed()) {
            assert_eq!(f.block_size(), (1, 1), "{f:?}");
            assert_eq!(f.bytes_per_texel(), f.block_bytes(), "{f:?}");
            assert_eq!(f.row_bytes(3), 3 * u64::from(f.block_bytes()), "{f:?}");
            assert_eq!(f.block_rows(5), 5, "{f:?}");
        }
        assert_eq!(Format::R11G11B10Float.block_bytes(), 4);
        assert_eq!(Format::Rg16Float.block_bytes(), 4);
    }

    #[test]
    fn block_formats_size_rows_by_blocks() {
        let cases = [
            (Format::Bc1RgbaUnorm, 8),
            (Format::Bc1RgbaUnormSrgb, 8),
            (Format::Bc4RUnorm, 8),
            (Format::Bc3RgbaUnorm, 16),
            (Format::Bc3RgbaUnormSrgb, 16),
            (Format::Bc5RgUnorm, 16),
            (Format::Bc6hRgbUfloat, 16),
            (Format::Bc7RgbaUnorm, 16),
            (Format::Bc7RgbaUnormSrgb, 16),
            (Format::Astc4x4Unorm, 16),
            (Format::Astc4x4UnormSrgb, 16),
        ];
        for (f, bytes) in cases {
            assert!(f.is_compressed(), "{f:?}");
            assert_eq!(f.block_size(), (4, 4), "{f:?}");
            assert_eq!(f.block_bytes(), bytes, "{f:?}");
            assert_eq!(f.bytes_per_texel(), 0, "{f:?}");
            assert_eq!(f.row_bytes(4), u64::from(bytes), "{f:?}");
            assert_eq!(f.row_bytes(1), u64::from(bytes), "{f:?}");
            assert_eq!(f.row_bytes(9), 3 * u64::from(bytes), "{f:?}");
            assert_eq!(f.block_rows(4), 1, "{f:?}");
            assert_eq!(f.block_rows(5), 2, "{f:?}");
        }
    }

    #[test]
    fn all_lists_every_variant_once() {
        let mut names: Vec<String> = Format::ALL.iter().map(|f| format!("{f:?}")).collect();
        names.sort();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
        assert_eq!(count, 26);
    }
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
