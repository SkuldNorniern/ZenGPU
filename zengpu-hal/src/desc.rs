//! Plain descriptor data with no backend types.

use crate::handle::{ShaderHandle, TextureHandle};
use crate::types::{BufferUsage, Format, MemoryUsage, PresentMode, TextureUsage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TexDim {
    /// A flat image. `array_layers > 1` makes a 2D array.
    #[default]
    D2,
    /// A volume. `TextureDesc::depth` is the slice count.
    D3,
    /// Six square layers (+X, -X, +Y, -Y, +Z, -Z). `array_layers` must be `6`.
    Cube,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Depth slices for [`TexDim::D3`]. Must be `1` for `D2` and `Cube`.
    pub depth: u32,
    pub format: Format,
    pub usage: TextureUsage,
    /// MSAA sample count. `1` means no multisampling.
    pub samples: u32,
    pub dimension: TexDim,
    /// Mip levels, including the base level. `1` means no mip chain.
    pub mip_levels: u32,
    /// Array layer count. `1` means a single non-array texture. Must be `6` for [`TexDim::Cube`].
    pub array_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    /// MSAA sample count. `1` means no multisampling.
    pub samples: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    /// SPIR-V words as raw bytes. The length must be a multiple of 4.
    Spirv(&'a [u8]),
    /// Null-terminated PTX assembly (CUDA Driver API format).
    Ptx(&'a [u8]),
    /// Metal Shading Language source as UTF-8 bytes.
    Msl(&'a [u8]),
    /// HIP C++ kernel source as UTF-8 bytes, compiled at runtime via hipRTC.
    Hip(&'a [u8]),
    /// CUDA C++ kernel source as UTF-8 bytes, compiled at runtime via NVRTC.
    CudaSrc(&'a [u8]),
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderDesc<'a> {
    pub source: ShaderSource<'a>,
}

impl<'a> ShaderDesc<'a> {
    pub fn spirv(bytes: &'a [u8]) -> Self {
        Self {
            source: ShaderSource::Spirv(bytes),
        }
    }
    pub fn ptx(bytes: &'a [u8]) -> Self {
        Self {
            source: ShaderSource::Ptx(bytes),
        }
    }
    pub fn msl(source: &'a str) -> Self {
        Self {
            source: ShaderSource::Msl(source.as_bytes()),
        }
    }
    pub fn hip(source: &'a str) -> Self {
        Self {
            source: ShaderSource::Hip(source.as_bytes()),
        }
    }
    pub fn cuda_src(source: &'a str) -> Self {
        Self {
            source: ShaderSource::CudaSrc(source.as_bytes()),
        }
    }
}

/// Thread-block dimensions for backends that need them at creation (CUDA, HIP). `[0, 0, 0]` means
/// the backend default. Vulkan ignores this, since the block size is in SPIR-V `LocalSizeX/Y/Z`.
#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineDesc<'a> {
    pub shader: ShaderHandle,
    pub entry: &'a str,
    /// Thread-block dimensions. `[0, 0, 0]` means the backend default.
    pub block: [u32; 3],
}

/// Tuning requests a backend may ignore. Results never depend on them, only speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputePipelineHints {
    /// lanes per subgroup, e.g. 64 for wave64 on RDNA (drivers often default to wave32). Ignored
    /// outside the device range.
    pub subgroup_size: Option<u32>,
    /// Keeps the driver resource report for [`crate::GpuDevice::pipeline_stats`]. Ignored where it is
    /// always reported.
    pub capture_stats: bool,
}

/// Unreported fields are zero, and `waves_per_simd` is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PipelineStats {
    /// Vector registers per lane.
    pub vgprs: u32,
    /// Scalar registers per wave.
    pub sgprs: u32,
    /// Private (spill / stack) bytes per lane. Non-zero means spilling.
    pub scratch_bytes: u32,
    pub lds_bytes: u32,
    pub max_threads: u32,
    /// Waves resident per SIMD at this register use, when the driver says.
    pub waves_per_simd: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Border color for clamp-to-border. The three presets need no extra device feature on any backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderColor {
    #[default]
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDesc {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub address: AddressMode,
    /// Maximum anisotropic samples. `1` disables it. Clamped to the device maximum, and ignored
    /// where unsupported.
    pub anisotropy: u8,
    pub lod_min: f32,
    pub lod_max: f32,
    /// `None` for a normal sampler. Set this for shadow-map PCF.
    pub compare: Option<CompareFn>,
    /// Used only by clamp-to-border address modes.
    pub border: BorderColor,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
            anisotropy: 1,
            lod_min: 0.0,
            lod_max: 1000.0,
            compare: None,
            border: BorderColor::TransparentBlack,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleStrip,
    LineList,
    PointList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Snorm8x4,
    Unorm16x2,
    Unorm16x4,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint8x4Unorm,
}

impl VertexFormat {
    /// Size of one attribute in bytes.
    pub const fn size_bytes(self) -> u32 {
        match self {
            VertexFormat::Float16x2
            | VertexFormat::Float32
            | VertexFormat::Snorm8x4
            | VertexFormat::Unorm16x2
            | VertexFormat::Uint32
            | VertexFormat::Uint8x4Unorm => 4,
            VertexFormat::Float16x4
            | VertexFormat::Float32x2
            | VertexFormat::Unorm16x4
            | VertexFormat::Sint32x2 => 8,
            VertexFormat::Float32x3 | VertexFormat::Sint32x3 => 12,
            VertexFormat::Float32x4 | VertexFormat::Sint32x4 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    /// Byte offset within the vertex.
    pub offset: u32,
    pub format: VertexFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepMode {
    #[default]
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VertexLayout<'a> {
    /// Bytes between consecutive vertices.
    pub stride: u32,
    pub attributes: &'a [VertexAttribute],
    pub step_mode: StepMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    AlphaBlend,
    /// Second color output used as the source factor. Needs the feature reported
    /// by `GraphicsDevice::supports_dual_source_blending`.
    DualSourceAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub op: BlendOp,
}

/// Portable subset: only the factors every backend can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// `None` in `ColorTargetState::blend` means an opaque write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const ALPHA_BLEND: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
    };

    /// Dual-source blend factor. Needs `GraphicsDevice::supports_dual_source_blending`.
    pub const DUAL_SOURCE_ALPHA: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::Src1Color,
            dst_factor: BlendFactor::OneMinusSrc1Color,
            op: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Src1Alpha,
            dst_factor: BlendFactor::OneMinusSrc1Alpha,
            op: BlendOp::Add,
        },
    };
}

#[derive(Debug, Clone, Copy)]
pub struct ColorTargetState {
    pub format: Format,
    pub blend: Option<BlendState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrontFace {
    #[default]
    Ccw,
    Cw,
}

/// `Line` and `Point` need `GraphicsDevice::supports_non_solid_fill` (Vulkan `fillModeNonSolid`).
/// Pipeline creation fails without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RasterState {
    pub cull: CullMode,
    pub front_face: FrontFace,
    pub polygon: PolygonMode,
}

/// The default is `Less`. For reverse-Z, use `Greater` with a `1.0` depth clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompareFn {
    Never,
    #[default]
    Less,
    Equal,
    LessEqual,
    Greater,
    GreaterEqual,
    NotEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DepthState {
    pub test: bool,
    pub write: bool,
    /// Used only when `test` is `true`.
    pub compare: CompareFn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFaceState {
    pub compare: CompareFn,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    /// One layout per vertex buffer, indexed by buffer binding (see
    /// `RenderCommands::set_vertex_buffer`). An empty list means no vertex buffers. For mixed
    /// per-vertex and per-instance data, list both, with differing `step_mode`.
    pub vertex_layouts: &'a [VertexLayout<'a>],
    pub topology: PrimitiveTopology,
    pub color_targets: &'a [ColorTargetState],
    pub depth_format: Option<Format>,
    pub depth: DepthState,
    pub stencil: Option<StencilState>,
    pub raster: RasterState,
    /// MSAA sample count. `1` means no multisampling.
    pub samples: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceConfig {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_format_sizes() {
        assert_eq!(VertexFormat::Float16x2.size_bytes(), 4);
        assert_eq!(VertexFormat::Float16x4.size_bytes(), 8);
        assert_eq!(VertexFormat::Snorm8x4.size_bytes(), 4);
        assert_eq!(VertexFormat::Unorm16x2.size_bytes(), 4);
        assert_eq!(VertexFormat::Unorm16x4.size_bytes(), 8);
        assert_eq!(VertexFormat::Sint32x2.size_bytes(), 8);
        assert_eq!(VertexFormat::Sint32x3.size_bytes(), 12);
        assert_eq!(VertexFormat::Sint32x4.size_bytes(), 16);
        assert_eq!(VertexFormat::Float32x3.size_bytes(), 12);
        assert_eq!(VertexFormat::Uint8x4Unorm.size_bytes(), 4);
    }

    #[test]
    fn sampler_default_is_linear_clamp() {
        let s = SamplerDesc::default();
        assert_eq!(s.min_filter, FilterMode::Linear);
        assert_eq!(s.address, AddressMode::ClampToEdge);
    }
}

/// A rectangle in texels of one mip level of one layer (or cube face).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRegion {
    pub mip_level: u32,
    pub layer: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl TextureRegion {
    pub const fn rect(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            mip_level: 0,
            layer: 0,
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureWrite<'a> {
    pub texture: TextureHandle,
    pub region: TextureRegion,
    /// Tightly packed texel rows, top row first.
    pub data: &'a [u8],
    /// Bytes from one row to the next. Must be at least the region width.
    pub bytes_per_row: u32,
}
