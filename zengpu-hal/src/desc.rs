//! Resource and pipeline descriptors. Plain data with no backend or consumer
//! types. Backends consume these to create resources.

use crate::handle::ShaderHandle;
use crate::types::{BufferUsage, Format, MemoryUsage, PresentMode, TextureUsage};

/// Describes a GPU buffer to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: u64,
    /// How the buffer may be used.
    pub usage: BufferUsage,
    /// Residency intent.
    pub memory: MemoryUsage,
}

/// Texture dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TexDim {
    /// A flat image, or — with `array_layers > 1` — a 2D array (shadow
    /// cascades, texture atlases).
    #[default]
    D2,
    /// A volume texture; `TextureDesc::depth` is the number of depth slices.
    D3,
    /// Six square layers (+X, -X, +Y, -Y, +Z, -Z) sampled as a cubemap.
    /// `TextureDesc::array_layers` must be `6`.
    Cube,
}

/// Describes a texture to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Depth slices for [`TexDim::D3`]; must be `1` for `D2`/`Cube`.
    pub depth: u32,
    pub format: Format,
    pub usage: TextureUsage,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
    pub dimension: TexDim,
    /// Mip levels, base level included; `1` means no mip chain.
    pub mip_levels: u32,
    /// Array layer count; `1` means a single non-array texture. Must be `6`
    /// for [`TexDim::Cube`].
    pub array_layers: u32,
}

/// Describes an offscreen render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
}

/// Shader source bytes, tagged by format.
#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    /// SPIR-V words as raw bytes (length must be a multiple of 4).
    Spirv(&'a [u8]),
    /// Null-terminated PTX assembly (CUDA Driver API format).
    Ptx(&'a [u8]),
    /// Metal Shading Language source as UTF-8 bytes.
    Msl(&'a [u8]),
}

/// Describes a shader module to create.
#[derive(Debug, Clone, Copy)]
pub struct ShaderDesc<'a> {
    pub source: ShaderSource<'a>,
}

impl<'a> ShaderDesc<'a> {
    pub fn spirv(bytes: &'a [u8]) -> Self {
        Self { source: ShaderSource::Spirv(bytes) }
    }
    pub fn ptx(bytes: &'a [u8]) -> Self {
        Self { source: ShaderSource::Ptx(bytes) }
    }
    /// Metal Shading Language source.
    pub fn msl(source: &'a str) -> Self {
        Self { source: ShaderSource::Msl(source.as_bytes()) }
    }
}

/// Describes a compute pipeline.
///
/// `block` specifies the thread-block dimensions for backends that require
/// them at pipeline creation time (CUDA, HIP). `[0, 0, 0]` lets the backend
/// choose a default; Vulkan ignores this field entirely (block size is encoded
/// in the SPIR-V `LocalSizeX/Y/Z` execution mode).
#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineDesc<'a> {
    pub shader: ShaderHandle,
    /// Entry-point name in the shader module.
    pub entry: &'a str,
    /// Thread-block dimensions; `[0, 0, 0]` = backend default.
    pub block: [u32; 3],
}

/// Texture sampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    #[default]
    Nearest,
    Linear,
}

/// Texture coordinate addressing outside `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Border color sampled when [`AddressMode::ClampToEdge`] would otherwise
/// sample outside the texture at the very edge texel — used by backends that
/// support a clamp-to-border mode. The three presets below need no extra
/// device feature/extension on any backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderColor {
    #[default]
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

/// Describes a sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDesc {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    /// Filter used between mip levels.
    pub mip_filter: FilterMode,
    pub address: AddressMode,
    /// Maximum anisotropic samples; `1` disables anisotropic filtering.
    /// Clamped to the device's maximum; ignored where the device cannot do
    /// anisotropic filtering at all.
    pub anisotropy: u8,
    /// Minimum clamp for the computed mip LOD.
    pub lod_min: f32,
    /// Maximum clamp for the computed mip LOD.
    pub lod_max: f32,
    /// Comparison function for a shadow-map PCF sampler; `None` is a normal
    /// (non-comparison) sampler.
    pub compare: Option<CompareFn>,
    /// Border color used by [`AddressMode`] variants that clamp to a border
    /// rather than the edge texel.
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

/// Primitive assembly mode for a graphics pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleStrip,
    LineList,
    PointList,
}

/// Per-attribute vertex format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint8x4Unorm,
}

impl VertexFormat {
    /// Size of one attribute in bytes.
    pub const fn size_bytes(self) -> u32 {
        match self {
            VertexFormat::Float32 | VertexFormat::Uint32 | VertexFormat::Uint8x4Unorm => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
        }
    }
}

/// One vertex attribute within a vertex buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Shader input location.
    pub location: u32,
    /// Byte offset within the vertex.
    pub offset: u32,
    pub format: VertexFormat,
}

/// Whether a vertex buffer advances per-vertex or per-instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepMode {
    /// Advance to the next element for each vertex (default).
    #[default]
    Vertex,
    /// Advance to the next element for each instance.
    Instance,
}

/// The layout of one vertex buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexLayout<'a> {
    /// Bytes between consecutive vertices.
    pub stride: u32,
    pub attributes: &'a [VertexAttribute],
    pub step_mode: StepMode,
}

/// Color blending for a graphics pipeline (kept simple to start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Source replaces destination.
    #[default]
    Opaque,
    /// Standard src-alpha over (premultiplied-friendly) blend.
    AlphaBlend,
    /// Dual-source blend for coverage-based text rendering: the fragment
    /// shader writes a second color output (`layout(location = 0, index =
    /// 1)`) used as the source-blend factor. Requires
    /// [`GraphicsDevice::supports_dual_source_blending`](crate::graphics::GraphicsDevice::supports_dual_source_blending).
    DualSourceAlpha,
}

/// Backface culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// Render both front and back faces.
    #[default]
    None,
    Front,
    Back,
}

/// Which winding order is considered front-facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrontFace {
    /// Counter-clockwise winding is front-facing (default).
    #[default]
    Ccw,
    /// Clockwise winding is front-facing.
    Cw,
}

/// How rasterized primitives are filled.
///
/// `Line` and `Point` require [`GraphicsDevice::supports_non_solid_fill`](crate::graphics::GraphicsDevice::supports_non_solid_fill)
/// (Vulkan's `fillModeNonSolid`); pipeline creation fails if requested without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    /// Wireframe rendering — the engine's debug/wireframe path.
    Line,
    Point,
}

/// Rasterizer state: culling, winding, and fill mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RasterState {
    pub cull: CullMode,
    pub front_face: FrontFace,
    pub polygon: PolygonMode,
}

/// Depth comparison function.
///
/// Selects which fragments pass the depth test. Defaults to [`CompareFn::Less`].
/// Use [`CompareFn::Greater`] with a `1.0` depth clear for reverse-Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareFn {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    GreaterEqual,
    NotEqual,
    Always,
}

impl Default for CompareFn {
    fn default() -> Self {
        Self::Less
    }
}

/// Depth test/write configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DepthState {
    /// Whether the depth test is enabled.
    pub test: bool,
    /// Whether depth is written.
    pub write: bool,
    /// Depth comparison function. Only meaningful when `test` is `true`.
    pub compare: CompareFn,
}

/// Describes a graphics pipeline with 3D support.
#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    /// One layout per bound vertex buffer; the slice index is the buffer
    /// binding, matching [`RenderCommands::set_vertex_buffer`](crate::graphics::RenderCommands::set_vertex_buffer)'s
    /// `slot`. An empty slice means no vertex buffers (vertices generated in the
    /// shader). Mixed per-vertex/per-instance pipelines list multiple layouts
    /// with differing [`VertexLayout::step_mode`].
    pub vertex_layouts: &'a [VertexLayout<'a>],
    pub topology: PrimitiveTopology,
    /// Color attachment format.
    pub color_format: Format,
    /// Depth attachment format, if any.
    pub depth_format: Option<Format>,
    pub depth: DepthState,
    pub blend: BlendMode,
    pub raster: RasterState,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
}

/// Configuration for a presentable surface.
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
