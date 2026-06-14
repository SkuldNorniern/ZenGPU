//! Resource and pipeline descriptors (plan §5, §15, §20). Plain data — no
//! backend or consumer types (D10). Backends consume these to create resources.

use crate::handle::ShaderHandle;
use crate::types::{BufferUsage, Format, MemoryUsage, PresentMode, TextureUsage};

/// Describes a GPU buffer to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: u64,
    /// How the buffer may be used.
    pub usage: BufferUsage,
    /// Residency intent (plan §7).
    pub memory: MemoryUsage,
}

/// Describes a texture to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub usage: TextureUsage,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
}

/// Describes an offscreen render target (plan §15 / D13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
}

/// A shader module's source. SPIR-V to start (plan §14); other formats later.
#[derive(Debug, Clone, Copy)]
pub struct ShaderDesc<'a> {
    /// SPIR-V words as raw bytes (length must be a multiple of 4).
    pub spirv: &'a [u8],
}

/// Describes a compute pipeline.
#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineDesc<'a> {
    pub shader: ShaderHandle,
    /// Entry-point name in the shader module.
    pub entry: &'a str,
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

/// Describes a sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerDesc {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub address: AddressMode,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
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

/// The layout of one vertex buffer.
#[derive(Debug, Clone, Copy)]
pub struct VertexLayout<'a> {
    /// Bytes between consecutive vertices.
    pub stride: u32,
    pub attributes: &'a [VertexAttribute],
}

/// Color blending for a graphics pipeline (kept simple to start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Source replaces destination.
    #[default]
    Opaque,
    /// Standard src-alpha over (premultiplied-friendly) blend.
    AlphaBlend,
}

/// Depth test/write configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DepthState {
    /// Whether the depth test is enabled.
    pub test: bool,
    /// Whether depth is written.
    pub write: bool,
}

/// Describes a graphics pipeline (3D-capable from day one — plan D12).
#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    pub vertex_layout: VertexLayout<'a>,
    pub topology: PrimitiveTopology,
    /// Color attachment format.
    pub color_format: Format,
    /// Depth attachment format, if any.
    pub depth_format: Option<Format>,
    pub depth: DepthState,
    pub blend: BlendMode,
    /// MSAA sample count; `1` means no multisampling.
    pub samples: u32,
}

/// Configuration for a presentable surface (plan D11).
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
