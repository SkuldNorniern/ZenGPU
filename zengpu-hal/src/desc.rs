//! Plain descriptor data with no backend types.

use crate::handle::ShaderHandle;
use crate::types::{BufferUsage, Format, MemoryUsage, PresentMode, TextureUsage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub usage: TextureUsage,
    /// MSAA sample count. `1` means no multisampling.
    pub samples: u32,
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
pub struct ShaderDesc<'a> {
    /// SPIR-V words as raw bytes. The length must be a multiple of 4.
    pub spirv: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineDesc<'a> {
    pub shader: ShaderHandle,
    pub entry: &'a str,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    /// Byte offset within the vertex.
    pub offset: u32,
    pub format: VertexFormat,
}

#[derive(Debug, Clone, Copy)]
pub struct VertexLayout<'a> {
    /// Bytes between consecutive vertices.
    pub stride: u32,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    AlphaBlend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DepthState {
    pub test: bool,
    pub write: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    pub vertex_layout: VertexLayout<'a>,
    pub topology: PrimitiveTopology,
    pub color_format: Format,
    pub depth_format: Option<Format>,
    pub depth: DepthState,
    pub blend: BlendMode,
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
