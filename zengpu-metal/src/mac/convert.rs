//! Maps HAL value types to their Metal counterparts.

use metal::{
    MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode, MTLPixelFormat,
    MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLSamplerAddressMode, MTLSamplerBorderColor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLStencilOperation, MTLVertexFormat,
    MTLVertexStepFunction, MTLWinding,
};
use zengpu_hal::{
    AddressMode, BlendFactor, BlendOp, BorderColor, CompareFn, CullMode, FilterMode, Format,
    FrontFace, PrimitiveTopology, StencilOp, StepMode, VertexFormat,
};

pub(crate) fn pixel_format(format: Format) -> MTLPixelFormat {
    match format {
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::Depth32Float => MTLPixelFormat::Depth32Float,
        Format::Depth24PlusStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

pub(crate) fn has_stencil(format: Format) -> bool {
    matches!(format, Format::Depth24PlusStencil8)
}

pub(crate) fn vertex_format(format: VertexFormat) -> MTLVertexFormat {
    match format {
        VertexFormat::Float16x2 => MTLVertexFormat::Half2,
        VertexFormat::Float16x4 => MTLVertexFormat::Half4,
        VertexFormat::Float32 => MTLVertexFormat::Float,
        VertexFormat::Float32x2 => MTLVertexFormat::Float2,
        VertexFormat::Float32x3 => MTLVertexFormat::Float3,
        VertexFormat::Float32x4 => MTLVertexFormat::Float4,
        VertexFormat::Snorm8x4 => MTLVertexFormat::Char4Normalized,
        VertexFormat::Unorm16x2 => MTLVertexFormat::UShort2Normalized,
        VertexFormat::Unorm16x4 => MTLVertexFormat::UShort4Normalized,
        VertexFormat::Sint32x2 => MTLVertexFormat::Int2,
        VertexFormat::Sint32x3 => MTLVertexFormat::Int3,
        VertexFormat::Sint32x4 => MTLVertexFormat::Int4,
        VertexFormat::Uint32 => MTLVertexFormat::UInt,
        VertexFormat::Uint8x4Unorm => MTLVertexFormat::UChar4Normalized,
    }
}

pub(crate) fn step_function(step: StepMode) -> MTLVertexStepFunction {
    match step {
        StepMode::Vertex => MTLVertexStepFunction::PerVertex,
        StepMode::Instance => MTLVertexStepFunction::PerInstance,
    }
}

pub(crate) fn primitive(
    topology: PrimitiveTopology,
) -> (MTLPrimitiveType, MTLPrimitiveTopologyClass) {
    match topology {
        PrimitiveTopology::TriangleList => (
            MTLPrimitiveType::Triangle,
            MTLPrimitiveTopologyClass::Triangle,
        ),
        PrimitiveTopology::TriangleStrip => (
            MTLPrimitiveType::TriangleStrip,
            MTLPrimitiveTopologyClass::Triangle,
        ),
        PrimitiveTopology::LineList => (MTLPrimitiveType::Line, MTLPrimitiveTopologyClass::Line),
        PrimitiveTopology::PointList => (MTLPrimitiveType::Point, MTLPrimitiveTopologyClass::Point),
    }
}

pub(crate) fn blend_factor(factor: BlendFactor) -> MTLBlendFactor {
    match factor {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::OneMinusSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::DstColor => MTLBlendFactor::DestinationColor,
        BlendFactor::OneMinusDstColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::Src1Color => MTLBlendFactor::Source1Color,
        BlendFactor::OneMinusSrc1Color => MTLBlendFactor::OneMinusSource1Color,
        BlendFactor::Src1Alpha => MTLBlendFactor::Source1Alpha,
        BlendFactor::OneMinusSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

pub(crate) fn blend_op(op: BlendOp) -> MTLBlendOperation {
    match op {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

pub(crate) fn compare(f: CompareFn) -> MTLCompareFunction {
    match f {
        CompareFn::Never => MTLCompareFunction::Never,
        CompareFn::Less => MTLCompareFunction::Less,
        CompareFn::Equal => MTLCompareFunction::Equal,
        CompareFn::LessEqual => MTLCompareFunction::LessEqual,
        CompareFn::Greater => MTLCompareFunction::Greater,
        CompareFn::GreaterEqual => MTLCompareFunction::GreaterEqual,
        CompareFn::NotEqual => MTLCompareFunction::NotEqual,
        CompareFn::Always => MTLCompareFunction::Always,
    }
}

pub(crate) fn stencil_op(op: StencilOp) -> MTLStencilOperation {
    match op {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

pub(crate) fn cull(mode: CullMode) -> MTLCullMode {
    match mode {
        CullMode::None => MTLCullMode::None,
        CullMode::Front => MTLCullMode::Front,
        CullMode::Back => MTLCullMode::Back,
    }
}

/// ZSL vertex shaders flip clip y on Metal to match Vulkan's +y down. The flip mirrors triangles:
/// winding reads opposite on screen.
pub(crate) fn winding(face: FrontFace) -> MTLWinding {
    match face {
        FrontFace::Ccw => MTLWinding::Clockwise,
        FrontFace::Cw => MTLWinding::CounterClockwise,
    }
}

pub(crate) fn min_mag_filter(f: FilterMode) -> MTLSamplerMinMagFilter {
    match f {
        FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

pub(crate) fn mip_filter(f: FilterMode, mip_levels_wanted: bool) -> MTLSamplerMipFilter {
    if !mip_levels_wanted {
        return MTLSamplerMipFilter::NotMipmapped;
    }
    match f {
        FilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

pub(crate) fn address(mode: AddressMode) -> MTLSamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

pub(crate) fn border(color: BorderColor) -> MTLSamplerBorderColor {
    match color {
        BorderColor::TransparentBlack => MTLSamplerBorderColor::TransparentBlack,
        BorderColor::OpaqueBlack => MTLSamplerBorderColor::OpaqueBlack,
        BorderColor::OpaqueWhite => MTLSamplerBorderColor::OpaqueWhite,
    }
}
