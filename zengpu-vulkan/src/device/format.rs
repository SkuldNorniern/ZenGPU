//! Pure HAL-to-Vulkan enum/flag conversion helpers, shared across the
//! `device` submodules.

use ash::vk;
use zengpu_hal::{
    AddressMode, BlendComponent, BlendFactor, BlendOp, BorderColor, BufferUsage, CompareFn,
    CullMode, FilterMode, Format, FrontFace, MemoryUsage, PolygonMode, PrimitiveTopology, StepMode,
    VertexFormat,
};

pub(crate) fn filter_to_vk(f: FilterMode) -> vk::Filter {
    match f {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

pub(crate) fn address_to_vk(a: AddressMode) -> vk::SamplerAddressMode {
    match a {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

pub(crate) fn border_color_to_vk(b: BorderColor) -> vk::BorderColor {
    match b {
        BorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        BorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        BorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}

pub(crate) fn hal_format_to_vk(format: Format) -> vk::Format {
    match format {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
        Format::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
    }
}

pub(crate) fn buffer_usage_to_vk(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsage::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(BufferUsage::TRANSFER_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::TRANSFER_DST) || usage.contains(BufferUsage::READBACK) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if flags.is_empty() {
        flags = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    }
    flags
}

pub(crate) fn memory_usage_to_vk(usage: MemoryUsage) -> vk::MemoryPropertyFlags {
    match usage {
        MemoryUsage::GpuOnly | MemoryUsage::Pooled => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryUsage::Upload | MemoryUsage::Transient | MemoryUsage::Persistent => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUsage::CpuToGpu => {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUsage::Readback => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
        }
    }
}

pub(crate) fn memory_usage_fallback(usage: MemoryUsage) -> Option<vk::MemoryPropertyFlags> {
    match usage {
        MemoryUsage::CpuToGpu | MemoryUsage::Readback => {
            Some(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        }
        _ => None,
    }
}

pub(crate) fn vertex_format_to_vk(f: VertexFormat) -> vk::Format {
    match f {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Uint8x4Unorm => vk::Format::R8G8B8A8_UNORM,
    }
}

pub(crate) fn step_mode_to_vk(s: StepMode) -> vk::VertexInputRate {
    match s {
        StepMode::Vertex => vk::VertexInputRate::VERTEX,
        StepMode::Instance => vk::VertexInputRate::INSTANCE,
    }
}

pub(crate) fn topology_to_vk(t: PrimitiveTopology) -> vk::PrimitiveTopology {
    match t {
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
    }
}

pub(crate) fn blend_component_to_vk(
    c: BlendComponent,
) -> (vk::BlendFactor, vk::BlendFactor, vk::BlendOp) {
    let factor = |factor| match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::Src1Color => vk::BlendFactor::SRC1_COLOR,
        BlendFactor::OneMinusSrc1Color => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        BlendFactor::Src1Alpha => vk::BlendFactor::SRC1_ALPHA,
        BlendFactor::OneMinusSrc1Alpha => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
    };
    let op = match c.op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    };
    (factor(c.src_factor), factor(c.dst_factor), op)
}

pub(crate) fn cull_mode_to_vk(c: CullMode) -> vk::CullModeFlags {
    match c {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

pub(crate) fn front_face_to_vk(f: FrontFace) -> vk::FrontFace {
    match f {
        FrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
        FrontFace::Cw => vk::FrontFace::CLOCKWISE,
    }
}

pub(crate) fn polygon_mode_to_vk(p: PolygonMode) -> vk::PolygonMode {
    match p {
        PolygonMode::Fill => vk::PolygonMode::FILL,
        PolygonMode::Line => vk::PolygonMode::LINE,
        PolygonMode::Point => vk::PolygonMode::POINT,
    }
}

pub(crate) fn compare_fn_to_vk(c: CompareFn) -> vk::CompareOp {
    match c {
        CompareFn::Never => vk::CompareOp::NEVER,
        CompareFn::Less => vk::CompareOp::LESS,
        CompareFn::Equal => vk::CompareOp::EQUAL,
        CompareFn::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFn::Greater => vk::CompareOp::GREATER,
        CompareFn::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFn::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFn::Always => vk::CompareOp::ALWAYS,
    }
}

pub(crate) fn sample_count_to_vk(samples: u32) -> vk::SampleCountFlags {
    match samples {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}
