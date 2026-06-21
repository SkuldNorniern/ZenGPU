//! Isolation repro for a `vkCreateGraphicsPipelines` fault.
//!
//! Builds the exact pipeline `asterra` builds (3×Vec3 vertex inputs, a
//! push-constant block of two mat4, one Vec3 varying, depth on), but from
//! glslang-compiled GLSL — i.e. known-good SPIR-V. If this succeeds while the
//! ZSL-generated path faults, the defect is in ZSL's SPIR-V; if it also faults,
//! the defect is in the pipeline-state code or the driver.
//!
//! Run: cargo run -p zengpu-vulkan --example pipeline_repro

use inline_spirv::inline_spirv;
use zengpu_hal::{
    BlendMode, DepthState, DeviceRequest, Format, GpuDevice, GraphicsDevice, GraphicsPipelineDesc,
    PrimitiveTopology, ShaderDesc, StepMode, VertexAttribute, VertexFormat, VertexLayout,
};
use zengpu_vulkan::{DepthTarget, OffscreenTarget, VulkanInstance};

const VS: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec3 pos;
    layout(location = 1) in vec3 col;
    layout(location = 2) in vec3 nrm;
    layout(location = 0) out vec3 v_col;
    layout(push_constant) uniform PC { mat4 model; mat4 view_proj; } pc;
    void main() {
        gl_Position = pc.view_proj * (pc.model * vec4(pos, 1.0));
        v_col = col;
    }
    "#,
    vert,
    glsl,
    vulkan1_2
);

const FS: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec3 v_col;
    layout(location = 0) out vec4 o_col;
    void main() { o_col = vec4(v_col, 1.0); }
    "#,
    frag,
    glsl,
    vulkan1_2
);

const ATTRS: &[VertexAttribute] = &[
    VertexAttribute {
        location: 0,
        offset: 0,
        format: VertexFormat::Float32x3,
    },
    VertexAttribute {
        location: 1,
        offset: 12,
        format: VertexFormat::Float32x3,
    },
    VertexAttribute {
        location: 2,
        offset: 24,
        format: VertexFormat::Float32x3,
    },
];
const LAYOUT: VertexLayout<'static> = VertexLayout {
    stride: 36,
    attributes: ATTRS,
    step_mode: StepMode::Vertex,
};

fn main() {
    eprintln!("repro: opening device");
    let inst = VulkanInstance::new().expect("instance");
    let adapter = inst.request_vulkan_adapter().expect("adapter");
    let device = adapter
        .open_headless(DeviceRequest::default())
        .expect("device");

    let (w, h) = (256u32, 256u32);
    let _offscreen = OffscreenTarget::new(&device, Format::Rgba8Unorm, w, h).expect("offscreen");
    let depth = DepthTarget::new(&device.context(), w, h).expect("depth");
    let _depth_handle = device.register_depth_target(&depth);

    eprintln!("repro: create shaders (glslang SPIR-V)");
    let vs = device
        .create_shader(ShaderDesc::spirv(bytemuck_cast(VS)))
        .expect("vs");
    let fs = device
        .create_shader(ShaderDesc::spirv(bytemuck_cast(FS)))
        .expect("fs");

    eprintln!("repro: create_graphics_pipeline");
    let pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc {
        vertex_shader: vs,
        fragment_shader: fs,
        vertex_layouts: &[LAYOUT],
        topology: PrimitiveTopology::TriangleList,
        color_format: Format::Rgba8Unorm,
        depth_format: Some(Format::Depth32Float),
        depth: DepthState {
            test: true,
            write: true,
            ..Default::default()
        },
        blend: BlendMode::Opaque,
        samples: 1,
    });

    match pipeline {
        Ok(_) => eprintln!("repro: PIPELINE CREATED OK — glslang SPIR-V works"),
        Err(e) => eprintln!("repro: pipeline error: {e}"),
    }

    eprintln!(
        "\n=== glslang VERTEX disassembly ===\n{}",
        zengpu_spv::disassemble(VS)
    );
    eprintln!(
        "\n=== glslang FRAGMENT disassembly ===\n{}",
        zengpu_spv::disassemble(FS)
    );
}

/// Reinterpret a `&[u32]` SPIR-V slice as `&[u8]` without pulling in bytemuck.
fn bytemuck_cast(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) }
}
