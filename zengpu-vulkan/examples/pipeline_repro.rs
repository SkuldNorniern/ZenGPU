//! Isolation repro for a `vkCreateGraphicsPipelines` fault.
//!
//! Builds the exact pipeline `asterra` builds (3×Vec3 vertex inputs, a
//! push-constant block of two mat4, one Vec3 varying, depth on), but from
//! glslang-compiled GLSL — i.e. known-good SPIR-V. If this succeeds while the
//! ZSL-generated path faults, the defect is in ZSL's SPIR-V; if it also faults,
//! the defect is in the pipeline-state code or the driver.
//!
//! Run: cargo run -p zengpu-vulkan --example pipeline_repro

use zengpu_hal::{
    ColorTargetState, DepthState, DeviceRequest, Format, GpuDevice, GraphicsDevice,
    GraphicsPipelineDesc, PrimitiveTopology, ShaderDesc, StepMode, VertexAttribute, VertexFormat,
    VertexLayout,
};
use zengpu_spirv::{ZslShader, zsl};
use zengpu_vulkan::{DepthTarget, OffscreenTarget, VulkanInstance};

const VS: ZslShader = zsl!(
    push P { model: mat4x4<f32>, view_proj: mat4x4<f32> }
    vertex vs(
        @location(0) pos: f32x3,
        @location(1) col: f32x3,
        @location(2) nrm: f32x3,
        p: P,
    ) -> (f32x4, f32x3) {
        (p.view_proj * (p.model * pos.extend(1.0)), col)
    }
);

const FS: ZslShader = zsl!(
    fragment fs(@location(0) v_col: f32x3) -> f32x4 {
        v_col.extend(1.0)
    }
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
        .create_shader(ShaderDesc::spirv(bytemuck_cast(VS.spv)))
        .expect("vs");
    let fs = device
        .create_shader(ShaderDesc::spirv(bytemuck_cast(FS.spv)))
        .expect("fs");

    eprintln!("repro: create_graphics_pipeline");
    let pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc {
        vertex_shader: vs,
        fragment_shader: fs,
        vertex_layouts: &[LAYOUT],
        topology: PrimitiveTopology::TriangleList,
        color_targets: &[ColorTargetState {
            format: Format::Rgba8Unorm,
            blend: None,
        }],
        depth_format: Some(Format::Depth32Float),
        depth: DepthState {
            test: true,
            write: true,
            ..Default::default()
        },
        raster: Default::default(),
        samples: 1,
    });

    match pipeline {
        Ok(_) => eprintln!("repro: PIPELINE CREATED OK — glslang SPIR-V works"),
        Err(e) => eprintln!("repro: pipeline error: {e}"),
    }

    eprintln!(
        "\n=== glslang VERTEX disassembly ===\n{}",
        zengpu_spv::disassemble(VS.spv)
    );
    eprintln!(
        "\n=== glslang FRAGMENT disassembly ===\n{}",
        zengpu_spv::disassemble(FS.spv)
    );
}

/// Reinterpret a `&[u32]` SPIR-V slice as `&[u8]` without pulling in bytemuck.
fn bytemuck_cast(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) }
}
