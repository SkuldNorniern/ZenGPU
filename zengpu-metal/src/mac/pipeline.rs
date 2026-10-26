//! Builds graphics pipelines from the HAL's backend-neutral descriptor.

use std::sync::Arc;

use metal::{
    DepthStencilDescriptor, FunctionRef, Library, MTLPixelFormat, MTLTriangleFillMode,
    RenderPipelineDescriptor, StencilDescriptor, VertexDescriptor,
};
use zengpu_hal::{
    GpuError, GraphicsPipelineDesc, PipelineHandle, PolygonMode, Result, StencilFaceState,
};

use super::convert;
use super::device::{MetalDevice, MetalPipeline, RenderPipeline, lock};
use super::table::MAX_VERTEX_BUFFERS;

/// The name ZSL gives every MSL entry point.
const ZSL_ENTRY: &str = "zsl_main";

/// The library's entry point: ZSL's, or the only function there is.
fn entry(library: &Library) -> Result<metal::Function> {
    if let Ok(f) = library.get_function(ZSL_ENTRY, None) {
        return Ok(f);
    }
    match library.function_names().as_slice() {
        [only] => library
            .get_function(only, None)
            .map_err(|e| GpuError::PipelineCreation(format!("metal: function `{only}`: {e}"))),
        _ => Err(GpuError::PipelineCreation(format!(
            "metal: shader has no `{ZSL_ENTRY}` and more than one function"
        ))),
    }
}

fn stencil_face(face: StencilFaceState, read_mask: u32, write_mask: u32) -> StencilDescriptor {
    let d = StencilDescriptor::new();
    d.set_stencil_compare_function(convert::compare(face.compare));
    d.set_stencil_failure_operation(convert::stencil_op(face.fail_op));
    d.set_depth_failure_operation(convert::stencil_op(face.depth_fail_op));
    d.set_depth_stencil_pass_operation(convert::stencil_op(face.pass_op));
    d.set_read_mask(read_mask);
    d.set_write_mask(write_mask);
    d
}

fn vertex_descriptor(desc: &GraphicsPipelineDesc<'_>) -> Result<Option<VertexDescriptor>> {
    if desc.vertex_layouts.is_empty() {
        return Ok(None);
    }
    if desc.vertex_layouts.len() > MAX_VERTEX_BUFFERS as usize {
        return Err(GpuError::PipelineCreation(format!(
            "metal: at most {MAX_VERTEX_BUFFERS} vertex buffers"
        )));
    }
    let vd = VertexDescriptor::new().to_owned();
    for (slot, layout) in desc.vertex_layouts.iter().enumerate() {
        let l = vd
            .layouts()
            .object_at(slot as u64)
            .ok_or_else(|| GpuError::PipelineCreation("metal: vertex layout slot".into()))?;
        l.set_stride(u64::from(layout.stride));
        l.set_step_function(convert::step_function(layout.step_mode));
        l.set_step_rate(1);
        for attr in layout.attributes {
            let a = vd
                .attributes()
                .object_at(u64::from(attr.location))
                .ok_or_else(|| {
                    GpuError::PipelineCreation("metal: vertex attribute location".into())
                })?;
            a.set_format(convert::vertex_format(attr.format));
            a.set_offset(u64::from(attr.offset));
            a.set_buffer_index(slot as u64);
        }
    }
    Ok(Some(vd))
}

impl MetalDevice {
    pub(crate) fn create_render_pipeline(
        &self,
        desc: GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle> {
        let fill = match desc.raster.polygon {
            PolygonMode::Fill => MTLTriangleFillMode::Fill,
            PolygonMode::Line => MTLTriangleFillMode::Lines,
            PolygonMode::Point => {
                return Err(GpuError::PipelineCreation(
                    "metal: point polygon mode is not supported".into(),
                ));
            }
        };
        let (vs, fs, texture_table) = {
            let shaders = lock(&self.inner.shaders)?;
            let get = |h, what: &str| {
                shaders
                    .get(h)
                    .ok_or_else(|| {
                        GpuError::PipelineCreation(format!("stale {what} shader handle"))
                    })
                    .and_then(|shader| entry(&shader.library))
            };
            (
                get(desc.vertex_shader, "vertex")?,
                get(desc.fragment_shader, "fragment")?,
                [desc.vertex_shader, desc.fragment_shader]
                    .iter()
                    .any(|h| shaders.get(*h).is_some_and(|s| s.texture_table)),
            )
        };
        let (primitive, class) = convert::primitive(desc.topology);

        let pd = RenderPipelineDescriptor::new();
        pd.set_vertex_function(Some(vs.as_ref() as &FunctionRef));
        pd.set_fragment_function(Some(fs.as_ref() as &FunctionRef));
        pd.set_input_primitive_topology(class);
        pd.set_raster_sample_count(u64::from(desc.samples.max(1)));
        if let Some(vd) = vertex_descriptor(&desc)? {
            pd.set_vertex_descriptor(Some(&vd));
        }
        for (i, target) in desc.color_targets.iter().enumerate() {
            let ca = pd.color_attachments().object_at(i as u64).ok_or_else(|| {
                GpuError::PipelineCreation("metal: color attachment index".into())
            })?;
            ca.set_pixel_format(convert::pixel_format(target.format));
            if let Some(blend) = target.blend {
                ca.set_blending_enabled(true);
                ca.set_source_rgb_blend_factor(convert::blend_factor(blend.color.src_factor));
                ca.set_destination_rgb_blend_factor(convert::blend_factor(blend.color.dst_factor));
                ca.set_rgb_blend_operation(convert::blend_op(blend.color.op));
                ca.set_source_alpha_blend_factor(convert::blend_factor(blend.alpha.src_factor));
                ca.set_destination_alpha_blend_factor(convert::blend_factor(
                    blend.alpha.dst_factor,
                ));
                ca.set_alpha_blend_operation(convert::blend_op(blend.alpha.op));
            }
        }
        if let Some(depth_format) = desc.depth_format {
            pd.set_depth_attachment_pixel_format(convert::pixel_format(depth_format));
            if convert::has_stencil(depth_format) {
                pd.set_stencil_attachment_pixel_format(MTLPixelFormat::Depth32Float_Stencil8);
            }
        }
        let state = self
            .inner
            .device
            .new_render_pipeline_state(&pd)
            .map_err(|e| GpuError::PipelineCreation(format!("metal: render pipeline: {e}")))?;

        let depth = (desc.depth_format.is_some() || desc.stencil.is_some()).then(|| {
            let dd = DepthStencilDescriptor::new();
            if desc.depth.test {
                dd.set_depth_compare_function(convert::compare(desc.depth.compare));
            } else {
                dd.set_depth_compare_function(metal::MTLCompareFunction::Always);
            }
            dd.set_depth_write_enabled(desc.depth.test && desc.depth.write);
            if let Some(s) = desc.stencil {
                dd.set_front_face_stencil(Some(&stencil_face(s.front, s.read_mask, s.write_mask)));
                dd.set_back_face_stencil(Some(&stencil_face(s.back, s.read_mask, s.write_mask)));
            }
            self.inner.device.new_depth_stencil_state(&dd)
        });

        let pipeline = RenderPipeline {
            texture_table,
            state,
            depth,
            primitive,
            cull: convert::cull(desc.raster.cull),
            winding: convert::winding(desc.raster.front_face),
            fill,
        };
        Ok(lock(&self.inner.pipelines)?.insert(MetalPipeline::Render(Arc::new(pipeline))))
    }
}
