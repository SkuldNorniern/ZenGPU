//! Render command recording straight into an `MTLCommandBuffer`.

use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use metal::{
    Buffer, CommandBuffer, MTLClearColor, MTLIndexType, MTLLoadAction, MTLResourceUsage,
    MTLScissorRect, MTLStoreAction, MTLViewport, RenderCommandEncoder, RenderPassDescriptor,
    ResourceRef, Texture,
};
use objc::rc::autoreleasepool;
use objc::{msg_send, sel, sel_impl};
use zengpu_hal::{
    Bindings, BufferHandle, LoadOp, PipelineHandle, QueryPoolHandle, RenderCommands,
    RenderPassDesc, TargetHandle, ViewportScissor,
};

use super::convert;
use super::device::{Inner, MetalPipeline, RenderPipeline, pack_scalars};
use super::table::{PUSH_BUFFER_INDEX, TEXTURE_BUFFER_INDEX};

/// Bytes per tightly packed `MTLDrawPrimitivesIndirectArguments`.
const DRAW_INDIRECT_STRIDE: u64 = 16;
/// Bytes per tightly packed `MTLDrawIndexedPrimitivesIndirectArguments`.
const DRAW_INDEXED_INDIRECT_STRIDE: u64 = 20;

/// Invalid calls (a stale handle, or a draw outside a pass) skip the command and are logged.
///
/// Methods return nothing.
pub struct MetalCommandList {
    inner: Arc<Inner>,
    cmd: Option<CommandBuffer>,
    encoder: Option<RenderCommandEncoder>,
    pipeline: Option<Arc<RenderPipeline>>,
    index: Option<Buffer>,
    /// The current pass's attachment size, which scissors must stay inside.
    extent: (u32, u32),
}

// SAFETY: used only through `&mut self`, so one thread encodes at a time.
unsafe impl Send for MetalCommandList {}

impl MetalCommandList {
    pub(crate) fn new(inner: Arc<Inner>) -> Self {
        let cmd = inner.command_buffer();
        Self {
            inner,
            cmd: Some(cmd),
            encoder: None,
            pipeline: None,
            index: None,
            extent: (0, 0),
        }
    }

    pub(crate) fn finish(mut self) -> CommandBuffer {
        self.end_encoder();
        match self.cmd.take() {
            Some(cmd) => cmd,
            None => self.inner.command_buffer(),
        }
    }

    fn end_encoder(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.end_encoding();
        }
        self.pipeline = None;
    }

    fn target(&self, handle: TargetHandle) -> Option<(Texture, zengpu_hal::Format)> {
        let targets = self.inner.targets.lock().ok()?;
        targets.get(handle).map(|t| (t.tex.clone(), t.format))
    }

    fn encoder(&self, what: &str) -> Option<&RenderCommandEncoder> {
        if self.encoder.is_none() {
            log::error!("[zengpu-metal] {what} outside a render pass; skipped");
        }
        self.encoder.as_ref()
    }

    fn full_scissor(&self) -> MTLScissorRect {
        MTLScissorRect {
            x: 0,
            y: 0,
            width: u64::from(self.extent.0),
            height: u64::from(self.extent.1),
        }
    }
}

impl Drop for MetalCommandList {
    fn drop(&mut self) {
        self.end_encoder();
    }
}

fn load_action(load: LoadOp) -> MTLLoadAction {
    match load {
        LoadOp::Clear(_) => MTLLoadAction::Clear,
        LoadOp::Load => MTLLoadAction::Load,
        LoadOp::DontCare => MTLLoadAction::DontCare,
    }
}

fn store_action(store: bool) -> MTLStoreAction {
    if store {
        MTLStoreAction::Store
    } else {
        MTLStoreAction::DontCare
    }
}

impl RenderCommands for MetalCommandList {
    fn push_debug_group(&mut self, label: &str) {
        match (&self.encoder, &self.cmd) {
            (Some(enc), _) => enc.push_debug_group(label),
            (None, Some(cmd)) => cmd.push_debug_group(label),
            (None, None) => {}
        }
    }

    fn pop_debug_group(&mut self) {
        match (&self.encoder, &self.cmd) {
            (Some(enc), _) => enc.pop_debug_group(),
            (None, Some(cmd)) => cmd.pop_debug_group(),
            (None, None) => {}
        }
    }

    fn insert_debug_label(&mut self, label: &str) {
        if let Some(enc) = &self.encoder {
            enc.insert_debug_signpost(label);
        }
    }

    fn write_timestamp(&mut self, _pool: QueryPoolHandle, _index: u32) {
    }

    fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        self.end_encoder();
        let Some(cmd) = self.cmd.as_ref() else { return };
        let rpd = RenderPassDescriptor::new();
        let mut extent = None;
        for (i, att) in desc.color.iter().enumerate() {
            let Some((tex, _)) = self.target(att.target) else {
                log::error!("[zengpu-metal] stale color target; render pass skipped");
                return;
            };
            let Some(ca) = rpd.color_attachments().object_at(i as u64) else {
                return;
            };
            ca.set_texture(Some(&tex));
            ca.set_load_action(load_action(att.load));
            if let LoadOp::Clear([r, g, b, a]) = att.load {
                ca.set_clear_color(MTLClearColor::new(
                    f64::from(r),
                    f64::from(g),
                    f64::from(b),
                    f64::from(a),
                ));
            }
            ca.set_store_action(store_action(att.store));
            extent.get_or_insert((tex.width() as u32, tex.height() as u32));
        }
        if let Some(depth) = desc.depth {
            let Some((tex, format)) = self.target(depth.target) else {
                log::error!("[zengpu-metal] stale depth target; render pass skipped");
                return;
            };
            if let Some(da) = rpd.depth_attachment() {
                da.set_texture(Some(&tex));
                da.set_load_action(load_action(depth.load));
                if let LoadOp::Clear([d, ..]) = depth.load {
                    da.set_clear_depth(f64::from(d));
                }
                da.set_store_action(store_action(depth.store));
            }
            if convert::has_stencil(format)
                && let Some(sa) = rpd.stencil_attachment()
            {
                sa.set_texture(Some(&tex));
                sa.set_load_action(load_action(depth.load));
                sa.set_clear_stencil(0);
                sa.set_store_action(store_action(depth.store));
            }
            extent.get_or_insert((tex.width() as u32, tex.height() as u32));
        }
        self.extent = extent.unwrap_or((0, 0));

        let resident = self
            .inner
            .table
            .lock()
            .map(|t| t.resident())
            .unwrap_or_default();
        let table = self.inner.table.lock().ok().map(|t| t.buffer().clone());
        let enc = autoreleasepool(|| cmd.new_render_command_encoder(rpd).to_owned());
        if let Some(table) = &table {
            enc.set_vertex_buffer(TEXTURE_BUFFER_INDEX, Some(table), 0);
            enc.set_fragment_buffer(TEXTURE_BUFFER_INDEX, Some(table), 0);
        }
        if !resident.is_empty() {
            let refs: Vec<&ResourceRef> = resident
                .iter()
                .map(|t| t.as_ref() as &ResourceRef)
                .collect();
            enc.use_resources(
                &refs,
                MTLResourceUsage::Read,
                metal::MTLRenderStages::Vertex | metal::MTLRenderStages::Fragment,
            );
        }
        self.encoder = Some(enc);
    }

    fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        let found = self
            .inner
            .pipelines
            .lock()
            .ok()
            .and_then(|p| match p.get(pipeline) {
                Some(MetalPipeline::Render(r)) => Some(Arc::clone(r)),
                _ => None,
            });
        let Some(p) = found else {
            log::error!("[zengpu-metal] set_pipeline: not a graphics pipeline; skipped");
            return;
        };
        let Some(enc) = self.encoder("set_pipeline") else {
            return;
        };
        enc.set_render_pipeline_state(&p.state);
        if let Some(depth) = &p.depth {
            enc.set_depth_stencil_state(depth);
        }
        enc.set_cull_mode(p.cull);
        enc.set_front_facing_winding(p.winding);
        enc.set_triangle_fill_mode(p.fill);
        self.pipeline = Some(p);
    }

    fn set_viewport_scissor(&mut self, vs: ViewportScissor) {
        let full = self.full_scissor();
        let Some(enc) = self.encoder("set_viewport_scissor") else {
            return;
        };
        let v = vs.viewport;
        enc.set_viewport(MTLViewport {
            originX: f64::from(v.x),
            originY: f64::from(v.y),
            width: f64::from(v.width),
            height: f64::from(v.height),
            znear: f64::from(v.min_depth),
            zfar: f64::from(v.max_depth),
        });
        let rect = match vs.scissor {
            None => full,
            Some(r) => {
                let x0 = (r.x.max(0.0) as u64).min(full.width);
                let y0 = (r.y.max(0.0) as u64).min(full.height);
                let x1 = ((r.x + r.width).max(0.0).ceil() as u64).clamp(x0, full.width);
                let y1 = ((r.y + r.height).max(0.0).ceil() as u64).clamp(y0, full.height);
                MTLScissorRect {
                    x: x0,
                    y: y0,
                    width: x1 - x0,
                    height: y1 - y0,
                }
            }
        };
        enc.set_scissor_rect(rect);
    }

    fn bind(&mut self, bindings: Bindings<'_>) {
        let Some(enc) = self.encoder("bind") else {
            return;
        };
        if !bindings.buffers.is_empty() {
            log::warn!("[zengpu-metal] graphics storage buffers are not supported; ignored");
        }
        if !bindings.scalars.is_empty() {
            let packed = pack_scalars(bindings.scalars);
            let (len, ptr) = (packed.len() as u64, packed.as_ptr() as *const c_void);
            enc.set_vertex_bytes(PUSH_BUFFER_INDEX, len, ptr);
            enc.set_fragment_bytes(PUSH_BUFFER_INDEX, len, ptr);
        }
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle) {
        let buf = self
            .inner
            .buffers
            .lock()
            .ok()
            .and_then(|b| b.get(buffer).map(|b| b.buf.clone()));
        let Some(buf) = buf else {
            log::error!("[zengpu-metal] set_vertex_buffer: stale buffer; skipped");
            return;
        };
        let Some(enc) = self.encoder("set_vertex_buffer") else {
            return;
        };
        enc.set_vertex_buffer(u64::from(slot), Some(&buf), 0);
    }

    fn set_index_buffer(&mut self, buffer: BufferHandle) {
        self.index = self
            .inner
            .buffers
            .lock()
            .ok()
            .and_then(|b| b.get(buffer).map(|b| b.buf.clone()));
        if self.index.is_none() {
            log::error!("[zengpu-metal] set_index_buffer: stale buffer");
        }
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        let Some(p) = self.pipeline.clone() else {
            log::error!("[zengpu-metal] draw without a pipeline; skipped");
            return;
        };
        let Some(enc) = self.encoder("draw") else {
            return;
        };
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        enc.draw_primitives_instanced_base_instance(
            p.primitive,
            u64::from(vertices.start),
            u64::from(vertices.end - vertices.start),
            u64::from(instances.end - instances.start),
            u64::from(instances.start),
        );
    }

    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>) {
        let (Some(p), Some(index)) = (self.pipeline.clone(), self.index.clone()) else {
            log::error!("[zengpu-metal] draw_indexed without a pipeline or index buffer; skipped");
            return;
        };
        let Some(enc) = self.encoder("draw_indexed") else {
            return;
        };
        if indices.is_empty() || instances.is_empty() {
            return;
        }
        let index_count = u64::from(indices.end - indices.start);
        let offset = u64::from(indices.start) * 4;
        let instance_count = u64::from(instances.end - instances.start);
        let base_instance = u64::from(instances.start);
        let base_vertex: i64 = 0;
        let enc_ref: &metal::RenderCommandEncoderRef = enc;
        unsafe {
            let _: () = msg_send![enc_ref,
                drawIndexedPrimitives: p.primitive
                indexCount: index_count
                indexType: MTLIndexType::UInt32
                indexBuffer: index.as_ref() as &metal::BufferRef
                indexBufferOffset: offset
                instanceCount: instance_count
                baseVertex: base_vertex
                baseInstance: base_instance
            ];
        }
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        let Some(p) = self.pipeline.clone() else {
            log::error!("[zengpu-metal] draw_indirect without a pipeline; skipped");
            return;
        };
        let buf = self
            .inner
            .buffers
            .lock()
            .ok()
            .and_then(|b| b.get(buffer).map(|b| b.buf.clone()));
        let Some(buf) = buf else { return };
        let Some(enc) = self.encoder("draw_indirect") else {
            return;
        };
        let stride = if stride == 0 {
            DRAW_INDIRECT_STRIDE
        } else {
            u64::from(stride)
        };
        for i in 0..u64::from(draw_count) {
            enc.draw_primitives_indirect(p.primitive, &buf, offset + i * stride);
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let (Some(p), Some(index)) = (self.pipeline.clone(), self.index.clone()) else {
            log::error!(
                "[zengpu-metal] draw_indexed_indirect without a pipeline or index buffer; skipped"
            );
            return;
        };
        let buf = self
            .inner
            .buffers
            .lock()
            .ok()
            .and_then(|b| b.get(buffer).map(|b| b.buf.clone()));
        let Some(buf) = buf else { return };
        let Some(enc) = self.encoder("draw_indexed_indirect") else {
            return;
        };
        let stride = if stride == 0 {
            DRAW_INDEXED_INDIRECT_STRIDE
        } else {
            u64::from(stride)
        };
        for i in 0..u64::from(draw_count) {
            enc.draw_indexed_primitives_indirect(
                p.primitive,
                MTLIndexType::UInt32,
                &index,
                0,
                &buf,
                offset + i * stride,
            );
        }
    }

    fn end_render_pass(&mut self) {
        self.end_encoder();
    }
}
