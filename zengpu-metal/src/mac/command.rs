//! Render command recording straight into an `MTLCommandBuffer`.

use std::ffi::c_void;
use std::mem::{size_of_val, take};
use std::ops::Range;
use std::sync::Arc;

use metal::{
    Buffer, CommandBuffer, MTLClearColor, MTLIndexType, MTLLoadAction, MTLResourceUsage,
    MTLScissorRect, MTLStoreAction, MTLViewport, RenderCommandEncoder, RenderCommandEncoderRef,
    RenderPassDescriptor, ResourceRef, Texture,
};
use objc::rc::autoreleasepool;
use objc::{msg_send, sel, sel_impl};
use zengpu_hal::{
    Bindings, BufferHandle, IndexFormat, LoadOp, PipelineHandle, QueryPoolHandle, RenderCommands,
    RenderPassDesc, Scalar, TargetHandle, ViewportScissor,
};

use super::convert;
use super::count::{self, Clamp, DRAW_WORDS, INDEXED_WORDS};
use super::device::{Inner, MetalPipeline, RenderPipeline};
use super::table::{
    DEPTH_TEXTURE_BUFFER_INDEX, GRAPHICS_BUFFER_INDEX, MAX_GRAPHICS_BUFFERS, PUSH_BUFFER_INDEX,
    TEXTURE_BUFFER_INDEX, TEXTURE_IDS_BUFFER_INDEX,
};
use super::timestamps;

/// The push block's size limit, as on Vulkan: 64 four-byte words.
const PUSH_BYTES: usize = 256;
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
    /// Bound index buffer, the byte offset its indices start at and their width.
    index: Option<(Buffer, u64, IndexFormat)>,
    binding_valid: bool,
    /// The current pass's attachment size, which scissors must stay inside.
    extent: (u32, u32),
    /// Work the GPU must do before this list: count driven draws clamping their arguments.
    pre: Option<CommandBuffer>,
    /// Timestamps written inside the open pass, taken when it ends.
    pass_samples: Vec<(QueryPoolHandle, u32)>,
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
            binding_valid: true,
            extent: (0, 0),
            pre: None,
            pass_samples: Vec::new(),
        }
    }

    pub(crate) fn finish(mut self) -> CommandBuffer {
        self.end_encoder();
        // committed first, so the queue runs it before the list
        if let Some(pre) = self.pre.take() {
            self.inner.commit(&pre);
        }
        match self.cmd.take() {
            Some(cmd) => cmd,
            None => self.inner.command_buffer(),
        }
    }

    fn end_encoder(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.end_encoding();
        }
        for (pool, index) in take(&mut self.pass_samples) {
            self.sample(pool, index);
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

    /// Storage buffers of a draw, one slot each from `GRAPHICS_BUFFER_INDEX` for both stages.
    /// False when a draw has more than Metal takes or names a buffer that is gone.
    fn bind_buffers(&self, enc: &RenderCommandEncoderRef, slots: &[u32]) -> bool {
        if slots.is_empty() {
            return true;
        }
        if slots.len() > MAX_GRAPHICS_BUFFERS {
            log::error!(
                "[zengpu-metal] {} storage buffers, a draw takes {MAX_GRAPHICS_BUFFERS}; draw skipped",
                slots.len()
            );
            return false;
        }
        let Ok(table) = self.inner.buffers.lock() else {
            return false;
        };
        for (at, &slot) in (GRAPHICS_BUFFER_INDEX..).zip(slots) {
            let Some(buffer) = table.get_by_slot_index(slot) else {
                log::error!("[zengpu-metal] stale storage buffer {slot}; draw skipped");
                return false;
            };
            enc.set_vertex_buffer(at, Some(&buffer.buf), 0);
            enc.set_fragment_buffer(at, Some(&buffer.buf), 0);
        }
        true
    }

    /// Arguments of a count driven draw, clamped to the count in a pass before this list. `None`
    /// when the draw cannot run: no pipeline or pass, a stale buffer, or arguments out of range.
    fn clamp(
        &mut self,
        what: &str,
        args: BufferHandle,
        count: BufferHandle,
        clamp: Clamp,
    ) -> Option<Buffer> {
        if !self.binding_valid || clamp.max_draws == 0 {
            return None;
        }
        if self.pipeline.is_none() || self.encoder.is_none() {
            log::error!("[zengpu-metal] {what} without a pipeline in a pass; skipped");
            return None;
        }
        let (args, count) = {
            let buffers = self.inner.buffers.lock().ok()?;
            match (buffers.get(args), buffers.get(count)) {
                (Some(a), Some(c)) => {
                    if let Err(why) = clamp.check(a.size, c.size) {
                        log::error!("[zengpu-metal] {what}: {why}; skipped");
                        return None;
                    }
                    (a.buf.clone(), c.buf.clone())
                }
                _ => {
                    log::error!("[zengpu-metal] {what}: stale buffer; skipped");
                    return None;
                }
            }
        };
        let pipeline = count::pipeline(&self.inner.count_clamp, &self.inner.device)?;
        let inner = Arc::clone(&self.inner);
        let pre = self.pre.get_or_insert_with(|| inner.command_buffer());
        Some(clamp.encode(&inner.device, &pipeline, pre, &args, &count))
    }

    /// Takes timestamp `index` of `pool` here, between encoders, and marks this list its writer.
    fn sample(&mut self, pool: QueryPoolHandle, index: u32) {
        let Some(cmd) = self.cmd.as_ref() else {
            return;
        };
        let Ok(mut pools) = self.inner.query_pools.lock() else {
            return;
        };
        let Some(p) = pools.get_mut(pool) else {
            log::error!("[zengpu-metal] write_timestamp: stale query pool; skipped");
            return;
        };
        let Some(writer) = p.writers.get_mut(index as usize) else {
            log::error!("[zengpu-metal] write_timestamp: {index} past the pool; skipped");
            return;
        };
        *writer = Some(cmd.clone());
        timestamps::sample(&self.inner.device, cmd, &p.samples, index);
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

    /// Apple GPUs sample only where encoders start or end: inside a pass the timestamp is taken
    /// when the pass ends.
    fn write_timestamp(&mut self, pool: QueryPoolHandle, index: u32) {
        if self.encoder.is_some() {
            self.pass_samples.push((pool, index));
        } else {
            self.sample(pool, index);
        }
    }

    fn reset_queries(&mut self, pool: QueryPoolHandle, first: u32, count: u32) {
        if let Ok(mut pools) = self.inner.query_pools.lock()
            && let Some(p) = pools.get_mut(pool)
        {
            let end = first.saturating_add(count) as usize;
            for writer in p.writers.iter_mut().take(end).skip(first as usize) {
                *writer = None;
            }
        }
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
        let table = self
            .inner
            .table
            .lock()
            .ok()
            .and_then(|t| (!t.tier1).then(|| t.buffer().clone()));
        let (depth_table, depth_resident) = self
            .inner
            .depth_table
            .lock()
            .ok()
            .and_then(|t| t.as_ref().map(|t| (t.buffer().clone(), t.resident())))
            .unzip();
        let mut resident = resident;
        resident.extend(depth_resident.into_iter().flatten());
        let enc = autoreleasepool(|| cmd.new_render_command_encoder(rpd).to_owned());
        if let Some(table) = &table {
            enc.set_vertex_buffer(TEXTURE_BUFFER_INDEX, Some(table), 0);
            enc.set_fragment_buffer(TEXTURE_BUFFER_INDEX, Some(table), 0);
        }
        if let Some(depth) = &depth_table {
            enc.set_vertex_buffer(DEPTH_TEXTURE_BUFFER_INDEX, Some(depth), 0);
            enc.set_fragment_buffer(DEPTH_TEXTURE_BUFFER_INDEX, Some(depth), 0);
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
        self.binding_valid = true;
        let Some(enc) = self.encoder("bind") else {
            return;
        };
        if let Ok(table) = self.inner.table.lock()
            && table.tier1
            && self.pipeline.as_ref().is_some_and(|p| p.texture_table)
        {
            match table.snapshot(&self.inner.device, bindings.textures) {
                Ok((buffer, ids, resident)) => {
                    enc.set_vertex_buffer(TEXTURE_BUFFER_INDEX, Some(&buffer), 0);
                    enc.set_fragment_buffer(TEXTURE_BUFFER_INDEX, Some(&buffer), 0);
                    let bytes = size_of_val(&ids) as u64;
                    let ptr = ids.as_ptr().cast();
                    enc.set_vertex_bytes(TEXTURE_IDS_BUFFER_INDEX, bytes, ptr);
                    enc.set_fragment_bytes(TEXTURE_IDS_BUFFER_INDEX, bytes, ptr);
                    for texture in &resident {
                        enc.use_resource_at(
                            texture,
                            MTLResourceUsage::Read,
                            metal::MTLRenderStages::Vertex | metal::MTLRenderStages::Fragment,
                        );
                    }
                }
                Err(error) => {
                    log::error!("[zengpu-metal] {error}; draw skipped");
                    self.binding_valid = false;
                    return;
                }
            }
        }
        if !self.bind_buffers(enc, bindings.buffers) {
            self.binding_valid = false;
            return;
        }
        let mut pc = [0u8; PUSH_BYTES];
        let mut len = 0usize;
        let words = bindings
            .scalars
            .iter()
            .map(|s| match s {
                Scalar::U32(v) => v.to_le_bytes(),
                Scalar::I32(v) => v.to_le_bytes(),
                Scalar::F32(v) => v.to_le_bytes(),
            })
            .chain(bindings.textures.iter().map(|i| i.to_le_bytes()));
        for w in words {
            if let Some(dst) = pc.get_mut(len..len + 4) {
                dst.copy_from_slice(&w);
                len += 4;
            }
        }
        if len == 0 {
            return;
        }
        let len = len.next_multiple_of(16).min(PUSH_BYTES);
        let ptr = pc.as_ptr() as *const c_void;
        enc.set_vertex_bytes(PUSH_BUFFER_INDEX, len as u64, ptr);
        enc.set_fragment_bytes(PUSH_BUFFER_INDEX, len as u64, ptr);
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64) {
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
        enc.set_vertex_buffer(u64::from(slot), Some(&buf), offset);
    }

    fn set_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat) {
        self.index = self
            .inner
            .buffers
            .lock()
            .ok()
            .and_then(|b| b.get(buffer).map(|b| (b.buf.clone(), offset, format)));
        if self.index.is_none() {
            log::error!("[zengpu-metal] set_index_buffer: stale buffer");
        }
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if !self.binding_valid {
            return;
        }
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

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        if !self.binding_valid {
            return;
        }
        let (Some(p), Some((index, start, format))) = (self.pipeline.clone(), self.index.clone())
        else {
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
        let offset = start + u64::from(indices.start) * format.bytes();
        let instance_count = u64::from(instances.end - instances.start);
        let base_instance = u64::from(instances.start);
        let base_vertex = i64::from(base_vertex);
        let index_type = index_type(format);
        let enc_ref: &metal::RenderCommandEncoderRef = enc;
        unsafe {
            let _: () = msg_send![enc_ref,
                drawIndexedPrimitives: p.primitive
                indexCount: index_count
                indexType: index_type
                indexBuffer: index.as_ref() as &metal::BufferRef
                indexBufferOffset: offset
                instanceCount: instance_count
                baseVertex: base_vertex
                baseInstance: base_instance
            ];
        }
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        if !self.binding_valid {
            return;
        }
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
        if !self.binding_valid {
            return;
        }
        let (Some(p), Some((index, start, format))) = (self.pipeline.clone(), self.index.clone())
        else {
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
                index_type(format),
                &index,
                start,
                &buf,
                offset + i * stride,
            );
        }
    }

    fn draw_indirect_count(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        count_buffer: BufferHandle,
        count_offset: u64,
        max_draws: u32,
        stride: u32,
    ) {
        let stride = if stride == 0 {
            DRAW_INDIRECT_STRIDE
        } else {
            u64::from(stride)
        };
        let clamp = Clamp {
            offset,
            stride,
            words: DRAW_WORDS,
            count_offset,
            max_draws,
        };
        let Some(scratch) = self.clamp("draw_indirect_count", buffer, count_buffer, clamp) else {
            return;
        };
        let (Some(p), Some(enc)) = (self.pipeline.clone(), self.encoder.as_ref()) else {
            return;
        };
        for i in 0..u64::from(max_draws) {
            enc.draw_primitives_indirect(p.primitive, &scratch, i * DRAW_INDIRECT_STRIDE);
        }
    }

    fn draw_indexed_indirect_count(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        count_buffer: BufferHandle,
        count_offset: u64,
        max_draws: u32,
        stride: u32,
    ) {
        let stride = if stride == 0 {
            DRAW_INDEXED_INDIRECT_STRIDE
        } else {
            u64::from(stride)
        };
        let clamp = Clamp {
            offset,
            stride,
            words: INDEXED_WORDS,
            count_offset,
            max_draws,
        };
        let Some((index, start, format)) = self.index.clone() else {
            log::error!(
                "[zengpu-metal] draw_indexed_indirect_count without an index buffer; skipped"
            );
            return;
        };
        let Some(scratch) = self.clamp("draw_indexed_indirect_count", buffer, count_buffer, clamp)
        else {
            return;
        };
        let (Some(p), Some(enc)) = (self.pipeline.clone(), self.encoder.as_ref()) else {
            return;
        };
        for i in 0..u64::from(max_draws) {
            enc.draw_indexed_primitives_indirect(
                p.primitive,
                index_type(format),
                &index,
                start,
                &scratch,
                i * DRAW_INDEXED_INDIRECT_STRIDE,
            );
        }
    }

    fn end_render_pass(&mut self) {
        self.end_encoder();
    }
}

fn index_type(format: IndexFormat) -> MTLIndexType {
    match format {
        IndexFormat::U16 => MTLIndexType::UInt16,
        IndexFormat::U32 => MTLIndexType::UInt32,
    }
}
