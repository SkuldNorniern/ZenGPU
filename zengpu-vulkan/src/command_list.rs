//! Pooled command-buffer recorder implementing [`zengpu_hal::RenderCommands`].
//!
//! Records straight to `ash`: no intermediate command list, no per-frame allocation after warmup.

use std::{
    ffi::CString,
    mem,
    ops::Range,
    ptr,
    sync::{Arc, Mutex},
};

use ash::vk;
use zengpu_hal::{
    Bindings, BufferHandle, ColorAttachment, DepthAttachment, GpuError, LoadOp, PipelineHandle,
    QueryPoolHandle, RenderCommands, RenderPassDesc, Result, Scalar, SlotMap, TargetHandle,
    ViewportScissor, marker,
};

use crate::device::{
    VulkanBuffer, VulkanDeviceInner, VulkanPipeline, VulkanQueryPool, VulkanRenderTarget,
    VulkanTexture,
};

pub(crate) const COLOR_SUBRESOURCE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

const DEPTH_SUBRESOURCE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::DEPTH,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Pool of `vk::CommandBuffer`s for [`VulkanCommandList`]. One command pool per buffer, since
/// recording is externally synchronized per pool.
pub(crate) struct CmdListPool {
    inner: Arc<VulkanDeviceInner>,
    free: Mutex<Vec<vk::CommandBuffer>>,
    pools: Mutex<Vec<vk::CommandPool>>,
}

impl CmdListPool {
    pub(crate) fn new(inner: Arc<VulkanDeviceInner>) -> Result<Self> {
        Ok(Self {
            inner,
            free: Mutex::new(Vec::new()),
            pools: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn acquire(&self) -> Result<vk::CommandBuffer> {
        let (cmd, reused) = match self.free.lock().unwrap().pop() {
            Some(cmd) => (cmd, true),
            None => {
                let pool = unsafe {
                    self.inner.device.create_command_pool(
                        &vk::CommandPoolCreateInfo {
                            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            queue_family_index: self.inner.queue_family,
                            ..Default::default()
                        },
                        None,
                    )
                }
                .map_err(|e| GpuError::Backend(format!("create command list pool: {e}")))?;
                let bufs = unsafe {
                    self.inner
                        .device
                        .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                            command_pool: pool,
                            level: vk::CommandBufferLevel::PRIMARY,
                            command_buffer_count: 1,
                            ..Default::default()
                        })
                };
                let bufs = match bufs {
                    Ok(bufs) => bufs,
                    Err(e) => {
                        unsafe { self.inner.device.destroy_command_pool(pool, None) };
                        return Err(GpuError::Backend(format!("allocate command buffer: {e}")));
                    }
                };
                self.pools.lock().unwrap().push(pool);
                (bufs[0], false)
            }
        };
        unsafe {
            if reused {
                self.inner
                    .device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| GpuError::Backend(format!("reset command buffer: {e}")))?;
            }
        }
        unsafe {
            self.inner.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| GpuError::Backend(format!("begin command buffer: {e}")))?;
        Ok(cmd)
    }

    /// The caller must wait until the GPU is done with `cmd` (for example, its frame-in-flight
    /// fence is signaled).
    pub(crate) fn release(&self, cmd: vk::CommandBuffer) {
        self.free.lock().unwrap().push(cmd);
    }
}

impl Drop for CmdListPool {
    fn drop(&mut self) {
        unsafe {
            for pool in self.pools.get_mut().unwrap().drain(..) {
                self.inner.device.destroy_command_pool(pool, None);
            }
        }
    }
}

fn color_load_op(load: LoadOp) -> (vk::AttachmentLoadOp, vk::ClearValue) {
    match load {
        LoadOp::Clear(c) => (
            vk::AttachmentLoadOp::CLEAR,
            vk::ClearValue {
                color: vk::ClearColorValue { float32: c },
            },
        ),
        LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        LoadOp::DontCare => (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default()),
    }
}

/// Depth variant. Clear depth is component 0 of [`LoadOp::clear_depth`].
fn depth_load_op(load: LoadOp) -> (vk::AttachmentLoadOp, vk::ClearValue) {
    match load {
        LoadOp::Clear(c) => (
            vk::AttachmentLoadOp::CLEAR,
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: c[0],
                    stencil: 0,
                },
            },
        ),
        LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        LoadOp::DontCare => (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default()),
    }
}

fn store_op(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}

/// Maximum color attachments per render pass, so `begin_render_pass` can use a stack array.
const MAX_COLOR_ATTACHMENTS: usize = 4;
const MAX_VERTEX_BUFFERS: usize = 8;

/// Records draw commands into a pooled `vk::CommandBuffer` via `VK_KHR_dynamic_rendering`.
///
/// Shares the device's slotmaps (`Arc<Mutex<...>>`) to resolve handles.
pub struct VulkanCommandList {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    #[allow(dead_code)]
    pub(crate) pool: Arc<CmdListPool>,
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
    pub(crate) render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    /// Whether each color target's texture holds texels. Don't lock while `render_targets` is held
    /// (`register_color_target` takes them in reverse).
    pub(crate) textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    pub(crate) buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
    pub(crate) query_pools: Arc<Mutex<SlotMap<marker::QueryPool, VulkanQueryPool>>>,
    pub(crate) bindless_set: vk::DescriptorSet,
    current_pipeline: Option<PipelineHandle>,
    /// Layout of the bound graphics pipeline. Scopes push constants for `bind`.
    current_layout: Option<vk::PipelineLayout>,
    current_vertex_buffers: [Option<BufferHandle>; MAX_VERTEX_BUFFERS],
    current_index_buffer: Option<BufferHandle>,
    /// Targets from this pass with `sample_after` set: moved to
    /// `SHADER_READ_ONLY_OPTIMAL` in `end_render_pass`.
    pending_shader_read: [Option<TargetHandle>; MAX_COLOR_ATTACHMENTS],
}

impl VulkanCommandList {
    /// `cmd` must already be recording.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inner: Arc<VulkanDeviceInner>,
        pool: Arc<CmdListPool>,
        cmd: vk::CommandBuffer,
        pipelines: Arc<Mutex<SlotMap<marker::Pipeline, VulkanPipeline>>>,
        render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
        textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
        buffers: Arc<Mutex<SlotMap<marker::Buffer, VulkanBuffer>>>,
        query_pools: Arc<Mutex<SlotMap<marker::QueryPool, VulkanQueryPool>>>,
        bindless_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            inner,
            pool,
            cmd,
            pipelines,
            render_targets,
            textures,
            buffers,
            query_pools,
            bindless_set,
            current_pipeline: None,
            current_layout: None,
            current_vertex_buffers: [None; MAX_VERTEX_BUFFERS],
            current_index_buffer: None,
            pending_shader_read: [None; MAX_COLOR_ATTACHMENTS],
        }
    }

    /// The raw command buffer, for `surface` to end and submit.
    pub(crate) fn raw(&self) -> vk::CommandBuffer {
        self.cmd
    }

    /// Returns the buffer to its pool once the GPU is done with it.
    ///
    /// `surface` does this after a fence wait.
    #[allow(dead_code)]
    pub(crate) fn release(&self) {
        self.pool.release(self.cmd);
    }

    fn transition_color(&self, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
        if old == new {
            return;
        }
        let barrier = vk::ImageMemoryBarrier {
            old_layout: old,
            new_layout: new,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            image,
            subresource_range: COLOR_SUBRESOURCE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..Default::default()
        };
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Transitions to the resting layout (`SHADER_READ_ONLY_OPTIMAL` for sampled)
    /// for `sample_after`.
    ///
    /// Shaders and copies read it next.
    fn transition_to_shader_read(&self, image: vk::Image, resting: vk::ImageLayout) {
        let barrier = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: resting,
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            image,
            subresource_range: COLOR_SUBRESOURCE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..Default::default()
        };
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    fn transition_depth(&self, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
        if old == new {
            return;
        }
        let barrier = vk::ImageMemoryBarrier {
            old_layout: old,
            new_layout: new,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image,
            subresource_range: DEPTH_SUBRESOURCE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..Default::default()
        };
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    /// Per color attachment: uploaded to since the layout was last tracked. Read with one
    /// lock at a time.
    fn uploaded_targets(&self, color: &[ColorAttachment]) -> [bool; MAX_COLOR_ATTACHMENTS] {
        let mut backing = [None; MAX_COLOR_ATTACHMENTS];
        {
            let targets = self.render_targets.lock().unwrap();
            for (slot, att) in color.iter().enumerate() {
                backing[slot] = targets
                    .get(att.target)
                    .filter(|rt| rt.layout == vk::ImageLayout::UNDEFINED)
                    .and_then(|rt| rt.texture);
            }
        }
        let mut uploaded = [false; MAX_COLOR_ATTACHMENTS];
        if backing.iter().any(Option::is_some) {
            let textures = self.textures.lock().unwrap();
            for (slot, texture) in backing.into_iter().enumerate() {
                uploaded[slot] = texture
                    .and_then(|t| textures.get(t))
                    .is_some_and(|tex| tex.is_ready(0, 0));
            }
        }
        uploaded
    }

    /// `None` for a stale handle.
    fn color_attachment_info(
        &self,
        targets: &mut SlotMap<marker::RenderTarget, VulkanRenderTarget>,
        att: &ColorAttachment,
    ) -> Option<(vk::RenderingAttachmentInfo<'static>, vk::Extent2D)> {
        let rt = targets.get_mut(att.target)?;
        self.transition_color(
            rt.image,
            rt.layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        rt.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let (load_op, clear_value) = color_load_op(att.load);
        Some((
            vk::RenderingAttachmentInfo {
                image_view: rt.view,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op,
                store_op: store_op(att.store),
                clear_value,
                ..Default::default()
            },
            rt.extent,
        ))
    }

    /// `None` for a stale handle.
    fn depth_attachment_info(
        &self,
        targets: &mut SlotMap<marker::RenderTarget, VulkanRenderTarget>,
        att: &DepthAttachment,
    ) -> Option<(vk::RenderingAttachmentInfo<'static>, vk::Extent2D)> {
        let rt = targets.get_mut(att.target)?;
        self.transition_depth(
            rt.image,
            rt.layout,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );
        rt.layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
        let (load_op, clear_value) = depth_load_op(att.load);
        Some((
            vk::RenderingAttachmentInfo {
                image_view: rt.view,
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                load_op,
                store_op: store_op(att.store),
                clear_value,
                ..Default::default()
            },
            rt.extent,
        ))
    }
}

impl RenderCommands for VulkanCommandList {
    fn push_debug_group(&mut self, label: &str) {
        let Some(debug_utils) = &self.inner.debug_utils else {
            return;
        };
        let Ok(label) = CString::new(label) else {
            return;
        };
        let info = vk::DebugUtilsLabelEXT {
            p_label_name: label.as_ptr(),
            color: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        unsafe { debug_utils.cmd_begin_debug_utils_label(self.cmd, &info) };
    }

    fn pop_debug_group(&mut self) {
        if let Some(debug_utils) = &self.inner.debug_utils {
            unsafe { debug_utils.cmd_end_debug_utils_label(self.cmd) };
        }
    }

    fn insert_debug_label(&mut self, label: &str) {
        let Some(debug_utils) = &self.inner.debug_utils else {
            return;
        };
        let Ok(label) = CString::new(label) else {
            return;
        };
        let info = vk::DebugUtilsLabelEXT {
            p_label_name: label.as_ptr(),
            color: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        unsafe { debug_utils.cmd_insert_debug_utils_label(self.cmd, &info) };
    }

    fn write_timestamp(&mut self, pool: QueryPoolHandle, index: u32) {
        let query_pools = self.query_pools.lock().unwrap();
        let Some(query_pool) = query_pools.get(pool) else {
            return;
        };
        if index >= query_pool.count {
            return;
        }
        unsafe {
            self.inner.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool.pool,
                index,
            );
        }
    }

    fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        assert!(
            desc.color.len() <= MAX_COLOR_ATTACHMENTS,
            "render pass has {} color attachments, but ZenGPU currently supports at most {}",
            desc.color.len(),
            MAX_COLOR_ATTACHMENTS
        );
        let mut color_attachments = [vk::RenderingAttachmentInfo::default(); MAX_COLOR_ATTACHMENTS];
        let mut pending_shader_read = [None; MAX_COLOR_ATTACHMENTS];
        let mut extent = vk::Extent2D::default();
        let count = desc.color.len();
        let uploaded = self.uploaded_targets(desc.color);
        let mut targets = self.render_targets.lock().unwrap();
        for (i, att) in desc.color.iter().enumerate() {
            if uploaded[i]
                && let Some(rt) = targets.get_mut(att.target)
                && rt.layout == vk::ImageLayout::UNDEFINED
            {
                rt.layout = rt.resting;
            }
            if let Some((info, e)) = self.color_attachment_info(&mut targets, att) {
                color_attachments[i] = info;
                extent = e;
            }
            if att.sample_after {
                pending_shader_read[i] = Some(att.target);
            }
        }
        self.pending_shader_read = pending_shader_read;

        let depth_info = desc
            .depth
            .and_then(|d| self.depth_attachment_info(&mut targets, &d));
        if let Some((_, e)) = &depth_info {
            extent = *e;
        }
        drop(targets);

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            },
            layer_count: 1,
            color_attachment_count: count as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: depth_info.as_ref().map_or(ptr::null(), |(info, _)| info),
            ..Default::default()
        };
        unsafe {
            self.inner
                .dynamic_rendering
                .cmd_begin_rendering(self.cmd, &rendering_info);
        }
    }

    fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        if self.current_pipeline == Some(pipeline) {
            return;
        }
        let pipelines = self.pipelines.lock().unwrap();
        let Some(VulkanPipeline::Graphics {
            layout,
            pipeline: vk_pipeline,
        }) = pipelines.get(pipeline)
        else {
            return;
        };
        let (layout, vk_pipeline) = (*layout, *vk_pipeline);
        drop(pipelines);
        unsafe {
            self.inner.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                vk_pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[self.bindless_set],
                &[],
            );
        }
        self.current_pipeline = Some(pipeline);
        self.current_layout = Some(layout);
    }

    fn set_viewport_scissor(&mut self, vs: ViewportScissor) {
        let viewport = vk::Viewport {
            x: vs.viewport.x,
            y: vs.viewport.y,
            width: vs.viewport.width,
            height: vs.viewport.height,
            min_depth: vs.viewport.min_depth,
            max_depth: vs.viewport.max_depth,
        };
        let scissor = match vs.scissor {
            Some(r) => vk::Rect2D {
                offset: vk::Offset2D {
                    x: r.x as i32,
                    y: r.y as i32,
                },
                extent: vk::Extent2D {
                    width: r.width as u32,
                    height: r.height as u32,
                },
            },
            None => vk::Rect2D {
                offset: vk::Offset2D {
                    x: vs.viewport.x as i32,
                    y: vs.viewport.y as i32,
                },
                extent: vk::Extent2D {
                    width: vs.viewport.width as u32,
                    height: vs.viewport.height as u32,
                },
            },
        };
        unsafe {
            self.inner.device.cmd_set_viewport(self.cmd, 0, &[viewport]);
            self.inner.device.cmd_set_scissor(self.cmd, 0, &[scissor]);
        }
    }

    fn bind(&mut self, bindings: Bindings<'_>) {
        let Some(layout) = self.current_layout else {
            return;
        };

        let mut pc = [0u8; 256];
        let mut len = 0usize;
        let mut push = |bytes: [u8; 4]| {
            if len + 4 <= pc.len() {
                pc[len..len + 4].copy_from_slice(&bytes);
                len += 4;
            }
        };
        for scalar in bindings.scalars {
            push(match scalar {
                Scalar::U32(v) => v.to_ne_bytes(),
                Scalar::I32(v) => v.to_ne_bytes(),
                Scalar::F32(v) => v.to_bits().to_ne_bytes(),
            });
        }
        for &idx in bindings.textures {
            push(idx.to_ne_bytes());
        }
        for &idx in bindings.buffers {
            push(idx.to_ne_bytes());
        }
        if len == 0 {
            return;
        }
        unsafe {
            self.inner.device.cmd_push_constants(
                self.cmd,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &pc[..len],
            );
        }
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle) {
        if self.current_vertex_buffers.get(slot as usize) == Some(&Some(buffer)) {
            return;
        }
        let buffers = self.buffers.lock().unwrap();
        let Some(buf) = buffers.get(buffer) else {
            return;
        };
        let vk_buf = buf.buffer;
        drop(buffers);
        unsafe {
            self.inner
                .device
                .cmd_bind_vertex_buffers(self.cmd, slot, &[vk_buf], &[0]);
        }
        if let Some(current) = self.current_vertex_buffers.get_mut(slot as usize) {
            *current = Some(buffer);
        }
    }

    fn set_index_buffer(&mut self, buffer: BufferHandle) {
        if self.current_index_buffer == Some(buffer) {
            return;
        }
        let buffers = self.buffers.lock().unwrap();
        let Some(buf) = buffers.get(buffer) else {
            return;
        };
        let vk_buf = buf.buffer;
        drop(buffers);
        unsafe {
            self.inner
                .device
                .cmd_bind_index_buffer(self.cmd, vk_buf, 0, vk::IndexType::UINT32);
        }
        self.current_index_buffer = Some(buffer);
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        unsafe {
            self.inner.device.cmd_draw(
                self.cmd,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>) {
        unsafe {
            self.inner.device.cmd_draw_indexed(
                self.cmd,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                0,
                instances.start,
            );
        }
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        let buffers = self.buffers.lock().unwrap();
        let Some(buf) = buffers.get(buffer) else {
            return;
        };
        let vk_buf = buf.buffer;
        drop(buffers);
        let stride = if stride == 0 {
            mem::size_of::<vk::DrawIndirectCommand>() as u32
        } else {
            stride
        };
        unsafe {
            self.inner
                .device
                .cmd_draw_indirect(self.cmd, vk_buf, offset, draw_count, stride);
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let buffers = self.buffers.lock().unwrap();
        let Some(buf) = buffers.get(buffer) else {
            return;
        };
        let vk_buf = buf.buffer;
        drop(buffers);
        let stride = if stride == 0 {
            mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32
        } else {
            stride
        };
        unsafe {
            self.inner
                .device
                .cmd_draw_indexed_indirect(self.cmd, vk_buf, offset, draw_count, stride);
        }
    }

    fn end_render_pass(&mut self) {
        unsafe {
            self.inner.dynamic_rendering.cmd_end_rendering(self.cmd);
        }
        let pending = mem::replace(&mut self.pending_shader_read, [None; MAX_COLOR_ATTACHMENTS]);
        let mut rendered = [None; MAX_COLOR_ATTACHMENTS];
        let mut targets = self.render_targets.lock().unwrap();
        for (slot, target) in pending.into_iter().enumerate() {
            if let Some(rt) = target.and_then(|t| targets.get_mut(t)) {
                let image = rt.image;
                rt.layout = rt.resting;
                rendered[slot] = rt.texture;
                self.transition_to_shader_read(image, rt.resting);
            }
        }
        drop(targets);
        if rendered.iter().any(Option::is_some) {
            let mut textures = self.textures.lock().unwrap();
            for texture in rendered.into_iter().flatten() {
                if let Some(tex) = textures.get_mut(texture) {
                    tex.mark_ready(0, 0);
                }
            }
        }
    }
}
