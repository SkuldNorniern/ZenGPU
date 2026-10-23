//! `GpuDevice` for Metal: resources, transfers, compute.

use std::any::Any;
use std::ffi::c_void;
use std::ptr::copy_nonoverlapping;
use std::str::from_utf8;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use block::ConcreteBlock;
use metal::{
    BlitCommandEncoderRef, Buffer, CommandBuffer, CommandBufferRef, CommandQueue,
    ComputeCommandEncoderRef, ComputePipelineState, DepthStencilState, Device, Library,
    MTLBlitOption, MTLCommandBufferStatus, MTLGPUFamily, MTLOrigin, MTLPrimitiveType,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    RenderPipelineState, SamplerDescriptor, SamplerState, Texture, TextureDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::{BOOL, NO, Object};
use objc::{msg_send, sel, sel_impl};
use zengpu_hal::{
    BackendPreference, Bindings, BufferCopyOp, BufferDesc, BufferHandle, BufferUsage, ComputeCaps,
    ComputeOp, ComputePipelineDesc, DType, DeviceLimits, DispatchOp, Features, Format, FormatCaps,
    GpuDevice, GpuError, GpuSubmission, HalCapabilities, MemoryBudget, MmaShape, PipelineHandle,
    Result, SamplerDesc, SamplerHandle, Scalar, ShaderDesc, ShaderHandle, ShaderSource, SlotMap,
    Submission, SubmissionStatus, TargetHandle, TexDim, TextureDesc, TextureHandle, TextureRegion,
    TextureUsage, TextureWrite, TransferCaps, UsageError, marker,
};

use super::convert;
use super::table::{TABLE_CAPACITY, TextureTable};
use crate::transfer;

pub(crate) struct MetalBuffer {
    pub(crate) buf: Buffer,
    pub(crate) size: u64,
    pub(crate) usage: BufferUsage,
}

pub(crate) struct MetalTexture {
    pub(crate) tex: Texture,
    pub(crate) desc: TextureDesc,
}

/// Draw target: a `RENDER_TARGET` or `DEPTH_STENCIL` texture, or a drawable's.
pub(crate) struct MetalTarget {
    pub(crate) tex: Texture,
    pub(crate) format: Format,
}

pub(crate) struct RenderPipeline {
    pub(crate) state: RenderPipelineState,
    pub(crate) depth: Option<DepthStencilState>,
    pub(crate) primitive: MTLPrimitiveType,
    pub(crate) cull: metal::MTLCullMode,
    pub(crate) winding: metal::MTLWinding,
    pub(crate) fill: metal::MTLTriangleFillMode,
}

pub(crate) enum MetalPipeline {
    Compute {
        state: ComputePipelineState,
        /// Threadgroup size (`@workgroup_size`), used as threads-per-threadgroup.
        block: [u32; 3],
    },
    Render(Arc<RenderPipeline>),
}

/// Shared by the device, its command lists and its surfaces.
pub(crate) struct Inner {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    pub(crate) buffers: Mutex<SlotMap<marker::Buffer, MetalBuffer>>,
    pub(crate) shaders: Mutex<SlotMap<marker::Shader, Library>>,
    pub(crate) pipelines: Mutex<SlotMap<marker::Pipeline, MetalPipeline>>,
    pub(crate) textures: Mutex<SlotMap<marker::Texture, MetalTexture>>,
    pub(crate) samplers: Mutex<SlotMap<marker::Sampler, SamplerState>>,
    pub(crate) targets: Mutex<SlotMap<marker::RenderTarget, MetalTarget>>,
    pub(crate) table: Mutex<TextureTable>,
    /// Last committed buffer. Earlier ones finish first, so waiting on it waits for the device.
    pub(crate) last_submit: Mutex<Option<CommandBuffer>>,
    pub(crate) lost: AtomicBool,
    /// Signalled as each command buffer completes.
    pub(crate) completions: Arc<Completions>,
    /// Async submissions and their buffer slots. The host must not touch those buffers until done.
    pub(crate) in_flight: Mutex<Vec<(CommandBuffer, Vec<u32>)>>,
}

/// Completed-buffer count. Waiters block on the condvar instead of polling.
#[derive(Default)]
pub(crate) struct Completions {
    count: Mutex<u64>,
    rung: Condvar,
}

impl Completions {
    fn ring(&self) {
        if let Ok(mut count) = self.count.lock() {
            *count += 1;
        }
        self.rung.notify_all();
    }
}

// SAFETY: Metal objects are thread-safe. Rust-side tables sit behind a Mutex.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// Locks the table. A poisoned lock is treated as a lost device.
pub(crate) fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock().map_err(|_| GpuError::DeviceLost)
}

impl Inner {
    /// A new command buffer, retained past the autorelease pool it came from.
    pub(crate) fn command_buffer(&self) -> CommandBuffer {
        autoreleasepool(|| self.queue.new_command_buffer().to_owned())
    }

    pub(crate) fn commit(&self, cmd: &CommandBufferRef) {
        let completions = Arc::clone(&self.completions);
        let handler = ConcreteBlock::new(move |_: &CommandBufferRef| completions.ring()).copy();
        cmd.add_completed_handler(&handler);
        if let Ok(mut last) = self.last_submit.lock() {
            cmd.commit();
            *last = Some(cmd.to_owned());
        } else {
            cmd.commit();
        }
    }

    pub(crate) fn commit_and_wait(&self, cmd: &CommandBufferRef) -> Result<()> {
        self.commit(cmd);
        self.wait_for(cmd)
    }

    pub(crate) fn wait_for(&self, cmd: &CommandBufferRef) -> Result<()> {
        cmd.wait_until_completed();
        self.check(cmd)
    }

    /// GPU faults are errors. Some mean the device is lost.
    pub(crate) fn check(&self, cmd: &CommandBufferRef) -> Result<()> {
        if cmd.status() != MTLCommandBufferStatus::Error {
            return Ok(());
        }
        let code: i64 = unsafe {
            let err: *mut Object = msg_send![cmd, error];
            if err.is_null() {
                0
            } else {
                msg_send![err, code]
            }
        };
        if code == 11 {
            self.lost.store(true, Ordering::Release);
            return Err(GpuError::DeviceLost);
        }
        Err(GpuError::Backend(format!(
            "metal: command buffer failed (MTLCommandBufferError {code})"
        )))
    }

    pub(crate) fn wait_idle(&self) -> Result<()> {
        let last = lock(&self.last_submit)?.take();
        if let Some(cmd) = last {
            cmd.wait_until_completed();
            self.check(&cmd)?;
        }
        Ok(())
    }

    fn texture_storage(&self, desc: &TextureDesc) -> MTLStorageMode {
        if desc.format.is_depth_stencil() || desc.samples > 1 {
            MTLStorageMode::Private
        } else if self.device.has_unified_memory() {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        }
    }
}

/// A depth buffer made by [`MetalDevice::create_depth_target`]. Its texture goes with it.
pub struct DepthTarget {
    inner: Arc<Inner>,
    texture: TextureHandle,
    extent: (u32, u32),
}

impl DepthTarget {
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }
}

impl Drop for DepthTarget {
    fn drop(&mut self) {
        if let Ok(mut textures) = self.inner.textures.lock() {
            textures.remove(self.texture);
        }
    }
}

/// Holds committed copies still in flight, which keeps their resources alive.
///
/// Dropping one never waits.
pub(crate) struct MetalSubmission {
    inner: Arc<Inner>,
    cycle_id: u64,
    cmds: Vec<CommandBuffer>,
}

impl MetalSubmission {
    pub(crate) fn boxed(inner: &Arc<Inner>, cmds: Vec<CommandBuffer>) -> Submission {
        Self::boxed_cycle(inner, 0, cmds)
    }

    pub(crate) fn boxed_cycle(
        inner: &Arc<Inner>,
        cycle_id: u64,
        cmds: Vec<CommandBuffer>,
    ) -> Submission {
        Box::new(Self {
            inner: Arc::clone(inner),
            cycle_id,
            cmds,
        })
    }
}

impl GpuSubmission for MetalSubmission {
    fn cycle_id(&self) -> u64 {
        self.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        let mut status = SubmissionStatus::Complete;
        for cmd in &self.cmds {
            match cmd.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => self.inner.check(cmd)?,
                _ => status = SubmissionStatus::Pending,
            }
        }
        Ok(status)
    }

    /// A timeout too long to add to the clock waits with no deadline.
    fn wait(&self, timeout: Duration) -> Result<()> {
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            for cmd in &self.cmds {
                self.inner.wait_for(cmd)?;
            }
            return Ok(());
        };
        let completions = &self.inner.completions;
        let mut count = lock(&completions.count)?;
        while self.poll()? == SubmissionStatus::Pending {
            let now = Instant::now();
            if now >= deadline {
                return Err(GpuError::Timeout);
            }
            let seen = *count;
            while *count == seen {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                count = completions
                    .rung
                    .wait_timeout(count, left)
                    .map_err(|_| GpuError::DeviceLost)?
                    .0;
            }
        }
        Ok(())
    }
}

pub struct MetalDevice {
    pub(crate) inner: Arc<Inner>,
}

impl MetalDevice {
    pub(crate) fn new(device: Device) -> Result<Self> {
        let queue = device.new_command_queue();
        let table = TextureTable::new(&device)?;
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                queue,
                buffers: Mutex::new(SlotMap::default()),
                shaders: Mutex::new(SlotMap::default()),
                pipelines: Mutex::new(SlotMap::default()),
                textures: Mutex::new(SlotMap::default()),
                samplers: Mutex::new(SlotMap::default()),
                targets: Mutex::new(SlotMap::default()),
                table: Mutex::new(table),
                last_submit: Mutex::new(None),
                lost: AtomicBool::new(false),
                completions: Arc::default(),
                in_flight: Mutex::new(Vec::new()),
            }),
        })
    }

    pub fn raw_device(&self) -> &metal::DeviceRef {
        &self.inner.device
    }

    pub fn raw_queue(&self) -> &metal::CommandQueueRef {
        &self.inner.queue
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.inner.wait_idle()
    }

    pub fn submit_and_wait(&self, list: super::command::MetalCommandList) -> Result<()> {
        let cmd = list.finish();
        self.inner.commit_and_wait(&cmd)
    }

    /// Puts `texture` at its own slot in the bindless table. Returns `None` for a stale handle or a
    /// slot past capacity.
    pub fn bind_texture(&self, texture: TextureHandle, sampler: SamplerHandle) -> Option<u32> {
        if texture.index() >= TABLE_CAPACITY {
            return None;
        }
        let tex = self.inner.textures.lock().ok()?.get(texture)?.tex.clone();
        let smp = self.inner.samplers.lock().ok()?.get(sampler)?.clone();
        let mut table = self.inner.table.lock().ok()?;
        table
            .bind(texture.index(), &tex, &smp)
            .then_some(texture.index())
    }

    pub fn bindless_texture_capacity(&self) -> u32 {
        TABLE_CAPACITY
    }

    pub fn bindless_texture_used(&self) -> u32 {
        self.inner.table.lock().map(|t| t.used()).unwrap_or(0)
    }

    /// Returns the render target for `ColorAttachment::target` or `DepthAttachment::target`, or
    /// `None` if stale.
    pub fn register_color_target(&self, texture: TextureHandle) -> Option<TargetHandle> {
        let (tex, format) = {
            let textures = self.inner.textures.lock().ok()?;
            let t = textures.get(texture)?;
            (t.tex.clone(), t.desc.format)
        };
        Some(
            self.inner
                .targets
                .lock()
                .ok()?
                .insert(MetalTarget { tex, format }),
        )
    }

    /// `Depth32Float` buffer for `register_depth_target`.
    pub fn create_depth_target(&self, width: u32, height: u32) -> Result<DepthTarget> {
        let texture = self.create_texture(TextureDesc {
            width: width.max(1),
            height: height.max(1),
            depth: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_STENCIL,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        })?;
        Ok(DepthTarget {
            inner: Arc::clone(&self.inner),
            texture,
            extent: (width.max(1), height.max(1)),
        })
    }

    /// Re-register after recreating the depth buffer. Unregister the old one first.
    pub fn register_depth_target(&self, depth: &DepthTarget) -> TargetHandle {
        let tex = self
            .inner
            .textures
            .lock()
            .ok()
            .and_then(|t| t.get(depth.texture).map(|t| t.tex.clone()));
        match (tex, self.inner.targets.lock()) {
            (Some(tex), Ok(mut targets)) => targets.insert(MetalTarget {
                tex,
                format: Format::Depth32Float,
            }),
            _ => TargetHandle::from_parts(u32::MAX, 0),
        }
    }

    pub fn unregister_render_target(&self, handle: TargetHandle) {
        if let Ok(mut targets) = self.inner.targets.lock() {
            targets.remove(handle);
        }
    }

    fn texture(&self, handle: TextureHandle) -> Result<(Texture, TextureDesc)> {
        lock(&self.inner.textures)?
            .get(handle)
            .map(|t| (t.tex.clone(), t.desc))
            .ok_or_else(|| GpuError::Backend("metal: invalid texture handle".into()))
    }

    /// Metal aborts the process on a descriptor it cannot create, so limits are checked first.
    fn check_texture(&self, desc: &TextureDesc) -> Result<()> {
        let unsupported = |what: String| Err(GpuError::Unsupported(format!("metal: {what}")));
        let (w, h) = (desc.width, desc.height);
        let depth = if desc.dimension == TexDim::D3 {
            desc.depth.max(1)
        } else {
            1
        };
        let layers = desc.array_layers.max(1);
        let samples = desc.samples.max(1);
        let max_extent = if desc.dimension == TexDim::D3 {
            MAX_TEXTURE_3D
        } else {
            MAX_TEXTURE_2D
        };
        if w.max(h).max(depth) > max_extent {
            return unsupported(format!(
                "{w}x{h}x{depth} texture exceeds {max_extent} per side"
            ));
        }
        if desc.dimension == TexDim::D2 && layers > MAX_TEXTURE_LAYERS {
            return unsupported(format!("{layers} array layers exceed {MAX_TEXTURE_LAYERS}"));
        }
        if desc.dimension == TexDim::Cube && (w != h || layers != 6) {
            return unsupported(format!(
                "a cube texture needs square faces and 6 layers, got {w}x{h} with {layers}"
            ));
        }
        let max_mips = 32 - w.max(h).max(depth).leading_zeros();
        if desc.mip_levels > max_mips {
            return unsupported(format!(
                "{} mip levels on a {w}x{h}x{depth} texture; at most {max_mips}",
                desc.mip_levels
            ));
        }
        if samples > 1 {
            if !self
                .inner
                .device
                .supports_texture_sample_count(u64::from(samples))
            {
                return unsupported(format!("{samples} samples per texel"));
            }
            if desc.dimension != TexDim::D2 || layers > 1 || desc.mip_levels > 1 {
                return unsupported(
                    "multisampled textures are single-layer 2D with one mip level".into(),
                );
            }
        }
        Ok(())
    }

    /// Fails while async work uses the buffer. Its contents belong to the GPU until done.
    fn ensure_not_busy(&self, index: u32, op: &str) -> Result<()> {
        let mut in_flight = lock(&self.inner.in_flight)?;
        in_flight.retain(|(cmd, _)| {
            !matches!(
                cmd.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        });
        if in_flight.iter().any(|(_, slots)| slots.contains(&index)) {
            return Err(UsageError::BindingMismatch(format!(
                "metal: {op} cannot touch a buffer pending GPU work still uses"
            ))
            .into());
        }
        Ok(())
    }

    fn in_bounds(offset: u64, len: u64, size: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= size => Ok(()),
            _ => Err(GpuError::Backend(
                "metal: buffer access out of bounds".into(),
            )),
        }
    }

    /// Queues a checked upload after earlier GPU reads, for any storage mode.
    fn stage_into(
        &self,
        tex: &Texture,
        layout: &transfer::Layout,
        region: TextureRegion,
        data: &[u8],
    ) -> Result<CommandBuffer> {
        let staging = self.stage_rows(layout, region, data)?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            Self::encode_upload(blit, &staging, tex, layout, region);
            blit.end_encoding();
        });
        self.inner.commit(&cmd);
        Ok(cmd)
    }

    fn stage_rows(
        &self,
        layout: &transfer::Layout,
        region: TextureRegion,
        data: &[u8],
    ) -> Result<Buffer> {
        let row = region.width * layout.texel_bytes;
        let image_bytes = u64::from(row) * u64::from(region.height);
        if image_bytes > self.inner.device.max_buffer_length() {
            return Err(GpuError::Backend(
                "metal: staging upload exceeds buffer limit".into(),
            ));
        }
        let staging = self
            .inner
            .device
            .new_buffer(image_bytes, MTLResourceOptions::StorageModeShared);
        if layout.row_bytes == row {
            // SAFETY: validated source span equals the packed allocation size.
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr(),
                    staging.contents().cast(),
                    image_bytes as usize,
                );
            }
        } else {
            for y in 0..region.height as usize {
                // SAFETY: layout checked per row. Staging has no GPU readers yet.
                unsafe {
                    copy_nonoverlapping(
                        data.as_ptr().add(y * layout.row_bytes as usize),
                        (staging.contents() as *mut u8).add(y * row as usize),
                        row as usize,
                    );
                }
            }
        }
        Ok(staging)
    }

    fn encode_upload(
        blit: &BlitCommandEncoderRef,
        staging: &Buffer,
        tex: &Texture,
        layout: &transfer::Layout,
        region: TextureRegion,
    ) {
        let row = region.width * layout.texel_bytes;
        let image_bytes = u64::from(row) * u64::from(region.height);
        blit.copy_from_buffer_to_texture(
            staging,
            0,
            u64::from(row),
            image_bytes,
            MTLSize::new(u64::from(region.width), u64::from(region.height), 1),
            tex,
            u64::from(region.layer),
            u64::from(region.mip_level),
            MTLOrigin {
                x: u64::from(region.x),
                y: u64::from(region.y),
                z: 0,
            },
            MTLBlitOption::empty(),
        );
    }

    /// Write `data` into `handle`'s region and wait until it has landed.
    fn write_region(
        &self,
        handle: TextureHandle,
        region: TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
    ) -> Result<()> {
        let cmd = self.queue_region(handle, region, data, bytes_per_row)?;
        self.inner.wait_for(&cmd)
    }

    /// Queues a copy for any storage mode. Queue order and tracked hazards keep it after
    /// earlier readers.
    fn queue_region(
        &self,
        handle: TextureHandle,
        region: TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
    ) -> Result<CommandBuffer> {
        let (tex, desc) = self.texture(handle)?;
        let layout = transfer::texture_layout(
            &desc,
            region,
            bytes_per_row,
            TextureUsage::TRANSFER_DST,
            true,
        )?;
        transfer::data_len(&layout, data.len() as u64)?;
        self.stage_into(&tex, &layout, region, data)
    }

    /// Dispatches a compute pipeline.
    ///
    /// Buffers in `bindings.buffers` bind to `[[buffer(0..n)]]`, and scalars pack into `Push` at
    /// `[[buffer(n)]]`. Textures in `bindings.textures` (table slots) bind to `[[texture(0..t)]]`,
    /// samplers to `[[sampler(0..t)]]`, and storage images follow. `grid` is the threadgroup count.
    /// The pipeline's `block` is threads per threadgroup.
    fn resolve_dispatch(&self, op: &DispatchOp<'_>) -> Result<ResolvedDispatch> {
        let (state, block) = match lock(&self.inner.pipelines)?.get(op.pipeline) {
            Some(MetalPipeline::Compute { state, block }) => (state.clone(), *block),
            Some(MetalPipeline::Render(_)) => {
                return Err(GpuError::Dispatch(
                    "metal: dispatch needs a compute pipeline".into(),
                ));
            }
            None => return Err(GpuError::Backend("metal: invalid pipeline handle".into())),
        };
        let threads = u64::from(block[0]) * u64::from(block[1]) * u64::from(block[2]);
        if threads == 0 || threads > state.max_total_threads_per_threadgroup() {
            return Err(GpuError::Dispatch(format!(
                "metal: workgroup of {threads} threads; this pipeline allows {}",
                state.max_total_threads_per_threadgroup()
            )));
        }
        let b = &op.bindings;
        let buffer_slots = b.buffers.len() + usize::from(!b.scalars.is_empty());
        let texture_slots = b.textures.len() + b.images.len();
        if buffer_slots > MAX_BUFFER_SLOTS
            || b.textures.len() > MAX_SAMPLER_SLOTS
            || texture_slots > MAX_TEXTURE_SLOTS
        {
            return Err(GpuError::Dispatch(format!(
                "metal: {} buffers{}, {} sampled textures and {} images exceed the argument \
                 table ({MAX_BUFFER_SLOTS} buffer, {MAX_SAMPLER_SLOTS} sampler and \
                 {MAX_TEXTURE_SLOTS} texture slots)",
                b.buffers.len(),
                if b.scalars.is_empty() {
                    ""
                } else {
                    " plus push data"
                },
                b.textures.len(),
                b.images.len(),
            )));
        }
        if b.scalars.len() * 4 > MAX_PUSH_BYTES {
            return Err(GpuError::Dispatch(format!(
                "metal: {} scalars exceed the {MAX_PUSH_BYTES}-byte push block",
                b.scalars.len()
            )));
        }
        let buffers: Vec<Buffer> = {
            let table = lock(&self.inner.buffers)?;
            op.bindings
                .buffers
                .iter()
                .map(|&slot| {
                    table
                        .get_by_slot_index(slot)
                        .map(|b| b.buf.clone())
                        .ok_or_else(|| {
                            GpuError::Backend("metal: invalid buffer slot in bindings".into())
                        })
                })
                .collect::<Result<_>>()?
        };
        let sampled: Vec<(Texture, SamplerState)> = {
            let table = lock(&self.inner.table)?;
            op.bindings
                .textures
                .iter()
                .map(|&slot| {
                    table.get(slot).ok_or_else(|| {
                        GpuError::Backend("metal: texture slot in bindings is not bound".into())
                    })
                })
                .collect::<Result<_>>()?
        };
        let images: Vec<Texture> = {
            let textures = lock(&self.inner.textures)?;
            op.bindings
                .images
                .iter()
                .map(|&slot| match textures.get_by_slot_index(slot) {
                    Some(t) if t.desc.usage.contains(TextureUsage::STORAGE) => Ok(t.tex.clone()),
                    Some(_) => Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                        resource: "bound image",
                        needed: "STORAGE",
                    })),
                    None => Err(GpuError::Backend(
                        "metal: invalid image slot in bindings".into(),
                    )),
                })
                .collect::<Result<_>>()?
        };
        Ok(ResolvedDispatch {
            state,
            block,
            grid: op.grid,
            buffers,
            sampled,
            images,
            push: pack_scalars(op.bindings.scalars),
        })
    }

    fn resolve_copy(&self, copy: &BufferCopyOp) -> Result<ResolvedCopy> {
        let buffers = lock(&self.inner.buffers)?;
        let (Some(src), Some(dst)) = (buffers.get(copy.src), buffers.get(copy.dst)) else {
            return Err(GpuError::Backend("metal: invalid buffer handle".into()));
        };
        Self::in_bounds(copy.src_offset, copy.len, src.size)?;
        Self::in_bounds(copy.dst_offset, copy.len, dst.size)?;
        Ok(ResolvedCopy {
            src: src.buf.clone(),
            src_offset: copy.src_offset,
            dst: dst.buf.clone(),
            dst_offset: copy.dst_offset,
            len: copy.len,
        })
    }

    /// Records ops into one command buffer and commits it. All ops are checked before
    /// any is encoded.
    ///
    /// Dispatches share a serial encoder, so later ops see earlier writes without a barrier.
    fn commit_ops(&self, ops: &[ComputeOp<'_>]) -> Result<CommandBuffer> {
        let resolved = ops
            .iter()
            .map(|op| match op {
                ComputeOp::Dispatch(d) => self.resolve_dispatch(d).map(Resolved::Dispatch),
                ComputeOp::CopyBuffer(c) => self.resolve_copy(c).map(Resolved::Copy),
            })
            .collect::<Result<Vec<_>>>()?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let mut rest = resolved.as_slice();
            while let Some(first) = rest.first() {
                let run = rest
                    .iter()
                    .position(|r| {
                        matches!(r, Resolved::Dispatch(_)) != matches!(first, Resolved::Dispatch(_))
                    })
                    .unwrap_or(rest.len());
                let (group, tail) = rest.split_at(run);
                rest = tail;
                if matches!(first, Resolved::Dispatch(_)) {
                    let enc = cmd.new_compute_command_encoder();
                    for r in group {
                        if let Resolved::Dispatch(d) = r {
                            d.encode(enc);
                        }
                    }
                    enc.end_encoding();
                } else {
                    let blit = cmd.new_blit_command_encoder();
                    for r in group {
                        if let Resolved::Copy(c) = r {
                            blit.copy_from_buffer(
                                &c.src,
                                c.src_offset,
                                &c.dst,
                                c.dst_offset,
                                c.len,
                            );
                        }
                    }
                    blit.end_encoding();
                }
            }
        });
        self.inner.commit(&cmd);
        Ok(cmd)
    }
}

/// Texture size limits on every Metal GPU family ZenGPU runs on (Apple3+ and Mac2).
const MAX_TEXTURE_2D: u32 = 16384;
const MAX_TEXTURE_3D: u32 = 2048;
const MAX_TEXTURE_LAYERS: u32 = 2048;

/// Compute argument table sizes on every Metal GPU family ZenGPU runs on.
const MAX_BUFFER_SLOTS: usize = 31;
const MAX_SAMPLER_SLOTS: usize = 16;
const MAX_TEXTURE_SLOTS: usize = 128;
/// `setBytes` limit.
const MAX_PUSH_BYTES: usize = 4096;

struct ResolvedDispatch {
    state: ComputePipelineState,
    block: [u32; 3],
    grid: [u32; 3],
    buffers: Vec<Buffer>,
    sampled: Vec<(Texture, SamplerState)>,
    images: Vec<Texture>,
    push: Vec<u8>,
}

impl ResolvedDispatch {
    fn encode(&self, enc: &ComputeCommandEncoderRef) {
        // Metal rejects an empty grid; it runs nothing anyway.
        if self.grid.contains(&0) {
            return;
        }
        enc.set_compute_pipeline_state(&self.state);
        for (i, b) in self.buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, (tex, smp)) in self.sampled.iter().enumerate() {
            enc.set_texture(i as u64, Some(tex));
            enc.set_sampler_state(i as u64, Some(smp));
        }
        for (i, tex) in self.images.iter().enumerate() {
            enc.set_texture((self.sampled.len() + i) as u64, Some(tex));
        }
        if !self.push.is_empty() {
            enc.set_bytes(
                self.buffers.len() as u64,
                self.push.len() as u64,
                self.push.as_ptr() as *const c_void,
            );
        }
        let [gx, gy, gz] = self.grid.map(u64::from);
        let [bx, by, bz] = self.block.map(u64::from);
        enc.dispatch_thread_groups(MTLSize::new(gx, gy, gz), MTLSize::new(bx, by, bz));
    }
}

struct ResolvedCopy {
    src: Buffer,
    src_offset: u64,
    dst: Buffer,
    dst_offset: u64,
    len: u64,
}

enum Resolved {
    Dispatch(ResolvedDispatch),
    Copy(ResolvedCopy),
}

/// Four bytes per scalar, in the field order of ZSL's MSL `struct Push`.
pub(crate) fn pack_scalars(scalars: &[Scalar]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scalars.len() * 4);
    for s in scalars {
        match s {
            Scalar::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Scalar::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Scalar::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
    out
}

pub(crate) fn capabilities() -> HalCapabilities {
    HalCapabilities {
        graphics: true,
        compute: true,
        features: Features::COMPUTE
            | Features::GRAPHICS
            | Features::DESCRIPTOR_INDEXING
            | Features::UNIFIED_MEMORY,
        transfers: TransferCaps {
            texture_region_write: true,
            buffer_texture_copy: true,
            staged_uploads: false,
            async_transfers: true,
        },
    }
}

impl GpuDevice for MetalDevice {
    fn backend(&self) -> BackendPreference {
        BackendPreference::Metal
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        let mut caps = capabilities();
        if !self.inner.device.has_unified_memory() {
            caps.features = Features::COMPUTE | Features::GRAPHICS | Features::DESCRIPTOR_INDEXING;
        }
        caps
    }

    fn format_caps(&self, format: Format) -> FormatCaps {
        let device = &self.inner.device;
        let float32_filtering = device.supports_family(MTLGPUFamily::Apple7)
            || device.supports_family(MTLGPUFamily::Mac2);
        convert::format_caps(format, float32_filtering)
    }

    fn compute_caps(&self) -> ComputeCaps {
        let device = &self.inner.device;
        let mut mma = Vec::new();
        if device.supports_family(MTLGPUFamily::Apple7) {
            mma.push(MmaShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::F16,
                b: DType::F16,
                acc: DType::F32,
                subgroup: 32,
            });
            // ZSL runs i8 operands as half tiles and keeps the i32 sum exact.
            mma.push(MmaShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::I8,
                b: DType::I8,
                acc: DType::I32,
                subgroup: 32,
            });
            mma.push(MmaShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::F32,
                b: DType::F32,
                acc: DType::F32,
                subgroup: 32,
            });
        }
        ComputeCaps {
            subgroup_sizes: (32, 32),
            default_subgroup: 32,
            shared_bytes: u32::try_from(device.max_threadgroup_memory_length()).unwrap_or(u32::MAX),
            max_workgroup: 1024,
            mma,
        }
    }

    fn limits(&self) -> DeviceLimits {
        let max = self.inner.device.max_threads_per_threadgroup();
        let cap = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        DeviceLimits {
            max_workgroup_size: [cap(max.width), cap(max.height), cap(max.depth)],
            max_workgroup_invocations: 1024,
            max_storage_buffer_range: self.inner.device.max_buffer_length(),
            max_push_constant_size: 4096,
            max_sampled_textures: TABLE_CAPACITY,
            max_storage_textures: TABLE_CAPACITY,
            ..DeviceLimits::default()
        }
    }

    fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let device = &self.inner.device;
        if desc.size > device.max_buffer_length() {
            return Err(GpuError::OutOfMemory(desc.memory));
        }
        let buf = device.new_buffer(desc.size.max(1), MTLResourceOptions::StorageModeShared);
        // A failed allocation is nil, and messaging nil returns 0.
        if buf.length() == 0 {
            return Err(GpuError::OutOfMemory(desc.memory));
        }
        Ok(lock(&self.inner.buffers)?.insert(MetalBuffer {
            buf,
            size: desc.size,
            usage: desc.usage,
        }))
    }

    fn buffer_size(&self, buffer: BufferHandle) -> Result<u64> {
        let buffers = lock(&self.inner.buffers)?;
        buffers
            .get(buffer)
            .map(|buffer| buffer.size)
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.ensure_not_busy(buffer.index(), "write_buffer")?;
        let buffers = lock(&self.inner.buffers)?;
        let b = buffers
            .get(buffer)
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
        Self::in_bounds(offset, data.len() as u64, b.size)?;
        // SAFETY: shared-storage `contents()` is valid for `size` bytes. Bounds checked above.
        unsafe {
            let ptr = (b.buf.contents() as *mut u8).add(offset as usize);
            copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let mut out = vec![
            0u8;
            usize::try_from(len)
                .map_err(|_| GpuError::Backend("metal: read too large".into()))?
        ];
        self.read_buffer_into(buffer, offset, &mut out)?;
        Ok(out)
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.ensure_not_busy(buffer.index(), "read_buffer")?;
        let buffers = lock(&self.inner.buffers)?;
        let b = buffers
            .get(buffer)
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
        if !b.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        Self::in_bounds(offset, dst.len() as u64, b.size)?;
        // SAFETY: as above.
        unsafe {
            let ptr = (b.buf.contents() as *const u8).add(offset as usize);
            copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        let op = ComputeOp::CopyBuffer(BufferCopyOp {
            src,
            src_offset,
            dst,
            dst_offset,
            len,
        });
        let cmd = self.commit_ops(&[op])?;
        self.inner.wait_for(&cmd)
    }

    fn copy_texture_to_buffer(&self, texture: TextureHandle, buffer: BufferHandle) -> Result<()> {
        let (_, desc) = self.texture(texture)?;
        let row = desc
            .width
            .checked_mul(desc.format.bytes_per_texel())
            .ok_or_else(|| {
                GpuError::from(UsageError::BindingMismatch(
                    "metal: texture row size overflow".into(),
                ))
            })?;
        self.copy_texture_region_to_buffer(
            texture,
            TextureRegion::rect(0, 0, desc.width, desc.height),
            buffer,
            0,
            row,
        )?
        .wait(Duration::MAX)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        if let Ok(mut buffers) = self.inner.buffers.lock() {
            buffers.remove(buffer);
        }
    }

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        if desc.width == 0 || desc.height == 0 {
            return Err(GpuError::Backend("metal: zero-sized texture".into()));
        }
        self.check_texture(&desc)?;
        let td = TextureDescriptor::new();
        let layers = desc.array_layers.max(1);
        let (ty, depth, array) = match desc.dimension {
            TexDim::D2 if desc.samples > 1 => (MTLTextureType::D2Multisample, 1, 1),
            TexDim::D2 if layers > 1 => (MTLTextureType::D2Array, 1, layers),
            TexDim::D2 => (MTLTextureType::D2, 1, 1),
            TexDim::D3 => (MTLTextureType::D3, desc.depth.max(1), 1),
            TexDim::Cube => (MTLTextureType::Cube, 1, 1),
        };
        td.set_texture_type(ty);
        td.set_pixel_format(convert::pixel_format(desc.format));
        td.set_width(u64::from(desc.width));
        td.set_height(u64::from(desc.height));
        td.set_depth(u64::from(depth));
        td.set_array_length(u64::from(array));
        td.set_mipmap_level_count(u64::from(desc.mip_levels.max(1)));
        td.set_sample_count(u64::from(desc.samples.max(1)));
        let mut usage = MTLTextureUsage::empty();
        if desc.usage.contains(TextureUsage::SAMPLED) {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            usage |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.contains(TextureUsage::RENDER_TARGET)
            || desc.usage.contains(TextureUsage::DEPTH_STENCIL)
        {
            usage |= MTLTextureUsage::RenderTarget;
        }
        if usage.is_empty() {
            usage = MTLTextureUsage::ShaderRead;
        }
        td.set_usage(usage);
        td.set_storage_mode(self.inner.texture_storage(&desc));
        let tex = self.inner.device.new_texture(&td);
        Ok(lock(&self.inner.textures)?.insert(MetalTexture { tex, desc }))
    }

    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        let (_, desc) = self.texture(texture)?;
        let region = TextureRegion::rect(0, 0, desc.width, desc.height);
        let layout = transfer::texture_layout(&desc, region, 0, TextureUsage::TRANSFER_DST, true)?;
        let layer_bytes = usize::try_from(layout.image_bytes).map_err(|_| {
            GpuError::from(UsageError::BindingMismatch(
                "metal: texture data size overflow".into(),
            ))
        })?;
        let layers = if desc.dimension == TexDim::Cube {
            6
        } else {
            desc.array_layers.max(1)
        };
        let total = layer_bytes.checked_mul(layers as usize).ok_or_else(|| {
            GpuError::from(UsageError::BindingMismatch(
                "metal: texture data size overflow".into(),
            ))
        })?;
        if data.len() < total {
            return Err(UsageError::BindingMismatch(
                "metal: texture data shorter than all its layers".into(),
            )
            .into());
        }
        let (tex, _) = self.texture(texture)?;
        let staged = (0..layers)
            .map(|layer| {
                let start = layer as usize * layer_bytes;
                let region = TextureRegion { layer, ..region };
                self.stage_rows(&layout, region, &data[start..start + layer_bytes])
                    .map(|staging| (staging, region))
            })
            .collect::<Result<Vec<_>>>()?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            for (staging, region) in &staged {
                Self::encode_upload(blit, staging, &tex, &layout, *region);
            }
            blit.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)
    }

    fn upload_texture_data_region(
        &self,
        texture: TextureHandle,
        mip_level: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        let (_, desc) = self.texture(texture)?;
        if mip_level >= desc.mip_levels.max(1) || mip_level >= u32::BITS {
            return Err(GpuError::Backend("metal: texture mip out of bounds".into()));
        }
        let (w, h) = (
            (desc.width >> mip_level).max(1),
            (desc.height >> mip_level).max(1),
        );
        let region = TextureRegion {
            mip_level,
            layer,
            ..TextureRegion::rect(0, 0, w, h)
        };
        self.write_region(texture, region, data, 0)
    }

    fn write_textures(&self, writes: &[TextureWrite<'_>]) -> Result<Submission> {
        if writes.is_empty() {
            return Err(UsageError::BindingMismatch(
                "metal: write_textures needs at least one write".into(),
            )
            .into());
        }
        let mut uploads = Vec::with_capacity(writes.len());
        for w in writes {
            let (tex, desc) = self.texture(w.texture)?;
            let layout = transfer::texture_layout(
                &desc,
                w.region,
                w.bytes_per_row,
                TextureUsage::TRANSFER_DST,
                false,
            )?;
            transfer::data_len(&layout, w.data.len() as u64)?;
            uploads.push((tex, layout, w));
        }
        let staging = uploads
            .iter()
            .map(|(_, layout, w)| self.stage_rows(layout, w.region, w.data))
            .collect::<Result<Vec<_>>>()?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            for ((tex, layout, w), buffer) in uploads.iter().zip(&staging) {
                Self::encode_upload(blit, buffer, tex, layout, w.region);
            }
            blit.end_encoding();
        });
        self.inner.commit(&cmd);
        Ok(MetalSubmission::boxed(&self.inner, vec![cmd]))
    }

    fn copy_buffer_to_texture(
        &self,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
        texture: TextureHandle,
        region: TextureRegion,
    ) -> Result<Submission> {
        let (tex, desc) = self.texture(texture)?;
        let layout = transfer::texture_layout(
            &desc,
            region,
            bytes_per_row,
            TextureUsage::TRANSFER_DST,
            false,
        )?;
        let src = {
            let buffers = lock(&self.inner.buffers)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            transfer::buffer_range(&layout, b.size, b.usage, offset, BufferUsage::TRANSFER_SRC)?;
            b.buf.clone()
        };
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer_to_texture(
                &src,
                offset,
                u64::from(layout.row_bytes),
                layout.image_bytes,
                MTLSize::new(u64::from(region.width), u64::from(region.height), 1),
                &tex,
                u64::from(region.layer),
                u64::from(region.mip_level),
                MTLOrigin {
                    x: u64::from(region.x),
                    y: u64::from(region.y),
                    z: 0,
                },
                MTLBlitOption::empty(),
            );
            blit.end_encoding();
        });
        self.inner.commit(&cmd);
        Ok(MetalSubmission::boxed(&self.inner, vec![cmd]))
    }

    fn copy_texture_region_to_buffer(
        &self,
        texture: TextureHandle,
        region: TextureRegion,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
    ) -> Result<Submission> {
        let (tex, desc) = self.texture(texture)?;
        let layout = transfer::texture_layout(
            &desc,
            region,
            bytes_per_row,
            TextureUsage::TRANSFER_SRC,
            false,
        )?;
        let dst = {
            let buffers = lock(&self.inner.buffers)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            transfer::buffer_range(&layout, b.size, b.usage, offset, BufferUsage::TRANSFER_DST)?;
            b.buf.clone()
        };
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_texture_to_buffer(
                &tex,
                u64::from(region.layer),
                u64::from(region.mip_level),
                MTLOrigin {
                    x: u64::from(region.x),
                    y: u64::from(region.y),
                    z: 0,
                },
                MTLSize::new(u64::from(region.width), u64::from(region.height), 1),
                &dst,
                offset,
                u64::from(layout.row_bytes),
                layout.image_bytes,
                MTLBlitOption::empty(),
            );
            blit.end_encoding();
        });
        self.inner.commit(&cmd);
        Ok(MetalSubmission::boxed(&self.inner, vec![cmd]))
    }

    fn generate_mipmaps(&self, texture: TextureHandle) -> Result<()> {
        let (tex, _) = self.texture(texture)?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.generate_mipmaps(&tex);
            blit.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)
    }

    fn destroy_texture(&self, texture: TextureHandle) {
        let removed = self
            .inner
            .textures
            .lock()
            .ok()
            .and_then(|mut t| t.remove(texture));
        if let (Some(t), Ok(mut table)) = (removed, self.inner.table.lock()) {
            table.unbind(texture.index(), &t.tex);
        }
    }

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle> {
        let sd = SamplerDescriptor::new();
        sd.set_min_filter(convert::min_mag_filter(desc.min_filter));
        sd.set_mag_filter(convert::min_mag_filter(desc.mag_filter));
        sd.set_mip_filter(convert::mip_filter(desc.mip_filter, desc.lod_max > 0.0));
        let address = convert::address(desc.address);
        sd.set_address_mode_s(address);
        sd.set_address_mode_t(address);
        sd.set_address_mode_r(address);
        sd.set_max_anisotropy(u64::from(desc.anisotropy.clamp(1, 16)));
        sd.set_lod_min_clamp(desc.lod_min);
        sd.set_lod_max_clamp(desc.lod_max);
        sd.set_border_color(convert::border(desc.border));
        if let Some(c) = desc.compare {
            sd.set_compare_function(convert::compare(c));
        }
        sd.set_support_argument_buffers(true);
        let state = self.inner.device.new_sampler(&sd);
        Ok(lock(&self.inner.samplers)?.insert(state))
    }

    fn destroy_sampler(&self, sampler: SamplerHandle) {
        if let Ok(mut samplers) = self.inner.samplers.lock() {
            samplers.remove(sampler);
        }
    }

    fn supports_anisotropic_filtering(&self) -> bool {
        true
    }

    fn memory_budget(&self) -> Option<MemoryBudget> {
        let device = &self.inner.device;
        // iOS has the working set size from 16 on; before, nothing reports a budget.
        // SAFETY: `respondsToSelector:` is on every NSObject; the device is live.
        let known: BOOL = unsafe {
            msg_send![device.as_ref(), respondsToSelector: sel!(recommendedMaxWorkingSetSize)]
        };
        if known == NO {
            return None;
        }
        let budget = device.recommended_max_working_set_size();
        Some(MemoryBudget {
            device_local: budget,
            budget,
            usage: device.current_allocated_size(),
        })
    }

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let bytes = match desc.source {
            ShaderSource::Msl(bytes) => bytes,
            ShaderSource::Spirv(_)
            | ShaderSource::Ptx(_)
            | ShaderSource::Hip(_)
            | ShaderSource::CudaSrc(_) => {
                return Err(GpuError::Unsupported(
                    "metal: only ShaderSource::Msl is supported (use zsl!().msl)".into(),
                ));
            }
        };
        let source = from_utf8(bytes)
            .map_err(|_| GpuError::ShaderCompile("metal: MSL source is not valid UTF-8".into()))?;
        let library = self
            .inner
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| GpuError::ShaderCompile(format!("metal: MSL compile failed: {e}")))?;
        Ok(lock(&self.inner.shaders)?.insert(library))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        if let Ok(mut shaders) = self.inner.shaders.lock() {
            shaders.remove(shader);
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let function = {
            let shaders = lock(&self.inner.shaders)?;
            let library = shaders
                .get(desc.shader)
                .ok_or_else(|| GpuError::Backend("metal: invalid shader handle".into()))?;
            library.get_function(desc.entry, None).map_err(|e| {
                GpuError::PipelineCreation(format!("metal: function `{}`: {e}", desc.entry))
            })?
        };
        let state = self
            .inner
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| GpuError::PipelineCreation(format!("metal: pipeline: {e}")))?;
        Ok(lock(&self.inner.pipelines)?.insert(MetalPipeline::Compute {
            state,
            block: desc.block,
        }))
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        if let Ok(mut pipelines) = self.inner.pipelines.lock() {
            pipelines.remove(pipeline);
        }
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.dispatch_batch(&[DispatchOp {
            pipeline,
            bindings,
            grid,
        }])
    }

    /// One command buffer for the whole batch, waited on once.
    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        let ops: Vec<ComputeOp<'_>> = ops.iter().copied().map(ComputeOp::Dispatch).collect();
        let cmd = self.commit_ops(&ops)?;
        self.inner.wait_for(&cmd)
    }

    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        let ops: Vec<ComputeOp<'_>> = ops.iter().copied().map(ComputeOp::Dispatch).collect();
        self.submit_compute_ops(cycle_id, &ops)
    }

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        let cmd = self.commit_ops(ops)?;
        let slots = ops
            .iter()
            .flat_map(|op| match op {
                ComputeOp::Dispatch(d) => d.bindings.buffers.to_vec(),
                ComputeOp::CopyBuffer(c) => vec![c.src.index(), c.dst.index()],
            })
            .collect();
        lock(&self.inner.in_flight)?.push((cmd.clone(), slots));
        Ok(MetalSubmission::boxed_cycle(
            &self.inner,
            cycle_id,
            vec![cmd],
        ))
    }
}
