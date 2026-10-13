//! The Metal device: resources, transfers, compute, and the [`GpuDevice`]
//! contract.

use std::any::Any;
use std::ffi::c_void;
use std::ptr::copy_nonoverlapping;
use std::str::from_utf8;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputePipelineState, DepthStencilState,
    Device, Library, MTLBlitOption, MTLCommandBufferStatus, MTLOrigin, MTLPrimitiveType, MTLRegion,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    RenderPipelineState, SamplerDescriptor, SamplerState, Texture, TextureDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use zengpu_hal::{
    Bindings, BufferDesc, BufferHandle, CompletedSubmission, ComputePipelineDesc, DeviceLimits,
    Features, Format, GpuDevice, GpuError, HalCapabilities, PipelineHandle, Result, SamplerDesc,
    SamplerHandle, Scalar, ShaderDesc, ShaderHandle, ShaderSource, SlotMap, Submission,
    TargetHandle, TexDim, TextureDesc, TextureHandle, TextureRegion, TextureUsage, TextureWrite,
    TransferCaps, marker,
};

use super::convert;
use super::table::{TABLE_CAPACITY, TextureTable};

pub(crate) struct MetalBuffer {
    pub(crate) buf: Buffer,
    pub(crate) size: u64,
}

pub(crate) struct MetalTexture {
    pub(crate) tex: Texture,
    pub(crate) desc: TextureDesc,
}

/// Something a render pass can draw into: a texture made with
/// `RENDER_TARGET` or `DEPTH_STENCIL`, or a swapchain drawable's texture.
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
    /// The last command buffer committed, which every earlier one finishes
    /// before: waiting on it waits for the device.
    pub(crate) last_submit: Mutex<Option<CommandBuffer>>,
    pub(crate) lost: AtomicBool,
}

// SAFETY: Metal devices, queues and resources are documented thread-safe;
// every Rust-side table is behind a Mutex.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// Lock, treating a poisoned table as a lost device.
pub(crate) fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock().map_err(|_| GpuError::DeviceLost)
}

impl Inner {
    /// A new command buffer, retained past the autorelease pool it came from.
    pub(crate) fn command_buffer(&self) -> CommandBuffer {
        autoreleasepool(|| self.queue.new_command_buffer().to_owned())
    }

    /// Commit `cmd` and remember it as the newest submission.
    pub(crate) fn commit(&self, cmd: &CommandBufferRef) {
        cmd.commit();
        if let Ok(mut last) = self.last_submit.lock() {
            *last = Some(cmd.to_owned());
        }
    }

    /// Commit `cmd`, wait for it, and report whether the GPU ran it cleanly.
    pub(crate) fn commit_and_wait(&self, cmd: &CommandBufferRef) -> Result<()> {
        self.commit(cmd);
        cmd.wait_until_completed();
        self.check(cmd)
    }

    /// A finished command buffer's outcome: a GPU fault is an error, and
    /// some faults mean the device is gone.
    pub(crate) fn check(&self, cmd: &CommandBufferRef) -> Result<()> {
        if cmd.status() != MTLCommandBufferStatus::Error {
            return Ok(());
        }
        // MTLCommandBufferErrorDeviceRemoved = 11, also treat a page fault
        // or a hang the same way the driver does: the work is gone.
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
            // Depth and multisampled textures live on the GPU only on macOS.
            MTLStorageMode::Private
        } else if self.device.has_unified_memory() {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        }
    }
}

/// A depth buffer made by [`MetalDevice::create_depth_target`]; its texture
/// goes with it.
pub struct DepthTarget {
    inner: Arc<Inner>,
    texture: TextureHandle,
    extent: (u32, u32),
}

impl DepthTarget {
    /// Width and height in pixels.
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

/// An opened Metal device: graphics and compute.
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
            }),
        })
    }

    /// The underlying `MTLDevice`, for Metal-specific work.
    pub fn raw_device(&self) -> &metal::DeviceRef {
        &self.inner.device
    }

    /// The device's command queue.
    pub fn raw_queue(&self) -> &metal::CommandQueueRef {
        &self.inner.queue
    }

    /// Wait until all work submitted to this device has completed.
    pub fn wait_idle(&self) -> Result<()> {
        self.inner.wait_idle()
    }

    /// Commit a recorded [`MetalCommandList`](crate::MetalCommandList) and
    /// block until the GPU has finished it.
    pub fn submit_and_wait(&self, list: super::command::MetalCommandList) -> Result<()> {
        let cmd = list.finish();
        self.inner.commit_and_wait(&cmd)
    }

    /// Register `texture` (sampled with `sampler`) in the bindless texture
    /// table at `texture`'s own slot index, for shaders that sample the
    /// table. Returns the slot, or `None` for a stale handle or a slot past
    /// the table's capacity.
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

    /// Slots the bindless texture table has.
    pub fn bindless_texture_capacity(&self) -> u32 {
        TABLE_CAPACITY
    }

    /// Slots in the bindless texture table that hold a texture.
    pub fn bindless_texture_used(&self) -> u32 {
        self.inner.table.lock().map(|t| t.used()).unwrap_or(0)
    }

    /// Register `texture` as a render target, for
    /// [`zengpu_hal::ColorAttachment::target`] (a texture made with
    /// [`TextureUsage::RENDER_TARGET`]) or
    /// [`zengpu_hal::DepthAttachment::target`] (one made with
    /// [`TextureUsage::DEPTH_STENCIL`]). `None` for a stale handle.
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

    /// A `Depth32Float` depth buffer of `width`×`height`, for
    /// [`register_depth_target`](Self::register_depth_target). Named alike on
    /// every graphics backend.
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

    /// Register `depth` as a render target for
    /// [`zengpu_hal::DepthAttachment::target`]. Make a new registration after
    /// recreating the depth buffer, and drop the old one with
    /// [`unregister_render_target`](Self::unregister_render_target).
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
            // A depth target owns its texture, so it cannot be stale.
            _ => TargetHandle::from_parts(u32::MAX, 0),
        }
    }

    /// Drop a render-target registration.
    pub fn unregister_render_target(&self, handle: TargetHandle) {
        if let Ok(mut targets) = self.inner.targets.lock() {
            targets.remove(handle);
        }
    }

    fn buffer(&self, handle: BufferHandle) -> Result<Buffer> {
        lock(&self.inner.buffers)?
            .get(handle)
            .map(|b| b.buf.clone())
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))
    }

    fn texture(&self, handle: TextureHandle) -> Result<(Texture, TextureDesc)> {
        lock(&self.inner.textures)?
            .get(handle)
            .map(|t| (t.tex.clone(), t.desc))
            .ok_or_else(|| GpuError::Backend("metal: invalid texture handle".into()))
    }

    /// Bounds-check `offset..offset+len` against a buffer of `size` bytes.
    fn in_bounds(offset: u64, len: u64, size: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= size => Ok(()),
            _ => Err(GpuError::Backend(
                "metal: buffer access out of bounds".into(),
            )),
        }
    }

    /// Copy `data` into `region` of a host-visible texture.
    fn replace(
        tex: &Texture,
        desc: &TextureDesc,
        region: TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
    ) -> Result<()> {
        let bpp = desc.format.bytes_per_texel();
        let row = if bytes_per_row == 0 {
            region.width * bpp
        } else {
            bytes_per_row
        };
        let need = u64::from(row) * u64::from(region.height);
        if (data.len() as u64) < need || row < region.width * bpp {
            return Err(GpuError::Backend(
                "metal: texture data shorter than its region".into(),
            ));
        }
        if region.x + region.width > tex.width() as u32 >> region.mip_level
            || region.y + region.height > (tex.height() as u32 >> region.mip_level).max(1)
        {
            return Err(GpuError::Backend(
                "metal: texture region out of bounds".into(),
            ));
        }
        let mtl_region = MTLRegion::new_2d(
            u64::from(region.x),
            u64::from(region.y),
            u64::from(region.width),
            u64::from(region.height),
        );
        tex.replace_region_in_slice(
            mtl_region,
            u64::from(region.mip_level),
            u64::from(region.layer),
            data.as_ptr() as *const c_void,
            u64::from(row),
            need,
        );
        Ok(())
    }

    /// Copy `data` into a Private texture through a staging buffer.
    fn stage_into(
        &self,
        tex: &Texture,
        desc: &TextureDesc,
        region: TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
    ) -> Result<()> {
        let row = if bytes_per_row == 0 {
            region.width * desc.format.bytes_per_texel()
        } else {
            bytes_per_row
        };
        let staging = self.inner.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer_to_texture(
                &staging,
                0,
                u64::from(row),
                u64::from(row) * u64::from(region.height),
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
            blit.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)
    }

    fn write_region(
        &self,
        handle: TextureHandle,
        region: TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
    ) -> Result<()> {
        let (tex, desc) = self.texture(handle)?;
        if tex.storage_mode() == MTLStorageMode::Private {
            self.stage_into(&tex, &desc, region, data, bytes_per_row)
        } else {
            Self::replace(&tex, &desc, region, data, bytes_per_row)
        }
    }

    /// Encode + submit one compute dispatch, blocking until completion. Buffers
    /// in `bindings.buffers` (slot indices) bind to `[[buffer(0..n)]]`; scalars
    /// pack into a `Push` struct at `[[buffer(n)]]`; `grid` is the threadgroup
    /// count and the pipeline's `block` is threads-per-threadgroup.
    fn dispatch_one(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let (state, block) = match lock(&self.inner.pipelines)?.get(pipeline) {
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
        let buffers: Vec<Buffer> = {
            let table = lock(&self.inner.buffers)?;
            bindings
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
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            for (i, b) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            if !bindings.scalars.is_empty() {
                let packed = pack_scalars(bindings.scalars);
                enc.set_bytes(
                    buffers.len() as u64,
                    packed.len() as u64,
                    packed.as_ptr() as *const c_void,
                );
            }
            enc.dispatch_thread_groups(
                MTLSize::new(u64::from(grid[0]), u64::from(grid[1]), u64::from(grid[2])),
                MTLSize::new(
                    u64::from(block[0]),
                    u64::from(block[1]),
                    u64::from(block[2]),
                ),
            );
            enc.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)
    }
}

/// Pack inline scalars into a tightly-laid-out `Push` struct (4 bytes each,
/// matching the MSL `struct Push` field order the ZSL→MSL backend emits).
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
            async_transfers: false,
        },
    }
}

impl GpuDevice for MetalDevice {
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

    fn limits(&self) -> DeviceLimits {
        let max = self.inner.device.max_threads_per_threadgroup();
        let cap = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        DeviceLimits {
            max_workgroup_size: [cap(max.width), cap(max.height), cap(max.depth)],
            max_workgroup_invocations: 1024,
            max_storage_buffer_range: self.inner.device.max_buffer_length(),
            max_push_constant_size: 4096,
            max_sampled_textures: TABLE_CAPACITY,
            ..DeviceLimits::default()
        }
    }

    fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        // Metal rejects zero-length buffers; round up to 1 byte.
        let buf = self
            .inner
            .device
            .new_buffer(desc.size.max(1), MTLResourceOptions::StorageModeShared);
        Ok(lock(&self.inner.buffers)?.insert(MetalBuffer {
            buf,
            size: desc.size,
        }))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffers = lock(&self.inner.buffers)?;
        let b = buffers
            .get(buffer)
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
        Self::in_bounds(offset, data.len() as u64, b.size)?;
        // SAFETY: Shared-storage contents() is a valid host pointer for `size`
        // bytes; the bounds check above keeps the copy in range.
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
        let buffers = lock(&self.inner.buffers)?;
        let b = buffers
            .get(buffer)
            .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
        Self::in_bounds(offset, dst.len() as u64, b.size)?;
        // SAFETY: as above; the bounds check keeps the copy within `size`.
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
        {
            let buffers = lock(&self.inner.buffers)?;
            let size = |h| buffers.get(h).map(|b: &MetalBuffer| b.size);
            let (Some(s), Some(d)) = (size(src), size(dst)) else {
                return Err(GpuError::Backend("metal: invalid buffer handle".into()));
            };
            Self::in_bounds(src_offset, len, s)?;
            Self::in_bounds(dst_offset, len, d)?;
        }
        let (s, d) = (self.buffer(src)?, self.buffer(dst)?);
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&s, src_offset, &d, dst_offset, len);
            blit.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)
    }

    fn copy_texture_to_buffer(&self, texture: TextureHandle, buffer: BufferHandle) -> Result<()> {
        let (_, desc) = self.texture(texture)?;
        let row = desc.width * desc.format.bytes_per_texel();
        self.copy_texture_region_to_buffer(
            texture,
            TextureRegion::rect(0, 0, desc.width, desc.height),
            buffer,
            0,
            row,
        )
        .map(|_| ())
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
        let layer_bytes = (desc.width * desc.height * desc.format.bytes_per_texel()) as usize;
        let layers = if desc.dimension == TexDim::Cube {
            6
        } else {
            desc.array_layers.max(1)
        };
        for layer in 0..layers {
            let start = layer as usize * layer_bytes;
            let Some(chunk) = data.get(start..start + layer_bytes) else {
                if layer == 0 {
                    return Err(GpuError::Backend(
                        "metal: texture data shorter than the texture".into(),
                    ));
                }
                break;
            };
            let region = TextureRegion {
                layer,
                ..TextureRegion::rect(0, 0, desc.width, desc.height)
            };
            self.write_region(texture, region, chunk, 0)?;
        }
        Ok(())
    }

    fn upload_texture_data_region(
        &self,
        texture: TextureHandle,
        mip_level: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        let (_, desc) = self.texture(texture)?;
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
        for w in writes {
            self.write_region(w.texture, w.region, w.data, w.bytes_per_row)?;
        }
        Ok(Box::new(CompletedSubmission::new(0)))
    }

    fn copy_buffer_to_texture(
        &self,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
        texture: TextureHandle,
        region: TextureRegion,
    ) -> Result<Submission> {
        let src = self.buffer(buffer)?;
        let (tex, _) = self.texture(texture)?;
        let cmd = self.inner.command_buffer();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer_to_texture(
                &src,
                offset,
                u64::from(bytes_per_row),
                u64::from(bytes_per_row) * u64::from(region.height),
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
        self.inner.commit_and_wait(&cmd)?;
        Ok(Box::new(CompletedSubmission::new(0)))
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
        {
            let buffers = lock(&self.inner.buffers)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            let row = u64::from(bytes_per_row.max(region.width * desc.format.bytes_per_texel()));
            Self::in_bounds(offset, row * u64::from(region.height), b.size)?;
        }
        let dst = self.buffer(buffer)?;
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
                u64::from(bytes_per_row),
                u64::from(bytes_per_row) * u64::from(region.height),
                MTLBlitOption::empty(),
            );
            blit.end_encoding();
        });
        self.inner.commit_and_wait(&cmd)?;
        Ok(Box::new(CompletedSubmission::new(0)))
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
        // The table holds samplers in an argument buffer.
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

    // ── Compute ─────────────────────────────────────────────────────────────

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
        self.dispatch_one(pipeline, bindings, grid)
    }
}
