//! Backend-independent device trait. Object-safe: `Box<dyn GpuDevice>` or `Arc<dyn GpuDevice>`
//! select at runtime.

use core::any::Any;

use crate::caps::ComputeCaps;
use crate::command::{Bindings, ComputeOp, DispatchOp};
use crate::desc::{
    BufferDesc, ComputePipelineDesc, ComputePipelineHints, PipelineStats, SamplerDesc, ShaderDesc,
    TextureDesc, TextureRegion, TextureWrite,
};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::{DeviceLimits, FormatCaps, HalCapabilities, MemoryBudget};
use crate::submission::{CompletedReadback, CompletedSubmission, ReadSubmission, Submission};
use crate::types::{BackendPreference, Features, Format};

/// `Send + Sync`, so worker threads can allocate and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Returns the backend device for backend-specific calls, e.g. a Vulkan swapchain
    /// from a `VulkanDevice`.
    fn as_any(&self) -> &dyn Any;

    fn capabilities(&self) -> HalCapabilities;

    /// `Auto` means the backend did not say, so callers fall back to SPIR-V.
    fn backend(&self) -> BackendPreference {
        BackendPreference::Auto
    }

    fn limits(&self) -> DeviceLimits {
        DeviceLimits::default()
    }

    /// The default claims scalar work only. Backends report what they checked.
    fn compute_caps(&self) -> ComputeCaps {
        ComputeCaps::scalar_only()
    }

    /// What textures of `format` can do here. The default claims nothing.
    fn format_caps(&self, format: Format) -> FormatCaps {
        let _ = format;
        FormatCaps::default()
    }

    /// `None` when the backend cannot report a budget.
    fn memory_budget(&self) -> Option<MemoryBudget> {
        None
    }

    /// The handle is generational. Contents are unspecified, so write every byte before reading it.
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    /// Validates the handle and returns its byte capacity. No transfer or GPU wait happens.
    fn buffer_size(&self, _buffer: BufferHandle) -> Result<u64> {
        Err(GpuError::Unsupported("buffer size query".into()))
    }

    /// Copies `data` into `buffer` and returns once it is visible to later work. Waits only for
    /// pending submissions that use `buffer`, never for unrelated work.
    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Requires `BufferUsage::READBACK`. Waits only for pending writes to `buffer`, never for
    /// unrelated work.
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Starts an async write of `data` into `buffer`. Returns before the copy finishes. `data` is
    /// copied on return, so the caller can reuse it. Earlier submissions using `buffer` finish
    /// first. It is allowed while others are pending, so poll the handle.
    ///
    /// The default calls `write_buffer` and returns a completed handle.
    fn submit_write_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<Submission> {
        self.write_buffer(buffer, offset, data)?;
        Ok(Box::new(CompletedSubmission::new(cycle_id)))
    }

    /// Starts an async read of `buffer`. Requires `READBACK`. The read observes every earlier write
    /// to `buffer`. Take the bytes via `into_bytes` or `read_into`, which wait if needed.
    ///
    /// The default calls `read_buffer` and returns a completed handle.
    fn submit_read_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        let bytes = self.read_buffer(buffer, offset, len)?;
        Ok(Box::new(CompletedReadback::new(cycle_id, bytes)))
    }

    /// Starts an async read into `dst` from `offset`. Requires `READBACK`, and the range is
    /// `offset..offset + dst.len()`.
    ///
    /// Real-time backends copy straight into `dst`. The default goes through `read_buffer`.
    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        let bytes = self.read_buffer(buffer, offset, dst.len() as u64)?;
        dst.copy_from_slice(&bytes);
        Ok(())
    }

    /// Copies a range from `src` to `dst`, synchronously. Bytes are visible to later ops on return.
    /// `src` needs `TRANSFER_SRC` and `dst` needs `TRANSFER_DST`. Same-buffer copies are rejected.
    /// A zero length is a no-op after validation.
    fn copy_buffer(
        &self,
        _src: BufferHandle,
        _src_offset: u64,
        _dst: BufferHandle,
        _dst_offset: u64,
        _len: u64,
    ) -> Result<()> {
        Err(GpuError::Unsupported("buffer copy".into()))
    }

    /// Copies mip 0, layer 0 of a texture into `buffer` at byte 0, synchronously. `buffer` needs
    /// `TRANSFER_DST`, or `READBACK` for host reads.
    ///
    /// The default returns `GpuError::Unsupported`.
    fn copy_texture_to_buffer(&self, _texture: TextureHandle, _buffer: BufferHandle) -> Result<()> {
        Err(GpuError::Unsupported("texture-to-buffer copy".into()))
    }

    /// Backend device ordinal, or `-1` when the backend does not expose one.
    fn device_ordinal(&self) -> i32 {
        -1
    }

    /// Whether direct device-to-device copies from `peer_ordinal` are usable.
    fn can_peer(&self, _peer_ordinal: i32) -> bool {
        false
    }

    /// The default returns an error, since multi-device copies are not supported.
    fn copy_from_peer(
        &self,
        _dst: BufferHandle,
        _src_device: &dyn GpuDevice,
        _src: BufferHandle,
        _bytes: u64,
    ) -> Result<()> {
        Err(GpuError::Unsupported("peer device copy".into()))
    }

    /// A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Uploads tightly packed pixels in the texture's creation format (RGBA8 is 4 bytes per pixel).
    /// Blocks until done.
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Replaces one whole mip of one layer (or cube face). For a sub-rectangle, use
    /// `write_textures` (check `TransferCaps::texture_region_write`). `upload_texture_data` only
    /// targets mip 0, layer 0.
    ///
    /// The default returns `GpuError::UnsupportedFeatures`.
    fn upload_texture_data_region(
        &self,
        _texture: TextureHandle,
        _mip_level: u32,
        _layer: u32,
        _data: &[u8],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::GRAPHICS))
    }

    /// Writes each of `writes` into its region and keeps the rest of the texture. This is one
    /// submission. It returns once `data` is copied out, before the GPU has seen it. Later work
    /// sees the texels. Check `TransferCaps::texture_region_write`, since the default refuses.
    fn write_textures(&self, _writes: &[TextureWrite<'_>]) -> Result<Submission> {
        Err(GpuError::Unsupported("texture region write".into()))
    }

    /// Copies rows of `buffer` (from `offset`, `bytes_per_row` apart) into `region`. `buffer` needs
    /// `TRANSFER_SRC`. Later work sees the copy. Check `TransferCaps::buffer_texture_copy`, since
    /// the default refuses.
    fn copy_buffer_to_texture(
        &self,
        _buffer: BufferHandle,
        _offset: u64,
        _bytes_per_row: u32,
        _texture: TextureHandle,
        _region: TextureRegion,
    ) -> Result<Submission> {
        Err(GpuError::Unsupported("buffer-to-texture copy".into()))
    }

    /// Copies `region` of `texture` into `buffer`, starting `offset` bytes
    /// in, rows `bytes_per_row` apart, on the device. `buffer` needs
    /// [`crate::BufferUsage::TRANSFER_DST`], and
    /// [`crate::BufferUsage::READBACK`] to be read on the host once the
    /// submission completes.
    ///
    /// Reading back without stalling a frame:
    ///
    /// 1. Create the buffer once, with `READBACK | TRANSFER_DST`, and reuse it.
    /// 2. Call this and keep the returned [`Submission`]. Do not wait on it.
    /// 3. Each frame, `poll` it. Polling never blocks.
    /// 4. When it reports `Complete`, `read_buffer_into` a caller-owned slice.
    ///    Reading before completion fails rather than waiting.
    ///
    /// Drop the submission only after it completes: on Vulkan, dropping a
    /// pending one waits for the GPU. Metal does not wait on drop; it keeps the
    /// command buffer alive until the GPU finishes.
    ///
    /// ```rust,no_run
    /// # use zengpu_hal::{BufferDesc, BufferHandle, BufferUsage, GpuDevice, MemoryUsage};
    /// # use zengpu_hal::{Result, Submission, SubmissionStatus, TextureHandle, TextureRegion};
    /// # fn example(dev: &dyn GpuDevice, tex: TextureHandle) -> Result<()> {
    /// let buf = dev.create_buffer(BufferDesc {
    ///     size: 256 * 256 * 4,
    ///     usage: BufferUsage::READBACK | BufferUsage::TRANSFER_DST,
    ///     memory: MemoryUsage::Readback,
    /// })?;
    /// // Allocated once, outside the frame loop.
    /// let mut pixels = vec![0u8; 256 * 256 * 4];
    /// let mut copy: Option<Submission> =
    ///     Some(dev.copy_texture_region_to_buffer(tex, TextureRegion::rect(0, 0, 256, 256), buf, 0, 1024)?);
    /// // Once per frame, after this frame's own work is submitted:
    /// if let Some(sub) = &copy {
    ///     if sub.poll()? == SubmissionStatus::Complete {
    ///         dev.read_buffer_into(buf, 0, &mut pixels)?;
    ///         copy = None;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Check [`crate::TransferCaps::buffer_texture_copy`]: the default
    /// refuses.
    fn copy_texture_region_to_buffer(
        &self,
        _texture: TextureHandle,
        _region: TextureRegion,
        _buffer: BufferHandle,
        _offset: u64,
        _bytes_per_row: u32,
    ) -> Result<Submission> {
        Err(GpuError::Unsupported(
            "texture-to-buffer region copy".into(),
        ))
    }

    /// Fills the mip chain by blitting each level from the previous one at half size. Needs
    /// `mip_levels > 1` and `TRANSFER_SRC | TRANSFER_DST`.
    ///
    /// The default returns `GpuError::UnsupportedFeatures`.
    fn generate_mipmaps(&self, _texture: TextureHandle) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::GRAPHICS))
    }

    /// Nothing may still reference the image, as with `destroy_buffer`. A stale handle is a no-op.
    fn destroy_texture(&self, texture: TextureHandle);

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    fn destroy_sampler(&self, sampler: SamplerHandle);

    /// When `false`, `create_sampler` treats any anisotropy as `1` instead of failing. This is a
    /// quality setting, not a hard requirement.
    fn supports_anisotropic_filtering(&self) -> bool {
        false
    }

    /// Whether the device is lost: reset, removed, or hung. A lost device never recovers. Every
    /// handle is dead, and calls fail with `GpuError::DeviceLost`. To recover, drop it, open a
    /// fresh device, and re-create resources from host copies. The check is cheap enough to poll
    /// once a frame. The default is `false`.
    fn is_lost(&self) -> bool {
        false
    }

    /// The default returns `UnsupportedFeatures`, since there is no compute.
    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    fn destroy_shader(&self, _shader: ShaderHandle) {}

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Same as `create_compute_pipeline`. `hints` is applied where supported and ignored elsewhere.
    fn create_compute_pipeline_with(
        &self,
        desc: ComputePipelineDesc<'_>,
        hints: ComputePipelineHints,
    ) -> Result<PipelineHandle> {
        let _ = hints;
        self.create_compute_pipeline(desc)
    }

    fn destroy_pipeline(&self, _pipeline: PipelineHandle) {}

    /// `None` when unreported, or when `ComputePipelineHints::capture_stats` is not set on backends
    /// that need it.
    fn pipeline_stats(&self, _pipeline: PipelineHandle) -> Option<PipelineStats> {
        None
    }

    /// Dispatches a compute kernel and blocks until the GPU work completes. `submit` is the
    /// pollable, bounded version.
    ///
    /// `bindings.buffers` are bindless SSBO slots, read with `nonuniformEXT`. `bindings.scalars`
    /// become push constants, packed after the buffer indices. `grid` is the number of workgroups
    /// per axis.
    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Dispatches a batch of ops as one submission that blocks once, instead of one per op. Chained
    /// kernels (e.g. `relu(add(a, b))`) should use this instead of a `dispatch` loop.
    ///
    /// The default calls `dispatch` per op, so unimplemented backends stay correct.
    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        for op in ops {
            self.dispatch(op.pipeline, op.bindings, op.grid)?;
        }
        Ok(())
    }

    /// Starts a dispatch and returns its submission. `cycle_id` is opaque backend metadata, used to
    /// reject late completions.
    ///
    /// The default dispatches synchronously and returns a completed handle.
    fn submit(
        &self,
        cycle_id: u64,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<Submission> {
        self.submit_batch(
            cycle_id,
            &[DispatchOp {
                pipeline,
                bindings,
                grid,
            }],
        )
    }

    /// Waits for a submission, up to a timeout. A `wait` timeout does not cancel the work. Keep the
    /// handle and all referenced resources until it completes.
    ///
    /// The default has no async submission and runs synchronously.
    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.dispatch_batch(ops)?;
        Ok(Box::new(CompletedSubmission::new(cycle_id)))
    }

    /// Runs a mixed batch of ordered copies and dispatches as one unit, so later ops see
    /// earlier writes.
    ///
    /// There is no native mixing, so the default runs synchronously.
    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        for op in ops {
            match op {
                ComputeOp::CopyBuffer(copy) => self.copy_buffer(
                    copy.src,
                    copy.src_offset,
                    copy.dst,
                    copy.dst_offset,
                    copy.len,
                )?,
                ComputeOp::Dispatch(dispatch) => {
                    self.dispatch(dispatch.pipeline, dispatch.bindings, dispatch.grid)?
                }
            }
        }
        Ok(Box::new(CompletedSubmission::new(cycle_id)))
    }
}
