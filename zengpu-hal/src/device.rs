//! Backend-independent device trait. Object-safe: `Box<dyn GpuDevice>` or `Arc<dyn GpuDevice>`
//! select at runtime.

use core::any::Any;

use crate::command::{Bindings, ComputeOp, DispatchOp};
use crate::desc::{BufferDesc, ComputePipelineDesc, SamplerDesc, ShaderDesc, TextureDesc};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::{DeviceLimits, HalCapabilities};
use crate::submission::{CompletedSubmission, Submission};
use crate::types::Features;

/// `Send + Sync`, so worker threads can allocate and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Returns the backend device for backend-specific calls, e.g. a Vulkan swapchain
    /// from a `VulkanDevice`.
    fn as_any(&self) -> &dyn Any;

    fn capabilities(&self) -> HalCapabilities;

    fn limits(&self) -> DeviceLimits {
        DeviceLimits::default()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Requires `BufferUsage::READBACK`.
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

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

    /// Uploads `data` to one mip level of one array layer (or cube face). Use this for texture
    /// arrays, cubemaps and pre-baked mip chains.
    ///
    /// [`Self::upload_texture_data`] only targets mip `0` and layer `0`. Backends without an
    /// implementation return [`GpuError::UnsupportedFeatures`].
    fn upload_texture_data_region(
        &self,
        _texture: TextureHandle,
        _mip_level: u32,
        _layer: u32,
        _data: &[u8],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::GRAPHICS))
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

    fn destroy_pipeline(&self, _pipeline: PipelineHandle) {}

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
