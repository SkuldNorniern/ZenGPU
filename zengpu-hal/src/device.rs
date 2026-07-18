//! The device trait — the backend-independent handle onto a GPU (or the CPU
//! reference). This is the first slice of the split HAL; it grows as
//! backends come online (compute dispatch, graphics passes, surfaces).
//!
//! Object-safe, so a backend can be held as `Box<dyn GpuDevice>` / `Arc<dyn
//! GpuDevice>` and selected at runtime.

use core::any::Any;

use crate::command::{Bindings, ComputeOp, DispatchOp};
use crate::desc::{BufferDesc, ComputePipelineDesc, SamplerDesc, ShaderDesc, TextureDesc};
use crate::error::{GpuError, Result};
use crate::handle::{BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, TextureHandle};
use crate::request::{DeviceLimits, HalCapabilities};
use crate::submission::{CompletedSubmission, Submission};
use crate::types::Features;

/// A created device: owns GPU resources and runs work. `Send + Sync` so worker
/// threads can allocate and upload concurrently.
pub trait GpuDevice: Send + Sync {
    /// Downcast to the concrete backend type. Required for backend-specific
    /// operations (e.g. creating a Vulkan swapchain from a VulkanDevice).
    fn as_any(&self) -> &dyn Any;

    /// Which HAL capabilities this backend implements.
    fn capabilities(&self) -> HalCapabilities;

    /// Limits of the created logical device and selected queue path.
    fn limits(&self) -> DeviceLimits {
        DeviceLimits::default()
    }

    /// Create a buffer. The returned handle is generational.
    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle>;

    /// Upload `data` into `buffer` starting at byte `offset`.
    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;

    /// Read `len` bytes from `buffer` starting at byte `offset`. The buffer must
    /// have been created with [`crate::BufferUsage::READBACK`].
    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Read bytes into caller-owned storage without requiring an allocation.
    /// The buffer must have [`crate::BufferUsage::READBACK`]; the read range is
    /// `offset..offset + dst.len()`.
    ///
    /// The default preserves compatibility by using [`Self::read_buffer`].
    /// Real-time backends override this with a direct copy into `dst`.
    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        let bytes = self.read_buffer(buffer, offset, dst.len() as u64)?;
        dst.copy_from_slice(&bytes);
        Ok(())
    }

    /// Copy `len` bytes between buffers on this device. `src` must have
    /// [`crate::BufferUsage::TRANSFER_SRC`] and `dst` must have
    /// [`crate::BufferUsage::TRANSFER_DST`]. The call is synchronous: the
    /// copied bytes are available to subsequent device or host operations
    /// when it returns.
    ///
    /// Copies within the same buffer are rejected so every backend has the
    /// same overlap semantics. A zero-length copy is a no-op after handle and
    /// usage validation.
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

    /// Backend device ordinal, or `-1` when the backend does not expose one.
    fn device_ordinal(&self) -> i32 {
        -1
    }

    /// Whether direct device-to-device copies from `peer_ordinal` are usable.
    fn can_peer(&self, _peer_ordinal: i32) -> bool {
        false
    }

    /// Copy `bytes` from a buffer owned by `src_device` into `dst`.
    /// Backends without multi-device copy support return an error by default.
    fn copy_from_peer(
        &self,
        _dst: BufferHandle,
        _src_device: &dyn GpuDevice,
        _src: BufferHandle,
        _bytes: u64,
    ) -> Result<()> {
        Err(GpuError::Unsupported("peer device copy".into()))
    }

    /// Destroy `buffer`, invalidating its handle. A stale handle is a no-op.
    fn destroy_buffer(&self, buffer: BufferHandle);

    // ── Texture API ───────────────────────────────────────────────────────────

    /// Allocate a GPU-resident texture. Data must be uploaded separately via
    /// [`Self::upload_texture_data`].
    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle>;

    /// Upload `data` to the texture's base mip level, layer `0`. `data` must
    /// be tightly packed pixels in the format specified at creation (RGBA8 =
    /// 4 bytes per pixel). Blocks until the upload is complete; an
    /// asynchronous path may be added later.
    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()>;

    /// Upload `data` to one mip level of one array layer (or cube face) of
    /// the texture. Use this for texture arrays, cubemaps, and pre-baked mip
    /// chains; [`Self::upload_texture_data`] only ever targets mip `0`, layer
    /// `0`. Default implementation reports
    /// [`GpuError::UnsupportedFeatures`] on backends that have not
    /// implemented it.
    fn upload_texture_data_region(
        &self,
        _texture: TextureHandle,
        _mip_level: u32,
        _layer: u32,
        _data: &[u8],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::GRAPHICS))
    }

    /// Generate the full mip chain for `texture` from its base level via
    /// successive blits, each level half the resolution of the last. The
    /// texture must have been created with `mip_levels > 1` and
    /// `TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST`. Default
    /// implementation reports [`GpuError::UnsupportedFeatures`] on backends
    /// that have not implemented it.
    fn generate_mipmaps(&self, _texture: TextureHandle) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::GRAPHICS))
    }

    /// Destroy a texture, invalidating its handle. A stale handle is a no-op.
    /// The caller must ensure no surface / pipeline is still referencing the
    /// underlying image (the same contract as `destroy_buffer`).
    fn destroy_texture(&self, texture: TextureHandle);

    /// Create a texture sampler.
    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle>;

    /// Destroy a sampler, invalidating its handle.
    fn destroy_sampler(&self, sampler: SamplerHandle);

    /// Whether the device supports anisotropic filtering
    /// ([`SamplerDesc::anisotropy`] above `1`). When `false`,
    /// [`create_sampler`](Self::create_sampler) silently treats any requested
    /// anisotropy as `1` rather than failing, since it is a quality setting,
    /// not a hard requirement of the sampled output.
    fn supports_anisotropic_filtering(&self) -> bool {
        false
    }

    // ── Compute API ───────────────────────────────────────────────────────────

    /// Upload a SPIR-V shader module. Returns a handle usable for pipeline
    /// creation. Default impl returns `UnsupportedFeatures` on backends that do
    /// not support compute.
    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Destroy a shader module.
    fn destroy_shader(&self, _shader: ShaderHandle) {}

    /// Create a compute pipeline from a shader module + entry point.
    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let _ = desc;
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Destroy a compute pipeline.
    fn destroy_pipeline(&self, _pipeline: PipelineHandle) {}

    /// Synchronously dispatch a compute pipeline. Blocks until GPU work
    /// completes. Use [`Self::submit`] for pollable, bounded completion.
    ///
    /// `bindings.buffers` contains slot indices into the bindless SSBO
    /// table; the shader indexes them with `nonuniformEXT`. `bindings.scalars`
    /// become push-constant values (packed after the buffer indices). `grid` is
    /// the workgroup count along each axis.
    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        Err(GpuError::UnsupportedFeatures(Features::COMPUTE))
    }

    /// Dispatch a sequence of compute pipelines as one GPU submission,
    /// blocking once for the whole batch instead of once per op. Chained
    /// elementwise/kernel sequences (e.g. `relu(add(a, b))`) should use this
    /// instead of calling [`Self::dispatch`] in a loop — each call to
    /// `dispatch` round-trips to the GPU and back, which serializes
    /// independent CPU/GPU work for no benefit when the caller has no
    /// intermediate result to inspect.
    ///
    /// Default implementation calls [`Self::dispatch`] once per op, so
    /// backends that have not implemented batching stay correct.
    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        for op in ops {
            self.dispatch(op.pipeline, op.bindings, op.grid)?;
        }
        Ok(())
    }

    /// Submit one compute dispatch and return a pollable completion handle.
    ///
    /// `cycle_id` is opaque backend metadata used by real-time callers to
    /// reject late completions. The default implementation dispatches
    /// synchronously and returns an already-complete handle.
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

    /// Submit a compute batch and return a pollable completion handle.
    ///
    /// A timeout reported by [`crate::GpuSubmission::wait`] never implies
    /// cancellation. The caller must retain the handle and all referenced
    /// resources until completion. Backends without asynchronous submission
    /// use the synchronous default implementation.
    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.dispatch_batch(ops)?;
        Ok(Box::new(CompletedSubmission::new(cycle_id)))
    }

    /// Submit ordered device-local copies and compute dispatches as one unit.
    /// A later operation observes writes made by every earlier operation.
    /// Backends without native mixed batching execute synchronously.
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
