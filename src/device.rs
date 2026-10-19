use core::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use zengpu_hal::{
    BackendPreference, Bindings, BufferDesc, BufferHandle, ComputeOp, ComputePipelineDesc,
    ComputePipelineHints, DeviceLimits, DispatchOp, Format, FormatCaps, GpuDevice, HalCapabilities,
    MemoryBudget, PipelineHandle, ReadSubmission, Result, SamplerDesc, SamplerHandle, ShaderDesc,
    ShaderHandle, Submission, TextureDesc, TextureHandle, TextureRegion, TextureWrite,
};

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("Device")
            .field("capabilities", &self.inner.capabilities())
            .finish_non_exhaustive()
    }
}

/// Logical GPU (or CPU) device from any enabled backend. Proxies [`GpuDevice`] methods, so the HAL
/// trait need not be imported.
///
/// Backend-specific work: use the `as_*` downcasts.
pub struct Device {
    pub(crate) inner: Box<dyn GpuDevice>,
}

impl Device {
    pub fn capabilities(&self) -> HalCapabilities {
        self.inner.capabilities()
    }

    pub fn limits(&self) -> DeviceLimits {
        self.inner.limits()
    }

    /// Whether the device is lost; see [`GpuDevice::is_lost`] for how to recover.
    pub fn is_lost(&self) -> bool {
        self.inner.is_lost()
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.inner.create_buffer(desc)
    }

    pub fn buffer_size(&self, buffer: BufferHandle) -> Result<u64> {
        self.inner.buffer_size(buffer)
    }

    pub fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.inner.write_buffer(buffer, offset, data)
    }

    pub fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_buffer(buffer, offset, len)
    }

    /// Queues an upload into `buffer` on the asynchronous path. Wait on the returned `Submission`.
    pub fn submit_write_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<Submission> {
        self.inner
            .submit_write_buffer(cycle_id, buffer, offset, data)
    }

    /// Queues a readback from `buffer` on the asynchronous path. Read the bytes from the returned
    /// `ReadSubmission` once it completes.
    pub fn submit_read_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        self.inner.submit_read_buffer(cycle_id, buffer, offset, len)
    }

    pub fn read_buffer_into(
        &self,
        buffer: BufferHandle,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        self.inner.read_buffer_into(buffer, offset, dst)
    }

    pub fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        self.inner
            .copy_buffer(src, src_offset, dst, dst_offset, len)
    }

    pub fn destroy_buffer(&self, buffer: BufferHandle) {
        self.inner.destroy_buffer(buffer)
    }

    pub fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        self.inner.create_texture(desc)
    }

    pub fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        self.inner.upload_texture_data(texture, data)
    }

    /// Non-blocking texture rect write from host memory. Check
    /// [`TransferCaps::texture_region_write`](crate::hal::TransferCaps) first.
    pub fn write_textures(&self, writes: &[TextureWrite<'_>]) -> Result<Submission> {
        self.inner.write_textures(writes)
    }

    pub fn copy_buffer_to_texture(
        &self,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
        texture: TextureHandle,
        region: TextureRegion,
    ) -> Result<Submission> {
        self.inner
            .copy_buffer_to_texture(buffer, offset, bytes_per_row, texture, region)
    }

    pub fn copy_texture_region_to_buffer(
        &self,
        texture: TextureHandle,
        region: TextureRegion,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
    ) -> Result<Submission> {
        self.inner
            .copy_texture_region_to_buffer(texture, region, buffer, offset, bytes_per_row)
    }

    pub fn destroy_texture(&self, texture: TextureHandle) {
        self.inner.destroy_texture(texture)
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle> {
        self.inner.create_sampler(desc)
    }

    pub fn destroy_sampler(&self, sampler: SamplerHandle) {
        self.inner.destroy_sampler(sampler)
    }

    pub fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        self.inner.create_shader(desc)
    }

    pub fn destroy_shader(&self, shader: ShaderHandle) {
        self.inner.destroy_shader(shader)
    }

    pub fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        self.inner.create_compute_pipeline(desc)
    }

    pub fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        self.inner.destroy_pipeline(pipeline)
    }

    pub fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.inner.dispatch(pipeline, bindings, grid)
    }

    pub fn submit(
        &self,
        cycle_id: u64,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<Submission> {
        self.inner.submit(cycle_id, pipeline, bindings, grid)
    }

    pub fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.inner.submit_batch(cycle_id, ops)
    }

    pub fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        self.inner.submit_compute_ops(cycle_id, ops)
    }

    /// Underlying [`zengpu_hal::GpuDevice`] trait object, for compute/BLAS APIs taking
    /// `&dyn GpuDevice`.
    pub fn as_dyn(&self) -> &dyn GpuDevice {
        &*self.inner
    }

    /// `VulkanDevice` for swapchain, frame-graph, render passes. `None` if not Vulkan.
    #[cfg(feature = "vulkan")]
    pub fn as_vulkan(&self) -> Option<&zengpu_vulkan::VulkanDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// `CpuDevice` for `register_kernel`. `None` if not CPU.
    #[cfg(feature = "cpu")]
    pub fn as_cpu(&self) -> Option<&zengpu_cpu::CpuDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// `CudaDevice`, or `None` if not CUDA.
    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> Option<&zengpu_cuda::CudaDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// `MetalDevice`, or `None` if not Metal.
    #[cfg(feature = "metal")]
    pub fn as_metal(&self) -> Option<&zengpu_metal::MetalDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// `HipDevice`, or `None` if not HIP.
    #[cfg(feature = "hip")]
    pub fn as_hip(&self) -> Option<&zengpu_hip::HipDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// `Dx12Device`, or `None` if not DX12.
    #[cfg(feature = "dx12")]
    pub fn as_dx12(&self) -> Option<&zengpu_dx12::Dx12Device> {
        self.inner.as_any().downcast_ref()
    }
}

/// Lets `&Device` coerce to `&dyn GpuDevice`.
///
/// Inherent methods with the same name win for direct calls. The trait methods apply only when the
/// trait is in scope.
impl GpuDevice for Device {
    fn backend(&self) -> BackendPreference {
        self.inner.backend()
    }

    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn capabilities(&self) -> HalCapabilities {
        self.inner.capabilities()
    }

    fn limits(&self) -> DeviceLimits {
        self.inner.limits()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.inner.create_buffer(desc)
    }

    fn buffer_size(&self, buffer: BufferHandle) -> Result<u64> {
        self.inner.buffer_size(buffer)
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_buffer(buffer, offset, len)
    }

    /// Queues an upload into `buffer` on the asynchronous path. Wait on the returned `Submission`.
    fn submit_write_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<Submission> {
        self.inner
            .submit_write_buffer(cycle_id, buffer, offset, data)
    }

    /// Queues a readback from `buffer` on the asynchronous path. Read the bytes from the returned
    /// `ReadSubmission` once it completes.
    fn submit_read_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        self.inner.submit_read_buffer(cycle_id, buffer, offset, len)
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.inner.read_buffer_into(buffer, offset, dst)
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        self.inner
            .copy_buffer(src, src_offset, dst, dst_offset, len)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        self.inner.destroy_buffer(buffer)
    }

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        self.inner.create_texture(desc)
    }

    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        self.inner.upload_texture_data(texture, data)
    }

    fn destroy_texture(&self, texture: TextureHandle) {
        self.inner.destroy_texture(texture)
    }

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle> {
        self.inner.create_sampler(desc)
    }

    fn destroy_sampler(&self, sampler: SamplerHandle) {
        self.inner.destroy_sampler(sampler)
    }

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        self.inner.create_shader(desc)
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        self.inner.destroy_shader(shader)
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        self.inner.create_compute_pipeline(desc)
    }

    fn create_compute_pipeline_with(
        &self,
        desc: ComputePipelineDesc<'_>,
        hints: ComputePipelineHints,
    ) -> Result<PipelineHandle> {
        self.inner.create_compute_pipeline_with(desc, hints)
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        self.inner.destroy_pipeline(pipeline)
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.inner.dispatch(pipeline, bindings, grid)
    }

    fn submit(
        &self,
        cycle_id: u64,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<Submission> {
        self.inner.submit(cycle_id, pipeline, bindings, grid)
    }

    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.inner.submit_batch(cycle_id, ops)
    }

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        self.inner.submit_compute_ops(cycle_id, ops)
    }

    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        self.inner.dispatch_batch(ops)
    }

    fn format_caps(&self, format: Format) -> FormatCaps {
        self.inner.format_caps(format)
    }

    fn memory_budget(&self) -> Option<MemoryBudget> {
        self.inner.memory_budget()
    }

    fn is_lost(&self) -> bool {
        self.inner.is_lost()
    }

    fn device_ordinal(&self) -> i32 {
        self.inner.device_ordinal()
    }

    fn can_peer(&self, peer_ordinal: i32) -> bool {
        self.inner.can_peer(peer_ordinal)
    }

    fn copy_from_peer(
        &self,
        dst: BufferHandle,
        src_device: &dyn GpuDevice,
        src: BufferHandle,
        bytes: u64,
    ) -> Result<()> {
        self.inner.copy_from_peer(dst, src_device, src, bytes)
    }

    fn copy_texture_to_buffer(&self, texture: TextureHandle, buffer: BufferHandle) -> Result<()> {
        self.inner.copy_texture_to_buffer(texture, buffer)
    }

    fn upload_texture_data_region(
        &self,
        texture: TextureHandle,
        mip_level: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        self.inner
            .upload_texture_data_region(texture, mip_level, layer, data)
    }

    fn write_textures(&self, writes: &[TextureWrite<'_>]) -> Result<Submission> {
        self.inner.write_textures(writes)
    }

    fn copy_buffer_to_texture(
        &self,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
        texture: TextureHandle,
        region: TextureRegion,
    ) -> Result<Submission> {
        self.inner
            .copy_buffer_to_texture(buffer, offset, bytes_per_row, texture, region)
    }

    fn copy_texture_region_to_buffer(
        &self,
        texture: TextureHandle,
        region: TextureRegion,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
    ) -> Result<Submission> {
        self.inner
            .copy_texture_region_to_buffer(texture, region, buffer, offset, bytes_per_row)
    }

    fn generate_mipmaps(&self, texture: TextureHandle) -> Result<()> {
        self.inner.generate_mipmaps(texture)
    }

    fn supports_anisotropic_filtering(&self) -> bool {
        self.inner.supports_anisotropic_filtering()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_hal::{CompletedReadback, CompletedSubmission, GpuError, SlotMap, marker};

    struct AsyncOnly;

    fn unsupported<T>() -> Result<T> {
        Err(GpuError::Unsupported(
            "unexpected synchronous fallback".into(),
        ))
    }

    impl GpuDevice for AsyncOnly {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn capabilities(&self) -> HalCapabilities {
            HalCapabilities::compute_only()
        }
        fn create_buffer(&self, _: BufferDesc) -> Result<BufferHandle> {
            unsupported()
        }
        fn buffer_size(&self, _: BufferHandle) -> Result<u64> {
            Ok(64)
        }
        fn write_buffer(&self, _: BufferHandle, _: u64, _: &[u8]) -> Result<()> {
            unsupported()
        }
        fn read_buffer(&self, _: BufferHandle, _: u64, _: u64) -> Result<Vec<u8>> {
            unsupported()
        }
        fn destroy_buffer(&self, _: BufferHandle) {}
        fn create_texture(&self, _: TextureDesc) -> Result<TextureHandle> {
            unsupported()
        }
        fn upload_texture_data(&self, _: TextureHandle, _: &[u8]) -> Result<()> {
            unsupported()
        }
        fn destroy_texture(&self, _: TextureHandle) {}
        fn create_sampler(&self, _: SamplerDesc) -> Result<SamplerHandle> {
            unsupported()
        }
        fn destroy_sampler(&self, _: SamplerHandle) {}
        fn submit_write_buffer(
            &self,
            cycle: u64,
            _: BufferHandle,
            offset: u64,
            data: &[u8],
        ) -> Result<Submission> {
            assert_eq!(offset, 3);
            assert_eq!(data, [1, 2]);
            Ok(Box::new(CompletedSubmission::new(cycle)))
        }
        fn submit_read_buffer(
            &self,
            cycle: u64,
            _: BufferHandle,
            offset: u64,
            len: u64,
        ) -> Result<ReadSubmission> {
            assert_eq!((offset, len), (5, 2));
            Ok(Box::new(CompletedReadback::new(cycle, vec![4, 5])))
        }
    }

    #[test]
    fn public_device_preserves_async_backend_overrides() {
        let device = Device {
            inner: Box::new(AsyncOnly),
        };
        let mut buffers = SlotMap::<marker::Buffer, ()>::new();
        let buffer = buffers.insert(());
        assert_eq!(device.buffer_size(buffer).expect("capacity"), 64);
        assert_eq!(
            device
                .submit_write_buffer(7, buffer, 3, &[1, 2])
                .expect("async upload")
                .cycle_id(),
            7
        );
        let read = device
            .submit_read_buffer(8, buffer, 5, 2)
            .expect("async read");
        assert_eq!(read.cycle_id(), 8);
        assert_eq!(read.into_bytes().expect("bytes"), [4, 5]);
        let hal: &dyn GpuDevice = &device;
        assert_eq!(hal.buffer_size(buffer).expect("HAL capacity"), 64);
        assert_eq!(
            hal.submit_write_buffer(9, buffer, 3, &[1, 2])
                .expect("HAL async upload")
                .cycle_id(),
            9
        );
        let read = hal
            .submit_read_buffer(10, buffer, 5, 2)
            .expect("HAL async read");
        assert_eq!(read.cycle_id(), 10);
        assert_eq!(read.into_bytes().expect("HAL bytes"), [4, 5]);
    }
}
