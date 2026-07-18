use core::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use zengpu_hal::{
    Bindings, BufferDesc, BufferHandle, ComputePipelineDesc, DeviceLimits, DispatchOp, GpuDevice,
    HalCapabilities, PipelineHandle, Result, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    Submission, TextureDesc, TextureHandle,
};

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("Device")
            .field("capabilities", &self.inner.capabilities())
            .finish_non_exhaustive()
    }
}

/// An opened logical GPU (or CPU) device from any enabled backend. Obtained
/// from [`crate::Adapter::open`].
///
/// Proxies every [`GpuDevice`] method so callers work against `Device`
/// without importing the HAL trait. For backend-specific operations (Vulkan
/// swapchains, CUDA context tuning) use the downcast helpers below.
pub struct Device {
    pub(crate) inner: Box<dyn GpuDevice>,
}

impl Device {
    // ── Capability ────────────────────────────────────────────────────────────

    pub fn capabilities(&self) -> HalCapabilities {
        self.inner.capabilities()
    }

    pub fn limits(&self) -> DeviceLimits {
        self.inner.limits()
    }

    // ── Buffer API ────────────────────────────────────────────────────────────

    pub fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.inner.create_buffer(desc)
    }

    pub fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.inner.write_buffer(buffer, offset, data)
    }

    pub fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_buffer(buffer, offset, len)
    }

    pub fn destroy_buffer(&self, buffer: BufferHandle) {
        self.inner.destroy_buffer(buffer)
    }

    // ── Texture API ───────────────────────────────────────────────────────────

    pub fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        self.inner.create_texture(desc)
    }

    pub fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        self.inner.upload_texture_data(texture, data)
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

    // ── Compute API ───────────────────────────────────────────────────────────

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

    // ── Backend downcast ──────────────────────────────────────────────────────

    /// Access the underlying [`zengpu_hal::GpuDevice`] trait object directly.
    /// Needed when passing to compute/BLAS APIs that accept `&dyn GpuDevice`.
    pub fn as_dyn(&self) -> &dyn GpuDevice {
        &*self.inner
    }

    /// Downcast to [`zengpu_vulkan::VulkanDevice`] for graphics-specific work
    /// (swapchain creation, frame-graph, render passes).
    /// Returns `None` when the underlying backend is not Vulkan.
    #[cfg(feature = "vulkan")]
    pub fn as_vulkan(&self) -> Option<&zengpu_vulkan::VulkanDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// Downcast to [`zengpu_cpu::CpuDevice`] for registering conformance
    /// kernels via [`zengpu_cpu::CpuDevice::register_kernel`].
    /// Returns `None` when the underlying backend is not CPU.
    #[cfg(feature = "cpu")]
    pub fn as_cpu(&self) -> Option<&zengpu_cpu::CpuDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// Downcast to [`zengpu_metal::MetalDevice`] for Metal-specific work.
    /// Returns `None` when the underlying backend is not Metal.
    #[cfg(feature = "metal")]
    pub fn as_metal(&self) -> Option<&zengpu_metal::MetalDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// Downcast to [`zengpu_hip::HipDevice`] for HIP-specific work.
    /// Returns `None` when the underlying backend is not HIP.
    #[cfg(feature = "hip")]
    pub fn as_hip(&self) -> Option<&zengpu_hip::HipDevice> {
        self.inner.as_any().downcast_ref()
    }

    /// Downcast to [`zengpu_dx12::Dx12Device`] for DirectX 12-specific work.
    /// Returns `None` when the underlying backend is not DX12.
    #[cfg(feature = "dx12")]
    pub fn as_dx12(&self) -> Option<&zengpu_dx12::Dx12Device> {
        self.inner.as_any().downcast_ref()
    }
}

/// Blanket [`GpuDevice`] impl lets `&Device` coerce to `&dyn GpuDevice`.
/// Inherent methods (same name) still take precedence for direct `.method()`
/// calls; this impl only activates when the trait is in scope and a trait
/// object / bound is required.
impl GpuDevice for Device {
    fn as_any(&self) -> &dyn Any {
        // Delegate into the concrete backend so `downcast_ref::<VulkanDevice>()` works.
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

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_buffer(buffer, offset, len)
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
}
