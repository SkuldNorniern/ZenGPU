//! ZenGPU AMD ROCm/HIP backend — compute HAL only.
//!
//! Skeleton: instance construction only. HIP runtime discovery and device
//! enumeration are wired up once `hipGetDeviceCount` / `hipDeviceGetName`
//! bindings land. All device operations return
//! `GpuError::Backend("not yet implemented")`.

#[allow(unused_imports)]
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, DeviceRequest,
    DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, Result,
    SamplerDesc, SamplerHandle, TextureDesc, TextureHandle,
};

// ── HipInstance ───────────────────────────────────────────────────────────────

/// Entry-point for the ROCm/HIP backend. Enumerates AMD GPUs via the HIP
/// runtime once bindings are added; returns empty in this skeleton.
pub struct HipInstance;

impl HipInstance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HipInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for HipInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        log::debug!("hip: adapter enumeration not yet implemented");
        Vec::new()
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        None
    }
}

// ── HipAdapter ────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct HipAdapter {
    info: AdapterInfo,
}

impl GpuAdapter for HipAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Err(GpuError::Backend("hip: not yet implemented".into()))
    }
}

// ── HipDevice ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct HipDevice;

impl GpuDevice for HipDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, _desc: BufferDesc) -> Result<BufferHandle> {
        Err(GpuError::Backend("hip: not yet implemented".into()))
    }

    fn write_buffer(&self, _buffer: BufferHandle, _offset: u64, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("hip: not yet implemented".into()))
    }

    fn read_buffer(&self, _buffer: BufferHandle, _offset: u64, _len: u64) -> Result<Vec<u8>> {
        Err(GpuError::Backend("hip: not yet implemented".into()))
    }

    fn destroy_buffer(&self, _buffer: BufferHandle) {}

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("hip: compute-only; no texture support".into()))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("hip: compute-only; no texture support".into()))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("hip: compute-only; no sampler support".into()))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = HipInstance::new();
        let _ = inst.enumerate_adapters();
    }
}
