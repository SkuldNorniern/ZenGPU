//! ZenGPU DirectX 12 backend — graphics + compute on Windows.
//!
//! Skeleton: instance construction and platform detection only. DXGI adapter
//! enumeration and D3D12 device creation are wired up once `windows-sys` /
//! `d3d12` bindings land. All device operations return
//! `GpuError::Backend("not yet implemented")`.

#[allow(unused_imports)]
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, DeviceRequest,
    DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, Result, SamplerDesc,
    SamplerHandle, TextureDesc, TextureHandle,
};

// ── Dx12Instance ──────────────────────────────────────────────────────────────

/// Entry-point for the DirectX 12 backend. On non-Windows platforms
/// [`enumerate_adapters`] always returns empty.
///
/// [`enumerate_adapters`]: Dx12Instance::enumerate_adapters
pub struct Dx12Instance;

impl Dx12Instance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Dx12Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for Dx12Instance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        if !cfg!(target_os = "windows") {
            return Vec::new();
        }
        log::debug!("dx12: adapter enumeration not yet implemented");
        Vec::new()
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        None
    }
}

// ── Dx12Adapter ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct Dx12Adapter {
    info: AdapterInfo,
}

impl GpuAdapter for Dx12Adapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }
}

// ── Dx12Device ────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct Dx12Device;

impl GpuDevice for Dx12Device {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, _desc: BufferDesc) -> Result<BufferHandle> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn write_buffer(&self, _buffer: BufferHandle, _offset: u64, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn read_buffer(&self, _buffer: BufferHandle, _offset: u64, _len: u64) -> Result<Vec<u8>> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn destroy_buffer(&self, _buffer: BufferHandle) {}

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("dx12: not yet implemented".into()))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = Dx12Instance::new();
        let _ = inst.enumerate_adapters();
    }
}
