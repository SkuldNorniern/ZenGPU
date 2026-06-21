//! ZenGPU Apple Metal backend — graphics + compute on macOS/iOS.
//!
//! Skeleton: instance construction and platform detection only. All device
//! operations return `GpuError::Backend("not yet implemented")` until the
//! Metal API bindings are wired up.

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, DeviceRequest,
    DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, Result,
    SamplerDesc, SamplerHandle, TextureDesc, TextureHandle,
};

// ── MetalInstance ─────────────────────────────────────────────────────────────

/// Entry-point for the Metal backend. On non-Apple platforms
/// [`enumerate_adapters`] always returns empty.
///
/// [`enumerate_adapters`]: MetalInstance::enumerate_adapters
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetalInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for MetalInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        if !cfg!(target_os = "macos") {
            return Vec::new();
        }
        log::debug!("metal: adapter enumeration not yet implemented");
        Vec::new()
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        None
    }
}

// ── MetalAdapter ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct MetalAdapter {
    info: AdapterInfo,
}

impl GpuAdapter for MetalAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }
}

// ── MetalDevice ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct MetalDevice;

impl GpuDevice for MetalDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, _desc: BufferDesc) -> Result<BufferHandle> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn write_buffer(&self, _buffer: BufferHandle, _offset: u64, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn read_buffer(&self, _buffer: BufferHandle, _offset: u64, _len: u64) -> Result<Vec<u8>> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn destroy_buffer(&self, _buffer: BufferHandle) {}

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("metal: not yet implemented".into()))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = MetalInstance::new();
        let _ = inst.enumerate_adapters();
    }
}
