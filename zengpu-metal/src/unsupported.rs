//! Metal off Apple platforms: no adapters, and a device type that cannot be
//! made, so code naming it still builds.

use std::any::Any;

use zengpu_hal::{
    AdapterRequest, BackendPreference, BufferDesc, BufferHandle, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, HalCapabilities, Result, SamplerDesc, SamplerHandle, TextureDesc, TextureHandle,
};

/// Finds no adapters here.
#[derive(Default)]
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Self {
        Self
    }
}

impl GpuInstance for MetalInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        Vec::new()
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        None
    }
}

/// Never constructed off Apple platforms.
pub enum MetalDevice {}

fn unavailable<T>() -> Result<T> {
    Err(GpuError::Backend(
        "metal: unavailable on this platform".into(),
    ))
}

impl GpuDevice for MetalDevice {
    fn backend(&self) -> BackendPreference {
        BackendPreference::Metal
    }

    fn as_any(&self) -> &dyn Any {
        match *self {}
    }
    fn capabilities(&self) -> HalCapabilities {
        match *self {}
    }
    fn create_buffer(&self, _desc: BufferDesc) -> Result<BufferHandle> {
        unavailable()
    }
    fn write_buffer(&self, _buffer: BufferHandle, _offset: u64, _data: &[u8]) -> Result<()> {
        unavailable()
    }
    fn read_buffer(&self, _buffer: BufferHandle, _offset: u64, _len: u64) -> Result<Vec<u8>> {
        unavailable()
    }
    fn destroy_buffer(&self, _buffer: BufferHandle) {}
    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        unavailable()
    }
    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        unavailable()
    }
    fn destroy_texture(&self, _texture: TextureHandle) {}
    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        unavailable()
    }
    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}
