//! ZenGPU Apple Metal backend: graphics and compute on macOS/iOS.
//!
//! On macOS the instance enumerates all `MTLDevice`s, including eGPUs. Elsewhere it compiles but
//! returns no adapters.

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BufferDesc, BufferHandle, DeviceRequest, GpuAdapter, GpuDevice,
    GpuError, GpuInstance, HalCapabilities, Result, SamplerDesc, SamplerHandle, TextureDesc,
    TextureHandle,
};

#[cfg(target_os = "macos")]
use zengpu_hal::{BackendPreference, DeviceType};

/// Metal backend entry point. On macOS, [`enumerate_adapters`] returns one entry per `MTLDevice`
/// (integrated and eGPUs). Elsewhere it returns empty.
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
        #[cfg(target_os = "macos")]
        {
            metal::Device::all()
                .into_iter()
                .map(|dev| {
                    let device_type = if dev.is_low_power() {
                        DeviceType::Integrated
                    } else {
                        DeviceType::Discrete
                    };
                    let info = AdapterInfo {
                        name: dev.name().to_string(),
                        vendor: 0x106b,
                        device: 0,
                        device_type,
                        backend: BackendPreference::Metal,
                    };
                    Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Vec::new()
        }
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        #[cfg(target_os = "macos")]
        {
            let all = self.enumerate_adapters();
            metal::Device::system_default().map(|dev| {
                let device_type = if dev.is_low_power() {
                    DeviceType::Integrated
                } else {
                    DeviceType::Discrete
                };
                let info = AdapterInfo {
                    name: dev.name().to_string(),
                    vendor: 0x106b,
                    device: 0,
                    device_type,
                    backend: BackendPreference::Metal,
                };
                Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
            })
            .or_else(|| all.into_iter().next())
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

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
        Err(GpuError::Backend(
            "metal: device open not yet implemented".into(),
        ))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = MetalInstance::new();
        let _ = inst.enumerate_adapters();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn enumerates_at_least_one_adapter_on_macos() {
        let inst = MetalInstance::new();
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty(), "expected at least one Metal adapter on macOS");
        for a in &adapters {
            assert!(!a.info().name.is_empty());
            assert!(a.capabilities().graphics);
        }
    }
}
