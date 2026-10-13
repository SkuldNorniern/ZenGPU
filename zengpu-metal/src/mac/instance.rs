//! Finding Metal devices.

use metal::Device;
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceRequest, DeviceType, GpuAdapter,
    GpuDevice, GpuInstance, HalCapabilities, PowerPreference, Result,
};

use super::device::{MetalDevice, capabilities};

/// Apple's PCI vendor ID, reported for every Metal device.
const APPLE_VENDOR: u32 = 0x106b;

/// Entry point for the Metal backend: one adapter per `MTLDevice`,
/// including eGPUs.
#[derive(Default)]
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Self {
        Self
    }

    /// The same as [`new`](Self::new): Metal needs nothing extra to present.
    /// Named like the Vulkan constructor so the two are interchangeable.
    pub fn new_with_surface() -> Result<Self> {
        Ok(Self)
    }

    /// Every adapter, under the name every graphics backend shares.
    pub fn graphics_adapters(&self) -> Vec<MetalAdapter> {
        self.metal_adapters()
    }

    /// The preferred adapter, under the shared graphics-backend name.
    pub fn request_graphics_adapter(&self) -> Option<MetalAdapter> {
        self.request_metal_adapter()
    }

    /// The adapter with this UUID, under the shared graphics-backend name.
    pub fn graphics_adapter_by_uuid(&self, uuid: [u8; 16]) -> Option<MetalAdapter> {
        self.metal_adapter_by_uuid(uuid)
    }

    /// Every Metal device, as concrete adapters.
    pub fn metal_adapters(&self) -> Vec<MetalAdapter> {
        Device::all().into_iter().map(MetalAdapter::new).collect()
    }

    /// The system's preferred device: the one macOS would pick for a
    /// high-performance app.
    pub fn request_metal_adapter(&self) -> Option<MetalAdapter> {
        self.request_metal_adapter_with_pref(PowerPreference::HighPerformance)
    }

    /// The device matching `pref`: a low-power one when asked for and
    /// present, otherwise the system default.
    pub fn request_metal_adapter_with_pref(&self, pref: PowerPreference) -> Option<MetalAdapter> {
        if pref == PowerPreference::LowPower
            && let Some(low) = self
                .metal_adapters()
                .into_iter()
                .find(|a| a.device.is_low_power())
        {
            return Some(low);
        }
        Device::system_default()
            .map(MetalAdapter::new)
            .or_else(|| self.metal_adapters().into_iter().next())
    }

    /// The adapter with this [`MetalAdapter::uuid`], if it is still there.
    pub fn metal_adapter_by_uuid(&self, uuid: [u8; 16]) -> Option<MetalAdapter> {
        self.metal_adapters().into_iter().find(|a| a.uuid() == uuid)
    }
}

impl GpuInstance for MetalInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        self.metal_adapters()
            .into_iter()
            .map(|a| Box::new(a) as Box<dyn GpuAdapter>)
            .collect()
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        self.request_metal_adapter_with_pref(req.power)
            .map(|a| Box::new(a) as Box<dyn GpuAdapter>)
    }
}

/// One `MTLDevice`. Opening it opens that very device.
pub struct MetalAdapter {
    device: Device,
    info: AdapterInfo,
}

// SAFETY: MTLDevice is documented thread-safe.
unsafe impl Send for MetalAdapter {}
unsafe impl Sync for MetalAdapter {}

impl MetalAdapter {
    fn new(device: Device) -> Self {
        // Apple GPUs share memory with the CPU; a removable one is an eGPU.
        let device_type =
            if device.has_unified_memory() || (device.is_low_power() && !device.is_removable()) {
                DeviceType::Integrated
            } else {
                DeviceType::Discrete
            };
        let info = AdapterInfo {
            name: device.name().to_string(),
            vendor: APPLE_VENDOR,
            // The registry ID's low bits tell devices apart within a run.
            device: (device.registry_id() & 0xffff_ffff) as u32,
            device_type,
            backend: BackendPreference::Metal,
        };
        Self { device, info }
    }

    /// An ID for this device that stays the same from one run to the next:
    /// what to save for a "which GPU" setting. Built from the IORegistry
    /// entry ID, which macOS keeps for a device across launches.
    pub fn uuid(&self) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[..4].copy_from_slice(b"MTL\0");
        id[8..].copy_from_slice(&self.device.registry_id().to_le_bytes());
        id
    }

    /// Open this adapter as a concrete [`MetalDevice`], for the graphics API.
    pub fn open_metal(&self, _req: DeviceRequest) -> Result<MetalDevice> {
        MetalDevice::new(self.device.clone())
    }

    /// Open a device that can present, under the shared graphics-backend
    /// name.
    pub fn open_graphics(&self, req: DeviceRequest) -> Result<MetalDevice> {
        self.open_metal(req)
    }
}

impl GpuAdapter for MetalAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        capabilities()
    }

    fn open(&self, req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Ok(Box::new(self.open_metal(req)?))
    }
}
