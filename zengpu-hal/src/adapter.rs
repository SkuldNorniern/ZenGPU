//! Instance and adapter traits — the top of the HAL entry-point chain (plan §22).
//!
//! `GpuInstance` enumerates physical adapters visible to a backend.
//! `GpuAdapter` opens a logical device from one of those adapters.
//! Both are object-safe so the backend is selectable at runtime without
//! monomorphisation at the call site.

use crate::device::GpuDevice;
use crate::error::Result;
use crate::request::{AdapterRequest, DeviceRequest, HalCapabilities};
use crate::surface::{GpuSurface, WindowHandles};
use crate::types::BackendPreference;

/// Physical device class reported by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Discrete GPU — separate chip, dedicated VRAM (preferred for throughput).
    Discrete,
    /// Integrated GPU — on-die, shared system memory.
    Integrated,
    /// CPU software implementation (e.g. the ZenGPU reference oracle).
    Cpu,
    /// Virtualised device inside a VM or emulator.
    Virtual,
    /// Driver could not classify the device.
    Unknown,
}

/// Human-readable description of a physical adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    /// PCI vendor ID (0 for software/CPU adapters).
    pub vendor: u32,
    /// PCI device ID (0 for software/CPU adapters).
    pub device: u32,
    pub device_type: DeviceType,
    pub backend: BackendPreference,
}

/// A physical adapter (GPU, CPU, or virtual): reports capabilities and opens a
/// logical device. `Send + Sync` so adapter selection can happen off the main
/// thread (plan D5).
pub trait GpuAdapter: Send + Sync {
    /// Static description of this physical adapter.
    fn info(&self) -> &AdapterInfo;

    /// Which HAL shapes this adapter can expose.
    fn capabilities(&self) -> HalCapabilities;

    /// Open a logical [`GpuDevice`] from this adapter.
    ///
    /// Returns [`crate::GpuError`] if `req.required` features are absent or
    /// device creation fails.
    fn open(&self, req: DeviceRequest) -> Result<Box<dyn GpuDevice>>;
}

/// Entry-point for a backend: enumerates physical adapters, selects the best
/// one for a given request, and creates presentable surfaces.
///
/// `Send + Sync` — the instance can be created once and shared across threads.
pub trait GpuInstance: Send + Sync {
    /// Every adapter this backend can see, in driver enumeration order.
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>>;

    /// Pick the best adapter for `req` following priority: device-type match,
    /// then power preference, then enumeration order. Returns `None` when no
    /// adapter satisfies the request.
    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>>;

    /// Create a fully configured presentable surface (plan G2).
    ///
    /// The surface and swapchain are created together because Vulkan separates
    /// instance-level surface creation from device-level swapchain creation.
    /// Pass the device that will render to this surface and the initial config.
    /// The window must outlive the returned surface.
    fn create_surface(
        &self,
        handles: &WindowHandles,
        device: &dyn crate::device::GpuDevice,
        config: crate::desc::SurfaceConfig,
    ) -> Result<Box<dyn GpuSurface>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_type_is_copy() {
        let t = DeviceType::Discrete;
        let _ = t;
        let _ = t; // copy, not move
    }

    #[test]
    fn adapter_info_fields() {
        let info = AdapterInfo {
            name: "Test Adapter".to_string(),
            vendor: 0x10de,
            device: 0x2684,
            device_type: DeviceType::Discrete,
            backend: BackendPreference::Vulkan,
        };
        assert_eq!(info.vendor, 0x10de);
        assert_eq!(info.device_type, DeviceType::Discrete);
    }
}
