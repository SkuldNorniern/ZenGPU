//! Instance and adapter traits, the entry point into the HAL.
//!
//! `GpuInstance` enumerates adapters. `GpuAdapter` opens a logical device. Both are object-safe, so
//! the backend is picked at runtime without monomorphisation.

use crate::device::GpuDevice;
use crate::error::Result;
use crate::request::{AdapterRequest, DeviceRequest, HalCapabilities};
use crate::surface::{GpuSurface, WindowHandles};
use crate::types::BackendPreference;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Discrete,
    Integrated,
    /// Software adapter, e.g. the ZenGPU reference oracle.
    Cpu,
    Virtual,
    Unknown,
}

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

/// A physical adapter (GPU, CPU, or virtual). Reports capabilities and opens a logical device.
/// `Send + Sync` so adapter selection can run off the main thread.
pub trait GpuAdapter: Send + Sync {
    fn info(&self) -> &AdapterInfo;

    fn capabilities(&self) -> HalCapabilities;

    /// Opens a logical device for this adapter.
    ///
    /// # Errors
    ///
    /// Returns [`crate::GpuError`] if `req.required` features are missing or creation fails.
    fn open(&self, req: DeviceRequest) -> Result<Box<dyn GpuDevice>>;
}

/// Entry point for a backend: enumerates adapters, selects one for a request, and creates
/// presentable surfaces.
///
/// `Send + Sync`: create once, share across threads.
pub trait GpuInstance: Send + Sync {
    /// Returns the adapters in driver enumeration order.
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>>;

    /// Picks the best adapter by priority: device type, power preference, then enumeration order.
    /// Returns `None` if nothing matches.
    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>>;

    /// Creates the surface and swapchain together. Window must outlive the returned surface.
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
        let _ = t;
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
