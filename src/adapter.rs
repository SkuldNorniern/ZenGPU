use zengpu_hal::{AdapterInfo, DeviceRequest, GpuAdapter, HalCapabilities, Result};

use crate::device::Device;

/// A physical GPU (or CPU) adapter from any enabled backend. Obtained from
/// [`crate::Instance::enumerate_adapters`] or [`crate::Instance::request_adapter`].
pub struct Adapter {
    pub(crate) inner: Box<dyn GpuAdapter>,
}

impl Adapter {
    /// Human-readable adapter name, PCI IDs, and backend tag.
    pub fn info(&self) -> &AdapterInfo {
        self.inner.info()
    }

    /// Which HAL shapes (graphics / compute) this adapter can expose.
    pub fn capabilities(&self) -> HalCapabilities {
        self.inner.capabilities()
    }

    /// Open a logical [`Device`] from this adapter.
    ///
    /// Returns an error if `req.required` features are absent on the adapter
    /// or device creation fails.
    pub fn open(&self, req: DeviceRequest) -> Result<Device> {
        self.inner.open(req).map(|inner| Device { inner })
    }
}
