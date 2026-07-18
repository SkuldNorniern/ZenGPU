use std::fmt::{Debug, Formatter, Result as FmtResult};

use zengpu_hal::{AdapterInfo, DeviceLimits, DeviceRequest, GpuAdapter, HalCapabilities, Result};

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

    /// Portable physical-device limits reported by the backend.
    pub fn limits(&self) -> DeviceLimits {
        self.inner.limits()
    }

    /// Open a logical [`Device`] from this adapter.
    ///
    /// Returns an error if `req.required` features are absent on the adapter
    /// or device creation fails.
    pub fn open(&self, req: DeviceRequest) -> Result<Device> {
        self.inner.open(req).map(|inner| Device { inner })
    }

    /// Open a logical [`Device`] with default settings.
    ///
    /// Shorthand for `open(DeviceRequest::default())`.
    pub fn open_default(&self) -> Result<Device> {
        self.open(DeviceRequest::default())
    }
}

impl Debug for Adapter {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let info = self.inner.info();
        f.debug_struct("Adapter")
            .field("name", &info.name)
            .field("backend", &info.backend)
            .finish()
    }
}
