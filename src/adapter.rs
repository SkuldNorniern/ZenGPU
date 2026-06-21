use zengpu_hal::{AdapterInfo, DeviceRequest, GpuAdapter, HalCapabilities, Result};

use crate::device::Device;

/// Physical GPU or CPU adapter from any enabled backend. Get one from
/// [`crate::Instance::request_adapter`].
pub struct Adapter {
    pub(crate) inner: Box<dyn GpuAdapter>,
}

impl Adapter {
    /// Human-readable adapter name, PCI IDs, and backend tag.
    pub fn info(&self) -> &AdapterInfo {
        self.inner.info()
    }

    /// Graphics and compute capabilities this adapter can expose.
    pub fn capabilities(&self) -> HalCapabilities {
        self.inner.capabilities()
    }

    /// Opens a logical [`Device`].
    ///
    /// # Errors
    ///
    /// Returns an error if a feature in `req.required` is missing or device creation fails.
    pub fn open(&self, req: DeviceRequest) -> Result<Device> {
        self.inner.open(req).map(|inner| Device { inner })
    }

    /// Shorthand for `open(DeviceRequest::default())`.
    pub fn open_default(&self) -> Result<Device> {
        self.open(DeviceRequest::default())
    }
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info = self.inner.info();
        f.debug_struct("Adapter")
            .field("name", &info.name)
            .field("backend", &info.backend)
            .finish()
    }
}
