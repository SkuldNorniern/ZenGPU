//! Vulkan physical-device adapter (plan §22).

use std::sync::Arc;

use ash::vk;
use zengpu_hal::{AdapterInfo, DeviceRequest, GpuAdapter, GpuDevice, HalCapabilities};

use crate::device::VulkanDevice;
use crate::instance::VulkanShared;

/// Wraps a `VkPhysicalDevice` and implements [`GpuAdapter`].
pub struct VulkanAdapter {
    pub(crate) shared: Arc<VulkanShared>,
    pub(crate) physical: vk::PhysicalDevice,
    info: AdapterInfo,
}

// vk::PhysicalDevice is a handle (u64 on 64-bit), safe to send across threads.
unsafe impl Send for VulkanAdapter {}
unsafe impl Sync for VulkanAdapter {}

impl VulkanAdapter {
    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        info: AdapterInfo,
    ) -> Self {
        Self {
            shared,
            physical,
            info,
        }
    }
}

impl GpuAdapter for VulkanAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn open(&self, req: DeviceRequest) -> zengpu_hal::Result<Box<dyn GpuDevice>> {
        let device =
            VulkanDevice::new(Arc::clone(&self.shared), self.physical, req)?;
        Ok(Box::new(device))
    }
}
