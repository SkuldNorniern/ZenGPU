
use std::sync::Arc;

use ash::vk;
use zengpu_hal::{AdapterInfo, DeviceRequest, GpuAdapter, GpuDevice, HalCapabilities};

use crate::device::VulkanDevice;
use crate::instance::VulkanShared;

/// Wraps a `VkPhysicalDevice`.
pub struct VulkanAdapter {
    pub(crate) shared: Arc<VulkanShared>,
    pub(crate) physical: vk::PhysicalDevice,
    info: AdapterInfo,
}

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

    /// Opens a compute-only device (no swapchain extension).
    fn open(&self, req: DeviceRequest) -> zengpu_hal::Result<Box<dyn GpuDevice>> {
        let device = VulkanDevice::new(Arc::clone(&self.shared), self.physical, req)?;
        Ok(Box::new(device))
    }
}

impl VulkanAdapter {
    /// Whether the swapchain extension is enabled. Required for presenting via [`crate::Swapchain`]
    /// or [`crate::Vulkan2dSurface`].
    pub fn open_with_surface(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        VulkanDevice::new_with_swapchain(Arc::clone(&self.shared), self.physical, req)
    }
}
