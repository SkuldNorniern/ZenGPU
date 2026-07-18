//! Vulkan physical-device adapter.

use std::sync::Arc;

use ash::vk;
use zengpu_hal::{
    AdapterInfo, DeviceLimits, DeviceRequest, Features, GpuAdapter, GpuDevice, HalCapabilities,
};

use crate::device::{VulkanDevice, physical_device_limits, queue_family};
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
        HalCapabilities::all().with_features(Features::DESCRIPTOR_INDEXING)
    }

    fn limits(&self) -> DeviceLimits {
        queue_family(&self.shared.instance, self.physical, false)
            .map(|family| physical_device_limits(&self.shared.instance, self.physical, family))
            .unwrap_or_default()
    }

    /// Open a compute-only device (no swapchain extension).
    fn open(&self, req: DeviceRequest) -> zengpu_hal::Result<Box<dyn GpuDevice>> {
        let device = VulkanDevice::new(Arc::clone(&self.shared), self.physical, req)?;
        Ok(Box::new(device))
    }
}

impl VulkanAdapter {
    /// Open a headless device (no swapchain extension) and return the concrete
    /// [`VulkanDevice`]. For windowed rendering use [`open_with_surface`].
    ///
    /// [`open_with_surface`]: VulkanAdapter::open_with_surface
    pub fn open_headless(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        VulkanDevice::new_headless_graphics(Arc::clone(&self.shared), self.physical, req)
    }

    /// Open a device with the swapchain extension enabled (required for
    /// presenting through a [`crate::Swapchain`]).
    pub fn open_with_surface(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        VulkanDevice::new_with_swapchain(Arc::clone(&self.shared), self.physical, req)
    }
}
