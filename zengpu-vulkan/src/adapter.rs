use std::sync::Arc;

use ash::ext::pci_bus_info;
use ash::vk;
use zengpu_hal::{
    AdapterInfo, DeviceLimits, DeviceRequest, Features, GpuAdapter, GpuDevice, HalCapabilities,
};

use crate::device::{VulkanDevice, physical_device_limits, queue_family};
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
        HalCapabilities::all().with_features(Features::DESCRIPTOR_INDEXING)
    }

    fn limits(&self) -> DeviceLimits {
        queue_family(&self.shared.instance, self.physical, false)
            .map(|family| physical_device_limits(&self.shared.instance, self.physical, family))
            .unwrap_or_default()
    }

    /// Opens a compute-only device (no swapchain extension).
    fn open(&self, req: DeviceRequest) -> zengpu_hal::Result<Box<dyn GpuDevice>> {
        let device = VulkanDevice::new(Arc::clone(&self.shared), self.physical, req)?;
        Ok(Box::new(device))
    }
}

impl VulkanAdapter {
    /// Device UUID, stable for the same GPU and driver across runs. Save it to reopen this adapter
    /// after a device loss.
    pub fn uuid(&self) -> [u8; 16] {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe {
            self.shared
                .instance
                .get_physical_device_properties2(self.physical, &mut props)
        };
        id.device_uuid
    }

    /// PCI address as `domain:bus:device.function` in lowercase hex, e.g. `0000:0b:00.0`, the
    /// same text `hipDeviceGetPCIBusId` gives. `None` without `VK_EXT_pci_bus_info`.
    pub fn pci_bus_id(&self) -> Option<String> {
        let instance = &self.shared.instance;
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(self.physical) }.ok()?;
        let supported = extensions.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|n| n == pci_bus_info::NAME)
        });
        if !supported {
            return None;
        }
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
        unsafe { instance.get_physical_device_properties2(self.physical, &mut props) };
        Some(format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function
        ))
    }

    /// Driver name and version text, e.g. `("radv", "Mesa 25.2.8")`. `None` before Vulkan 1.2.
    pub fn driver(&self) -> Option<(String, String)> {
        let instance = &self.shared.instance;
        let base = unsafe { instance.get_physical_device_properties(self.physical) };
        if base.api_version < vk::API_VERSION_1_2 {
            return None;
        }
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        unsafe { instance.get_physical_device_properties2(self.physical, &mut props) };
        let name = driver
            .driver_name_as_c_str()
            .ok()?
            .to_string_lossy()
            .into_owned();
        let info = driver
            .driver_info_as_c_str()
            .ok()?
            .to_string_lossy()
            .into_owned();
        Some((name, info))
    }

    /// Opens a headless graphics device with no swapchain extension. For windowed rendering,
    /// use `open_with_surface`.
    pub fn open_headless(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        VulkanDevice::new_headless_graphics(Arc::clone(&self.shared), self.physical, req)
    }

    /// Opens a device with the swapchain extension, required for presenting
    /// through a [`crate::Swapchain`].
    pub fn open_with_surface(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        VulkanDevice::new_with_swapchain(Arc::clone(&self.shared), self.physical, req)
    }

    /// Opens a device that can present when the instance has surface support, else a headless
    /// one. Same name on every backend (see `zengpu::native`).
    pub fn open_graphics(&self, req: DeviceRequest) -> zengpu_hal::Result<VulkanDevice> {
        if self.shared.surface_extensions {
            self.open_with_surface(req)
        } else {
            self.open_headless(req)
        }
    }
}
