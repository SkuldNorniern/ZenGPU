//! Vulkan entry point and instance (plan §22 / D15).

use std::sync::Arc;

use ash::{Entry, Instance, vk};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuSurface, PowerPreference, SurfaceConfig, WindowHandles,
};

use crate::adapter::VulkanAdapter;
use crate::swapchain::VulkanSwapchain;

/// Shared ownership of the Vulkan loader and `VkInstance`.
pub(crate) struct VulkanShared {
    pub entry: Entry,
    pub instance: Instance,
}

unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// Vulkan [`GpuInstance`].
pub struct VulkanInstance {
    pub(crate) shared: Arc<VulkanShared>,
    pub(crate) has_surface: bool,
}

impl VulkanInstance {
    fn create(surface_extensions: bool) -> zengpu_hal::Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| GpuError::Backend(format!("Vulkan loader: {e}")))?;

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let mut ext_names: Vec<*const i8> = Vec::new();
        if surface_extensions {
            ext_names.push(ash::khr::surface::NAME.as_ptr());
            #[cfg(target_os = "windows")]
            ext_names.push(ash::khr::win32_surface::NAME.as_ptr());
            #[cfg(target_os = "linux")]
            ext_names.push(ash::khr::xlib_surface::NAME.as_ptr());
            #[cfg(target_os = "macos")]
            ext_names.push(ash::mvk::macos_surface::NAME.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: ext_names.len() as u32,
            pp_enabled_extension_names: if ext_names.is_empty() {
                std::ptr::null()
            } else {
                ext_names.as_ptr()
            },
            ..Default::default()
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateInstance: {e}")))?
        };

        Ok(Self {
            shared: Arc::new(VulkanShared { entry, instance }),
            has_surface: surface_extensions,
        })
    }

    /// Compute-only instance (no surface/display extensions).
    pub fn new() -> zengpu_hal::Result<Self> {
        Self::create(false)
    }

    /// Instance with surface extensions enabled — required for presenting to windows.
    pub fn new_with_surface() -> zengpu_hal::Result<Self> {
        Self::create(true)
    }

    /// Return the first available adapter as a concrete [`VulkanAdapter`], or
    /// `None` if no Vulkan physical device is found.
    ///
    /// Use this instead of [`GpuInstance::enumerate_adapters`] when you need
    /// to call [`VulkanAdapter::open_with_surface`], which is not part of the
    /// trait object API.
    pub fn request_vulkan_adapter(&self) -> Option<VulkanAdapter> {
        let physicals = unsafe {
            self.shared.instance.enumerate_physical_devices().unwrap_or_default()
        };
        physicals.into_iter().next().map(|phys| {
            let props = unsafe { self.shared.instance.get_physical_device_properties(phys) };
            let name = unsafe {
                std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            VulkanAdapter::new(
                Arc::clone(&self.shared),
                phys,
                AdapterInfo {
                    name,
                    vendor: props.vendor_id,
                    device: props.device_id,
                    device_type: vk_device_type(props.device_type),
                    backend: BackendPreference::Vulkan,
                },
            )
        })
    }
}

fn vk_device_type(t: vk::PhysicalDeviceType) -> DeviceType {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
        vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::Virtual,
        _ => DeviceType::Unknown,
    }
}

fn type_score(t: DeviceType, pref: PowerPreference) -> u32 {
    match (t, pref) {
        (DeviceType::Discrete, PowerPreference::HighPerformance) => 3,
        (DeviceType::Integrated, PowerPreference::LowPower) => 3,
        (DeviceType::Discrete, PowerPreference::LowPower) => 2,
        (DeviceType::Integrated, PowerPreference::HighPerformance) => 1,
        _ => 0,
    }
}

impl GpuInstance for VulkanInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        let physicals = unsafe {
            self.shared
                .instance
                .enumerate_physical_devices()
                .unwrap_or_default()
        };
        physicals
            .into_iter()
            .map(|phys| {
                let props = unsafe {
                    self.shared.instance.get_physical_device_properties(phys)
                };
                let name = unsafe {
                    std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                let info = AdapterInfo {
                    name,
                    vendor: props.vendor_id,
                    device: props.device_id,
                    device_type: vk_device_type(props.device_type),
                    backend: BackendPreference::Vulkan,
                };
                Box::new(VulkanAdapter::new(Arc::clone(&self.shared), phys, info))
                    as Box<dyn GpuAdapter>
            })
            .collect()
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        let mut adapters = self.enumerate_adapters();
        adapters.sort_by_key(|a| std::cmp::Reverse(type_score(a.info().device_type, req.power)));
        adapters.into_iter().next()
    }

    fn create_surface(
        &self,
        handles: &WindowHandles,
        device: &dyn GpuDevice,
        config: SurfaceConfig,
    ) -> zengpu_hal::Result<Box<dyn GpuSurface>> {
        if !self.has_surface {
            return Err(GpuError::Backend(
                "create_surface requires VulkanInstance::new_with_surface()".to_string(),
            ));
        }
        let vk_dev = device
            .as_any()
            .downcast_ref::<crate::device::VulkanDevice>()
            .ok_or_else(|| {
                GpuError::Backend("create_surface requires a VulkanDevice".to_string())
            })?;
        let swapchain =
            VulkanSwapchain::new(Arc::clone(&self.shared), vk_dev, handles, config)?;
        Ok(Box::new(swapchain))
    }
}
