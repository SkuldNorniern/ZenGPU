//! Vulkan entry point and instance.

use std::sync::Arc;

use ash::{Entry, Instance, vk};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuError, GpuInstance,
    PowerPreference,
};

use crate::adapter::VulkanAdapter;

/// Shared ownership of the Vulkan loader and `VkInstance`.
pub(crate) struct VulkanShared {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_extensions: bool,
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
        #[cfg(target_os = "macos")]
        let mut flags = vk::InstanceCreateFlags::empty();
        #[cfg(not(target_os = "macos"))]
        let flags = vk::InstanceCreateFlags::empty();
        if surface_extensions {
            let available = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| GpuError::Backend(format!("enumerate instance extensions: {e}")))?;
            let supports = |name: &std::ffi::CStr| {
                available.iter().any(|extension| unsafe {
                    std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) == name
                })
            };
            let mut require = |name: &'static std::ffi::CStr| -> zengpu_hal::Result<()> {
                if !supports(name) {
                    return Err(GpuError::Backend(format!(
                        "required Vulkan instance extension is unavailable: {}",
                        name.to_string_lossy()
                    )));
                }
                ext_names.push(name.as_ptr());
                Ok(())
            };

            require(ash::khr::surface::NAME)?;
            #[cfg(target_os = "windows")]
            require(ash::khr::win32_surface::NAME)?;
            #[cfg(target_os = "linux")]
            {
                if supports(ash::khr::xcb_surface::NAME) {
                    ext_names.push(ash::khr::xcb_surface::NAME.as_ptr());
                }
                if supports(ash::khr::wayland_surface::NAME) {
                    ext_names.push(ash::khr::wayland_surface::NAME.as_ptr());
                }
                if !supports(ash::khr::xcb_surface::NAME)
                    && !supports(ash::khr::wayland_surface::NAME)
                {
                    return Err(GpuError::Backend(
                        "Vulkan loader supports neither XCB nor Wayland surfaces".to_string(),
                    ));
                }
            }
            #[cfg(target_os = "macos")]
            {
                require(ash::mvk::macos_surface::NAME)?;
                if supports(ash::khr::portability_enumeration::NAME) {
                    ext_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
                    flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
                }
            }
        }

        let create_info = vk::InstanceCreateInfo {
            flags,
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
            shared: Arc::new(VulkanShared {
                entry,
                instance,
                surface_extensions,
            }),
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
            self.shared
                .instance
                .enumerate_physical_devices()
                .unwrap_or_default()
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
                let props = unsafe { self.shared.instance.get_physical_device_properties(phys) };
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
}
