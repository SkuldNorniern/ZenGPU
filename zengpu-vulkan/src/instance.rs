//! Vulkan entry point and instance.

use std::sync::Arc;

use ash::{Entry, Instance, ext, vk};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuError, GpuInstance,
    PowerPreference,
};

use crate::adapter::VulkanAdapter;

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { std::ffi::CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {msg}");
    } else {
        log::warn!("[Vulkan] {msg}");
    }
    vk::FALSE
}

/// Shared ownership of the Vulkan loader and `VkInstance`.
pub(crate) struct VulkanShared {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_extensions: bool,
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe {
            if let Some((du, messenger)) = self.debug_utils.take() {
                du.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
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

        let available_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();

        let has_ext = |name: &std::ffi::CStr| {
            available_exts
                .iter()
                .any(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == name })
        };
        let has_layer = |name: &[u8]| {
            available_layers.iter().any(|l| {
                let s = unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) };
                s.to_bytes() == name
            })
        };

        let want_validation =
            has_layer(b"VK_LAYER_KHRONOS_validation") && has_ext(ext::debug_utils::NAME);

        let mut ext_names: Vec<*const i8> = Vec::new();
        let mut layer_names: Vec<*const i8> = Vec::new();

        #[cfg(target_os = "macos")]
        let mut flags = vk::InstanceCreateFlags::empty();
        #[cfg(not(target_os = "macos"))]
        let flags = vk::InstanceCreateFlags::empty();

        if want_validation {
            log::debug!("[zengpu-vulkan] enabling VK_LAYER_KHRONOS_validation");
            layer_names.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            ext_names.push(ext::debug_utils::NAME.as_ptr());
        }

        if surface_extensions {
            let require = |name: &'static std::ffi::CStr| -> zengpu_hal::Result<*const i8> {
                if !has_ext(name) {
                    return Err(GpuError::Backend(format!(
                        "required Vulkan instance extension is unavailable: {}",
                        name.to_string_lossy()
                    )));
                }
                Ok(name.as_ptr())
            };

            ext_names.push(require(ash::khr::surface::NAME)?);
            #[cfg(target_os = "windows")]
            ext_names.push(require(ash::khr::win32_surface::NAME)?);
            #[cfg(target_os = "linux")]
            {
                if has_ext(ash::khr::xcb_surface::NAME) {
                    ext_names.push(ash::khr::xcb_surface::NAME.as_ptr());
                }
                if has_ext(ash::khr::wayland_surface::NAME) {
                    ext_names.push(ash::khr::wayland_surface::NAME.as_ptr());
                }
                if !has_ext(ash::khr::xcb_surface::NAME)
                    && !has_ext(ash::khr::wayland_surface::NAME)
                {
                    return Err(GpuError::Backend(
                        "Vulkan loader supports neither XCB nor Wayland surfaces".to_string(),
                    ));
                }
            }
            #[cfg(target_os = "macos")]
            {
                ext_names.push(require(ash::mvk::macos_surface::NAME)?);
                if has_ext(ash::khr::portability_enumeration::NAME) {
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
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: if layer_names.is_empty() {
                std::ptr::null()
            } else {
                layer_names.as_ptr()
            },
            ..Default::default()
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateInstance: {e}")))?
        };

        let debug_utils = if want_validation {
            let du = ext::debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };
            match unsafe { du.create_debug_utils_messenger(&messenger_info, None) } {
                Ok(m) => Some((du, m)),
                Err(e) => {
                    log::warn!("[zengpu-vulkan] debug messenger failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            shared: Arc::new(VulkanShared {
                entry,
                instance,
                surface_extensions,
                debug_utils,
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
    /// Return the best available adapter for headless rendering: prefers
    /// discrete GPUs over integrated ones, uses `type_score` to rank.
    pub fn request_vulkan_adapter(&self) -> Option<VulkanAdapter> {
        self.request_vulkan_adapter_with_pref(PowerPreference::HighPerformance)
    }

    /// Return the best adapter ranked by `pref`. Discrete beats integrated for
    /// `HighPerformance`; integrated beats discrete for `LowPower`.
    pub fn request_vulkan_adapter_with_pref(&self, pref: PowerPreference) -> Option<VulkanAdapter> {
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
                let dt = vk_device_type(props.device_type);
                let score = type_score(dt, pref);
                let name = unsafe {
                    std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                (score, phys, props, name, dt)
            })
            .max_by_key(|(score, _, _, _, _)| *score)
            .map(|(_, phys, props, name, dt)| {
                log::info!("[zengpu-vulkan] selected adapter: {name} ({dt:?})");
                VulkanAdapter::new(
                    Arc::clone(&self.shared),
                    phys,
                    AdapterInfo {
                        name,
                        vendor: props.vendor_id,
                        device: props.device_id,
                        device_type: dt,
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
