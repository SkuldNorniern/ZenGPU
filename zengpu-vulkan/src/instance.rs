use std::{
    cmp::Reverse,
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::Arc,
};

use ash::{Entry, Instance, ext, khr, vk};
#[cfg(target_os = "macos")]
use std::env::current_exe;
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuError, GpuInstance,
    PowerPreference,
};

use crate::adapter::VulkanAdapter;

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    #[cfg(test)]
    eprintln!("[Vulkan validation] {msg}");
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {msg}");
    } else {
        log::warn!("[Vulkan] {msg}");
    }
    vk::FALSE
}

pub(crate) struct VulkanShared {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_extensions: bool,
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanShared {
    pub(crate) fn debug_utils_enabled(&self) -> bool {
        self.debug_utils.is_some()
    }
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

pub struct VulkanInstance {
    pub(crate) shared: Arc<VulkanShared>,
}

/// Load the Vulkan loader. On macOS, a binary run outside `cargo run` misses Homebrew, `/usr/local`
/// and app-bundle Frameworks, so those are tried by full path after the default lookup.
fn load_entry() -> zengpu_hal::Result<Entry> {
    // SAFETY: loading the Vulkan loader library runs no Vulkan calls.
    let first = unsafe { Entry::load() };
    #[cfg(target_os = "macos")]
    let first = first.or_else(|e| {
        let mut paths: Vec<PathBuf> = Vec::new();
        if let Some(dir) = current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        {
            paths.push(dir.join("../Frameworks/libvulkan.1.dylib"));
            paths.push(dir.join("libvulkan.1.dylib"));
        }
        for dir in ["/opt/homebrew/lib", "/usr/local/lib"] {
            paths.push(Path::new(dir).join("libvulkan.1.dylib"));
        }
        paths
            .iter()
            .filter(|p| p.exists())
            // SAFETY: as above, for the same library found by path.
            .find_map(|p| unsafe { Entry::load_from(p) }.ok())
            .ok_or(e)
    });
    first.map_err(|e| GpuError::Backend(format!("Vulkan loader: {e}")))
}

impl VulkanInstance {
    fn create(surface_extensions: bool) -> zengpu_hal::Result<Self> {
        let entry = load_entry()?;

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let available_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();

        let has_ext = |name: &CStr| {
            available_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == name })
        };
        let has_layer = |name: &[u8]| {
            available_layers.iter().any(|l| {
                let s = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                s.to_bytes() == name
            })
        };

        let want_validation =
            has_layer(b"VK_LAYER_KHRONOS_validation") && has_ext(ext::debug_utils::NAME);

        let mut ext_names: Vec<*const c_char> = Vec::new();
        let mut layer_names: Vec<*const c_char> = Vec::new();

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
            let require = |name: &'static CStr| -> zengpu_hal::Result<*const c_char> {
                if !has_ext(name) {
                    return Err(GpuError::Backend(format!(
                        "required Vulkan instance extension is unavailable: {}",
                        name.to_string_lossy()
                    )));
                }
                Ok(name.as_ptr())
            };

            ext_names.push(require(khr::surface::NAME)?);
            #[cfg(target_os = "windows")]
            ext_names.push(require(khr::win32_surface::NAME)?);
            #[cfg(target_os = "linux")]
            {
                if has_ext(khr::xcb_surface::NAME) {
                    ext_names.push(khr::xcb_surface::NAME.as_ptr());
                }
                if has_ext(khr::wayland_surface::NAME) {
                    ext_names.push(khr::wayland_surface::NAME.as_ptr());
                }
                if !has_ext(khr::xcb_surface::NAME) && !has_ext(khr::wayland_surface::NAME) {
                    return Err(GpuError::Backend(
                        "Vulkan loader supports neither XCB nor Wayland surfaces".to_string(),
                    ));
                }
            }
            #[cfg(target_os = "macos")]
            ext_names.push(require(ext::metal_surface::NAME)?);
            #[cfg(target_os = "android")]
            ext_names.push(require(khr::android_surface::NAME)?);
        }

        #[cfg(target_os = "macos")]
        if has_ext(khr::portability_enumeration::NAME) {
            ext_names.push(khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let create_info = vk::InstanceCreateInfo {
            flags,
            p_application_info: &app_info,
            enabled_extension_count: ext_names.len() as u32,
            pp_enabled_extension_names: if ext_names.is_empty() {
                ptr::null()
            } else {
                ext_names.as_ptr()
            },
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: if layer_names.is_empty() {
                ptr::null()
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

    /// `true` if a Vulkan loader is present, without creating an instance. For tests and CI that
    /// skip GPU work when there is no driver.
    pub fn is_available() -> bool {
        // SAFETY: `Entry::load` only loads a shared library; no Vulkan objects are allocated.
        load_entry().is_ok()
    }

    /// Compute-only instance (no surface/display extensions).
    pub fn new() -> zengpu_hal::Result<Self> {
        Self::create(false)
    }

    /// Instance with surface extensions, required for presenting to windows.
    pub fn new_with_surface() -> zengpu_hal::Result<Self> {
        Self::create(true)
    }

    /// First adapter as a concrete `VulkanAdapter`, or `None`. Use it, not `enumerate_adapters`, to
    /// call `open_with_surface`. Every adapter, as the concrete type.
    pub fn vulkan_adapters(&self) -> Vec<VulkanAdapter> {
        let physicals = unsafe {
            self.shared
                .instance
                .enumerate_physical_devices()
                .unwrap_or_default()
        };
        physicals
            .into_iter()
            .map(|phys| VulkanAdapter::new(Arc::clone(&self.shared), phys, self.adapter_info(phys)))
            .collect()
    }

    /// Adapter with this `VulkanAdapter::uuid`, if still present.
    pub fn graphics_adapters(&self) -> Vec<VulkanAdapter> {
        self.vulkan_adapters()
    }

    pub fn request_graphics_adapter(&self) -> Option<VulkanAdapter> {
        self.request_vulkan_adapter()
    }

    pub fn graphics_adapter_by_uuid(&self, uuid: [u8; 16]) -> Option<VulkanAdapter> {
        self.vulkan_adapter_by_uuid(uuid)
    }

    pub fn vulkan_adapter_by_uuid(&self, uuid: [u8; 16]) -> Option<VulkanAdapter> {
        self.vulkan_adapters()
            .into_iter()
            .find(|a| a.uuid() == uuid)
    }

    fn adapter_info(&self, phys: vk::PhysicalDevice) -> AdapterInfo {
        let props = unsafe { self.shared.instance.get_physical_device_properties(phys) };
        let name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        AdapterInfo {
            name,
            vendor: props.vendor_id,
            device: props.device_id,
            device_type: vk_device_type(props.device_type),
            backend: BackendPreference::Vulkan,
        }
    }

    /// Best adapter for headless rendering: discrete over integrated, by `type_score`.
    pub fn request_vulkan_adapter(&self) -> Option<VulkanAdapter> {
        self.request_vulkan_adapter_with_pref(PowerPreference::HighPerformance)
    }

    /// Best adapter by `pref`: discrete wins for `HighPerformance`, integrated for `LowPower`.
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
                    CStr::from_ptr(props.device_name.as_ptr())
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
                    CStr::from_ptr(props.device_name.as_ptr())
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
        adapters.sort_by_key(|a| Reverse(type_score(a.info().device_type, req.power)));
        adapters.into_iter().next()
    }
}
