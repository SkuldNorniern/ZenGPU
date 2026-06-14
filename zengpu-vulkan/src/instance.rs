//! Vulkan entry point and instance (plan §22 / D15).

use std::sync::Arc;

use ash::{Entry, Instance, vk};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuSurface, PowerPreference, SamplerHandle, SurfaceConfig, TextureHandle,
    WindowHandles,
};

use crate::adapter::VulkanAdapter;
use crate::swapchain_triangle::VulkanSwapchain;
use crate::swapchain_2d::Vulkan2dSurface;
use crate::swapchain_textured::VulkanTexturedSwapchain;

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

    /// Create a surface that renders a fullscreen quad sampling from a bindless
    /// texture array.  `texture` and `sampler` are registered at slot 0; all
    /// other slots are pre-filled with a 1×1 white placeholder.
    ///
    /// The underlying texture and sampler must remain alive for the lifetime of
    /// the returned surface.
    pub fn create_textured_surface(
        &self,
        handles: &WindowHandles,
        device: &crate::device::VulkanDevice,
        config: SurfaceConfig,
        texture: TextureHandle,
        sampler: SamplerHandle,
    ) -> zengpu_hal::Result<Box<dyn GpuSurface>> {
        if !self.has_surface {
            return Err(GpuError::Backend(
                "create_textured_surface requires VulkanInstance::new_with_surface()".to_string(),
            ));
        }
        let sc =
            VulkanTexturedSwapchain::new(device, handles, config, texture, sampler)?;
        Ok(Box::new(sc))
    }

    /// Create a surface that paints batches of instanced solid-colour
    /// rectangles (aurea's 2D path, G4 / Rung 1).  Call
    /// [`Vulkan2dSurface::present`] each frame with the clear colour and rects.
    pub fn create_2d_surface(
        &self,
        handles: &WindowHandles,
        device: &crate::device::VulkanDevice,
        config: SurfaceConfig,
    ) -> zengpu_hal::Result<Vulkan2dSurface> {
        if !self.has_surface {
            return Err(GpuError::Backend(
                "create_2d_surface requires VulkanInstance::new_with_surface()".to_string(),
            ));
        }
        Vulkan2dSurface::new(device, handles, config)
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
