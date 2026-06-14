//! Vulkan entry point and instance (plan §22 / D15).

use std::sync::{Arc, Mutex};

use ash::{Entry, Instance, vk};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuSurface, PowerPreference, SurfaceConfig, SurfaceFrame, WindowHandles,
};

use crate::adapter::VulkanAdapter;
use crate::swapchain::{BeginFrame, DeviceContext, Swapchain};
use crate::swapchain_2d::Vulkan2dSurface;

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

    /// Create a surface that paints batches of instanced solid-colour
    /// rectangles (aurea's 2D path).  Call
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
        Ok(Box::new(ClearSurface::new(vk_dev, handles, config)?))
    }
}

// ── ClearSurface ─────────────────────────────────────────────────────────────
//
// Minimal GpuSurface backed by Swapchain: does a solid-colour clear each frame,
// no pipeline. Used as the GpuInstance::create_surface implementation so that
// code depending on the HAL trait still works. Render-pass-shaped work (pipelines,
// vertex buffers, descriptor sets) belongs on the caller side — see examples/.

struct ClearSurface {
    ctx: DeviceContext,
    render_pass: vk::RenderPass,
    framebuffers: Mutex<Vec<vk::Framebuffer>>,
    pending: Mutex<Option<BeginFrame>>,
    sc: Swapchain, // LAST: drops after render_pass/framebuffers are destroyed
}

unsafe impl Send for ClearSurface {}
unsafe impl Sync for ClearSurface {}

impl ClearSurface {
    fn new(
        device: &crate::device::VulkanDevice,
        handles: &WindowHandles,
        config: SurfaceConfig,
    ) -> zengpu_hal::Result<Self> {
        let sc = Swapchain::new(device, handles, config, 2)?;
        let ctx = sc.context();
        let render_pass = cs_render_pass(ctx.device(), sc.format())?;
        let fbs = cs_framebuffers(ctx.device(), render_pass, &sc.image_views(), sc.extent())?;
        Ok(Self {
            ctx,
            render_pass,
            framebuffers: Mutex::new(fbs),
            pending: Mutex::new(None),
            sc,
        })
    }

    fn rebuild_framebuffers(&self) -> zengpu_hal::Result<()> {
        let new_fbs =
            cs_framebuffers(self.ctx.device(), self.render_pass, &self.sc.image_views(), self.sc.extent())?;
        let mut fbs = self.framebuffers.lock().unwrap();
        for &fb in fbs.iter() {
            unsafe { self.ctx.device().destroy_framebuffer(fb, None); }
        }
        *fbs = new_fbs;
        Ok(())
    }
}

impl Drop for ClearSurface {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let fbs = self.framebuffers.lock().unwrap();
            for &fb in fbs.iter() {
                self.ctx.device().destroy_framebuffer(fb, None);
            }
            drop(fbs);
            self.ctx.device().destroy_render_pass(self.render_pass, None);
        }
        // self.sc drops here — swapchain/image_views/surface/sync destroyed last
    }
}

impl GpuSurface for ClearSurface {
    fn configure(&self, config: SurfaceConfig) -> zengpu_hal::Result<()> {
        self.sc.resize(config.width, config.height)?;
        self.rebuild_framebuffers()
    }

    fn acquire_frame(&self) -> zengpu_hal::Result<SurfaceFrame> {
        loop {
            match self.sc.begin_frame()? {
                BeginFrame::Skip => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                BeginFrame::Recreated => {
                    self.rebuild_framebuffers()?;
                }
                bf @ BeginFrame::Image { index, .. } => {
                    *self.pending.lock().unwrap() = Some(bf);
                    return Ok(SurfaceFrame { index });
                }
            }
        }
    }

    fn present_frame(&self, _frame: SurfaceFrame) -> zengpu_hal::Result<()> {
        let bf = match self.pending.lock().unwrap().take() {
            Some(bf) => bf,
            None => return Ok(()),
        };
        let (current, index) = match bf {
            BeginFrame::Image { current, index } => (current, index),
            _ => return Ok(()),
        };
        let bf = BeginFrame::Image { current, index };
        let cmd = self.sc.cmd_buffer(current);
        let dev = self.ctx.device();
        let fbs = self.framebuffers.lock().unwrap();
        let extent = self.sc.extent();
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| GpuError::Backend(format!("reset_command_buffer: {e}")))?;
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| GpuError::Backend(format!("begin_command_buffer: {e}")))?;
            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.02, 0.02, 0.02, 1.0] },
            };
            dev.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: fbs[index as usize],
                    render_area: vk::Rect2D { offset: vk::Offset2D::default(), extent },
                    clear_value_count: 1,
                    p_clear_values: &clear,
                    ..Default::default()
                },
                vk::SubpassContents::INLINE,
            );
            dev.cmd_end_render_pass(cmd);
            dev.end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
        drop(fbs);
        if self.sc.end_frame(&bf, cmd)? {
            self.rebuild_framebuffers()?;
        }
        Ok(())
    }

    fn size(&self) -> (u32, u32) {
        let e = self.sc.extent();
        (e.width, e.height)
    }

    fn image_count(&self) -> u32 {
        self.sc.image_count() as u32
    }
}

fn cs_render_pass(device: &ash::Device, format: vk::Format) -> zengpu_hal::Result<vk::RenderPass> {
    let attachment = vk::AttachmentDescription {
        format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };
    let color_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_ref,
        ..Default::default()
    };
    let dependency = vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ..Default::default()
    };
    unsafe {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo {
                    attachment_count: 1,
                    p_attachments: &attachment,
                    subpass_count: 1,
                    p_subpasses: &subpass,
                    dependency_count: 1,
                    p_dependencies: &dependency,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_render_pass: {e}")))
    }
}

fn cs_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> zengpu_hal::Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&view| {
            let attachments = [view];
            unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo {
                            render_pass,
                            attachment_count: 1,
                            p_attachments: attachments.as_ptr(),
                            width: extent.width,
                            height: extent.height,
                            layers: 1,
                            ..Default::default()
                        },
                        None,
                    )
                    .map_err(|e| GpuError::Backend(format!("create_framebuffer: {e}")))
            }
        })
        .collect()
}
