//! Vulkan swapchain, render pass, graphics pipeline, and present (plan G2).
//!
//! G2 scope: a fullscreen triangle drawn from `gl_VertexIndex` (no vertex
//! buffer).  The command buffers are pre-recorded at swapchain creation; the
//! render loop is acquire → (pre-recorded draw) → present.

use std::sync::{Arc, Mutex};

use ash::{khr, vk};
use inline_spirv::inline_spirv;
use zengpu_hal::{GpuError, GpuSurface, PresentMode, Result, SurfaceConfig, SurfaceFrame};

use crate::device::VulkanDeviceInner;
use crate::instance::VulkanShared;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// ── Compiled shaders ──────────────────────────────────────────────────────────

const VERT_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    void main() {
        vec2 pos[3] = vec2[](
            vec2(-0.5,  0.5),
            vec2( 0.5,  0.5),
            vec2( 0.0, -0.5)
        );
        gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    }
    "#,
    vert,
    vulkan1_0
);

const FRAG_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) out vec4 o_color;
    void main() {
        o_color = vec4(0.1, 0.6, 1.0, 1.0);
    }
    "#,
    frag,
    vulkan1_0
);

// ── Frame sync state ──────────────────────────────────────────────────────────

struct FrameState {
    current: usize,
}

// ── VulkanSwapchain ───────────────────────────────────────────────────────────

/// Vulkan swapchain + graphics pipeline + pre-recorded triangle command buffers.
/// Implements [`GpuSurface`] so it can be held as `Box<dyn GpuSurface>`.
pub struct VulkanSwapchain {
    inner: Arc<VulkanDeviceInner>,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame_state: Mutex<FrameState>,
    extent: Mutex<vk::Extent2D>,
    // Kept for swapchain recreation on resize (deferred to post-G2).
    #[allow(dead_code)]
    format: vk::Format,
}

// Safety: all mutable state protected by Mutex; ash types are Send + Sync.
unsafe impl Send for VulkanSwapchain {}
unsafe impl Sync for VulkanSwapchain {}

impl VulkanSwapchain {
    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        device: &crate::device::VulkanDevice,
        handles: &zengpu_hal::WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self> {
        let inner = Arc::clone(&device.inner);
        let surface_loader =
            khr::surface::Instance::new(&shared.entry, &shared.instance);

        let surface = create_platform_surface(&shared, handles)?;

        // Verify queue family can present to this surface.
        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                inner.physical,
                inner.queue_family,
                surface,
            )
        }
        .map_err(|e| GpuError::Backend(format!("surface support query: {e}")))?;

        if !supports_present {
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(GpuError::Backend(
                "selected queue family cannot present to this surface".to_string(),
            ));
        }

        let swapchain_loader = khr::swapchain::Device::new(&shared.instance, &inner.device);

        let (swapchain, images, format, extent) = create_swapchain(
            &surface_loader,
            &swapchain_loader,
            inner.physical,
            surface,
            &config,
            vk::SwapchainKHR::null(),
        )?;

        let image_views = create_image_views(&inner.device, &images, format)?;
        let render_pass = create_render_pass(&inner.device, format)?;
        let framebuffers = create_framebuffers(&inner.device, render_pass, &image_views, extent)?;
        let (pipeline_layout, pipeline) =
            create_pipeline(&inner.device, render_pass, extent)?;

        let cmd_pool = create_command_pool(&inner.device, inner.queue_family)?;
        let cmd_buffers = allocate_and_record_cmd_buffers(
            &inner.device,
            cmd_pool,
            render_pass,
            &framebuffers,
            pipeline,
            extent,
        )?;

        let (image_available, render_finished, in_flight) =
            create_sync(&inner.device, MAX_FRAMES_IN_FLIGHT)?;

        Ok(Self {
            inner,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            images,
            image_views,
            render_pass,
            framebuffers,
            pipeline_layout,
            pipeline,
            cmd_pool,
            cmd_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_state: Mutex::new(FrameState { current: 0 }),
            extent: Mutex::new(extent),
            format,
        })
    }

    fn destroy_swapchain_resources(&self) {
        unsafe {
            let dev = &self.inner.device;
            dev.free_command_buffers(self.cmd_pool, &self.cmd_buffers);
            for &fb in &self.framebuffers {
                dev.destroy_framebuffer(fb, None);
            }
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_render_pass(self.render_pass, None);
            for &iv in &self.image_views {
                dev.destroy_image_view(iv, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        self.destroy_swapchain_resources();
        unsafe {
            let dev = &self.inner.device;
            dev.destroy_command_pool(self.cmd_pool, None);
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                dev.destroy_semaphore(self.image_available[i], None);
                dev.destroy_semaphore(self.render_finished[i], None);
                dev.destroy_fence(self.in_flight[i], None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl GpuSurface for VulkanSwapchain {
    fn configure(&self, _config: SurfaceConfig) -> Result<()> {
        // Swapchain recreation on resize — deferred to post-G2.
        Ok(())
    }

    fn acquire_frame(&self) -> Result<SurfaceFrame> {
        let mut state = self.frame_state.lock().unwrap();
        let current = state.current;

        unsafe {
            self.inner
                .device
                .wait_for_fences(&[self.in_flight[current]], true, u64::MAX)
                .map_err(|e| GpuError::Backend(format!("wait_for_fences: {e}")))?;
            self.inner
                .device
                .reset_fences(&[self.in_flight[current]])
                .map_err(|e| GpuError::Backend(format!("reset_fences: {e}")))?;
        }

        let (image_index, _suboptimal) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available[current],
                    vk::Fence::null(),
                )
                .map_err(|e| GpuError::Surface(match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => zengpu_hal::SurfaceError::Outdated,
                    vk::Result::ERROR_SURFACE_LOST_KHR => zengpu_hal::SurfaceError::Lost,
                    _ => zengpu_hal::SurfaceError::OutOfMemory,
                }))?
        };

        state.current = current; // present_frame will advance it
        Ok(SurfaceFrame { index: image_index })
    }

    fn present_frame(&self, frame: SurfaceFrame) -> Result<()> {
        let mut state = self.frame_state.lock().unwrap();
        let current = state.current;
        let image_index = frame.index;

        let wait_semaphores = [self.image_available[current]];
        let signal_semaphores = [self.render_finished[current]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [self.cmd_buffers[image_index as usize]];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: cmd_bufs.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &[submit_info], self.in_flight[current])
                .map_err(|e| GpuError::Backend(format!("queue_submit: {e}")))?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.swapchain_loader
                .queue_present(self.inner.queue, &present_info)
                .map_err(|e| GpuError::Surface(match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => zengpu_hal::SurfaceError::Outdated,
                    vk::Result::ERROR_SURFACE_LOST_KHR => zengpu_hal::SurfaceError::Lost,
                    _ => zengpu_hal::SurfaceError::OutOfMemory,
                }))?;
        }

        state.current = (current + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn size(&self) -> (u32, u32) {
        let ext = *self.extent.lock().unwrap();
        (ext.width, ext.height)
    }

    fn image_count(&self) -> u32 {
        self.images.len() as u32
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use raw_window_handle::RawWindowHandle;

    let win32_loader = khr::win32_surface::Instance::new(&shared.entry, &shared.instance);

    let RawWindowHandle::Win32(win32) = handles.window else {
        return Err(GpuError::Backend(
            "expected Win32 window handle on Windows".to_string(),
        ));
    };

    let create_info = vk::Win32SurfaceCreateInfoKHR {
        hwnd: win32.hwnd.get() as vk::HANDLE,
        hinstance: win32
            .hinstance
            .map(|h| h.get() as vk::HINSTANCE)
            .unwrap_or(0isize as vk::HINSTANCE),
        ..Default::default()
    };

    unsafe {
        win32_loader
            .create_win32_surface(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateWin32SurfaceKHR: {e}")))
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

    match (handles.window, handles.display) {
        (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
            let connection = display.connection.ok_or_else(|| {
                GpuError::Backend("XCB display handle has no connection".to_string())
            })?;
            let loader = khr::xcb_surface::Instance::new(&shared.entry, &shared.instance);
            let create_info = vk::XcbSurfaceCreateInfoKHR {
                connection: connection.as_ptr(),
                window: window.window.get(),
                ..Default::default()
            };
            unsafe {
                loader
                    .create_xcb_surface(&create_info, None)
                    .map_err(|e| GpuError::Backend(format!("vkCreateXcbSurfaceKHR: {e}")))
            }
        }
        (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
            let loader = khr::wayland_surface::Instance::new(&shared.entry, &shared.instance);
            let create_info = vk::WaylandSurfaceCreateInfoKHR {
                display: display.display.as_ptr(),
                surface: window.surface.as_ptr(),
                ..Default::default()
            };
            unsafe {
                loader
                    .create_wayland_surface(&create_info, None)
                    .map_err(|e| GpuError::Backend(format!("vkCreateWaylandSurfaceKHR: {e}")))
            }
        }
        _ => Err(GpuError::Backend(
            "expected matching XCB or Wayland window/display handles on Linux".to_string(),
        )),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn create_platform_surface(
    shared: &VulkanShared,
    handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    use raw_window_handle::RawWindowHandle;

    let RawWindowHandle::AppKit(appkit) = handles.window else {
        return Err(GpuError::Backend(
            "expected AppKit window handle on macOS".to_string(),
        ));
    };
    let loader = ash::mvk::macos_surface::Instance::new(&shared.entry, &shared.instance);
    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        p_view: appkit.ns_view.as_ptr(),
        ..Default::default()
    };
    unsafe {
        loader
            .create_mac_os_surface(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateMacOSSurfaceMVK: {e}")))
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub(crate) fn create_platform_surface(
    _shared: &VulkanShared,
    _handles: &zengpu_hal::WindowHandles,
) -> Result<vk::SurfaceKHR> {
    Err(GpuError::Backend("unsupported surface platform".to_string()))
}

fn create_swapchain(
    surface_loader: &khr::surface::Instance,
    swapchain_loader: &khr::swapchain::Device,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    config: &SurfaceConfig,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
    let caps = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface capabilities: {e}")))?
    };

    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface formats: {e}")))?
    };

    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface present modes: {e}")))?
    };

    // Pick format: prefer BGRA8_SRGB; fall back to first available.
    let surface_format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .ok_or_else(|| GpuError::Backend("no surface formats".to_string()))?;

    // Pick present mode from config.
    let desired_mode = match config.present_mode {
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    let present_mode = if present_modes.contains(&desired_mode) {
        desired_mode
    } else {
        vk::PresentModeKHR::FIFO // always supported
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: config.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: config
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: image_count,
        image_format: surface_format.format,
        image_color_space: surface_format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        pre_transform: caps.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        old_swapchain,
        ..Default::default()
    };

    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .map_err(|e| GpuError::Backend(format!("vkCreateSwapchainKHR: {e}")))?
    };
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|e| GpuError::Backend(format!("get_swapchain_images: {e}")))?
    };

    Ok((swapchain, images, surface_format.format, extent))
}

fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            unsafe {
                device
                    .create_image_view(&info, None)
                    .map_err(|e| GpuError::Backend(format!("create_image_view: {e}")))
            }
        })
        .collect()
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
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

    let info = vk::RenderPassCreateInfo {
        attachment_count: 1,
        p_attachments: &attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 1,
        p_dependencies: &dependency,
        ..Default::default()
    };

    unsafe {
        device
            .create_render_pass(&info, None)
            .map_err(|e| GpuError::Backend(format!("create_render_pass: {e}")))
    }
}

fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&view| {
            let attachments = [view];
            let info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: attachments.as_ptr(),
                width: extent.width,
                height: extent.height,
                layers: 1,
                ..Default::default()
            };
            unsafe {
                device
                    .create_framebuffer(&info, None)
                    .map_err(|e| GpuError::Backend(format!("create_framebuffer: {e}")))
            }
        })
        .collect()
}

fn create_shader_module(device: &ash::Device, spv: &[u32]) -> Result<vk::ShaderModule> {
    let info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * 4,
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    unsafe {
        device
            .create_shader_module(&info, None)
            .map_err(|e| GpuError::Backend(format!("create_shader_module: {e}")))
    }
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert = create_shader_module(device, VERT_SPV)?;
    let frag = create_shader_module(device, FRAG_SPV)?;

    let entry = std::ffi::CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent,
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,
        ..Default::default()
    };

    let raster = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };

    let ms = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };

    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::RGBA,
        ..Default::default()
    };
    let blend = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: 1,
        p_attachments: &blend_attachment,
        ..Default::default()
    };

    let layout_info = vk::PipelineLayoutCreateInfo::default();
    let layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| GpuError::Backend(format!("create_pipeline_layout: {e}")))?
    };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &ms,
        p_color_blend_state: &blend,
        layout,
        render_pass,
        subpass: 0,
        ..Default::default()
    };

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| GpuError::Backend(format!("create_graphics_pipelines: {e}")))?
            .into_iter()
            .next()
            .unwrap()
    };

    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }

    Ok((layout, pipeline))
}

fn create_command_pool(device: &ash::Device, queue_family: u32) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo {
        queue_family_index: queue_family,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    unsafe {
        device
            .create_command_pool(&info, None)
            .map_err(|e| GpuError::Backend(format!("create_command_pool: {e}")))
    }
}

fn allocate_and_record_cmd_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    render_pass: vk::RenderPass,
    framebuffers: &[vk::Framebuffer],
    pipeline: vk::Pipeline,
    extent: vk::Extent2D,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: framebuffers.len() as u32,
        ..Default::default()
    };
    let cmd_buffers = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| GpuError::Backend(format!("allocate_command_buffers: {e}")))?
    };

    for (i, &cb) in cmd_buffers.iter().enumerate() {
        let begin = vk::CommandBufferBeginInfo::default();
        unsafe {
            device
                .begin_command_buffer(cb, &begin)
                .map_err(|e| GpuError::Backend(format!("begin_command_buffer: {e}")))?;
        }

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.02, 1.0],
            },
        };
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer: framebuffers[i],
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            },
            clear_value_count: 1,
            p_clear_values: &clear,
            ..Default::default()
        };

        unsafe {
            device.cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_draw(cb, 3, 1, 0, 0);
            device.cmd_end_render_pass(cb);
            device
                .end_command_buffer(cb)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
    }

    Ok(cmd_buffers)
}

fn create_sync(
    device: &ash::Device,
    count: usize,
) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
    let sem_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo {
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };

    let mut image_available = Vec::with_capacity(count);
    let mut render_finished = Vec::with_capacity(count);
    let mut fences = Vec::with_capacity(count);

    for _ in 0..count {
        unsafe {
            image_available.push(
                device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_semaphore: {e}")))?,
            );
            render_finished.push(
                device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_semaphore: {e}")))?,
            );
            fences.push(
                device
                    .create_fence(&fence_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_fence: {e}")))?,
            );
        }
    }

    Ok((image_available, render_finished, fences))
}
