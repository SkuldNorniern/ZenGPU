//! Vulkan 2D rect painter — instanced solid-colour quads (aurea G4 / Rung 1).
//!
//! Unlike [`crate::swapchain`] (which pre-records a static triangle), the 2D
//! surface re-records its command buffer every frame because the rect set
//! changes: `present` does acquire → upload instances → record → submit →
//! present in one call.
//!
//! Geometry is a unit quad expanded in the vertex shader from `gl_VertexIndex`
//! (6 vertices, two triangles); per-instance data is `[x, y, w, h]` in physical
//! pixels plus straight RGBA.  A push-constant viewport size maps pixels → NDC.
//!
//! The swapchain deliberately prefers a **non-sRGB** (`B8G8R8A8_UNORM`) format
//! so the straight sRGB colour bytes are written through unchanged, matching
//! the CPU rasterizer's appearance (which writes sRGB bytes directly).

use std::sync::{Arc, Mutex};

use ash::{khr, vk};
use inline_spirv::inline_spirv;
use zengpu_hal::{GpuError, PresentMode, Result, SurfaceConfig, SurfaceError};

use crate::device::VulkanDeviceInner;
use crate::instance::VulkanShared;
use crate::swapchain::create_platform_surface;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame instance-buffer capacity in rectangles. Rects beyond this are
/// dropped for the frame (Rung 1 keeps a fixed allocation rather than growing).
const MAX_RECTS: usize = 65_536;

/// One solid-colour rectangle instance: `rect` is `[x, y, w, h]` in physical
/// pixels, `color` is straight RGBA in `0.0..=1.0`. `#[repr(C)]` so a slice
/// uploads directly as the per-instance vertex attributes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RectInstance {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

const INSTANCE_SIZE: usize = std::mem::size_of::<RectInstance>();

// ── Compiled shaders ──────────────────────────────────────────────────────────

const VERT_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec4 i_rect;   // x, y, w, h  (physical pixels)
    layout(location = 1) in vec4 i_color;  // straight RGBA
    layout(push_constant) uniform PC { vec2 viewport; } pc;
    layout(location = 0) out vec4 v_color;
    void main() {
        vec2 corners[6] = vec2[](
            vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
            vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
        );
        vec2 corner = corners[gl_VertexIndex];
        vec2 px = i_rect.xy + corner * i_rect.zw;
        // Vulkan NDC: top-left is (-1, -1), +y points down — matches pixel space.
        vec2 ndc = (px / pc.viewport) * 2.0 - 1.0;
        gl_Position = vec4(ndc, 0.0, 1.0);
        v_color = i_color;
    }
    "#,
    vert,
    vulkan1_0
);

const FRAG_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec4 v_color;
    layout(location = 0) out vec4 o_color;
    void main() { o_color = v_color; }
    "#,
    frag,
    vulkan1_0
);

// ── Per-frame instance buffer ───────────────────────────────────────────────

/// A persistently-mapped host-visible vertex buffer holding one frame's rect
/// instances. One per frame-in-flight so the CPU can fill frame N+1 while the
/// GPU still reads frame N.
struct InstanceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

impl InstanceBuffer {
    fn new(inner: &VulkanDeviceInner) -> Result<Self> {
        let size = (MAX_RECTS * INSTANCE_SIZE) as u64;
        let info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe {
            inner
                .device
                .create_buffer(&info, None)
                .map_err(|e| GpuError::Backend(format!("create instance buffer: {e}")))?
        };
        let reqs = unsafe { inner.device.get_buffer_memory_requirements(buffer) };
        let type_index = find_memory_type(
            inner,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| GpuError::Backend("no host-visible memory for instances".to_string()))?;

        let memory = unsafe {
            inner
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: reqs.size,
                        memory_type_index: type_index,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| {
                    inner.device.destroy_buffer(buffer, None);
                    GpuError::Backend(format!("allocate instance memory: {e}"))
                })?
        };
        unsafe {
            if let Err(e) = inner.device.bind_buffer_memory(buffer, memory, 0) {
                inner.device.destroy_buffer(buffer, None);
                inner.device.free_memory(memory, None);
                return Err(GpuError::Backend(format!("bind instance memory: {e}")));
            }
        }
        let mapped = unsafe {
            inner
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    inner.device.destroy_buffer(buffer, None);
                    inner.device.free_memory(memory, None);
                    GpuError::Backend(format!("map instance memory: {e}"))
                })? as *mut u8
        };

        Ok(Self { buffer, memory, mapped })
    }

    /// Copy up to `MAX_RECTS` instances in; returns the count actually written.
    fn upload(&self, rects: &[RectInstance]) -> u32 {
        let count = rects.len().min(MAX_RECTS);
        unsafe {
            std::ptr::copy_nonoverlapping(
                rects.as_ptr() as *const u8,
                self.mapped,
                count * INSTANCE_SIZE,
            );
        }
        count as u32
    }

    fn destroy(&self, inner: &VulkanDeviceInner) {
        unsafe {
            inner.device.unmap_memory(self.memory);
            inner.device.destroy_buffer(self.buffer, None);
            inner.device.free_memory(self.memory, None);
        }
    }
}

// ── Vulkan2dSurface ──────────────────────────────────────────────────────────

struct FrameState {
    current: usize,
}

/// Vulkan swapchain that draws a batch of instanced rectangles per frame.
pub struct Vulkan2dSurface {
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
    /// One command buffer per frame-in-flight (re-recorded each present),
    /// guarded by `in_flight[current]`. The render pass targets
    /// `framebuffers[image_index]`, so a per-frame buffer is sufficient.
    cmd_buffers: Vec<vk::CommandBuffer>,
    instance_buffers: Vec<InstanceBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame_state: Mutex<FrameState>,
    extent: vk::Extent2D,
    #[allow(dead_code)]
    format: vk::Format,
}

// Safety: all mutable cross-frame state is behind Mutex; ash types are Send+Sync.
unsafe impl Send for Vulkan2dSurface {}
unsafe impl Sync for Vulkan2dSurface {}

impl Vulkan2dSurface {
    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        device: &crate::device::VulkanDevice,
        handles: &zengpu_hal::WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self> {
        let inner = Arc::clone(&device.inner);
        let surface_loader = khr::surface::Instance::new(&shared.entry, &shared.instance);
        let surface = create_platform_surface(&shared, handles)?;

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
        let (swapchain, images, format, extent) =
            create_swapchain(&surface_loader, &swapchain_loader, inner.physical, surface, &config)?;

        let image_views = create_image_views(&inner.device, &images, format)?;
        let render_pass = create_render_pass(&inner.device, format)?;
        let framebuffers = create_framebuffers(&inner.device, render_pass, &image_views, extent)?;
        let (pipeline_layout, pipeline) = create_pipeline(&inner.device, render_pass, extent)?;

        let cmd_pool = create_command_pool(&inner.device, inner.queue_family)?;
        let cmd_buffers = allocate_cmd_buffers(&inner.device, cmd_pool, MAX_FRAMES_IN_FLIGHT)?;

        let mut instance_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            instance_buffers.push(InstanceBuffer::new(&inner)?);
        }

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
            instance_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_state: Mutex::new(FrameState { current: 0 }),
            extent,
            format,
        })
    }

    /// Swapchain extent in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// Number of swapchain images.
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Clear to `clear` (defaults to opaque black) and draw `rects`, then
    /// present. Performs the full acquire → upload → record → submit → present
    /// cycle for one frame.
    pub fn present(&self, clear: Option<[f32; 4]>, rects: &[RectInstance]) -> Result<()> {
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
                .map_err(map_surface_err)?
        };

        let instance_count = self.instance_buffers[current].upload(rects);

        let cmd = self.cmd_buffers[current];
        self.record(cmd, image_index as usize, current, clear, instance_count)?;

        let wait_semaphores = [self.image_available[current]];
        let signal_semaphores = [self.render_finished[current]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [cmd];
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
                .map_err(map_surface_err)?;
        }

        state.current = (current + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn record(
        &self,
        cmd: vk::CommandBuffer,
        image_index: usize,
        frame: usize,
        clear: Option<[f32; 4]>,
        instance_count: u32,
    ) -> Result<()> {
        let dev = &self.inner.device;
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| GpuError::Backend(format!("reset_command_buffer: {e}")))?;
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| GpuError::Backend(format!("begin_command_buffer: {e}")))?;
        }

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            },
        };
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers[image_index],
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.extent,
            },
            clear_value_count: 1,
            p_clear_values: &clear_value,
            ..Default::default()
        };

        let viewport = [self.extent.width as f32, self.extent.height as f32];

        unsafe {
            dev.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            if instance_count > 0 {
                dev.cmd_bind_vertex_buffers(cmd, 0, &[self.instance_buffers[frame].buffer], &[0]);
                dev.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(viewport.as_ptr() as *const u8, 8),
                );
                dev.cmd_draw(cmd, 6, instance_count, 0, 0);
            }
            dev.cmd_end_render_pass(cmd);
            dev.end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
        Ok(())
    }

    fn destroy_resources(&self) {
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
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for Vulkan2dSurface {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        for ib in &self.instance_buffers {
            ib.destroy(&self.inner);
        }
        self.destroy_resources();
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

// ── Helpers ──────────────────────────────────────────────────────────────────

fn map_surface_err(e: vk::Result) -> GpuError {
    GpuError::Surface(match e {
        vk::Result::ERROR_OUT_OF_DATE_KHR => SurfaceError::Outdated,
        vk::Result::ERROR_SURFACE_LOST_KHR => SurfaceError::Lost,
        _ => SurfaceError::OutOfMemory,
    })
}

fn find_memory_type(
    inner: &VulkanDeviceInner,
    type_bits: u32,
    props: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mem_props = unsafe {
        inner
            .shared
            .instance
            .get_physical_device_memory_properties(inner.physical)
    };
    (0..mem_props.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(props)
    })
}

fn create_swapchain(
    surface_loader: &khr::surface::Instance,
    swapchain_loader: &khr::swapchain::Device,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    config: &SurfaceConfig,
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

    // Prefer a non-sRGB BGRA format so straight sRGB colour bytes pass through
    // unchanged (matching the CPU rasterizer). Fall back to whatever's first.
    let surface_format = formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.first())
        .copied()
        .ok_or_else(|| GpuError::Backend("no surface formats".to_string()))?;

    let desired_mode = match config.present_mode {
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    let present_mode = if present_modes.contains(&desired_mode) {
        desired_mode
    } else {
        vk::PresentModeKHR::FIFO
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: config
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
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
        old_swapchain: vk::SwapchainKHR::null(),
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

    // One per-instance binding: RectInstance (32 bytes), two vec4 attributes.
    let binding = vk::VertexInputBindingDescription {
        binding: 0,
        stride: INSTANCE_SIZE as u32,
        input_rate: vk::VertexInputRate::INSTANCE,
    };
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 16,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count: 1,
        p_vertex_binding_descriptions: &binding,
        vertex_attribute_description_count: attributes.len() as u32,
        p_vertex_attribute_descriptions: attributes.as_ptr(),
        ..Default::default()
    };
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
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };
    let ms = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };

    // Straight-alpha blending so translucent rect fills composite correctly.
    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let blend = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: 1,
        p_attachments: &blend_attachment,
        ..Default::default()
    };

    let push_constant = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: 8, // vec2 viewport
    };
    let layout_info = vk::PipelineLayoutCreateInfo {
        push_constant_range_count: 1,
        p_push_constant_ranges: &push_constant,
        ..Default::default()
    };
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

fn allocate_cmd_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    count: usize,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: count as u32,
        ..Default::default()
    };
    unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| GpuError::Backend(format!("allocate_command_buffers: {e}")))
    }
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
