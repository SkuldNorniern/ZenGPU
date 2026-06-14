//! Vulkan textured-quad swapchain with bindless descriptor indexing (plan G3).
//!
//! The pipeline exposes an array of 64 combined image samplers (binding 0,
//! set 0) accessible by a `uint tex_index` push constant.  All unused slots
//! are pre-filled with a 1×1 white placeholder so no slot is ever invalid.
//!
//! Command buffers are pre-recorded at creation; the render loop is the same
//! acquire → (pre-recorded draw) → present pattern as `VulkanSwapchain`.

use std::sync::{Arc, Mutex};

use ash::{Device, khr, vk};
use inline_spirv::inline_spirv;
use zengpu_hal::{
    GpuError, GpuSurface, PresentMode, Result, SamplerHandle, SurfaceConfig, SurfaceFrame,
    TextureHandle,
};

use crate::device::VulkanDevice;
use crate::instance::VulkanShared;
use crate::swapchain::create_platform_surface;

/// Maximum number of textures in the bindless array (must match the shader).
pub const BINDLESS_CAPACITY: u32 = 64;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// ── Compiled shaders ──────────────────────────────────────────────────────────

const VERT_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) out vec2 v_uv;
    void main() {
        float x = float((gl_VertexIndex & 1) * 2);
        float y = float((gl_VertexIndex >> 1) * 2);
        v_uv = vec2(x * 0.5, y * 0.5);
        gl_Position = vec4(x - 1.0, y - 1.0, 0.0, 1.0);
    }
    "#,
    vert,
    vulkan1_0
);

const FRAG_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(set = 0, binding = 0) uniform sampler2D textures[64];
    layout(push_constant) uniform PC { uint tex_index; } pc;
    layout(location = 0) in vec2 v_uv;
    layout(location = 0) out vec4 o_color;
    void main() {
        o_color = texture(textures[pc.tex_index], v_uv);
    }
    "#,
    frag,
    vulkan1_0
);

// ── Frame sync state ──────────────────────────────────────────────────────────

struct FrameState {
    current: usize,
}

// ── Placeholder texture (fills unused bindless slots) ─────────────────────────

struct Placeholder {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    sampler: vk::Sampler,
}

unsafe impl Send for Placeholder {}
unsafe impl Sync for Placeholder {}

fn create_placeholder(device: &VulkanDevice) -> Result<Placeholder> {
    let image_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent: vk::Extent3D { width: 1, height: 1, depth: 1 },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        initial_layout: vk::ImageLayout::UNDEFINED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let image = unsafe {
        device
            .inner
            .device
            .create_image(&image_info, None)
            .map_err(|e| GpuError::Backend(format!("placeholder vkCreateImage: {e}")))?
    };
    let mem_reqs = unsafe { device.inner.device.get_image_memory_requirements(image) };
    let mem_props = unsafe {
        device
            .inner
            .shared
            .instance
            .get_physical_device_memory_properties(device.inner.physical)
    };
    let type_index = (0..mem_props.memory_type_count)
        .find(|&i| {
            mem_reqs.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| {
            unsafe { device.inner.device.destroy_image(image, None) };
            GpuError::Backend("no device-local memory for placeholder".to_string())
        })?;
    let memory = unsafe {
        device
            .inner
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: mem_reqs.size,
                    memory_type_index: type_index,
                    ..Default::default()
                },
                None,
            )
            .map_err(|_| {
                device.inner.device.destroy_image(image, None);
                GpuError::Backend("placeholder OOM".to_string())
            })?
    };
    unsafe {
        device
            .inner
            .device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                device.inner.device.destroy_image(image, None);
                device.inner.device.free_memory(memory, None);
                GpuError::Backend(format!("placeholder bind: {e}"))
            })?
    };
    let view = unsafe {
        device
            .inner
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| {
                device.inner.device.destroy_image(image, None);
                device.inner.device.free_memory(memory, None);
                GpuError::Backend(format!("placeholder view: {e}"))
            })?
    };
    let sampler = unsafe {
        device
            .inner
            .device
            .create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| {
                device.inner.device.destroy_image_view(view, None);
                device.inner.device.destroy_image(image, None);
                device.inner.device.free_memory(memory, None);
                GpuError::Backend(format!("placeholder sampler: {e}"))
            })?
    };

    // Upload 1×1 white pixel.
    device.one_shot_submit(|dev, cmd| {
        unsafe {
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                }],
            );
            dev.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }],
            );
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
            );
        }
        Ok(())
    })?;

    Ok(Placeholder { image, view, memory, sampler })
}

// ── VulkanTexturedSwapchain ───────────────────────────────────────────────────

/// Vulkan swapchain that renders a fullscreen quad sampling from a bindless
/// texture array.  Slot 0 is the primary texture; unused slots hold a 1×1
/// white placeholder so the descriptor set is always complete.
pub struct VulkanTexturedSwapchain {
    inner: Arc<crate::device::VulkanDeviceInner>,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    // Bindless descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)] // bound into pre-recorded command buffers; kept for future re-recording
    descriptor_set: vk::DescriptorSet,
    // Pipeline (has push constant for tex_index)
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Commands
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    // Sync
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame_state: Mutex<FrameState>,
    extent: Mutex<vk::Extent2D>,
    // Placeholder texture that fills unused bindless slots
    placeholder: Placeholder,
    // Kept for swapchain recreation on resize (deferred to post-G3).
    #[allow(dead_code)]
    format: vk::Format,
}

unsafe impl Send for VulkanTexturedSwapchain {}
unsafe impl Sync for VulkanTexturedSwapchain {}

impl VulkanTexturedSwapchain {
    /// Create a textured swapchain.  `texture` and `sampler` must refer to
    /// live handles in `device`; they will be bound into slot 0 of the
    /// bindless array and must remain alive for the lifetime of this surface.
    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        device: &VulkanDevice,
        handles: &zengpu_hal::WindowHandles,
        config: SurfaceConfig,
        texture: TextureHandle,
        sampler: SamplerHandle,
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

        let (swapchain, images, format, extent) = create_swapchain(
            &surface_loader,
            &swapchain_loader,
            inner.physical,
            surface,
            &config,
        )?;

        let image_views = create_image_views(&inner.device, &images, format)?;
        let render_pass = create_render_pass(&inner.device, format)?;
        let framebuffers = create_framebuffers(&inner.device, render_pass, &image_views, extent)?;

        let (descriptor_pool, descriptor_set_layout, descriptor_set) =
            create_bindless_descriptors(&inner.device)?;

        let (pipeline_layout, pipeline) =
            create_textured_pipeline(&inner.device, render_pass, extent, descriptor_set_layout)?;

        // Create placeholder and fill all bindless slots with it.
        let placeholder = create_placeholder(device)?;
        fill_bindless_slots(
            &inner.device,
            descriptor_set,
            placeholder.view,
            placeholder.sampler,
        );

        // Register the user texture at slot 0.
        let tex_view = device.texture_view(texture).ok_or_else(|| {
            GpuError::Backend("create_textured_surface: stale TextureHandle".to_string())
        })?;
        let samp_vk = device.sampler_vk(sampler).ok_or_else(|| {
            GpuError::Backend("create_textured_surface: stale SamplerHandle".to_string())
        })?;
        update_bindless_slot(&inner.device, descriptor_set, 0, tex_view, samp_vk);

        let cmd_pool = create_command_pool(&inner.device, inner.queue_family)?;
        let cmd_buffers = record_cmd_buffers(
            &inner.device,
            cmd_pool,
            render_pass,
            &framebuffers,
            pipeline,
            pipeline_layout,
            descriptor_set,
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
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            cmd_pool,
            cmd_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_state: Mutex::new(FrameState { current: 0 }),
            extent: Mutex::new(extent),
            placeholder,
            format,
        })
    }
}

impl Drop for VulkanTexturedSwapchain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            let dev = &self.inner.device;

            dev.free_command_buffers(self.cmd_pool, &self.cmd_buffers);
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                dev.destroy_semaphore(self.image_available[i], None);
                dev.destroy_semaphore(self.render_finished[i], None);
                dev.destroy_fence(self.in_flight[i], None);
            }
            dev.destroy_command_pool(self.cmd_pool, None);

            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_pool(self.descriptor_pool, None);
            dev.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            dev.destroy_sampler(self.placeholder.sampler, None);
            dev.destroy_image_view(self.placeholder.view, None);
            dev.destroy_image(self.placeholder.image, None);
            dev.free_memory(self.placeholder.memory, None);

            for &fb in &self.framebuffers {
                dev.destroy_framebuffer(fb, None);
            }
            dev.destroy_render_pass(self.render_pass, None);
            for &iv in &self.image_views {
                dev.destroy_image_view(iv, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl GpuSurface for VulkanTexturedSwapchain {
    fn configure(&self, _config: SurfaceConfig) -> Result<()> {
        // Resize/recreation deferred to post-G3.
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
                .map_err(|e| {
                    GpuError::Surface(match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => zengpu_hal::SurfaceError::Outdated,
                        vk::Result::ERROR_SURFACE_LOST_KHR => zengpu_hal::SurfaceError::Lost,
                        _ => zengpu_hal::SurfaceError::OutOfMemory,
                    })
                })?
        };

        state.current = current;
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
                .map_err(|e| {
                    GpuError::Surface(match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => zengpu_hal::SurfaceError::Outdated,
                        vk::Result::ERROR_SURFACE_LOST_KHR => zengpu_hal::SurfaceError::Lost,
                        _ => zengpu_hal::SurfaceError::OutOfMemory,
                    })
                })?;
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

    let surface_format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .ok_or_else(|| GpuError::Backend("no surface formats".to_string()))?;

    let desired = match config.present_mode {
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    let present_mode = if present_modes.contains(&desired) { desired } else { vk::PresentModeKHR::FIFO };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: config.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: config.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
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

fn create_image_views(device: &Device, images: &[vk::Image], format: vk::Format) -> Result<Vec<vk::ImageView>> {
    images.iter().map(|&image| {
        unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_image_view: {e}")))
        }
    }).collect()
}

fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
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

fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views.iter().map(|&view| {
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
    }).collect()
}

fn create_bindless_descriptors(
    device: &Device,
) -> Result<(vk::DescriptorPool, vk::DescriptorSetLayout, vk::DescriptorSet)> {
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: BINDLESS_CAPACITY,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        ..Default::default()
    };
    let layout = unsafe {
        device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 1,
                    p_bindings: &binding,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_descriptor_set_layout: {e}")))?
    };

    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: BINDLESS_CAPACITY,
    };
    let pool = unsafe {
        device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 1,
                    p_pool_sizes: &pool_size,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| {
                device.destroy_descriptor_set_layout(layout, None);
                GpuError::Backend(format!("create_descriptor_pool: {e}"))
            })?
    };

    let set = unsafe {
        device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &layout,
                ..Default::default()
            })
            .map_err(|e| {
                device.destroy_descriptor_pool(pool, None);
                device.destroy_descriptor_set_layout(layout, None);
                GpuError::Backend(format!("allocate_descriptor_sets: {e}"))
            })?[0]
    };

    Ok((pool, layout, set))
}

/// Fill all BINDLESS_CAPACITY slots with the placeholder image+sampler so
/// every slot in the array is valid before any real texture is registered.
fn fill_bindless_slots(
    device: &Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_infos: Vec<vk::DescriptorImageInfo> = (0..BINDLESS_CAPACITY)
        .map(|_| vk::DescriptorImageInfo {
            sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        })
        .collect();

    let write = vk::WriteDescriptorSet {
        dst_set: set,
        dst_binding: 0,
        dst_array_element: 0,
        descriptor_count: BINDLESS_CAPACITY,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: image_infos.as_ptr(),
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

/// Update a single slot in the bindless array with a real texture.
fn update_bindless_slot(
    device: &Device,
    set: vk::DescriptorSet,
    slot: u32,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let write = vk::WriteDescriptorSet {
        dst_set: set,
        dst_binding: 0,
        dst_array_element: slot,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: &image_info,
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn create_shader_module(device: &Device, spv: &[u32]) -> Result<vk::ShaderModule> {
    unsafe {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: spv.len() * 4,
                    p_code: spv.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_shader_module: {e}")))
    }
}

fn create_textured_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    set_layout: vk::DescriptorSetLayout,
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

    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: 4, // u32 tex_index
    };

    let layout = unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &set_layout,
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &push_range,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_pipeline_layout: {e}")))?
    };

    let viewport = vk::Viewport {
        x: 0.0, y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0, max_depth: 1.0,
    };
    let scissor = vk::Rect2D { offset: vk::Offset2D::default(), extent };

    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::RGBA,
        ..Default::default()
    };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo::default(),
        p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        },
        p_viewport_state: &vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        },
        p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        },
        p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        },
        p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &blend_attachment,
            ..Default::default()
        },
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

fn create_command_pool(device: &Device, queue_family: u32) -> Result<vk::CommandPool> {
    unsafe {
        device
            .create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: queue_family,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| GpuError::Backend(format!("create_command_pool: {e}")))
    }
}

#[allow(clippy::too_many_arguments)]
fn record_cmd_buffers(
    device: &Device,
    pool: vk::CommandPool,
    render_pass: vk::RenderPass,
    framebuffers: &[vk::Framebuffer],
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    extent: vk::Extent2D,
) -> Result<Vec<vk::CommandBuffer>> {
    let cmd_buffers = unsafe {
        device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: framebuffers.len() as u32,
                ..Default::default()
            })
            .map_err(|e| GpuError::Backend(format!("allocate_command_buffers: {e}")))?
    };

    let tex_index_bytes = 0u32.to_ne_bytes();

    for (i, &cb) in cmd_buffers.iter().enumerate() {
        unsafe {
            device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| GpuError::Backend(format!("begin_command_buffer: {e}")))?;

            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.02, 0.02, 0.02, 1.0] },
            };
            device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo {
                    render_pass,
                    framebuffer: framebuffers[i],
                    render_area: vk::Rect2D { offset: vk::Offset2D::default(), extent },
                    clear_value_count: 1,
                    p_clear_values: &clear,
                    ..Default::default()
                },
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &tex_index_bytes,
            );
            device.cmd_draw(cb, 3, 1, 0, 0);
            device.cmd_end_render_pass(cb);
            device
                .end_command_buffer(cb)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
    }

    Ok(cmd_buffers)
}

fn create_sync(device: &Device, count: usize) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
    let sem_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo { flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() };

    let mut image_available = Vec::with_capacity(count);
    let mut render_finished = Vec::with_capacity(count);
    let mut fences = Vec::with_capacity(count);

    for _ in 0..count {
        unsafe {
            image_available.push(
                device.create_semaphore(&sem_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_semaphore: {e}")))?,
            );
            render_finished.push(
                device.create_semaphore(&sem_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_semaphore: {e}")))?,
            );
            fences.push(
                device.create_fence(&fence_info, None)
                    .map_err(|e| GpuError::Backend(format!("create_fence: {e}")))?,
            );
        }
    }

    Ok((image_available, render_finished, fences))
}
