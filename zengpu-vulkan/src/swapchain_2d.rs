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
//! The swapchain prefers a **non-sRGB** (`B8G8R8A8_UNORM`) format so straight
//! sRGB colour bytes are written through unchanged, matching the CPU rasterizer.
//!
//! Resize and surface-loss are handled by recreating the swapchain-dependent
//! resources ([`SwapchainResources`]); the pipeline uses **dynamic** viewport
//! and scissor so it survives a resize untouched. Instance buffers grow on
//! demand from a small base allocation rather than reserving a fixed maximum.

use std::sync::{Arc, Mutex};

use ash::{khr, vk};
use inline_spirv::inline_spirv;
use zengpu_hal::{GpuError, PresentMode, Result, SurfaceError};

use crate::device::VulkanDeviceInner;
use crate::instance::VulkanShared;
use crate::swapchain::create_platform_surface;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Initial per-frame instance-buffer capacity, in rectangles. Buffers double
/// on demand when a frame needs more, so idle/typical scenes stay small.
const INITIAL_RECTS: usize = 256;

/// One solid-colour rectangle instance: `rect` is `[x, y, w, h]` in physical
/// pixels, `color` is straight RGBA in `0.0..=1.0`. `#[repr(C)]` so a slice
/// uploads directly as the per-instance vertex attributes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RectInstance {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

/// One filled-circle instance: `center_radius` is `[cx, cy, radius, _]` in
/// physical pixels, `color` is straight RGBA. Same 32-byte layout as
/// [`RectInstance`], so it shares the vertex-input binding and instance buffer.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CircleInstance {
    pub center_radius: [f32; 4],
    pub color: [f32; 4],
}

/// One 2-stop gradient fill over a rect. `kind` in `a[3]` selects linear
/// (`0.0`) vs radial (`1.0`).
///
/// - **Linear:** `a = [start.x, start.y, _, 0.0]`, `b = [end.x, end.y, _, _]`.
/// - **Radial:** `a = [center.x, center.y, radius, 1.0]`, `b` unused.
///
/// `color0`/`color1` are straight RGBA. 80-byte `#[repr(C)]` (five `vec4`).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GradientInstance {
    pub rect: [f32; 4],
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub color0: [f32; 4],
    pub color1: [f32; 4],
}

/// One frame's 2D primitives to draw. Within a primitive kind the order is
/// back-to-front (painter's algorithm); across kinds the draw order is fixed
/// (rects, gradients, circles), so cross-kind z-order is not yet preserved —
/// true interleaving is a scene-graph concern. Borrows the instance slices.
#[derive(Default, Clone, Copy)]
pub struct Frame2d<'a> {
    pub clear: Option<[f32; 4]>,
    pub rects: &'a [RectInstance],
    pub gradients: &'a [GradientInstance],
    pub circles: &'a [CircleInstance],
}

/// Rects and circles are 32 bytes; the shared 32-byte buffer/binding rely on it.
const INSTANCE_SIZE: usize = std::mem::size_of::<RectInstance>();
const _: () = assert!(std::mem::size_of::<CircleInstance>() == INSTANCE_SIZE);
const GRADIENT_SIZE: usize = std::mem::size_of::<GradientInstance>();
const _: () = assert!(GRADIENT_SIZE == 80);

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

// Circle: expand the instance's bounding-box quad, then evaluate a signed
// distance field in the fragment shader for a 1px-antialiased edge.
const CIRCLE_VERT_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec4 i_data;   // cx, cy, radius, _
    layout(location = 1) in vec4 i_color;  // straight RGBA
    layout(push_constant) uniform PC { vec2 viewport; } pc;
    layout(location = 0) out vec2 v_local;   // offset from centre (px)
    layout(location = 1) out vec4 v_color;
    layout(location = 2) out float v_radius;
    void main() {
        vec2 corners[6] = vec2[](
            vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
            vec2( 1.0, -1.0), vec2(1.0,  1.0), vec2(-1.0, 1.0)
        );
        vec2 corner = corners[gl_VertexIndex];
        float r = i_data.z;
        vec2 px = i_data.xy + corner * r;
        v_local = corner * r;
        v_radius = r;
        v_color = i_color;
        vec2 ndc = (px / pc.viewport) * 2.0 - 1.0;
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    "#,
    vert,
    vulkan1_0
);

const CIRCLE_FRAG_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec2 v_local;
    layout(location = 1) in vec4 v_color;
    layout(location = 2) in float v_radius;
    layout(location = 0) out vec4 o_color;
    void main() {
        float dist = length(v_local);
        float alpha = 1.0 - smoothstep(v_radius - 1.0, v_radius, dist);
        if (alpha <= 0.0) discard;
        o_color = vec4(v_color.rgb, v_color.a * alpha);
    }
    "#,
    frag,
    vulkan1_0
);

// Gradient: expand the fill rect, then in the fragment shader compute the
// gradient parameter `t` (linear projection or radial distance) and mix the two
// stop colours.
const GRADIENT_VERT_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec4 i_rect;    // x, y, w, h (fill area, px)
    layout(location = 1) in vec4 i_a;        // linear start.xy / radial centre.xy,.z=r,.w=kind
    layout(location = 2) in vec4 i_b;        // linear end.xy
    layout(location = 3) in vec4 i_color0;
    layout(location = 4) in vec4 i_color1;
    layout(push_constant) uniform PC { vec2 viewport; } pc;
    layout(location = 0) out vec2 v_px;
    layout(location = 1) out vec4 v_a;
    layout(location = 2) out vec4 v_b;
    layout(location = 3) out vec4 v_c0;
    layout(location = 4) out vec4 v_c1;
    void main() {
        vec2 corners[6] = vec2[](
            vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
            vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
        );
        vec2 corner = corners[gl_VertexIndex];
        vec2 px = i_rect.xy + corner * i_rect.zw;
        v_px = px; v_a = i_a; v_b = i_b; v_c0 = i_color0; v_c1 = i_color1;
        vec2 ndc = (px / pc.viewport) * 2.0 - 1.0;
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    "#,
    vert,
    vulkan1_0
);

const GRADIENT_FRAG_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    layout(location = 0) in vec2 v_px;
    layout(location = 1) in vec4 v_a;
    layout(location = 2) in vec4 v_b;
    layout(location = 3) in vec4 v_c0;
    layout(location = 4) in vec4 v_c1;
    layout(location = 0) out vec4 o_color;
    void main() {
        float t;
        if (v_a.w < 0.5) {
            vec2 d = v_b.xy - v_a.xy;
            t = dot(v_px - v_a.xy, d) / max(dot(d, d), 1e-6);
        } else {
            t = length(v_px - v_a.xy) / max(v_a.z, 1e-6);
        }
        o_color = mix(v_c0, v_c1, clamp(t, 0.0, 1.0));
    }
    "#,
    frag,
    vulkan1_0
);

// ── Per-frame instance buffer (growable) ────────────────────────────────────

/// A persistently-mapped host-visible vertex buffer holding one frame's rect
/// instances. One per frame-in-flight so the CPU can fill frame N+1 while the
/// GPU still reads frame N. Grows (reallocates) when a frame needs more rects.
struct InstanceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    /// Size of one instance in bytes (32 for rects/circles, 80 for gradients).
    elem_size: usize,
    /// Capacity in instances.
    capacity: usize,
}

impl InstanceBuffer {
    fn new(inner: &VulkanDeviceInner, elem_size: usize, capacity: usize) -> Result<Self> {
        let (buffer, memory, mapped) = alloc_mapped_vertex_buffer(inner, elem_size, capacity)?;
        Ok(Self { buffer, memory, mapped, elem_size, capacity })
    }

    /// Ensure room for `needed` instances, reallocating (doubling) if required.
    fn ensure_capacity(&mut self, inner: &VulkanDeviceInner, needed: usize) -> Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }
        let mut new_cap = self.capacity.max(1);
        while new_cap < needed {
            new_cap *= 2;
        }
        let (buffer, memory, mapped) = alloc_mapped_vertex_buffer(inner, self.elem_size, new_cap)?;
        // Swap in the new allocation, then free the old one.
        let old = InstanceBuffer {
            buffer: self.buffer,
            memory: self.memory,
            mapped: self.mapped,
            elem_size: self.elem_size,
            capacity: self.capacity,
        };
        self.buffer = buffer;
        self.memory = memory;
        self.mapped = mapped;
        self.capacity = new_cap;
        old.destroy(inner);
        Ok(())
    }

    /// Copy `items` into the mapped buffer; caller guarantees capacity. `T` must
    /// match this buffer's element size.
    fn upload_bytes<T>(&self, items: &[T]) {
        debug_assert_eq!(std::mem::size_of::<T>(), self.elem_size);
        if items.is_empty() {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr() as *const u8,
                self.mapped,
                std::mem::size_of_val(items),
            );
        }
    }

    fn destroy(&self, inner: &VulkanDeviceInner) {
        unsafe {
            inner.device.unmap_memory(self.memory);
            inner.device.destroy_buffer(self.buffer, None);
            inner.device.free_memory(self.memory, None);
        }
    }
}

/// Allocate a host-visible, persistently-mapped vertex buffer for `capacity`
/// instances of `elem_size` bytes each.
fn alloc_mapped_vertex_buffer(
    inner: &VulkanDeviceInner,
    elem_size: usize,
    capacity: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
    let size = (capacity.max(1) * elem_size) as u64;
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
    .ok_or_else(|| {
        unsafe { inner.device.destroy_buffer(buffer, None) };
        GpuError::Backend("no host-visible memory for instances".to_string())
    })?;

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
    Ok((buffer, memory, mapped))
}

// ── Swapchain-dependent resources (recreated on resize / surface loss) ──────

struct SwapchainResources {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    extent: vk::Extent2D,
}

impl SwapchainResources {
    /// Destroy the per-swapchain objects (not the swapchain itself, which the
    /// caller may pass as `old_swapchain` during recreation).
    fn destroy_views_framebuffers(&self, device: &ash::Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            for &iv in &self.image_views {
                device.destroy_image_view(iv, None);
            }
        }
    }
}

struct FrameState {
    current: usize,
}

/// Vulkan swapchain that draws a batch of instanced rectangles per frame.
pub struct Vulkan2dSurface {
    inner: Arc<VulkanDeviceInner>,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    // Extent-independent (dynamic viewport/scissor), so they survive resize.
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    rect_pipeline: vk::Pipeline,
    circle_pipeline: vk::Pipeline,
    gradient_pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    /// One command buffer per frame-in-flight (re-recorded each present).
    cmd_buffers: Vec<vk::CommandBuffer>,
    /// Growable instance buffers per frame-in-flight, one set per primitive kind.
    rect_buffers: Vec<Mutex<InstanceBuffer>>,
    circle_buffers: Vec<Mutex<InstanceBuffer>>,
    gradient_buffers: Vec<Mutex<InstanceBuffer>>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    resources: Mutex<SwapchainResources>,
    frame_state: Mutex<FrameState>,
    format: vk::Format,
    present_mode: vk::PresentModeKHR,
}

// Safety: all mutable cross-frame state is behind Mutex; ash types are Send+Sync.
unsafe impl Send for Vulkan2dSurface {}
unsafe impl Sync for Vulkan2dSurface {}

impl Vulkan2dSurface {
    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        device: &crate::device::VulkanDevice,
        handles: &zengpu_hal::WindowHandles,
        config: zengpu_hal::SurfaceConfig,
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
        let present_mode = pick_present_mode(&surface_loader, inner.physical, surface, config.present_mode)?;
        let format = pick_format(&surface_loader, inner.physical, surface)?;

        let (swapchain, images, extent) = create_swapchain(
            &surface_loader,
            &swapchain_loader,
            inner.physical,
            surface,
            format,
            present_mode,
            config.width,
            config.height,
            vk::SwapchainKHR::null(),
        )?;
        let image_views = create_image_views(&inner.device, &images, format)?;
        let render_pass = create_render_pass(&inner.device, format)?;
        let framebuffers = create_framebuffers(&inner.device, render_pass, &image_views, extent)?;
        let pipeline_layout = create_pipeline_layout(&inner.device)?;
        let rect_pipeline = create_pipeline(
            &inner.device,
            pipeline_layout,
            render_pass,
            VERT_SPV,
            FRAG_SPV,
            INSTANCE_SIZE as u32,
            2,
        )?;
        let circle_pipeline = create_pipeline(
            &inner.device,
            pipeline_layout,
            render_pass,
            CIRCLE_VERT_SPV,
            CIRCLE_FRAG_SPV,
            INSTANCE_SIZE as u32,
            2,
        )?;
        let gradient_pipeline = create_pipeline(
            &inner.device,
            pipeline_layout,
            render_pass,
            GRADIENT_VERT_SPV,
            GRADIENT_FRAG_SPV,
            GRADIENT_SIZE as u32,
            5,
        )?;

        let cmd_pool = create_command_pool(&inner.device, inner.queue_family)?;
        let cmd_buffers = allocate_cmd_buffers(&inner.device, cmd_pool, MAX_FRAMES_IN_FLIGHT)?;

        let mut rect_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut circle_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut gradient_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            rect_buffers.push(Mutex::new(InstanceBuffer::new(&inner, INSTANCE_SIZE, INITIAL_RECTS)?));
            circle_buffers
                .push(Mutex::new(InstanceBuffer::new(&inner, INSTANCE_SIZE, INITIAL_RECTS)?));
            gradient_buffers
                .push(Mutex::new(InstanceBuffer::new(&inner, GRADIENT_SIZE, INITIAL_RECTS)?));
        }

        let (image_available, render_finished, in_flight) =
            create_sync(&inner.device, MAX_FRAMES_IN_FLIGHT)?;

        Ok(Self {
            inner,
            surface_loader,
            swapchain_loader,
            surface,
            render_pass,
            pipeline_layout,
            rect_pipeline,
            circle_pipeline,
            gradient_pipeline,
            cmd_pool,
            cmd_buffers,
            rect_buffers,
            circle_buffers,
            gradient_buffers,
            image_available,
            render_finished,
            in_flight,
            resources: Mutex::new(SwapchainResources {
                swapchain,
                images,
                image_views,
                framebuffers,
                extent,
            }),
            frame_state: Mutex::new(FrameState { current: 0 }),
            format,
            present_mode,
        })
    }

    /// Swapchain extent in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        let res = self.resources.lock().unwrap();
        (res.extent.width, res.extent.height)
    }

    /// Number of swapchain images.
    pub fn image_count(&self) -> u32 {
        self.resources.lock().unwrap().images.len() as u32
    }

    /// Recreate the swapchain (e.g. after a resize or surface-lost). Safe to
    /// call when the window is minimised — bails out while the extent is zero.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        let mut res = self.resources.lock().unwrap();
        self.recreate(&mut res, width, height)
    }

    /// Draw `frame`'s primitives (clear, then rects, then circles) and present.
    /// Recreates the swapchain transparently on resize / surface loss.
    pub fn present(&self, frame: Frame2d) -> Result<()> {
        let mut state = self.frame_state.lock().unwrap();
        let current = state.current;

        unsafe {
            self.inner
                .device
                .wait_for_fences(&[self.in_flight[current]], true, u64::MAX)
                .map_err(|e| GpuError::Backend(format!("wait_for_fences: {e}")))?;
        }

        let mut res = self.resources.lock().unwrap();
        if res.extent.width == 0 || res.extent.height == 0 {
            return Ok(()); // minimised — nothing to present
        }

        let image_index = match unsafe {
            self.swapchain_loader.acquire_next_image(
                res.swapchain,
                u64::MAX,
                self.image_available[current],
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Swapchain stale: recreate and skip this frame. The fence was
                // not reset, so the slot stays usable next call.
                let (w, h) = (res.extent.width, res.extent.height);
                return self.recreate(&mut res, w, h);
            }
            Err(e) => return Err(map_surface_err(e)),
        };

        unsafe {
            self.inner
                .device
                .reset_fences(&[self.in_flight[current]])
                .map_err(|e| GpuError::Backend(format!("reset_fences: {e}")))?;
        }

        // Upload instances (growing each buffer if needed).
        let rect_buf = {
            let mut ib = self.rect_buffers[current].lock().unwrap();
            ib.ensure_capacity(&self.inner, frame.rects.len())?;
            ib.upload_bytes(frame.rects);
            ib.buffer
        };
        let circle_buf = {
            let mut ib = self.circle_buffers[current].lock().unwrap();
            ib.ensure_capacity(&self.inner, frame.circles.len())?;
            ib.upload_bytes(frame.circles);
            ib.buffer
        };
        let gradient_buf = {
            let mut ib = self.gradient_buffers[current].lock().unwrap();
            ib.ensure_capacity(&self.inner, frame.gradients.len())?;
            ib.upload_bytes(frame.gradients);
            ib.buffer
        };

        let cmd = self.cmd_buffers[current];
        self.record(
            cmd,
            res.framebuffers[image_index as usize],
            res.extent,
            &frame,
            [rect_buf, gradient_buf, circle_buf],
        )?;

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

        let swapchains = [res.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };
        match unsafe { self.swapchain_loader.queue_present(self.inner.queue, &present_info) } {
            Ok(_suboptimal) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let (w, h) = (res.extent.width, res.extent.height);
                self.recreate(&mut res, w, h)?;
            }
            Err(e) => return Err(map_surface_err(e)),
        }

        state.current = (current + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Recreate swapchain + image views + framebuffers in place. Keeps the
    /// render pass and pipeline (format-only, viewport is dynamic).
    fn recreate(&self, res: &mut SwapchainResources, width: u32, height: u32) -> Result<()> {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        let old_swapchain = res.swapchain;
        let (swapchain, images, extent) = create_swapchain(
            &self.surface_loader,
            &self.swapchain_loader,
            self.inner.physical,
            self.surface,
            self.format,
            self.present_mode,
            width,
            height,
            old_swapchain,
        )?;

        res.destroy_views_framebuffers(&self.inner.device);
        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        let image_views = create_image_views(&self.inner.device, &images, self.format)?;
        let framebuffers =
            create_framebuffers(&self.inner.device, self.render_pass, &image_views, extent)?;

        res.swapchain = swapchain;
        res.images = images;
        res.image_views = image_views;
        res.framebuffers = framebuffers;
        res.extent = extent;
        Ok(())
    }

    fn record(
        &self,
        cmd: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        frame: &Frame2d,
        bufs: [vk::Buffer; 3], // [rect, gradient, circle]
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
                float32: frame.clear.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            },
        };
        let rp_begin = vk::RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            },
            clear_value_count: 1,
            p_clear_values: &clear_value,
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
        let push = [extent.width as f32, extent.height as f32];

        unsafe {
            dev.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            dev.cmd_set_viewport(cmd, 0, &[viewport]);
            dev.cmd_set_scissor(cmd, 0, &[scissor]);
            dev.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(push.as_ptr() as *const u8, 8),
            );

            let [rect_buf, gradient_buf, circle_buf] = bufs;
            // Fixed per-kind order: rects (backgrounds/panels), then gradient
            // fills, then circles on top. See Frame2d note on cross-kind z-order.
            if !frame.rects.is_empty() {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rect_pipeline);
                dev.cmd_bind_vertex_buffers(cmd, 0, &[rect_buf], &[0]);
                dev.cmd_draw(cmd, 6, frame.rects.len() as u32, 0, 0);
            }
            if !frame.gradients.is_empty() {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.gradient_pipeline);
                dev.cmd_bind_vertex_buffers(cmd, 0, &[gradient_buf], &[0]);
                dev.cmd_draw(cmd, 6, frame.gradients.len() as u32, 0, 0);
            }
            if !frame.circles.is_empty() {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.circle_pipeline);
                dev.cmd_bind_vertex_buffers(cmd, 0, &[circle_buf], &[0]);
                dev.cmd_draw(cmd, 6, frame.circles.len() as u32, 0, 0);
            }

            dev.cmd_end_render_pass(cmd);
            dev.end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for Vulkan2dSurface {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
        }
        for ib in self
            .rect_buffers
            .iter()
            .chain(&self.circle_buffers)
            .chain(&self.gradient_buffers)
        {
            ib.lock().unwrap().destroy(&self.inner);
        }
        {
            let res = self.resources.lock().unwrap();
            res.destroy_views_framebuffers(&self.inner.device);
            unsafe {
                self.swapchain_loader.destroy_swapchain(res.swapchain, None);
            }
        }
        unsafe {
            let dev = &self.inner.device;
            dev.free_command_buffers(self.cmd_pool, &self.cmd_buffers);
            dev.destroy_pipeline(self.rect_pipeline, None);
            dev.destroy_pipeline(self.circle_pipeline, None);
            dev.destroy_pipeline(self.gradient_pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_render_pass(self.render_pass, None);
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

fn pick_format(
    surface_loader: &khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::Format> {
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface formats: {e}")))?
    };
    // Prefer a non-sRGB BGRA format so straight sRGB colour bytes pass through
    // unchanged (matching the CPU rasterizer). Fall back to whatever's first.
    formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.first())
        .map(|f| f.format)
        .ok_or_else(|| GpuError::Backend("no surface formats".to_string()))
}

fn pick_present_mode(
    surface_loader: &khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    requested: PresentMode,
) -> Result<vk::PresentModeKHR> {
    let modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface present modes: {e}")))?
    };
    let desired = match requested {
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    Ok(if modes.contains(&desired) {
        desired
    } else {
        vk::PresentModeKHR::FIFO // guaranteed available
    })
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain(
    surface_loader: &khr::surface::Instance,
    swapchain_loader: &khr::swapchain::Device,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    format: vk::Format,
    present_mode: vk::PresentModeKHR,
    width: u32,
    height: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D)> {
    let caps = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface capabilities: {e}")))?
    };
    let color_space = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| GpuError::Backend(format!("surface formats: {e}")))?
    }
    .iter()
    .find(|f| f.format == format)
    .map(|f| f.color_space)
    .unwrap_or(vk::ColorSpaceKHR::SRGB_NONLINEAR);

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    // Minimised window: zero extent. Return an empty swapchain handle so the
    // caller can detect and skip presenting.
    if extent.width == 0 || extent.height == 0 {
        return Ok((vk::SwapchainKHR::null(), Vec::new(), extent));
    }

    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: image_count,
        image_format: format,
        image_color_space: color_space,
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
    Ok((swapchain, images, extent))
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

/// Pipeline layout shared by the rect and circle pipelines (same vertex push
/// constant). Created once.
fn create_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
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
    unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| GpuError::Backend(format!("create_pipeline_layout: {e}")))
    }
}

/// `vec4` vertex attributes at consecutive 16-byte offsets, one per `location`
/// in `0..count`. Both 2D instance layouts are packed `vec4`s, so this fully
/// describes either.
fn vec4_attributes(count: u32) -> Vec<vk::VertexInputAttributeDescription> {
    (0..count)
        .map(|i| vk::VertexInputAttributeDescription {
            location: i,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: i * 16,
        })
        .collect()
}

/// Build an instanced 2D pipeline. Pipelines differ only in shaders and vertex
/// layout (`stride` + `vec4` attribute `count`); everything else (dynamic
/// viewport/scissor, alpha blend) is identical.
fn create_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vert_spv: &[u32],
    frag_spv: &[u32],
    stride: u32,
    attr_count: u32,
) -> Result<vk::Pipeline> {
    let vert = create_shader_module(device, vert_spv)?;
    let frag = create_shader_module(device, frag_spv)?;
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

    // One per-instance binding; attributes are consecutive vec4s.
    let binding = vk::VertexInputBindingDescription {
        binding: 0,
        stride,
        input_rate: vk::VertexInputRate::INSTANCE,
    };
    let attributes = vec4_attributes(attr_count);
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

    // Viewport and scissor are dynamic so the pipeline survives resize; only a
    // count is fixed here.
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count: 1,
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
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

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &ms,
        p_color_blend_state: &blend,
        p_dynamic_state: &dynamic_state,
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
    Ok(pipeline)
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
