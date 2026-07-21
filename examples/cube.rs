//! ZenGPU 3D cube rendered through the unified graphics API.
//!
//! Run: `cargo run --example cube`

use core::array::from_fn;
use std::mem::{size_of, size_of_val};
use std::slice::from_raw_parts;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use zengpu::hal::{CompareFn, RasterState};
use zengpu::vulkan::{DepthTarget, VulkanSurface};
use zengpu::{
    Acquire, Bindings, BufferDesc, BufferHandle, BufferUsage, ColorAttachment, ColorTargetState,
    DepthAttachment, DepthState, DeviceRequest, Format, Frame, GpuDevice, GraphicsDevice,
    GraphicsPipelineDesc, LoadOp, MemoryUsage, PipelineHandle, PresentMode, PrimitiveTopology,
    Rect, RenderCommands, RenderPassDesc, Scalar, ShaderDesc, Surface, SurfaceConfig, TargetHandle,
    VertexAttribute, VertexFormat, VertexLayout, Viewport, ViewportScissor, VulkanDevice,
    VulkanInstance, WindowHandles, zsl,
};

/// Bridge a winit window (which speaks `raw-window-handle`) into ZenGPU's
/// `zen-window-handle` types. Examples use `raw-window-handle` as a dev-dep;
/// the ZenGPU library crates depend only on `zen-window-handle`. A future
/// in-house windowing crate will produce `zen-window-handle` types directly.
fn zen_handles<W>(win: &W) -> WindowHandles
where
    W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    use raw_window_handle::{RawDisplayHandle as D, RawWindowHandle as R};
    use zen_window_handle as z;

    let window = match win.window_handle().expect("window handle").as_raw() {
        R::AppKit(h) => z::WindowHandle::AppKit(z::AppKitWindowHandle { ns_view: h.ns_view }),
        R::Win32(h) => z::WindowHandle::Win32(z::Win32WindowHandle {
            hwnd: h.hwnd,
            hinstance: h.hinstance,
        }),
        R::Xcb(h) => z::WindowHandle::Xcb(z::XcbWindowHandle { window: h.window }),
        R::Wayland(h) => z::WindowHandle::Wayland(z::WaylandWindowHandle { surface: h.surface }),
        other => panic!("unsupported window handle: {other:?}"),
    };
    let display = match win.display_handle().expect("display handle").as_raw() {
        D::AppKit(_) => z::DisplayHandle::AppKit,
        D::Windows(_) => z::DisplayHandle::Windows,
        D::Xcb(h) => z::DisplayHandle::Xcb(z::XcbDisplayHandle {
            connection: h.connection,
        }),
        D::Wayland(h) => z::DisplayHandle::Wayland(z::WaylandDisplayHandle { display: h.display }),
        other => panic!("unsupported display handle: {other:?}"),
    };
    WindowHandles::from_raw(window, display)
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex3d {
    pos: [f32; 3],
    color: [f32; 3],
}

fn cube_vertices() -> [Vertex3d; 8] {
    let v = |x: f32, y: f32, z: f32| Vertex3d {
        pos: [x, y, z],
        color: [x * 0.5 + 0.5, y * 0.5 + 0.5, z * 0.5 + 0.5],
    };
    [
        v(-1.0, -1.0, -1.0),
        v(1.0, -1.0, -1.0),
        v(1.0, 1.0, -1.0),
        v(-1.0, 1.0, -1.0),
        v(-1.0, -1.0, 1.0),
        v(1.0, -1.0, 1.0),
        v(1.0, 1.0, 1.0),
        v(-1.0, 1.0, 1.0),
    ]
}

/// 36 indices, each face wound CCW as seen from outside (right-handed coords).
#[rustfmt::skip]
const CUBE_INDICES: [u32; 36] = [
    4, 5, 6,  4, 6, 7, // +Z front
    1, 0, 3,  1, 3, 2, // -Z back
    0, 4, 7,  0, 7, 3, // -X left
    5, 1, 2,  5, 2, 6, // +X right
    3, 7, 6,  3, 6, 2, // +Y top
    0, 1, 5,  0, 5, 4, // -Y bottom
];

type Mat4 = [f32; 16];

fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

fn identity() -> Mat4 {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = identity();
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

fn rotate_y(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    let mut m = identity();
    m[0] = c;
    m[8] = s;
    m[2] = -s;
    m[10] = c;
    m
}

fn rotate_x(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    let mut m = identity();
    m[5] = c;
    m[9] = -s;
    m[6] = s;
    m[10] = c;
    m
}

/// Standard right-handed perspective for Vulkan NDC.
///
/// Uses a **negative viewport** (`y = H, height = -H`) to flip Y in the
/// rasterizer, so this matrix uses the natural `+f` (no manual Y-flip) and
/// CCW winding stays CCW on screen.  Depth range: `0..1` (Vulkan).
fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy * 0.5).tan();
    let mut m = [0.0f32; 16];
    m[0] = f / aspect;
    m[5] = f; // no Y-flip here; viewport flip handles Vulkan's +Y-down NDC
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = (far * near) / (near - far);
    m
}

use zengpu::ZslShader;

const VERT_ZSL: ZslShader = zsl!(
    push P { mvp: mat4x4<f32> }
    vertex vs(@location(0) in_pos: f32x3, @location(1) in_color: f32x3, p: P) -> (f32x4, f32x3) {
        (p.mvp * in_pos.extend(1.0), in_color)
    }
);

const FRAG_ZSL: ZslShader = zsl!(
    fragment fs(@location(0) v_color: f32x3) -> f32x4 {
        v_color.extend(1.0)
    }
);

fn spv_bytes(words: &[u32]) -> &[u8] {
    unsafe { from_raw_parts(words.as_ptr() as *const u8, size_of_val(words)) }
}

fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { from_raw_parts(slice.as_ptr() as *const u8, size_of_val(slice)) }
}

struct CubeRenderState {
    surface: VulkanSurface,
    pipeline: PipelineHandle,
    vertex_buffer: BufferHandle,
    index_buffer: BufferHandle,
    depth: DepthTarget,
    depth_target: TargetHandle,
}

impl CubeRenderState {
    fn new(device: &VulkanDevice, window: &Window) -> zengpu::Result<Self> {
        let size = window.inner_size();
        let config = SurfaceConfig {
            format: Format::Bgra8Unorm,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::Mailbox,
        };
        let surface = device.create_surface(&zen_handles(window), config)?;
        let vertex_shader = device.create_shader(ShaderDesc::spirv(spv_bytes(VERT_ZSL.spv)))?;
        let fragment_shader = device.create_shader(ShaderDesc::spirv(spv_bytes(FRAG_ZSL.spv)))?;
        let pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc {
            vertex_shader,
            fragment_shader,
            vertex_layouts: &[VertexLayout {
                stride: size_of::<Vertex3d>() as u32,
                attributes: &[
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x3,
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 12,
                        format: VertexFormat::Float32x3,
                    },
                ],
                ..Default::default()
            }],
            topology: PrimitiveTopology::TriangleList,
            color_targets: &[ColorTargetState {
                format: config.format,
                blend: None,
            }],
            depth_format: Some(Format::Depth32Float),
            depth: DepthState {
                test: true,
                write: true,
                compare: CompareFn::default(),
            },
            raster: RasterState::default(),
            samples: 1,
        })?;

        let vertices = cube_vertices();
        let vertex_bytes = as_bytes(&vertices);
        let vertex_buffer = device.create_buffer(BufferDesc {
            size: vertex_bytes.len() as u64,
            usage: BufferUsage::VERTEX,
            memory: MemoryUsage::Upload,
        })?;
        device.write_buffer(vertex_buffer, 0, vertex_bytes)?;
        let index_bytes = as_bytes(&CUBE_INDICES);
        let index_buffer = device.create_buffer(BufferDesc {
            size: index_bytes.len() as u64,
            usage: BufferUsage::INDEX,
            memory: MemoryUsage::Upload,
        })?;
        device.write_buffer(index_buffer, 0, index_bytes)?;

        let depth = DepthTarget::new(&device.context(), config.width, config.height)?;
        let depth_target = device.register_depth_target(&depth);
        Ok(Self {
            surface,
            pipeline,
            vertex_buffer,
            index_buffer,
            depth,
            depth_target,
        })
    }

    fn ensure_depth_size(&mut self, device: &VulkanDevice) -> zengpu::Result<(u32, u32)> {
        let (width, height) = self.surface.size();
        if self.depth.extent() != (width, height) {
            device.unregister_render_target(self.depth_target);
            self.depth = DepthTarget::new(&device.context(), width.max(1), height.max(1))?;
            self.depth_target = device.register_depth_target(&self.depth);
        }
        Ok((width, height))
    }
}

struct App {
    render: Option<CubeRenderState>,
    device: Option<VulkanDevice>,
    _instance: Option<VulkanInstance>,
    window: Option<Window>,
    start: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            render: None,
            device: None,
            _instance: None,
            window: None,
            start: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(Window::default_attributes().with_title("ZenGPU — 3D cube"))
            .expect("create window");
        let instance = VulkanInstance::new_with_surface().expect("vulkan instance");
        let adapter = instance
            .request_vulkan_adapter()
            .expect("no vulkan adapter");
        let device = adapter
            .open_with_surface(DeviceRequest::default())
            .expect("open device");
        let render = CubeRenderState::new(&device, &window).expect("create cube renderer");
        self.render = Some(render);
        self.device = Some(device);
        self._instance = Some(instance);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(render) = &self.render {
                    let _ = render.surface.resize(size.width.max(1), size.height.max(1));
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(render), Some(device), Some(window)) =
                    (&mut self.render, &self.device, &self.window)
                {
                    let Ok((width, height)) = render.ensure_depth_size(device) else {
                        window.request_redraw();
                        return;
                    };
                    let frame = match render.surface.acquire() {
                        Ok(Acquire::Frame(frame)) => frame,
                        Ok(Acquire::Skip) | Err(_) => {
                            window.request_redraw();
                            return;
                        }
                    };
                    let t = self.start.elapsed().as_secs_f32();
                    let model = mat_mul(&rotate_y(t * 0.6), &rotate_x(t * 0.3));
                    let view = translate(0.0, 0.0, -5.0);
                    let proj = perspective(
                        60f32.to_radians(),
                        width as f32 / height.max(1) as f32,
                        0.1,
                        100.0,
                    );
                    let mvp = mat_mul(&proj, &mat_mul(&view, &model));
                    let scalars: [Scalar; 16] = from_fn(|i| Scalar::F32(mvp[i]));
                    if let Ok(mut list) = device.create_command_list() {
                        list.begin_render_pass(&RenderPassDesc {
                            color: &[ColorAttachment {
                                target: frame.target(),
                                load: LoadOp::clear_rgb(0.02, 0.02, 0.05),
                                store: true,
                                sample_after: false,
                            }],
                            depth: Some(DepthAttachment {
                                target: render.depth_target,
                                load: LoadOp::clear_depth(1.0),
                                store: false,
                            }),
                        });
                        list.set_pipeline(render.pipeline);
                        list.set_viewport_scissor(ViewportScissor {
                            viewport: Viewport {
                                x: 0.0,
                                y: height as f32,
                                width: width as f32,
                                height: -(height as f32),
                                min_depth: 0.0,
                                max_depth: 1.0,
                            },
                            scissor: Some(Rect {
                                x: 0.0,
                                y: 0.0,
                                width: width as f32,
                                height: height as f32,
                            }),
                        });
                        list.bind(Bindings {
                            scalars: &scalars,
                            ..Default::default()
                        });
                        list.set_vertex_buffer(0, render.vertex_buffer);
                        list.set_index_buffer(render.index_buffer);
                        list.draw_indexed(0..CUBE_INDICES.len() as u32, 0..1);
                        list.end_render_pass();
                        let _ = render.surface.present(frame, list);
                    }
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).expect("run app");
}
