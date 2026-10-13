//! ZenGPU Apple Metal backend: graphics and compute on macOS.
//!
//! Shaders come from ZSL (`zsl!(..).msl`). The backend binds what ZSL's MSL
//! expects: push constants at `[[buffer(16)]]` and the bindless texture table
//! at `[[buffer(17)]]` for graphics, buffers then push constants for compute.
//! Vertex shaders flip clip-space y, so a ZSL shader draws the same way here
//! as on Vulkan.
//!
//! On other platforms the crate builds, finds no adapters, and every call
//! fails, so it can be enabled everywhere.

#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
pub use mac::{
    DepthTarget, MetalAdapter, MetalCommandList, MetalDevice, MetalFrame, MetalInstance,
    MetalSurface, OffscreenTarget,
};

#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(not(target_os = "macos"))]
pub use unsupported::{MetalDevice, MetalInstance};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use std::mem::size_of_val;
    #[cfg(target_os = "macos")]
    use std::slice::from_raw_parts;
    use zengpu_hal::GpuInstance;
    #[cfg(target_os = "macos")]
    use zengpu_hal::{
        AdapterRequest, BufferDesc, BufferHandle, DeviceRequest, GpuDevice, Scalar, ShaderDesc,
    };

    #[cfg(target_os = "macos")]
    fn open_metal() -> Option<Box<dyn GpuDevice>> {
        let inst = MetalInstance::new();
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    #[cfg(target_os = "macos")]
    fn make_f32(device: &dyn GpuDevice, data: &[f32], readback: bool) -> BufferHandle {
        use zengpu_hal::{BufferUsage, MemoryUsage};
        let usage = if readback {
            BufferUsage::STORAGE | BufferUsage::READBACK
        } else {
            BufferUsage::STORAGE
        };
        let buf = device
            .create_buffer(BufferDesc {
                size: (data.len() * 4) as u64,
                usage,
                memory: MemoryUsage::Upload,
            })
            .unwrap();
        device.write_buffer(buf, 0, bytemuck_cast(data)).unwrap();
        buf
    }

    #[cfg(target_os = "macos")]
    fn read_f32(device: &dyn GpuDevice, buf: BufferHandle, n: usize) -> Vec<f32> {
        let bytes = device.read_buffer(buf, 0, (n * 4) as u64).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn instance_constructs_without_panic() {
        let inst = MetalInstance::new();
        let _ = inst.enumerate_adapters();
    }

    /// saxpy: `y = y + alpha*x` — exercises 2 buffers + a 2-field push struct
    /// (`n`, `alpha`) and the scalar push param actually being used in the kernel.
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_saxpy() {
        use zengpu_hal::{Bindings, ComputePipelineDesc};
        use zengpu_spirv::{ZslShader, zsl};

        const SAXPY: ZslShader = zsl!(
            push P { n: u32, alpha: f32 }
            @workgroup_size(64)
            kernel saxpy(x: device buffer<f32>, y: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n { y[i] = y[i] + p.alpha * x[i] }
            }
        );
        let Some(dev) = open_metal() else { return };
        let n = 10usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
        let bx = make_f32(&*dev, &x, false);
        let by = make_f32(&*dev, &y, true);
        let sh = dev.create_shader(ShaderDesc::msl(SAXPY.msl)).unwrap();
        let p = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry: "zsl_main",
                block: [64, 1, 1],
            })
            .unwrap();
        let groups = n.div_ceil(64) as u32;
        dev.dispatch(
            p,
            Bindings {
                buffers: &[bx.index(), by.index()],
                textures: &[],
                scalars: &[Scalar::U32(n as u32), Scalar::F32(2.0)],
            },
            [groups, 1, 1],
        )
        .unwrap();
        let out = read_f32(&*dev, by, n);
        let expect: Vec<f32> = (0..n).map(|i| (i * 10) as f32 + 2.0 * i as f32).collect();
        assert_eq!(out, expect);
    }

    /// Large `vec_add` over 4 threadgroups — exercises the threadgroup-count math
    /// and the `i < n` bounds guard on the tail group. `a[i]+b[i] = i+(n-i) = n`.
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_vec_add_multi_threadgroup() {
        use zengpu_hal::{Bindings, ComputePipelineDesc};
        use zengpu_spirv::{ZslShader, zsl};

        const ADD: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(256)
            kernel add(a: device buffer<f32>, b: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n { out[i] = a[i] + b[i] }
            }
        );
        let Some(dev) = open_metal() else { return };
        let n = 1000usize; // not a multiple of 256 → tail group hits the guard
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let ba = make_f32(&*dev, &a, false);
        let bb = make_f32(&*dev, &b, false);
        let bout = make_f32(&*dev, &vec![-1.0f32; n], true);
        let sh = dev.create_shader(ShaderDesc::msl(ADD.msl)).unwrap();
        let p = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry: "zsl_main",
                block: [256, 1, 1],
            })
            .unwrap();
        let groups = n.div_ceil(256) as u32; // 4
        dev.dispatch(
            p,
            Bindings {
                buffers: &[ba.index(), bb.index(), bout.index()],
                textures: &[],
                scalars: &[Scalar::U32(n as u32)],
            },
            [groups, 1, 1],
        )
        .unwrap();
        let out = read_f32(&*dev, bout, n);
        assert!(
            out.iter().all(|&v| (v - n as f32).abs() < 1e-3),
            "every element should equal n={n}; got e.g. {}",
            out[0]
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn enumerates_at_least_one_adapter_on_macos() {
        let inst = MetalInstance::new();
        let adapters = inst.enumerate_adapters();
        assert!(
            !adapters.is_empty(),
            "expected at least one Metal adapter on macOS"
        );
        for a in &adapters {
            assert!(!a.info().name.is_empty());
            assert!(a.capabilities().graphics);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn buffer_write_read_round_trip() {
        let inst = MetalInstance::new();
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return; // no Metal device in this environment
        };
        let device = adapter
            .open(DeviceRequest::default())
            .expect("open MTLDevice");

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes = bytemuck_cast(&data);
        let buf = device
            .create_buffer(BufferDesc {
                size: bytes.len() as u64,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::Upload,
            })
            .expect("create buffer");
        device.write_buffer(buf, 0, bytes).expect("write");
        let out = device
            .read_buffer(buf, 0, bytes.len() as u64)
            .expect("read");
        assert_eq!(out, bytes);

        // Out-of-bounds write is rejected.
        assert!(
            device
                .write_buffer(buf, bytes.len() as u64, &[0u8; 4])
                .is_err()
        );
        // An offset that wraps when the length is added is rejected too,
        // rather than writing past the buffer.
        assert!(device.write_buffer(buf, u64::MAX - 1, &[0u8; 4]).is_err());
        assert!(device.read_buffer(buf, u64::MAX - 1, 4).is_err());
        device.destroy_buffer(buf);
    }

    #[cfg(target_os = "macos")]
    fn bytemuck_cast(data: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding/invalid bit patterns; viewing as bytes is sound.
        unsafe { from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn vec_add_compute_on_metal() {
        use zengpu_hal::{Bindings, BufferUsage, ComputePipelineDesc, MemoryUsage};
        use zengpu_spirv::{ZslShader, zsl};

        const ADD: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(256)
            kernel add(
                a: device buffer<f32>,
                b: device buffer<f32>,
                out: device mut buffer<f32>,
                p: P,
                id: global_id,
            ) {
                let i = id.x
                if i < p.n {
                    out[i] = a[i] + b[i]
                }
            }
        );

        let inst = MetalInstance::new();
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let device = adapter.open(DeviceRequest::default()).expect("open");

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();
        let bytes = (n * 4) as u64;

        let mk = |usage| {
            device
                .create_buffer(BufferDesc {
                    size: bytes,
                    usage,
                    memory: MemoryUsage::Upload,
                })
                .expect("buffer")
        };
        let ba = mk(BufferUsage::STORAGE);
        let bb = mk(BufferUsage::STORAGE);
        let bout = mk(BufferUsage::STORAGE | BufferUsage::READBACK);
        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

        let shader = device
            .create_shader(ShaderDesc::msl(ADD.msl))
            .expect("shader");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "zsl_main",
                block: [256, 1, 1],
            })
            .expect("pipeline");

        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[ba.index(), bb.index(), bout.index()],
                    textures: &[],
                    scalars: &[Scalar::U32(n as u32)],
                },
                [1, 1, 1],
            )
            .expect("dispatch");

        let out = device.read_buffer(bout, 0, bytes).expect("read");
        let result: &[f32] = unsafe { from_raw_parts(out.as_ptr() as *const f32, n) };
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }
}

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod graphics_tests {
    //! The Metal graphics path through the HAL, offscreen: what a ZSL shader
    //! draws here must match what it draws on Vulkan.

    use std::mem::size_of_val;
    use std::slice::from_raw_parts;

    use crate::{MetalDevice, MetalInstance};
    use zengpu_hal::{
        Bindings, BlendState, BufferDesc, BufferHandle, BufferUsage, ColorAttachment,
        ColorTargetState, CompareFn, DepthAttachment, DepthState, DeviceRequest, DeviceType,
        FilterMode, Format, GpuDevice, GraphicsDevice, GraphicsPipelineDesc, LoadOp, MemoryUsage,
        PipelineHandle, PrimitiveTopology, RasterState, RenderCommands, RenderPassDesc,
        SamplerDesc, Scalar, ShaderDesc, StepMode, TargetHandle, TexDim, TextureDesc,
        TextureHandle, TextureUsage, VertexAttribute, VertexFormat, VertexLayout, Viewport,
        ViewportScissor,
    };
    use zengpu_spirv::{ZslShader, zsl};

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];

    fn open() -> Option<MetalDevice> {
        MetalInstance::new()
            .request_metal_adapter()?
            .open_metal(DeviceRequest::default())
            .ok()
    }

    fn bytes<T>(data: &[T]) -> &[u8] {
        // SAFETY: only plain f32/u32 arrays are viewed, which have no padding.
        unsafe { from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
    }

    fn texture(
        dev: &MetalDevice,
        format: Format,
        w: u32,
        h: u32,
        usage: TextureUsage,
    ) -> TextureHandle {
        dev.create_texture(TextureDesc {
            width: w,
            height: h,
            depth: 1,
            format,
            usage,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        })
        .expect("texture")
    }

    fn color_target(dev: &MetalDevice, w: u32, h: u32) -> (TextureHandle, TargetHandle) {
        let usage =
            TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::TRANSFER_SRC;
        let tex = texture(dev, Format::Rgba8Unorm, w, h, usage);
        let target = dev.register_color_target(tex).expect("target");
        (tex, target)
    }

    fn buffer(dev: &MetalDevice, data: &[u8], usage: BufferUsage) -> BufferHandle {
        let b = dev
            .create_buffer(BufferDesc {
                size: data.len() as u64,
                usage,
                memory: MemoryUsage::Upload,
            })
            .expect("buffer");
        dev.write_buffer(b, 0, data).expect("write");
        b
    }

    /// Every pixel of `tex`, row by row from the top.
    fn pixels(dev: &MetalDevice, tex: TextureHandle, w: u32, h: u32) -> Vec<[u8; 4]> {
        let len = u64::from(w * h * 4);
        let buf = dev
            .create_buffer(BufferDesc {
                size: len,
                usage: BufferUsage::READBACK,
                memory: MemoryUsage::Readback,
            })
            .expect("readback buffer");
        dev.copy_texture_to_buffer(tex, buf).expect("copy");
        let raw = dev.read_buffer(buf, 0, len).expect("read");
        raw.chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }

    fn pipeline(
        dev: &MetalDevice,
        vs: &ZslShader,
        fs: &ZslShader,
        layouts: &[VertexLayout<'_>],
        blend: Option<BlendState>,
        depth: Option<DepthState>,
    ) -> PipelineHandle {
        let v = dev
            .create_shader(ShaderDesc::msl(vs.msl))
            .expect("vertex shader");
        let f = dev
            .create_shader(ShaderDesc::msl(fs.msl))
            .expect("fragment shader");
        let p = dev
            .create_graphics_pipeline(GraphicsPipelineDesc {
                vertex_shader: v,
                fragment_shader: f,
                vertex_layouts: layouts,
                topology: PrimitiveTopology::TriangleList,
                color_targets: &[ColorTargetState {
                    format: Format::Rgba8Unorm,
                    blend,
                }],
                depth_format: depth.map(|_| Format::Depth32Float),
                depth: depth.unwrap_or_default(),
                stencil: None,
                raster: RasterState::default(),
                samples: 1,
            })
            .expect("pipeline");
        dev.destroy_shader(v);
        dev.destroy_shader(f);
        p
    }

    fn pass<'a>(
        target: TargetHandle,
        load: LoadOp,
        depth: Option<DepthAttachment>,
        color: &'a mut [ColorAttachment; 1],
    ) -> RenderPassDesc<'a> {
        color[0] = ColorAttachment {
            target,
            load,
            store: true,
            sample_after: false,
        };
        RenderPassDesc {
            color: &color[..],
            depth,
        }
    }

    fn viewport(w: u32, h: u32) -> ViewportScissor {
        ViewportScissor {
            viewport: Viewport {
                x: 0.0,
                y: 0.0,
                width: w as f32,
                height: h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: None,
        }
    }

    const RED_FS: ZslShader = zsl!(
        fragment fs() -> f32x4 { f32x4(1.0, 0.0, 0.0, 1.0) }
    );

    /// ZSL clip space has +y down, as on Vulkan: y in -1..0 is the top half.
    #[test]
    fn clip_space_y_points_down() {
        const TOP_HALF: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> f32x4 {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                f32x4(corner.x * 2.0 - 1.0, corner.y - 1.0, 0.0, 1.0)
            }
        );
        let Some(dev) = open() else { return };
        let (tex, target) = color_target(&dev, 4, 4);
        let p = pipeline(&dev, &TOP_HALF, &RED_FS, &[], None, None);
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 0.0),
            None,
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(4, 4));
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        let px = pixels(&dev, tex, 4, 4);
        for (i, p) in px.iter().enumerate() {
            let want = if i / 4 < 2 { RED } else { BLACK };
            assert_eq!(*p, want, "pixel {i} (row {})", i / 4);
        }
    }

    /// A texture bound in the table and picked by a push-constant slot reads
    /// back in place: the top-left texel lands top left.
    #[test]
    fn samples_the_texture_table_through_push_constants() {
        const QUAD_VS: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> (f32x4, f32x2) {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                (f32x4(corner * 2.0 - 1.0, 0.0, 1.0), corner)
            }
        );
        const SAMPLE_FS: ZslShader = zsl!(
            push P { slot: u32 }
            fragment fs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
                tex[p.slot].sample(uv)
            }
        );
        let Some(dev) = open() else { return };
        let texels: [[u8; 4]; 4] = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 255, 255],
        ];
        let src = texture(
            &dev,
            Format::Rgba8Unorm,
            2,
            2,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        );
        dev.upload_texture_data(src, texels.as_flattened())
            .expect("upload");
        let sampler = dev
            .create_sampler(SamplerDesc {
                min_filter: FilterMode::Nearest,
                mag_filter: FilterMode::Nearest,
                mip_filter: FilterMode::Nearest,
                ..SamplerDesc::default()
            })
            .expect("sampler");
        let slot = dev.bind_texture(src, sampler).expect("slot");

        let (tex, target) = color_target(&dev, 2, 2);
        let p = pipeline(&dev, &QUAD_VS, &SAMPLE_FS, &[], None, None);
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 0.0),
            None,
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(2, 2));
        cmd.bind(Bindings {
            buffers: &[],
            textures: &[],
            scalars: &[Scalar::U32(slot)],
        });
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        assert_eq!(pixels(&dev, tex, 2, 2), texels.to_vec());
    }

    /// The push block is packed as on Vulkan, scalars then texture indices, and
    /// padded out to the MSL struct's size: a `float2` then a slot from
    /// `Bindings::textures` is a 16-byte struct fed from 12 bytes.
    #[test]
    fn texture_indices_follow_scalars_in_the_push_block() {
        const QUAD_VS: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> (f32x4, f32x2) {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                (f32x4(corner * 2.0 - 1.0, 0.0, 1.0), corner)
            }
        );
        const TINTED_FS: ZslShader = zsl!(
            push P { tint: f32x2, slot: u32 }
            fragment fs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
                tex[p.slot].sample(uv) * f32x4(p.tint.x, p.tint.y, 1.0, 1.0)
            }
        );
        let Some(dev) = open() else { return };
        let src = texture(&dev, Format::Rgba8Unorm, 1, 1, TextureUsage::SAMPLED);
        dev.upload_texture_data(src, &[255, 255, 255, 255])
            .expect("upload");
        let sampler = dev.create_sampler(SamplerDesc::default()).expect("sampler");
        let slot = dev.bind_texture(src, sampler).expect("slot");
        let (tex, target) = color_target(&dev, 2, 2);
        let p = pipeline(&dev, &QUAD_VS, &TINTED_FS, &[], None, None);
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 0.0),
            None,
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(2, 2));
        cmd.bind(Bindings {
            buffers: &[],
            textures: &[slot],
            scalars: &[Scalar::F32(1.0), Scalar::F32(0.0)],
        });
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        assert!(
            pixels(&dev, tex, 2, 2)
                .iter()
                .all(|p| *p == [255, 0, 255, 255])
        );
    }

    /// Indexed draws honour the instance range: two instances, each covering
    /// half the target from a per-instance vertex buffer, fill all of it.
    #[test]
    fn indexed_draws_draw_every_instance() {
        const HALF_VS: ZslShader = zsl!(
            vertex vs(@location(0) left: f32, vi: vertex_index) -> f32x4 {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                f32x4(left + corner.x, corner.y * 2.0 - 1.0, 0.0, 1.0)
            }
        );
        let Some(dev) = open() else { return };
        let (tex, target) = color_target(&dev, 4, 2);
        let attrs = [VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32,
        }];
        let layouts = [VertexLayout {
            stride: 4,
            attributes: &attrs,
            step_mode: StepMode::Instance,
        }];
        let p = pipeline(&dev, &HALF_VS, &RED_FS, &layouts, None, None);
        let lefts = buffer(&dev, bytes(&[-1.0f32, 0.0]), BufferUsage::VERTEX);
        let indices = buffer(&dev, bytes(&[0u32, 1, 2, 3, 4, 5]), BufferUsage::INDEX);
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 0.0),
            None,
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(4, 2));
        cmd.set_vertex_buffer(0, lefts);
        cmd.set_index_buffer(indices);
        cmd.draw_indexed(0..6, 0..2);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        assert!(pixels(&dev, tex, 4, 2).iter().all(|p| *p == RED));
    }

    #[test]
    fn alpha_blends_over_the_clear_colour() {
        const FULL_VS: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> f32x4 {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                f32x4(corner * 2.0 - 1.0, 0.0, 1.0)
            }
        );
        const HALF_RED_FS: ZslShader = zsl!(
            fragment fs() -> f32x4 { f32x4(1.0, 0.0, 0.0, 0.5) }
        );
        let Some(dev) = open() else { return };
        let (tex, target) = color_target(&dev, 2, 2);
        let p = pipeline(
            &dev,
            &FULL_VS,
            &HALF_RED_FS,
            &[],
            Some(BlendState::ALPHA_BLEND),
            None,
        );
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 1.0),
            None,
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(2, 2));
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        for p in pixels(&dev, tex, 2, 2) {
            assert!(
                p[0].abs_diff(128) <= 1 && p[1] == 0 && p[2].abs_diff(127) <= 1,
                "{p:?}"
            );
        }
    }

    #[test]
    fn depth_test_keeps_the_nearest() {
        const DEPTH_VS: ZslShader = zsl!(
            push P { z: f32, r: f32, g: f32, b: f32 }
            vertex vs(vi: vertex_index, p: P) -> (f32x4, f32x4) {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                (f32x4(corner * 2.0 - 1.0, p.z, 1.0), f32x4(p.r, p.g, p.b, 1.0))
            }
        );
        const COLOR_FS: ZslShader = zsl!(
            fragment fs(@location(0) c: f32x4) -> f32x4 { c }
        );
        let Some(dev) = open() else { return };
        let (tex, target) = color_target(&dev, 2, 2);
        let depth_buffer = dev.create_depth_target(2, 2).expect("depth buffer");
        let depth_target = dev.register_depth_target(&depth_buffer);
        let depth = DepthState {
            test: true,
            write: true,
            compare: CompareFn::Less,
        };
        let p = pipeline(&dev, &DEPTH_VS, &COLOR_FS, &[], None, Some(depth));
        let mut cmd = dev.create_command_list().expect("list");
        let mut color = [ColorAttachment {
            target,
            load: LoadOp::DontCare,
            store: true,
            sample_after: false,
        }];
        let depth_att = DepthAttachment {
            target: depth_target,
            load: LoadOp::clear_depth(1.0),
            store: false,
        };
        cmd.begin_render_pass(&pass(
            target,
            LoadOp::clear_rgb(0.0, 0.0, 0.0),
            Some(depth_att),
            &mut color,
        ));
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(viewport(2, 2));
        for (z, rgb) in [
            (0.8, [1.0, 0.0, 0.0]),
            (0.2, [0.0, 1.0, 0.0]),
            (0.5, [0.0, 0.0, 1.0]),
        ] {
            let scalars = [
                Scalar::F32(z),
                Scalar::F32(rgb[0]),
                Scalar::F32(rgb[1]),
                Scalar::F32(rgb[2]),
            ];
            cmd.bind(Bindings {
                buffers: &[],
                textures: &[],
                scalars: &scalars,
            });
            cmd.draw(0..6, 0..1);
        }
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        assert!(
            pixels(&dev, tex, 2, 2)
                .iter()
                .all(|p| *p == [0, 255, 0, 255])
        );
    }

    #[test]
    fn adapters_keep_their_uuid_and_open_themselves() {
        let inst = MetalInstance::new();
        let Some(first) = inst.metal_adapters().into_iter().next() else {
            return;
        };
        let again = inst
            .metal_adapter_by_uuid(first.uuid())
            .expect("same adapter by uuid");
        assert_eq!(again.uuid(), first.uuid());
        let dev = again.open_metal(DeviceRequest::default()).expect("open");
        // Apple GPUs share memory with the CPU.
        if dev.raw_device().has_unified_memory() {
            use zengpu_hal::GpuAdapter;
            assert_eq!(first.info().device_type, DeviceType::Integrated);
        }
    }
}
