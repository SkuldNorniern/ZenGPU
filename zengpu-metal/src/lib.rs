//! ZenGPU Apple Metal backend: graphics and compute on macOS.
//!
//! Shaders come from ZSL (`zsl!(..).msl`). Bindings follow ZSL's MSL: push constants at
//! `[[buffer(16)]]`, the bindless texture table at `[[buffer(17)]]` for graphics; buffers then push
//! constants for compute. Vertex shaders flip clip-space y, so a ZSL shader draws as on Vulkan.
//!
//! Other platforms build, find no adapters, and every call fails.

#[cfg(target_os = "macos")]
mod mac;
#[cfg(any(target_os = "macos", test))]
mod transfer;
#[cfg(target_os = "macos")]
pub use mac::{
    DepthTarget, MetalAdapter, MetalCommandList, MetalDevice, MetalFrame, MetalInstance,
    MetalSurface, OffscreenTarget,
};

#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(not(target_os = "macos"))]
pub use unsupported::{MetalDevice, MetalInstance};

#[cfg(all(test, target_os = "macos"))]
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Bindings for saxpy: two buffers and a push struct with `n` and `alpha`.
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
        let sh = dev
            .create_shader(ShaderDesc::msl(SAXPY.msl.unwrap()))
            .unwrap();
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
                images: &[],
                scalars: &[Scalar::U32(n as u32), Scalar::F32(2.0)],
            },
            [groups, 1, 1],
        )
        .unwrap();
        let out = read_f32(&*dev, by, n);
        let expect: Vec<f32> = (0..n).map(|i| (i * 10) as f32 + 2.0 * i as f32).collect();
        assert_eq!(out, expect);
    }

    /// Metal has 31 buffer slots and the push block takes the one after the buffers: binding
    /// past that is an error, not an out-of-range `setBuffer`.
    #[test]
    #[cfg(target_os = "macos")]
    fn too_many_bindings_is_an_error() {
        use zengpu_hal::{Bindings, ComputePipelineDesc};
        use zengpu_spirv::{ZslShader, zsl};

        const FILL: ZslShader = zsl!(
            push P { v: f32 }
            @workgroup_size(1)
            kernel fill(y: device mut buffer<f32>, p: P) {
                y[0] = p.v
            }
        );
        let Some(dev) = open_metal() else { return };
        let y = make_f32(&*dev, &[0.0], true);
        let sh = dev
            .create_shader(ShaderDesc::msl(FILL.msl.unwrap()))
            .unwrap();
        let p = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry: "zsl_main",
                block: [1, 1, 1],
            })
            .unwrap();
        let bind = |n: usize, scalars: &[Scalar]| {
            let slots = vec![y.index(); n];
            dev.dispatch(
                p,
                Bindings {
                    buffers: &slots,
                    textures: &[],
                    images: &[],
                    scalars,
                },
                [1, 1, 1],
            )
        };
        assert!(bind(1, &[Scalar::F32(1.0)]).is_ok());
        assert!(bind(30, &[Scalar::F32(1.0)]).is_ok());
        assert!(bind(31, &[]).is_ok());
        assert!(bind(31, &[Scalar::F32(1.0)]).is_err());
        assert!(bind(32, &[]).is_err());
        assert!(bind(1, &[Scalar::F32(0.0); 1025]).is_err());
        // An empty grid runs nothing.
        for grid in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
            let b = Bindings {
                buffers: &[y.index()],
                textures: &[],
                images: &[],
                scalars: &[Scalar::F32(5.0)],
            };
            dev.dispatch(p, b, grid).unwrap();
        }
        assert_eq!(read_f32(&*dev, y, 1), vec![1.0]);
    }

    /// Resources Metal cannot create are errors. Metal itself aborts the process on them.
    #[test]
    #[cfg(target_os = "macos")]
    fn impossible_resources_are_errors() {
        use zengpu_hal::{
            BufferUsage, Format, GpuError, MemoryUsage, TexDim, TextureDesc, TextureUsage,
        };
        let Some(dev) = open_metal() else { return };
        let huge = dev.create_buffer(BufferDesc {
            size: 1 << 42,
            usage: BufferUsage::STORAGE,
            memory: MemoryUsage::GpuOnly,
        });
        assert!(matches!(huge, Err(GpuError::OutOfMemory(_))), "{huge:?}");

        let base = TextureDesc {
            width: 16,
            height: 16,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        };
        let bad = [
            TextureDesc {
                width: 1 << 16,
                ..base
            },
            TextureDesc {
                mip_levels: 6,
                ..base
            },
            TextureDesc {
                dimension: TexDim::D3,
                depth: 4096,
                ..base
            },
            TextureDesc {
                array_layers: 4096,
                ..base
            },
            TextureDesc {
                samples: 3,
                usage: TextureUsage::RENDER_TARGET,
                ..base
            },
            TextureDesc {
                samples: 4,
                array_layers: 2,
                usage: TextureUsage::RENDER_TARGET,
                ..base
            },
            TextureDesc {
                dimension: TexDim::Cube,
                height: 8,
                array_layers: 6,
                ..base
            },
        ];
        for desc in bad {
            assert!(dev.create_texture(desc).is_err(), "{desc:?}");
        }
        let full_chain = dev.create_texture(TextureDesc {
            mip_levels: 5,
            ..base
        });
        assert!(full_chain.is_ok(), "{full_chain:?}");
        // The device still works.
        let buf = make_f32(&*dev, &[1.0, 2.0], true);
        assert_eq!(read_f32(&*dev, buf, 2), vec![1.0, 2.0]);
    }

    /// `round_even`, `fast_exp` and the integer forms of `abs`/`min`/`max`/`clamp` compile and
    /// match the host.
    #[test]
    #[cfg(target_os = "macos")]
    fn integer_and_rounding_builtins_match_the_host() {
        use zengpu_hal::{Bindings, BufferUsage, ComputePipelineDesc, MemoryUsage};
        use zengpu_spirv::{ZslShader, zsl};

        const OPS: ZslShader = zsl!(
            @workgroup_size(64)
            kernel ops(xs: device buffer<i32>, fs: device buffer<f32>, o: device mut buffer<u32>, id: global_id) {
                let i = id.x
                if i < 8 {
                    let x = xs[i]
                    let u = bitcast_u32(x)
                    let k = i * 8
                    o[k] = bitcast_u32(abs(x))
                    o[k + 1] = bitcast_u32(min(x, -3))
                    o[k + 2] = bitcast_u32(max(x, 2))
                    o[k + 3] = bitcast_u32(clamp(x, -5, 5))
                    o[k + 4] = min(u, 7u)
                    o[k + 5] = max(u, 7u)
                    o[k + 6] = bitcast_u32(round_even(fs[i]))
                    o[k + 7] = bitcast_u32(fast_exp(fs[i]))
                }
            }
        );
        let Some(dev) = open_metal() else { return };
        let xs: [i32; 8] = [0, 1, -1, 9, -9, i32::MAX, i32::MIN + 1, 4];
        let fs: [f32; 8] = [0.5, 1.5, 2.5, -0.5, -1.5, 3.4, 0.0, 1.0];
        let mk = |bytes: &[u8]| {
            let h = dev
                .create_buffer(BufferDesc {
                    size: bytes.len() as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                    memory: MemoryUsage::Upload,
                })
                .unwrap();
            dev.write_buffer(h, 0, bytes).unwrap();
            h
        };
        let bx = mk(&xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let bf = mk(&fs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let bo = mk(&[0u8; 8 * 8 * 4]);
        let sh = dev.create_shader(ShaderDesc::msl(OPS.msl.unwrap())).unwrap();
        let p = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry: "zsl_main",
                block: [64, 1, 1],
            })
            .unwrap();
        dev.dispatch(
            p,
            Bindings {
                buffers: &[bx.index(), bf.index(), bo.index()],
                textures: &[],
                images: &[],
                scalars: &[],
            },
            [1, 1, 1],
        )
        .unwrap();
        let out: Vec<u32> = dev
            .read_buffer(bo, 0, 8 * 8 * 4)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        for (i, (&x, &f)) in xs.iter().zip(&fs).enumerate() {
            let u = x as u32;
            let got = &out[i * 8..i * 8 + 8];
            assert_eq!(got[0] as i32, x.abs(), "abs({x})");
            assert_eq!(got[1] as i32, x.min(-3), "min({x})");
            assert_eq!(got[2] as i32, x.max(2), "max({x})");
            assert_eq!(got[3] as i32, x.clamp(-5, 5), "clamp({x})");
            assert_eq!(got[4], u.min(7), "min({u}u)");
            assert_eq!(got[5], u.max(7), "max({u}u)");
            assert_eq!(f32::from_bits(got[6]), f.round_ties_even(), "round_even({f})");
            let e = f32::from_bits(got[7]);
            assert!((e - f.exp()).abs() <= 1e-3 * f.exp(), "fast_exp({f}) = {e}");
        }
    }

    /// `vec_add` over 4 threadgroups: the `i < n` guard covers the tail group.
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
        let n = 1000usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let ba = make_f32(&*dev, &a, false);
        let bb = make_f32(&*dev, &b, false);
        let bout = make_f32(&*dev, &vec![-1.0f32; n], true);
        let sh = dev
            .create_shader(ShaderDesc::msl(ADD.msl.unwrap()))
            .unwrap();
        let p = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry: "zsl_main",
                block: [256, 1, 1],
            })
            .unwrap();
        let groups = n.div_ceil(256) as u32;
        dev.dispatch(
            p,
            Bindings {
                buffers: &[ba.index(), bb.index(), bout.index()],
                textures: &[],
                images: &[],
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
            return;
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

        assert!(
            device
                .write_buffer(buf, bytes.len() as u64, &[0u8; 4])
                .is_err()
        );
        assert!(device.write_buffer(buf, u64::MAX - 1, &[0u8; 4]).is_err());
        assert!(device.read_buffer(buf, u64::MAX - 1, 4).is_err());
        device.destroy_buffer(buf);

        let storage = device
            .create_buffer(BufferDesc {
                size: 4,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create buffer");
        assert!(device.read_buffer(storage, 0, 4).is_err());
        device.destroy_buffer(storage);
    }

    #[cfg(target_os = "macos")]
    fn bytemuck_cast(data: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding/invalid bit patterns; viewing as bytes is sound.
        unsafe { from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
    }

    /// A cube's six faces arrive in one upload, each at its own layer.
    #[cfg(target_os = "macos")]
    #[test]
    fn cube_uploads_land_on_their_faces() {
        use std::time::Duration;
        use zengpu_hal::{
            BufferUsage, Format, MemoryUsage, TexDim, TextureDesc, TextureRegion, TextureUsage,
        };
        let Some(dev) = open_metal() else { return };
        let tex = dev
            .create_texture(TextureDesc {
                width: 2,
                height: 2,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_SRC
                    | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::Cube,
                mip_levels: 1,
                array_layers: 6,
            })
            .unwrap();
        let faces: Vec<u8> = (0..6u8).flat_map(|face| [face * 40 + 1; 16]).collect();
        dev.upload_texture_data(tex, &faces).unwrap();
        let buf = dev
            .create_buffer(BufferDesc {
                size: 16,
                usage: BufferUsage::READBACK | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::Readback,
            })
            .unwrap();
        for layer in 0..6u32 {
            let region = TextureRegion {
                layer,
                ..TextureRegion::rect(0, 0, 2, 2)
            };
            dev.copy_texture_region_to_buffer(tex, region, buf, 0, 8)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            let mut out = [0u8; 16];
            dev.read_buffer_into(buf, 0, &mut out).unwrap();
            assert_eq!(out, [layer as u8 * 40 + 1; 16], "face {layer}");
        }
    }

    /// Each dispatch reads the one before; the final copy sees them all.
    #[test]
    #[cfg(target_os = "macos")]
    fn batched_ops_run_in_order() {
        use std::time::Duration;
        use zengpu_hal::{
            Bindings, BufferCopyOp, BufferUsage, ComputeOp, ComputePipelineDesc, DispatchOp,
            MemoryUsage, SubmissionStatus,
        };
        use zengpu_spirv::{ZslShader, zsl};

        const INC: ZslShader = zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel inc(data: device mut buffer<u32>, p: P, id: global_id) {
                if id.x < p.n {
                    data[id.x] = data[id.x] + 1
                }
            }
        );
        let Some(dev) = open_metal() else { return };
        let shader = dev
            .create_shader(ShaderDesc::msl(INC.msl.unwrap()))
            .unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "zsl_main",
                block: [64, 1, 1],
            })
            .unwrap();
        let n = 1000u32;
        let desc = BufferDesc {
            size: u64::from(n) * 4,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        };
        let (data, copy) = (
            dev.create_buffer(desc).unwrap(),
            dev.create_buffer(desc).unwrap(),
        );
        dev.write_buffer(data, 0, &vec![0; n as usize * 4]).unwrap();
        let scalars = [Scalar::U32(n)];
        let bindings = Bindings {
            buffers: &[data.index()],
            textures: &[],
            images: &[],
            scalars: &scalars,
        };
        let dispatch = DispatchOp {
            pipeline,
            bindings,
            grid: [n.div_ceil(64), 1, 1],
        };
        dev.dispatch_batch(&[dispatch; 100]).unwrap();
        let mut ops = vec![ComputeOp::Dispatch(dispatch); 50];
        ops.push(ComputeOp::CopyBuffer(BufferCopyOp {
            src: data,
            src_offset: 0,
            dst: copy,
            dst_offset: 0,
            len: desc.size,
        }));
        let other = dev.create_buffer(desc).unwrap();
        let sub = dev.submit_compute_ops(7, &ops).unwrap();
        assert_eq!(sub.cycle_id(), 7);
        dev.write_buffer(other, 0, &[1; 4]).unwrap();
        if dev.read_buffer(copy, 0, 4).is_ok() {
            assert_eq!(sub.poll().unwrap(), SubmissionStatus::Complete);
        }
        sub.wait(Duration::from_secs(10)).unwrap();
        let words = |b| {
            dev.read_buffer(b, 0, desc.size)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<_>>()
        };
        assert!(words(data).iter().all(|&v| v == 150));
        assert!(words(copy).iter().all(|&v| v == 150));
        dev.destroy_buffer(data);
        dev.destroy_buffer(copy);
        dev.destroy_buffer(other);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }
}

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod graphics_tests {
    //! Offscreen graphics via the HAL. Output must match Vulkan for the same ZSL shader.

    use std::mem::size_of_val;
    use std::slice::from_raw_parts;

    use crate::{MetalDevice, MetalInstance};
    use zengpu_hal::{
        Bindings, BlendState, BufferDesc, BufferHandle, BufferUsage, ColorAttachment,
        ColorTargetState, CompareFn, DepthAttachment, DepthState, DeviceRequest, FilterMode,
        Format, GpuDevice, GraphicsDevice, GraphicsPipelineDesc, LoadOp, MemoryUsage,
        PipelineHandle, PrimitiveTopology, RasterState, RenderCommands, RenderPassDesc,
        SamplerDesc, Scalar, ShaderDesc, StepMode, TargetHandle, TexDim, TextureDesc,
        TextureHandle, TextureUsage, VertexAttribute, VertexFormat, VertexLayout, Viewport,
        ViewportScissor,
    };
    use zengpu_spirv::{ZslShader, zsl};

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];

    fn open() -> Option<MetalDevice> {
        use std::env;
        use zengpu_hal::GpuAdapter;
        let adapter = MetalInstance::new().request_metal_adapter();
        let dev = adapter.as_ref().and_then(|adapter| {
            eprintln!("Metal regression adapter: {}", adapter.info().name);
            adapter.open_metal(DeviceRequest::default()).ok()
        });
        assert!(
            env::var_os("ZENGPU_REQUIRE_METAL").is_none() || dev.is_some(),
            "GPU-required Metal tests could not open an adapter"
        );
        dev
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

    #[test]
    fn region_transfers_validate_shared_and_private_textures() {
        use std::time::Duration;
        use zengpu_hal::{TextureRegion, TextureWrite};
        let Some(dev) = open() else { return };
        for (format, storage) in [
            (Format::Rgba8Unorm, TextureUsage::SAMPLED),
            (Format::Depth32Float, TextureUsage::DEPTH_STENCIL),
        ] {
            let tex = texture(
                &dev,
                format,
                4,
                4,
                storage | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST,
            );
            let valid = TextureRegion::rect(0, 0, 4, 4);
            assert!(
                dev.write_textures(&[TextureWrite {
                    texture: tex,
                    region: valid,
                    data: &[0; 63],
                    bytes_per_row: 16
                }])
                .is_err()
            );
            for region in [
                TextureRegion {
                    mip_level: 32,
                    ..valid
                },
                TextureRegion { layer: 1, ..valid },
                TextureRegion {
                    x: u32::MAX,
                    ..valid
                },
            ] {
                assert!(
                    dev.write_textures(&[TextureWrite {
                        texture: tex,
                        region,
                        data: &[0; 64],
                        bytes_per_row: 16
                    }])
                    .is_err()
                );
            }
            for pitch in [0, 15, 17] {
                assert!(
                    dev.write_textures(&[TextureWrite {
                        texture: tex,
                        region: valid,
                        data: &[0; 64],
                        bytes_per_row: pitch
                    }])
                    .is_err()
                );
            }
            let invalid_buffer = buffer(&dev, &[0; 64], BufferUsage::STORAGE);
            assert!(
                dev.copy_buffer_to_texture(invalid_buffer, 0, 16, tex, valid)
                    .is_err()
            );
            assert!(
                dev.copy_texture_region_to_buffer(tex, valid, invalid_buffer, 0, 16)
                    .is_err()
            );
            let upload = buffer(&dev, &[0; 64], BufferUsage::TRANSFER_SRC);
            assert!(
                dev.copy_buffer_to_texture(upload, 4, 16, tex, valid)
                    .is_err()
            );
            let readback = buffer(
                &dev,
                &[0; 64],
                BufferUsage::READBACK | BufferUsage::TRANSFER_DST,
            );
            assert!(
                dev.copy_texture_region_to_buffer(tex, valid, readback, u64::MAX - 3, 16)
                    .is_err()
            );
            for byte in 1..=16 {
                let data = [byte; 64];
                let wrote = dev
                    .write_textures(&[TextureWrite {
                        texture: tex,
                        region: valid,
                        data: &data,
                        bytes_per_row: 16,
                    }])
                    .unwrap();
                let read = dev
                    .copy_texture_region_to_buffer(tex, valid, readback, 0, 16)
                    .unwrap();
                read.wait(Duration::MAX).unwrap();
                wrote.wait(Duration::MAX).unwrap();
                assert_eq!(dev.read_buffer(readback, 0, 64).unwrap(), data);
            }
            let packed = [42; 64];
            let padded = [77; 76];
            dev.write_textures(&[
                TextureWrite {
                    texture: tex,
                    region: valid,
                    data: &packed,
                    bytes_per_row: 16,
                },
                TextureWrite {
                    texture: tex,
                    region: valid,
                    data: &padded,
                    bytes_per_row: 20,
                },
            ])
            .unwrap()
            .wait(Duration::MAX)
            .unwrap();
            assert!(
                dev.write_textures(&[
                    TextureWrite {
                        texture: tex,
                        region: valid,
                        data: &packed,
                        bytes_per_row: 16
                    },
                    TextureWrite {
                        texture: tex,
                        region: valid,
                        data: &[0; 63],
                        bytes_per_row: 16
                    },
                ])
                .is_err()
            );
            dev.copy_texture_region_to_buffer(tex, valid, readback, 0, 16)
                .unwrap()
                .wait(Duration::MAX)
                .unwrap();
            assert_eq!(dev.read_buffer(readback, 0, 64).unwrap(), [77; 64]);
        }
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

    /// Returns every pixel of `tex`, row by row from the top.
    fn pixels(dev: &MetalDevice, tex: TextureHandle, w: u32, h: u32) -> Vec<[u8; 4]> {
        let len = u64::from(w * h * 4);
        let buf = dev
            .create_buffer(BufferDesc {
                size: len,
                usage: BufferUsage::READBACK | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::Readback,
            })
            .expect("readback buffer");
        dev.copy_texture_to_buffer(tex, buf).expect("copy");
        let raw = dev.read_buffer(buf, 0, len).expect("read");
        raw.as_chunks::<4>()
            .0
            .iter()
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
            .create_shader(ShaderDesc::msl(vs.msl.unwrap()))
            .expect("vertex shader");
        let f = dev
            .create_shader(ShaderDesc::msl(fs.msl.unwrap()))
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

    /// Texture picked by a push slot: the top-left texel lands at the top left.
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
            images: &[],
            scalars: &[Scalar::U32(slot)],
        });
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        assert_eq!(pixels(&dev, tex, 2, 2), texels.to_vec());
    }

    /// Push block packed as on Vulkan, padded to the MSL struct size.
    ///
    /// A `float2` followed by a texture slot is a 16-byte struct fed from 12 bytes.
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
        let src = texture(
            &dev,
            Format::Rgba8Unorm,
            1,
            1,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        );
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
            images: &[],
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

    /// Indexed draws honour the instance range: two instances fill the target.
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
                images: &[],
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

    /// Metal twin of zengpu-vulkan's `compute_chains_texture_passes_in_one_submission`.
    #[test]
    fn compute_chains_texture_passes() {
        use zengpu_hal::{ComputeOp, ComputePipelineDesc, DispatchOp};

        const INVERT: ZslShader = zsl!(
            push P { w: u32, h: u32 }
            @workgroup_size(8, 8)
            kernel invert(src: texture, dst: image, p: P, id: global_id) {
                let x = id.x
                let y = id.y
                if x < p.w && y < p.h {
                    dst.store(x, y, 1.0 - src.load(x, y).r, 1.0 - src.load(x, y).g, 1.0 - src.load(x, y).b, src.load(x, y).a)
                }
            }
        );
        const COMBINE: ZslShader = zsl!(
            @workgroup_size(8, 8)
            kernel combine(inv: image, smooth: texture, out: image, id: global_id) {
                let x = id.x
                let y = id.y
                if x < out.width() && y < out.height() {
                    let u = (x + 0.5) / out.width()
                    let v = (y + 0.5) / out.height()
                    let r = (inv.load(x, y).r + smooth.sample(u, v).r) * 0.5
                    out.store(x, y, r, inv.load(x, y).g, smooth.sample_level(u, v, 0.0).b, 1.0)
                }
            }
        );

        let Some(dev) = open() else { return };
        let pipeline = |shader: &ZslShader| {
            let shader = dev
                .create_shader(ShaderDesc::msl(shader.msl.unwrap()))
                .unwrap();
            dev.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "zsl_main",
                block: [8, 8, 1],
            })
            .unwrap()
        };
        let (invert, combine) = (pipeline(&INVERT), pipeline(&COMBINE));
        let (w, h) = (5, 3);
        let src = texture(
            &dev,
            Format::Rgba8Unorm,
            w,
            h,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        );
        let mid = texture(
            &dev,
            Format::Rgba8Unorm,
            w,
            h,
            TextureUsage::STORAGE | TextureUsage::SAMPLED,
        );
        let out = texture(
            &dev,
            Format::Rgba8Unorm,
            w,
            h,
            TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC,
        );
        let texels: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 % 256) as u8).collect();
        dev.upload_texture_data(src, &texels).unwrap();
        let nearest = dev.create_sampler(SamplerDesc::default()).unwrap();
        let src_slot = dev.bind_texture(src, nearest).unwrap();
        let mid_slot = dev.bind_texture(mid, nearest).unwrap();
        let src_textures = [src_slot];
        let mid_images = [mid.index()];
        let dimensions = [Scalar::U32(w), Scalar::U32(h)];
        let mid_textures = [mid_slot];
        let out_images = [mid.index(), out.index()];
        let ops = [
            ComputeOp::Dispatch(DispatchOp {
                pipeline: invert,
                bindings: Bindings {
                    buffers: &[],
                    textures: &src_textures,
                    images: &mid_images,
                    scalars: &dimensions,
                },
                grid: [1, 1, 1],
            }),
            ComputeOp::Dispatch(DispatchOp {
                pipeline: combine,
                bindings: Bindings {
                    buffers: &[],
                    textures: &mid_textures,
                    images: &out_images,
                    scalars: &[],
                },
                grid: [1, 1, 1],
            }),
        ];
        dev.submit_compute_ops(1, &ops).unwrap();
        for (i, (got, src)) in pixels(&dev, out, w, h)
            .iter()
            .zip(texels.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let want = [255 - src[0], 255 - src[1], 255 - src[2], 255];
            for (g, w) in got.iter().zip(want) {
                assert!(g.abs_diff(w) <= 1, "texel {i}: {got:?} != {want:?}");
            }
        }
    }
}
