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
