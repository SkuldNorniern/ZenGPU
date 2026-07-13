//! ZenGPU Apple Metal backend — graphics + compute on macOS/iOS.
//!
//! On macOS the instance enumerates all `MTLDevice` objects (including eGPUs).
//! On non-Apple platforms the instance compiles but returns no adapters.
//! Device open (`MTLDevice` creation, command queues, buffers) lands in the
//! next commit once the surface extension story is settled.

use zengpu_hal::{
    AdapterInfo, AdapterRequest, Bindings, BufferDesc, BufferHandle, ComputePipelineDesc,
    DeviceRequest, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, PipelineHandle,
    Result, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle, ShaderSource, TextureDesc,
    TextureHandle,
};

#[cfg(target_os = "macos")]
use zengpu_hal::{BackendPreference, DeviceType, Scalar, SlotMap, marker};

// ── MetalInstance ─────────────────────────────────────────────────────────────

/// Entry-point for the Metal backend.
///
/// On macOS, [`enumerate_adapters`] returns one entry per `MTLDevice` —
/// including Apple Silicon integrated GPUs and any connected eGPUs.
/// On other platforms it always returns empty.
///
/// [`enumerate_adapters`]: MetalInstance::enumerate_adapters
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetalInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for MetalInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        #[cfg(target_os = "macos")]
        {
            metal::Device::all()
                .into_iter()
                .map(|dev| {
                    let device_type = if dev.is_low_power() {
                        DeviceType::Integrated
                    } else {
                        DeviceType::Discrete
                    };
                    let info = AdapterInfo {
                        name: dev.name().to_string(),
                        vendor: 0x106b, // Apple PCI vendor ID
                        device: 0,
                        device_type,
                        backend: BackendPreference::Metal,
                    };
                    Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Vec::new()
        }
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        // On macOS, prefer the non-low-power device if multiple are present.
        #[cfg(target_os = "macos")]
        {
            let all = self.enumerate_adapters();
            // system_default() is the OS-preferred device; use it.
            metal::Device::system_default()
                .map(|dev| {
                    let device_type = if dev.is_low_power() {
                        DeviceType::Integrated
                    } else {
                        DeviceType::Discrete
                    };
                    let info = AdapterInfo {
                        name: dev.name().to_string(),
                        vendor: 0x106b,
                        device: 0,
                        device_type,
                        backend: BackendPreference::Metal,
                    };
                    Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
                })
                // Fall back to first enumerated device if system_default is None.
                .or_else(|| all.into_iter().next())
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

// ── MetalAdapter ──────────────────────────────────────────────────────────────

pub struct MetalAdapter {
    info: AdapterInfo,
}

impl GpuAdapter for MetalAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        #[cfg(target_os = "macos")]
        {
            let device = metal::Device::system_default()
                .ok_or_else(|| GpuError::Backend("metal: no MTLDevice available".into()))?;
            let queue = device.new_command_queue();
            Ok(Box::new(MetalDevice {
                inner: MacDevice {
                    device,
                    queue,
                    buffers: std::sync::Mutex::new(SlotMap::default()),
                    shaders: std::sync::Mutex::new(SlotMap::default()),
                    pipelines: std::sync::Mutex::new(SlotMap::default()),
                },
            }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(GpuError::Backend(
                "metal: unavailable on this platform".into(),
            ))
        }
    }
}

// ── MetalDevice ───────────────────────────────────────────────────────────────

/// A GPU-resident buffer. On Apple Silicon, `Shared` storage is host-visible and
/// device-visible (unified memory), so reads/writes are plain `memcpy`.
#[cfg(target_os = "macos")]
struct MetalBuffer {
    buf: metal::Buffer,
    size: u64,
}

#[cfg(target_os = "macos")]
struct MetalShader {
    library: metal::Library,
}

#[cfg(target_os = "macos")]
struct MetalPipeline {
    state: metal::ComputePipelineState,
    /// Threadgroup size (`@workgroup_size`), used as threads-per-threadgroup.
    block: [u32; 3],
}

#[cfg(target_os = "macos")]
struct MacDevice {
    device: metal::Device,
    queue: metal::CommandQueue,
    buffers: std::sync::Mutex<SlotMap<marker::Buffer, MetalBuffer>>,
    shaders: std::sync::Mutex<SlotMap<marker::Shader, MetalShader>>,
    pipelines: std::sync::Mutex<SlotMap<marker::Pipeline, MetalPipeline>>,
}

/// An opened Metal device. Buffers today; compute/graphics submission and the
/// ZSL→MSL shader path follow.
pub struct MetalDevice {
    #[cfg(target_os = "macos")]
    inner: MacDevice,
}

// SAFETY: the contained Metal objects are reference-counted Obj-C handles; all
// mutable state (the buffer slot map) is guarded by a Mutex, and no raw pointer
// is shared across threads.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalDevice {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalDevice {}

impl GpuDevice for MetalDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        #[cfg(target_os = "macos")]
        {
            // Metal rejects zero-length buffers; round up to 1 byte.
            let len = desc.size.max(1);
            let buf = self
                .inner
                .device
                .new_buffer(len, metal::MTLResourceOptions::StorageModeShared);
            let mut buffers = self
                .inner
                .buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)?;
            Ok(buffers.insert(MetalBuffer {
                buf,
                size: desc.size,
            }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = desc;
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self
                .inner
                .buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            if offset + data.len() as u64 > b.size {
                return Err(GpuError::Backend("metal: write out of bounds".into()));
            }
            // SAFETY: Shared-storage contents() is a valid host pointer for `size`
            // bytes; the bounds check above keeps the copy in range.
            unsafe {
                let ptr = (b.buf.contents() as *mut u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (buffer, offset, data);
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self
                .inner
                .buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            if offset + len > b.size {
                return Err(GpuError::Backend("metal: read out of bounds".into()));
            }
            let mut out = vec![0u8; len as usize];
            // SAFETY: as above; the bounds check keeps the copy within `size`.
            unsafe {
                let ptr = (b.buf.contents() as *const u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len as usize);
            }
            Ok(out)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (buffer, offset, len);
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut buffers) = self.inner.buffers.lock() {
                buffers.remove(buffer);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = buffer;
        }
    }

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend(
            "metal: textures not yet implemented".into(),
        ))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend(
            "metal: textures not yet implemented".into(),
        ))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend(
            "metal: samplers not yet implemented".into(),
        ))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}

    // ── Compute ─────────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        #[cfg(target_os = "macos")]
        {
            let ShaderSource::Msl(bytes) = desc.source else {
                return Err(GpuError::Backend(
                    "metal: only ShaderSource::Msl is supported (use zsl!().msl)".into(),
                ));
            };
            let source = std::str::from_utf8(bytes)
                .map_err(|_| GpuError::Backend("metal: MSL source is not valid UTF-8".into()))?;
            let library = self
                .inner
                .device
                .new_library_with_source(source, &metal::CompileOptions::new())
                .map_err(|e| GpuError::Backend(format!("metal: MSL compile failed: {e}")))?;
            let mut shaders = self
                .inner
                .shaders
                .lock()
                .map_err(|_| GpuError::DeviceLost)?;
            Ok(shaders.insert(MetalShader { library }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = desc;
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut shaders) = self.inner.shaders.lock() {
                shaders.remove(shader);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = shader;
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        #[cfg(target_os = "macos")]
        {
            let function = {
                let shaders = self
                    .inner
                    .shaders
                    .lock()
                    .map_err(|_| GpuError::DeviceLost)?;
                let shader = shaders
                    .get(desc.shader)
                    .ok_or_else(|| GpuError::Backend("metal: invalid shader handle".into()))?;
                shader.library.get_function(desc.entry, None).map_err(|e| {
                    GpuError::Backend(format!("metal: function `{}`: {e}", desc.entry))
                })?
            };
            let state = self
                .inner
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| GpuError::Backend(format!("metal: pipeline: {e}")))?;
            let mut pipelines = self
                .inner
                .pipelines
                .lock()
                .map_err(|_| GpuError::DeviceLost)?;
            Ok(pipelines.insert(MetalPipeline {
                state,
                block: desc.block,
            }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = desc;
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut pipelines) = self.inner.pipelines.lock() {
                pipelines.remove(pipeline);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = pipeline;
        }
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            self.dispatch_one(pipeline, bindings, grid)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (pipeline, bindings, grid);
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }
}

// ── Compute dispatch (macOS) ────────────────────────────────────────────────

#[cfg(target_os = "macos")]
impl MetalDevice {
    /// Encode + submit one compute dispatch, blocking until completion. Buffers
    /// in `bindings.buffers` (slot indices) bind to `[[buffer(0..n)]]`; scalars
    /// pack into a `Push` struct at `[[buffer(n)]]`; `grid` is the threadgroup
    /// count and the pipeline's `block` is threads-per-threadgroup.
    fn dispatch_one(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let pipelines = self
            .inner
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?;
        let pipe = pipelines
            .get(pipeline)
            .ok_or_else(|| GpuError::Backend("metal: invalid pipeline handle".into()))?;
        let buffers = self
            .inner
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?;

        let cmd = self.inner.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe.state);

        for (i, &slot) in bindings.buffers.iter().enumerate() {
            let b = buffers.get_by_slot_index(slot).ok_or_else(|| {
                GpuError::Backend("metal: invalid buffer slot in bindings".into())
            })?;
            enc.set_buffer(i as u64, Some(&b.buf), 0);
        }

        if !bindings.scalars.is_empty() {
            let packed = pack_scalars(bindings.scalars);
            enc.set_bytes(
                bindings.buffers.len() as u64,
                packed.len() as u64,
                packed.as_ptr() as *const std::ffi::c_void,
            );
        }

        let tg = metal::MTLSize {
            width: grid[0] as u64,
            height: grid[1] as u64,
            depth: grid[2] as u64,
        };
        let tptg = metal::MTLSize {
            width: pipe.block[0] as u64,
            height: pipe.block[1] as u64,
            depth: pipe.block[2] as u64,
        };
        enc.dispatch_thread_groups(tg, tptg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }
}

/// Pack inline scalars into a tightly-laid-out `Push` struct (4 bytes each,
/// matching the MSL `struct Push` field order the ZSL→MSL backend emits).
#[cfg(target_os = "macos")]
fn pack_scalars(scalars: &[Scalar]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scalars.len() * 4);
    for s in scalars {
        match s {
            Scalar::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Scalar::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Scalar::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        device.destroy_buffer(buf);
    }

    #[cfg(target_os = "macos")]
    fn bytemuck_cast(data: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding/invalid bit patterns; viewing as bytes is sound.
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        }
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
        let result: &[f32] = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const f32, n) };
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Render a full-target triangle (ZSL→MSL vertex/fragment) to an offscreen
    /// RGBA8 texture and read it back — validates the Metal graphics path on the
    /// GPU without a window.
    #[test]
    #[cfg(target_os = "macos")]
    fn offscreen_triangle_render_on_metal() {
        use zengpu_spirv::{ZslShader, zsl};

        const VS_SHADER: ZslShader = zsl!(
            vertex vs(@location(0) pos: f32x4) -> f32x4 { pos }
        );
        const FS_SHADER: ZslShader = zsl!(
            fragment fs() -> f32x4 { f32x4(1.0, 0.0, 0.0, 1.0) }
        );
        let vs = VS_SHADER.msl;
        let fs = FS_SHADER.msl;

        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();

        // Oversized triangle covering the whole NDC viewport, in clip space.
        let verts: [f32; 12] = [
            -1.0, -1.0, 0.0, 1.0, //
            3.0, -1.0, 0.0, 1.0, //
            -1.0, 3.0, 0.0, 1.0, //
        ];
        let vbuf = device.new_buffer_with_data(
            verts.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&verts) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let opts = metal::CompileOptions::new();
        let vs_fn = device
            .new_library_with_source(vs, &opts)
            .unwrap()
            .get_function("zsl_main", None)
            .unwrap();
        let fs_fn = device
            .new_library_with_source(fs, &opts)
            .unwrap()
            .get_function("zsl_main", None)
            .unwrap();

        const W: u64 = 4;
        let tex_desc = metal::TextureDescriptor::new();
        tex_desc.set_pixel_format(metal::MTLPixelFormat::RGBA8Unorm);
        tex_desc.set_width(W);
        tex_desc.set_height(W);
        tex_desc.set_usage(metal::MTLTextureUsage::RenderTarget);
        tex_desc.set_storage_mode(metal::MTLStorageMode::Shared);
        let tex = device.new_texture(&tex_desc);

        let pdesc = metal::RenderPipelineDescriptor::new();
        pdesc.set_vertex_function(Some(&vs_fn));
        pdesc.set_fragment_function(Some(&fs_fn));
        pdesc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(metal::MTLPixelFormat::RGBA8Unorm);

        let vd = metal::VertexDescriptor::new();
        let attr = vd.attributes().object_at(0).unwrap();
        attr.set_format(metal::MTLVertexFormat::Float4);
        attr.set_offset(0);
        attr.set_buffer_index(0);
        vd.layouts().object_at(0).unwrap().set_stride(16);
        pdesc.set_vertex_descriptor(Some(vd));

        let pso = device
            .new_render_pipeline_state(&pdesc)
            .expect("render pipeline");

        let rpd = metal::RenderPassDescriptor::new();
        let ca = rpd.color_attachments().object_at(0).unwrap();
        ca.set_texture(Some(&tex));
        ca.set_load_action(metal::MTLLoadAction::Clear);
        ca.set_clear_color(metal::MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        ca.set_store_action(metal::MTLStoreAction::Store);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_render_command_encoder(rpd);
        enc.set_render_pipeline_state(&pso);
        enc.set_vertex_buffer(0, Some(&vbuf), 0);
        enc.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut px = vec![0u8; (W * W * 4) as usize];
        tex.get_bytes(
            px.as_mut_ptr() as *mut std::ffi::c_void,
            W * 4,
            metal::MTLRegion::new_2d(0, 0, W, W),
            0,
        );
        // The triangle covers the whole target → every texel is opaque red.
        assert_eq!(
            &px[0..4],
            &[255, 0, 0, 255],
            "expected red texel, got {:?}",
            &px[0..4]
        );
    }
}
