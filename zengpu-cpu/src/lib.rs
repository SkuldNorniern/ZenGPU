//! ZenGPU CPU reference backend — the conformance oracle.
//!
//! Plain Rust over `Vec<u8>` buffers: correctness and determinism over speed.
//! It is the reference the GPU backends are validated against, **not**
//! a product fallback (consumers like aurea keep their own CPU paths).
//!
//! # Compute dispatch
//!
//! The CPU backend cannot execute SPIR-V directly. Instead, callers create a
//! pipeline as usual and then register a Rust function for that
//! [`PipelineHandle`] via [`CpuDevice::register_kernel`]. When
//! [`GpuDevice::dispatch`] is called, the backend looks up the function
//! registered for the dispatched pipeline, lends the bound buffers' data to
//! it in a [`CpuKernelCtx`], calls it, and puts the data back.
//! This is the "oracle" model: CPU kernels are hand-written Rust
//! functions that should produce bit-identical results to the GPU SPIR-V.
//!
//! Keying by pipeline handle (rather than SPIR-V entry-point name) lets two
//! different ops both use the GLSL-conventional entry point `"main"` in their
//! SPIR-V (each its own shader module / pipeline) without colliding in the
//! registry.

#![forbid(unsafe_code)]
// This is a backend HAL implementation; a failed driver/backend call here is unrecoverable and is meant to panic loudly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::any::Any;
use std::collections::HashMap;
use std::mem::take;
use std::sync::Mutex;

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, ComputePipelineDesc, DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, HalCapabilities, PipelineHandle, Result, SamplerDesc, SamplerHandle, Scalar,
    ShaderDesc, ShaderHandle, ShaderSource, SlotMap, TextureDesc, TextureHandle, UsageError,
    marker,
};

// ── Kernel registry types ─────────────────────────────────────────────────────

/// Arguments passed to a CPU kernel by [`CpuDevice::dispatch`].
///
/// `buffers[i]` holds the data of the buffer at slot index
/// `bindings.buffers[i]`, lent for the call. Modify it in place, keeping its
/// length; it goes back to its buffer when the kernel returns.
pub struct CpuKernelCtx {
    pub buffers: Vec<Vec<u8>>,
    pub scalars: Vec<Scalar>,
    pub grid: [u32; 3],
}

/// A CPU-side kernel function. Registered with [`CpuDevice::register_kernel`]
/// for the [`PipelineHandle`] it implements.
pub type CpuKernel = Box<dyn Fn(&mut CpuKernelCtx) + Send + Sync>;

// ── Internal slot types ───────────────────────────────────────────────────────

struct CpuBuffer {
    data: Vec<u8>,
    usage: BufferUsage,
}

struct CpuPipeline {
    entry: String,
}

type BufferMap = SlotMap<marker::Buffer, CpuBuffer>;

/// A CPU-backed [`GpuDevice`]. All state lives in host memory behind mutexes,
/// so the device is `Send + Sync`.
pub struct CpuDevice {
    buffers: Mutex<BufferMap>,
    shaders: Mutex<SlotMap<marker::Shader, Vec<u8>>>,
    pipelines: Mutex<SlotMap<marker::Pipeline, CpuPipeline>>,
    kernels: Mutex<HashMap<PipelineHandle, CpuKernel>>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(SlotMap::new()),
            shaders: Mutex::new(SlotMap::new()),
            pipelines: Mutex::new(SlotMap::new()),
            kernels: Mutex::new(HashMap::new()),
        }
    }

    /// Register a CPU kernel for `pipeline` (returned by
    /// [`GpuDevice::create_compute_pipeline`]).
    ///
    /// When [`GpuDevice::dispatch`] is called with this `pipeline`, `kernel`
    /// is invoked with the bound buffer data.
    pub fn register_kernel(&self, pipeline: PipelineHandle, kernel: CpuKernel) {
        self.kernels.lock().unwrap().insert(pipeline, kernel);
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

// ── Error helpers ─────────────────────────────────────────────────────────────

fn stale(handle: BufferHandle, buffers: &BufferMap) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: buffers.generation_at(handle.index()).unwrap_or(u32::MAX),
    })
}

fn out_of_bounds(start: usize, end: usize, len: usize) -> GpuError {
    GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
        "range {start}..{end} exceeds buffer size {len}"
    )))
}

// ── GpuDevice ─────────────────────────────────────────────────────────────────

impl GpuDevice for CpuDevice {
    fn backend(&self) -> BackendPreference {
        BackendPreference::Cpu
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let buffer = CpuBuffer {
            data: vec![0u8; desc.size as usize],
            usage: desc.usage,
        };
        Ok(self.buffers.lock().unwrap().insert(buffer))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.get(buffer).is_none() {
            return Err(stale(buffer, &buffers));
        }
        let buf = buffers.get_mut(buffer).unwrap();
        let start = offset as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| out_of_bounds(start, usize::MAX, buf.data.len()))?;
        if end > buf.data.len() {
            return Err(out_of_bounds(start, end, buf.data.len()));
        }
        buf.data[start..end].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or_else(|| out_of_bounds(start, usize::MAX, buf.data.len()))?;
        if end > buf.data.len() {
            return Err(out_of_bounds(start, end, buf.data.len()));
        }
        Ok(buf.data[start..end].to_vec())
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        let start = offset as usize;
        let end = start
            .checked_add(dst.len())
            .ok_or_else(|| out_of_bounds(start, usize::MAX, buf.data.len()))?;
        if end > buf.data.len() {
            return Err(out_of_bounds(start, end, buf.data.len()));
        }
        dst.copy_from_slice(&buf.data[start..end]);
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let src_buf = buffers.get(src).ok_or_else(|| stale(src, &buffers))?;
        if !src_buf.usage.contains(BufferUsage::TRANSFER_SRC) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "source buffer",
                needed: "TRANSFER_SRC",
            }));
        }
        let src_start = src_offset as usize;
        let src_end = src_start
            .checked_add(len as usize)
            .ok_or_else(|| out_of_bounds(src_start, usize::MAX, src_buf.data.len()))?;
        if src_end > src_buf.data.len() {
            return Err(out_of_bounds(src_start, src_end, src_buf.data.len()));
        }
        let bytes = src_buf.data[src_start..src_end].to_vec();

        if src == dst {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "copy_buffer requires distinct source and destination buffers".into(),
            )));
        }
        if buffers.get(dst).is_none() {
            return Err(stale(dst, &buffers));
        }
        let dst_buf = buffers.get_mut(dst).unwrap();
        if !dst_buf.usage.contains(BufferUsage::TRANSFER_DST) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "destination buffer",
                needed: "TRANSFER_DST",
            }));
        }
        let dst_start = dst_offset as usize;
        let dst_end = dst_start
            .checked_add(len as usize)
            .ok_or_else(|| out_of_bounds(dst_start, usize::MAX, dst_buf.data.len()))?;
        if dst_end > dst_buf.data.len() {
            return Err(out_of_bounds(dst_start, dst_end, dst_buf.data.len()));
        }
        dst_buf.data[dst_start..dst_end].copy_from_slice(&bytes);
        Ok(())
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        self.buffers.lock().unwrap().remove(buffer);
    }

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend(
            "CPU backend does not support textures".to_string(),
        ))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend(
            "CPU backend does not support textures".to_string(),
        ))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend(
            "CPU backend does not support samplers".to_string(),
        ))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}

    // ── Compute ───────────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let bytes = match desc.source {
            ShaderSource::Spirv(b) | ShaderSource::Ptx(b) | ShaderSource::Msl(b) => b,
            ShaderSource::Hip(_) | ShaderSource::CudaSrc(_) => {
                return Err(GpuError::Unsupported(
                    "CPU backend does not support HIP or CUDA source shaders".to_string(),
                ));
            }
        };
        Ok(self.shaders.lock().unwrap().insert(bytes.to_vec()))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        self.shaders.lock().unwrap().remove(shader);
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        {
            let shaders = self.shaders.lock().unwrap();
            if shaders.get(desc.shader).is_none() {
                return Err(GpuError::PipelineCreation(format!(
                    "shader handle {:?} is stale",
                    desc.shader
                )));
            }
        }
        let pipeline = CpuPipeline {
            entry: desc.entry.to_string(),
        };
        Ok(self.pipelines.lock().unwrap().insert(pipeline))
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        self.pipelines.lock().unwrap().remove(pipeline);
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let entry = {
            let pipelines = self.pipelines.lock().unwrap();
            let p = pipelines
                .get(pipeline)
                .ok_or_else(|| GpuError::Dispatch("stale pipeline handle".to_string()))?;
            p.entry.clone()
        };
        let kernels = self.kernels.lock().unwrap();
        let kernel = kernels.get(&pipeline).ok_or_else(|| {
            GpuError::Dispatch(format!(
                "no CPU kernel registered for pipeline (entry '{entry}'); \
                 call CpuDevice::register_kernel before dispatching"
            ))
        })?;

        // The bound buffers are lent to the kernel rather than copied, and
        // the table stays locked meanwhile so nobody sees them taken. Each
        // buffer is bound once: two copies of one would leave which write
        // wins to chance.
        let mut buffers = self.buffers.lock().unwrap();
        for (i, &idx) in bindings.buffers.iter().enumerate() {
            if buffers.get_by_slot_index(idx).is_none() {
                return Err(GpuError::Dispatch(format!(
                    "binding {i} names buffer slot {idx}, which holds no buffer"
                )));
            }
            if bindings.buffers[..i].contains(&idx) {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    format!(
                        "buffer slot {idx} is bound twice; the CPU backend binds each buffer once"
                    ),
                )));
            }
        }
        let lent: Vec<Vec<u8>> = bindings
            .buffers
            .iter()
            .map(|&idx| take(&mut buffers.get_mut_by_slot_index(idx).unwrap().data))
            .collect();
        let sizes: Vec<usize> = lent.iter().map(Vec::len).collect();
        let mut ctx = CpuKernelCtx {
            buffers: lent,
            scalars: bindings.scalars.to_vec(),
            grid,
        };
        kernel(&mut ctx);

        // A kernel cannot resize a buffer; one that tried gets its buffer
        // back at its own size and an error.
        let mut resized = None;
        for ((&idx, mut data), size) in bindings.buffers.iter().zip(ctx.buffers).zip(sizes) {
            if data.len() != size {
                resized = Some(idx);
                data.resize(size, 0);
            }
            buffers.get_mut_by_slot_index(idx).unwrap().data = data;
        }
        if let Some(idx) = resized {
            return Err(GpuError::Dispatch(format!(
                "the kernel for entry '{entry}' resized buffer slot {idx}"
            )));
        }
        Ok(())
    }
}

// ── Adapter + Instance ────────────────────────────────────────────────────────

pub struct CpuAdapter {
    info: AdapterInfo,
}

impl CpuAdapter {
    pub fn new() -> Self {
        Self {
            info: AdapterInfo {
                name: "ZenGPU CPU Reference".to_string(),
                vendor: 0,
                device: 0,
                device_type: DeviceType::Cpu,
                backend: BackendPreference::Cpu,
            },
        }
    }
}

impl Default for CpuAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuAdapter for CpuAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Ok(Box::new(CpuDevice::new()))
    }
}

pub struct CpuInstance;

impl GpuInstance for CpuInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        vec![Box::new(CpuAdapter::new())]
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        Some(Box::new(CpuAdapter::new()))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_hal::MemoryUsage;

    fn rw_desc(size: u64) -> BufferDesc {
        BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        }
    }

    #[test]
    fn foreign_and_recreated_device_handles_cannot_access_buffers() {
        let first = CpuDevice::new();
        let old = first.create_buffer(rw_desc(4)).unwrap();
        let second = CpuDevice::new();
        let own = second.create_buffer(rw_desc(4)).unwrap();
        second.write_buffer(own, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(old.index(), own.index());
        assert_eq!(old.generation(), own.generation());
        assert!(second.read_buffer(old, 0, 4).is_err());
        assert!(second.write_buffer(old, 0, &[9; 4]).is_err());
        second.destroy_buffer(old);
        assert_eq!(second.read_buffer(own, 0, 4).unwrap(), [1, 2, 3, 4]);
        drop(first);
        let recreated = CpuDevice::new();
        let new = recreated.create_buffer(rw_desc(4)).unwrap();
        assert!(recreated.write_buffer(old, 0, &[9; 4]).is_err());
        assert!(recreated.write_buffer(new, 0, &[5; 4]).is_ok());
    }

    #[test]
    fn buffer_roundtrip() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(dev.read_buffer(h, 2, 2).unwrap(), vec![3, 4]);
    }

    #[test]
    fn read_without_readback_usage_fails() {
        let dev = CpuDevice::new();
        let h = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage {
                needed: "READBACK",
                ..
            })
        ));
    }

    #[test]
    fn use_after_destroy_is_stale() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.destroy_buffer(h);
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        match err {
            GpuError::InvalidUsage(UsageError::StaleHandle {
                expected_gen,
                actual_gen,
                ..
            }) => {
                assert_ne!(expected_gen, actual_gen)
            }
            other => panic!("expected StaleHandle, got {other}"),
        }
    }

    #[test]
    fn out_of_bounds_write_fails() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));
    }

    #[test]
    fn reports_compute_only_capabilities() {
        let dev = CpuDevice::new();
        assert!(dev.capabilities().compute);
        assert!(!dev.capabilities().graphics);
    }

    #[test]
    fn usable_as_dyn_device() {
        let dev: Box<dyn GpuDevice> = Box::new(CpuDevice::new());
        let h = dev.create_buffer(rw_desc(2)).unwrap();
        dev.write_buffer(h, 0, &[9, 8]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 2).unwrap(), vec![9, 8]);
    }

    #[test]
    fn adapter_opens_cpu_device() {
        let adapter = CpuAdapter::new();
        assert_eq!(adapter.info().name, "ZenGPU CPU Reference");
        assert!(!adapter.capabilities().graphics);
        assert!(adapter.capabilities().compute);
        let dev = adapter.open(DeviceRequest::default()).unwrap();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn instance_enumerates_one_adapter() {
        let inst = CpuInstance;
        let adapters = inst.enumerate_adapters();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].info().name, "ZenGPU CPU Reference");
    }

    #[test]
    fn instance_request_adapter_always_returns_cpu() {
        let inst = CpuInstance;
        let adapter = inst.request_adapter(AdapterRequest::default()).unwrap();
        let dev = adapter.open(DeviceRequest::default()).unwrap();
        assert!(dev.capabilities().compute);
    }

    // ── Compute tests ─────────────────────────────────────────────────────────

    #[test]
    fn dispatch_vec_add_f32() {
        let dev = CpuDevice::new();

        let n: u32 = 8;
        let size = (n as u64) * 4;
        let usage = BufferUsage::STORAGE | BufferUsage::READBACK;
        let mem = MemoryUsage::Upload;

        let ha = dev
            .create_buffer(BufferDesc {
                size,
                usage,
                memory: mem,
            })
            .unwrap();
        let hb = dev
            .create_buffer(BufferDesc {
                size,
                usage,
                memory: mem,
            })
            .unwrap();
        let hout = dev
            .create_buffer(BufferDesc {
                size,
                usage,
                memory: mem,
            })
            .unwrap();

        let a_data: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let b_data: Vec<u8> = (0..n)
            .flat_map(|i| (100.0f32 * i as f32).to_le_bytes())
            .collect();
        dev.write_buffer(ha, 0, &a_data).unwrap();
        dev.write_buffer(hb, 0, &b_data).unwrap();

        let dummy_spv = [0x03, 0x02, 0x23, 0x07u8]; // fake SPIR-V magic
        let shader = dev.create_shader(ShaderDesc::spirv(&dummy_spv)).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [0, 0, 0],
            })
            .unwrap();

        // Register a Rust vec_add kernel for this pipeline.
        dev.register_kernel(
            pipeline,
            Box::new(|ctx: &mut CpuKernelCtx| {
                let len = match ctx.scalars.first() {
                    Some(&Scalar::U32(n)) => n as usize,
                    _ => return,
                };
                // buffers[0]=a, [1]=b, [2]=out (4 bytes per f32)
                let read_f32 = |buf: &[u8], i: usize| -> f32 {
                    f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
                };
                let write_f32 = |buf: &mut Vec<u8>, i: usize, v: f32| {
                    buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                };
                if ctx.buffers.len() < 3 {
                    return;
                }
                // Can't borrow a[..] and out[..] simultaneously from same vec; clone a+b first.
                let a: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[0], i)).collect();
                let b: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[1], i)).collect();
                for i in 0..len {
                    write_f32(&mut ctx.buffers[2], i, a[i] + b[i]);
                }
            }),
        );

        dev.dispatch(
            pipeline,
            Bindings {
                buffers: &[ha.index(), hb.index(), hout.index()],
                scalars: &[Scalar::U32(n)],
                textures: &[],
                images: &[],
            },
            [1, 1, 1],
        )
        .unwrap();

        let out = dev.read_buffer(hout, 0, size).unwrap();
        for i in 0..n as usize {
            let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            let expected = i as f32 + 100.0 * i as f32;
            assert!(
                (got - expected).abs() < 1e-5,
                "out[{i}] = {got}, expected {expected}"
            );
        }

        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
        dev.destroy_buffer(ha);
        dev.destroy_buffer(hb);
        dev.destroy_buffer(hout);
    }

    fn noop_pipeline(dev: &CpuDevice, kernel: CpuKernel) -> PipelineHandle {
        let shader = dev.create_shader(ShaderDesc::spirv(&[0u8; 4])).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [1, 1, 1],
            })
            .unwrap();
        dev.register_kernel(pipeline, kernel);
        pipeline
    }

    fn bound(buffers: &[u32]) -> Bindings<'_> {
        Bindings {
            buffers,
            ..Bindings::default()
        }
    }

    #[test]
    fn dispatch_rejects_missing_and_repeated_buffers() {
        let dev = CpuDevice::new();
        let pipeline = noop_pipeline(&dev, Box::new(|_| {}));
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        assert!(matches!(
            dev.dispatch(pipeline, bound(&[h.index() + 1]), [1, 1, 1]),
            Err(GpuError::Dispatch(_))
        ));
        assert!(matches!(
            dev.dispatch(pipeline, bound(&[h.index(), h.index()]), [1, 1, 1]),
            Err(GpuError::InvalidUsage(UsageError::BindingMismatch(_)))
        ));
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn a_kernel_cannot_resize_a_buffer() {
        let dev = CpuDevice::new();
        let pipeline = noop_pipeline(
            &dev,
            Box::new(|ctx| {
                ctx.buffers[0][0] = 7;
                ctx.buffers[0].push(9);
            }),
        );
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        assert!(
            dev.dispatch(pipeline, bound(&[h.index()]), [1, 1, 1])
                .is_err()
        );
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [7, 0, 0, 0]);
        assert!(dev.read_buffer(h, 0, 5).is_err());
    }

    #[test]
    fn dispatch_stale_pipeline_fails() {
        let dev = CpuDevice::new();
        let dummy_spv = [0u8; 4];
        let shader = dev.create_shader(ShaderDesc::spirv(&dummy_spv)).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [0, 0, 0],
            })
            .unwrap();
        dev.destroy_pipeline(pipeline);
        let err = dev
            .dispatch(pipeline, Bindings::default(), [1, 1, 1])
            .unwrap_err();
        assert!(matches!(err, GpuError::Dispatch(_)));
    }

    #[test]
    fn dispatch_missing_kernel_fails() {
        let dev = CpuDevice::new();
        let dummy_spv = [0u8; 4];
        let shader = dev.create_shader(ShaderDesc::spirv(&dummy_spv)).unwrap();
        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "main",
                block: [0, 0, 0],
            })
            .unwrap();
        // No kernel registered → should fail with Dispatch error.
        let err = dev
            .dispatch(pipeline, Bindings::default(), [1, 1, 1])
            .unwrap_err();
        assert!(
            matches!(err, GpuError::Dispatch(ref msg) if msg.contains("no CPU kernel")),
            "unexpected error: {err}"
        );
    }
}
