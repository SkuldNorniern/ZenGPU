//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Uses cuda-oxide for Driver API access; absent CUDA yields an empty
//! adapter list rather than a build or link error (the stub library path
//! returns ErrorCode::StubLibrary from cuInit).

mod error;

use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::Mutex;

use cuda_oxide::{
    Cuda,
    context::{Context, Handle},
    device::Device,
    mem::{DeviceBox, DevicePtr},
};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    ComputePipelineDesc, DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance,
    HalCapabilities, PipelineHandle, Result, Scalar, SamplerDesc, SamplerHandle, ShaderDesc,
    ShaderHandle, ShaderSource, SlotMap, TextureDesc, TextureHandle, marker,
};

use error::{cu, from_cuda};

// ── CudaInstance ──────────────────────────────────────────────────────────────

/// Entry-point for the CUDA backend. Calls `cuInit` at construction; if CUDA
/// is absent (stub library or no driver), `enumerate_adapters` returns empty.
pub struct CudaInstance {
    initialized: bool,
}

impl CudaInstance {
    pub fn new() -> Self {
        let initialized = match Cuda::init() {
            Ok(()) => true,
            Err(e) => {
                log::debug!("cuda: init failed ({e:?}); no CUDA adapters available");
                false
            }
        };
        Self { initialized }
    }
}

impl Default for CudaInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for CudaInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        if !self.initialized {
            return Vec::new();
        }
        match Cuda::list_devices() {
            Ok(devices) => devices
                .into_iter()
                .map(|dev| {
                    let name = dev
                        .name()
                        .unwrap_or_else(|_| "Unknown CUDA Device".into());
                    let info = AdapterInfo {
                        name,
                        vendor: 0x10de, // NVIDIA PCI vendor ID
                        device: 0,
                        device_type: DeviceType::Discrete,
                        backend: BackendPreference::Cuda,
                    };
                    Box::new(CudaAdapter { dev, info }) as Box<dyn GpuAdapter>
                })
                .collect(),
            Err(e) => {
                log::warn!("cuda: list_devices failed: {e:?}");
                Vec::new()
            }
        }
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        // Ordinal 0 is the primary GPU. Future: honour req.power for multi-GPU.
        let _ = req;
        self.enumerate_adapters().into_iter().next()
    }
}

// ── CudaAdapter ───────────────────────────────────────────────────────────────

pub struct CudaAdapter {
    dev: Device,
    info: AdapterInfo,
}

// SAFETY: Device is a newtype over a CUdevice (c_int ordinal); safe across threads.
unsafe impl Send for CudaAdapter {}
unsafe impl Sync for CudaAdapter {}

impl GpuAdapter for CudaAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        let ctx = from_cuda(Context::new(&self.dev))?;
        Ok(Box::new(CudaDevice {
            ctx: UnsafeCell::new(ctx),
            ctx_lock: Mutex::new(()),
            buffers:   Mutex::new(SlotMap::new()),
            shaders:   Mutex::new(SlotMap::new()),
            pipelines: Mutex::new(SlotMap::new()),
        }))
    }
}

// ── CudaBuffer ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaBuffer {
    ptr: u64,
    len: u64,
}

// ── CudaShader ────────────────────────────────────────────────────────────────

struct CudaShader {
    module: cuda_oxide::sys::CUmodule,
    // Keep the null-terminated PTX alive for the module's lifetime.
    _ptx: Vec<u8>,
}

// SAFETY: CUmodule is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaShader {}
unsafe impl Sync for CudaShader {}

// ── CudaPipeline ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaPipeline {
    func:  cuda_oxide::sys::CUfunction,
    block: [u32; 3],
}

// SAFETY: CUfunction is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaPipeline {}
unsafe impl Sync for CudaPipeline {}

// ── CudaDevice ────────────────────────────────────────────────────────────────

/// An opened CUDA device. Provides compute-only buffer, shader, and dispatch
/// operations via the CUDA Driver API; graphics is not supported.
///
/// # Thread safety
///
/// `Context` in cuda-oxide is `!Send` because `Handle` uses an `Rc`. We
/// compensate by holding `ctx_lock` across every operation that enters the
/// context: only one thread at a time ever touches `ctx`, and the
/// `Rc<Handle>` is always created and destroyed within the same locked
/// method call — it never crosses a thread boundary.
pub struct CudaDevice {
    ctx:       UnsafeCell<Context>,
    ctx_lock:  Mutex<()>,
    buffers:   Mutex<SlotMap<marker::Buffer,   CudaBuffer>>,
    shaders:   Mutex<SlotMap<marker::Shader,   CudaShader>>,
    pipelines: Mutex<SlotMap<marker::Pipeline, CudaPipeline>>,
}

// SAFETY: see the CudaDevice doc comment above.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Enter the CUDA context, run `f`, exit the context. `ctx_lock` must be
    /// held for the lifetime of `f` to serialise cross-thread access.
    fn with_context<F, T>(&self, f: F) -> Result<T>
    where
        F: for<'h> FnOnce(Rc<Handle<'h>>) -> Result<T>,
    {
        let _guard = self.ctx_lock.lock().map_err(|_| GpuError::DeviceLost)?;
        // SAFETY: `ctx_lock` is held; no other thread can reach this point.
        let handle = from_cuda(unsafe { &mut *self.ctx.get() }.enter())?;
        f(handle)
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // `&mut self` provides exclusive access — ctx_lock not needed.
        let buffers:   Vec<CudaBuffer>   = self.buffers.get_mut().unwrap().drain().collect();
        let shaders:   Vec<CudaShader>   = self.shaders.get_mut().unwrap().drain().collect();
        let pipelines: Vec<CudaPipeline> = self.pipelines.get_mut().unwrap().drain().collect();

        if buffers.is_empty() && shaders.is_empty() && pipelines.is_empty() {
            return;
        }
        // UnsafeCell::get_mut is safe here because of the exclusive &mut self.
        if let Ok(handle) = self.ctx.get_mut().enter() {
            for cb in buffers {
                // SAFETY: ptr/len came from a DeviceBox we explicitly leaked.
                let dp = unsafe { DevicePtr::from_raw_parts(handle.clone(), cb.ptr, cb.len) };
                let db = unsafe { DeviceBox::from_raw(dp) };
                drop(db);
            }
            let _ = pipelines; // CUfunction handles are owned by their modules; no explicit free.
            for cs in shaders {
                let _ = unsafe { cuda_oxide::sys::cuModuleUnload(cs.module) };
            }
        }
        // If enter() fails the device is already dead; resources are leaked.
    }
}

impl GpuDevice for CudaDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.with_context(|handle| {
            let db = from_cuda(DeviceBox::alloc(&handle, desc.size))?;
            // Extract the raw device pointer before leaking so we can track it.
            let ptr = db.as_raw();
            let len = db.len();
            // Prevent Drop from calling cuMemFree; we manage the allocation manually.
            db.leak();
            let bh = self
                .buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)?
                .insert(CudaBuffer { ptr, len });
            Ok(bh)
        })
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let cb = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;

        if offset + data.len() as u64 > cb.len {
            return Err(GpuError::Backend("cuda: write out of bounds".into()));
        }

        self.with_context(|handle| {
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + data.len() as u64);
            from_cuda(view.store(data))
        })
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let cb = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;

        if offset + len > cb.len {
            return Err(GpuError::Backend("cuda: read out of bounds".into()));
        }

        self.with_context(|handle| {
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + len);
            from_cuda(view.load())
        })
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let cb = match self.buffers.lock() {
            Ok(mut g) => g.remove(buffer),
            Err(_) => return,
        };
        if let Some(cb) = cb {
            let _ = self.with_context(|handle| {
                // SAFETY: ptr/len from our buffer table; context is current.
                let dp = unsafe { DevicePtr::from_raw_parts(handle.clone(), cb.ptr, cb.len) };
                let db = unsafe { DeviceBox::from_raw(dp) };
                drop(db); // calls cuMemFree while context is current
                Ok(())
            });
        }
    }

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend(
            "cuda: compute-only; no texture support".into(),
        ))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend(
            "cuda: compute-only; no texture support".into(),
        ))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend(
            "cuda: compute-only; no sampler support".into(),
        ))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let ShaderSource::Ptx(ptx) = desc.source else {
            return Err(GpuError::Backend(
                "cuda: only PTX shaders are supported".into(),
            ));
        };
        // cuModuleLoadData requires a null-terminated PTX string.
        let mut ptx_vec: Vec<u8> = ptx.to_vec();
        if ptx_vec.last() != Some(&0) {
            ptx_vec.push(0);
        }
        let module = self.with_context(|_handle| {
            let mut m: cuda_oxide::sys::CUmodule = std::ptr::null_mut();
            cu(unsafe {
                cuda_oxide::sys::cuModuleLoadData(
                    &mut m,
                    ptx_vec.as_ptr() as *const std::ffi::c_void,
                )
            })?;
            Ok(m)
        })?;
        let handle = self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaShader { module, _ptx: ptx_vec });
        Ok(handle)
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let cs = match self.shaders.lock() {
            Ok(mut g) => g.remove(shader),
            Err(_) => return,
        };
        if let Some(cs) = cs {
            let _ = self.with_context(|_handle| {
                cu(unsafe { cuda_oxide::sys::cuModuleUnload(cs.module) })
            });
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let module = self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(desc.shader)
            .map(|s| s.module)
            .ok_or_else(|| GpuError::Backend("cuda: invalid shader handle".into()))?;

        let entry = std::ffi::CString::new(desc.entry)
            .map_err(|_| GpuError::Backend("cuda: entry point name contains NUL byte".into()))?;

        let func = self.with_context(|_handle| {
            let mut f: cuda_oxide::sys::CUfunction = std::ptr::null_mut();
            cu(unsafe {
                cuda_oxide::sys::cuModuleGetFunction(&mut f, module, entry.as_ptr())
            })?;
            Ok(f)
        })?;

        let block = if desc.block == [0, 0, 0] { [256, 1, 1] } else { desc.block };
        let handle = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaPipeline { func, block });
        Ok(handle)
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        if let Ok(mut g) = self.pipelines.lock() {
            g.remove(pipeline);
        }
        // CUfunction handles are owned by their parent module; no explicit free.
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let cp = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(pipeline)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid pipeline handle".into()))?;

        // Resolve buffer slot indices → raw CUdeviceptr values (u64).
        let buf_guard = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
        let mut ptrs: Vec<u64> = Vec::with_capacity(bindings.buffers.len());
        for &slot in bindings.buffers {
            let cb = buf_guard
                .get_by_slot_index(slot)
                .ok_or_else(|| GpuError::Backend("cuda: invalid buffer slot in bindings".into()))?;
            ptrs.push(cb.ptr);
        }
        drop(buf_guard);

        // Build kernel parameter storage: [ptr0: u64, ptr1: u64, ..., scalar0, scalar1, ...].
        // cuLaunchKernel wants a *mut *mut c_void where each pointer points to the param value.
        let mut param_storage: Vec<Vec<u8>> = Vec::new();
        for p in &ptrs {
            param_storage.push(p.to_le_bytes().to_vec());
        }
        for s in bindings.scalars {
            match s {
                Scalar::U32(v) => param_storage.push(v.to_le_bytes().to_vec()),
                Scalar::I32(v) => param_storage.push(v.to_le_bytes().to_vec()),
                Scalar::F32(v) => param_storage.push(v.to_bits().to_le_bytes().to_vec()),
            }
        }
        let mut kernel_params: Vec<*mut std::ffi::c_void> = param_storage
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut std::ffi::c_void)
            .collect();

        self.with_context(|_handle| {
            let mut stream: cuda_oxide::sys::CUstream = std::ptr::null_mut();
            cu(unsafe {
                cuda_oxide::sys::cuStreamCreate(
                    &mut stream,
                    cuda_oxide::sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
                )
            })?;

            let launch = cu(unsafe {
                cuda_oxide::sys::cuLaunchKernel(
                    cp.func,
                    grid[0], grid[1], grid[2],
                    cp.block[0], cp.block[1], cp.block[2],
                    0,
                    stream,
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            });
            let sync = cu(unsafe { cuda_oxide::sys::cuStreamSynchronize(stream) });
            unsafe { cuda_oxide::sys::cuStreamDestroy_v2(stream) };
            launch?;
            sync
        })
    }
}

// ── PTX kernels ───────────────────────────────────────────────────────────────

/// `c[i] = a[i] + b[i]` for `n` f32 elements.
///
/// Kernel signature (CUDA-C equivalent):
/// `__global__ void vec_add_f32(float* a, float* b, float* c, uint32_t n)`
///
/// Params: `(a: u64, b: u64, c: u64, n: u32)` — raw device pointers then scalar.
#[cfg(test)]
const VEC_ADD_PTX: &[u8] = b"\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry vec_add_f32(\n\
    .param .u64 param_a,\n\
    .param .u64 param_b,\n\
    .param .u64 param_c,\n\
    .param .u32 param_n\n\
)\n\
{\n\
    .reg .pred  %p0;\n\
    .reg .u32   %r<5>;\n\
    .reg .u64   %rd<4>;\n\
    .reg .f32   %f<3>;\n\
\n\
    ld.param.u64  %rd0, [param_a];\n\
    ld.param.u64  %rd1, [param_b];\n\
    ld.param.u64  %rd2, [param_c];\n\
    ld.param.u32  %r0,  [param_n];\n\
\n\
    mov.u32       %r1, %tid.x;\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mad.lo.u32    %r4, %r3, %r2, %r1;\n\
\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra      done;\n\
\n\
    cvt.u64.u32   %rd3, %r4;\n\
    shl.b64       %rd3, %rd3, 2;\n\
    add.u64       %rd0, %rd0, %rd3;\n\
    add.u64       %rd1, %rd1, %rd3;\n\
    add.u64       %rd2, %rd2, %rd3;\n\
\n\
    ld.global.f32 %f0, [%rd0];\n\
    ld.global.f32 %f1, [%rd1];\n\
    add.f32       %f2, %f0, %f1;\n\
    st.global.f32 [%rd2], %f2;\n\
\n\
done:\n\
    ret;\n\
}\n\0";

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = CudaInstance::new();
        let _ = inst.enumerate_adapters();
    }

    #[test]
    fn adapter_capabilities_are_compute_only() {
        let inst = CudaInstance::new();
        for adapter in inst.enumerate_adapters() {
            assert!(adapter.capabilities().compute);
            assert!(!adapter.capabilities().graphics);
        }
    }

    /// Create a CUDA device from the first available adapter, or return None to skip.
    fn cuda_device() -> Option<Box<dyn GpuDevice>> {
        let inst = CudaInstance::new();
        let adapter = inst.enumerate_adapters().into_iter().next()?;
        Some(adapter.open(DeviceRequest::default()).expect("open failed"))
    }

    #[test]
    fn open_and_buffer_round_trip() {
        let Some(device) = cuda_device() else { return };
        let buf = device
            .create_buffer(BufferDesc {
                size: 256,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create_buffer failed");
        let data: Vec<u8> = (0u8..=255).collect();
        device.write_buffer(buf, 0, &data).expect("write failed");
        let read_back = device.read_buffer(buf, 0, 256).expect("read failed");
        assert_eq!(read_back, data);
        device.destroy_buffer(buf);
    }

    #[test]
    fn buffer_offset_read_write() {
        let Some(device) = cuda_device() else { return };
        let buf = device
            .create_buffer(BufferDesc {
                size: 512,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create_buffer");
        // Write two halves independently.
        let first: Vec<u8> = (0u8..128).collect();
        let second: Vec<u8> = (128u8..=255).collect();
        device.write_buffer(buf, 0, &first).expect("write first");
        device.write_buffer(buf, 128, &second).expect("write second");
        // Read back first half and second half separately.
        let rb_first = device.read_buffer(buf, 0, 128).expect("read first");
        let rb_second = device.read_buffer(buf, 128, 128).expect("read second");
        assert_eq!(rb_first, first);
        assert_eq!(rb_second, second);
        device.destroy_buffer(buf);
    }

    #[test]
    fn multiple_independent_buffers() {
        let Some(device) = cuda_device() else { return };
        let desc = BufferDesc {
            size: 128,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let a = device.create_buffer(desc).expect("create a");
        let b = device.create_buffer(desc).expect("create b");
        let c = device.create_buffer(desc).expect("create c");

        let da: Vec<u8> = (0..128).map(|i| (i * 3) as u8).collect();
        let db: Vec<u8> = (0..128).map(|i| (i * 7 + 1) as u8).collect();
        let dc: Vec<u8> = (0..128).map(|i| (i * 11 + 5) as u8).collect();

        device.write_buffer(a, 0, &da).unwrap();
        device.write_buffer(b, 0, &db).unwrap();
        device.write_buffer(c, 0, &dc).unwrap();

        assert_eq!(device.read_buffer(a, 0, 128).unwrap(), da);
        assert_eq!(device.read_buffer(b, 0, 128).unwrap(), db);
        assert_eq!(device.read_buffer(c, 0, 128).unwrap(), dc);

        device.destroy_buffer(a);
        device.destroy_buffer(b);
        device.destroy_buffer(c);
    }

    #[test]
    fn large_buffer_round_trip() {
        let Some(device) = cuda_device() else { return };
        const SIZE: u64 = 16 * 1024 * 1024; // 16 MiB
        let buf = device
            .create_buffer(BufferDesc {
                size: SIZE,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create 16 MiB buffer");
        let data: Vec<u8> = (0..SIZE as usize).map(|i| (i ^ (i >> 8)) as u8).collect();
        device.write_buffer(buf, 0, &data).expect("write 16 MiB");
        let rb = device.read_buffer(buf, 0, SIZE).expect("read 16 MiB");
        assert_eq!(rb, data, "16 MiB round-trip mismatch");
        device.destroy_buffer(buf);
    }

    /// CPU-vs-CUDA conformance: `c[i] = a[i] + b[i]` on 1024 f32 elements.
    #[test]
    fn vec_add_cpu_vs_cuda() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 1024;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (100 * i) as f32).collect();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let size = (N * std::mem::size_of::<f32>()) as u64;
        let buf_a   = device.create_buffer(BufferDesc { size, usage: zengpu_hal::BufferUsage::STORAGE, memory: zengpu_hal::MemoryUsage::GpuOnly }).unwrap();
        let buf_b   = device.create_buffer(BufferDesc { size, usage: zengpu_hal::BufferUsage::STORAGE, memory: zengpu_hal::MemoryUsage::GpuOnly }).unwrap();
        let buf_out = device.create_buffer(BufferDesc { size, usage: zengpu_hal::BufferUsage::STORAGE, memory: zengpu_hal::MemoryUsage::GpuOnly }).unwrap();

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.write_buffer(buf_a, 0, &a_bytes).unwrap();
        device.write_buffer(buf_b, 0, &b_bytes).unwrap();

        let shader = device.create_shader(ShaderDesc::ptx(VEC_ADD_PTX)).expect("load PTX");
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "vec_add_f32",
            block: [256, 1, 1],
        }).expect("create pipeline");

        device.dispatch(
            pipeline,
            Bindings {
                buffers:  &[buf_a.index(), buf_b.index(), buf_out.index()],
                scalars:  &[Scalar::U32(N as u32)],
                textures: &[],
            },
            [(N as u32).div_ceil(256), 1, 1],
        ).expect("dispatch");

        let raw = device.read_buffer(buf_out, 0, size).unwrap();
        for i in 0..N {
            let got = f32::from_le_bytes(raw[i*4..i*4+4].try_into().unwrap());
            assert!((got - expected[i]).abs() < 1e-4, "out[{i}] = {got}, expected {}", expected[i]);
        }

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_out);
    }
}
