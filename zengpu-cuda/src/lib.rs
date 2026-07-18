//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Uses cuda-oxide for Driver API access; absent CUDA yields an empty
//! adapter list rather than a build or link error (the stub library path
//! returns ErrorCode::StubLibrary from cuInit).

mod error;

use std::any::Any;
use std::cell::UnsafeCell;
use std::ffi::{CString, c_char, c_void};
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cuda_oxide::{
    Cuda,
    context::{Context, Handle},
    device::Device,
    mem::{DeviceBox, DevicePtr},
    sys,
};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, ComputePipelineDesc, DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuSubmission, HalCapabilities, PipelineHandle, Result, SamplerDesc,
    SamplerHandle, Scalar, ShaderDesc, ShaderHandle, ShaderSource, SlotMap, Submission,
    SubmissionStatus, TextureDesc, TextureHandle, UsageError, marker,
};

use error::{cu, from_cuda};

// ── NVRTC bindings ────────────────────────────────────────────────────────────

type NvrtcProgram = *mut c_void;

#[link(name = "nvrtc")]
unsafe extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut NvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn nvrtcCompileProgram(prog: NvrtcProgram, num_opts: i32, opts: *const *const c_char) -> i32;
    fn nvrtcGetPTXSize(prog: NvrtcProgram, sz: *mut usize) -> i32;
    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut c_char) -> i32;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> i32;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, sz: *mut usize) -> i32;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut c_char) -> i32;
}

const NVRTC_SUCCESS: i32 = 0;

fn nvrtc_compile_to_ptx(src: &[u8]) -> Result<Vec<u8>> {
    let src_cstr = CString::new(src)
        .map_err(|_| GpuError::Backend("nvrtc: CUDA C++ source contains NUL byte".into()))?;
    let name_cstr = CString::new("zsl_kernel.cu").unwrap();

    let mut prog: NvrtcProgram = ptr::null_mut();
    let r = unsafe {
        nvrtcCreateProgram(
            &mut prog,
            src_cstr.as_ptr(),
            name_cstr.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    if r != NVRTC_SUCCESS {
        return Err(GpuError::Backend(format!("nvrtcCreateProgram failed: {r}")));
    }

    let r = unsafe { nvrtcCompileProgram(prog, 0, ptr::null()) };
    if r != NVRTC_SUCCESS {
        let log = unsafe {
            let mut log_size = 0usize;
            nvrtcGetProgramLogSize(prog, &mut log_size);
            let mut buf = vec![0i8; log_size];
            nvrtcGetProgramLog(prog, buf.as_mut_ptr());
            nvrtcDestroyProgram(&mut prog);
            String::from_utf8_lossy(&buf.iter().map(|&b| b as u8).collect::<Vec<_>>()).into_owned()
        };
        return Err(GpuError::Backend(format!(
            "nvrtc compilation failed:\n{log}"
        )));
    }

    let ptx = unsafe {
        let mut ptx_size = 0usize;
        nvrtcGetPTXSize(prog, &mut ptx_size);
        let mut buf = vec![0i8; ptx_size];
        nvrtcGetPTX(prog, buf.as_mut_ptr());
        nvrtcDestroyProgram(&mut prog);
        buf.iter().map(|&b| b as u8).collect::<Vec<_>>()
    };
    Ok(ptx)
}

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
                    let name = dev.name().unwrap_or_else(|_| "Unknown CUDA Device".into());
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
        let mut ctx = from_cuda(Context::new(&self.dev))?;
        let stream = {
            let _handle = from_cuda(ctx.enter())?;
            let mut s: sys::CUstream = ptr::null_mut();
            cu(unsafe {
                sys::cuStreamCreate(&mut s, sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING)
            })?;
            s
        };
        Ok(Box::new(CudaDevice {
            shared: Arc::new(CudaShared {
                ctx: UnsafeCell::new(ctx),
                ctx_lock: Mutex::new(()),
                stream,
            }),
            buffers: Arc::new(Mutex::new(SlotMap::new())),
            shaders: Arc::new(Mutex::new(SlotMap::new())),
            pipelines: Arc::new(Mutex::new(SlotMap::new())),
            lifetime: Arc::new(Mutex::new(CudaLifetimeState::default())),
        }))
    }
}

// ── CudaBuffer ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaBuffer {
    ptr: u64,
    len: u64,
    usage: BufferUsage,
}

// ── CudaShader ────────────────────────────────────────────────────────────────

struct CudaShader {
    module: sys::CUmodule,
    // Keep the null-terminated PTX alive for the module's lifetime.
    _ptx: Vec<u8>,
}

// SAFETY: CUmodule is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaShader {}
unsafe impl Sync for CudaShader {}

// ── CudaPipeline ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaPipeline {
    func: sys::CUfunction,
    block: [u32; 3],
}

// SAFETY: CUfunction is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaPipeline {}
unsafe impl Sync for CudaPipeline {}

enum DeferredCudaResource {
    Buffer(u32, CudaBuffer),
    Shader(u32, CudaShader),
    Pipeline(u32),
}

#[derive(Default)]
struct CudaLifetimeState {
    in_flight: usize,
    deferred: Vec<DeferredCudaResource>,
}

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
struct CudaShared {
    ctx: UnsafeCell<Context>,
    ctx_lock: Mutex<()>,
    /// Persistent compute stream — created once in `open()`, reused across all
    /// dispatches to eliminate per-kernel create/destroy overhead.
    stream: sys::CUstream,
}

// SAFETY: every context entry is serialized by ctx_lock.
unsafe impl Send for CudaShared {}
unsafe impl Sync for CudaShared {}

impl CudaShared {
    fn with_context<F, T>(&self, f: F) -> Result<T>
    where
        F: for<'h> FnOnce(Rc<Handle<'h>>) -> Result<T>,
    {
        let _guard = self.ctx_lock.lock().map_err(|_| GpuError::DeviceLost)?;
        // SAFETY: ctx_lock is held; no other thread can enter this context.
        let handle = from_cuda(unsafe { &mut *self.ctx.get() }.enter())?;
        f(handle)
    }
}

impl Drop for CudaShared {
    fn drop(&mut self) {
        if self.ctx.get_mut().enter().is_ok() {
            unsafe { sys::cuStreamSynchronize(self.stream) };
            unsafe { sys::cuStreamDestroy_v2(self.stream) };
        }
    }
}

pub struct CudaDevice {
    shared: Arc<CudaShared>,
    buffers: Arc<Mutex<SlotMap<marker::Buffer, CudaBuffer>>>,
    shaders: Arc<Mutex<SlotMap<marker::Shader, CudaShader>>>,
    pipelines: Arc<Mutex<SlotMap<marker::Pipeline, CudaPipeline>>>,
    lifetime: Arc<Mutex<CudaLifetimeState>>,
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
        self.shared.with_context(f)
    }
}

struct CudaSubmission {
    cycle_id: u64,
    shared: Arc<CudaShared>,
    event: Mutex<Option<sys::CUevent>>,
    buffers: Arc<Mutex<SlotMap<marker::Buffer, CudaBuffer>>>,
    shaders: Arc<Mutex<SlotMap<marker::Shader, CudaShader>>>,
    pipelines: Arc<Mutex<SlotMap<marker::Pipeline, CudaPipeline>>>,
    lifetime: Arc<Mutex<CudaLifetimeState>>,
}

// SAFETY: CUevent is opaque and every operation enters the serialized context.
unsafe impl Send for CudaSubmission {}
unsafe impl Sync for CudaSubmission {}

impl CudaSubmission {
    fn release_in_flight(&self) {
        release_cuda_in_flight(
            &self.shared,
            &self.buffers,
            &self.shaders,
            &self.pipelines,
            &self.lifetime,
        );
    }

    fn query(&self) -> Result<SubmissionStatus> {
        let mut event = self.event.lock().map_err(|_| GpuError::DeviceLost)?;
        let Some(raw) = *event else {
            return Ok(SubmissionStatus::Complete);
        };
        let code = self
            .shared
            .with_context(|_| Ok(unsafe { sys::cuEventQuery(raw) }))?;
        if code == 0 {
            self.shared
                .with_context(|_| cu(unsafe { sys::cuEventDestroy_v2(raw) }))?;
            *event = None;
            self.release_in_flight();
            Ok(SubmissionStatus::Complete)
        } else if code == sys::cudaError_enum_CUDA_ERROR_NOT_READY {
            Ok(SubmissionStatus::Pending)
        } else {
            cu(code)?;
            unreachable!()
        }
    }
}

impl GpuSubmission for CudaSubmission {
    fn cycle_id(&self) -> u64 {
        self.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        self.query()
    }

    fn wait(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            if self.query()? == SubmissionStatus::Complete {
                return Ok(());
            }
            if timeout == Duration::ZERO
                || deadline.is_some_and(|deadline| Instant::now() >= deadline)
            {
                return Err(GpuError::Timeout);
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for CudaSubmission {
    fn drop(&mut self) {
        let Ok(event) = self.event.get_mut() else {
            return;
        };
        let Some(raw) = event.take() else {
            return;
        };
        // Preserve the resource-lifetime contract even if a pending token is
        // dropped: wait for its exact stream position, then destroy the event.
        let _ = self.shared.with_context(|_| {
            cu(unsafe { sys::cuEventSynchronize(raw) })?;
            cu(unsafe { sys::cuEventDestroy_v2(raw) })
        });
        self.release_in_flight();
    }
}

fn release_cuda_in_flight(
    shared: &CudaShared,
    buffers: &Mutex<SlotMap<marker::Buffer, CudaBuffer>>,
    shaders: &Mutex<SlotMap<marker::Shader, CudaShader>>,
    pipelines: &Mutex<SlotMap<marker::Pipeline, CudaPipeline>>,
    lifetime: &Mutex<CudaLifetimeState>,
) {
    let deferred = {
        let mut lifetime = lifetime.lock().unwrap();
        debug_assert!(lifetime.in_flight > 0);
        lifetime.in_flight = lifetime.in_flight.saturating_sub(1);
        if lifetime.in_flight == 0 {
            std::mem::take(&mut lifetime.deferred)
        } else {
            Vec::new()
        }
    };
    for resource in deferred {
        match resource {
            DeferredCudaResource::Buffer(index, buffer) => {
                let _ = shared.with_context(|_| cu(unsafe { sys::cuMemFree_v2(buffer.ptr) }));
                buffers.lock().unwrap().release_retired(index);
            }
            DeferredCudaResource::Shader(index, shader) => {
                let _ = shared.with_context(|_| cu(unsafe { sys::cuModuleUnload(shader.module) }));
                shaders.lock().unwrap().release_retired(index);
            }
            DeferredCudaResource::Pipeline(index) => {
                pipelines.lock().unwrap().release_retired(index);
            }
        }
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // `&mut self` provides exclusive access — ctx_lock not needed.
        let buffers: Vec<CudaBuffer> = self.buffers.lock().unwrap().drain().collect();
        let shaders: Vec<CudaShader> = self.shaders.lock().unwrap().drain().collect();
        let pipelines: Vec<CudaPipeline> = self.pipelines.lock().unwrap().drain().collect();

        let _ = self.shared.with_context(|handle| {
            // A device may be dropped before its submission handles. Drain
            // work before freeing the resources owned by this device.
            cu(unsafe { sys::cuStreamSynchronize(self.shared.stream) })?;
            for cb in buffers {
                // SAFETY: ptr/len came from a DeviceBox we explicitly leaked.
                let dp = unsafe { DevicePtr::from_raw_parts(handle.clone(), cb.ptr, cb.len) };
                let db = unsafe { DeviceBox::from_raw(dp) };
                drop(db);
            }
            let _ = pipelines; // CUfunction handles are owned by their modules; no explicit free.
            for cs in shaders {
                let _ = unsafe { sys::cuModuleUnload(cs.module) };
            }
            Ok(())
        });
    }
}

impl GpuDevice for CudaDevice {
    fn as_any(&self) -> &dyn Any {
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
                .insert(CudaBuffer {
                    ptr,
                    len,
                    usage: desc.usage,
                });
            Ok(bh)
        })
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.with_context(|handle| {
            let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            let cb = buffers
                .get(buffer)
                .copied()
                .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;
            if self
                .lifetime
                .lock()
                .map_err(|_| GpuError::DeviceLost)?
                .in_flight
                != 0
            {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "write_buffer cannot mutate buffers while an asynchronous submission is pending"
                        .into(),
                )));
            }
            if offset + data.len() as u64 > cb.len {
                return Err(GpuError::Backend("cuda: write out of bounds".into()));
            }
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + data.len() as u64);
            from_cuda(view.store(data))
        })
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.with_context(|handle| {
            let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            let cb = buffers
                .get(buffer)
                .copied()
                .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;
            if self
                .lifetime
                .lock()
                .map_err(|_| GpuError::DeviceLost)?
                .in_flight
                != 0
            {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "read_buffer cannot observe buffers while an asynchronous submission is pending"
                        .into(),
                )));
            }
            if offset + len > cb.len {
                return Err(GpuError::Backend("cuda: read out of bounds".into()));
            }
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + len);
            from_cuda(view.load())
        })
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        self.with_context(|handle| {
            let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            if self
                .lifetime
                .lock()
                .map_err(|_| GpuError::DeviceLost)?
                .in_flight
                != 0
            {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "copy_buffer cannot run while an asynchronous submission is pending".into(),
                )));
            }
            let src_buf = buffers
                .get(src)
                .copied()
                .ok_or_else(|| GpuError::Backend("cuda: invalid source buffer handle".into()))?;
            let dst_buf = buffers.get(dst).copied().ok_or_else(|| {
                GpuError::Backend("cuda: invalid destination buffer handle".into())
            })?;
            if !src_buf.usage.contains(BufferUsage::TRANSFER_SRC) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "source buffer",
                    needed: "TRANSFER_SRC",
                }));
            }
            if !dst_buf.usage.contains(BufferUsage::TRANSFER_DST) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "destination buffer",
                    needed: "TRANSFER_DST",
                }));
            }
            if src == dst {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "copy_buffer requires distinct source and destination buffers".into(),
                )));
            }
            let src_end = src_offset.checked_add(len).ok_or_else(|| {
                GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "source copy range overflows u64".into(),
                ))
            })?;
            let dst_end = dst_offset.checked_add(len).ok_or_else(|| {
                GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "destination copy range overflows u64".into(),
                ))
            })?;
            if src_end > src_buf.len || dst_end > dst_buf.len {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    format!(
                        "copy ranges src {src_offset}..{src_end}/{} dst {dst_offset}..{dst_end}/{}",
                        src_buf.len, dst_buf.len
                    ),
                )));
            }
            if len == 0 {
                return Ok(());
            }
            let src_ptr =
                unsafe { DevicePtr::from_raw_parts(handle.clone(), src_buf.ptr, src_buf.len) };
            let dst_ptr = unsafe { DevicePtr::from_raw_parts(handle, dst_buf.ptr, dst_buf.len) };
            from_cuda(
                src_ptr
                    .subslice(src_offset, src_end)
                    .copy_to(&dst_ptr.subslice(dst_offset, dst_end)),
            )
        })
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let cb = {
            let mut buffers = match self.buffers.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let mut lifetime = match self.lifetime.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            if lifetime.in_flight != 0 {
                if let Some(cb) = buffers.retire(buffer) {
                    lifetime
                        .deferred
                        .push(DeferredCudaResource::Buffer(buffer.index(), cb));
                }
                None
            } else {
                buffers.remove(buffer)
            }
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
        let mut ptx_vec: Vec<u8> = match desc.source {
            ShaderSource::Ptx(ptx) => ptx.to_vec(),
            ShaderSource::CudaSrc(src) => nvrtc_compile_to_ptx(src)?,
            _ => {
                return Err(GpuError::Backend(
                    "cuda: only PTX or CUDA C++ (CudaSrc) shaders are supported".into(),
                ));
            }
        };
        // cuModuleLoadData requires a null-terminated PTX string.
        if ptx_vec.last() != Some(&0) {
            ptx_vec.push(0);
        }
        let module = self.with_context(|_handle| {
            let mut m: sys::CUmodule = ptr::null_mut();
            cu(unsafe { sys::cuModuleLoadData(&mut m, ptx_vec.as_ptr() as *const c_void) })?;
            Ok(m)
        })?;
        let handle = self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaShader {
                module,
                _ptx: ptx_vec,
            });
        Ok(handle)
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let cs = {
            let mut shaders = match self.shaders.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let mut lifetime = match self.lifetime.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            if lifetime.in_flight != 0 {
                if let Some(cs) = shaders.retire(shader) {
                    lifetime
                        .deferred
                        .push(DeferredCudaResource::Shader(shader.index(), cs));
                }
                None
            } else {
                shaders.remove(shader)
            }
        };
        if let Some(cs) = cs {
            let _ = self.with_context(|_handle| cu(unsafe { sys::cuModuleUnload(cs.module) }));
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

        let entry = CString::new(desc.entry)
            .map_err(|_| GpuError::Backend("cuda: entry point name contains NUL byte".into()))?;

        let func = self.with_context(|_handle| {
            let mut f: sys::CUfunction = ptr::null_mut();
            cu(unsafe { sys::cuModuleGetFunction(&mut f, module, entry.as_ptr()) })?;
            Ok(f)
        })?;

        let block = if desc.block == [0, 0, 0] {
            [256, 1, 1]
        } else {
            desc.block
        };
        let handle = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaPipeline { func, block });
        Ok(handle)
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let Ok(mut pipelines) = self.pipelines.lock() else {
            return;
        };
        let Ok(mut lifetime) = self.lifetime.lock() else {
            return;
        };
        if lifetime.in_flight != 0 {
            if pipelines.retire(pipeline).is_some() {
                lifetime
                    .deferred
                    .push(DeferredCudaResource::Pipeline(pipeline.index()));
            }
        } else {
            pipelines.remove(pipeline);
        }
        // CUfunction handles are owned by their parent module; no explicit free.
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.dispatch_batch(&[zengpu_hal::DispatchOp {
            pipeline,
            bindings,
            grid,
        }])
    }

    fn dispatch_batch(&self, ops: &[zengpu_hal::DispatchOp<'_>]) -> Result<()> {
        self.submit_batch(0, ops)?.wait(Duration::MAX)
    }

    fn submit_batch(
        &self,
        cycle_id: u64,
        ops: &[zengpu_hal::DispatchOp<'_>],
    ) -> Result<Submission> {
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        // Kernel parameters live in a fixed-size, per-op stack buffer — no
        // heap allocation per dispatch. Each slot is 8 bytes regardless of
        // the value's actual size (4-byte scalars only use the low bytes);
        // `cuLaunchKernel` reads exactly as many bytes as the kernel
        // signature declares for that parameter, so the unused tail is inert.
        const MAX_PARAMS: usize = 32;

        let stream = self.shared.stream;
        self.lifetime
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .in_flight += 1;
        // Same-stream launches execute and become globally visible in
        // submission order on the device, so a later op safely reads an
        // earlier op's output with no explicit barrier. The event recorded
        // after the final launch represents completion of the whole batch.
        let event = self.with_context(|_handle| {
            for op in ops {
                let cp = self
                    .pipelines
                    .lock()
                    .map_err(|_| GpuError::DeviceLost)?
                    .get(op.pipeline)
                    .copied()
                    .ok_or_else(|| GpuError::Backend("cuda: invalid pipeline handle".into()))?;

                let mut storage = [[0u8; 8]; MAX_PARAMS];
                let mut n_params = 0usize;
                {
                    let buf_guard = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
                    for &slot in op.bindings.buffers {
                        if n_params >= MAX_PARAMS {
                            return Err(GpuError::Dispatch(format!(
                                "dispatch: more than {MAX_PARAMS} kernel parameters"
                            )));
                        }
                        let cb = buf_guard.get_by_slot_index(slot).ok_or_else(|| {
                            GpuError::Backend("cuda: invalid buffer slot in bindings".into())
                        })?;
                        storage[n_params] = cb.ptr.to_le_bytes();
                        n_params += 1;
                    }
                }
                for s in op.bindings.scalars {
                    if n_params >= MAX_PARAMS {
                        return Err(GpuError::Dispatch(format!(
                            "dispatch: more than {MAX_PARAMS} kernel parameters"
                        )));
                    }
                    let bytes4 = match s {
                        Scalar::U32(v) => v.to_le_bytes(),
                        Scalar::I32(v) => v.to_le_bytes(),
                        Scalar::F32(v) => v.to_bits().to_le_bytes(),
                    };
                    storage[n_params][..4].copy_from_slice(&bytes4);
                    n_params += 1;
                }

                let mut kernel_params: [*mut c_void; MAX_PARAMS] = [ptr::null_mut(); MAX_PARAMS];
                for (i, slot) in storage.iter_mut().take(n_params).enumerate() {
                    kernel_params[i] = slot.as_mut_ptr() as *mut c_void;
                }

                cu(unsafe {
                    sys::cuLaunchKernel(
                        cp.func,
                        op.grid[0],
                        op.grid[1],
                        op.grid[2],
                        cp.block[0],
                        cp.block[1],
                        cp.block[2],
                        0,
                        stream,
                        kernel_params.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                })?;
            }
            let mut event: sys::CUevent = ptr::null_mut();
            cu(unsafe {
                sys::cuEventCreate(&mut event, sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING)
            })?;
            if let Err(e) = cu(unsafe { sys::cuEventRecord(event, stream) }) {
                unsafe { sys::cuEventDestroy_v2(event) };
                return Err(e);
            }
            Ok(event)
        });
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                // Some earlier launches may already be queued. Drain them
                // before releasing any concurrently retired resources.
                let _ = self.with_context(|_| cu(unsafe { sys::cuStreamSynchronize(stream) }));
                release_cuda_in_flight(
                    &self.shared,
                    &self.buffers,
                    &self.shaders,
                    &self.pipelines,
                    &self.lifetime,
                );
                return Err(error);
            }
        };
        Ok(Box::new(CudaSubmission {
            cycle_id,
            shared: Arc::clone(&self.shared),
            event: Mutex::new(Some(event)),
            buffers: Arc::clone(&self.buffers),
            shaders: Arc::clone(&self.shaders),
            pipelines: Arc::clone(&self.pipelines),
            lifetime: Arc::clone(&self.lifetime),
        }))
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

/// CUDA C++ equivalent of [`VEC_ADD_PTX`], compiled at runtime with NVRTC.
#[cfg(test)]
const VEC_ADD_CUDA: &str = r#"
extern "C" __global__ void vec_add_f32(
    const float* a,
    const float* b,
    float* c,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_spirv::{ZslShader, zsl};

    const TRIG_ZSL: ZslShader = zsl!(
        push P { len: u32 }
        @workgroup_size(64)
        kernel trig(src: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
            let i = id.x
            if i < p.len {
                let x = src[i]
                out[i] = sin(x) + cos(x) + tan(x)
            }
        }
    );

    const TYPED_BUFFER_ZSL: ZslShader = zsl!(
        push P { len: u32 }
        @workgroup_size(64)
        kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, p: P, id: global_id) {
            let i = id.x
            if i < p.len {
                out_u[i] = src_u[i]
                out_i[i] = src_i[i]
            }
        }
    );

    const FINITE_ZSL: ZslShader = zsl!(
        push P { len: u32 }
        @workgroup_size(64)
        kernel classify(src: device buffer<f32>, out: device mut buffer<u32>, p: P, id: global_id) {
            let i = id.x
            if i < p.len {
                let x = src[i]
                if isfinite(x) {
                    out[i] = 1
                } else {
                    if isnan(x) {
                        out[i] = 2
                    } else {
                        if isinf(x) { out[i] = 3 } else { out[i] = 4 }
                    }
                }
            }
        }
    );

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
    fn device_info() {
        // CudaInstance::new() calls cuInit; without it list_devices returns empty.
        let inst = CudaInstance::new();
        if !inst.initialized {
            println!("no CUDA driver — skip");
            return;
        }
        let devices = Cuda::list_devices().unwrap_or_default();
        if devices.is_empty() {
            println!("no CUDA devices — skip");
            return;
        }
        for (i, dev) in devices.iter().enumerate() {
            let name = dev.name().unwrap_or_else(|_| "?".into());
            let vram_mb = dev.memory_size().unwrap_or(0) / (1024 * 1024);
            let cc = dev
                .compute_capability()
                .map(|v| format!("sm_{}{}", v.major, v.minor))
                .unwrap_or_else(|_| "?".into());
            println!("CUDA device {i}: {name}  |  {vram_mb} MiB  |  {cc}");
        }
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
        device
            .write_buffer(buf, 128, &second)
            .expect("write second");
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

    #[test]
    fn device_to_device_buffer_copy_round_trip() {
        const SIZE: usize = 4 * 1024 * 1024;
        let Some(dev) = cuda_device() else { return };
        let src = dev
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::TRANSFER_SRC,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let dst = dev
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let data: Vec<u8> = (0..SIZE)
            .map(|i| (i as u32).wrapping_mul(17).wrapping_add(3) as u8)
            .collect();
        dev.write_buffer(src, 0, &data).unwrap();
        dev.copy_buffer(src, 0, dst, 0, SIZE as u64).unwrap();
        assert_eq!(dev.read_buffer(dst, 0, SIZE as u64).unwrap(), data);
        dev.destroy_buffer(src);
        dev.destroy_buffer(dst);
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
        let buf_a = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_b = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_out = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.write_buffer(buf_a, 0, &a_bytes).unwrap();
        device.write_buffer(buf_b, 0, &b_bytes).unwrap();

        let shader = device
            .create_shader(ShaderDesc::ptx(VEC_ADD_PTX))
            .expect("load PTX");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "vec_add_f32",
                block: [256, 1, 1],
            })
            .expect("create pipeline");

        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[buf_a.index(), buf_b.index(), buf_out.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                },
                [(N as u32).div_ceil(256), 1, 1],
            )
            .expect("dispatch");

        let raw = device.read_buffer(buf_out, 0, size).unwrap();
        for i in 0..N {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(
                (got - expected[i]).abs() < 1e-4,
                "out[{i}] = {got}, expected {}",
                expected[i]
            );
        }

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_out);
    }

    /// End-to-end CUDA C++ path: NVRTC compilation, module load, launch, and readback.
    #[test]
    fn cuda_source_nvrtc_vec_add() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 1024;
        let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..N).map(|i| 1000.0 - i as f32 * 0.5).collect();
        let size = (N * std::mem::size_of::<f32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let buf_a = device.create_buffer(desc).expect("create a");
        let buf_b = device.create_buffer(desc).expect("create b");
        let buf_out = device.create_buffer(desc).expect("create output");

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.write_buffer(buf_a, 0, &a_bytes).expect("upload a");
        device.write_buffer(buf_b, 0, &b_bytes).expect("upload b");

        let shader = device
            .create_shader(ShaderDesc::cuda_src(VEC_ADD_CUDA))
            .expect("compile CUDA C++ with NVRTC");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "vec_add_f32",
                block: [256, 1, 1],
            })
            .expect("create NVRTC pipeline");
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[buf_a.index(), buf_b.index(), buf_out.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                },
                [(N as u32).div_ceil(256), 1, 1],
            )
            .expect("dispatch NVRTC kernel");

        let raw = device.read_buffer(buf_out, 0, size).expect("read output");
        for i in 0..N {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            let expected = a[i] + b[i];
            assert!(
                (got - expected).abs() < 1e-4,
                "out[{i}] = {got}, expected {expected}"
            );
        }

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_out);
    }

    #[test]
    fn zsl_trigonometry_runs_through_cuda() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 128;
        let input: Vec<f32> = (0..N).map(|i| -0.5 + i as f32 / N as f32).collect();
        let size = (N * std::mem::size_of::<f32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let src = device.create_buffer(desc).expect("create source");
        let out = device.create_buffer(desc).expect("create output");
        let input_bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        device
            .write_buffer(src, 0, &input_bytes)
            .expect("upload source");

        let (shader_desc, entry) = TRIG_ZSL.for_backend(BackendPreference::Cuda);
        let shader = device.create_shader(shader_desc).expect("compile ZSL CUDA");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .expect("create ZSL CUDA pipeline");
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[src.index(), out.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                },
                [(N as u32).div_ceil(64), 1, 1],
            )
            .expect("dispatch ZSL CUDA trigonometry");

        let raw = device.read_buffer(out, 0, size).expect("read output");
        for (i, value) in input.iter().copied().enumerate() {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            let expected = value.sin() + value.cos() + value.tan();
            assert!((got - expected).abs() < 2e-5, "out[{i}] mismatch");
        }

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src);
        device.destroy_buffer(out);
    }

    #[test]
    fn zsl_integer_buffers_round_trip_through_cuda() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 128;
        let input_u: Vec<u32> = (0..N as u32)
            .map(|value| value.wrapping_mul(2_654_435_761))
            .collect();
        let input_i: Vec<i32> = (0..N as i32).map(|value| value - 64).collect();
        let size = (N * std::mem::size_of::<u32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let src_u = device.create_buffer(desc).expect("create src_u");
        let src_i = device.create_buffer(desc).expect("create src_i");
        let out_u = device.create_buffer(desc).expect("create out_u");
        let out_i = device.create_buffer(desc).expect("create out_i");
        let bytes_u: Vec<u8> = input_u
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let bytes_i: Vec<u8> = input_i
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        device.write_buffer(src_u, 0, &bytes_u).expect("upload u32");
        device.write_buffer(src_i, 0, &bytes_i).expect("upload i32");

        let (shader_desc, entry) = TYPED_BUFFER_ZSL.for_backend(BackendPreference::Cuda);
        let shader = device
            .create_shader(shader_desc)
            .expect("compile typed ZSL CUDA");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .expect("create typed CUDA pipeline");
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[src_u.index(), src_i.index(), out_u.index(), out_i.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                },
                [(N as u32).div_ceil(64), 1, 1],
            )
            .expect("dispatch typed CUDA kernel");

        assert_eq!(device.read_buffer(out_u, 0, size).unwrap(), bytes_u);
        assert_eq!(device.read_buffer(out_i, 0, size).unwrap(), bytes_i);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src_u);
        device.destroy_buffer(src_i);
        device.destroy_buffer(out_u);
        device.destroy_buffer(out_i);
    }

    #[test]
    fn zsl_finite_classification_runs_through_cuda() {
        let Some(device) = cuda_device() else { return };
        let input = [0.0f32, -12.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let expected = [1u32, 1, 2, 3, 3];
        let size = std::mem::size_of_val(&input) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let src = device.create_buffer(desc).expect("create source");
        let out = device.create_buffer(desc).expect("create output");
        let bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        device.write_buffer(src, 0, &bytes).expect("upload values");

        let (shader_desc, entry) = FINITE_ZSL.for_backend(BackendPreference::Cuda);
        let shader = device
            .create_shader(shader_desc)
            .expect("compile finite ZSL CUDA");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .expect("create finite CUDA pipeline");
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[src.index(), out.index()],
                    scalars: &[Scalar::U32(input.len() as u32)],
                    textures: &[],
                },
                [1, 1, 1],
            )
            .expect("dispatch finite CUDA kernel");

        let raw = device
            .read_buffer(out, 0, size)
            .expect("read classifications");
        let actual: Vec<u32> = raw
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src);
        device.destroy_buffer(out);
    }

    #[test]
    fn submission_batch_chains_ops_on_one_stream() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (10 * i) as f32).collect();
        // sum = a + b; doubled = sum + sum, batched as one submission.
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 2.0 * (x + y)).collect();

        let size = (N * std::mem::size_of::<f32>()) as u64;
        let gpu = |size| BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let buf_a = device.create_buffer(gpu(size)).unwrap();
        let buf_b = device.create_buffer(gpu(size)).unwrap();
        let buf_sum = device.create_buffer(gpu(size)).unwrap();
        let buf_out = device.create_buffer(gpu(size)).unwrap();

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.write_buffer(buf_a, 0, &a_bytes).unwrap();
        device.write_buffer(buf_b, 0, &b_bytes).unwrap();

        let shader = device
            .create_shader(ShaderDesc::ptx(VEC_ADD_PTX))
            .expect("load PTX");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "vec_add_f32",
                block: [256, 1, 1],
            })
            .expect("create pipeline");

        let grid = [(N as u32).div_ceil(256), 1, 1];
        let submission = device
            .submit_batch(
                77,
                &[
                    zengpu_hal::DispatchOp {
                        pipeline,
                        bindings: Bindings {
                            buffers: &[buf_a.index(), buf_b.index(), buf_sum.index()],
                            scalars: &[Scalar::U32(N as u32)],
                            textures: &[],
                        },
                        grid,
                    },
                    zengpu_hal::DispatchOp {
                        pipeline,
                        bindings: Bindings {
                            buffers: &[buf_sum.index(), buf_sum.index(), buf_out.index()],
                            scalars: &[Scalar::U32(N as u32)],
                            textures: &[],
                        },
                        grid,
                    },
                ],
            )
            .expect("submit_batch");
        assert_eq!(submission.cycle_id(), 77);
        let second_out = device.create_buffer(gpu(size)).unwrap();
        let second_submission = device
            .submit(
                78,
                pipeline,
                Bindings {
                    buffers: &[buf_a.index(), buf_b.index(), second_out.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                },
                grid,
            )
            .unwrap();
        assert!(device.write_buffer(buf_a, 0, &[0; 4]).is_err());
        let retired_sum_index = buf_sum.index();
        device.destroy_buffer(buf_sum);
        device.destroy_pipeline(pipeline);
        assert!(device.write_buffer(buf_sum, 0, &[0; 4]).is_err());
        let allocated_while_pending = device.create_buffer(gpu(size)).unwrap();
        assert_ne!(allocated_while_pending.index(), retired_sum_index);
        let zero_wait = submission.wait(Duration::ZERO);
        assert!(zero_wait.is_ok() || matches!(zero_wait, Err(GpuError::Timeout)));
        submission.wait(Duration::from_secs(5)).unwrap();
        assert_eq!(submission.poll().unwrap(), SubmissionStatus::Complete);
        let allocated_after_first = device.create_buffer(gpu(size)).unwrap();
        assert_ne!(allocated_after_first.index(), retired_sum_index);
        second_submission.wait(Duration::from_secs(5)).unwrap();
        assert_eq!(
            second_submission.poll().unwrap(),
            SubmissionStatus::Complete
        );
        let allocated_after_completion = device.create_buffer(gpu(size)).unwrap();
        assert_eq!(allocated_after_completion.index(), retired_sum_index);

        let raw = device.read_buffer(buf_out, 0, size).unwrap();
        for i in 0..N {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(
                (got - expected[i]).abs() < 1e-3,
                "out[{i}] = {got}, expected {}",
                expected[i]
            );
        }

        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_sum);
        device.destroy_buffer(buf_out);
        device.destroy_buffer(allocated_while_pending);
        device.destroy_buffer(allocated_after_first);
        device.destroy_buffer(allocated_after_completion);
        device.destroy_buffer(second_out);
    }
}
