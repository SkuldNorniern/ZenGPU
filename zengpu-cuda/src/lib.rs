//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Uses cuda-oxide for Driver API access; absent CUDA yields an empty
//! adapter list rather than a build or link error (the stub library path
//! returns ErrorCode::StubLibrary from cuInit).

// This is a backend HAL implementation; a failed driver/backend call here is unrecoverable and is meant to panic loudly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod error;

use std::any::Any;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::mem::take;
use std::sync::{Arc, Mutex};
use std::thread::yield_now;
use std::time::{Duration, Instant};
use std::{ptr, slice};

use cuda_oxide::{Cuda, context::Context, device::Device, sys};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, ComputeOp, ComputePipelineDesc, DeviceRequest, DeviceType, GpuAdapter, GpuDevice,
    GpuError, GpuInstance, GpuSubmission, HalCapabilities, PipelineHandle, Result, SamplerDesc,
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
    fn nvrtcGetCUBINSize(prog: NvrtcProgram, sz: *mut usize) -> i32;
    fn nvrtcGetCUBIN(prog: NvrtcProgram, cubin: *mut c_char) -> i32;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> i32;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, sz: *mut usize) -> i32;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut c_char) -> i32;
}

const NVRTC_SUCCESS: i32 = 0;

/// Device code ready for `cuModuleLoadData`.
enum ModuleImage {
    /// Architecture-specific binary for the device; loaded without JIT.
    Cubin(Vec<u8>),
    /// PTX text, JIT-compiled by the driver. Null-terminated.
    Ptx(Vec<u8>),
}

impl ModuleImage {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Cubin(bytes) | Self::Ptx(bytes) => bytes,
        }
    }
}

/// Compile CUDA C++ for the device's architecture. Tries `sm_XY` cubin first,
/// then `compute_XY` PTX, then NVRTC's default PTX, so an NVRTC that predates
/// the GPU still produces a runnable module.
fn nvrtc_compile(src: &[u8], (major, minor): (u32, u32)) -> Result<ModuleImage> {
    let cubin_arch = format!("--gpu-architecture=sm_{major}{minor}");
    let ptx_arch = format!("--gpu-architecture=compute_{major}{minor}");
    let attempts = [
        (Some(cubin_arch.as_str()), true),
        (Some(ptx_arch.as_str()), false),
        (None, false),
    ];
    let mut last_error = None;
    for (arch, want_cubin) in attempts {
        match nvrtc_build(src, arch, want_cubin) {
            Ok(image) => return Ok(image),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.expect("nvrtc attempted at least one build"))
}

fn nvrtc_build(src: &[u8], arch: Option<&str>, want_cubin: bool) -> Result<ModuleImage> {
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

    let opts: Vec<CString> = arch.map(|a| CString::new(a).unwrap()).into_iter().collect();
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    let r = unsafe { nvrtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr()) };
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

    let image = nvrtc_read_output(prog, want_cubin);
    unsafe { nvrtcDestroyProgram(&mut prog) };
    image
}

/// Copy the cubin or PTX produced by a successful NVRTC compile into Rust memory.
fn nvrtc_read_output(prog: NvrtcProgram, want_cubin: bool) -> Result<ModuleImage> {
    let read_failed = |what: &str, r: i32| GpuError::Backend(format!("nvrtc: {what} failed: {r}"));
    let mut size = 0usize;
    let (r, bytes) = if want_cubin {
        let r = unsafe { nvrtcGetCUBINSize(prog, &mut size) };
        if r != NVRTC_SUCCESS {
            return Err(read_failed("nvrtcGetCUBINSize", r));
        }
        let mut buf = vec![0i8; size];
        let r = unsafe { nvrtcGetCUBIN(prog, buf.as_mut_ptr()) };
        (r, buf)
    } else {
        let r = unsafe { nvrtcGetPTXSize(prog, &mut size) };
        if r != NVRTC_SUCCESS {
            return Err(read_failed("nvrtcGetPTXSize", r));
        }
        let mut buf = vec![0i8; size];
        let r = unsafe { nvrtcGetPTX(prog, buf.as_mut_ptr()) };
        (r, buf)
    };
    if r != NVRTC_SUCCESS {
        return Err(read_failed("output read", r));
    }
    let bytes: Vec<u8> = bytes.into_iter().map(|b| b as u8).collect();
    Ok(if want_cubin {
        ModuleImage::Cubin(bytes)
    } else {
        ModuleImage::Ptx(bytes)
    })
}

/// Convert a byte count to the driver's `size_t`, failing instead of truncating.
fn driver_len(n: impl TryInto<sys::size_t>) -> Result<sys::size_t> {
    n.try_into().map_err(|_| {
        GpuError::InvalidUsage(UsageError::BindingMismatch(
            "cuda: byte count exceeds driver range".into(),
        ))
    })
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
        let sm = from_cuda(self.dev.compute_capability())?;
        let mut ctx = from_cuda(Context::new(&self.dev))?;
        let (stream, raw_ctx) = {
            let _handle = from_cuda(ctx.enter())?;
            // SAFETY: the context was just entered on this thread, so the query returns it.
            let mut raw: sys::CUcontext = ptr::null_mut();
            cu(unsafe { sys::cuCtxGetCurrent(&mut raw) })?;
            let mut s: sys::CUstream = ptr::null_mut();
            cu(unsafe {
                sys::cuStreamCreate(&mut s, sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING)
            })?;
            (s, raw)
        };
        Ok(Box::new(CudaDevice {
            shared: Arc::new(CudaShared {
                _ctx: ctx,
                raw_ctx,
                ctx_lock: Mutex::new(()),
                stream,
                sm: (sm.major, sm.minor),
                events: Mutex::new(Vec::new()),
                staging: Mutex::new(None),
                graphs: Mutex::new(GraphCache::default()),
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
    // Keep the module image alive for the module's lifetime.
    _image: ModuleImage,
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

// ── Pinned staging ────────────────────────────────────────────────────────────

/// Bytes per pinned staging buffer. Chosen by benchmarking 4, 8 and 16 MiB on
/// an RTX 4090 (WSL); 4 MiB was marginally best of the three (transfer rates within noise).
const STAGING_CHUNK: usize = 4 << 20;

/// Host transfers at least this large go through the pinned double buffer;
/// smaller ones are a single synchronous driver copy.
const STAGING_MIN: usize = 1 << 20;

/// Two pinned host buffers, each with an event marking its last DMA. `pending[i]`
/// is set while a DMA issued from `bufs[i]` may still be in flight.
struct Staging {
    bufs: [*mut c_void; 2],
    events: [sys::CUevent; 2],
    pending: [bool; 2],
}

impl Staging {
    /// Allocate the buffers and events. The context must be current.
    fn new() -> Result<Self> {
        let mut st = Staging {
            bufs: [ptr::null_mut(); 2],
            events: [ptr::null_mut(); 2],
            pending: [false; 2],
        };
        // On error `st` drops here and frees whatever was already allocated.
        for buf in &mut st.bufs {
            // SAFETY: writes one pinned allocation pointer into `buf`.
            cu(unsafe {
                sys::cuMemHostAlloc(buf as *mut *mut c_void, driver_len(STAGING_CHUNK)?, 0)
            })?;
        }
        for ev in &mut st.events {
            cu(unsafe { sys::cuEventCreate(ev, sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING) })?;
        }
        Ok(st)
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // SAFETY: only called with the context current (see CudaShared); null
        // entries were never allocated and are skipped.
        for &buf in &self.bufs {
            if !buf.is_null() {
                unsafe { sys::cuMemFreeHost(buf) };
            }
        }
        for &ev in &self.events {
            if !ev.is_null() {
                unsafe { sys::cuEventDestroy_v2(ev) };
            }
        }
    }
}

/// Spin on `cuEventQuery` until `ev` completes. Used instead of
/// `cuEventSynchronize`: the context is created with blocking-sync scheduling,
/// which would add an OS sleep to every wait. The context must be current.
fn wait_event(ev: sys::CUevent) -> Result<()> {
    loop {
        // SAFETY: `ev` is a live event owned by the caller's pool or staging buffers.
        let code = unsafe { sys::cuEventQuery(ev) };
        if code == 0 {
            return Ok(());
        }
        if code != sys::cudaError_enum_CUDA_ERROR_NOT_READY {
            return cu(code);
        }
        yield_now();
    }
}

// ── Graph replay ──────────────────────────────────────────────────────────────

/// Batches with at least this many ops are candidates for graph replay.
const GRAPH_MIN_OPS: usize = 4;

/// Instantiated graphs kept per device; one is evicted to make room for a new one.
const GRAPH_MAX_READY: usize = 64;

/// Cache entries that are not instantiated graphs are dropped wholesale past this count.
const GRAPH_MAX_UNREADY: usize = 1024;

/// Kernel parameter slots per launch. Fixed size keeps launches heap-free.
const MAX_PARAMS: usize = 32;

/// One resolved driver call of a batch. Every value a launch needs is held by
/// value, so the same list can be issued directly or recorded into a graph.
#[derive(Clone, Copy)]
#[allow(
    clippy::large_enum_variant,
    reason = "plans are short-lived; inline params avoid a heap allocation per launch"
)]
enum Launch {
    Kernel {
        func: sys::CUfunction,
        grid: [u32; 3],
        block: [u32; 3],
        /// Each parameter takes an 8-byte slot; scalars use the low 4 bytes.
        params: [[u8; 8]; MAX_PARAMS],
        n_params: usize,
    },
    Copy {
        dst: u64,
        src: u64,
        len: sys::size_t,
    },
}

#[derive(Clone, Copy)]
enum GraphEntry {
    /// Launched once directly; the next identical batch is captured.
    Seen,
    /// Instantiated graph, replayed with `cuGraphLaunch`.
    Ready(sys::CUgraphExec),
    /// Capture or instantiation failed; always launched directly.
    Uncachable,
}

/// Per-device graph cache. Its keys are the exact resolved launch values, so a
/// hit means the same kernels run with the same pointers and scalars.
#[derive(Default)]
struct GraphCache {
    entries: HashMap<Vec<u64>, GraphEntry>,
    ready: usize,
}

impl GraphCache {
    /// Destroy every instantiated graph and forget all entries. The context must be current.
    fn clear(&mut self) {
        for (_, entry) in self.entries.drain() {
            if let GraphEntry::Ready(exec) = entry {
                // SAFETY: `exec` came from cuGraphInstantiate_v2 and leaves the map here, once.
                unsafe { sys::cuGraphExecDestroy(exec) };
            }
        }
        self.ready = 0;
    }

    /// Store a freshly instantiated graph, evicting one at capacity. The context must be current.
    fn insert_ready(&mut self, key: Vec<u64>, exec: sys::CUgraphExec) {
        if self.ready >= GRAPH_MAX_READY {
            let victim = self
                .entries
                .iter()
                .find(|(_, entry)| matches!(entry, GraphEntry::Ready(_)))
                .map(|(key, _)| key.clone());
            if let Some(victim) = victim
                && let Some(GraphEntry::Ready(old)) = self.entries.remove(&victim)
            {
                // SAFETY: `old` is no longer reachable through the cache.
                unsafe { sys::cuGraphExecDestroy(old) };
                self.ready -= 1;
            }
        }
        self.entries.insert(key, GraphEntry::Ready(exec));
        self.ready += 1;
    }

    /// Record a first sighting of `key`, dropping unready entries when too many build up.
    fn note_seen(&mut self, key: Vec<u64>) {
        if self.entries.len() - self.ready >= GRAPH_MAX_UNREADY {
            self.entries
                .retain(|_, entry| matches!(entry, GraphEntry::Ready(_)));
        }
        self.entries.insert(key, GraphEntry::Seen);
    }
}

/// Flatten a resolved batch into the values that a replay depends on.
fn plan_key(plan: &[Launch]) -> Vec<u64> {
    const KERNEL_TAG: u64 = 0;
    const COPY_TAG: u64 = 1;
    let mut key = Vec::new();
    for launch in plan {
        match launch {
            Launch::Kernel {
                func,
                grid,
                block,
                params,
                n_params,
            } => {
                key.push(KERNEL_TAG);
                key.push(*func as usize as u64);
                key.extend(grid.iter().chain(block).map(|&v| u64::from(v)));
                key.push(*n_params as u64);
                key.extend(
                    params[..*n_params]
                        .iter()
                        .map(|slot| u64::from_le_bytes(*slot)),
                );
            }
            Launch::Copy { dst, src, len } => {
                key.extend([COPY_TAG, *dst, *src, *len as u64]);
            }
        }
    }
    key
}

/// Resolve ops in order until one fails. Returns the launches resolved before the
/// failure and that failure, so the caller can still issue the prefix as a per-op
/// loop always has.
fn resolve_all<T>(
    ops: &[T],
    mut resolve: impl FnMut(&T) -> Result<Option<Launch>>,
) -> (Vec<Launch>, Option<GpuError>) {
    let mut plan = Vec::with_capacity(ops.len());
    for op in ops {
        match resolve(op) {
            Ok(Some(launch)) => plan.push(launch),
            Ok(None) => {}
            Err(error) => return (plan, Some(error)),
        }
    }
    (plan, None)
}

// ── CudaDevice ────────────────────────────────────────────────────────────────

/// An opened CUDA device. Provides compute-only buffer, shader, and dispatch
/// operations via the CUDA Driver API; graphics is not supported.
///
/// # Thread safety
///
/// `Context` in cuda-oxide is `!Send`. We compensate by holding `ctx_lock`
/// across every operation that touches the context: only one thread at a time
/// ever runs driver calls against it, and the context is only ever made
/// current on the thread that holds the lock.
struct CudaShared {
    /// Owns the CUDA context; destroyed after `Drop for CudaShared` runs.
    _ctx: Context,
    /// Raw handle of `_ctx`, made current via `with_current` before driver calls.
    raw_ctx: sys::CUcontext,
    ctx_lock: Mutex<()>,
    /// Persistent compute stream — created once in `open()`, reused across all
    /// dispatches to eliminate per-kernel create/destroy overhead.
    stream: sys::CUstream,
    /// Compute capability, used as the NVRTC target.
    sm: (u32, u32),
    /// Timing-disabled events recycled between submissions.
    events: Mutex<Vec<sys::CUevent>>,
    /// Pinned staging for large host transfers, allocated on first use.
    staging: Mutex<Option<Staging>>,
    /// Captured batch graphs, keyed by the exact launch values of the batch.
    graphs: Mutex<GraphCache>,
}

// SAFETY: every context entry is serialized by ctx_lock.
unsafe impl Send for CudaShared {}
unsafe impl Sync for CudaShared {}

impl CudaShared {
    /// Run `f` with the context current on this thread. The context is only
    /// re-set when the thread's current context differs, so the common path is
    /// one `cuCtxGetCurrent`. The current context is never cleared.
    fn with_current<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let _guard = self.ctx_lock.lock().map_err(|_| GpuError::DeviceLost)?;
        let mut current: sys::CUcontext = ptr::null_mut();
        // SAFETY: plain thread-local query; writes one pointer.
        cu(unsafe { sys::cuCtxGetCurrent(&mut current) })?;
        if current != self.raw_ctx {
            // SAFETY: raw_ctx is the live context owned by `_ctx`; ctx_lock is held.
            cu(unsafe { sys::cuCtxSetCurrent(self.raw_ctx) })?;
        }
        f()
    }

    /// Take a pooled event, or create one. The context must be current.
    fn take_event(&self) -> Result<sys::CUevent> {
        if let Some(ev) = self.events.lock().map_err(|_| GpuError::DeviceLost)?.pop() {
            return Ok(ev);
        }
        let mut ev: sys::CUevent = ptr::null_mut();
        cu(unsafe {
            sys::cuEventCreate(&mut ev, sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING)
        })?;
        Ok(ev)
    }

    /// Return a finished event to the pool. The context must be current.
    fn return_event(&self, ev: sys::CUevent) {
        match self.events.lock() {
            Ok(mut pool) => pool.push(ev),
            Err(_) => {
                unsafe { sys::cuEventDestroy_v2(ev) };
            }
        }
    }

    /// Record a completion event on the stream. The context must be current.
    fn record_event(&self) -> Result<sys::CUevent> {
        let ev = self.take_event()?;
        if let Err(error) = cu(unsafe { sys::cuEventRecord(ev, self.stream) }) {
            self.return_event(ev);
            return Err(error);
        }
        Ok(ev)
    }

    /// Wait for everything queued on the stream, spinning rather than sleeping.
    /// The context must be current.
    fn sync_stream(&self) -> Result<()> {
        let ev = self.record_event()?;
        let waited = wait_event(ev);
        self.return_event(ev);
        waited
    }

    /// Issue `plan` on the stream in order. The context must be current and
    /// `ctx_lock` held. Serves direct launches and graph capture alike.
    fn issue(&self, plan: &[Launch]) -> Result<()> {
        for launch in plan {
            match *launch {
                Launch::Kernel {
                    func,
                    grid,
                    block,
                    mut params,
                    n_params,
                } => {
                    let mut kernel_params: [*mut c_void; MAX_PARAMS] =
                        [ptr::null_mut(); MAX_PARAMS];
                    for (i, slot) in params.iter_mut().take(n_params).enumerate() {
                        kernel_params[i] = slot.as_mut_ptr() as *mut c_void;
                    }
                    // SAFETY: each pointer aims into `params`, which lives across the call.
                    cu(unsafe {
                        sys::cuLaunchKernel(
                            func,
                            grid[0],
                            grid[1],
                            grid[2],
                            block[0],
                            block[1],
                            block[2],
                            0,
                            self.stream,
                            kernel_params.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    })?;
                }
                Launch::Copy { dst, src, len } => {
                    // SAFETY: both ranges were bounds-checked against their buffers when resolved.
                    cu(unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, len, self.stream) })?;
                }
            }
        }
        Ok(())
    }

    /// Record `plan` into a graph and instantiate it without executing anything.
    /// Returns None, with the capture ended, if any step fails. The context must be current.
    fn capture(&self, plan: &[Launch]) -> Option<sys::CUgraphExec> {
        // SAFETY: `stream` is this device's live stream and is not capturing.
        cu(unsafe {
            sys::cuStreamBeginCapture_v2(
                self.stream,
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        })
        .ok()?;
        let recorded = self.issue(plan).is_ok();
        let mut graph: sys::CUgraph = ptr::null_mut();
        // SAFETY: writes one graph handle into `graph`; this ends the capture on every path.
        let ended = cu(unsafe { sys::cuStreamEndCapture(self.stream, &mut graph) }).is_ok();
        if !recorded || !ended {
            if !graph.is_null() {
                // SAFETY: `graph` was produced by the EndCapture call above and is unused.
                unsafe { sys::cuGraphDestroy(graph) };
            }
            return None;
        }
        let mut exec: sys::CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is the complete capture; no error-node or log output is requested.
        let instantiated = cu(unsafe {
            sys::cuGraphInstantiate_v2(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0)
        })
        .is_ok();
        // SAFETY: the template graph is not needed once instantiation has run.
        unsafe { sys::cuGraphDestroy(graph) };
        instantiated.then_some(exec)
    }

    /// Launch `plan` for a batch of `op_count` ops. A batch seen before is captured
    /// on its second sighting and replayed from then on. The context must be
    /// current and `ctx_lock` held.
    fn launch_plan(&self, plan: &[Launch], op_count: usize) -> Result<()> {
        if op_count < GRAPH_MIN_OPS || plan.is_empty() {
            return self.issue(plan);
        }
        let key = plan_key(plan);
        let mut graphs = self.graphs.lock().map_err(|_| GpuError::DeviceLost)?;
        let entry = graphs.entries.get(&key).copied();
        match entry {
            Some(GraphEntry::Ready(exec)) => {
                drop(graphs);
                // SAFETY: `exec` is cached, so it is live; it is destroyed only under ctx_lock.
                cu(unsafe { sys::cuGraphLaunch(exec, self.stream) })
            }
            Some(GraphEntry::Uncachable) => {
                drop(graphs);
                self.issue(plan)
            }
            Some(GraphEntry::Seen) => {
                drop(graphs);
                let Some(exec) = self.capture(plan) else {
                    self.graphs
                        .lock()
                        .map_err(|_| GpuError::DeviceLost)?
                        .entries
                        .insert(key, GraphEntry::Uncachable);
                    return self.issue(plan);
                };
                match self.graphs.lock() {
                    Ok(mut graphs) => graphs.insert_ready(key, exec),
                    Err(_) => {
                        // SAFETY: nothing else holds `exec`; it is destroyed rather than leaked.
                        unsafe { sys::cuGraphExecDestroy(exec) };
                        return Err(GpuError::DeviceLost);
                    }
                }
                // SAFETY: `exec` was just instantiated and is cached, so it is live.
                cu(unsafe { sys::cuGraphLaunch(exec, self.stream) })
            }
            None => {
                graphs.note_seen(key);
                drop(graphs);
                self.issue(plan)
            }
        }
    }

    /// Destroy all cached graphs, e.g. before the kernels they reference are unloaded.
    /// The context must be current and `ctx_lock` held.
    fn clear_graphs(&self) {
        if let Ok(mut graphs) = self.graphs.lock() {
            graphs.clear();
        }
    }

    /// Copy host `src` to device address `dst`. The context must be current and
    /// `ctx_lock` held.
    fn htod(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.len() < STAGING_MIN {
            // SAFETY: src is a valid host slice of src.len() bytes; pageable
            // memory is staged before the call returns, and we sync below.
            cu(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    src.as_ptr() as *const c_void,
                    driver_len(src.len())?,
                    self.stream,
                )
            })?;
            return self.sync_stream();
        }
        let mut guard = self.staging.lock().map_err(|_| GpuError::DeviceLost)?;
        if guard.is_none() {
            *guard = Some(Staging::new()?);
        }
        let st = guard.as_mut().unwrap();
        for (i, chunk) in src.chunks(STAGING_CHUNK).enumerate() {
            let b = i % 2;
            if st.pending[b] {
                // The previous DMA reading bufs[b] must finish before we overwrite it.
                wait_event(st.events[b])?;
                st.pending[b] = false;
            }
            // SAFETY: bufs[b] is a live STAGING_CHUNK-byte pinned allocation, and
            // chunk.len() <= STAGING_CHUNK.
            unsafe { ptr::copy_nonoverlapping(chunk.as_ptr(), st.bufs[b] as *mut u8, chunk.len()) };
            let off = (i * STAGING_CHUNK) as u64;
            // SAFETY: the destination range was bounds-checked by the caller.
            cu(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst + off,
                    st.bufs[b] as *const c_void,
                    driver_len(chunk.len())?,
                    self.stream,
                )
            })?;
            cu(unsafe { sys::cuEventRecord(st.events[b], self.stream) })?;
            st.pending[b] = true;
        }
        // Return only once the device side is complete.
        for (pending, &ev) in st.pending.iter_mut().zip(&st.events) {
            if *pending {
                wait_event(ev)?;
                *pending = false;
            }
        }
        Ok(())
    }

    /// Copy device address `src` into host `dst`. The context must be current
    /// and `ctx_lock` held.
    fn dtoh(&self, src: u64, dst: &mut [u8]) -> Result<()> {
        if dst.len() < STAGING_MIN {
            // SAFETY: dst is a valid mutable host slice of dst.len() bytes and
            // stays borrowed until the stream sync below.
            cu(unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    dst.as_mut_ptr() as *mut c_void,
                    src,
                    driver_len(dst.len())?,
                    self.stream,
                )
            })?;
            return self.sync_stream();
        }
        let mut guard = self.staging.lock().map_err(|_| GpuError::DeviceLost)?;
        if guard.is_none() {
            *guard = Some(Staging::new()?);
        }
        let st = guard.as_mut().unwrap();
        let total = dst.len();
        // (offset, len) of chunk `j` within `dst`.
        let chunk = |j: usize| {
            let off = j * STAGING_CHUNK;
            (off, STAGING_CHUNK.min(total - off))
        };
        let chunks = total.div_ceil(STAGING_CHUNK);
        // Software pipeline: issue the DMA for chunk i, then copy chunk i-1 out
        // of its buffer while the DMA for chunk i runs.
        for i in 0..=chunks {
            if i < chunks {
                let b = i % 2;
                let (off, len) = chunk(i);
                if st.pending[b] {
                    // bufs[b] still holds chunk i-2 until its DMA completes.
                    wait_event(st.events[b])?;
                    st.pending[b] = false;
                }
                // SAFETY: bufs[b] is a live STAGING_CHUNK-byte pinned allocation
                // and len <= STAGING_CHUNK; the source range was bounds-checked.
                cu(unsafe {
                    sys::cuMemcpyDtoHAsync_v2(
                        st.bufs[b],
                        src + off as u64,
                        driver_len(len)?,
                        self.stream,
                    )
                })?;
                cu(unsafe { sys::cuEventRecord(st.events[b], self.stream) })?;
                st.pending[b] = true;
            }
            if i > 0 {
                let j = i - 1;
                let b = j % 2;
                let (off, len) = chunk(j);
                wait_event(st.events[b])?;
                st.pending[b] = false;
                // SAFETY: the DMA into bufs[b] completed (waited above), so the
                // first `len` bytes are initialised and not being written.
                let staged = unsafe { slice::from_raw_parts(st.bufs[b] as *const u8, len) };
                dst[off..off + len].copy_from_slice(staged);
            }
        }
        Ok(())
    }
}

impl Drop for CudaShared {
    fn drop(&mut self) {
        // SAFETY: `&mut self` gives exclusive access. `raw_ctx` is still alive
        // because `_ctx` is dropped only after this body.
        if unsafe { sys::cuCtxSetCurrent(self.raw_ctx) } == 0 {
            unsafe { sys::cuStreamSynchronize(self.stream) };
            // Graph execs and staging are released while the context is current.
            if let Ok(graphs) = self.graphs.get_mut() {
                graphs.clear();
            }
            if let Ok(staging) = self.staging.get_mut() {
                *staging = None;
            }
            if let Ok(pool) = self.events.get_mut() {
                for ev in pool.drain(..) {
                    unsafe { sys::cuEventDestroy_v2(ev) };
                }
            }
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

// SAFETY: see the CudaShared doc comment above.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

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
        // One context entry covers the query and, on completion, the pool return.
        let code = self.shared.with_current(|| {
            // SAFETY: `raw` is this submission's live event.
            let code = unsafe { sys::cuEventQuery(raw) };
            if code == 0 {
                self.shared.return_event(raw);
            }
            Ok(code)
        })?;
        if code == 0 {
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
            yield_now();
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
        // dropped: wait for its exact stream position, then recycle the event.
        let _ = self.shared.with_current(|| {
            wait_event(raw)?;
            self.shared.return_event(raw);
            Ok(())
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
            take(&mut lifetime.deferred)
        } else {
            Vec::new()
        }
    };
    for resource in deferred {
        match resource {
            DeferredCudaResource::Buffer(index, buffer) => {
                let _ = shared.with_current(|| cu(unsafe { sys::cuMemFree_v2(buffer.ptr) }));
                buffers.lock().unwrap().release_retired(index);
            }
            DeferredCudaResource::Shader(index, shader) => {
                let _ = shared.with_current(|| {
                    shared.clear_graphs();
                    cu(unsafe { sys::cuModuleUnload(shader.module) })
                });
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

        let _ = self.shared.with_current(|| {
            // A device may be dropped before its submission handles. Drain
            // work before freeing the resources owned by this device.
            cu(unsafe { sys::cuStreamSynchronize(self.shared.stream) })?;
            // Graphs reference kernels in the modules unloaded below.
            self.shared.clear_graphs();
            for cb in buffers {
                // SAFETY: ptr came from cuMemAlloc_v2 and is freed exactly once here.
                let _ = unsafe { sys::cuMemFree_v2(cb.ptr) };
            }
            let _ = pipelines; // CUfunction handles are owned by their modules; no explicit free.
            for cs in shaders {
                let _ = unsafe { sys::cuModuleUnload(cs.module) };
            }
            Ok(())
        });
    }
}

impl CudaDevice {
    /// Resolve a dispatch into its launch values. No driver call is made, so the
    /// context need not be current.
    fn resolve_dispatch(&self, op: &zengpu_hal::DispatchOp<'_>) -> Result<Launch> {
        let cp = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(op.pipeline)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid pipeline handle".into()))?;

        let mut params = [[0u8; 8]; MAX_PARAMS];
        let mut n_params = 0usize;
        {
            let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            for &slot in op.bindings.buffers {
                if n_params >= MAX_PARAMS {
                    return Err(GpuError::Dispatch(format!(
                        "dispatch: more than {MAX_PARAMS} kernel parameters"
                    )));
                }
                let cb = buffers.get_by_slot_index(slot).ok_or_else(|| {
                    GpuError::Backend("cuda: invalid buffer slot in bindings".into())
                })?;
                params[n_params] = cb.ptr.to_le_bytes();
                n_params += 1;
            }
        }
        for scalar in op.bindings.scalars {
            if n_params >= MAX_PARAMS {
                return Err(GpuError::Dispatch(format!(
                    "dispatch: more than {MAX_PARAMS} kernel parameters"
                )));
            }
            let bytes = match scalar {
                Scalar::U32(v) => v.to_le_bytes(),
                Scalar::I32(v) => v.to_le_bytes(),
                Scalar::F32(v) => v.to_bits().to_le_bytes(),
            };
            params[n_params][..4].copy_from_slice(&bytes);
            n_params += 1;
        }
        Ok(Launch::Kernel {
            func: cp.func,
            grid: op.grid,
            block: cp.block,
            params,
            n_params,
        })
    }

    /// Validate a buffer copy and resolve it. A zero-length copy needs no launch.
    fn resolve_copy(&self, copy: &zengpu_hal::BufferCopyOp) -> Result<Option<Launch>> {
        let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
        let src = buffers
            .get(copy.src)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid source buffer handle".into()))?;
        let dst = buffers
            .get(copy.dst)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid destination buffer handle".into()))?;
        if copy.src == copy.dst {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "copy_buffer requires distinct source and destination buffers".into(),
            )));
        }
        if !src.usage.contains(BufferUsage::TRANSFER_SRC) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "source buffer",
                needed: "TRANSFER_SRC",
            }));
        }
        if !dst.usage.contains(BufferUsage::TRANSFER_DST) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "destination buffer",
                needed: "TRANSFER_DST",
            }));
        }
        let src_end = copy.src_offset.checked_add(copy.len).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "source copy range overflows u64".into(),
            ))
        })?;
        let dst_end = copy.dst_offset.checked_add(copy.len).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "destination copy range overflows u64".into(),
            ))
        })?;
        if src_end > src.len || dst_end > dst.len {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "cuda: mixed-submission copy range exceeds buffer".into(),
            )));
        }
        if copy.len == 0 {
            return Ok(None);
        }
        let len = sys::size_t::try_from(copy.len).map_err(|_| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "cuda: copy exceeds driver byte-count range".into(),
            ))
        })?;
        Ok(Some(Launch::Copy {
            dst: dst.ptr + copy.dst_offset,
            src: src.ptr + copy.src_offset,
            len,
        }))
    }
}

impl GpuDevice for CudaDevice {
    fn backend(&self) -> BackendPreference {
        BackendPreference::Cuda
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.shared.with_current(|| {
            let mut ptr: u64 = 0;
            // SAFETY: writes one device pointer into `ptr`.
            cu(unsafe { sys::cuMemAlloc_v2(&mut ptr, driver_len(desc.size)?) })?;
            self.buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)
                .map(|mut buffers| {
                    buffers.insert(CudaBuffer {
                        ptr,
                        len: desc.size,
                        usage: desc.usage,
                    })
                })
        })
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.shared.with_current(|| {
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
            self.shared.htod(cb.ptr + offset, data)
        })
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let len = usize::try_from(len).map_err(|_| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "cuda: read length exceeds host address range".into(),
            ))
        })?;
        // Bounds-check before allocating, so a bogus length errors instead of aborting.
        let size = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .map(|cb| cb.len)
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;
        if offset.checked_add(len as u64).is_none_or(|end| end > size) {
            return Err(GpuError::Backend("cuda: read out of bounds".into()));
        }
        // Zeroed allocation is calloc-backed, so large reads do not pay a memset.
        let mut out = vec![0u8; len];
        self.read_buffer_into(buffer, offset, &mut out)?;
        Ok(out)
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.shared.with_current(|| {
            let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            let cb = buffers
                .get(buffer)
                .copied()
                .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;
            if !cb.usage.contains(BufferUsage::READBACK) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "buffer",
                    needed: "READBACK",
                }));
            }
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
            let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
                GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "cuda: read range overflows u64".into(),
                ))
            })?;
            if end > cb.len {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    format!("cuda: read range {offset}..{end} exceeds {}", cb.len),
                )));
            }
            self.shared.dtoh(cb.ptr + offset, dst)
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
        self.shared.with_current(|| {
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
            // Device-to-device copies return before completion even through the
            // synchronous API, so queue on our stream (ordered with the staged
            // transfers and dispatches) and wait for it.
            // SAFETY: both ranges were bounds-checked above and the buffers are distinct.
            cu(unsafe {
                sys::cuMemcpyDtoDAsync_v2(
                    dst_buf.ptr + dst_offset,
                    src_buf.ptr + src_offset,
                    driver_len(len)?,
                    self.shared.stream,
                )
            })?;
            self.shared.sync_stream()
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
            // SAFETY: cb.ptr came from cuMemAlloc_v2 and is no longer in the table.
            let _ = self
                .shared
                .with_current(|| cu(unsafe { sys::cuMemFree_v2(cb.ptr) }));
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
        let mut image = match desc.source {
            ShaderSource::Ptx(ptx) => ModuleImage::Ptx(ptx.to_vec()),
            ShaderSource::CudaSrc(src) => nvrtc_compile(src, self.shared.sm)?,
            _ => {
                return Err(GpuError::Backend(
                    "cuda: only PTX or CUDA C++ (CudaSrc) shaders are supported".into(),
                ));
            }
        };
        // cuModuleLoadData requires a null-terminated PTX string. Cubin is raw.
        if let ModuleImage::Ptx(bytes) = &mut image
            && bytes.last() != Some(&0)
        {
            bytes.push(0);
        }
        let module = self.shared.with_current(|| {
            let mut m: sys::CUmodule = ptr::null_mut();
            // SAFETY: `image` outlives the call (it is moved into the shader below).
            cu(unsafe { sys::cuModuleLoadData(&mut m, image.bytes().as_ptr() as *const c_void) })?;
            Ok(m)
        })?;
        let handle = self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaShader {
                module,
                _image: image,
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
            let _ = self.shared.with_current(|| {
                self.shared.clear_graphs();
                cu(unsafe { sys::cuModuleUnload(cs.module) })
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

        let entry = CString::new(desc.entry)
            .map_err(|_| GpuError::Backend("cuda: entry point name contains NUL byte".into()))?;

        let func = self.shared.with_current(|| {
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
        // Graphs captured from this pipeline hold its function handle.
        let _ = self.shared.with_current(|| {
            self.shared.clear_graphs();
            Ok(())
        });
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

        let stream = self.shared.stream;
        self.lifetime
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .in_flight += 1;
        // Same-stream launches execute and become globally visible in
        // submission order on the device, so a later op safely reads an
        // earlier op's output with no explicit barrier. The event recorded
        // after the final launch represents completion of the whole batch.
        let event = self.shared.with_current(|| {
            let (plan, resolve_error) = resolve_all(ops, |op| self.resolve_dispatch(op).map(Some));
            if let Some(error) = resolve_error {
                // Launch the resolved prefix, as the per-op loop did, then fail.
                self.shared.issue(&plan)?;
                return Err(error);
            }
            self.shared.launch_plan(&plan, ops.len())?;
            self.shared.record_event()
        });
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                // Some earlier launches may already be queued. Drain them
                // before releasing any concurrently retired resources.
                let _ = self
                    .shared
                    .with_current(|| cu(unsafe { sys::cuStreamSynchronize(stream) }));
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

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        let stream = self.shared.stream;
        self.lifetime
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .in_flight += 1;
        let event = self.shared.with_current(|| {
            let (plan, resolve_error) = resolve_all(ops, |op| match op {
                ComputeOp::CopyBuffer(copy) => self.resolve_copy(copy),
                ComputeOp::Dispatch(op) => self.resolve_dispatch(op).map(Some),
            });
            if let Some(error) = resolve_error {
                // Launch the resolved prefix, as the per-op loop did, then fail.
                self.shared.issue(&plan)?;
                return Err(error);
            }
            self.shared.launch_plan(&plan, ops.len())?;
            self.shared.record_event()
        });
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                let _ = self
                    .shared
                    .with_current(|| cu(unsafe { sys::cuStreamSynchronize(stream) }));
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
    use std::mem::{size_of, size_of_val};
    use std::thread;

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
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create_buffer failed");
        let data: Vec<u8> = (0u8..=255).collect();
        device.write_buffer(buf, 0, &data).expect("write failed");
        let read_back = device.read_buffer(buf, 0, 256).expect("read failed");
        assert_eq!(read_back, data);
        let mut caller_owned = [0u8; 32];
        device
            .read_buffer_into(buf, 17, &mut caller_owned)
            .expect("read into caller-owned storage failed");
        assert_eq!(caller_owned.as_slice(), &data[17..49]);
        device.destroy_buffer(buf);
    }

    #[test]
    fn buffer_offset_read_write() {
        let Some(device) = cuda_device() else { return };
        let buf = device
            .create_buffer(BufferDesc {
                size: 512,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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

        let size = (N * size_of::<f32>()) as u64;
        let buf_a = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_b = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_out = device
            .create_buffer(BufferDesc {
                size,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                    images: &[],
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
        let size = (N * size_of::<f32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                    images: &[],
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
        let size = (N * size_of::<f32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                    images: &[],
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
        let size = (N * size_of::<u32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                    images: &[],
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
        let size = size_of_val(&input) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                    images: &[],
                },
                [1, 1, 1],
            )
            .expect("dispatch finite CUDA kernel");

        let raw = device
            .read_buffer(out, 0, size)
            .expect("read classifications");
        let actual: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(actual, expected);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src);
        device.destroy_buffer(out);
    }

    #[test]
    fn mixed_copy_dispatch_submission_round_trips_device_local_data() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * 4) as u64;
        let upload = device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_SRC,
                memory: zengpu_hal::MemoryUsage::Upload,
            })
            .unwrap();
        let gpu_input = |size| BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let a = device.create_buffer(gpu_input(size)).unwrap();
        let b = device.create_buffer(gpu_input(size)).unwrap();
        let output = device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let readback = device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::Readback,
            })
            .unwrap();
        let values: Vec<f32> = (0..N).map(|i| i as f32 - 31.0).collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        device.write_buffer(upload, 0, &bytes).unwrap();
        let shader = device.create_shader(ShaderDesc::ptx(VEC_ADD_PTX)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "vec_add_f32",
                block: [256, 1, 1],
            })
            .unwrap();
        let copy_to = |dst| {
            ComputeOp::CopyBuffer(zengpu_hal::BufferCopyOp {
                src: upload,
                src_offset: 0,
                dst,
                dst_offset: 0,
                len: size,
            })
        };
        let submission = device
            .submit_compute_ops(
                0x4d_49_58_45_44,
                &[
                    copy_to(a),
                    copy_to(b),
                    ComputeOp::Dispatch(zengpu_hal::DispatchOp {
                        pipeline,
                        bindings: Bindings {
                            buffers: &[a.index(), b.index(), output.index()],
                            scalars: &[Scalar::U32(N as u32)],
                            textures: &[],
                            images: &[],
                        },
                        grid: [1, 1, 1],
                    }),
                    ComputeOp::CopyBuffer(zengpu_hal::BufferCopyOp {
                        src: output,
                        src_offset: 0,
                        dst: readback,
                        dst_offset: 0,
                        len: size,
                    }),
                ],
            )
            .unwrap();
        assert_eq!(submission.cycle_id(), 0x4d_49_58_45_44);
        submission.wait(Duration::from_secs(5)).unwrap();
        let actual = device.read_buffer(readback, 0, size).unwrap();
        for (index, bytes) in actual.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(f32::from_le_bytes(*bytes), values[index] * 2.0);
        }
        device.destroy_buffer(upload);
        device.destroy_buffer(a);
        device.destroy_buffer(b);
        device.destroy_buffer(output);
        device.destroy_buffer(readback);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
    }

    #[test]
    fn submission_batch_chains_ops_on_one_stream() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (10 * i) as f32).collect();
        // sum = a + b; doubled = sum + sum, batched as one submission.
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 2.0 * (x + y)).collect();

        let size = (N * size_of::<f32>()) as u64;
        let gpu = |size| BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
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
                            images: &[],
                        },
                        grid,
                    },
                    zengpu_hal::DispatchOp {
                        pipeline,
                        bindings: Bindings {
                            buffers: &[buf_sum.index(), buf_sum.index(), buf_out.index()],
                            scalars: &[Scalar::U32(N as u32)],
                            textures: &[],
                            images: &[],
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
                    images: &[],
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

    /// Staged transfers: a size that is not a multiple of STAGING_CHUNK and an
    /// odd offset, through write_buffer, read_buffer and read_buffer_into.
    #[test]
    fn staged_transfers_round_trip_unaligned_size_and_offset() {
        let Some(device) = cuda_device() else { return };
        const OFFSET: u64 = 7;
        const LEN: usize = 3 * STAGING_CHUNK + 12345;
        let buf = device
            .create_buffer(BufferDesc {
                size: OFFSET + LEN as u64,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create staged buffer");
        let data: Vec<u8> = (0..LEN)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 13)) as u8)
            .collect();
        device
            .write_buffer(buf, OFFSET, &data)
            .expect("staged write");
        assert_eq!(
            device
                .read_buffer(buf, OFFSET, LEN as u64)
                .expect("staged read"),
            data
        );
        let mut into = vec![0u8; LEN];
        device
            .read_buffer_into(buf, OFFSET, &mut into)
            .expect("staged read_into");
        assert_eq!(into, data);
        device.destroy_buffer(buf);
    }

    /// The first call on a thread with no current context must still work, and
    /// the device must remain usable from a different thread afterwards.
    #[test]
    fn transfers_work_from_another_thread() {
        let Some(device) = cuda_device() else { return };
        let buf = device
            .create_buffer(BufferDesc {
                size: 4096,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create buffer");
        let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        thread::scope(|scope| {
            scope.spawn(|| {
                device
                    .write_buffer(buf, 0, &data)
                    .expect("write off-thread")
            });
        });
        let back = thread::scope(|scope| {
            scope
                .spawn(|| device.read_buffer(buf, 0, 4096).expect("read off-thread"))
                .join()
                .expect("reader thread")
        });
        assert_eq!(back, data);
        device.destroy_buffer(buf);
    }

    /// NVRTC should target the device's own architecture, so the module is a
    /// cubin that loads without driver JIT.
    #[test]
    fn nvrtc_targets_device_architecture() {
        let inst = CudaInstance::new();
        if !inst.initialized {
            return;
        }
        let Ok(devices) = Cuda::list_devices() else {
            return;
        };
        let Some(dev) = devices.first() else {
            return;
        };
        let sm = dev
            .compute_capability()
            .map(|v| (v.major, v.minor))
            .expect("compute capability");
        let image = nvrtc_compile(VEC_ADD_CUDA.as_bytes(), sm).expect("compile for device");
        assert!(
            matches!(image, ModuleImage::Cubin(_)),
            "expected an sm_{}{} cubin",
            sm.0,
            sm.1
        );
    }

    const ACCUM_CUDA: &str = r#"
extern "C" __global__ void accum(float* out, float v, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += v;
    }
}
"#;

    const SCALE_ADD_CUDA: &str = r#"
extern "C" __global__ void scale_add(float* out, float v, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = out[i] * 2.0f + v;
    }
}
"#;

    /// Device buffer that can be read back, copied from and dispatched on.
    fn graph_buffer(device: &dyn GpuDevice, size: u64) -> BufferHandle {
        device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap()
    }

    fn read_f32(device: &dyn GpuDevice, buffer: BufferHandle, n: usize) -> Vec<f32> {
        let bytes = device
            .read_buffer(buffer, 0, (n * size_of::<f32>()) as u64)
            .unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// Submit one `accum` dispatch per entry of `adds`, all over `out`, and wait.
    fn submit_accum(
        device: &dyn GpuDevice,
        pipeline: PipelineHandle,
        out: BufferHandle,
        adds: &[f32],
        n: u32,
    ) {
        let slots = [out.index()];
        let scalars: Vec<[Scalar; 2]> = adds
            .iter()
            .map(|&v| [Scalar::F32(v), Scalar::U32(n)])
            .collect();
        let ops: Vec<zengpu_hal::DispatchOp<'_>> = scalars
            .iter()
            .map(|s| zengpu_hal::DispatchOp {
                pipeline,
                bindings: Bindings {
                    buffers: &slots,
                    scalars: s,
                    textures: &[],
                    images: &[],
                },
                grid: [n.div_ceil(64), 1, 1],
            })
            .collect();
        device
            .submit_batch(0, &ops)
            .unwrap()
            .wait(Duration::from_secs(5))
            .unwrap();
    }

    #[test]
    fn graph_replay_matches_direct_launches() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        let out = graph_buffer(&*device, size);
        let init: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
        device.write_buffer(out, 0, &init).unwrap();
        let shader = device
            .create_shader(ShaderDesc::cuda_src(ACCUM_CUDA))
            .expect("compile accum");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "accum",
                block: [64, 1, 1],
            })
            .expect("create pipeline");

        // Five identical batches cross Seen -> Ready -> replay.
        let adds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..5 {
            submit_accum(&*device, pipeline, out, &adds, N as u32);
        }
        let mut expected: Vec<f32> = (0..N).map(|i| i as f32 + 5.0 * 36.0).collect();
        assert_eq!(read_f32(&*device, out, N), expected);

        // A changed scalar is a different batch: it misses, then is captured on its second run.
        let changed = [100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..3 {
            submit_accum(&*device, pipeline, out, &changed, N as u32);
        }
        for v in &mut expected {
            *v += 3.0 * 135.0;
        }
        assert_eq!(read_f32(&*device, out, N), expected);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(out);
    }

    #[test]
    fn graph_cache_is_invalidated_when_pipeline_is_destroyed() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        let out = graph_buffer(&*device, size);
        let init: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
        device.write_buffer(out, 0, &init).unwrap();
        let adds = [1.0, 2.0, 3.0, 4.0];

        let shader = device
            .create_shader(ShaderDesc::cuda_src(ACCUM_CUDA))
            .expect("compile accum");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "accum",
                block: [64, 1, 1],
            })
            .expect("create pipeline");
        // Three runs: the batch is captured and replayed before it is invalidated.
        for _ in 0..3 {
            submit_accum(&*device, pipeline, out, &adds, N as u32);
        }
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);

        // Same grid and buffers, different kernel: a stale replay would add instead of scale.
        let shader = device
            .create_shader(ShaderDesc::cuda_src(SCALE_ADD_CUDA))
            .expect("compile scale_add");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "scale_add",
                block: [64, 1, 1],
            })
            .expect("create pipeline");
        for _ in 0..3 {
            submit_accum(&*device, pipeline, out, &adds, N as u32);
        }

        let mut expected: Vec<f32> = (0..N).map(|i| i as f32 + 3.0 * 10.0).collect();
        for _ in 0..3 {
            for &v in &adds {
                for x in &mut expected {
                    *x = *x * 2.0 + v;
                }
            }
        }
        assert_eq!(read_f32(&*device, out, N), expected);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(out);
    }

    #[test]
    fn mixed_copy_and_dispatch_batch_replays_correctly() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        let base = device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let acc = graph_buffer(&*device, size);
        let init: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
        device.write_buffer(base, 0, &init).unwrap();

        let shader = device
            .create_shader(ShaderDesc::cuda_src(ACCUM_CUDA))
            .expect("compile accum");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "accum",
                block: [64, 1, 1],
            })
            .expect("create pipeline");

        // Each run re-copies `base` into `acc` and adds 1 + 2 + 3 to it, so the
        // result is the same every time only if the copy is replayed too.
        let slots = [acc.index()];
        let scalars: Vec<[Scalar; 2]> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| [Scalar::F32(v), Scalar::U32(N as u32)])
            .collect();
        let copy = zengpu_hal::BufferCopyOp {
            src: base,
            src_offset: 0,
            dst: acc,
            dst_offset: 0,
            len: size,
        };
        let mut ops = vec![ComputeOp::CopyBuffer(copy)];
        ops.extend(scalars.iter().map(|s| {
            ComputeOp::Dispatch(zengpu_hal::DispatchOp {
                pipeline,
                bindings: Bindings {
                    buffers: &slots,
                    scalars: s,
                    textures: &[],
                    images: &[],
                },
                grid: [(N as u32).div_ceil(64), 1, 1],
            })
        }));

        let expected: Vec<f32> = (0..N).map(|i| i as f32 + 6.0).collect();
        for _ in 0..3 {
            device
                .submit_compute_ops(0, &ops)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            assert_eq!(read_f32(&*device, acc, N), expected);
        }

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(base);
        device.destroy_buffer(acc);
    }
}
