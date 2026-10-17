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
use std::mem::{take, zeroed};
use std::sync::{Arc, Mutex};
use std::thread::{self, yield_now};
use std::time::{Duration, Instant};
use std::{ptr, slice};

use cuda_oxide::{Cuda, context::Context, device::Device, sys};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, CompletedReadback, CompletedSubmission, ComputeOp, ComputePipelineDesc,
    DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, GpuReadback,
    GpuSubmission, HalCapabilities, PipelineHandle, ReadSubmission, Result, SamplerDesc,
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
        let (streams, raw_ctx) = {
            let _handle = from_cuda(ctx.enter())?;
            // SAFETY: the context was just entered on this thread, so the query returns it.
            let mut raw: sys::CUcontext = ptr::null_mut();
            cu(unsafe { sys::cuCtxGetCurrent(&mut raw) })?;
            let mut streams = [ptr::null_mut(); 3];
            for stream in &mut streams {
                // On error the streams already created leak with the context; the context is torn down.
                cu(unsafe {
                    sys::cuStreamCreate(stream, sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING)
                })?;
            }
            (streams, raw)
        };
        let [stream, upload_stream, download_stream] = streams;
        Ok(Box::new(CudaDevice {
            shared: Arc::new(CudaShared {
                _ctx: ctx,
                raw_ctx,
                ctx_lock: Mutex::new(()),
                stream,
                upload_stream,
                download_stream,
                sm: (sm.major, sm.minor),
                events: Arc::new(EventPool(Mutex::new(Vec::new()))),
                staging: Mutex::new(None),
                pinned: Arc::new(PinnedPool(Mutex::new(Vec::new()))),
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
    /// CUDA C++ text the module was compiled from, read for kernel signatures.
    source: Option<String>,
}

// SAFETY: CUmodule is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaShader {}
unsafe impl Sync for CudaShader {}

// ── CudaPipeline ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaPipeline {
    func: sys::CUfunction,
    block: [u32; 3],
    /// Number of pointer parameters in the kernel signature. Zero when it was not readable.
    ptr_params: usize,
    /// Bit `i` is set when pointer parameter `i` may be written. See `writes_buffer`.
    writes: u32,
}

// SAFETY: CUfunction is an opaque handle; all access is serialised through ctx_lock.
unsafe impl Send for CudaPipeline {}
unsafe impl Sync for CudaPipeline {}

enum DeferredCudaResource {
    Buffer(u32, CudaBuffer),
    Shader(u32, CudaShader),
    Pipeline(u32),
}

/// Last accesses to one buffer, by base device pointer. Each event is a
/// completion point that covers every earlier access to the buffer, so only the
/// latest read and the latest write need to be remembered.
#[derive(Default)]
struct BufferHazard {
    last_access: Option<Access>,
    last_write: Option<Access>,
}

#[derive(Clone)]
struct Access {
    event: Arc<OwnedEvent>,
    /// The stream the event was recorded on.
    lane: Lane,
}

/// The streams work is queued on. Work on one lane is ordered; ordering across
/// lanes needs an event wait.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Compute,
    Upload,
    Download,
}

#[derive(Default)]
struct CudaLifetimeState {
    in_flight: usize,
    deferred: Vec<DeferredCudaResource>,
    hazards: HashMap<u64, BufferHazard>,
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

/// Host copies at least this large are split across threads. One core cannot keep
/// up with the copy engines, which move 13 GB/s or more.
const PARALLEL_COPY_MIN: usize = 8 << 20;

/// Threads used for a large host copy.
const PARALLEL_COPY_THREADS: usize = 4;

/// Copy `src` into `dst` (same length), in parallel when the copy is large.
fn copy_host(dst: &mut [u8], src: &[u8]) {
    if dst.len() < PARALLEL_COPY_MIN {
        dst.copy_from_slice(src);
        return;
    }
    let chunk = dst.len().div_ceil(PARALLEL_COPY_THREADS);
    thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || d.copy_from_slice(s));
        }
    });
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

/// Recycled timing-disabled events. Shared with every `OwnedEvent`, so an event
/// returns here when its last reference drops.
struct EventPool(Mutex<Vec<sys::CUevent>>);

// SAFETY: CUevent handles are opaque and the pool is only touched under its mutex.
unsafe impl Send for EventPool {}
unsafe impl Sync for EventPool {}

impl EventPool {
    fn put(&self, ev: sys::CUevent) {
        match self.0.lock() {
            Ok(mut pool) => pool.push(ev),
            Err(_) => {
                unsafe { sys::cuEventDestroy_v2(ev) };
            }
        }
    }
}

impl Drop for EventPool {
    fn drop(&mut self) {
        for ev in self
            .0
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            unsafe { sys::cuEventDestroy_v2(ev) };
        }
    }
}

/// A recorded event that is recycled only after its last holder drops it, so a
/// buffer hazard and a submission can both refer to one stream position.
struct OwnedEvent {
    ev: sys::CUevent,
    pool: Arc<EventPool>,
}

// SAFETY: the event is an opaque handle; it is only recorded and queried in the serialized context.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

impl OwnedEvent {
    /// Record a new event on `stream`. The context must be current.
    fn record(shared: &CudaShared, stream: sys::CUstream) -> Result<Arc<Self>> {
        let ev = shared.take_event()?;
        if let Err(error) = cu(unsafe { sys::cuEventRecord(ev, stream) }) {
            shared.events.put(ev);
            return Err(error);
        }
        Ok(Arc::new(OwnedEvent {
            ev,
            pool: Arc::clone(&shared.events),
        }))
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        self.pool.put(self.ev);
    }
}

/// Pinned host memory block used to stage one asynchronous transfer.
#[derive(Clone, Copy)]
struct PinnedBlock {
    ptr: *mut c_void,
    cap: usize,
}

// SAFETY: a pinned block is plain host memory; access is ordered by the events of the transfer using it.
unsafe impl Send for PinnedBlock {}
unsafe impl Sync for PinnedBlock {}

/// Free pinned blocks for asynchronous transfers. Blocks are leased per transfer
/// and reused, so pinned allocation is not paid on every call.
struct PinnedPool(Mutex<Vec<PinnedBlock>>);

impl PinnedPool {
    /// Lease a block of at least `len` bytes. The context must be current.
    fn lease(self: &Arc<Self>, len: usize) -> Result<PinnedLease> {
        let reused = {
            let mut free = self.0.lock().map_err(|_| GpuError::DeviceLost)?;
            let best = free
                .iter()
                .enumerate()
                .filter(|(_, b)| b.cap >= len)
                .min_by_key(|(_, b)| b.cap)
                .map(|(i, _)| i);
            best.map(|i| free.swap_remove(i))
        };
        let block = match reused {
            Some(block) => block,
            None => {
                let mut ptr: *mut c_void = ptr::null_mut();
                // SAFETY: writes one pinned allocation pointer into `ptr`.
                cu(unsafe { sys::cuMemHostAlloc(&mut ptr, driver_len(len)?, 0) })?;
                PinnedBlock { ptr, cap: len }
            }
        };
        Ok(PinnedLease {
            block,
            pool: Arc::clone(self),
        })
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        // Only reached with the context current (see CudaShared::drop), or on error paths
        // where the driver call simply fails.
        for block in self
            .0
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            unsafe { sys::cuMemFreeHost(block.ptr) };
        }
    }
}

/// A pinned block held until the transfer using it has finished and, for a
/// readback, its bytes have been taken. Returns the block to its pool on drop.
struct PinnedLease {
    block: PinnedBlock,
    pool: Arc<PinnedPool>,
}

impl Drop for PinnedLease {
    fn drop(&mut self) {
        if let Ok(mut free) = self.pool.0.lock() {
            free.push(self.block);
        }
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
        /// The first `n_buffers` parameters are buffer base pointers, the rest scalars.
        n_buffers: usize,
        /// Written-buffer bits, indexed like the buffer parameters. See `writes_buffer`.
        writes: u32,
    },
    Copy {
        dst: u64,
        src: u64,
        len: sys::size_t,
        /// Base pointers of the whole buffers, so a sub-range maps to its buffer.
        dst_buf: u64,
        src_buf: u64,
    },
}

/// An instantiated batch graph. The template `graph` is kept alive beside `exec`
/// because the node handles that parameter updates need belong to it.
struct Replay {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    /// One entry per op, in stream order, with the values its node was last set to.
    nodes: Vec<(sys::CUgraphNode, Launch)>,
}

impl Replay {
    /// Destroy the exec and its template. The context must be current.
    fn destroy(self) {
        // SAFETY: both handles came from `CudaShared::build_graph` and leave the cache here, once.
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}

enum GraphEntry {
    /// Launched once directly; the next batch with the same structure is built.
    Seen,
    /// Instantiated graph, replayed with `cuGraphLaunch` after patching changed nodes.
    Ready(Replay),
    /// Build or patch failed; always launched directly.
    Uncachable,
}

/// Per-device graph cache. Keys are the structure of a batch (see `plan_key`), so a
/// hit replays the same kernels and copies with whatever values the batch carries.
#[derive(Default)]
struct GraphCache {
    entries: HashMap<Vec<u64>, GraphEntry>,
    ready: usize,
}

impl GraphCache {
    /// Destroy every instantiated graph and forget all entries. The context must be current.
    fn clear(&mut self) {
        for (_, entry) in self.entries.drain() {
            if let GraphEntry::Ready(replay) = entry {
                replay.destroy();
            }
        }
        self.ready = 0;
    }

    /// Store a freshly built graph, evicting one at capacity. The context must be current.
    fn insert_ready(&mut self, key: Vec<u64>, replay: Replay) {
        if self.ready >= GRAPH_MAX_READY {
            let victim = self
                .entries
                .iter()
                .find(|(_, entry)| matches!(entry, GraphEntry::Ready(_)))
                .map(|(key, _)| key.clone());
            if let Some(victim) = victim
                && let Some(GraphEntry::Ready(old)) = self.entries.remove(&victim)
            {
                old.destroy();
                self.ready -= 1;
            }
        }
        self.entries.insert(key, GraphEntry::Ready(replay));
        self.ready += 1;
    }

    /// Replace `key`'s entry with `Uncachable`, destroying any graph it held. The context must be current.
    fn disable(&mut self, key: Vec<u64>) {
        if let Some(GraphEntry::Ready(replay)) = self.entries.insert(key, GraphEntry::Uncachable) {
            replay.destroy();
            self.ready -= 1;
        }
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

/// Whether pointer parameter `i` of a kernel may be written. Parameters past bit 31 are
/// assumed to be written.
fn writes_buffer(writes: u32, i: usize) -> bool {
    i >= 32 || (writes >> i) & 1 != 0
}

/// Buffers a launch touches, as `(base device pointer, written)`, in parameter order.
fn launch_accesses(launch: &Launch, out: &mut Vec<(u64, bool)>) {
    out.clear();
    match launch {
        Launch::Kernel {
            params,
            n_buffers,
            writes,
            ..
        } => {
            for (i, slot) in params.iter().enumerate().take(*n_buffers) {
                out.push((u64::from_le_bytes(*slot), writes_buffer(*writes, i)));
            }
        }
        Launch::Copy {
            src_buf, dst_buf, ..
        } => {
            out.push((*src_buf, false));
            out.push((*dst_buf, true));
        }
    }
}

/// Flatten the structure of a resolved batch: everything a graph's topology depends on.
/// Pointers, scalar values and copy addresses are left out; they are patched into the
/// nodes when they change. Buffers are replaced by their first-occurrence number in the
/// batch, so the key also records which launches share a buffer.
fn plan_key(plan: &[Launch]) -> Vec<u64> {
    const KERNEL_TAG: u64 = 0;
    const COPY_TAG: u64 = 1;
    let mut key = Vec::new();
    let mut ids: HashMap<u64, u64> = HashMap::new();
    let mut accesses = Vec::new();
    let mut id_of = |base: u64| {
        let next = ids.len() as u64;
        *ids.entry(base).or_insert(next)
    };
    for launch in plan {
        match launch {
            Launch::Kernel {
                func,
                grid,
                block,
                n_params,
                n_buffers,
                writes,
                ..
            } => {
                key.push(KERNEL_TAG);
                key.push(*func as usize as u64);
                key.extend(grid.iter().chain(block).map(|&v| u64::from(v)));
                key.extend([*n_params as u64, *n_buffers as u64, u64::from(*writes)]);
                launch_accesses(launch, &mut accesses);
                key.extend(accesses.iter().map(|&(base, _)| id_of(base)));
            }
            Launch::Copy {
                len,
                dst_buf,
                src_buf,
                ..
            } => {
                key.extend([COPY_TAG, *len as u64, id_of(*src_buf), id_of(*dst_buf)]);
            }
        }
    }
    key
}

/// For each launch of `plan`, the earlier launches it must wait for. A launch depends on
/// the last writer of every buffer it touches (RAW and WAW), and a writer also depends on
/// every reader of its buffer since that writer (WAR). Indices are sorted and unique.
fn launch_deps(plan: &[Launch]) -> Vec<Vec<usize>> {
    #[derive(Default)]
    struct Resource {
        last_writer: Option<usize>,
        readers: Vec<usize>,
    }
    let mut resources: HashMap<u64, Resource> = HashMap::new();
    let mut accesses = Vec::new();
    let mut all_deps = Vec::with_capacity(plan.len());
    for (i, launch) in plan.iter().enumerate() {
        launch_accesses(launch, &mut accesses);
        let mut deps = Vec::new();
        for &(base, written) in &accesses {
            if let Some(res) = resources.get(&base) {
                deps.extend(res.last_writer);
                if written {
                    deps.extend(res.readers.iter().copied());
                }
            }
        }
        deps.sort_unstable();
        deps.dedup();
        all_deps.push(deps);
        for &(base, written) in &accesses {
            let res = resources.entry(base).or_default();
            if written {
                res.last_writer = Some(i);
                res.readers.clear();
            } else if !accesses.iter().any(|&(b, w)| b == base && w) {
                res.readers.push(i);
            }
        }
    }
    all_deps
}

/// Parse the pointer parameters of kernel `entry` in CUDA C++ `src`. Each entry is whether
/// that parameter may be written: a pointer is read-only only if `const` precedes its `*`.
/// Returns None when the signature cannot be read reliably: no such kernel, or a parameter
/// list with templates, macros, arrays or function pointers.
fn kernel_pointer_writes(src: &str, entry: &str) -> Option<Vec<bool>> {
    const GLOBAL: &str = "__global__";
    let mut from = 0;
    while let Some(pos) = src[from..].find(GLOBAL) {
        let start = from + pos + GLOBAL.len();
        from = start;
        let header_end = src[start..].find('{').map_or(src.len(), |end| start + end);
        let header = &src[start..header_end];
        let Some(open) = find_call_open(header, entry) else {
            continue;
        };
        let body = &header[open + 1..];
        let mut depth = 0usize;
        let close = body.char_indices().find_map(|(i, c)| match c {
            '(' => {
                depth += 1;
                None
            }
            ')' if depth == 0 => Some(i),
            ')' => {
                depth -= 1;
                None
            }
            _ => None,
        })?;
        let params = &body[..close];
        if params.contains(['<', '>', '#', '[', '(']) {
            return None;
        }
        let mut writes = Vec::new();
        for param in params.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let Some(star) = param.find('*') else {
                continue;
            };
            let read_only = param[..star]
                .split(|c: char| !is_ident_char(c))
                .any(|word| word == "const");
            writes.push(!read_only);
        }
        return Some(writes);
    }
    None
}

/// Byte offset of the `(` that follows the whole identifier `name` in `text`, if any.
fn find_call_open(text: &str, name: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(pos) = text[from..].find(name) {
        let at = from + pos;
        from = at + name.len();
        let whole = text[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !is_ident_char(c));
        let rest = text[from..].trim_start();
        if whole && rest.starts_with('(') {
            return Some(text.len() - rest.len());
        }
    }
    None
}

/// Whether `c` can appear in a C identifier.
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether two launches with the same structure carry the same values.
fn same_values(a: &Launch, b: &Launch) -> bool {
    match (a, b) {
        (
            Launch::Kernel {
                params: pa,
                n_params: na,
                ..
            },
            Launch::Kernel {
                params: pb,
                n_params: nb,
                ..
            },
        ) => na == nb && pa[..*na] == pb[..*nb],
        (
            Launch::Copy {
                dst: da,
                src: sa,
                len: la,
                ..
            },
            Launch::Copy {
                dst: db,
                src: sb,
                len: lb,
                ..
            },
        ) => da == db && sa == sb && la == lb,
        _ => false,
    }
}

/// Pointers to the first `n_params` slots of `params`, in the form a kernel launch
/// takes them. They are only valid while `params` is.
fn kernel_arg_ptrs(
    params: &mut [[u8; 8]; MAX_PARAMS],
    n_params: usize,
) -> [*mut c_void; MAX_PARAMS] {
    let mut ptrs = [ptr::null_mut(); MAX_PARAMS];
    for (arg, slot) in ptrs.iter_mut().zip(params.iter_mut()).take(n_params) {
        *arg = slot.as_mut_ptr() as *mut c_void;
    }
    ptrs
}

/// Kernel node parameters for `func` launched with the argument list `args`.
fn kernel_node_params(
    func: sys::CUfunction,
    grid: [u32; 3],
    block: [u32; 3],
    args: *mut *mut c_void,
) -> sys::CUDA_KERNEL_NODE_PARAMS {
    sys::CUDA_KERNEL_NODE_PARAMS {
        func,
        gridDimX: grid[0],
        gridDimY: grid[1],
        gridDimZ: grid[2],
        blockDimX: block[0],
        blockDimY: block[1],
        blockDimZ: block[2],
        sharedMemBytes: 0,
        kernelParams: args,
        extra: ptr::null_mut(),
    }
}

/// Device-to-device copy node parameters for `len` bytes from `src` to `dst`.
fn copy_node_params(dst: u64, src: u64, len: sys::size_t) -> sys::CUDA_MEMCPY3D {
    // SAFETY: CUDA_MEMCPY3D is plain old data, and all-zero is a valid value for it.
    let mut copy: sys::CUDA_MEMCPY3D = unsafe { zeroed() };
    copy.srcMemoryType = sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = src;
    copy.dstMemoryType = sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = dst;
    copy.WidthInBytes = len;
    copy.Height = 1;
    copy.Depth = 1;
    copy
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
    /// Non-blocking streams for asynchronous transfers, one per direction, so
    /// uploads and readbacks run on the copy engines alongside kernels on `stream`.
    /// A readback waiting on a kernel does not hold up the next upload.
    upload_stream: sys::CUstream,
    download_stream: sys::CUstream,
    /// Compute capability, used as the NVRTC target.
    sm: (u32, u32),
    /// Timing-disabled events recycled between submissions.
    events: Arc<EventPool>,
    /// Pinned staging for large host transfers, allocated on first use.
    staging: Mutex<Option<Staging>>,
    /// Pinned blocks leased by asynchronous transfers.
    pinned: Arc<PinnedPool>,
    /// Instantiated batch graphs, keyed by the structure of the batch.
    graphs: Mutex<GraphCache>,
}

// SAFETY: every context entry is serialized by ctx_lock.
unsafe impl Send for CudaShared {}
unsafe impl Sync for CudaShared {}

impl CudaShared {
    /// The stream a lane's work is queued on.
    fn stream(&self, lane: Lane) -> sys::CUstream {
        match lane {
            Lane::Compute => self.stream,
            Lane::Upload => self.upload_stream,
            Lane::Download => self.download_stream,
        }
    }

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
        if let Some(ev) = self
            .events
            .0
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .pop()
        {
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
        self.events.put(ev);
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
    /// `ctx_lock` held.
    fn issue(&self, plan: &[Launch]) -> Result<()> {
        for launch in plan {
            match *launch {
                Launch::Kernel {
                    func,
                    grid,
                    block,
                    mut params,
                    n_params,
                    ..
                } => {
                    let mut kernel_params = kernel_arg_ptrs(&mut params, n_params);
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
                Launch::Copy { dst, src, len, .. } => {
                    // SAFETY: both ranges were bounds-checked against their buffers when resolved.
                    cu(unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, len, self.stream) })?;
                }
            }
        }
        Ok(())
    }

    /// Build `plan` into a graph and instantiate it without executing anything. Each node
    /// depends only on the earlier nodes it shares a buffer hazard with (see `launch_deps`),
    /// so independent launches may run concurrently. Returns None, with nothing left behind,
    /// if any step fails. The context must be current.
    fn build_graph(&self, plan: &[Launch]) -> Option<Replay> {
        let mut graph: sys::CUgraph = ptr::null_mut();
        // SAFETY: writes one graph handle into `graph`; no flags are set.
        cu(unsafe { sys::cuGraphCreate(&mut graph, 0) }).ok()?;
        let mut nodes = Vec::with_capacity(plan.len());
        let added = self.add_nodes(graph, plan, &mut nodes).is_ok();
        let mut exec: sys::CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is complete when added; no error-node or log output is requested.
        let instantiated = added
            && cu(unsafe {
                sys::cuGraphInstantiate_v2(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0)
            })
            .is_ok();
        if !instantiated {
            // SAFETY: `graph` is not referenced again after this point.
            unsafe { sys::cuGraphDestroy(graph) };
            return None;
        }
        Some(Replay { graph, exec, nodes })
    }

    /// Add one node per op to `graph`, each depending on the nodes `launch_deps` gives for
    /// it, and record the values each node is given. The context must be current.
    fn add_nodes(
        &self,
        graph: sys::CUgraph,
        plan: &[Launch],
        nodes: &mut Vec<(sys::CUgraphNode, Launch)>,
    ) -> Result<()> {
        for (launch, dep_ids) in plan.iter().zip(launch_deps(plan)) {
            let deps: Vec<sys::CUgraphNode> = dep_ids.iter().map(|&j| nodes[j].0).collect();
            let dep_count = driver_len(deps.len())?;
            let mut node: sys::CUgraphNode = ptr::null_mut();
            match *launch {
                Launch::Kernel {
                    func,
                    grid,
                    block,
                    mut params,
                    n_params,
                    ..
                } => {
                    let mut kernel_params = kernel_arg_ptrs(&mut params, n_params);
                    let node_params =
                        kernel_node_params(func, grid, block, kernel_params.as_mut_ptr());
                    // SAFETY: `deps` and `node_params` live across the call, and the
                    // parameter values are copied into the node.
                    cu(unsafe {
                        sys::cuGraphAddKernelNode(
                            &mut node,
                            graph,
                            deps.as_ptr(),
                            dep_count,
                            &node_params,
                        )
                    })?;
                }
                Launch::Copy { dst, src, len, .. } => {
                    let copy = copy_node_params(dst, src, len);
                    // SAFETY: `deps` and `copy` live across the call; both ranges were
                    // bounds-checked against their buffers when resolved.
                    cu(unsafe {
                        sys::cuGraphAddMemcpyNode(
                            &mut node,
                            graph,
                            deps.as_ptr(),
                            dep_count,
                            &copy,
                            self.raw_ctx,
                        )
                    })?;
                }
            }
            nodes.push((node, *launch));
        }
        Ok(())
    }

    /// Bring the nodes of `replay` up to `plan`, which has the same structure, by setting
    /// only the nodes whose values changed. The exec is updated in place, so it keeps its
    /// topology and enqueued launches are unaffected. The context must be current.
    fn patch_nodes(&self, replay: &mut Replay, plan: &[Launch]) -> Result<()> {
        let Replay { exec, nodes, .. } = replay;
        debug_assert_eq!(nodes.len(), plan.len());
        for ((node, applied), launch) in nodes.iter_mut().zip(plan) {
            if same_values(applied, launch) {
                continue;
            }
            match *launch {
                Launch::Kernel {
                    func,
                    grid,
                    block,
                    mut params,
                    n_params,
                    ..
                } => {
                    let mut kernel_params = kernel_arg_ptrs(&mut params, n_params);
                    let node_params =
                        kernel_node_params(func, grid, block, kernel_params.as_mut_ptr());
                    // SAFETY: `exec` and `node` belong to this graph and are live; the
                    // parameter values are copied during the call.
                    cu(unsafe { sys::cuGraphExecKernelNodeSetParams(*exec, *node, &node_params) })?;
                }
                Launch::Copy { dst, src, len, .. } => {
                    let copy = copy_node_params(dst, src, len);
                    // SAFETY: `exec` and `node` belong to this graph and are live; `copy`
                    // lives across the call.
                    cu(unsafe {
                        sys::cuGraphExecMemcpyNodeSetParams(*exec, *node, &copy, self.raw_ctx)
                    })?;
                }
            }
            *applied = *launch;
        }
        Ok(())
    }

    /// Launch `plan` for a batch of `op_count` ops. A batch structure seen before is built
    /// into a graph on its second sighting. Later batches of that structure patch the
    /// nodes whose values changed and replay. The structure includes which launches share
    /// buffers, so a replayed graph keeps the dependencies it was built with. The context
    /// must be current and `ctx_lock` held.
    fn launch_plan(&self, plan: &[Launch], op_count: usize) -> Result<()> {
        if op_count < GRAPH_MIN_OPS || plan.is_empty() {
            return self.issue(plan);
        }
        let key = plan_key(plan);
        let mut graphs = self.graphs.lock().map_err(|_| GpuError::DeviceLost)?;
        match graphs.entries.get_mut(&key) {
            Some(GraphEntry::Ready(replay)) => match self.patch_nodes(replay, plan) {
                Ok(()) => {
                    let exec = replay.exec;
                    drop(graphs);
                    // SAFETY: `exec` is cached, so it is live; it is destroyed only under ctx_lock.
                    cu(unsafe { sys::cuGraphLaunch(exec, self.stream) })
                }
                Err(_) => {
                    // A graph whose nodes cannot be patched is not worth patching again.
                    graphs.disable(key);
                    drop(graphs);
                    self.issue(plan)
                }
            },
            Some(GraphEntry::Uncachable) => {
                drop(graphs);
                self.issue(plan)
            }
            Some(GraphEntry::Seen) => {
                drop(graphs);
                let Some(replay) = self.build_graph(plan) else {
                    self.graphs
                        .lock()
                        .map_err(|_| GpuError::DeviceLost)?
                        .disable(key);
                    return self.issue(plan);
                };
                let exec = replay.exec;
                match self.graphs.lock() {
                    Ok(mut graphs) => graphs.insert_ready(key, replay),
                    Err(_) => {
                        replay.destroy();
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
            unsafe { sys::cuStreamSynchronize(self.upload_stream) };
            unsafe { sys::cuStreamSynchronize(self.download_stream) };
            // Graph execs and staging are released while the context is current.
            if let Ok(graphs) = self.graphs.get_mut() {
                graphs.clear();
            }
            if let Ok(staging) = self.staging.get_mut() {
                *staging = None;
            }
            if let Ok(mut pinned) = self.pinned.0.lock() {
                for block in pinned.drain(..) {
                    unsafe { sys::cuMemFreeHost(block.ptr) };
                }
            }
            if let Ok(mut pool) = self.events.0.lock() {
                for ev in pool.drain(..) {
                    unsafe { sys::cuEventDestroy_v2(ev) };
                }
            }
            unsafe { sys::cuStreamDestroy_v2(self.stream) };
            unsafe { sys::cuStreamDestroy_v2(self.upload_stream) };
            unsafe { sys::cuStreamDestroy_v2(self.download_stream) };
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
    event: Mutex<Option<Arc<OwnedEvent>>>,
    /// Pinned block of an asynchronous transfer. Held until this submission is
    /// dropped, so a readback's bytes stay valid until they are taken.
    lease: Option<PinnedLease>,
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
        let Some(raw) = event.as_ref().map(|ev| ev.ev) else {
            return Ok(SubmissionStatus::Complete);
        };
        // SAFETY (in closure): `raw` is kept alive by `event` while the lock is held.
        let code = self
            .shared
            .with_current(|| Ok(unsafe { sys::cuEventQuery(raw) }))?;
        if code == 0 {
            // Dropping the last reference returns the event to its pool.
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
        let Some(ev) = event.take() else {
            return;
        };
        // Preserve the resource-lifetime contract even if a pending token is
        // dropped: wait for its exact stream position before the lease and the
        // buffers it references can be released.
        let _ = self.shared.with_current(|| wait_event(ev.ev));
        drop(ev);
        self.release_in_flight();
    }
}

/// Handle to an asynchronous buffer readback. Its bytes are in the lease of
/// `sub`, valid once `sub` completes.
struct CudaReadback {
    sub: CudaSubmission,
    len: usize,
}

impl GpuSubmission for CudaReadback {
    fn cycle_id(&self) -> u64 {
        self.sub.cycle_id()
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        self.sub.poll()
    }

    fn wait(&self, timeout: Duration) -> Result<()> {
        self.sub.wait(timeout)
    }
}

impl GpuReadback for CudaReadback {
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.len];
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }

    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.len {
            return Err(GpuError::Backend(format!(
                "cuda: readback is {} bytes, destination is {}",
                self.len,
                dst.len()
            )));
        }
        self.sub.wait(Duration::MAX)?;
        let Some(lease) = self.sub.lease.as_ref() else {
            return Err(GpuError::Backend(
                "cuda: readback has no staging block".into(),
            ));
        };
        // SAFETY: the transfer into this block completed (waited above), and the
        // block stays leased to `self` for the duration of the copy.
        let staged = unsafe { slice::from_raw_parts(lease.block.ptr as *const u8, self.len) };
        copy_host(dst, staged);
        Ok(())
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
            cu(unsafe { sys::cuStreamSynchronize(self.shared.upload_stream) })?;
            cu(unsafe { sys::cuStreamSynchronize(self.shared.download_stream) })?;
            // Hazard events are released now; the streams they refer to are drained.
            if let Ok(mut lifetime) = self.lifetime.lock() {
                lifetime.hazards.clear();
            }
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
        let n_buffers = op.bindings.buffers.len();
        // Buffers beyond the signature's pointer parameters are unknown, so count them as written.
        let writes = if n_buffers <= cp.ptr_params {
            cp.writes
        } else {
            u32::MAX
        };
        Ok(Launch::Kernel {
            func: cp.func,
            grid: op.grid,
            block: cp.block,
            params,
            n_params,
            n_buffers,
            writes,
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
            dst_buf: dst.ptr,
            src_buf: src.ptr,
        }))
    }
}

/// Buffers a plan touches, as `(base pointer, written)`, in plan order.
fn plan_accesses(plan: &[Launch]) -> Vec<(u64, bool)> {
    let mut accesses = Vec::new();
    let mut one = Vec::new();
    for launch in plan {
        launch_accesses(launch, &mut one);
        accesses.append(&mut one);
    }
    accesses
}

impl CudaDevice {
    /// Make `stream` wait for the events of earlier transfers or compute work that
    /// touch `ptrs`, from the other stream. Same-stream events are ordered already.
    /// The context must be current and `stream` must be one of this device's streams.
    fn wait_hazards(&self, lane: Lane, ptrs: &[u64]) -> Result<()> {
        let stream = self.shared.stream(lane);
        let lifetime = self.lifetime.lock().map_err(|_| GpuError::DeviceLost)?;
        for ptr in ptrs {
            let Some(hazard) = lifetime.hazards.get(ptr) else {
                continue;
            };
            for access in [&hazard.last_access, &hazard.last_write]
                .into_iter()
                .flatten()
            {
                if access.lane != lane {
                    // SAFETY: the event is kept alive by `access` while `lifetime` is held.
                    cu(unsafe { sys::cuStreamWaitEvent(stream, access.event.ev, 0) })?;
                }
            }
        }
        Ok(())
    }

    /// Record that a work item completing at `event` on `lane` accessed `accesses`,
    /// as `(base pointer, written)`.
    fn publish_hazards(
        &self,
        accesses: &[(u64, bool)],
        event: &Arc<OwnedEvent>,
        lane: Lane,
    ) -> Result<()> {
        let mut lifetime = self.lifetime.lock().map_err(|_| GpuError::DeviceLost)?;
        for &(ptr, written) in accesses {
            let hazard = lifetime.hazards.entry(ptr).or_default();
            let access = Access {
                event: Arc::clone(event),
                lane,
            };
            if written {
                hazard.last_write = Some(access.clone());
            }
            hazard.last_access = Some(access);
        }
        Ok(())
    }

    /// Account for a new submission before any of its work is queued.
    fn begin_submission(&self) -> Result<()> {
        self.lifetime
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .in_flight += 1;
        Ok(())
    }

    /// Drain `stream` after a failed submission, then release its in-flight count.
    fn fail_submission(&self, lane: Lane, error: GpuError) -> GpuError {
        let stream = self.shared.stream(lane);
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
        error
    }

    /// Queue a resolved compute plan on the compute stream, after the copy-stream
    /// work it depends on, and return its completion event. `resolve_error` is set
    /// when resolution stopped early: the prefix is still queued, then the error returned.
    /// The context must be current.
    fn compute_plan(
        &self,
        plan: &[Launch],
        op_count: usize,
        resolve_error: Option<GpuError>,
    ) -> Result<Arc<OwnedEvent>> {
        let stream = self.shared.stream;
        let accesses = plan_accesses(plan);
        let ptrs: Vec<u64> = accesses.iter().map(|&(ptr, _)| ptr).collect();
        self.wait_hazards(Lane::Compute, &ptrs)?;
        if let Some(error) = resolve_error {
            // Launch the resolved prefix, as the per-op loop did, then fail.
            self.shared.issue(plan)?;
            return Err(error);
        }
        self.shared.launch_plan(plan, op_count)?;
        let event = OwnedEvent::record(&self.shared, stream)?;
        self.publish_hazards(&accesses, &event, Lane::Compute)?;
        Ok(event)
    }

    /// Wrap a copy-stream transfer's completion event as a submission.
    fn transfer_submission(
        &self,
        cycle_id: u64,
        event: Arc<OwnedEvent>,
        lease: PinnedLease,
    ) -> CudaSubmission {
        CudaSubmission {
            cycle_id,
            shared: Arc::clone(&self.shared),
            event: Mutex::new(Some(event)),
            lease: Some(lease),
            buffers: Arc::clone(&self.buffers),
            shaders: Arc::clone(&self.shaders),
            pipelines: Arc::clone(&self.pipelines),
            lifetime: Arc::clone(&self.lifetime),
        }
    }

    /// Resolve a buffer for an asynchronous transfer: its record and the base
    /// pointer, with the byte range checked against it.
    fn transfer_target(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<CudaBuffer> {
        let cb = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;
        if offset.checked_add(len).is_none_or(|end| end > cb.len) {
            return Err(GpuError::Backend("cuda: transfer out of bounds".into()));
        }
        Ok(cb)
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
                    lifetime.hazards.remove(&cb.ptr);
                    lifetime
                        .deferred
                        .push(DeferredCudaResource::Buffer(buffer.index(), cb));
                }
                None
            } else {
                let cb = buffers.remove(buffer);
                if let Some(cb) = cb {
                    lifetime.hazards.remove(&cb.ptr);
                }
                cb
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
        let mut source = None;
        let mut image = match desc.source {
            ShaderSource::Ptx(ptx) => ModuleImage::Ptx(ptx.to_vec()),
            ShaderSource::CudaSrc(src) => {
                source = str::from_utf8(src).ok().map(str::to_owned);
                nvrtc_compile(src, self.shared.sm)?
            }
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
                source,
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
        let (module, signature) = self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(desc.shader)
            .map(|s| {
                let signature = s
                    .source
                    .as_deref()
                    .and_then(|src| kernel_pointer_writes(src, desc.entry));
                (s.module, signature)
            })
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
        // An unreadable signature keeps `ptr_params` at zero, so every buffer counts as written.
        let (ptr_params, writes) = signature.map_or((0, 0), |written| {
            let mask = written
                .iter()
                .take(32)
                .enumerate()
                .fold(0u32, |mask, (i, &w)| if w { mask | 1 << i } else { mask });
            (written.len(), mask)
        });
        let handle = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaPipeline {
                func,
                block,
                ptr_params,
                writes,
            });
        Ok(handle)
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        // Graphs built from this pipeline hold its function handle.
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

        self.begin_submission()?;
        // Same-stream launches execute and become globally visible in
        // submission order on the device, so a later op safely reads an
        // earlier op's output with no explicit barrier. The event recorded
        // after the final launch represents completion of the whole batch.
        let event = self.shared.with_current(|| {
            let (plan, resolve_error) = resolve_all(ops, |op| self.resolve_dispatch(op).map(Some));
            self.compute_plan(&plan, ops.len(), resolve_error)
        });
        match event {
            Ok(event) => Ok(Box::new(CudaSubmission {
                cycle_id,
                shared: Arc::clone(&self.shared),
                event: Mutex::new(Some(event)),
                lease: None,
                buffers: Arc::clone(&self.buffers),
                shaders: Arc::clone(&self.shaders),
                pipelines: Arc::clone(&self.pipelines),
                lifetime: Arc::clone(&self.lifetime),
            })),
            // Some earlier launches may already be queued; `fail_submission` drains them
            // before releasing any concurrently retired resources.
            Err(error) => Err(self.fail_submission(Lane::Compute, error)),
        }
    }

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        self.begin_submission()?;
        let event = self.shared.with_current(|| {
            let (plan, resolve_error) = resolve_all(ops, |op| match op {
                ComputeOp::CopyBuffer(copy) => self.resolve_copy(copy),
                ComputeOp::Dispatch(op) => self.resolve_dispatch(op).map(Some),
            });
            self.compute_plan(&plan, ops.len(), resolve_error)
        });
        match event {
            Ok(event) => Ok(Box::new(CudaSubmission {
                cycle_id,
                shared: Arc::clone(&self.shared),
                event: Mutex::new(Some(event)),
                lease: None,
                buffers: Arc::clone(&self.buffers),
                shaders: Arc::clone(&self.shaders),
                pipelines: Arc::clone(&self.pipelines),
                lifetime: Arc::clone(&self.lifetime),
            })),
            Err(error) => Err(self.fail_submission(Lane::Compute, error)),
        }
    }

    fn submit_write_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<Submission> {
        if data.is_empty() {
            self.transfer_target(buffer, offset, 0)?;
            return Ok(Box::new(CompletedSubmission::new(cycle_id)));
        }
        let cb = self.transfer_target(buffer, offset, data.len() as u64)?;
        let stream = self.shared.stream(Lane::Upload);
        self.begin_submission()?;
        let queued = self.shared.with_current(|| {
            let lease = self.shared.pinned.lease(data.len())?;
            // SAFETY: the lease holds at least `data.len()` bytes and is not shared
            // with any other transfer until this one completes.
            let staged =
                unsafe { slice::from_raw_parts_mut(lease.block.ptr as *mut u8, data.len()) };
            copy_host(staged, data);
            self.wait_hazards(Lane::Upload, &[cb.ptr])?;
            // SAFETY: the destination range was bounds-checked against the buffer.
            cu(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    cb.ptr + offset,
                    lease.block.ptr as *const c_void,
                    driver_len(data.len())?,
                    stream,
                )
            })?;
            let published = OwnedEvent::record(&self.shared, stream).and_then(|event| {
                self.publish_hazards(&[(cb.ptr, true)], &event, Lane::Upload)
                    .map(|()| event)
            });
            let event = match published {
                Ok(event) => event,
                Err(error) => {
                    // The copy may already read `lease`: drain it before the lease is returned.
                    let _ = cu(unsafe { sys::cuStreamSynchronize(stream) });
                    return Err(error);
                }
            };
            Ok((event, lease))
        });
        match queued {
            Ok((event, lease)) => Ok(Box::new(self.transfer_submission(cycle_id, event, lease))),
            Err(error) => Err(self.fail_submission(Lane::Upload, error)),
        }
    }

    fn submit_read_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        let cb = self.transfer_target(buffer, offset, len)?;
        if !cb.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        let len_bytes = usize::try_from(len).map_err(|_| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "cuda: read length exceeds host address range".into(),
            ))
        })?;
        if len_bytes == 0 {
            return Ok(Box::new(CompletedReadback::new(cycle_id, Vec::new())));
        }
        let stream = self.shared.stream(Lane::Download);
        self.begin_submission()?;
        let queued = self.shared.with_current(|| {
            let lease = self.shared.pinned.lease(len_bytes)?;
            self.wait_hazards(Lane::Download, &[cb.ptr])?;
            // SAFETY: the lease holds at least `len_bytes` bytes; the source range was
            // bounds-checked against the buffer.
            cu(unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    lease.block.ptr,
                    cb.ptr + offset,
                    driver_len(len_bytes)?,
                    stream,
                )
            })?;
            let published = OwnedEvent::record(&self.shared, stream).and_then(|event| {
                self.publish_hazards(&[(cb.ptr, false)], &event, Lane::Download)
                    .map(|()| event)
            });
            let event = match published {
                Ok(event) => event,
                Err(error) => {
                    // The copy writes into `lease`: drain it before the lease is returned.
                    let _ = cu(unsafe { sys::cuStreamSynchronize(stream) });
                    return Err(error);
                }
            };
            Ok((event, lease))
        });
        match queued {
            Ok((event, lease)) => Ok(Box::new(CudaReadback {
                sub: self.transfer_submission(cycle_id, event, lease),
                len: len_bytes,
            })),
            Err(error) => Err(self.fail_submission(Lane::Download, error)),
        }
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

    use zengpu_blas::level3::GemmKernel;
    use zengpu_compute::BufferPool;
    use zengpu_hal::DType;
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

    /// `copy_add` writes `in + v` to `out`; `fill` writes `v` to `out`.
    const COPY_FILL_CUDA: &str = r#"
extern "C" __global__ void copy_add(const float* in, float* out, float v, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] + v;
    }
}

extern "C" __global__ void fill(float* out, float v, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = v;
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

        // A changed scalar keeps the batch structure: the node is patched and the graph replays.
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
        // Three runs: the batch is built and replayed before it is invalidated.
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
    /// Number of instantiated graphs cached for `device`.
    fn ready_graphs(device: &dyn GpuDevice) -> usize {
        let device = device
            .as_any()
            .downcast_ref::<CudaDevice>()
            .expect("cuda device");
        device.shared.graphs.lock().unwrap().ready
    }

    #[test]
    fn graph_replay_patches_changed_scalars() {
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

        // Every run changes every scalar. Runs 1-2 build the graph; runs 3-6 patch its nodes.
        let mut total = 0.0f32;
        for run in 0..6u32 {
            let adds: Vec<f32> = (0..8u32).map(|k| (run * 10 + k) as f32).collect();
            total += adds.iter().sum::<f32>();
            submit_accum(&*device, pipeline, out, &adds, N as u32);
            let expected: Vec<f32> = (0..N).map(|i| i as f32 + total).collect();
            assert_eq!(read_f32(&*device, out, N), expected, "run {run}");
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(out);
    }

    #[test]
    fn graph_replay_follows_changed_buffers() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        let outs = [graph_buffer(&*device, size), graph_buffer(&*device, size)];
        let init: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
        for &out in &outs {
            device.write_buffer(out, 0, &init).unwrap();
        }
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

        // The same structure alternates between two output buffers. Each run must land in
        // the buffer it names, with the other buffer left as it was.
        let adds = [1.0, 2.0, 3.0, 4.0];
        let mut totals = [0.0f32; 2];
        for run in 0..6 {
            let which = run % 2;
            submit_accum(&*device, pipeline, outs[which], &adds, N as u32);
            totals[which] += 10.0;
            for (out, total) in outs.iter().zip(totals) {
                let expected: Vec<f32> = (0..N).map(|i| i as f32 + total).collect();
                assert_eq!(read_f32(&*device, *out, N), expected, "run {run}");
            }
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for out in outs {
            device.destroy_buffer(out);
        }
    }

    #[test]
    fn graph_replay_follows_changed_copy_buffers() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        // Two sources with different contents, so a stale copy source shows up.
        let mut bases = Vec::new();
        for offset in [0.0f32, 1000.0] {
            let base = device
                .create_buffer(BufferDesc {
                    size,
                    usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                    memory: zengpu_hal::MemoryUsage::GpuOnly,
                })
                .unwrap();
            let init: Vec<u8> = (0..N)
                .flat_map(|i| (offset + i as f32).to_le_bytes())
                .collect();
            device.write_buffer(base, 0, &init).unwrap();
            bases.push(base);
        }
        let accs = [graph_buffer(&*device, size), graph_buffer(&*device, size)];
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

        // Each run copies one source over one accumulator, then adds 1 + 2 + 3 to it. The
        // copy's source and destination both change every run; the batch structure does not.
        const ADDS: [f32; 3] = [1.0, 2.0, 3.0];
        for run in 0..6 {
            let k = run % 2;
            let slots = [accs[k].index()];
            let scalars: Vec<[Scalar; 2]> = ADDS
                .iter()
                .map(|&v| [Scalar::F32(v), Scalar::U32(N as u32)])
                .collect();
            let copy = zengpu_hal::BufferCopyOp {
                src: bases[k],
                src_offset: 0,
                dst: accs[k],
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
            device
                .submit_compute_ops(0, &ops)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            let base_value = if k == 0 { 0.0 } else { 1000.0 };
            let expected: Vec<f32> = (0..N).map(|i| base_value + i as f32 + 6.0).collect();
            assert_eq!(read_f32(&*device, accs[k], N), expected, "run {run}");
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for base in bases {
            device.destroy_buffer(base);
        }
        for acc in accs {
            device.destroy_buffer(acc);
        }
    }

    /// One dispatch of `pipeline` over `slots` and `scalars`, covering `n` elements.
    fn graph_op<'a>(
        pipeline: PipelineHandle,
        slots: &'a [u32],
        scalars: &'a [Scalar],
        n: u32,
    ) -> ComputeOp<'a> {
        ComputeOp::Dispatch(zengpu_hal::DispatchOp {
            pipeline,
            bindings: Bindings {
                buffers: slots,
                scalars,
                textures: &[],
                images: &[],
            },
            grid: [n.div_ceil(64), 1, 1],
        })
    }

    #[test]
    fn kernel_signature_reads_const_pointers_as_read_only() {
        let zsl_style = r#"
extern "C" __global__ __launch_bounds__(64)
void trig(
    const float* __restrict__ src,
    float* __restrict__ out,
    unsigned int len
) {
}
"#;
        assert_eq!(
            kernel_pointer_writes(zsl_style, "trig"),
            Some(vec![false, true])
        );
        let qualifiers =
            "__global__ void k(const float* const a, float* const b, const unsigned int n) {}";
        assert_eq!(
            kernel_pointer_writes(qualifiers, "k"),
            Some(vec![false, true])
        );
    }

    #[test]
    fn kernel_signature_matches_whole_entry_names() {
        let two =
            "__global__ void foo2(float* a) {}\n__global__ void foo(const float* a, float* b) {}";
        assert_eq!(kernel_pointer_writes(two, "foo"), Some(vec![false, true]));
        assert_eq!(kernel_pointer_writes(two, "foo2"), Some(vec![true]));
        let suffix = "__global__ void my_foo(float* a) {}\n__global__ void foo(const float* a) {}";
        assert_eq!(kernel_pointer_writes(suffix, "foo"), Some(vec![false]));
    }

    #[test]
    fn kernel_signature_falls_back_when_unreadable() {
        // Not a kernel, no such kernel, or a parameter list the scan does not model.
        assert_eq!(kernel_pointer_writes("void k(float* a) {}", "k"), None);
        assert_eq!(
            kernel_pointer_writes("__global__ void k(float* a) {}", "other"),
            None
        );
        assert_eq!(
            kernel_pointer_writes("__global__ void k(Buf<float*, int> b) {}", "k"),
            None
        );
        assert_eq!(
            kernel_pointer_writes("__global__ void k(float a[], float* b) {}", "k"),
            None
        );
    }

    #[test]
    fn launch_deps_follow_buffer_hazards() {
        // Kernel over buffers `(base, written)`, in parameter order.
        let kernel = |bufs: &[(u64, bool)]| {
            let mut params = [[0u8; 8]; MAX_PARAMS];
            let mut writes = 0u32;
            for (i, &(base, written)) in bufs.iter().enumerate() {
                params[i] = base.to_le_bytes();
                if written {
                    writes |= 1 << i;
                }
            }
            Launch::Kernel {
                func: ptr::null_mut(),
                grid: [1, 1, 1],
                block: [1, 1, 1],
                params,
                n_params: bufs.len(),
                n_buffers: bufs.len(),
                writes,
            }
        };
        let plan = [
            kernel(&[(1, true)]),              // 0: writes A
            kernel(&[(1, false)]),             // 1: reads A (RAW on 0)
            kernel(&[(1, false)]),             // 2: reads A (RAW on 0)
            kernel(&[(1, true)]),              // 3: writes A (WAW on 0, WAR on 1 and 2)
            kernel(&[(2, true)]),              // 4: writes B, independent
            kernel(&[(1, false), (2, false)]), // 5: reads A and B
            kernel(&[(3, true), (3, true)]),   // 6: writes C through two parameters
        ];
        assert_eq!(
            launch_deps(&plan),
            vec![
                vec![],
                vec![0],
                vec![0],
                vec![0, 1, 2],
                vec![],
                vec![3, 4],
                vec![],
            ]
        );
        // Same buffer pattern, different buffers: same key. Shared buffer versus distinct: different.
        assert_eq!(
            plan_key(&[kernel(&[(7, false), (8, true)])]),
            plan_key(&[kernel(&[(9, false), (5, true)])])
        );
        assert_ne!(
            plan_key(&[kernel(&[(7, false), (7, true)])]),
            plan_key(&[kernel(&[(7, false), (8, true)])])
        );
    }

    #[test]
    fn graph_chain_through_distinct_buffers_honours_dependencies() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        const STAGES: usize = 8;
        let size = (N * size_of::<f32>()) as u64;
        let bufs: Vec<BufferHandle> = (0..=STAGES).map(|_| graph_buffer(&*device, size)).collect();
        let shader = device
            .create_shader(ShaderDesc::cuda_src(COPY_FILL_CUDA))
            .expect("compile copy_add");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "copy_add",
                block: [64, 1, 1],
            })
            .expect("create pipeline");

        // Stage k reads buffer k and writes buffer k + 1, so each stage waits for the one before.
        // Every run starts from new input, so a stage that ran early would read the previous run.
        let slots: Vec<[u32; 2]> = (0..STAGES)
            .map(|k| [bufs[k].index(), bufs[k + 1].index()])
            .collect();
        let adds: Vec<[Scalar; 2]> = (0..STAGES)
            .map(|k| [Scalar::F32((k + 1) as f32), Scalar::U32(N as u32)])
            .collect();
        let total: f32 = (1..=STAGES).map(|v| v as f32).sum();
        for run in 0..4 {
            let input: Vec<u8> = (0..N)
                .flat_map(|i| ((run * 1000 + i) as f32).to_le_bytes())
                .collect();
            device.write_buffer(bufs[0], 0, &input).unwrap();
            let ops: Vec<ComputeOp<'_>> = (0..STAGES)
                .map(|k| graph_op(pipeline, &slots[k], &adds[k], N as u32))
                .collect();
            device
                .submit_compute_ops(0, &ops)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            let expected: Vec<f32> = (0..N).map(|i| (run * 1000 + i) as f32 + total).collect();
            assert_eq!(read_f32(&*device, bufs[STAGES], N), expected, "run {run}");
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for buf in bufs {
            device.destroy_buffer(buf);
        }
    }

    #[test]
    fn graph_replay_honours_write_after_read() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        let size = (N * size_of::<f32>()) as u64;
        let x = graph_buffer(&*device, size);
        let y = graph_buffer(&*device, size);
        let z = graph_buffer(&*device, size);
        let w = graph_buffer(&*device, size);
        let shader = device
            .create_shader(ShaderDesc::cuda_src(COPY_FILL_CUDA))
            .expect("compile kernels");
        let copy = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "copy_add",
                block: [64, 1, 1],
            })
            .expect("create copy pipeline");
        let fill = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "fill",
                block: [64, 1, 1],
            })
            .expect("create fill pipeline");

        // Op 0 reads X into Y, op 1 then overwrites X. Op 0 must read the old X: a run that
        // reversed them would leave Y at 1000.
        let (x_slot, y_slot, z_slot, w_slot) = (x.index(), y.index(), z.index(), w.index());
        let zero = [Scalar::F32(0.0), Scalar::U32(N as u32)];
        let thousand = [Scalar::F32(1000.0), Scalar::U32(N as u32)];
        let two = [Scalar::F32(2.0), Scalar::U32(N as u32)];
        let x_only = [x_slot];
        let x_to_y = [x_slot, y_slot];
        let x_to_z = [x_slot, z_slot];
        let y_to_w = [y_slot, w_slot];
        let thousands: Vec<f32> = vec![1000.0; N];
        for run in 0..4 {
            let old: Vec<u8> = (0..N)
                .flat_map(|i| ((run * 100 + i) as f32).to_le_bytes())
                .collect();
            device.write_buffer(x, 0, &old).unwrap();
            let ops = [
                graph_op(copy, &x_to_y, &zero, N as u32),
                graph_op(fill, &x_only, &thousand, N as u32),
                graph_op(copy, &x_to_z, &two, N as u32),
                graph_op(copy, &y_to_w, &zero, N as u32),
            ];
            device
                .submit_compute_ops(0, &ops)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            let old_values: Vec<f32> = (0..N).map(|i| (run * 100 + i) as f32).collect();
            assert_eq!(read_f32(&*device, y, N), old_values, "run {run}: Y");
            assert_eq!(read_f32(&*device, w, N), old_values, "run {run}: W");
            assert_eq!(read_f32(&*device, x, N), thousands, "run {run}: X");
            assert_eq!(read_f32(&*device, z, N), vec![1002.0; N], "run {run}: Z");
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(copy);
        device.destroy_pipeline(fill);
        device.destroy_shader(shader);
        for buf in [x, y, z, w] {
            device.destroy_buffer(buf);
        }
    }

    #[test]
    fn graph_replay_runs_independent_ops_exactly() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 256;
        const OPS: usize = 8;
        let size = (N * size_of::<f32>()) as u64;
        let ins: Vec<BufferHandle> = (0..OPS).map(|_| graph_buffer(&*device, size)).collect();
        let outs: Vec<BufferHandle> = (0..OPS).map(|_| graph_buffer(&*device, size)).collect();
        let shader = device
            .create_shader(ShaderDesc::cuda_src(COPY_FILL_CUDA))
            .expect("compile copy_add");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "copy_add",
                block: [64, 1, 1],
            })
            .expect("create pipeline");

        // Each op reads its own input and writes its own output, so no op waits for another.
        let slots: Vec<[u32; 2]> = (0..OPS)
            .map(|k| [ins[k].index(), outs[k].index()])
            .collect();
        let adds: Vec<[Scalar; 2]> = (0..OPS)
            .map(|k| [Scalar::F32((k + 1) as f32), Scalar::U32(N as u32)])
            .collect();
        for run in 0..4 {
            for (k, &input) in ins.iter().enumerate() {
                let bytes: Vec<u8> = (0..N)
                    .flat_map(|i| ((run * 1000 + k * 10 + i) as f32).to_le_bytes())
                    .collect();
                device.write_buffer(input, 0, &bytes).unwrap();
            }
            let ops: Vec<ComputeOp<'_>> = (0..OPS)
                .map(|k| graph_op(pipeline, &slots[k], &adds[k], N as u32))
                .collect();
            device
                .submit_compute_ops(0, &ops)
                .unwrap()
                .wait(Duration::from_secs(5))
                .unwrap();
            for (k, &out) in outs.iter().enumerate() {
                let expected: Vec<f32> = (0..N)
                    .map(|i| (run * 1000 + k * 10 + i) as f32 + (k + 1) as f32)
                    .collect();
                assert_eq!(read_f32(&*device, out, N), expected, "run {run} op {k}");
            }
        }
        assert_eq!(ready_graphs(&*device), 1);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for buf in ins.into_iter().chain(outs) {
            device.destroy_buffer(buf);
        }
    }

    /// `slow_copy` spins for about `cycles` clock cycles in every thread, then copies `in` to `out`.
    const SLOW_COPY_CUDA: &str = r#"
extern "C" __global__ void slow_copy(const float* in, float* out, unsigned int n, unsigned int cycles) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cycles) {
    }
    if (i < n) {
        out[i] = in[i];
    }
}
"#;

    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(f32::to_le_bytes).collect()
    }

    /// Compile `entry` from `src` into a pipeline with 64-thread blocks.
    fn pipeline_for(
        device: &dyn GpuDevice,
        src: &str,
        entry: &str,
    ) -> (ShaderHandle, PipelineHandle) {
        let shader = device
            .create_shader(ShaderDesc::cuda_src(src))
            .expect("compile kernel");
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .expect("create pipeline");
        (shader, pipeline)
    }

    /// Submit one dispatch of a `pipeline` over `buffers` (bound in order) and `scalars`.
    fn submit_one(
        device: &dyn GpuDevice,
        pipeline: PipelineHandle,
        buffers: &[BufferHandle],
        scalars: &[Scalar],
        n: u32,
    ) -> Submission {
        let slots: Vec<u32> = buffers.iter().map(|b| b.index()).collect();
        let op = zengpu_hal::DispatchOp {
            pipeline,
            bindings: Bindings {
                buffers: &slots,
                scalars,
                textures: &[],
                images: &[],
            },
            grid: [n.div_ceil(64), 1, 1],
        };
        device
            .submit_batch(0, slice::from_ref(&op))
            .expect("submit failed")
    }

    #[test]
    fn async_write_dispatch_read_chain_needs_no_intermediate_waits() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 1024;
        let size = (N * size_of::<f32>()) as u64;
        let input = graph_buffer(&*device, size);
        let output = graph_buffer(&*device, size);
        let (shader, pipeline) = pipeline_for(&*device, COPY_FILL_CUDA, "copy_add");
        let data = f32_bytes((0..N).map(|i| i as f32));

        let write = device.submit_write_buffer(1, input, 0, &data).unwrap();
        let kernel = submit_one(
            &*device,
            pipeline,
            &[input, output],
            &[Scalar::F32(2.0), Scalar::U32(N as u32)],
            N as u32,
        );
        let readback = device.submit_read_buffer(2, output, 0, size).unwrap();
        assert_eq!(readback.cycle_id(), 2);
        let got = f32_values(&readback.into_bytes().unwrap());
        let expected: Vec<f32> = (0..N).map(|i| i as f32 + 2.0).collect();
        assert_eq!(got, expected);
        write.wait(Duration::from_secs(5)).unwrap();
        kernel.wait(Duration::from_secs(5)).unwrap();

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(input);
        device.destroy_buffer(output);
    }

    #[test]
    fn async_ping_pong_chain_waits_only_on_final_readbacks() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 512;
        const ROUNDS: usize = 6;
        let size = (N * size_of::<f32>()) as u64;
        let bufs = [graph_buffer(&*device, size), graph_buffer(&*device, size)];
        let (shader, pipeline) = pipeline_for(&*device, COPY_FILL_CUDA, "copy_add");
        let mut pending = Vec::new();
        let mut readbacks = Vec::new();
        for round in 0..ROUNDS {
            let (src, dst) = (bufs[round % 2], bufs[(round + 1) % 2]);
            let data = f32_bytes((0..N).map(|i| (round * 1000 + i) as f32));
            pending.push(
                device
                    .submit_write_buffer(round as u64, src, 0, &data)
                    .unwrap(),
            );
            pending.push(submit_one(
                &*device,
                pipeline,
                &[src, dst],
                &[Scalar::F32(1.0), Scalar::U32(N as u32)],
                N as u32,
            ));
            readbacks.push(
                device
                    .submit_read_buffer(round as u64, dst, 0, size)
                    .unwrap(),
            );
        }
        for (round, readback) in readbacks.into_iter().enumerate() {
            let got = f32_values(&readback.into_bytes().unwrap());
            let expected: Vec<f32> = (0..N).map(|i| (round * 1000 + i) as f32 + 1.0).collect();
            assert_eq!(got, expected, "round {round}");
        }
        drop(pending);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for buf in bufs {
            device.destroy_buffer(buf);
        }
    }

    #[test]
    fn async_write_lands_after_earlier_reader_finishes() {
        let Some(device) = cuda_device() else { return };
        const N: usize = 1024;
        let size = (N * size_of::<f32>()) as u64;
        let x = graph_buffer(&*device, size);
        let y = graph_buffer(&*device, size);
        let (shader, pipeline) = pipeline_for(&*device, SLOW_COPY_CUDA, "slow_copy");
        let old: Vec<f32> = (0..N).map(|i| i as f32 + 0.5).collect();
        let new: Vec<f32> = (0..N).map(|i| i as f32 + 1000.0).collect();
        device
            .write_buffer(x, 0, &f32_bytes(old.iter().copied()))
            .unwrap();

        // The kernel reads x for tens of milliseconds; the write must not overtake it.
        let kernel = submit_one(
            &*device,
            pipeline,
            &[x, y],
            &[Scalar::U32(N as u32), Scalar::U32(50_000_000)],
            N as u32,
        );
        let write = device
            .submit_write_buffer(1, x, 0, &f32_bytes(new.iter().copied()))
            .unwrap();
        let readback = device.submit_read_buffer(2, y, 0, size).unwrap();
        assert_eq!(f32_values(&readback.into_bytes().unwrap()), old);
        kernel.wait(Duration::from_secs(10)).unwrap();
        write.wait(Duration::from_secs(10)).unwrap();

        let check = device.submit_read_buffer(3, x, 0, size).unwrap();
        assert_eq!(f32_values(&check.into_bytes().unwrap()), new);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(x);
        device.destroy_buffer(y);
    }

    #[test]
    fn async_large_write_round_trips() {
        let Some(device) = cuda_device() else { return };
        const SIZE: usize = 80 * 1024 * 1024 + 13;
        let buf = device
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        let data: Vec<u8> = (0..SIZE)
            .map(|i| (i.wrapping_mul(31) % 251) as u8)
            .collect();
        let write = device.submit_write_buffer(0, buf, 0, &data).unwrap();
        let readback = device.submit_read_buffer(0, buf, 0, SIZE as u64).unwrap();
        assert!(readback.into_bytes().unwrap() == data);
        write.wait(Duration::from_secs(10)).unwrap();
        device.destroy_buffer(buf);
    }

    #[test]
    fn async_write_copies_data_before_returning() {
        let Some(device) = cuda_device() else { return };
        const SIZE: usize = 4096;
        let buf = graph_buffer(&*device, SIZE as u64);
        let mut data: Vec<u8> = (0..SIZE).map(|i| (i % 253) as u8).collect();
        let expected = data.clone();
        let write = device.submit_write_buffer(0, buf, 0, &data).unwrap();
        data.fill(0);
        let mut got = vec![0u8; SIZE];
        device
            .submit_read_buffer(0, buf, 0, SIZE as u64)
            .unwrap()
            .read_into(&mut got)
            .unwrap();
        assert!(got == expected);
        write.wait(Duration::from_secs(5)).unwrap();
        device.destroy_buffer(buf);
    }

    /// Values `i/8` for `i` in [-32, 32]. They are exact in f16 and bf16, and
    /// every product and partial sum is exact in f32, so any accumulation
    /// order (tensor-core, split-K, tiled) must match the CPU reference exactly.
    fn exact_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 65) as f32 / 8.0 - 4.0
            })
            .collect()
    }

    /// IEEE half bits of an [`exact_values`] entry (normal range, no rounding).
    fn f16_bits(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let bits = x.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        assert!(
            bits & 0x1fff == 0 && (1..31).contains(&exp),
            "{x} is not exact in f16"
        );
        (((bits >> 31) as u16) << 15) | ((exp as u16) << 10) | ((bits & 0x7f_ffff) >> 13) as u16
    }

    /// bfloat16 bits of an [`exact_values`] entry (the top 16 bits of f32).
    fn bf16_bits(x: f32) -> u16 {
        assert!(x.to_bits() & 0xffff == 0, "{x} is not exact in bf16");
        (x.to_bits() >> 16) as u16
    }

    fn cpu_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            let crow = &mut c[i * n..(i + 1) * n];
            for p in 0..k {
                let av = a[i * k + p];
                for (cv, bv) in crow.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                    *cv += av * bv;
                }
            }
        }
        c
    }

    /// A CUDA pool for the GEMM tests, or None to skip.
    fn cuda_pool() -> Option<BufferPool> {
        Some(BufferPool::new(Arc::from(cuda_device()?)))
    }

    /// C = A @ B on the device for `dtype` inputs (f32, f16 or bf16), checked
    /// against the CPU reference.
    fn check_gemm(
        pool: &BufferPool,
        kernel: &GemmKernel,
        dtype: DType,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let a_vals = exact_values(m * k, (m * 31 + k) as u64);
        let b_vals = exact_values(k * n, (n * 17 + k + 1) as u64);
        let want = cpu_gemm(m, n, k, &a_vals, &b_vals);
        let encode = |vals: &[f32]| -> Vec<u8> {
            match dtype {
                DType::F16 => vals
                    .iter()
                    .flat_map(|&v| f16_bits(v).to_le_bytes())
                    .collect(),
                DType::Bf16 => vals
                    .iter()
                    .flat_map(|&v| bf16_bits(v).to_le_bytes())
                    .collect(),
                _ => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }
        };
        let device = pool.device();
        let upload = |shape: Vec<u32>, bytes: &[u8]| {
            let array = pool.alloc(shape, dtype).expect("alloc");
            device.write_buffer(array.buffer, 0, bytes).expect("upload");
            array
        };
        let a = upload(vec![m as u32, k as u32], &encode(&a_vals));
        let b = upload(vec![k as u32, n as u32], &encode(&b_vals));
        let c = kernel.sgemm(device, pool, 1.0, &a, &b).expect("sgemm");
        let raw = device.read_buffer(c.buffer, 0, (m * n * 4) as u64).unwrap();
        let mut bad = 0;
        for (i, want) in want.iter().enumerate() {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            if got != *want {
                if bad < 3 {
                    println!(
                        "{dtype:?} {m}x{n}x{k} [{}, {}]: got {got}, want {want}",
                        i / n,
                        i % n
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{dtype:?} {m}x{n}x{k}: {bad} mismatches");
        pool.free(a);
        pool.free(b);
        pool.free(c);
    }

    /// Shapes covering the few-row split-K kernel (with and without a split),
    /// the 64×64 and 128×128 tiles, and partial tiles in M, N and K.
    const GEMM_SHAPES: [(usize, usize, usize); 7] = [
        (1, 4096, 4096),
        (16, 4096, 4096),
        (77, 200, 136),
        (513, 1000, 264),
        (1024, 1024, 1024),
        (2048, 2048, 64),
        (3, 16, 8),
    ];

    fn gemm_matches_reference(dtype: DType) {
        let Some(pool) = cuda_pool() else { return };
        let kernel = GemmKernel::new(pool.device()).expect("GEMM kernel");
        for (m, n, k) in GEMM_SHAPES {
            check_gemm(&pool, &kernel, dtype, m, n, k);
        }
        kernel.destroy(pool.device());
    }

    #[test]
    fn gemm_f32_matches_reference() {
        gemm_matches_reference(DType::F32);
    }

    #[test]
    fn gemm_f16_tensor_cores_match_reference() {
        gemm_matches_reference(DType::F16);
    }

    #[test]
    fn gemm_bf16_tensor_cores_match_reference() {
        gemm_matches_reference(DType::Bf16);
    }
}
