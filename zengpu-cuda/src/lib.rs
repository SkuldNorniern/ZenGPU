//! CUDA backend: compute HAL only, no graphics. Uses cuda-oxide for the Driver API. Without CUDA
//! the adapter list is empty, not a build or link error (the stub library returns
//! `ErrorCode::StubLibrary` from `cuInit`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod error;

use std::any::Any;
use std::collections::HashMap;
use std::ffi::{CString, OsStr, c_char, c_void};
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::{take, zeroed};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, yield_now};
use std::time::{Duration, Instant};
use std::{env, fs, io, process, ptr, slice};

use cuda_oxide::{Cuda, context::Context, device::Device, sys};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, CompletedReadback, CompletedSubmission, ComputeCaps, ComputeOp,
    ComputePipelineDesc, DType, DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuReadback, GpuSubmission, HalCapabilities, MemoryBudget, MmaShape,
    PipelineHandle, ReadSubmission, Result, SamplerDesc, SamplerHandle, Scalar, ShaderDesc,
    ShaderHandle, ShaderSource, SlotMap, Submission, SubmissionStatus, TextureDesc, TextureHandle,
    UsageError, marker,
};

use error::{cu, from_cuda};

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
    fn nvrtcVersion(major: *mut i32, minor: *mut i32) -> i32;
}

const NVRTC_SUCCESS: i32 = 0;

/// Device code ready for `cuModuleLoadData`.
enum ModuleImage {
    /// Architecture-specific binary, loaded without JIT.
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

    /// PTX image, null-terminated as `cuModuleLoadData` requires.
    fn ptx(mut bytes: Vec<u8>) -> Self {
        if bytes.last() != Some(&0) {
            bytes.push(0);
        }
        Self::Ptx(bytes)
    }
}

/// Build attempts for `sm`, in order: `sm_XY` cubin, `compute_XY` PTX, then NVRTC's default PTX.
/// The first that succeeds wins.
fn nvrtc_attempts((major, minor): (u32, u32)) -> [(Option<String>, bool); 3] {
    [
        (Some(format!("--gpu-architecture=sm_{major}{minor}")), true),
        (
            Some(format!("--gpu-architecture=compute_{major}{minor}")),
            false,
        ),
        (None, false),
    ]
}

/// Compile CUDA C++ for the device arch, trying [`nvrtc_attempts`] in order. An older NVRTC that
/// predates the GPU still gives a runnable module.
fn nvrtc_compile(src: &[u8], sm: (u32, u32)) -> Result<ModuleImage> {
    let mut last_error = None;
    for (arch, want_cubin) in nvrtc_attempts(sm) {
        match nvrtc_build(src, arch.as_deref(), want_cubin) {
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
        ModuleImage::ptx(bytes)
    })
}

/// NVRTC version as `(major, minor)`, or `(0, 0)` when it cannot be queried.
fn nvrtc_version() -> (i32, i32) {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both out-pointers are valid for writes for the duration of the call.
    if unsafe { nvrtcVersion(&mut major, &mut minor) } == NVRTC_SUCCESS {
        (major, minor)
    } else {
        (0, 0)
    }
}

/// Driver version from `cuDriverGetVersion` (e.g. 12040), or 0 when unavailable.
fn driver_version() -> i32 {
    let mut version = 0;
    // SAFETY: the out-pointer is valid for a write for the duration of the call.
    if unsafe { sys::cuDriverGetVersion(&mut version) } == 0 {
        version
    } else {
        0
    }
}

/// Cache format version, in [`cache_key`]. Bump on any layout change so old entries are
/// never loaded.
const CACHE_FORMAT_VERSION: u32 = 1;

/// FNV-1a 64-bit hash. Used instead of `DefaultHasher`, which is not stable across Rust releases.
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    /// Hash one field, length-prefixed so adjacent fields cannot run together.
    fn field(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }
}

/// Cache key for the module built from `src` for `sm` by [`nvrtc_compile`]. Covers every input to
/// the output: format version, source, all attempt options, NVRTC and driver versions.
fn cache_key(src: &[u8], sm: (u32, u32), nvrtc: (i32, i32), driver: i32) -> u64 {
    let mut hash = Fnv1a::new();
    hash.field(&CACHE_FORMAT_VERSION.to_le_bytes());
    hash.field(src);
    for (arch, cubin) in nvrtc_attempts(sm) {
        hash.field(arch.unwrap_or_default().as_bytes());
        hash.field(&[u8::from(cubin)]);
    }
    hash.field(format!("nvrtc {}.{}", nvrtc.0, nvrtc.1).as_bytes());
    hash.field(format!("driver {driver}").as_bytes());
    hash.0
}

/// Cache base from env: `ZENGPU_CACHE_DIR` (empty disables), `ZENGPU_NO_CACHE=1` disables.
/// Otherwise `$XDG_CACHE_HOME`, then `$HOME/.cache`, or `%LOCALAPPDATA%` on Windows. `None` when
/// caching is off.
fn resolve_cache_base(
    cache_dir: Option<&OsStr>,
    no_cache: bool,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
    local_app_data: Option<&OsStr>,
) -> Option<PathBuf> {
    if no_cache {
        return None;
    }
    let non_empty = |value: Option<&OsStr>| value.filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = cache_dir {
        return non_empty(Some(dir));
    }
    if cfg!(windows) {
        non_empty(local_app_data)
    } else {
        non_empty(xdg_cache_home).or_else(|| non_empty(home).map(|home| home.join(".cache")))
    }
}

fn default_cache_base() -> Option<PathBuf> {
    resolve_cache_base(
        env::var_os("ZENGPU_CACHE_DIR").as_deref(),
        env::var_os("ZENGPU_NO_CACHE").is_some_and(|v| v.to_str() == Some("1")),
        env::var_os("XDG_CACHE_HOME").as_deref(),
        env::var_os("HOME").as_deref(),
        env::var_os("LOCALAPPDATA").as_deref(),
    )
}

fn cache_entry(base: &Path, key: u64, cubin: bool) -> PathBuf {
    let ext = if cubin { "cubin" } else { "ptx" };
    base.join("zengpu")
        .join("cuda")
        .join(format!("{key:016x}.{ext}"))
}

/// Read the cached image for `key`, cubin first, then PTX. Read errors other than a missing entry
/// log and count as a miss.
fn cache_read(base: &Path, key: u64) -> Option<(ModuleImage, PathBuf)> {
    for cubin in [true, false] {
        let path = cache_entry(base, key, cubin);
        match fs::read(&path) {
            Ok(bytes) if cubin => return Some((ModuleImage::Cubin(bytes), path)),
            Ok(bytes) => return Some((ModuleImage::ptx(bytes), path)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => log::debug!("cuda: cache read {} failed: {e}", path.display()),
        }
    }
    None
}

/// Write `bytes` to `path` via temp file and rename, so readers never see a partial entry. Creates
/// parent dirs.
fn write_entry(path: &Path, bytes: &[u8]) -> io::Result<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().unwrap_or(path);
    fs::create_dir_all(dir)?;
    let (tmp, mut file) = loop {
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(
            ".tmp.{}.{}",
            process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp = PathBuf::from(tmp);
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => break (tmp, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let written = file.write_all(bytes);
    drop(file);
    written
        .and_then(|()| fs::rename(&tmp, path))
        .inspect_err(|_| {
            let _ = fs::remove_file(&tmp);
        })
}

/// Store `image` under `key`. Failures are logged, not returned: the module is already loaded.
fn cache_write(base: &Path, key: u64, image: &ModuleImage) {
    let path = cache_entry(base, key, matches!(image, ModuleImage::Cubin(_)));
    if let Err(e) = write_entry(&path, image.bytes()) {
        log::debug!("cuda: cache write {} failed: {e}", path.display());
    }
}

fn check_buffer_range(offset: u64, bytes: u64, capacity: u64) -> Result<()> {
    if offset.checked_add(bytes).is_none_or(|end| end > capacity) {
        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
            format!("cuda: buffer range at {offset} of length {bytes} exceeds {capacity}"),
        )));
    }
    Ok(())
}

/// Convert a byte count to the driver's `size_t`, failing instead of truncating.
fn driver_len(n: impl TryInto<sys::size_t>) -> Result<sys::size_t> {
    n.try_into().map_err(|_| {
        GpuError::InvalidUsage(UsageError::BindingMismatch(
            "cuda: byte count exceeds driver range".into(),
        ))
    })
}

/// CUDA entry point. Calls `cuInit` at construction. Without CUDA (stub library or no driver),
/// `enumerate_adapters` returns empty.
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
                        vendor: 0x10de,
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
        let _ = req;
        self.enumerate_adapters().into_iter().next()
    }
}

pub struct CudaAdapter {
    dev: Device,
    info: AdapterInfo,
}

// SAFETY: `Device` wraps a CUdevice (c_int ordinal), safe across threads.
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
                images: Mutex::new(HashMap::new()),
            }),
            buffers: Arc::new(Mutex::new(SlotMap::new())),
            shaders: Arc::new(Mutex::new(SlotMap::new())),
            pipelines: Arc::new(Mutex::new(SlotMap::new())),
            lifetime: Arc::new(Mutex::new(CudaLifetimeState::default())),
        }))
    }
}

#[derive(Clone, Copy)]
struct CudaBuffer {
    ptr: u64,
    len: u64,
    usage: BufferUsage,
}

struct CudaShader {
    module: sys::CUmodule,
    _image: Arc<ModuleImage>,
    /// CUDA C++ source, read for kernel signatures.
    source: Option<String>,
}

// SAFETY: CUmodule is opaque; all access goes through ctx_lock.
unsafe impl Send for CudaShader {}
unsafe impl Sync for CudaShader {}

#[derive(Clone, Copy)]
struct CudaPipeline {
    func: sys::CUfunction,
    block: [u32; 3],
    /// Pointer parameter count in the kernel signature. Zero if unreadable.
    ptr_params: usize,
    /// Bit `i` is set when pointer parameter `i` may be written. See `writes_buffer`.
    writes: u32,
}

// SAFETY: CUfunction is opaque; all access goes through ctx_lock.
unsafe impl Send for CudaPipeline {}
unsafe impl Sync for CudaPipeline {}

enum DeferredCudaResource {
    Buffer(u32, CudaBuffer),
    Shader(u32, CudaShader),
    Pipeline(u32),
}

/// Last accesses to a buffer, by base device pointer. Each event covers all earlier accesses, so
/// only the latest read and latest write are kept.
#[derive(Default)]
struct BufferHazard {
    last_access: Option<Access>,
    last_write: Option<Access>,
}

#[derive(Clone)]
struct Access {
    event: Arc<OwnedEvent>,
    lane: Lane,
}

/// Streams work is queued on. Order holds within one lane; across lanes, use an event wait.
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

/// Pinned staging buffer size. 4 MiB measured fastest of 4, 8, and 16 MiB on an RTX 4090 (WSL),
/// though the gap is within noise.
const STAGING_CHUNK: usize = 4 << 20;

/// Host transfers this size or larger use the pinned double buffer. Smaller ones are one
/// synchronous driver copy.
const STAGING_MIN: usize = 1 << 20;

/// Two pinned host buffers, each with an event on its last DMA. `pending[i]` is set while a DMA
/// from `bufs[i]` may be in flight.
struct Staging {
    bufs: [*mut c_void; 2],
    events: [sys::CUevent; 2],
    pending: [bool; 2],
}

impl Staging {
    /// Allocate buffers and events. Context must be current.
    fn new() -> Result<Self> {
        let mut st = Staging {
            bufs: [ptr::null_mut(); 2],
            events: [ptr::null_mut(); 2],
            pending: [false; 2],
        };
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

/// Host copies this large or larger are split across threads. One core cannot keep up with the copy
/// engines (13 GB/s or more).
const PARALLEL_COPY_MIN: usize = 8 << 20;

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

/// Spin on `cuEventQuery` until `ev` completes. Not `cuEventSynchronize`: the context uses
/// blocking-sync scheduling, which adds an OS sleep to every wait. Context must be current.
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

/// Recycled timing-disabled events. Each `OwnedEvent` shares the pool, so an event returns here
/// when its last reference drops.
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

/// Recorded event, recycled after its last holder drops it. A buffer hazard and a submission can
/// share one stream position.
struct OwnedEvent {
    ev: sys::CUevent,
    pool: Arc<EventPool>,
}

// SAFETY: the event is an opaque handle, recorded and queried only in the serialized context.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

impl OwnedEvent {
    /// Record an event on `stream`. Context must be current.
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

// SAFETY: pinned block is plain host memory; the transfer's events order access.
unsafe impl Send for PinnedBlock {}
unsafe impl Sync for PinnedBlock {}

/// Bytes per pinned half-block of an async upload. Uploads stage through two halves, so a large
/// write needs `2 * UPLOAD_CHUNK` pinned bytes.
const UPLOAD_CHUNK: usize = 8 << 20;

/// Pinned sizes round up to a class: powers of two up to this step, then multiples of it. Nearby
/// sizes share blocks.
const PINNED_CLASS_STEP: usize = 4 << 20;

/// Max pinned bytes kept free. Blocks returned past this are freed, oldest first, on
/// the next lease.
const PINNED_RETAIN_MAX: usize = 512 << 20;

fn pinned_class(len: usize) -> usize {
    if len <= PINNED_CLASS_STEP {
        len.next_power_of_two()
    } else {
        len.div_ceil(PINNED_CLASS_STEP) * PINNED_CLASS_STEP
    }
}

/// Free pinned blocks for async transfers. Leased per transfer and reused, so pinned allocation is
/// not paid per call.
struct PinnedPool(Mutex<Vec<PinnedBlock>>);

impl PinnedPool {
    /// Lease a block of at least `len` bytes. Context must be current.
    fn lease(self: &Arc<Self>, len: usize) -> Result<PinnedLease> {
        let class = pinned_class(len);
        let reused = {
            let mut free = self.0.lock().map_err(|_| GpuError::DeviceLost)?;
            let best = free
                .iter()
                .enumerate()
                .filter(|(_, b)| b.cap >= class)
                .min_by_key(|(_, b)| b.cap)
                .map(|(i, _)| i);
            best.map(|i| free.remove(i))
        };
        let block = match reused {
            Some(block) => block,
            None => {
                let mut ptr: *mut c_void = ptr::null_mut();
                // SAFETY: writes one pinned allocation pointer into `ptr`.
                cu(unsafe { sys::cuMemHostAlloc(&mut ptr, driver_len(class)?, 0) })?;
                PinnedBlock { ptr, cap: class }
            }
        };
        self.trim();
        Ok(PinnedLease {
            block,
            pool: Arc::clone(self),
        })
    }

    /// Free the oldest free blocks until retained bytes fit [`PINNED_RETAIN_MAX`]. Context
    /// must be current.
    fn trim(&self) {
        let Ok(mut free) = self.0.lock() else { return };
        let mut retained: usize = free.iter().map(|b| b.cap).sum();
        while retained > PINNED_RETAIN_MAX && !free.is_empty() {
            let block = free.remove(0);
            retained -= block.cap;
            // SAFETY: a free block is not leased, so nothing touches it.
            unsafe { sys::cuMemFreeHost(block.ptr) };
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
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

/// Pinned block held until its transfer finishes and a readback's bytes are taken. Returns the
/// block to its pool on drop.
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

/// Batches with at least this many ops are candidates for graph replay.
const GRAPH_MIN_OPS: usize = 4;

/// Instantiated graphs kept per device. One is evicted when full.
const GRAPH_MAX_READY: usize = 64;

/// Non-graph cache entries are dropped all at once past this count.
const GRAPH_MAX_UNREADY: usize = 1024;

/// Kernel param slots per launch. Fixed size keeps launches heap-free.
const MAX_PARAMS: usize = 32;

/// One resolved driver call. Values are held by value, so one list can be issued directly or
/// recorded into a graph.
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
        /// Each parameter takes an 8-byte slot. Scalars use the low 4 bytes.
        params: [[u8; 8]; MAX_PARAMS],
        n_params: usize,
        /// First `n_buffers` params are buffer base pointers, the rest scalars.
        n_buffers: usize,
        /// Written-buffer bits, by buffer param index. See `writes_buffer`.
        writes: u32,
    },
    Copy {
        dst: u64,
        src: u64,
        len: sys::size_t,
        /// Base pointers of whole buffers, so a sub-range maps back to its buffer.
        dst_buf: u64,
        src_buf: u64,
    },
}

/// Instantiated batch graph. The template `graph` lives beside `exec`, since parameter updates need
/// its node handles.
struct Replay {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    /// One entry per op, in stream order, with the values its node was last set to.
    nodes: Vec<(sys::CUgraphNode, Launch)>,
}

impl Replay {
    /// Destroy exec and template. Context must be current.
    fn destroy(self) {
        // SAFETY: both handles came from `CudaShared::build_graph` and leave the cache here, once.
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}

enum GraphEntry {
    /// First sighting: launched directly. The next batch with this structure is built.
    Seen,
    /// Instantiated graph, replayed with `cuGraphLaunch` after patching changed nodes.
    Ready(Replay),
    /// Build or patch failed. Always launched directly.
    Uncachable,
}

/// Per-device graph cache, keyed by batch structure (see `plan_key`). A hit replays the same
/// kernels and copies with the batch's values.
#[derive(Default)]
struct GraphCache {
    entries: HashMap<Vec<u64>, GraphEntry>,
    ready: usize,
}

impl GraphCache {
    /// Destroy all graphs and forget all entries. Context must be current.
    fn clear(&mut self) {
        for (_, entry) in self.entries.drain() {
            if let GraphEntry::Ready(replay) = entry {
                replay.destroy();
            }
        }
        self.ready = 0;
    }

    /// Store a built graph, evicting one at capacity. Context must be current.
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

    /// Replace `key`'s entry with `Uncachable`, destroying its graph. Context must be current.
    fn disable(&mut self, key: Vec<u64>) {
        if let Some(GraphEntry::Ready(replay)) = self.entries.insert(key, GraphEntry::Uncachable) {
            replay.destroy();
            self.ready -= 1;
        }
    }

    /// Record a first sighting of `key`. Unready entries are dropped when too many build up.
    fn note_seen(&mut self, key: Vec<u64>) {
        if self.entries.len() - self.ready >= GRAPH_MAX_UNREADY {
            self.entries
                .retain(|_, entry| matches!(entry, GraphEntry::Ready(_)));
        }
        self.entries.insert(key, GraphEntry::Seen);
    }
}

/// Whether pointer param `i` may be written. Params past bit 31 count as written.
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

/// Flatten a batch's structure, which graph topology depends on. Pointers, scalars and copy
/// addresses are patched in, not keyed. Buffers become their first-occurrence number, so the key
/// records which launches share a buffer.
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
                #[allow(clippy::unnecessary_cast)]
                let len = *len as u64;
                key.extend([COPY_TAG, len, id_of(*src_buf), id_of(*dst_buf)]);
            }
        }
    }
    key
}

/// For each launch of `plan`, the earlier launches it must wait for: last writer of each buffer it
/// touches (RAW, WAW), and readers since that writer (WAR). Sorted, unique.
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

/// Pointer params of kernel `entry` in CUDA C++ `src`: whether each may be written (read-only only
/// if `const` precedes `*`). None if unreliable: no such kernel, or templates, macros, arrays or
/// function pointers in the list.
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

/// Offset of the `(` after the whole identifier `name`.
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

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

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

/// Pointers to the first `n_params` slots of `params`, in launch form. Valid only while
/// `params` lives.
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

/// Resolves ops in order until one fails. Returns the launches resolved so far along with the
/// error, so the prefix can still be issued.
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

/// Opened CUDA device with compute only (buffers, shaders, dispatch) via the Driver API.
///
/// # Thread safety
///
/// `Context` from cuda-oxide is `!Send`. `ctx_lock` is held for every operation that touches it,
/// and the context is current only on the lock holder's thread.
struct CudaShared {
    /// Owns the context. Dropped after the `Drop` impl of `CudaShared` runs.
    _ctx: Context,
    /// Raw handle of `_ctx`, made current via `with_current` before driver calls.
    raw_ctx: sys::CUcontext,
    ctx_lock: Mutex<()>,
    /// Persistent compute stream, created in `open()` and reused, so kernels skip
    /// per-dispatch create/destroy.
    stream: sys::CUstream,
    /// Non-blocking streams for async transfers, one per direction. Uploads and readbacks run on
    /// copy engines beside kernels. A readback waiting on a kernel does not block the next upload.
    upload_stream: sys::CUstream,
    download_stream: sys::CUstream,
    /// Compute capability, used as the NVRTC target.
    sm: (u32, u32),
    /// Timing-disabled events recycled between submissions.
    events: Arc<EventPool>,
    /// Pinned staging for large host transfers, allocated on first use.
    staging: Mutex<Option<Staging>>,
    pinned: Arc<PinnedPool>,
    graphs: Mutex<GraphCache>,
    /// Device images compiled by this process, keyed by [`cache_key`]. One source compiles once per
    /// device even with the disk cache off.
    images: Mutex<HashMap<u64, Arc<ModuleImage>>>,
}

// SAFETY: every context entry is serialized by ctx_lock.
unsafe impl Send for CudaShared {}
unsafe impl Sync for CudaShared {}

impl CudaShared {
    fn stream(&self, lane: Lane) -> sys::CUstream {
        match lane {
            Lane::Compute => self.stream,
            Lane::Upload => self.upload_stream,
            Lane::Download => self.download_stream,
        }
    }

    /// Runs `f` with the context current on this thread. The context is set only when the
    /// thread's current context differs, so the common path is a single `cuCtxGetCurrent`. It
    /// is never cleared.
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

    /// Takes a pooled event, or creates one. The context must be current.
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

    /// Returns a finished event to the pool. The context must be current.
    fn return_event(&self, ev: sys::CUevent) {
        self.events.put(ev);
    }

    /// Records a completion event on the stream. The context must be current.
    fn record_event(&self) -> Result<sys::CUevent> {
        let ev = self.take_event()?;
        if let Err(error) = cu(unsafe { sys::cuEventRecord(ev, self.stream) }) {
            self.return_event(ev);
            return Err(error);
        }
        Ok(ev)
    }

    /// Waits for everything queued on the stream, spinning rather than sleeping. The context must
    /// be current.
    fn sync_stream(&self) -> Result<()> {
        let ev = self.record_event()?;
        let waited = wait_event(ev);
        self.return_event(ev);
        waited
    }

    /// Issues `plan` on the stream in order. The context must be current, and `ctx_lock` must
    /// be held.
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

    /// Builds `plan` into an instantiated graph without running it. Each node depends only on
    /// earlier nodes it shares a buffer hazard with (`launch_deps`), so independent launches may
    /// run concurrently. Returns `None`, with nothing left behind, if any step fails. The context
    /// must be current.
    fn build_graph(&self, plan: &[Launch]) -> Option<Replay> {
        let mut graph: sys::CUgraph = ptr::null_mut();
        // SAFETY: writes one graph handle into `graph`; no flags are set.
        cu(unsafe { sys::cuGraphCreate(&mut graph, 0) }).ok()?;
        let mut nodes = Vec::with_capacity(plan.len());
        let added = self.add_nodes(graph, plan, &mut nodes).is_ok();
        let mut exec: sys::CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is complete at this point; no error-node or log output is requested.
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

    /// Adds a node per op, each depending on the nodes `launch_deps` gives, and records the values
    /// each node was given. The context must be current.
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
                    // SAFETY: `deps` and `node_params` live across the call; values are copied into
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
                    // SAFETY: `deps` and `copy` live across the call; ranges were bounds-checked
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

    /// Brings `replay` up to `plan` (same structure) by setting only changed nodes. The exec
    /// is updated in place, so topology and enqueued launches are unaffected. The context
    /// must be current.
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
                    // SAFETY: `exec` and `node` belong to this graph and are live; values are
                    cu(unsafe { sys::cuGraphExecKernelNodeSetParams(*exec, *node, &node_params) })?;
                }
                Launch::Copy { dst, src, len, .. } => {
                    let copy = copy_node_params(dst, src, len);
                    // SAFETY: `exec` and `node` belong to this graph and are live; `copy` lives
                    cu(unsafe {
                        sys::cuGraphExecMemcpyNodeSetParams(*exec, *node, &copy, self.raw_ctx)
                    })?;
                }
            }
            *applied = *launch;
        }
        Ok(())
    }

    /// Launches `plan` for a batch of `op_count` ops. A structure is built into a graph the second
    /// time it is seen. Later batches patch changed nodes and replay. The structure includes buffer
    /// sharing, so a replayed graph keeps the deps it was built with. The context must be current,
    /// and `ctx_lock` must be held.
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
                // SAFETY: `exec` was instantiated above and is cached, so it is live.
                cu(unsafe { sys::cuGraphLaunch(exec, self.stream) })
            }
            None => {
                graphs.note_seen(key);
                drop(graphs);
                self.issue(plan)
            }
        }
    }

    /// Destroys all cached graphs, e.g. before their kernels are unloaded. The context must be
    /// current, and `ctx_lock` must be held.
    fn clear_graphs(&self) {
        if let Ok(mut graphs) = self.graphs.lock() {
            graphs.clear();
        }
    }

    /// Load `image` as a module. The driver copies it, so the caller keeps ownership.
    fn load_module(&self, image: &ModuleImage) -> Result<sys::CUmodule> {
        self.with_current(|| {
            let mut module: sys::CUmodule = ptr::null_mut();
            // SAFETY: `image` is alive for the call and its bytes are null-terminated for PTX.
            cu(unsafe {
                sys::cuModuleLoadData(&mut module, image.bytes().as_ptr() as *const c_void)
            })?;
            Ok(module)
        })
    }

    /// Builds the module for CUDA C++ `src`. Checks this process's memo first, then the disk cache
    /// at `cache` (`None` disables it). A cached image that fails to load is deleted and the source
    /// is recompiled.
    fn cuda_module(
        &self,
        src: &[u8],
        cache: Option<&Path>,
    ) -> Result<(sys::CUmodule, Arc<ModuleImage>)> {
        let key = cache_key(src, self.sm, nvrtc_version(), driver_version());
        let memo = self
            .images
            .lock()
            .ok()
            .and_then(|images| images.get(&key).cloned());
        if let Some(image) = memo {
            return Ok((self.load_module(&image)?, image));
        }
        if let Some(base) = cache
            && let Some((image, path)) = cache_read(base, key)
        {
            match self.load_module(&image) {
                Ok(module) => {
                    let image = Arc::new(image);
                    self.remember_image(key, &image);
                    return Ok((module, image));
                }
                Err(e) => {
                    log::debug!("cuda: cached image {} failed to load: {e}", path.display());
                    let _ = fs::remove_file(&path);
                }
            }
        }
        let image = Arc::new(nvrtc_compile(src, self.sm)?);
        let module = self.load_module(&image)?;
        if let Some(base) = cache {
            cache_write(base, key, &image);
        }
        self.remember_image(key, &image);
        Ok((module, image))
    }

    fn remember_image(&self, key: u64, image: &Arc<ModuleImage>) {
        if let Ok(mut images) = self.images.lock() {
            images.insert(key, Arc::clone(image));
        }
    }

    /// Copies host `src` to device `dst`. The context must be current, and `ctx_lock` must be held.
    fn htod(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.len() < STAGING_MIN {
            // SAFETY: `src` is valid for `src.len()` bytes. Pageable memory is staged before
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
                wait_event(st.events[b])?;
                st.pending[b] = false;
            }
            // SAFETY: `bufs[b]` is a live `STAGING_CHUNK`-byte pinned allocation, and `chunk.len()`
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
        for (pending, &ev) in st.pending.iter_mut().zip(&st.events) {
            if *pending {
                wait_event(ev)?;
                *pending = false;
            }
        }
        Ok(())
    }

    /// Copies device `src` into host `dst`. The context must be current, and `ctx_lock` must
    /// be held.
    fn dtoh(&self, src: u64, dst: &mut [u8]) -> Result<()> {
        if dst.len() < STAGING_MIN {
            // SAFETY: `dst` is a valid mutable host slice of `dst.len()` bytes, borrowed until the
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
        let chunk = |j: usize| {
            let off = j * STAGING_CHUNK;
            (off, STAGING_CHUNK.min(total - off))
        };
        let chunks = total.div_ceil(STAGING_CHUNK);
        for i in 0..=chunks {
            if i < chunks {
                let b = i % 2;
                let (off, len) = chunk(i);
                if st.pending[b] {
                    wait_event(st.events[b])?;
                    st.pending[b] = false;
                }
                // SAFETY: `bufs[b]` is a live `STAGING_CHUNK`-byte pinned allocation, `len` is at
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
                // SAFETY: the DMA into bufs[b] completed (waited above), so the first `len` bytes
                let staged = unsafe { slice::from_raw_parts(st.bufs[b] as *const u8, len) };
                dst[off..off + len].copy_from_slice(staged);
            }
        }
        Ok(())
    }
}

impl Drop for CudaShared {
    fn drop(&mut self) {
        // SAFETY: `&mut self` gives exclusive access. `raw_ctx` is alive since `_ctx` drops after
        if unsafe { sys::cuCtxSetCurrent(self.raw_ctx) } == 0 {
            unsafe { sys::cuStreamSynchronize(self.stream) };
            unsafe { sys::cuStreamSynchronize(self.upload_stream) };
            unsafe { sys::cuStreamSynchronize(self.download_stream) };
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
    /// Pinned block of an async transfer. Held until this submission drops, so a readback's bytes
    /// stay valid until taken.
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
        let _ = self.shared.with_current(|| wait_event(ev.ev));
        drop(ev);
        self.release_in_flight();
    }
}

/// Handle to an async buffer readback. Bytes are in the lease of `sub`, valid once `sub` completes.
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
        // SAFETY: the transfer into this block completed (waited above), and the block stays leased
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
        let buffers: Vec<CudaBuffer> = self.buffers.lock().unwrap().drain().collect();
        let shaders: Vec<CudaShader> = self.shaders.lock().unwrap().drain().collect();
        let pipelines: Vec<CudaPipeline> = self.pipelines.lock().unwrap().drain().collect();

        let _ = self.shared.with_current(|| {
            cu(unsafe { sys::cuStreamSynchronize(self.shared.stream) })?;
            cu(unsafe { sys::cuStreamSynchronize(self.shared.upload_stream) })?;
            cu(unsafe { sys::cuStreamSynchronize(self.shared.download_stream) })?;
            if let Ok(mut lifetime) = self.lifetime.lock() {
                lifetime.hazards.clear();
            }
            self.shared.clear_graphs();
            for cb in buffers {
                // SAFETY: ptr came from cuMemAlloc_v2 and is freed exactly once here.
                let _ = unsafe { sys::cuMemFree_v2(cb.ptr) };
            }
            let _ = pipelines;
            for cs in shaders {
                let _ = unsafe { sys::cuModuleUnload(cs.module) };
            }
            Ok(())
        });
    }
}

impl CudaDevice {
    /// Resolves a dispatch into launch values. No driver call is made, so the context need
    /// not be current.
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

    /// Validates and resolves a buffer copy. Zero-length copies need no launch.
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
    /// Builds a shader from CUDA C++ `src` through the compile cache at `cache`.
    fn cuda_shader(&self, src: &[u8], cache: Option<&Path>) -> Result<ShaderHandle> {
        let (module, image) = self.shared.cuda_module(src, cache)?;
        self.insert_shader(module, image, str::from_utf8(src).ok().map(str::to_owned))
    }

    fn insert_shader(
        &self,
        module: sys::CUmodule,
        image: Arc<ModuleImage>,
        source: Option<String>,
    ) -> Result<ShaderHandle> {
        Ok(self
            .shaders
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .insert(CudaShader {
                module,
                _image: image,
                source,
            }))
    }

    /// Makes `stream` wait for earlier transfers or compute events touching `ptrs` on the other
    /// stream. Same-stream events are already ordered. The context must be current, and `stream`
    /// must be one of this device's streams.
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

    /// Records that the work item finishing at `event` on `lane` accessed `accesses`, as `(base
    /// pointer, written)`.
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

    /// Accounts for a submission before any of its work is queued.
    fn begin_submission(&self) -> Result<()> {
        self.lifetime
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .in_flight += 1;
        Ok(())
    }

    /// Drains `stream` after a failed submission, then releases its in-flight count.
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

    /// Queues a resolved compute plan on the compute stream, after the copy-stream work it depends
    /// on. Returns its completion event. If resolution stopped early (`resolve_error`), the prefix
    /// is still queued, and then the error is returned. The context must be current.
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
            self.shared.issue(plan)?;
            return Err(error);
        }
        self.shared.launch_plan(plan, op_count)?;
        let event = OwnedEvent::record(&self.shared, stream)?;
        self.publish_hazards(&accesses, &event, Lane::Compute)?;
        Ok(event)
    }

    /// Stages `data` to `dst` on `stream` through the two halves of `lease`. Returns the event of
    /// the last copy. Blocks only when both halves are in flight. `lease` must hold
    /// `min(data.len(), 2 * UPLOAD_CHUNK)` bytes. The context must be current, and `stream` needs
    /// its hazard waits queued first.
    fn queue_upload(
        &self,
        stream: sys::CUstream,
        lease: &PinnedLease,
        dst: u64,
        data: &[u8],
    ) -> Result<Arc<OwnedEvent>> {
        let base = lease.block.ptr as *mut u8;
        let mut busy: [Option<Arc<OwnedEvent>>; 2] = [None, None];
        let mut last = None;
        for (i, chunk) in data.chunks(UPLOAD_CHUNK).enumerate() {
            let half = i % 2;
            if let Some(previous) = busy[half].take() {
                wait_event(previous.ev)?;
            }
            // SAFETY: the half is inside the lease and its previous copy has completed, so nothing
            let staged =
                unsafe { slice::from_raw_parts_mut(base.add(half * UPLOAD_CHUNK), chunk.len()) };
            copy_host(staged, chunk);
            // SAFETY: range bounds-checked by the caller.
            cu(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst + (i * UPLOAD_CHUNK) as u64,
                    staged.as_ptr() as *const c_void,
                    driver_len(chunk.len())?,
                    stream,
                )
            })?;
            let event = OwnedEvent::record(&self.shared, stream)?;
            busy[half] = Some(Arc::clone(&event));
            last = Some(event);
        }
        last.ok_or_else(|| GpuError::Backend("cuda: empty upload".into()))
    }

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

    fn compute_caps(&self) -> ComputeCaps {
        let (major, minor) = self.shared.sm;
        let queried = self.shared.with_current(|| {
            let mut dev: sys::CUdevice = 0;
            // SAFETY: the context is current; writes one device ordinal.
            cu(unsafe { sys::cuCtxGetDevice(&mut dev) })?;
            let get = |attr: sys::CUdevice_attribute_enum| -> Result<u32> {
                let mut value = 0i32;
                // SAFETY: writes one int for a valid attribute of `dev`.
                cu(unsafe { sys::cuDeviceGetAttribute(&mut value, attr, dev) })?;
                Ok(u32::try_from(value).unwrap_or(0))
            };
            Ok((
                get(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
                get(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
                get(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            ))
        });
        let mut caps = ComputeCaps::scalar_only();
        let Ok((warp, shared, threads)) = queried else {
            return caps;
        };
        if warp == 0 {
            return caps;
        }
        caps.subgroup_sizes = (warp, warp);
        caps.default_subgroup = warp;
        caps.shared_bytes = shared;
        caps.max_workgroup = threads;
        let sm = major * 10 + minor;
        if warp == 32 && sm >= 70 {
            caps.mma.push(MmaShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::F16,
                b: DType::F16,
                acc: DType::F32,
                subgroup: 32,
            });
        }
        if warp == 32 && sm >= 72 {
            caps.mma.push(MmaShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::I8,
                b: DType::I8,
                acc: DType::I32,
                subgroup: 32,
            });
        }
        caps
    }

    fn memory_budget(&self) -> Option<MemoryBudget> {
        self.shared
            .with_current(|| {
                let (mut free, mut total) = (0u64, 0u64);
                // SAFETY: context is current; each pointer is 8 bytes.
                cu(unsafe {
                    sys::cuMemGetInfo_v2(
                        (&raw mut free).cast::<sys::size_t>(),
                        (&raw mut total).cast::<sys::size_t>(),
                    )
                })?;
                Ok(MemoryBudget {
                    device_local: total,
                    budget: free,
                    usage: 0,
                })
            })
            .ok()
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

    fn buffer_size(&self, buffer: BufferHandle) -> Result<u64> {
        let buffers = self.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
        buffers
            .get(buffer)
            .map(|buffer| buffer.len)
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))
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
            check_buffer_range(offset, data.len() as u64, cb.len)?;
            self.shared.htod(cb.ptr + offset, data)
        })
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let len = usize::try_from(len).map_err(|_| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(
                "cuda: read length exceeds host address range".into(),
            ))
        })?;
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
            // SAFETY: `cb.ptr` came from cuMemAlloc_v2 and is no longer in the table.
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
        match desc.source {
            ShaderSource::Ptx(ptx) => {
                let image = Arc::new(ModuleImage::ptx(ptx.to_vec()));
                let module = self.shared.load_module(&image)?;
                self.insert_shader(module, image, None)
            }
            ShaderSource::CudaSrc(src) => self.cuda_shader(src, default_cache_base().as_deref()),
            _ => Err(GpuError::Backend(
                "cuda: only PTX or CUDA C++ (CudaSrc) shaders are supported".into(),
            )),
        }
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
            let lease = self.shared.pinned.lease(data.len().min(2 * UPLOAD_CHUNK))?;
            self.wait_hazards(Lane::Upload, &[cb.ptr])?;
            let event = match self.queue_upload(stream, &lease, cb.ptr + offset, data) {
                Ok(event) => event,
                Err(error) => {
                    let _ = cu(unsafe { sys::cuStreamSynchronize(stream) });
                    return Err(error);
                }
            };
            if let Err(error) = self.publish_hazards(&[(cb.ptr, true)], &event, Lane::Upload) {
                let _ = cu(unsafe { sys::cuStreamSynchronize(stream) });
                return Err(error);
            }
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
            // SAFETY: lease holds at least `len_bytes`; source range bounds-checked.
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

/// Params: `(a: u64, b: u64, c: u64, n: u32)`, device pointers then scalar.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{size_of, size_of_val};
    use std::sync::Barrier;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn cuda_device() -> Option<Box<dyn GpuDevice>> {
        let inst = CudaInstance::new();
        let adapter = inst.enumerate_adapters().into_iter().next()?;
        Some(adapter.open(DeviceRequest::default()).expect("open failed"))
    }

    #[test]
    fn device_info() {
        let inst = CudaInstance::new();
        if !inst.initialized {
            println!("no CUDA driver: skip");
            return;
        }
        let devices = Cuda::list_devices().unwrap_or_default();
        if devices.is_empty() {
            println!("no CUDA devices: skip");
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
        let first: Vec<u8> = (0u8..128).collect();
        let second: Vec<u8> = (128u8..=255).collect();
        device.write_buffer(buf, 0, &first).expect("write first");
        device
            .write_buffer(buf, 128, &second)
            .expect("write second");
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
        const SIZE: u64 = 16 * 1024 * 1024;
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

        let (shader_desc, entry) = TRIG_ZSL
            .for_backend(BackendPreference::Cuda)
            .expect("zsl cuda source");
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

        let (shader_desc, entry) = TYPED_BUFFER_ZSL
            .for_backend(BackendPreference::Cuda)
            .expect("zsl cuda source");
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

        let (shader_desc, entry) = FINITE_ZSL
            .for_backend(BackendPreference::Cuda)
            .expect("zsl cuda source");
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

    /// Round trip of a size that is not a multiple of `STAGING_CHUNK`, at an odd offset, through
    /// the write and read paths.
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

    /// The first call on a thread with no current context works, and the device stays usable from
    /// other threads.
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

    fn as_cuda(device: &dyn GpuDevice) -> &CudaDevice {
        device
            .as_any()
            .downcast_ref::<CudaDevice>()
            .expect("cuda device")
    }

    /// Uses a unique `src` (with a timestamp comment), so no earlier run has cached it.
    fn unique_source(src: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        format!("// {nanos}\n{src}")
    }

    fn unique_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        env::temp_dir().join(format!("zengpu-cuda-{name}-{}-{nanos}", process::id()))
    }

    fn only_entry(base: &Path) -> PathBuf {
        let entries: Vec<PathBuf> = fs::read_dir(base.join("zengpu").join("cuda"))
            .expect("cache directory")
            .map(|e| e.expect("entry").path())
            .collect();
        assert_eq!(entries.len(), 1, "cache entries: {entries:?}");
        entries[0].clone()
    }

    fn run_vec_add(device: &dyn GpuDevice, shader: ShaderHandle) {
        const N: usize = 1024;
        let size = (N * size_of::<f32>()) as u64;
        let desc = BufferDesc {
            size,
            usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let (a, b, out) = (
            device.create_buffer(desc).expect("create a"),
            device.create_buffer(desc).expect("create b"),
            device.create_buffer(desc).expect("create output"),
        );
        let a_bytes: Vec<u8> = (0..N)
            .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = (0..N)
            .flat_map(|i| (1000.0 - i as f32 * 0.5).to_le_bytes())
            .collect();
        device.write_buffer(a, 0, &a_bytes).expect("upload a");
        device.write_buffer(b, 0, &b_bytes).expect("upload b");
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
                    buffers: &[a.index(), b.index(), out.index()],
                    scalars: &[Scalar::U32(N as u32)],
                    textures: &[],
                    images: &[],
                },
                [(N as u32).div_ceil(256), 1, 1],
            )
            .expect("dispatch");
        let raw = device.read_buffer(out, 0, size).expect("read output");
        for i in 0..N {
            let got = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            let want = i as f32 * 0.25 + (1000.0 - i as f32 * 0.5);
            assert!(
                (got - want).abs() < 1e-4,
                "out[{i}] = {got}, expected {want}"
            );
        }
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        for h in [a, b, out] {
            device.destroy_buffer(h);
        }
    }

    /// A cache miss writes one entry, and a fresh device loads it unchanged and runs.
    #[test]
    fn compile_cache_round_trip_runs_kernel() {
        let Some(first) = cuda_device() else { return };
        let base = unique_dir("roundtrip");
        let src = unique_source(VEC_ADD_CUDA);
        let shader = as_cuda(&*first)
            .cuda_shader(src.as_bytes(), Some(&base))
            .expect("compile and cache");
        let entry = only_entry(&base);
        assert!(entry.extension().is_some_and(|e| e == "cubin"), "{entry:?}");
        let written = fs::read(&entry).expect("read entry");
        assert!(!written.is_empty());
        run_vec_add(&*first, shader);

        let Some(second) = cuda_device() else { return };
        let shader = as_cuda(&*second)
            .cuda_shader(src.as_bytes(), Some(&base))
            .expect("load from cache");
        assert_eq!(fs::read(&entry).expect("read entry"), written);
        run_vec_add(&*second, shader);
        let _ = fs::remove_dir_all(&base);
    }

    /// A bad entry is deleted, then rebuilt and rewritten.
    #[test]
    fn compile_cache_corrupt_entry_recompiles() {
        let Some(first) = cuda_device() else { return };
        let base = unique_dir("corrupt");
        let src = unique_source(VEC_ADD_CUDA);
        as_cuda(&*first)
            .cuda_shader(src.as_bytes(), Some(&base))
            .expect("compile and cache");
        let entry = only_entry(&base);

        let garbage = b"not a module image";
        fs::write(&entry, garbage).expect("corrupt entry");
        let Some(second) = cuda_device() else { return };
        let shader = as_cuda(&*second)
            .cuda_shader(src.as_bytes(), Some(&base))
            .expect("recompile after bad entry");
        assert_ne!(fs::read(&entry).expect("read entry"), garbage);
        run_vec_add(&*second, shader);
        let _ = fs::remove_dir_all(&base);
    }

    /// Without a cache base, nothing is written to disk.
    #[test]
    fn compile_cache_disabled_writes_nothing() {
        let Some(device) = cuda_device() else { return };
        let base = unique_dir("disabled");
        let src = unique_source(VEC_ADD_CUDA);
        let shader = as_cuda(&*device)
            .cuda_shader(src.as_bytes(), None)
            .expect("compile without cache");
        run_vec_add(&*device, shader);
        assert!(!base.exists());
    }

    #[test]
    fn compile_cache_memo_serves_repeat_source() {
        let Some(device) = cuda_device() else { return };
        let src = unique_source(VEC_ADD_CUDA);
        let cuda = as_cuda(&*device);
        let (_, first) = cuda
            .shared
            .cuda_module(src.as_bytes(), None)
            .expect("compile");
        let (_, second) = cuda.shared.cuda_module(src.as_bytes(), None).expect("memo");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn cache_base_resolution() {
        let some = |s: &'static str| Some(OsStr::new(s));
        assert_eq!(
            resolve_cache_base(some("/cache"), false, some("/xdg"), some("/home/u"), None),
            Some(PathBuf::from("/cache"))
        );
        assert_eq!(
            resolve_cache_base(some(""), false, some("/xdg"), some("/home/u"), None),
            None
        );
        assert_eq!(
            resolve_cache_base(some("/cache"), true, None, None, None),
            None
        );
        if cfg!(windows) {
            assert_eq!(
                resolve_cache_base(None, false, some("/xdg"), some("/home/u"), some("C:/Local")),
                Some(PathBuf::from("C:/Local"))
            );
        } else {
            assert_eq!(
                resolve_cache_base(None, false, some("/xdg"), some("/home/u"), None),
                Some(PathBuf::from("/xdg"))
            );
            assert_eq!(
                resolve_cache_base(None, false, None, some("/home/u"), None),
                Some(PathBuf::from("/home/u/.cache"))
            );
        }
        assert_eq!(resolve_cache_base(None, false, None, None, None), None);
    }

    /// Entries keep their image kind (cubin or PTX) on reload.
    #[test]
    fn cache_entries_keep_image_kind() {
        let base = unique_dir("kinds");
        let cubin = ModuleImage::Cubin(b"cubin bytes".to_vec());
        let ptx = ModuleImage::ptx(b"ptx text".to_vec());
        cache_write(&base, 1, &cubin);
        cache_write(&base, 2, &ptx);
        assert!(matches!(
            cache_read(&base, 1),
            Some((ModuleImage::Cubin(bytes), _)) if bytes == b"cubin bytes"
        ));
        assert!(matches!(
            cache_read(&base, 2),
            Some((ModuleImage::Ptx(bytes), _)) if bytes == b"ptx text\0"
        ));
        assert!(cache_read(&base, 3).is_none());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn concurrent_cache_writers_publish_whole_entries() {
        let base = unique_dir("concurrent-writers");
        let path = base.join("module.cubin");
        let barrier = Barrier::new(8);
        thread::scope(|scope| {
            for marker in 1..=8u8 {
                let path = &path;
                let barrier = &barrier;
                scope.spawn(move || {
                    let bytes = vec![marker; 64 * 1024];
                    barrier.wait();
                    for _ in 0..16 {
                        write_entry(path, &bytes).expect("publish cache entry");
                        let got = fs::read(path).expect("read published entry");
                        assert_eq!(got.len(), bytes.len());
                        assert!(got.iter().all(|&byte| byte == got[0]), "partial entry");
                    }
                });
            }
        });
        assert_eq!(fs::read_dir(&base).expect("cache dir").count(), 1);
        fs::remove_dir_all(base).expect("remove test cache");
    }

    #[test]
    fn fnv1a_matches_reference_vectors() {
        let hash = |bytes: &[u8]| {
            let mut h = Fnv1a::new();
            h.write(bytes);
            h.0
        };
        assert_eq!(hash(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(hash(b"foobar"), 0x8594_4171_f739_67e8);
    }

    /// Targets the device arch so the module is a cubin, avoiding driver JIT.
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

        let adds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..5 {
            submit_accum(&*device, pipeline, out, &adds, N as u32);
        }
        let mut expected: Vec<f32> = (0..N).map(|i| i as f32 + 5.0 * 36.0).collect();
        assert_eq!(read_f32(&*device, out, N), expected);

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
        for _ in 0..3 {
            submit_accum(&*device, pipeline, out, &adds, N as u32);
        }
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);

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
            kernel(&[(1, true)]),
            kernel(&[(1, false)]),
            kernel(&[(1, false)]),
            kernel(&[(1, true)]),
            kernel(&[(2, true)]),
            kernel(&[(1, false), (2, false)]),
            kernel(&[(3, true), (3, true)]),
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

    /// Spins for about `cycles` clock cycles per thread, then copies `in` to `out`.
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

    /// Writes asynchronously at an offset across several upload chunks, with a ragged tail.
    /// Neighbouring bytes stay untouched.
    #[test]
    fn async_chunked_write_at_offset_lands_exactly() {
        let Some(device) = cuda_device() else { return };
        const OFFSET: usize = 5;
        const LEN: usize = 2 * UPLOAD_CHUNK + 1000 + 3;
        const SIZE: usize = LEN + 2 * OFFSET;
        let buf = device
            .create_buffer(BufferDesc {
                size: SIZE as u64,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .unwrap();
        device.write_buffer(buf, 0, &vec![0xEEu8; SIZE]).unwrap();
        let data: Vec<u8> = (0..LEN).map(|i| (i % 253) as u8).collect();
        let write = device
            .submit_write_buffer(0, buf, OFFSET as u64, &data)
            .unwrap();
        let back = device
            .submit_read_buffer(0, buf, 0, SIZE as u64)
            .unwrap()
            .into_bytes()
            .unwrap();
        write.wait(Duration::from_secs(10)).unwrap();
        assert!(back[..OFFSET].iter().all(|&b| b == 0xEE));
        assert!(back[OFFSET..OFFSET + LEN] == data);
        assert!(back[OFFSET + LEN..].iter().all(|&b| b == 0xEE));
        device.destroy_buffer(buf);
    }

    #[test]
    fn pinned_requests_round_to_size_classes() {
        assert_eq!(pinned_class(1), 1);
        assert_eq!(pinned_class(5), 8);
        assert_eq!(pinned_class(PINNED_CLASS_STEP), PINNED_CLASS_STEP);
        assert_eq!(pinned_class(PINNED_CLASS_STEP + 1), 2 * PINNED_CLASS_STEP);
        assert_eq!(pinned_class((64 << 20) - 1), 64 << 20);
    }

    #[test]
    fn upload_bounds_reject_integer_overflow_and_accept_an_empty_tail() {
        assert!(check_buffer_range(u64::MAX, 1, 256).is_err());
        assert!(check_buffer_range(u64::MAX - 3, 8, u64::MAX).is_err());
        assert!(check_buffer_range(257, 0, 256).is_err());
        assert!(check_buffer_range(255, 2, 256).is_err());
        assert!(check_buffer_range(256, 0, 256).is_ok());
        assert!(check_buffer_range(255, 1, 256).is_ok());
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
}
