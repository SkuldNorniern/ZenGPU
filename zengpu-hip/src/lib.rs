//! ZenGPU AMD ROCm/HIP compute backend.
//!
//! ROCm 5.x to 7.x on gfx9, gfx10 (RDNA 1/2), gfx11 (RDNA 3) and gfx12 (RDNA 4). Kernels are
//! compiled by hipRTC for the exact gfx target, so parts outside AMD's support list run
//! without `HSA_OVERRIDE_GFX_VERSION`.
//!
//! Each [`HipDevice`] has three non-blocking streams: kernels and device copies, async uploads,
//! async downloads. Work is ordered as on one stream. Streams wait on each other's events only for
//! buffers both touch, so transfers overlap kernels.
//!
//! Batches are plain stream launches rather than hipGraph: on ROCm 7.2, graphs of 1-1000 small
//! kernels ran at 0.80-1.00x stream speed. Open one `HipDevice` per adapter for multi-GPU. They are
//! `Send + Sync`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::ffi::{CStr, CString, c_char, c_void};
use std::hash::{Hash, Hasher};
use std::mem::take;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::{sleep, yield_now};
use std::time::{Duration, Instant};

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, CompletedReadback, CompletedSubmission, ComputeOp, ComputePipelineDesc,
    DeviceLimits, DeviceRequest, DeviceType, DispatchOp, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, GpuReadback, GpuSubmission, HalCapabilities, MemoryBudget, MemoryUsage,
    PipelineHandle, PowerPreference, ReadSubmission, Result, SamplerDesc, SamplerHandle, Scalar,
    ShaderDesc, ShaderHandle, ShaderSource, SlotMap, Submission, SubmissionStatus, TextureDesc,
    TextureHandle, UsageError, marker,
};

mod hip_layout {
    include!(concat!(env!("OUT_DIR"), "/hip_layout.rs"));
}
use hip_layout::*;

pub mod version;
use version::{GfxFamily, HipCapabilities, RocmVersion};

pub mod workarounds;

mod ffi;
use ffi::*;

mod alloc;
use alloc::BlockCache;

mod cache;

mod peer;
use peer::PeerStaging;

mod pinned;
use pinned::{PinnedLease, PinnedPool};

const STAGE_CHUNK: usize = 1 << 20;

const HIP_SUCCESS: HipError = 0;
const HIP_ERROR_OUT_OF_MEMORY: HipError = 2;
const HIP_ERROR_INVALID_IMAGE: HipError = 200;
const HIP_ERROR_NO_BINARY_FOR_GPU: HipError = 209;
const HIP_ERROR_INVALID_KERNEL_FILE: HipError = 218;
const HIP_ERROR_NOT_FOUND: HipError = 500;
const HIP_ERROR_NOT_READY: HipError = 600;
const HIP_ERROR_ILLEGAL_ADDRESS: HipError = 700;
const HIP_ERROR_LAUNCH_OUT_OF_RESOURCES: HipError = 701;
const HIP_ERROR_LAUNCH_TIMEOUT: HipError = 702;
const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipError = 704;
const HIP_ERROR_LAUNCH_FAILURE: HipError = 719;

const HIP_MEMCPY_H2D: i32 = 1;
const HIP_MEMCPY_D2H: i32 = 2;
const HIP_MEMCPY_D2D: i32 = 3;
/// Inferred from unified addresses. Used for pinned host buffers (APU fallback), where the explicit
/// kinds would mislead.
const HIP_MEMCPY_DEFAULT: i32 = 4;

const HIP_HOST_MALLOC_MAPPED: u32 = 2;
/// First `hipFunction_attribute` member, the same in every ROCm release.
const HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 0;

const HIP_STREAM_NON_BLOCKING: u32 = 1;
const HIP_EVENT_DISABLE_TIMING: u32 = 2;

const SPIN_BEFORE_SLEEP: Duration = Duration::from_micros(200);
const POLL_SLEEP: Duration = Duration::from_micros(100);

/// `hipDeviceScheduleBlockingSync`: waiting threads sleep in the OS instead of spinning.
const HIP_DEVICE_SCHEDULE_BLOCKING_SYNC: u32 = 4;

pub const SPIN_WAIT_ENV: &str = "ZENGPU_HIP_SPIN_WAIT";

/// Parameter slots per launch, 8 bytes each. A 4-byte scalar uses the low half;
/// `hipModuleLaunchKernel` reads exactly the declared size.
const MAX_PARAMS: usize = 32;

fn error_name(code: HipError) -> String {
    // SAFETY: hipGetErrorName returns a static NUL-terminated string (or null
    let name = unsafe { hipGetErrorName(code) };
    if name.is_null() {
        return format!("HIP error {code}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

/// Whether `code` leaves the HIP context corrupt, so every later call on the device fails.
fn is_fatal(code: HipError) -> bool {
    matches!(
        code,
        HIP_ERROR_ILLEGAL_ADDRESS | HIP_ERROR_LAUNCH_FAILURE | HIP_ERROR_LAUNCH_TIMEOUT
    )
}

fn hip_error(code: HipError, ctx: &str) -> GpuError {
    match code {
        HIP_ERROR_OUT_OF_MEMORY => GpuError::OutOfMemory(MemoryUsage::GpuOnly),
        HIP_ERROR_INVALID_IMAGE | HIP_ERROR_NO_BINARY_FOR_GPU | HIP_ERROR_INVALID_KERNEL_FILE => {
            GpuError::ShaderCompile(format!("{ctx}: {} ({code})", error_name(code)))
        }
        HIP_ERROR_LAUNCH_OUT_OF_RESOURCES => {
            GpuError::Dispatch(format!("{ctx}: {} ({code})", error_name(code)))
        }
        code if is_fatal(code) => GpuError::DeviceLost,
        _ => GpuError::Backend(format!("{ctx}: {} ({code})", error_name(code))),
    }
}

#[inline]
fn check(code: HipError, ctx: &str) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(hip_error(code, ctx))
    }
}

fn binding_error(msg: impl Into<String>) -> GpuError {
    GpuError::InvalidUsage(UsageError::BindingMismatch(msg.into()))
}

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| GpuError::DeviceLost)
}

fn checked_range(offset: u64, len: u64, size: usize, what: &str) -> Result<usize> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| binding_error(format!("hip: {what} range overflows u64")))?;
    if end > size as u64 {
        return Err(binding_error(format!(
            "hip: {what} range {offset}..{end} exceeds buffer of {size} bytes"
        )));
    }
    Ok(offset as usize)
}

/// Larger than any ROCm `hipDeviceProp_t`, so a wrong probe size cannot make the runtime write past
/// the buffer.
const PROP_BLOB_SIZE: usize = if HIP_PROP_SIZE > 8192 {
    HIP_PROP_SIZE
} else {
    8192
};

/// Reads `gcnArchName` from an opaque `hipDeviceProp_t` blob.
///
/// Uses the build-time offset when the runtime's layout matches the build headers. Otherwise scans
/// for the NUL-delimited `gfx…` string. A gfx-named marketing name is the same target.
fn gcn_arch_from_prop(blob: &[u8], abi: Option<PropsAbi>) -> String {
    let headers_abi = if ROCM_VERSION_MAJOR >= 6 {
        PropsAbi::R0600
    } else {
        PropsAbi::R0000
    };
    if abi == Some(headers_abi)
        && let Some(name) = gfx_string_at(blob, HIP_PROP_GCN_ARCH_OFF)
    {
        return name;
    }
    (0..blob.len().saturating_sub(3))
        .filter(|&i| (i == 0 || blob[i - 1] == 0) && blob[i..].starts_with(b"gfx"))
        .find_map(|i| gfx_string_at(blob, i))
        .unwrap_or_default()
}

fn gfx_string_at(blob: &[u8], start: usize) -> Option<String> {
    let field = blob.get(start..(start + HIP_PROP_GCN_ARCH_LEN).min(blob.len()))?;
    let len = field.iter().position(|&b| b == 0)?;
    let name = str::from_utf8(&field[..len]).ok()?;
    let target = name.split(':').next()?;
    let valid = target.len() > 3
        && target.starts_with("gfx")
        && target[3..]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-');
    valid.then(|| name.to_string())
}

fn device_attribute(attr: i32, ordinal: i32) -> Option<i32> {
    let mut value = 0;
    let rc = unsafe { hipDeviceGetAttribute(&mut value, attr, ordinal) };
    (rc == HIP_SUCCESS).then_some(value)
}

fn positive_attribute(attr: i32, ordinal: i32) -> Option<u32> {
    device_attribute(attr, ordinal)
        .filter(|&v| v > 0)
        .map(|v| v as u32)
}

#[derive(Debug, Clone)]
pub struct HipDeviceInfo {
    pub ordinal: i32,
    pub name: String,
    /// Bare gfx target, e.g. "gfx1200". Empty: hipRTC picks the target.
    pub gfx_target: String,
    pub gfx_family: GfxFamily,
    pub total_mem: usize,
    pub cu_count: u32,
    pub clock_mhz: u32,
    pub wave_size: u32,
    pub integrated: bool,
    pub max_threads_per_block: u32,
    pub max_block_dim: [u32; 3],
    pub max_grid_dim: [u32; 3],
    pub capabilities: HipCapabilities,
}

impl HipDeviceInfo {
    fn query(ordinal: i32) -> Option<Self> {
        let mut name_buf = [0 as c_char; 256];
        unsafe {
            if hipDeviceGetName(name_buf.as_mut_ptr(), 256, ordinal) != HIP_SUCCESS {
                return None;
            }
        }
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total_mem: usize = 0;
        if unsafe { hipDeviceTotalMem(&mut total_mem, ordinal) } != HIP_SUCCESS {
            total_mem = 0;
        }

        let mut blob = vec![0u8; PROP_BLOB_SIZE];
        let prop_ok = unsafe { hipGetDeviceProperties(blob.as_mut_ptr(), ordinal) } == HIP_SUCCESS;
        let raw_arch = if prop_ok {
            gcn_arch_from_prop(&blob, props_abi())
        } else {
            String::new()
        };
        let gfx_target = raw_arch
            .split_once(':')
            .map(|(g, _)| g.to_string())
            .unwrap_or(raw_arch);

        let mut family = GfxFamily::from_gfx(&gfx_target);
        if family == GfxFamily::Unknown {
            let (mut major, mut minor) = (0, 0);
            if unsafe { hipDeviceComputeCapability(&mut major, &mut minor, ordinal) } == HIP_SUCCESS
            {
                family = GfxFamily::from_compute_capability(major, minor);
            }
        }

        let wave_size = if family.is_rdna() && env::var_os(version::WAVE64_ENV).is_some() {
            64
        } else {
            device_attribute(HIP_ATTR_WARP_SIZE, ordinal)
                .filter(|&w| w == 32 || w == 64)
                .map(|w| w as u32)
                .unwrap_or_else(|| family.wave_size())
        };

        let runtime = RocmVersion::runtime();
        let mut caps = HipCapabilities::for_family(runtime, runtime, family);
        caps.wave_size = wave_size;

        let max_threads_per_block =
            positive_attribute(HIP_ATTR_MAX_THREADS_PER_BLOCK, ordinal).unwrap_or(1024);
        let max_block_dim = [
            positive_attribute(HIP_ATTR_MAX_BLOCK_DIM_X, ordinal).unwrap_or(1024),
            positive_attribute(HIP_ATTR_MAX_BLOCK_DIM_Y, ordinal).unwrap_or(1024),
            positive_attribute(HIP_ATTR_MAX_BLOCK_DIM_Z, ordinal).unwrap_or(1024),
        ];
        let max_grid_dim = [
            positive_attribute(HIP_ATTR_MAX_GRID_DIM_X, ordinal).unwrap_or(i32::MAX as u32),
            positive_attribute(HIP_ATTR_MAX_GRID_DIM_Y, ordinal).unwrap_or(65535),
            positive_attribute(HIP_ATTR_MAX_GRID_DIM_Z, ordinal).unwrap_or(65535),
        ];

        Some(Self {
            ordinal,
            name,
            gfx_target,
            gfx_family: family,
            total_mem,
            cu_count: positive_attribute(HIP_ATTR_MULTIPROCESSOR_COUNT, ordinal).unwrap_or(0),
            clock_mhz: positive_attribute(HIP_ATTR_CLOCK_RATE, ordinal).unwrap_or(0) / 1000,
            wave_size,
            integrated: device_attribute(HIP_ATTR_INTEGRATED, ordinal).is_some_and(|v| v != 0),
            max_threads_per_block,
            max_block_dim,
            max_grid_dim,
            capabilities: caps,
        })
    }
}

/// Keyed by target, options and source: each code object compiles once.
type CodeCache = Mutex<HashMap<u64, Vec<(String, Vec<u8>, Vec<String>, Arc<Vec<u8>>)>>>;

fn code_cache() -> &'static CodeCache {
    static CACHE: OnceLock<CodeCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeOrigin {
    Memory,
    Disk,
    Compiled,
}

/// Compiles `src` for `target` (gfx name, or the device when unknown). Checks the in-process cache,
/// then `disk`.
fn compile_hip_src(
    src: &[u8],
    opts: &[String],
    target: &str,
    disk: Option<&Path>,
) -> Result<(Arc<Vec<u8>>, CodeOrigin)> {
    static COMPILE: Mutex<()> = Mutex::new(());
    let _serial = lock(&COMPILE)?;
    let mut hasher = DefaultHasher::new();
    target.hash(&mut hasher);
    src.hash(&mut hasher);
    opts.hash(&mut hasher);
    let key = hasher.finish();
    if let Some(bucket) = lock(code_cache())?.get(&key)
        && let Some((_, _, _, code)) = bucket
            .iter()
            .find(|(t, s, o, _)| t == target && s == src && o == opts)
    {
        return Ok((Arc::clone(code), CodeOrigin::Memory));
    }

    let disk_key = disk.map(|_| cache::CacheKey::new(src, opts, target, runtime_version_raw()));
    let from_disk = disk
        .zip(disk_key.as_ref())
        .and_then(|(dir, key)| cache::load(dir, key));
    let (code, origin) = match from_disk {
        Some(code) => (code, CodeOrigin::Disk),
        None => {
            let code = hiprtc_compile(src, opts)?;
            if let (Some(dir), Some(key)) = (disk, disk_key.as_ref()) {
                cache::store(dir, key, &code);
            }
            (code, CodeOrigin::Compiled)
        }
    };
    let code = Arc::new(code);
    lock(code_cache())?.entry(key).or_default().push((
        target.to_string(),
        src.to_vec(),
        opts.to_vec(),
        Arc::clone(&code),
    ));
    Ok((code, origin))
}

/// Compiles `src` for `gfx` without a device and returns the code object size, so kernels can be
/// checked for GPUs that are not installed.
pub fn compile_for_target(src: &str, gfx: &str) -> Result<usize> {
    let rocm = RocmVersion::runtime().min(RocmVersion::COMPILE_TIME);
    let caps = HipCapabilities::for_family(rocm, rocm, GfxFamily::from_gfx(gfx));
    let opts = caps.rtc_options(gfx);
    Ok(compile_hip_src(src.as_bytes(), &opts, gfx, None)?.0.len())
}

fn evict_code(src: &[u8], opts: &[String], target: &str, disk: Option<&Path>) {
    if let Ok(mut cache) = code_cache().lock() {
        for bucket in cache.values_mut() {
            bucket.retain(|(t, s, o, _)| !(t == target && s == src && o == opts));
        }
    }
    if let Some(dir) = disk {
        cache::evict(
            dir,
            &cache::CacheKey::new(src, opts, target, runtime_version_raw()),
        );
    }
}

fn runtime_version_raw() -> i32 {
    let mut version = 0;
    unsafe { hipRuntimeGetVersion(&mut version) };
    version
}

fn hiprtc_compile(src: &[u8], opts: &[String]) -> Result<Vec<u8>> {
    if !ffi::hiprtc_loaded() {
        return Err(GpuError::ShaderCompile(
            "hip: hipRTC (libhiprtc) not found; install the ROCm hiprtc package".into(),
        ));
    }
    let src_c = CString::new(src)
        .map_err(|_| GpuError::ShaderCompile("HIP source contains null byte".into()))?;
    let name_c = CString::new("zen_kernel.hip").unwrap();
    let opts_owned: Vec<CString> = opts
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let opt_ptrs: Vec<*const c_char> = opts_owned.iter().map(|s| s.as_ptr()).collect();

    let mut prog: HiprtcProg = ptr::null_mut();
    unsafe {
        let rc = hiprtcCreateProgram(
            &mut prog,
            src_c.as_ptr(),
            name_c.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        );
        if rc != 0 {
            return Err(GpuError::ShaderCompile(format!(
                "hiprtcCreateProgram failed ({rc})"
            )));
        }

        let rc = hiprtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        if rc != 0 {
            let log = fetch_rtc_log(prog);
            hiprtcDestroyProgram(&mut prog);
            return Err(GpuError::ShaderCompile(format!(
                "hiprtcCompileProgram failed ({rc}) with {opts:?}:\n{log}"
            )));
        }

        let mut code_sz: usize = 0;
        let rc = hiprtcGetCodeSize(prog, &mut code_sz);
        if rc != 0 || code_sz == 0 {
            hiprtcDestroyProgram(&mut prog);
            return Err(GpuError::ShaderCompile(format!(
                "hiprtcGetCodeSize failed ({rc}) or returned an empty code object"
            )));
        }
        let mut code = vec![0u8; code_sz];
        let rc = hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char);
        hiprtcDestroyProgram(&mut prog);
        if rc != 0 {
            return Err(GpuError::ShaderCompile(format!(
                "hiprtcGetCode failed ({rc})"
            )));
        }
        Ok(code)
    }
}

unsafe fn fetch_rtc_log(prog: HiprtcProg) -> String {
    let mut sz: usize = 0;
    unsafe { hiprtcGetProgramLogSize(prog, &mut sz) };
    if sz <= 1 {
        return "(no log)".into();
    }
    let mut buf = vec![0u8; sz];
    unsafe { hiprtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) };
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

#[derive(Clone, Copy)]
struct HipBuffer {
    ptr: *mut c_void,
    size: usize,
    usage: BufferUsage,
    /// Pinned host allocation backing this buffer (APU fallback), else null.
    host: *mut c_void,
    /// Allocated bytes at `ptr`: `size`, or more for a cached block.
    block: usize,
}

impl HipBuffer {
    fn copy_kind(&self, device_kind: i32) -> i32 {
        if self.host.is_null() {
            device_kind
        } else {
            HIP_MEMCPY_DEFAULT
        }
    }
}
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

/// Shared with pipelines, so destroying a shader does not unload code a pipeline still launches.
struct HipModule {
    raw: HipModuleRaw,
}
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl Drop for HipModule {
    fn drop(&mut self) {
        unsafe { hipModuleUnload(self.raw) };
    }
}

#[derive(Clone)]
struct HipPipeline {
    function: HipFunction,
    block: [u32; 3],
    _module: Arc<HipModule>,
}
unsafe impl Send for HipPipeline {}
unsafe impl Sync for HipPipeline {}

enum Deferred {
    Buffer(u32, HipBuffer),
    Shader(u32, Arc<HipModule>),
    Pipeline(u32, HipPipeline),
}

#[derive(Default)]
struct LifetimeState {
    in_flight: usize,
    deferred: Vec<Deferred>,
    /// Last queued access per buffer (by device pointer). Cleared when nothing is in flight.
    hazards: HashMap<usize, Hazard>,
}

/// Streams work is queued on. Order holds within one lane; across lanes, use an event wait.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lane {
    Compute,
    Upload,
    Download,
}

/// Destroyed when the last holder drops it.
struct OwnedEvent(HipEvent);

// SAFETY: hipEvent_t is an opaque handle and HIP event calls are thread-safe.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        unsafe { hipEventDestroy(self.0) };
    }
}

impl Lane {
    const ALL: [Lane; 3] = [Lane::Compute, Lane::Upload, Lane::Download];
}

/// Queued work on one buffer: last access per lane, and the last write.
#[derive(Default)]
struct Hazard {
    last_access: [Option<Arc<OwnedEvent>>; 3],
    last_write: Option<(Lane, Arc<OwnedEvent>)>,
}

/// Buffers `ops` touch as `(ptr, written)`. Kernels are taken to write every bound buffer.
fn accesses(mut list: Vec<(usize, bool)>) -> Vec<(usize, bool)> {
    list.sort_unstable();
    let mut out: Vec<(usize, bool)> = Vec::with_capacity(list.len());
    for (ptr, written) in list {
        match out.last_mut() {
            Some(last) if last.0 == ptr => last.1 |= written,
            _ => out.push((ptr, written)),
        }
    }
    out
}

struct HipCore {
    info: HipDeviceInfo,
    stream: HipStream,
    /// Host-to-device writes, overlapping kernels.
    upload: HipStream,
    download: HipStream,
    buffers: Mutex<SlotMap<marker::Buffer, HipBuffer>>,
    shaders: Mutex<SlotMap<marker::Shader, Arc<HipModule>>>,
    pipelines: Mutex<SlotMap<marker::Pipeline, HipPipeline>>,
    lifetime: Mutex<LifetimeState>,
    peer_access: Mutex<HashMap<i32, bool>>,
    cache_dir: Option<PathBuf>,
    /// Pinned staging for cross-GPU copies, made on first use. `Some(None)`: no pinned memory.
    peer_staging: Mutex<Option<Option<PeerStaging>>>,
    /// Freed blocks, reused on this device's stream.
    block_cache: Mutex<BlockCache>,
    pinned: Arc<PinnedPool>,
    allocated: AtomicU64,
    lost: AtomicBool,
}

// SAFETY: the raw stream is only used through HIP calls, which are
unsafe impl Send for HipCore {}
unsafe impl Sync for HipCore {}

impl HipCore {
    /// Makes this device current on the thread (HIP tracks it per thread). Fails once lost.
    fn activate(&self) -> Result<()> {
        if self.lost.load(Ordering::Acquire) {
            return Err(GpuError::DeviceLost);
        }
        unsafe { self.check(hipSetDevice(self.info.ordinal), "hipSetDevice") }
    }

    /// Converts `code` to a `Result`, marking the device lost when the error is fatal.
    fn check(&self, code: HipError, ctx: &str) -> Result<()> {
        if is_fatal(code) {
            log::error!(
                "hip: device {} lost: {ctx}: {}",
                self.info.ordinal,
                error_name(code)
            );
            self.lost.store(true, Ordering::Release);
        }
        check(code, ctx)
    }

    fn synchronize(&self) -> Result<()> {
        self.activate()?;
        unsafe { self.check(hipStreamSynchronize(self.stream), "hipStreamSynchronize") }
    }

    fn ensure_idle(&self, op: &str) -> Result<()> {
        if lock(&self.lifetime)?.in_flight != 0 {
            return Err(binding_error(format!(
                "{op} cannot run while an asynchronous submission is pending"
            )));
        }
        Ok(())
    }

    fn buffer(&self, handle: BufferHandle) -> Result<HipBuffer> {
        let buffers = lock(&self.buffers)?;
        buffers
            .get(handle)
            .copied()
            .ok_or_else(|| stale(&buffers, handle))
    }

    fn release_in_flight(&self) {
        let deferred = {
            let Ok(mut lifetime) = self.lifetime.lock() else {
                return;
            };
            debug_assert!(lifetime.in_flight > 0);
            lifetime.in_flight = lifetime.in_flight.saturating_sub(1);
            if lifetime.in_flight == 0 {
                lifetime.hazards.clear();
                take(&mut lifetime.deferred)
            } else {
                Vec::new()
            }
        };
        for resource in deferred {
            match resource {
                Deferred::Buffer(index, buffer) => {
                    self.free(buffer);
                    if let Ok(mut buffers) = self.buffers.lock() {
                        buffers.release_retired(index);
                    }
                }
                Deferred::Shader(index, module) => {
                    drop(module);
                    if let Ok(mut shaders) = self.shaders.lock() {
                        shaders.release_retired(index);
                    }
                }
                Deferred::Pipeline(index, pipeline) => {
                    drop(pipeline);
                    if let Ok(mut pipelines) = self.pipelines.lock() {
                        pipelines.release_retired(index);
                    }
                }
            }
        }
    }

    fn free(&self, buffer: HipBuffer) {
        self.allocated
            .fetch_sub(buffer.size as u64, Ordering::Relaxed);
        if !buffer.host.is_null() {
            alloc::free_host_later(buffer.host);
            return;
        }
        if buffer.ptr.is_null() {
            return;
        }
        let cached = self
            .block_cache
            .lock()
            .is_ok_and(|mut cache| cache.put(buffer.ptr, buffer.block));
        if !cached {
            alloc::free_later(self.info.ordinal, buffer.ptr);
        }
    }

    /// Returns cached blocks to HIP and waits for queued frees, so a failed allocation can retry.
    fn release_cached_memory(&self) {
        let blocks = self
            .block_cache
            .lock()
            .map(|mut cache| cache.drain())
            .unwrap_or_default();
        for ptr in blocks {
            unsafe {
                hipSetDevice(self.info.ordinal);
                hipFree(ptr);
            }
        }
        alloc::wait_for_frees(Duration::from_secs(5));
    }

    fn launch(&self, op: &DispatchOp<'_>) -> Result<()> {
        let pipe = {
            let pipelines = lock(&self.pipelines)?;
            pipelines
                .get(op.pipeline)
                .cloned()
                .ok_or_else(|| stale(&pipelines, op.pipeline))?
        };
        if !op.bindings.textures.is_empty() || !op.bindings.images.is_empty() {
            return Err(GpuError::Unsupported(
                "hip: compute-only backend has no texture or image bindings".into(),
            ));
        }
        let n_params = op.bindings.buffers.len() + op.bindings.scalars.len();
        if n_params > MAX_PARAMS {
            return Err(GpuError::Dispatch(format!(
                "hip: {n_params} kernel parameters exceed the limit of {MAX_PARAMS}"
            )));
        }
        if op.grid.contains(&0) {
            return Ok(());
        }
        for axis in 0..3 {
            if op.grid[axis] > self.info.max_grid_dim[axis] {
                return Err(GpuError::Dispatch(format!(
                    "hip: grid {:?} exceeds the device limit {:?}",
                    op.grid, self.info.max_grid_dim
                )));
            }
            if u64::from(op.grid[axis]) * u64::from(pipe.block[axis]) > u64::from(u32::MAX) {
                return Err(GpuError::Dispatch(format!(
                    "hip: grid {:?} × block {:?} exceeds 2^32 threads on one axis",
                    op.grid, pipe.block
                )));
            }
        }

        let mut storage = [0u64; MAX_PARAMS];
        let mut n = 0usize;
        {
            let buffers = lock(&self.buffers)?;
            for &slot in op.bindings.buffers {
                let buffer = buffers.get_by_slot_index(slot).ok_or_else(|| {
                    binding_error(format!("hip: binding refers to unknown buffer slot {slot}"))
                })?;
                storage[n] = buffer.ptr as u64;
                n += 1;
            }
        }
        for scalar in op.bindings.scalars {
            let bytes = match scalar {
                Scalar::U32(v) => v.to_ne_bytes(),
                Scalar::I32(v) => v.to_ne_bytes(),
                Scalar::F32(v) => v.to_bits().to_ne_bytes(),
            };
            let mut slot = [0u8; 8];
            slot[..4].copy_from_slice(&bytes);
            storage[n] = u64::from_ne_bytes(slot);
            n += 1;
        }
        let mut params = [ptr::null_mut::<c_void>(); MAX_PARAMS];
        for (param, slot) in params.iter_mut().zip(storage.iter_mut()).take(n) {
            *param = slot as *mut u64 as *mut c_void;
        }

        unsafe {
            self.check(
                hipModuleLaunchKernel(
                    pipe.function,
                    op.grid[0],
                    op.grid[1],
                    op.grid[2],
                    pipe.block[0],
                    pipe.block[1],
                    pipe.block[2],
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )
        }
    }

    fn enqueue_copy(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        if src == dst {
            return Err(binding_error(
                "copy_buffer requires distinct source and destination buffers",
            ));
        }
        let src_buf = self.buffer(src)?;
        let dst_buf = self.buffer(dst)?;
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
        let src_off = checked_range(src_offset, len, src_buf.size, "source copy")?;
        let dst_off = checked_range(dst_offset, len, dst_buf.size, "destination copy")?;
        if len == 0 {
            return Ok(());
        }
        let kind = if src_buf.host.is_null() {
            dst_buf.copy_kind(HIP_MEMCPY_D2D)
        } else {
            HIP_MEMCPY_DEFAULT
        };
        unsafe {
            self.check(
                hipMemcpyAsync(
                    (dst_buf.ptr as *mut u8).add(dst_off) as *mut c_void,
                    (src_buf.ptr as *const u8).add(src_off) as *const c_void,
                    len as usize,
                    kind,
                    self.stream,
                ),
                "hipMemcpyAsync D→D",
            )
        }
    }

    /// Resources destroyed while the batch is in flight stay alive until it completes.
    fn submit<T>(
        self: &Arc<Self>,
        cycle_id: u64,
        ops: &[T],
        touched: impl Fn(&Self, &T, &mut Vec<(usize, bool)>),
        enqueue: impl Fn(&Self, &T) -> Result<()>,
    ) -> Result<Submission> {
        if ops.is_empty() {
            return Ok(Box::new(CompletedSubmission::new(cycle_id)));
        }
        let mut list = Vec::new();
        for op in ops {
            touched(self, op, &mut list);
        }
        let submission = self.record(cycle_id, Lane::Compute, None, &accesses(list), |core| {
            ops.iter().try_for_each(|op| enqueue(core, op))
        })?;
        Ok(Box::new(submission))
    }

    fn lane_stream(&self, lane: Lane) -> HipStream {
        match lane {
            Lane::Compute => self.stream,
            Lane::Upload => self.upload,
            Lane::Download => self.download,
        }
    }

    /// Dispatch bindings, all taken as written.
    fn dispatch_touches(&self, op: &DispatchOp<'_>, out: &mut Vec<(usize, bool)>) {
        let Ok(buffers) = self.buffers.lock() else {
            return;
        };
        for &slot in op.bindings.buffers {
            if let Some(buffer) = buffers.get_by_slot_index(slot) {
                out.push((buffer.ptr as usize, true));
            }
        }
    }

    /// Runs `enqueue` after other lanes' queued work on `touched`. Keeps `lease` alive until done.
    fn record(
        self: &Arc<Self>,
        cycle_id: u64,
        lane: Lane,
        lease: Option<PinnedLease>,
        touched: &[(usize, bool)],
        enqueue: impl FnOnce(&Self) -> Result<()>,
    ) -> Result<HipSubmission> {
        self.activate()?;
        let stream = self.lane_stream(lane);
        {
            let mut lifetime = lock(&self.lifetime)?;
            lifetime.in_flight += 1;
            if let Err(e) = self.wait_hazards(&lifetime, lane, stream, touched) {
                drop(lifetime);
                self.release_in_flight();
                return Err(e);
            }
        }

        let recorded = (|| {
            enqueue(self)?;
            let mut raw: HipEvent = ptr::null_mut();
            unsafe {
                self.check(
                    hipEventCreateWithFlags(&mut raw, HIP_EVENT_DISABLE_TIMING),
                    "hipEventCreateWithFlags",
                )?;
            }
            let event = Arc::new(OwnedEvent(raw));
            unsafe { self.check(hipEventRecord(raw, stream), "hipEventRecord")? };
            let mut lifetime = lock(&self.lifetime)?;
            for &(ptr, written) in touched {
                let hazard = lifetime.hazards.entry(ptr).or_default();
                if written {
                    hazard.last_write = Some((lane, Arc::clone(&event)));
                }
                hazard.last_access[lane as usize] = Some(Arc::clone(&event));
            }
            Ok(event)
        })();

        match recorded {
            Ok(event) => Ok(HipSubmission {
                cycle_id,
                core: Arc::clone(self),
                event: Mutex::new(Some(event)),
                lease,
            }),
            Err(error) => {
                unsafe { hipStreamSynchronize(stream) };
                self.release_in_flight();
                Err(error)
            }
        }
    }

    /// A write waits for every lane's last access, a read only for the last write.
    fn wait_hazards(
        &self,
        lifetime: &LifetimeState,
        lane: Lane,
        stream: HipStream,
        touched: &[(usize, bool)],
    ) -> Result<()> {
        let wait = |event: &OwnedEvent| unsafe {
            self.check(hipStreamWaitEvent(stream, event.0, 0), "hipStreamWaitEvent")
        };
        for &(ptr, written) in touched {
            let Some(hazard) = lifetime.hazards.get(&ptr) else {
                continue;
            };
            if written {
                for other in Lane::ALL.into_iter().filter(|&l| l != lane) {
                    if let Some(event) = &hazard.last_access[other as usize] {
                        wait(event)?;
                    }
                }
            } else if let Some((writer, event)) = &hazard.last_write
                && *writer != lane
            {
                wait(event)?;
            }
        }
        Ok(())
    }
}

impl Drop for HipCore {
    fn drop(&mut self) {
        unsafe {
            hipSetDevice(self.info.ordinal);
            hipStreamSynchronize(self.stream);
            hipStreamSynchronize(self.upload);
            hipStreamSynchronize(self.download);
        }
        let buffers: Vec<HipBuffer> = self
            .buffers
            .get_mut()
            .map(|b| b.drain().collect())
            .unwrap_or_default();
        for buffer in buffers {
            self.free(buffer);
        }
        let deferred = self
            .lifetime
            .get_mut()
            .map(|l| take(&mut l.deferred))
            .unwrap_or_default();
        for resource in deferred {
            if let Deferred::Buffer(_, buffer) = resource {
                self.free(buffer);
            }
        }
        if let Ok(pipelines) = self.pipelines.get_mut() {
            drop(pipelines.drain().collect::<Vec<_>>());
        }
        if let Ok(shaders) = self.shaders.get_mut() {
            drop(shaders.drain().collect::<Vec<_>>());
        }
        let cached = self
            .block_cache
            .get_mut()
            .map(|cache| cache.drain())
            .unwrap_or_default();
        unsafe {
            for ptr in cached {
                hipFree(ptr);
            }
            hipStreamDestroy(self.stream);
            hipStreamDestroy(self.upload);
            hipStreamDestroy(self.download);
        }
    }
}

fn stale<K, V>(table: &SlotMap<K, V>, handle: zengpu_hal::Handle<K>) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: table.generation_at(handle.index()).unwrap_or(0),
    })
}

struct HipSubmission {
    cycle_id: u64,
    core: Arc<HipCore>,
    event: Mutex<Option<Arc<OwnedEvent>>>,
    /// Dropped only after the event completes.
    lease: Option<PinnedLease>,
}

impl HipSubmission {
    fn query(&self) -> Result<SubmissionStatus> {
        let mut event = lock(&self.event)?;
        let Some(raw) = event.as_ref().map(|e| e.0) else {
            return Ok(SubmissionStatus::Complete);
        };
        match unsafe { hipEventQuery(raw) } {
            HIP_SUCCESS => {
                *event = None;
                self.core.release_in_flight();
                Ok(SubmissionStatus::Complete)
            }
            HIP_ERROR_NOT_READY => Ok(SubmissionStatus::Pending),
            code => {
                *event = None;
                self.core.release_in_flight();
                Err(self
                    .core
                    .check(code, "hipEventQuery")
                    .err()
                    .unwrap_or(GpuError::DeviceLost))
            }
        }
    }
}

impl GpuSubmission for HipSubmission {
    fn cycle_id(&self) -> u64 {
        self.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        self.query()
    }

    fn wait(&self, timeout: Duration) -> Result<()> {
        if timeout == Duration::MAX {
            let event = lock(&self.event)?.clone();
            if let Some(event) = event {
                let code = unsafe { hipEventSynchronize(event.0) };
                if code != HIP_SUCCESS && code != HIP_ERROR_NOT_READY {
                    self.core.check(code, "hipEventSynchronize")?;
                }
            }
        }
        let start = Instant::now();
        let deadline = start.checked_add(timeout);
        loop {
            if self.query()? == SubmissionStatus::Complete {
                return Ok(());
            }
            let now = Instant::now();
            if timeout == Duration::ZERO || deadline.is_some_and(|d| now >= d) {
                return Err(GpuError::Timeout);
            }
            if now - start < SPIN_BEFORE_SLEEP {
                yield_now();
            } else {
                let left = deadline.map_or(POLL_SLEEP, |d| d - now);
                sleep(POLL_SLEEP.min(left));
            }
        }
    }
}

impl Drop for HipSubmission {
    fn drop(&mut self) {
        let Ok(event) = self.event.get_mut() else {
            return;
        };
        let Some(event) = event.take() else {
            return;
        };
        unsafe { hipEventSynchronize(event.0) };
        self.core.release_in_flight();
    }
}

struct HipReadback {
    sub: HipSubmission,
    len: usize,
}

impl HipReadback {
    fn bytes(&self) -> Result<&[u8]> {
        self.sub.wait(Duration::MAX)?;
        let lease = self.sub.lease.as_ref().ok_or(GpuError::DeviceLost)?;
        Ok(lease.bytes(self.len))
    }
}

impl GpuSubmission for HipReadback {
    fn cycle_id(&self) -> u64 {
        self.sub.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        self.sub.poll()
    }

    fn wait(&self, timeout: Duration) -> Result<()> {
        self.sub.wait(timeout)
    }
}

impl GpuReadback for HipReadback {
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>> {
        Ok(self.bytes()?.to_vec())
    }

    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()> {
        let bytes = self.bytes()?;
        if dst.len() != bytes.len() {
            return Err(binding_error(format!(
                "hip: read_into got {} bytes for a {}-byte read",
                dst.len(),
                bytes.len()
            )));
        }
        dst.copy_from_slice(bytes);
        Ok(())
    }
}

/// ```no_run
/// use zengpu_hip::HipInstance;
/// use zengpu_hal::{GpuInstance, AdapterRequest};
///
/// let inst = HipInstance::new().expect("ROCm not available");
/// for adapter in inst.enumerate_adapters() {
///     println!("{}", adapter.info().name);
/// }
/// ```
pub struct HipInstance {
    devices: Vec<HipDeviceInfo>,
}

impl HipInstance {
    pub fn new() -> Result<Self> {
        workarounds::apply_process_env();
        if !ffi::runtime_loaded() {
            return Err(GpuError::Unsupported(
                "hip: ROCm runtime (libamdhip64) not found; install ROCm or set ROCM_PATH".into(),
            ));
        }
        let rc = unsafe { hipInit(0) };
        let with_hint = |e: GpuError| match (e, workarounds::kfd_diagnostic()) {
            (GpuError::Backend(msg), Some(hint)) => GpuError::Backend(format!("{msg}; {hint}")),
            (e, _) => e,
        };
        check(rc, "hipInit").map_err(with_hint)?;

        let mut count: i32 = 0;
        check(
            unsafe { hipGetDeviceCount(&mut count) },
            "hipGetDeviceCount",
        )
        .map_err(with_hint)?;

        if env::var_os(SPIN_WAIT_ENV).is_none() {
            for ordinal in 0..count {
                unsafe {
                    if hipSetDevice(ordinal) == HIP_SUCCESS {
                        let _ = hipSetDeviceFlags(HIP_DEVICE_SCHEDULE_BLOCKING_SYNC);
                    }
                }
            }
        }

        let devices: Vec<HipDeviceInfo> = (0..count).filter_map(HipDeviceInfo::query).collect();

        for d in &devices {
            log::debug!(
                "hip: [{ordinal}] {name} ({gfx}/{family}): {mem} MiB, {cu} CUs @ {clk} MHz, wave{wave}, ROCm {rocm}",
                ordinal = d.ordinal,
                name = d.name,
                gfx = if d.gfx_target.is_empty() {
                    "unknown"
                } else {
                    &d.gfx_target
                },
                family = d.gfx_family.name(),
                mem = d.total_mem / (1024 * 1024),
                cu = d.cu_count,
                clk = d.clock_mhz,
                wave = d.wave_size,
                rocm = d.capabilities.rocm,
            );
            if !d.capabilities.target_supported() {
                log::warn!(
                    "hip: {} ({}) needs a newer ROCm than {}; hipRTC may reject it",
                    d.name,
                    d.gfx_target,
                    d.capabilities.rocm
                );
            }
        }

        Ok(Self { devices })
    }

    pub fn device_infos(&self) -> &[HipDeviceInfo] {
        &self.devices
    }

    fn adapter(d: &HipDeviceInfo) -> HipAdapter {
        HipAdapter {
            info: AdapterInfo {
                name: d.name.clone(),
                vendor: 0x1002,
                device: 0,
                device_type: if d.integrated {
                    DeviceType::Integrated
                } else {
                    DeviceType::Discrete
                },
                backend: BackendPreference::Hip,
            },
            dev_info: d.clone(),
        }
    }
}

impl GpuInstance for HipInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        self.devices
            .iter()
            .map(|d| Box::new(Self::adapter(d)) as Box<dyn GpuAdapter>)
            .collect()
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        pick_device(&self.devices, req.power == PowerPreference::LowPower)
            .map(|d| Box::new(Self::adapter(d)) as Box<dyn GpuAdapter>)
    }
}

/// Requested kind first, then most compute units. Ties keep the lowest ordinal.
fn pick_device(devices: &[HipDeviceInfo], want_integrated: bool) -> Option<&HipDeviceInfo> {
    devices
        .iter()
        .rev()
        .max_by_key(|d| (d.integrated == want_integrated, d.cu_count))
}

pub struct HipAdapter {
    info: AdapterInfo,
    dev_info: HipDeviceInfo,
}

impl GpuAdapter for HipAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        let ordinal = self.dev_info.ordinal;
        check(unsafe { hipSetDevice(ordinal) }, "hipSetDevice")?;
        let mut streams: [HipStream; 3] = [ptr::null_mut(); 3];
        for i in 0..streams.len() {
            let rc = unsafe { hipStreamCreateWithFlags(&mut streams[i], HIP_STREAM_NON_BLOCKING) };
            if let Err(e) = check(rc, "hipStreamCreateWithFlags") {
                for &made in &streams[..i] {
                    unsafe { hipStreamDestroy(made) };
                }
                return Err(e);
            }
        }
        let [stream, upload, download] = streams;
        for &copy in &[upload, download] {
            let mut event: HipEvent = ptr::null_mut();
            unsafe {
                if hipEventCreateWithFlags(&mut event, HIP_EVENT_DISABLE_TIMING) == HIP_SUCCESS {
                    hipEventRecord(event, copy);
                    hipEventSynchronize(event);
                    hipEventDestroy(event);
                }
            }
        }
        Ok(Box::new(HipDevice {
            core: Arc::new(HipCore {
                info: self.dev_info.clone(),
                stream,
                upload,
                download,
                buffers: Mutex::new(SlotMap::new()),
                shaders: Mutex::new(SlotMap::new()),
                pipelines: Mutex::new(SlotMap::new()),
                lifetime: Mutex::new(LifetimeState::default()),
                peer_access: Mutex::new(HashMap::new()),
                cache_dir: req.pipeline_cache_path.as_deref().map(cache::dir_for),
                peer_staging: Mutex::new(None),
                block_cache: Mutex::new(BlockCache::new(
                    (self.dev_info.total_mem / 8).min(512 << 20),
                )),
                pinned: Arc::default(),
                allocated: AtomicU64::new(0),
                lost: AtomicBool::new(false),
            }),
        }))
    }
}

/// Compute-only buffers, hipRTC shaders and kernel dispatch on one ordered stream.
pub struct HipDevice {
    core: Arc<HipCore>,
}

impl HipDevice {
    pub fn device_info(&self) -> &HipDeviceInfo {
        &self.core.info
    }

    /// Code cache index: the gfx target, or the device if hipRTC detects it.
    fn code_target(&self) -> String {
        let info = &self.core.info;
        if info.gfx_target.is_empty() {
            format!("device{}:{}", info.ordinal, info.name)
        } else {
            info.gfx_target.clone()
        }
    }

    fn compile(&self, src: &[u8]) -> Result<(Arc<Vec<u8>>, CodeOrigin)> {
        let info = &self.core.info;
        let opts = info.capabilities.rtc_options(&info.gfx_target);
        self.core.activate()?;
        compile_hip_src(
            src,
            &opts,
            &self.code_target(),
            self.core.cache_dir.as_deref(),
        )
        .map_err(|e| match e {
            GpuError::ShaderCompile(log) if !info.capabilities.target_supported() => {
                GpuError::ShaderCompile(format!(
                    "{log}\nhint: {} ({}) needs a newer ROCm than {}",
                    info.name, info.gfx_target, info.capabilities.rocm
                ))
            }
            other => other,
        })
    }

    fn copy_host_to_device(&self, buf: HipBuffer, offset: usize, data: &[u8]) -> Result<()> {
        let core = &self.core;
        unsafe {
            core.check(
                hipMemcpyAsync(
                    (buf.ptr as *mut u8).add(offset) as *mut c_void,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    buf.copy_kind(HIP_MEMCPY_H2D),
                    core.stream,
                ),
                "hipMemcpyAsync H→D",
            )?;
        }
        core.synchronize()
    }

    fn copy_device_to_host(&self, buf: HipBuffer, offset: usize, dst: &mut [u8]) -> Result<()> {
        let core = &self.core;
        unsafe {
            core.check(
                hipMemcpyAsync(
                    dst.as_mut_ptr() as *mut c_void,
                    (buf.ptr as *const u8).add(offset) as *const c_void,
                    dst.len(),
                    buf.copy_kind(HIP_MEMCPY_D2H),
                    core.stream,
                ),
                "hipMemcpyAsync D→H",
            )?;
        }
        core.synchronize()
    }

    fn readable(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<(HipBuffer, usize)> {
        let buf = self.core.buffer(buffer)?;
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        self.core.ensure_idle("read_buffer")?;
        let offset = checked_range(offset, len, buf.size, "read")?;
        Ok((buf, offset))
    }
}

impl GpuDevice for HipDevice {
    fn backend(&self) -> BackendPreference {
        BackendPreference::Hip
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn limits(&self) -> DeviceLimits {
        let info = &self.core.info;
        DeviceLimits {
            max_workgroup_size: info.max_block_dim,
            max_workgroup_invocations: info.max_threads_per_block,
            max_dispatch_size: info.max_grid_dim,
            max_storage_buffer_range: info.total_mem as u64,
            max_push_constant_size: (MAX_PARAMS * 4) as u32,
            max_storage_buffers: MAX_PARAMS as u32,
            compute_units: if info.gfx_family.is_rdna() {
                info.cu_count * 2
            } else {
                info.cu_count
            },
            ..DeviceLimits::default()
        }
    }

    fn memory_budget(&self) -> Option<MemoryBudget> {
        self.core.activate().ok()?;
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { hipMemGetInfo(&mut free, &mut total) } != HIP_SUCCESS {
            return None;
        }
        let usage = self.core.allocated.load(Ordering::Relaxed);
        let cached = self.core.block_cache.lock().map_or(0, |c| c.bytes()) as u64;
        Some(MemoryBudget {
            device_local: total as u64,
            budget: (free as u64)
                .saturating_add(usage)
                .saturating_add(cached)
                .min(total as u64),
            usage,
        })
    }

    fn is_lost(&self) -> bool {
        self.core.lost.load(Ordering::Acquire)
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let size =
            usize::try_from(desc.size).map_err(|_| GpuError::OutOfMemory(MemoryUsage::GpuOnly))?;
        self.core.activate()?;
        let mut ptr: *mut c_void = ptr::null_mut();
        let mut host: *mut c_void = ptr::null_mut();
        let mut block = size;
        if size >= alloc::CACHE_MIN_BYTES
            && let Some((cached, cached_block)) = lock(&self.core.block_cache)?.take(size)
        {
            ptr = cached;
            block = cached_block;
        }
        if size != 0 && ptr.is_null() {
            let mut rc = unsafe { hipMalloc(&mut ptr, size) };
            if rc == HIP_ERROR_OUT_OF_MEMORY {
                self.core.release_cached_memory();
                rc = unsafe { hipMalloc(&mut ptr, size) };
            }
            if rc == HIP_ERROR_OUT_OF_MEMORY && self.core.info.integrated {
                unsafe {
                    self.core.check(
                        hipHostMalloc(&mut host, size, HIP_HOST_MALLOC_MAPPED),
                        "hipHostMalloc (APU fallback)",
                    )?;
                    if let Err(e) = self.core.check(
                        hipHostGetDevicePointer(&mut ptr, host, 0),
                        "hipHostGetDevicePointer",
                    ) {
                        hipHostFree(host);
                        return Err(e);
                    }
                }
                log::debug!("hip: {size} B buffer placed in pinned host memory (APU)");
            } else {
                self.core.check(rc, "hipMalloc")?;
            }
        }
        self.core
            .allocated
            .fetch_add(size as u64, Ordering::Relaxed);
        Ok(lock(&self.core.buffers)?.insert(HipBuffer {
            ptr,
            size,
            usage: desc.usage,
            host,
            block,
        }))
    }

    fn buffer_size(&self, buffer: BufferHandle) -> Result<u64> {
        Ok(self.core.buffer(buffer)?.size as u64)
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buf = self.core.buffer(buffer)?;
        self.core.ensure_idle("write_buffer")?;
        let offset = checked_range(offset, data.len() as u64, buf.size, "write")?;
        if data.is_empty() {
            return Ok(());
        }
        self.core.activate()?;
        self.copy_host_to_device(buf, offset, data)
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (buf, offset) = self.readable(buffer, offset, len)?;
        let mut out = vec![0u8; len as usize];
        if len != 0 {
            self.core.activate()?;
            self.copy_device_to_host(buf, offset, &mut out)?;
        }
        Ok(out)
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        let (buf, offset) = self.readable(buffer, offset, dst.len() as u64)?;
        if dst.is_empty() {
            return Ok(());
        }
        self.core.activate()?;
        self.copy_device_to_host(buf, offset, dst)
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        self.core.ensure_idle("copy_buffer")?;
        self.core.activate()?;
        self.core
            .enqueue_copy(src, src_offset, dst, dst_offset, len)?;
        self.core.synchronize()
    }

    fn device_ordinal(&self) -> i32 {
        self.core.info.ordinal
    }

    fn can_peer(&self, peer_ordinal: i32) -> bool {
        let ordinal = self.core.info.ordinal;
        if peer_ordinal == ordinal || peer_ordinal < 0 {
            return false;
        }
        let Ok(mut cache) = self.core.peer_access.lock() else {
            return false;
        };
        if let Some(&usable) = cache.get(&peer_ordinal) {
            return usable;
        }

        let (mut major, mut minor) = (0, 0);
        let peer_family =
            if unsafe { hipDeviceComputeCapability(&mut major, &mut minor, peer_ordinal) }
                == HIP_SUCCESS
            {
                GfxFamily::from_compute_capability(major, minor)
            } else {
                GfxFamily::Unknown
            };
        let policy = env::var(workarounds::P2P_ENV).ok();
        if !workarounds::peer_copy_allowed(
            self.core.info.gfx_family,
            peer_family,
            policy.as_deref(),
        ) {
            cache.insert(peer_ordinal, false);
            return false;
        }

        let mut can_access = 0;
        let usable = unsafe {
            hipDeviceCanAccessPeer(&mut can_access, ordinal, peer_ordinal) == HIP_SUCCESS
                && can_access != 0
                && hipSetDevice(ordinal) == HIP_SUCCESS
                && matches!(
                    hipDeviceEnablePeerAccess(peer_ordinal, 0),
                    HIP_SUCCESS | HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED
                )
        };
        cache.insert(peer_ordinal, usable);
        usable
    }

    fn copy_from_peer(
        &self,
        dst: BufferHandle,
        src_device: &dyn GpuDevice,
        src: BufferHandle,
        bytes: u64,
    ) -> Result<()> {
        let src_device = src_device
            .as_any()
            .downcast_ref::<HipDevice>()
            .ok_or_else(|| GpuError::Unsupported("hip: peer source is not a HIP device".into()))?;
        let dst_buf = self.core.buffer(dst)?;
        let src_buf = src_device.core.buffer(src)?;
        checked_range(0, bytes, dst_buf.size, "peer copy destination")?;
        checked_range(0, bytes, src_buf.size, "peer copy source")?;
        self.core.ensure_idle("copy_from_peer")?;
        src_device.core.ensure_idle("copy_from_peer")?;
        if bytes == 0 {
            return Ok(());
        }
        let bytes = bytes as usize;

        let dst_ordinal = self.core.info.ordinal;
        let src_ordinal = src_device.core.info.ordinal;
        if Arc::ptr_eq(&self.core, &src_device.core) {
            if src == dst {
                return Err(binding_error(
                    "copy_from_peer requires distinct source and destination buffers",
                ));
            }
            self.core.activate()?;
            unsafe {
                self.core.check(
                    hipMemcpyAsync(
                        dst_buf.ptr,
                        src_buf.ptr,
                        bytes,
                        HIP_MEMCPY_DEFAULT,
                        self.core.stream,
                    ),
                    "hipMemcpyAsync D→D",
                )?;
            }
            return self.core.synchronize();
        }

        self.core.synchronize()?;
        src_device.core.synchronize()?;
        if self.can_peer(src_ordinal) {
            self.core.activate()?;
            unsafe {
                self.core.check(
                    hipMemcpyPeer(dst_buf.ptr, dst_ordinal, src_buf.ptr, src_ordinal, bytes),
                    "hipMemcpyPeer",
                )?;
            }
        } else if let Some(staging) = lock(&self.core.peer_staging)?
            .get_or_insert_with(PeerStaging::new)
            .as_ref()
        {
            staging.copy(&self.core, dst_buf, &src_device.core, src_buf, bytes)?;
        } else {
            let mut staging = vec![0u8; bytes.min(peer::CHUNK)];
            for offset in (0..bytes).step_by(peer::CHUNK) {
                let len = (bytes - offset).min(peer::CHUNK);
                let chunk = &mut staging[..len];
                src_device.core.activate()?;
                unsafe {
                    src_device.core.check(
                        hipMemcpy(
                            chunk.as_mut_ptr() as *mut c_void,
                            src_buf.ptr.cast::<u8>().add(offset).cast::<c_void>(),
                            len,
                            src_buf.copy_kind(HIP_MEMCPY_D2H),
                        ),
                        "hipMemcpy peer fallback D→H",
                    )?;
                }
                self.core.activate()?;
                unsafe {
                    self.core.check(
                        hipMemcpy(
                            dst_buf.ptr.cast::<u8>().add(offset).cast::<c_void>(),
                            chunk.as_ptr() as *const c_void,
                            len,
                            dst_buf.copy_kind(HIP_MEMCPY_H2D),
                        ),
                        "hipMemcpy peer fallback H→D",
                    )?;
                }
            }
        }
        Ok(())
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let removed = {
            let (Ok(mut buffers), Ok(mut lifetime)) =
                (self.core.buffers.lock(), self.core.lifetime.lock())
            else {
                return;
            };
            if lifetime.in_flight != 0 {
                if let Some(buf) = buffers.retire(buffer) {
                    lifetime
                        .deferred
                        .push(Deferred::Buffer(buffer.index(), buf));
                }
                None
            } else {
                buffers.remove(buffer)
            }
        };
        if let Some(buf) = removed {
            self.core.free(buf);
        }
    }

    fn create_texture(&self, _: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Unsupported(
            "hip: compute-only backend has no textures".into(),
        ))
    }
    fn upload_texture_data(&self, _: TextureHandle, _: &[u8]) -> Result<()> {
        Err(GpuError::Unsupported(
            "hip: compute-only backend has no textures".into(),
        ))
    }
    fn destroy_texture(&self, _: TextureHandle) {}

    fn create_sampler(&self, _: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Unsupported(
            "hip: compute-only backend has no samplers".into(),
        ))
    }
    fn destroy_sampler(&self, _: SamplerHandle) {}

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let src = match desc.source {
            ShaderSource::Hip(src) => src,
            _ => {
                return Err(GpuError::ShaderCompile(
                    "hip backend: only HIP C++ source is accepted; use ShaderDesc::hip() \
                     or a zsl! shader built with the `hip` feature"
                        .into(),
                ));
            }
        };

        let (mut code, mut origin) = self.compile(src)?;
        self.core.activate()?;
        let mut module: HipModuleRaw = ptr::null_mut();
        let mut rc = unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) };
        if rc != HIP_SUCCESS && origin != CodeOrigin::Compiled && !is_fatal(rc) {
            log::warn!(
                "hip: cached code object rejected ({}); recompiling",
                error_name(rc)
            );
            let info = &self.core.info;
            let opts = info.capabilities.rtc_options(&info.gfx_target);
            evict_code(
                src,
                &opts,
                &self.code_target(),
                self.core.cache_dir.as_deref(),
            );
            (code, origin) = self.compile(src)?;
            debug_assert_eq!(origin, CodeOrigin::Compiled);
            rc = unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) };
        }
        self.core.check(rc, "hipModuleLoadData")?;
        Ok(lock(&self.core.shaders)?.insert(Arc::new(HipModule { raw: module })))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let (Ok(mut shaders), Ok(mut lifetime)) =
            (self.core.shaders.lock(), self.core.lifetime.lock())
        else {
            return;
        };
        if lifetime.in_flight != 0 {
            if let Some(module) = shaders.retire(shader) {
                lifetime
                    .deferred
                    .push(Deferred::Shader(shader.index(), module));
            }
        } else {
            let removed = shaders.remove(shader);
            drop(shaders);
            drop(lifetime);
            drop(removed);
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let entry = CString::new(desc.entry)
            .map_err(|_| GpuError::PipelineCreation("entry point contains null byte".into()))?;
        let module = lock(&self.core.shaders)?
            .get(desc.shader)
            .cloned()
            .ok_or_else(|| GpuError::PipelineCreation("stale shader handle".into()))?;

        let block = if desc.block == [0, 0, 0] {
            [256, 1, 1]
        } else {
            desc.block
        };
        let info = &self.core.info;
        let threads = block.iter().map(|&b| u64::from(b)).product::<u64>();
        if block.contains(&0)
            || threads > u64::from(info.max_threads_per_block)
            || block
                .iter()
                .zip(info.max_block_dim)
                .any(|(&b, max)| b > max)
        {
            return Err(GpuError::PipelineCreation(format!(
                "hip: block {block:?} is outside the device limits ({} threads, {:?} per axis)",
                info.max_threads_per_block, info.max_block_dim
            )));
        }

        self.core.activate()?;
        let mut func: HipFunction = ptr::null_mut();
        let rc = unsafe { hipModuleGetFunction(&mut func, module.raw, entry.as_ptr()) };
        if rc == HIP_ERROR_NOT_FOUND {
            return Err(GpuError::PipelineCreation(format!(
                "hip: kernel `{}` not found in module (declare it extern \"C\" __global__)",
                desc.entry
            )));
        }
        self.core.check(rc, "hipModuleGetFunction")?;

        let mut kernel_max = 0;
        let rc = unsafe {
            hipFuncGetAttribute(
                &mut kernel_max,
                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                func,
            )
        };
        if rc == HIP_SUCCESS && kernel_max > 0 && threads > kernel_max as u64 {
            return Err(GpuError::PipelineCreation(format!(
                "hip: block {block:?} has {threads} threads but kernel `{}` allows at most \
                 {kernel_max} (check its __launch_bounds__ / @workgroup_size)",
                desc.entry
            )));
        }

        Ok(lock(&self.core.pipelines)?.insert(HipPipeline {
            function: func,
            block,
            _module: module,
        }))
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let (Ok(mut pipelines), Ok(mut lifetime)) =
            (self.core.pipelines.lock(), self.core.lifetime.lock())
        else {
            return;
        };
        if lifetime.in_flight != 0 {
            if let Some(pipe) = pipelines.retire(pipeline) {
                lifetime
                    .deferred
                    .push(Deferred::Pipeline(pipeline.index(), pipe));
            }
        } else {
            let removed = pipelines.remove(pipeline);
            drop(pipelines);
            drop(lifetime);
            drop(removed);
        }
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.dispatch_batch(&[DispatchOp {
            pipeline,
            bindings,
            grid,
        }])
    }

    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        self.submit_batch(0, ops)?.wait(Duration::MAX)
    }

    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.core
            .submit(cycle_id, ops, HipCore::dispatch_touches, HipCore::launch)
    }

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        let touches = |core: &HipCore, op: &ComputeOp<'_>, out: &mut Vec<(usize, bool)>| match op {
            ComputeOp::CopyBuffer(copy) => {
                for (handle, written) in [(copy.src, false), (copy.dst, true)] {
                    if let Ok(buffer) = core.buffer(handle) {
                        out.push((buffer.ptr as usize, written));
                    }
                }
            }
            ComputeOp::Dispatch(dispatch) => core.dispatch_touches(dispatch, out),
        };
        self.core
            .submit(cycle_id, ops, touches, |core, op| match op {
                ComputeOp::CopyBuffer(copy) => core.enqueue_copy(
                    copy.src,
                    copy.src_offset,
                    copy.dst,
                    copy.dst_offset,
                    copy.len,
                ),
                ComputeOp::Dispatch(dispatch) => core.launch(dispatch),
            })
    }

    fn submit_write_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<Submission> {
        let buf = self.core.buffer(buffer)?;
        let offset = checked_range(offset, data.len() as u64, buf.size, "write")?;
        if data.is_empty() {
            return Ok(Box::new(CompletedSubmission::new(cycle_id)));
        }
        self.core.activate()?;
        let lease = self.core.pinned.lease(data.len())?;
        let staging = lease.ptr();
        let touched = [(buf.ptr as usize, true)];
        let sub = self
            .core
            .record(cycle_id, Lane::Upload, Some(lease), &touched, |core| {
                for (i, chunk) in data.chunks(STAGE_CHUNK).enumerate() {
                    let at = i * STAGE_CHUNK;
                    unsafe {
                        // SAFETY: the lease holds `data.len()` bytes, and this
                        let src = staging.add(at);
                        ptr::copy_nonoverlapping(chunk.as_ptr(), src, chunk.len());
                        core.check(
                            hipMemcpyAsync(
                                (buf.ptr as *mut u8).add(offset + at) as *mut c_void,
                                src as *const c_void,
                                chunk.len(),
                                buf.copy_kind(HIP_MEMCPY_H2D),
                                core.upload,
                            ),
                            "hipMemcpyAsync H→D",
                        )?;
                    }
                }
                Ok(())
            })?;
        Ok(Box::new(sub))
    }

    fn submit_read_buffer(
        &self,
        cycle_id: u64,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        let buf = self.core.buffer(buffer)?;
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        let offset = checked_range(offset, len, buf.size, "read")?;
        let len = len as usize;
        if len == 0 {
            return Ok(Box::new(CompletedReadback::new(cycle_id, Vec::new())));
        }
        self.core.activate()?;
        let lease = self.core.pinned.lease(len)?;
        let dst = lease.ptr();
        let touched = [(buf.ptr as usize, false)];
        let sub = self.core.record(
            cycle_id,
            Lane::Download,
            Some(lease),
            &touched,
            |core| unsafe {
                core.check(
                    hipMemcpyAsync(
                        dst as *mut c_void,
                        (buf.ptr as *const u8).add(offset) as *const c_void,
                        len,
                        buf.copy_kind(HIP_MEMCPY_D2H),
                        core.download,
                    ),
                    "hipMemcpyAsync D→H",
                )
            },
        )?;
        Ok(Box::new(HipReadback { sub, len }))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::manual_div_ceil, clippy::unnecessary_cast)]

    use super::*;
    use std::env::temp_dir;
    use std::fs;
    use std::process;
    use std::slice::from_raw_parts;
    use std::sync::Arc;
    use std::thread::{sleep, spawn};
    use std::time::Instant;

    use zengpu_hal::{BufferUsage, GpuInstance, MemoryUsage};

    const VEC_ADD_SRC: &str = r#"
extern "C" __global__
void vec_add(const float* __restrict__ a,
             const float* __restrict__ b,
             float* __restrict__ c,
             unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

    fn try_instance() -> Option<HipInstance> {
        match HipInstance::new() {
            Ok(inst) => Some(inst),
            Err(e) => {
                println!("hip: skipped: {e}");
                None
            }
        }
    }

    #[test]
    fn enumerate_adapters() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        println!("hip: {} adapter(s) found", adapters.len());
        for a in &adapters {
            let i = a.info();
            println!("  [{}] {}", i.device, i.name);
        }
        assert!(
            !adapters.is_empty(),
            "ROCm initialised but found no adapters"
        );
    }

    #[test]
    fn device_info_gcn_target() {
        let Some(inst) = try_instance() else { return };
        for info in inst.device_infos() {
            println!(
                "  [{ordinal}] {name} → gfx target: {gfx:?}",
                ordinal = info.ordinal,
                name = info.name,
                gfx = info.gfx_target
            );
            assert!(
                !info.gfx_target.is_empty(),
                "no gfx target derived for {}",
                info.name
            );
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let Some(inst) = try_instance() else { return };
        let adapter = match inst.request_adapter(AdapterRequest::default()) {
            Some(a) => a,
            None => {
                println!("hip: no adapter");
                return;
            }
        };
        let device = adapter.open(DeviceRequest::default()).unwrap();

        let data: Vec<u8> = (0u8..128).collect();
        let buf = device
            .create_buffer(BufferDesc {
                size: data.len() as u64,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device.write_buffer(buf, 0, &data).unwrap();
        let back = device.read_buffer(buf, 0, data.len() as u64).unwrap();
        assert_eq!(data, back, "buffer roundtrip mismatch");
        device.destroy_buffer(buf);
    }

    #[test]
    fn peer_buffer_copy() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            println!(
                "hip: peer copy test needs ≥ 2 GPUs; found {}",
                adapters.len()
            );
            return;
        }

        let dev0 = adapters[0].open(DeviceRequest::default()).unwrap();
        let dev1 = adapters[1].open(DeviceRequest::default()).unwrap();
        let values = [1.25f32, -2.5, 3.75, 1024.5, -0.125, 42.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let desc = BufferDesc {
            size: data.len() as u64,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        };
        let dst = dev0.create_buffer(desc).unwrap();
        let src = dev1.create_buffer(desc).unwrap();
        dev1.write_buffer(src, 0, &data).unwrap();

        let direct = dev0.can_peer(dev1.device_ordinal());
        println!(
            "hip: GPU {} ← GPU {} peer copy: {}",
            dev0.device_ordinal(),
            dev1.device_ordinal(),
            if direct {
                "P2P fast path"
            } else {
                "host-staging fallback"
            }
        );
        dev0.copy_from_peer(dst, dev1.as_ref(), src, data.len() as u64)
            .unwrap();
        let copied = dev0.read_buffer(dst, 0, data.len() as u64).unwrap();
        assert_eq!(copied, data, "peer buffer copy mismatch");
        println!("hip: peer copy correctness: OK");

        dev0.destroy_buffer(dst);
        dev1.destroy_buffer(src);
    }

    #[test]
    fn vec_add_single_gpu() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        run_vec_add(
            adapters[0].open(DeviceRequest::default()).unwrap().as_ref(),
            "GPU 0",
        );
    }

    #[test]
    fn vec_add_all_gpus() {
        let Some(inst) = try_instance() else { return };
        for adapter in inst.enumerate_adapters() {
            let label = adapter.info().name.clone();
            let device = adapter.open(DeviceRequest::default()).unwrap();
            run_vec_add(device.as_ref(), &label);
        }
    }

    #[test]
    fn multi_gpu_parallel_dispatch() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            println!(
                "hip: multi-gpu test needs ≥ 2 GPUs; found {}",
                adapters.len()
            );
            return;
        }

        let devices: Vec<Arc<Box<dyn GpuDevice>>> = adapters
            .iter()
            .map(|a| Arc::new(a.open(DeviceRequest::default()).unwrap()))
            .collect();

        let handles: Vec<_> = devices
            .into_iter()
            .enumerate()
            .map(|(idx, dev)| {
                let label = format!("GPU {idx}");
                spawn(move || run_vec_add(dev.as_ref().as_ref(), &label))
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    fn run_vec_add(device: &dyn GpuDevice, label: &str) {
        const N: usize = 1024 * 256;

        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

        let byte_len = (N * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_a = device
            .create_buffer(BufferDesc {
                size: byte_len,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_b = device
            .create_buffer(BufferDesc {
                size: byte_len,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_c = device
            .create_buffer(BufferDesc {
                size: byte_len,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();

        device.write_buffer(buf_a, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(buf_b, 0, bytemuck_cast(&b)).unwrap();

        let shader = device.create_shader(ShaderDesc::hip(VEC_ADD_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "vec_add",
                block: [256, 1, 1],
            })
            .unwrap();

        let blocks = ((N as u32) + 255) / 256;
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[buf_a.index(), buf_b.index(), buf_c.index()],
                    textures: &[],
                    images: &[],
                    scalars: &[zengpu_hal::Scalar::U32(N as u32)],
                },
                [blocks, 1, 1],
            )
            .unwrap();

        let raw = device.read_buffer(buf_c, 0, byte_len).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);

        let expected = N as f32;
        let mismatches: Vec<_> = c
            .iter()
            .enumerate()
            .filter(|&(_, &v)| (v - expected).abs() > 1e-3)
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "{label}: vec_add result mismatch at indices {:?} (expected {expected})",
            mismatches.iter().map(|(i, _)| i).collect::<Vec<_>>()
        );
        println!(
            "{label}: vec_add {N} elements OK (c[0]={}, c[N-1]={})",
            c[0],
            c[N - 1]
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_c);
    }

    fn bytemuck_cast(v: &[f32]) -> &[u8] {
        unsafe { from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    fn bytemuck_cast_slice(v: &[u8]) -> &[f32] {
        assert_eq!(v.len() % 4, 0);
        unsafe { from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
    }

    /// 32x32 tiled SGEMM. Fills a full WGP on gfx1200 (1024 threads/WGP).
    const SGEMM_SRC: &str = r#"
#define TILE 32
extern "C" __global__
void sgemm(const float* __restrict__ A,
           const float* __restrict__ B,
           float* __restrict__ C,
           unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;
    float acc = 0.0f;
    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        int bk = t * TILE + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

    /// 64x64 tile with 4x4 register blocking: 16 outputs per thread. LDS is padded +1 on K to avoid
    /// bank conflicts. Expected about 4x faster than `SGEMM_SRC`.
    const SGEMM_OPT_SRC: &str = r#"
#define TILE 64
#define TK   16
#define DIM  16

extern "C" __global__ __launch_bounds__(DIM * DIM)
void sgemm_opt(const float* __restrict__ A,
               const float* __restrict__ B,
               float* __restrict__ C,
               unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TK + 1];   /* [m][k] padded to avoid bank conflicts */
    __shared__ float Bs[TILE][TK + 1];   /* [n][k] same layout                   */

    int tx  = (int)threadIdx.x;          /* 0..DIM-1 */
    int ty  = (int)threadIdx.y;          /* 0..DIM-1 */
    int bx  = (int)blockIdx.x;
    int by  = (int)blockIdx.y;
    int tid = ty * DIM + tx;             /* 0..255 linearised thread index        */

    float acc[4][4] = {};

    for (int k0 = 0; k0 < (int)K; k0 += TK) {
        /* ── Load As[TILE][TK]: 64×16 = 1024 elements, 4 per thread ──
           idx maps to (mi, ki) via mi=idx/TK, ki=idx%TK.
           16 consecutive tids share the same mi → ki runs 0..15 → coalesced. */
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int mi = idx / TK;
            int ki = idx % TK;
            int gm = by * TILE + mi;
            int gk = k0 + ki;
            As[mi][ki] = (gm < (int)M && gk < (int)K) ? A[gm * (int)K + gk] : 0.0f;
        }
        /* ── Load Bs[TILE][TK]: same size, ni=idx%TILE, ki=idx/TILE.
           64 consecutive tids share ki → ni runs 0..63 → full cache line. */
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int ni = idx % TILE;
            int ki = idx / TILE;
            int gn = bx * TILE + ni;
            int gk = k0 + ki;
            Bs[ni][ki] = (gk < (int)K && gn < (int)N) ? B[gk * (int)N + gn] : 0.0f;
        }
        __syncthreads();

        /* ── 4×4 register-blocked dot product ── */
        #pragma unroll
        for (int ki = 0; ki < TK; ki++) {
            float a0 = As[ty          ][ki];
            float a1 = As[ty + DIM    ][ki];
            float a2 = As[ty + 2*DIM  ][ki];
            float a3 = As[ty + 3*DIM  ][ki];
            float b0 = Bs[tx          ][ki];
            float b1 = Bs[tx + DIM    ][ki];
            float b2 = Bs[tx + 2*DIM  ][ki];
            float b3 = Bs[tx + 3*DIM  ][ki];
            acc[0][0] += a0*b0; acc[0][1] += a0*b1; acc[0][2] += a0*b2; acc[0][3] += a0*b3;
            acc[1][0] += a1*b0; acc[1][1] += a1*b1; acc[1][2] += a1*b2; acc[1][3] += a1*b3;
            acc[2][0] += a2*b0; acc[2][1] += a2*b1; acc[2][2] += a2*b2; acc[2][3] += a2*b3;
            acc[3][0] += a3*b0; acc[3][1] += a3*b1; acc[3][2] += a3*b2; acc[3][3] += a3*b3;
        }
        __syncthreads();
    }

    /* ── Write 4×4 output block ── */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = by * TILE + ty + i * DIM;
        if (gm < (int)M) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int gn = bx * TILE + tx + j * DIM;
                if (gn < (int)N) C[gm * (int)N + gn] = acc[i][j];
            }
        }
    }
}
"#;

    /// Tree reduction of N f32 values to one per block. Uses static LDS for 512 threads.
    const REDUCE_SRC: &str = r#"
#define BLOCK_SIZE 512
extern "C" __global__
void reduce_sum(const float* __restrict__ in, float* __restrict__ out, unsigned int n) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    float v = 0.0f;
    if (i              < n) v += in[i];
    if (i + BLOCK_SIZE < n) v += in[i + BLOCK_SIZE];
    sdata[tid] = v;
    __syncthreads();
    #pragma unroll
    for (unsigned int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
"#;

    const SCALE_SRC: &str = r#"
extern "C" __global__
void mem_scale(const float* __restrict__ in, float* __restrict__ out,
               float scale, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}
"#;

    fn elapsed_ms(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }

    #[test]
    fn heavy_sgemm() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let info = adapters[0].info();

        const M: usize = 2048;
        const N: usize = 2048;
        const K: usize = 2048;

        let a: Vec<f32> = (0..M * K).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..K * N).map(|i| (i % 13) as f32 * 0.01).collect();

        let bytes_mn = (M * N * 4) as u64;
        let bytes_mk = (M * K * 4) as u64;
        let bytes_kn = (K * N * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let ba = device
            .create_buffer(BufferDesc {
                size: bytes_mk,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bb = device
            .create_buffer(BufferDesc {
                size: bytes_kn,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bc = device
            .create_buffer(BufferDesc {
                size: bytes_mn,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();

        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

        let shader = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm",
                block: [32, 32, 1],
            })
            .unwrap();

        let grid = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            images: &[],
            scalars: &[
                zengpu_hal::Scalar::U32(M as u32),
                zengpu_hal::Scalar::U32(N as u32),
                zengpu_hal::Scalar::U32(K as u32),
            ],
        };
        device.dispatch(pipeline, bindings, grid).unwrap();

        const REPS: u32 = 5;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, grid).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let flop = 2.0 * M as f64 * N as f64 * K as f64;
        let gflops = flop / (ms * 1e6);

        println!(
            "[{}] SGEMM {M}×{K}×{N}: {ms:.2} ms → {gflops:.1} GFLOP/s",
            info.name
        );

        let raw = device.read_buffer(bc, 0, bytes_mn).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);
        let expected_c00: f32 = (0..K).map(|k| a[k] * b[k * N]).sum();
        let err = (c[0] - expected_c00).abs() / expected_c00.abs().max(1e-6);
        assert!(
            err < 1e-3,
            "SGEMM C[0][0] error {err:.2e}: got {}, expected {expected_c00}",
            c[0]
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
    }

    #[test]
    fn heavy_reduction() {
        let Some(inst) = try_instance() else { return };
        let device = adapters_or_skip(&inst);

        const N: usize = 128 * 1024 * 1024;
        const BLOCK: u32 = 512;

        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes = (N * 4) as u64;
        let num_blocks = ((N as u32) + BLOCK * 2 - 1) / (BLOCK * 2);
        let partial_bytes = (num_blocks as usize * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in = device
            .create_buffer(BufferDesc {
                size: src_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_out = device
            .create_buffer(BufferDesc {
                size: partial_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();

        device
            .write_buffer(buf_in, 0, bytemuck_cast(&data))
            .unwrap();

        let shader = device.create_shader(ShaderDesc::hip(REDUCE_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "reduce_sum",
                block: [BLOCK, 1, 1],
            })
            .unwrap();

        let bindings = Bindings {
            buffers: &[buf_in.index(), buf_out.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(N as u32)],
        };

        device
            .dispatch(pipeline, bindings, [num_blocks, 1, 1])
            .unwrap();

        const REPS: u32 = 3;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device
                .dispatch(pipeline, bindings, [num_blocks, 1, 1])
                .unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials: &[f32] = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "reduction sum wrong: got {sum}, expected {expected}, err={err:.2e}"
        );

        let bytes_moved = (N * 4 + num_blocks as usize * 4) as f64;
        let gb_s = bytes_moved / (ms * 1e6);
        println!("[GPU 0] reduce_sum {N} floats (512 MB): {ms:.2} ms → {gb_s:.1} GB/s (sum={sum})");

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    #[test]
    fn heavy_bandwidth() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        for adapter in &adapters {
            let device = adapter.open(DeviceRequest::default()).unwrap();
            let name = adapter.info().name.clone();

            const N: usize = 256 * 1024 * 1024;
            const BLOCK: u32 = 256;

            let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
            let bytes = (N * 4) as u64;
            let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

            let buf_in = device
                .create_buffer(BufferDesc {
                    size: bytes,
                    usage: storage,
                    memory: MemoryUsage::GpuOnly,
                })
                .unwrap();
            let buf_out = device
                .create_buffer(BufferDesc {
                    size: bytes,
                    usage: storage,
                    memory: MemoryUsage::GpuOnly,
                })
                .unwrap();
            device
                .write_buffer(buf_in, 0, bytemuck_cast(&data))
                .unwrap();

            let shader = device.create_shader(ShaderDesc::hip(SCALE_SRC)).unwrap();
            let pipeline = device
                .create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry: "mem_scale",
                    block: [BLOCK, 1, 1],
                })
                .unwrap();

            let grid = [(N as u32 + BLOCK - 1) / BLOCK, 1, 1];
            let bindings = Bindings {
                buffers: &[buf_in.index(), buf_out.index()],
                textures: &[],
                images: &[],
                scalars: &[
                    zengpu_hal::Scalar::F32(2.0f32),
                    zengpu_hal::Scalar::U32(N as u32),
                ],
            };

            device.dispatch(pipeline, bindings, grid).unwrap();

            const REPS: u32 = 5;
            let t0 = Instant::now();
            for _ in 0..REPS {
                device.dispatch(pipeline, bindings, grid).unwrap();
            }
            let ms = elapsed_ms(t0) / REPS as f64;

            let gb_s = (2.0 * bytes as f64) / (ms * 1e6);
            println!("[{name}] mem_scale {N} floats (1 GB r+w): {ms:.2} ms → {gb_s:.1} GB/s");

            let raw = device.read_buffer(buf_out, 0, 64).unwrap();
            let out: &[f32] = bytemuck_cast_slice(&raw);
            assert!((out[0] - 0.0f32).abs() < 1e-5, "out[0] wrong: {}", out[0]);
            assert!((out[1] - 2.0f32).abs() < 1e-5, "out[1] wrong: {}", out[1]);

            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
            device.destroy_buffer(buf_in);
            device.destroy_buffer(buf_out);
        }
    }

    #[test]
    fn heavy_multi_gpu_sgemm() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            println!(
                "hip: multi-gpu SGEMM needs ≥ 2 GPUs; found {}",
                adapters.len()
            );
        }

        let devices: Vec<Arc<Box<dyn GpuDevice>>> = adapters
            .iter()
            .map(|a| Arc::new(a.open(DeviceRequest::default()).unwrap()))
            .collect();
        let names: Vec<String> = adapters.iter().map(|a| a.info().name.clone()).collect();

        let handles: Vec<_> = devices
            .into_iter()
            .zip(names)
            .enumerate()
            .map(|(idx, (dev, name))| {
                spawn(move || {
                    const M: usize = 2048;
                    const N: usize = 2048;
                    const K: usize = 2048;
                    let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
                    let b: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 * 0.1).collect();
                    let bytes = |r: usize, c: usize| (r * c * 4) as u64;
                    let storage = BufferUsage::STORAGE | BufferUsage::READBACK;
                    let ba = dev
                        .create_buffer(BufferDesc {
                            size: bytes(M, K),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    let bb = dev
                        .create_buffer(BufferDesc {
                            size: bytes(K, N),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    let bc = dev
                        .create_buffer(BufferDesc {
                            size: bytes(M, N),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    dev.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
                    dev.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
                    let shader = dev.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
                    let pipeline = dev
                        .create_compute_pipeline(ComputePipelineDesc {
                            shader,
                            entry: "sgemm",
                            block: [32, 32, 1],
                        })
                        .unwrap();
                    let grid = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
                    let bindings = Bindings {
                        buffers: &[ba.index(), bb.index(), bc.index()],
                        textures: &[],
                        images: &[],
                        scalars: &[
                            zengpu_hal::Scalar::U32(M as u32),
                            zengpu_hal::Scalar::U32(N as u32),
                            zengpu_hal::Scalar::U32(K as u32),
                        ],
                    };
                    dev.dispatch(pipeline, bindings, grid).unwrap();
                    const REPS: u32 = 5;
                    let t0 = Instant::now();
                    for _ in 0..REPS {
                        dev.dispatch(pipeline, bindings, grid).unwrap();
                    }
                    let ms = elapsed_ms(t0) / REPS as f64;
                    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);
                    println!(
                        "[GPU {idx} - {name}] SGEMM {M}×{K}×{N}: {ms:.2} ms → {gflops:.1} GFLOP/s"
                    );
                    dev.destroy_pipeline(pipeline);
                    dev.destroy_shader(shader);
                    dev.destroy_buffer(ba);
                    dev.destroy_buffer(bb);
                    dev.destroy_buffer(bc);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    fn run_sgemm_opt(device: &dyn GpuDevice, m: usize, n: usize, k: usize) -> f64 {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.001).collect();
        let st = BufferUsage::STORAGE | BufferUsage::READBACK;
        let ba = device
            .create_buffer(BufferDesc {
                size: (m * k * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bb = device
            .create_buffer(BufferDesc {
                size: (k * n * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bc = device
            .create_buffer(BufferDesc {
                size: (m * n * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
        let shader = device
            .create_shader(ShaderDesc::hip(SGEMM_OPT_SRC))
            .unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm_opt",
                block: [16, 16, 1],
            })
            .unwrap();
        let grid = [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1];
        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            images: &[],
            scalars: &[
                zengpu_hal::Scalar::U32(m as u32),
                zengpu_hal::Scalar::U32(n as u32),
                zengpu_hal::Scalar::U32(k as u32),
            ],
        };
        device.dispatch(pipeline, bindings, grid).unwrap();
        device.dispatch(pipeline, bindings, grid).unwrap();
        device.read_buffer(bc, 0, 4).unwrap();
        const REPS: u32 = 3;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, grid).unwrap();
        }
        device.read_buffer(bc, 0, 4).unwrap();
        let ms = elapsed_ms(t0) / REPS as f64;
        let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e6);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
        gflops
    }

    #[test]
    fn heavy_sgemm_opt_sweep() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let adapter = &adapters[0];
        let device = adapter.open(DeviceRequest::default()).unwrap();
        let name = adapter.info().name.clone();

        for &sz in &[2048usize, 4096, 8192] {
            let mem_mb = 3 * sz * sz * 4 / (1024 * 1024);
            let info = inst.device_infos();
            let vram = info[0].total_mem / (1024 * 1024);
            if mem_mb as usize > vram * 3 / 4 {
                println!("[{name}] skip SGEMM {sz}³: need {mem_mb} MB, VRAM={vram} MB");
                continue;
            }

            let gflops_naive = if sz <= 4096 {
                Some(run_sgemm_naive(device.as_ref(), sz))
            } else {
                None
            };
            let gflops_opt = run_sgemm_opt(device.as_ref(), sz, sz, sz);

            match gflops_naive {
                Some(n) => println!(
                    "[{name}] SGEMM {sz}³  naive={n:.0}  opt={gflops_opt:.0} GFLOP/s  ({:.1}×)",
                    gflops_opt / n
                ),
                None => println!("[{name}] SGEMM {sz}³  opt={gflops_opt:.0} GFLOP/s"),
            }

            assert!(
                gflops_opt.is_finite() && gflops_opt > 0.0,
                "opt SGEMM {sz}³ produced an invalid throughput measurement: {gflops_opt}"
            );
            if let Some(n) = gflops_naive {
                assert!(
                    n.is_finite() && n > 0.0,
                    "naive SGEMM {sz}³ produced an invalid throughput measurement: {n}"
                );
            }
        }
    }

    fn run_sgemm_naive(device: &dyn GpuDevice, sz: usize) -> f64 {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 13) as f32 * 0.01).collect();
        let st = BufferUsage::STORAGE | BufferUsage::READBACK;
        let ba = device
            .create_buffer(BufferDesc {
                size: (sz * sz * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bb = device
            .create_buffer(BufferDesc {
                size: (sz * sz * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bc = device
            .create_buffer(BufferDesc {
                size: (sz * sz * 4) as u64,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
        let shader = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm",
                block: [32, 32, 1],
            })
            .unwrap();
        let grid = [((sz as u32) + 31) / 32, ((sz as u32) + 31) / 32, 1];
        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            images: &[],
            scalars: &[
                zengpu_hal::Scalar::U32(sz as u32),
                zengpu_hal::Scalar::U32(sz as u32),
                zengpu_hal::Scalar::U32(sz as u32),
            ],
        };
        device.dispatch(pipeline, bindings, grid).unwrap();
        device.read_buffer(bc, 0, 4).unwrap();
        const REPS: u32 = 3;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, grid).unwrap();
        }
        device.read_buffer(bc, 0, 4).unwrap();
        let ms = elapsed_ms(t0) / REPS as f64;
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
        2.0 * sz as f64 * sz as f64 * sz as f64 / (ms * 1e6)
    }

    #[test]
    fn heavy_multi_gpu_sgemm_opt_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.is_empty() {
            return;
        }

        let handles: Vec<_> = adapters
            .iter()
            .enumerate()
            .map(|(idx, a)| {
                let dev: Arc<dyn GpuDevice> = Arc::from(a.open(DeviceRequest::default()).unwrap());
                let name = a.info().name.clone();
                spawn(move || {
                    let gflops = run_sgemm_opt(dev.as_ref(), 4096, 4096, 4096);
                    println!("[GPU {idx} - {name}] opt SGEMM 4096³: {gflops:.0} GFLOP/s");
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn zsl_hip_vec_scale() {
        let Some(inst) = try_instance() else { return };
        let device = adapters_or_skip(&inst);

        const SCALE: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push Consts { n: u32 }
            @workgroup_size(256)
            kernel scale(
                id: global_id,
                a: device buffer<f32>,
                b: device mut buffer<f32>,
                c: Consts,
            ) {
                let i = id.x
                if i < c.n {
                    b[i] = a[i] * 2.0
                }
            }
        );
        let scale_src = SCALE.hip;

        const N: usize = 1 << 20;
        let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let bytes = (N * 4) as u64;
        let st = BufferUsage::STORAGE | BufferUsage::READBACK;

        let src = device
            .create_buffer(BufferDesc {
                size: bytes,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let dst = device
            .create_buffer(BufferDesc {
                size: bytes,
                usage: st,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device.write_buffer(src, 0, bytemuck_cast(&data)).unwrap();

        let shader = device.create_shader(ShaderDesc::hip(scale_src)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "zsl_kernel",
                block: [256, 1, 1],
            })
            .unwrap();

        let bindings = Bindings {
            buffers: &[src.index(), dst.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(N as u32)],
        };
        let grid = [(N as u32 + 255) / 256, 1, 1];
        device.dispatch(pipeline, bindings, grid).unwrap();

        let raw = device.read_buffer(dst, 0, 32).unwrap();
        let out = bytemuck_cast_slice(&raw);
        assert!((out[0] - 0.0f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 2.0f32).abs() < 1e-5, "out[1]={}", out[1]);
        assert!((out[2] - 4.0f32).abs() < 1e-5, "out[2]={}", out[2]);
        println!(
            "zsl! → hipRTC: vec_scale OK  out[0..3]={} {} {}",
            out[0], out[1], out[2]
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src);
        device.destroy_buffer(dst);
    }

    fn adapters_or_skip(inst: &HipInstance) -> Box<dyn GpuDevice> {
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty(), "no HIP adapters");
        adapters
            .into_iter()
            .next()
            .unwrap()
            .open(DeviceRequest::default())
            .unwrap()
    }

    #[test]
    fn heavy_sgemm_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let adapter = &adapters[0];
        let device = adapter.open(DeviceRequest::default()).unwrap();
        let info = adapter.info();

        const M: usize = 4096;
        const N: usize = 4096;
        const K: usize = 4096;

        let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.001).collect();
        let b: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.001).collect();

        let bytes_mn = (M * N * 4) as u64;
        let bytes_mk = (M * K * 4) as u64;
        let bytes_kn = (K * N * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let ba = device
            .create_buffer(BufferDesc {
                size: bytes_mk,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bb = device
            .create_buffer(BufferDesc {
                size: bytes_kn,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let bc = device
            .create_buffer(BufferDesc {
                size: bytes_mn,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();

        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

        let shader = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm",
                block: [32, 32, 1],
            })
            .unwrap();

        let grid = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            images: &[],
            scalars: &[
                zengpu_hal::Scalar::U32(M as u32),
                zengpu_hal::Scalar::U32(N as u32),
                zengpu_hal::Scalar::U32(K as u32),
            ],
        };

        device.dispatch(pipeline, bindings, grid).unwrap();
        device.dispatch(pipeline, bindings, grid).unwrap();

        const REPS: u32 = 3;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, grid).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;
        let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);

        println!(
            "[{}] SGEMM {M}×{K}×{N}: {ms:.1} ms → {gflops:.1} GFLOP/s",
            info.name
        );

        let raw = device.read_buffer(bc, 0, 64).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);
        let expected: f32 = (0..K).map(|k| a[k] * b[k * N]).sum();
        let err = (c[0] - expected).abs() / expected.abs().max(1e-6);
        assert!(
            err < 1e-2,
            "SGEMM 4096 C[0][0] err {err:.2e}: got {} expected {expected}",
            c[0]
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
    }

    /// 256 M floats (1 GB), larger than L2.
    #[test]
    fn heavy_reduction_256m() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let name = adapters[0].info().name.clone();

        const N: usize = 256 * 1024 * 1024;
        const BLOCK: u32 = 512;

        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes = (N * 4) as u64;
        let num_blocks = ((N as u32) + BLOCK * 2 - 1) / (BLOCK * 2);
        let partial_bytes = (num_blocks as usize * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in = device
            .create_buffer(BufferDesc {
                size: src_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_out = device
            .create_buffer(BufferDesc {
                size: partial_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device
            .write_buffer(buf_in, 0, bytemuck_cast(&data))
            .unwrap();

        let shader = device.create_shader(ShaderDesc::hip(REDUCE_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "reduce_sum",
                block: [BLOCK, 1, 1],
            })
            .unwrap();

        let bindings = Bindings {
            buffers: &[buf_in.index(), buf_out.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(N as u32)],
        };

        device
            .dispatch(pipeline, bindings, [num_blocks, 1, 1])
            .unwrap();

        const REPS: u32 = 3;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device
                .dispatch(pipeline, bindings, [num_blocks, 1, 1])
                .unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "reduction 256M: got {sum}, expected {expected}, err={err:.2e}"
        );

        let bytes_moved = (N * 4 + num_blocks as usize * 4) as f64;
        let gb_s = bytes_moved / (ms * 1e6);
        println!("[{name}] reduce_sum 256M floats (1 GB): {ms:.2} ms → {gb_s:.1} GB/s  sum={sum}");

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    /// RDNA 4 wave reduction via `__builtin_amdgcn_ds_swizzle`, no LDS. Falls back to LDS on older
    /// ROCm and non-RDNA.
    const WAVE_REDUCE_SRC: &str = r#"
/* Wave-level sum reduction using DPP butterfly swizzle (wave32 assumed).
   One block sums 1024 elements; outer loop handles the full array. */
#define BLOCK_SIZE 256
extern "C" __global__
void wave_reduce_sum(const float* __restrict__ in, float* __restrict__ out,
                     unsigned int n) {
    __shared__ float sdata[BLOCK_SIZE / 32]; // one slot per wave
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int wid  = tid >> 5; // wave index within block

    float v = 0.0f;
    for (unsigned int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += gridDim.x * BLOCK_SIZE)
        v += in[i];

    v += __shfl_xor(v, 16);
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 1);

    if (lane == 0) sdata[wid] = v;
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        unsigned int nw = BLOCK_SIZE / 32;
        for (unsigned int w = 0; w < nw; ++w) s += sdata[w];
        out[blockIdx.x] = s;
    }
}
"#;

    /// Wave reduction via `__shfl_xor` (DPP on RDNA). Skipped below ROCm 4.0.
    #[test]
    fn heavy_wave_reduce_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let hip_info = inst.device_infos();
        if !hip_info[0].capabilities.wave_reduction {
            println!("skip heavy_wave_reduce_4096: ROCm < 4.0 (no wave reduction builtins)");
            return;
        }

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let name = adapters[0].info().name.clone();

        const N: usize = 256 * 1024 * 1024;
        const BLOCK: u32 = 256;
        const NUM_BLOCKS: u32 = 2048;

        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes = (N * 4) as u64;
        let partial_bytes = (NUM_BLOCKS as usize * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in = device
            .create_buffer(BufferDesc {
                size: src_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let buf_out = device
            .create_buffer(BufferDesc {
                size: partial_bytes,
                usage: storage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        device
            .write_buffer(buf_in, 0, bytemuck_cast(&data))
            .unwrap();

        let shader = device
            .create_shader(ShaderDesc::hip(WAVE_REDUCE_SRC))
            .unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "wave_reduce_sum",
                block: [BLOCK, 1, 1],
            })
            .unwrap();

        let bindings = Bindings {
            buffers: &[buf_in.index(), buf_out.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(N as u32)],
        };

        device
            .dispatch(pipeline, bindings, [NUM_BLOCKS, 1, 1])
            .unwrap();

        const REPS: u32 = 5;
        let t0 = Instant::now();
        for _ in 0..REPS {
            device
                .dispatch(pipeline, bindings, [NUM_BLOCKS, 1, 1])
                .unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "wave reduce: got {sum}, expected {expected}, err={err:.2e}"
        );

        let bytes_moved = N as f64 * 4.0;
        let gb_s = bytes_moved / (ms * 1e6);
        println!(
            "[{name}] wave_reduce 256M floats: {ms:.2} ms → {gb_s:.1} GB/s  (wave{} DPP)",
            hip_info[0].capabilities.wave_size
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    #[test]
    fn heavy_multi_gpu_sgemm_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();

        if adapters.is_empty() {
            return;
        }
        if adapters.len() < 2 {
            println!("hip: multi-gpu 4096 needs ≥ 2 GPUs; running single-GPU only");
        }

        let devices: Vec<Arc<Box<dyn GpuDevice>>> = adapters
            .iter()
            .map(|a| Arc::new(a.open(DeviceRequest::default()).unwrap()))
            .collect();
        let names: Vec<String> = adapters.iter().map(|a| a.info().name.clone()).collect();

        let handles: Vec<_> = devices
            .into_iter()
            .zip(names)
            .enumerate()
            .map(|(idx, (dev, name))| {
                spawn(move || {
                    const M: usize = 4096;
                    const N: usize = 4096;
                    const K: usize = 4096;

                    let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.001).collect();
                    let b: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.001).collect();

                    let bytes = |r: usize, c: usize| (r * c * 4) as u64;
                    let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

                    let ba = dev
                        .create_buffer(BufferDesc {
                            size: bytes(M, K),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    let bb = dev
                        .create_buffer(BufferDesc {
                            size: bytes(K, N),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    let bc = dev
                        .create_buffer(BufferDesc {
                            size: bytes(M, N),
                            usage: storage,
                            memory: MemoryUsage::GpuOnly,
                        })
                        .unwrap();
                    dev.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
                    dev.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

                    let shader = dev.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
                    let pipeline = dev
                        .create_compute_pipeline(ComputePipelineDesc {
                            shader,
                            entry: "sgemm",
                            block: [32, 32, 1],
                        })
                        .unwrap();

                    let grid = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
                    let bindings = Bindings {
                        buffers: &[ba.index(), bb.index(), bc.index()],
                        textures: &[],
                        images: &[],
                        scalars: &[
                            zengpu_hal::Scalar::U32(M as u32),
                            zengpu_hal::Scalar::U32(N as u32),
                            zengpu_hal::Scalar::U32(K as u32),
                        ],
                    };

                    dev.dispatch(pipeline, bindings, grid).unwrap();
                    dev.dispatch(pipeline, bindings, grid).unwrap();

                    const REPS: u32 = 3;
                    let t0 = Instant::now();
                    for _ in 0..REPS {
                        dev.dispatch(pipeline, bindings, grid).unwrap();
                    }
                    let ms = elapsed_ms(t0) / REPS as f64;
                    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);
                    println!("[GPU {idx} - {name}] SGEMM 4096³: {ms:.1} ms → {gflops:.1} GFLOP/s");

                    dev.destroy_pipeline(pipeline);
                    dev.destroy_shader(shader);
                    dev.destroy_buffer(ba);
                    dev.destroy_buffer(bb);
                    dev.destroy_buffer(bc);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn capability_report() {
        let Some(inst) = try_instance() else { return };
        let infos = inst.device_infos();
        assert!(!infos.is_empty(), "no HIP adapters found");

        for info in infos {
            let report = info.capabilities.report(&info.name, &info.gfx_target);
            println!("─────────────────────────────────────────────────");
            println!("{report}");
            println!("─────────────────────────────────────────────────");

            let caps = &info.capabilities;

            assert!(
                caps.wave_size == 32 || caps.wave_size == 64,
                "unexpected wave size {} on {}",
                caps.wave_size,
                info.gfx_target,
            );
            if caps.mfma {
                assert!(
                    info.gfx_family.full_fp64(),
                    "mfma set but not a CDNA device: {}",
                    info.gfx_target,
                );
            }
            if !caps.hiprtc {
                assert!(
                    !caps.hiprtc_bitcode,
                    "bitcode without hipRTC on {}",
                    info.gfx_target
                );
            }
            let expect_wgp = info.gfx_family.is_rdna();
            assert_eq!(
                caps.wgp_default, expect_wgp,
                "WGP mode mismatch for {} ({:?})",
                info.gfx_target, info.gfx_family,
            );
        }
    }

    fn first_device() -> Option<Box<dyn GpuDevice>> {
        let inst = try_instance()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        Some(adapter.open(DeviceRequest::default()).unwrap())
    }

    fn buffer(device: &dyn GpuDevice, size: u64, usage: BufferUsage) -> BufferHandle {
        device
            .create_buffer(BufferDesc {
                size,
                usage,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap()
    }

    const FILL_SRC: &str = r#"
extern "C" __global__ void fill(unsigned int* out, unsigned int value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value + i;
}
"#;

    #[test]
    fn device_query_reports_real_properties() {
        let Some(inst) = try_instance() else { return };
        for info in inst.device_infos() {
            assert!(info.gfx_target.starts_with("gfx"), "{info:?}");
            assert_ne!(info.gfx_family, GfxFamily::Unknown, "{}", info.gfx_target);
            assert!(info.cu_count > 0, "CU count from attributes");
            assert!(info.clock_mhz > 0, "clock from attributes");
            assert!(info.wave_size == 32 || info.wave_size == 64);
            assert!(info.max_threads_per_block >= 256);
            if info.gfx_family.is_rdna() && env::var_os(version::WAVE64_ENV).is_none() {
                assert_eq!(info.wave_size, 32, "HIP runs RDNA in wave32");
            }
        }
    }

    #[test]
    fn out_of_bounds_and_usage_are_rejected() {
        let Some(device) = first_device() else { return };
        let rb = buffer(
            device.as_ref(),
            16,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        let storage_only = buffer(device.as_ref(), 16, BufferUsage::STORAGE);

        assert!(
            device.write_buffer(rb, 8, &[0u8; 16]).is_err(),
            "write past end"
        );
        assert!(
            device.write_buffer(rb, u64::MAX, &[1]).is_err(),
            "offset overflow"
        );
        assert!(device.read_buffer(rb, 0, 17).is_err(), "read past end");
        assert!(
            device.read_buffer(storage_only, 0, 4).is_err(),
            "needs READBACK"
        );
        assert!(
            device.write_buffer(rb, 16, &[]).is_ok(),
            "empty write at end"
        );
        assert!(device.read_buffer(rb, 16, 0).unwrap().is_empty());

        device.destroy_buffer(rb);
        assert!(device.read_buffer(rb, 0, 4).is_err(), "stale handle");
        device.destroy_buffer(rb);
        device.destroy_buffer(storage_only);
    }

    #[test]
    fn dispatch_validates_bindings_and_launch_shape() {
        let Some(device) = first_device() else { return };
        let shader = device.create_shader(ShaderDesc::hip(FILL_SRC)).unwrap();
        assert!(
            device
                .create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry: "fill",
                    block: [2048, 1, 1],
                })
                .is_err(),
            "block over the device limit"
        );
        assert!(
            device
                .create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry: "missing",
                    block: [64, 1, 1],
                })
                .is_err(),
            "unknown entry point"
        );
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "fill",
                block: [64, 1, 1],
            })
            .unwrap();
        device.destroy_shader(shader);

        let out = buffer(
            device.as_ref(),
            256 * 4,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        let bad = Bindings {
            buffers: &[9999],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(0), zengpu_hal::Scalar::U32(256)],
        };
        assert!(
            device.dispatch(pipeline, bad, [4, 1, 1]).is_err(),
            "unknown slot"
        );

        let good = Bindings {
            buffers: &[out.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(7), zengpu_hal::Scalar::U32(256)],
        };
        device.dispatch(pipeline, good, [0, 1, 1]).unwrap();
        device.dispatch(pipeline, good, [4, 1, 1]).unwrap();
        let raw = device.read_buffer(out, 0, 256 * 4).unwrap();
        for (i, chunk) in raw.as_chunks::<4>().0.iter().enumerate() {
            let v = u32::from_ne_bytes(*chunk);
            assert_eq!(v, 7 + i as u32);
        }
        assert!(!device.is_lost());

        device.destroy_pipeline(pipeline);
        device.destroy_buffer(out);
    }

    #[test]
    fn async_submission_defers_destruction_and_orders_copies() {
        let Some(device) = first_device() else { return };
        const N: u32 = 1 << 16;
        let shader = device.create_shader(ShaderDesc::hip(FILL_SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "fill",
                block: [256, 1, 1],
            })
            .unwrap();
        let src = buffer(
            device.as_ref(),
            u64::from(N) * 4,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        );
        let dst = buffer(
            device.as_ref(),
            u64::from(N) * 4,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
        );
        let bindings = Bindings {
            buffers: &[src.index()],
            textures: &[],
            images: &[],
            scalars: &[zengpu_hal::Scalar::U32(100), zengpu_hal::Scalar::U32(N)],
        };
        let ops = [
            ComputeOp::Dispatch(DispatchOp {
                pipeline,
                bindings,
                grid: [N / 256, 1, 1],
            }),
            ComputeOp::CopyBuffer(zengpu_hal::BufferCopyOp {
                src,
                src_offset: 0,
                dst,
                dst_offset: 0,
                len: u64::from(N) * 4,
            }),
        ];
        let submission = device.submit_compute_ops(42, &ops).unwrap();
        assert_eq!(submission.cycle_id(), 42);
        assert!(device.read_buffer(dst, 0, 4).is_err());
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(src);
        submission.wait(Duration::from_secs(30)).unwrap();
        assert_eq!(submission.poll().unwrap(), SubmissionStatus::Complete);

        let raw = device.read_buffer(dst, 0, u64::from(N) * 4).unwrap();
        assert_eq!(u32::from_ne_bytes(raw[0..4].try_into().unwrap()), 100);
        let last = (N as usize - 1) * 4;
        assert_eq!(
            u32::from_ne_bytes(raw[last..last + 4].try_into().unwrap()),
            100 + N - 1
        );
        device.destroy_buffer(dst);
    }

    #[test]
    fn copy_buffer_validates_and_copies() {
        let Some(device) = first_device() else { return };
        let src = buffer(
            device.as_ref(),
            64,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        );
        let dst = buffer(
            device.as_ref(),
            64,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
        );
        let data: Vec<u8> = (0..64).collect();
        device.write_buffer(src, 0, &data).unwrap();
        assert!(
            device.copy_buffer(src, 0, src, 32, 16).is_err(),
            "same buffer"
        );
        assert!(
            device.copy_buffer(dst, 0, src, 0, 16).is_err(),
            "missing usage"
        );
        assert!(device.copy_buffer(src, 60, dst, 0, 8).is_err(), "past end");
        device.copy_buffer(src, 16, dst, 0, 32).unwrap();
        let mut out = [0u8; 32];
        device.read_buffer_into(dst, 0, &mut out).unwrap();
        assert_eq!(&out[..], &data[16..48]);
        device.destroy_buffer(src);
        device.destroy_buffer(dst);
    }

    #[test]
    fn limits_and_budget_are_reported() {
        let Some(device) = first_device() else { return };
        let limits = device.limits();
        assert!(limits.max_workgroup_invocations >= 256);
        assert!(limits.max_dispatch_size[0] > 0);
        let before = device.memory_budget().expect("hipMemGetInfo");
        assert!(before.device_local > 0 && before.budget <= before.device_local);
        let b = buffer(device.as_ref(), 1 << 20, BufferUsage::STORAGE);
        let after = device.memory_budget().unwrap();
        assert_eq!(after.usage, before.usage + (1 << 20));
        device.destroy_buffer(b);
        assert_eq!(device.memory_budget().unwrap().usage, before.usage);
    }

    #[test]
    fn ieee_semantics_survive_compilation() {
        const SRC: &str = r#"
extern "C" __global__ void classify(const float* in, unsigned int* out) {
    unsigned int i = threadIdx.x;
    out[i] = (isnan(in[i]) ? 1u : 0u) | (isinf(in[i]) ? 2u : 0u);
}
"#;
        let Some(device) = first_device() else { return };
        let input = [0.0f32, f32::NAN, f32::INFINITY, -1.5];
        let inb = buffer(device.as_ref(), 16, BufferUsage::STORAGE);
        let outb = buffer(
            device.as_ref(),
            16,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        device.write_buffer(inb, 0, bytemuck_cast(&input)).unwrap();
        let shader = device.create_shader(ShaderDesc::hip(SRC)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "classify",
                block: [4, 1, 1],
            })
            .unwrap();
        device
            .dispatch(
                pipeline,
                Bindings {
                    buffers: &[inb.index(), outb.index()],
                    textures: &[],
                    images: &[],
                    scalars: &[],
                },
                [1, 1, 1],
            )
            .unwrap();
        let raw = device.read_buffer(outb, 0, 16).unwrap();
        let flags: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c))
            .collect();
        assert_eq!(flags, [0, 1, 2, 0]);
        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(inb);
        device.destroy_buffer(outb);
    }

    /// Shipped kernels must compile for older parts (RDNA 1/2, GCN 5, CDNA) without a device:
    /// compile only, no load.
    #[test]
    fn kernels_compile_for_legacy_targets() {
        if try_instance().is_none() {
            return;
        }
        let rocm = RocmVersion::runtime().min(RocmVersion::COMPILE_TIME);
        for gfx in [
            "gfx1010", "gfx1012", "gfx1030", "gfx1100", "gfx906", "gfx90a",
        ] {
            let family = GfxFamily::from_gfx(gfx);
            let caps = HipCapabilities::for_family(rocm, rocm, family);
            let opts = caps.rtc_options(gfx);
            for src in [VEC_ADD_SRC, SGEMM_OPT_SRC, REDUCE_SRC, WAVE_REDUCE_SRC] {
                let code = compile_hip_src(src.as_bytes(), &opts, gfx, None)
                    .unwrap_or_else(|e| panic!("{gfx}: {e}"))
                    .0;
                assert!(!code.is_empty(), "{gfx}: empty code object");
            }
        }
    }

    #[test]
    fn adapter_pick_prefers_kind_then_compute_units() {
        let dev = |ordinal, cu_count, integrated| {
            let family = GfxFamily::Rdna2;
            HipDeviceInfo {
                ordinal,
                name: String::new(),
                gfx_target: "gfx1030".into(),
                gfx_family: family,
                total_mem: 0,
                cu_count,
                clock_mhz: 0,
                wave_size: 32,
                integrated,
                max_threads_per_block: 1024,
                max_block_dim: [1024; 3],
                max_grid_dim: [1 << 30; 3],
                capabilities: HipCapabilities::for_family(
                    RocmVersion::COMPILE_TIME,
                    RocmVersion::COMPILE_TIME,
                    family,
                ),
            }
        };
        let pick = |devs: &[HipDeviceInfo], low| pick_device(devs, low).map(|d| d.ordinal);
        let devs = [dev(0, 2, true), dev(1, 28, false), dev(2, 80, false)];
        assert_eq!(pick(&devs, false), Some(2));
        assert_eq!(pick(&devs, true), Some(0));
        let twins = [dev(0, 32, false), dev(1, 32, false)];
        assert_eq!(pick(&twins, false), Some(0));
        assert_eq!(pick(&twins, true), Some(0));
        assert_eq!(pick(&[], false), None);
    }

    #[test]
    fn gcn_arch_found_in_any_props_layout() {
        let mut blob = vec![0u8; PROP_BLOB_SIZE];
        blob[..18].copy_from_slice(b"AMD Radeon RX 5700");
        blob[1003..1003 + 14].copy_from_slice(b"gfx1010:xnack-");
        let other_abi = if ROCM_VERSION_MAJOR >= 6 {
            PropsAbi::R0000
        } else {
            PropsAbi::R0600
        };
        assert_eq!(gcn_arch_from_prop(&blob, Some(other_abi)), "gfx1010:xnack-");
        assert_eq!(gcn_arch_from_prop(&blob, None), "gfx1010:xnack-");

        let mut junk = vec![0u8; 64];
        junk[..12].copy_from_slice(b"xxgfx1030 ok");
        assert_eq!(gcn_arch_from_prop(&junk, None), "");
        assert!(gfx_string_at(b"gfx\0", 0).is_none());
        assert!(gfx_string_at(b"gfx12-generic\0", 0).is_some());
    }

    #[test]
    fn block_beyond_kernel_launch_bounds_is_rejected() {
        const SRC: &str = r#"
extern "C" __global__ __launch_bounds__(64) void small(unsigned int* out) {
    out[threadIdx.x] = threadIdx.x;
}
"#;
        let Some(device) = first_device() else { return };
        let shader = device.create_shader(ShaderDesc::hip(SRC)).unwrap();
        let err = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "small",
                block: [256, 1, 1],
            })
            .expect_err("256 threads exceed __launch_bounds__(64)");
        assert!(format!("{err:?}").contains("at most"), "{err:?}");
        let ok = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "small",
                block: [64, 1, 1],
            })
            .unwrap();
        device.destroy_pipeline(ok);
        device.destroy_shader(shader);
    }

    #[test]
    fn disk_code_cache_survives_process_cache_and_corruption() {
        let Some(inst) = try_instance() else { return };
        let adapter = inst.request_adapter(AdapterRequest::default()).unwrap();
        let root = temp_dir().join(format!("zengpu-hip-e2e-{}", process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let request = DeviceRequest {
            pipeline_cache_path: Some(root.join("pipelines.bin")),
            ..DeviceRequest::default()
        };
        let src = format!("{FILL_SRC}\n// {:?}\n", Instant::now());

        let build = |label: &str| {
            let device = adapter.open(request.clone()).unwrap();
            let t = Instant::now();
            let shader = device.create_shader(ShaderDesc::hip(&src)).unwrap();
            println!("hip: {label}: {:?}", t.elapsed());
            let pipeline = device
                .create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry: "fill",
                    block: [64, 1, 1],
                })
                .unwrap();
            let out = buffer(
                device.as_ref(),
                256,
                BufferUsage::STORAGE | BufferUsage::READBACK,
            );
            let scalars = [zengpu_hal::Scalar::U32(3), zengpu_hal::Scalar::U32(64)];
            let bindings = Bindings {
                buffers: &[out.index()],
                textures: &[],
                images: &[],
                scalars: &scalars,
            };
            device.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
            let raw = device.read_buffer(out, 0, 8).unwrap();
            assert_eq!(raw, [3, 0, 0, 0, 4, 0, 0, 0], "{label}");
        };

        build("cold compile");
        let dir = root.join("pipelines.bin.hip");
        let entries: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(entries.len(), 1, "one cached code object");

        code_cache().lock().unwrap().clear();
        build("warm from disk");

        let path = entries[0].path();
        let mut data = fs::read(&path).unwrap();
        let len = data.len();
        data[len - 2048..].fill(0xAB);
        fs::write(&path, &data).unwrap();
        code_cache().lock().unwrap().clear();
        build("after corruption");

        let _ = fs::remove_dir_all(&root);
    }

    /// Destroying a buffer never calls `hipFree` inline (it waits for every kernel).
    #[test]
    fn destroying_large_buffer_does_not_wait_for_busy_gpu() {
        const SPIN: &str = r#"
extern "C" __global__ void spin(unsigned int ticks) {
    unsigned long long start = wall_clock64();
    while (wall_clock64() - start < ticks) {}
}
"#;
        let Some(inst) = try_instance() else { return };
        let adapter = inst.request_adapter(AdapterRequest::default()).unwrap();
        let busy = adapter.open(DeviceRequest::default()).unwrap();
        let other = adapter.open(DeviceRequest::default()).unwrap();

        let shader = busy.create_shader(ShaderDesc::hip(SPIN)).unwrap();
        let pipeline = busy
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "spin",
                block: [1, 1, 1],
            })
            .unwrap();
        let sizes: [u64; 3] = [64 << 20, 1 << 20, 600 << 20];
        let bufs: Vec<_> = sizes
            .iter()
            .map(|&size| buffer(other.as_ref(), size, BufferUsage::STORAGE))
            .collect();

        let ticks = zengpu_hal::Scalar::U32(50_000_000);
        let submission = busy
            .submit(
                1,
                pipeline,
                Bindings {
                    buffers: &[],
                    textures: &[],
                    images: &[],
                    scalars: &[ticks],
                },
                [1, 1, 1],
            )
            .unwrap();
        sleep(Duration::from_millis(50));
        let t = Instant::now();
        for b in bufs {
            other.destroy_buffer(b);
        }
        let took = t.elapsed();
        let pending = submission.poll().unwrap() == SubmissionStatus::Pending;
        submission.wait(Duration::from_secs(30)).unwrap();
        println!(
            "hip: destroy 64 MiB + 1 MiB + 600 MiB while GPU busy: {took:?} (kernel still running: {pending})"
        );
        assert!(
            pending,
            "spin kernel ended too early for the test to mean anything"
        );
        assert!(took < Duration::from_millis(100), "destroy waited {took:?}");

        busy.destroy_pipeline(pipeline);
        busy.destroy_shader(shader);
    }

    /// Host-staged copies use 4 MiB chunks. A size that is not a whole multiple leaves a
    /// short tail chunk.
    #[test]
    fn large_peer_copy_round_trips() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            return;
        }
        let dst_dev = adapters[0].open(DeviceRequest::default()).unwrap();
        let src_dev = adapters[1].open(DeviceRequest::default()).unwrap();
        let bytes = (64 << 20) + 4100;
        let data: Vec<u8> = (0..bytes).map(|i| (i * 7 % 251) as u8).collect();
        let src = buffer(src_dev.as_ref(), bytes as u64, BufferUsage::STORAGE);
        let dst = buffer(
            dst_dev.as_ref(),
            bytes as u64,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        src_dev.write_buffer(src, 0, &data).unwrap();
        dst_dev
            .copy_from_peer(dst, src_dev.as_ref(), src, bytes as u64)
            .unwrap();
        let t = Instant::now();
        dst_dev
            .copy_from_peer(dst, src_dev.as_ref(), src, bytes as u64)
            .unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "hip: peer copy 64 MiB  {:.1} GB/s",
            bytes as f64 / secs / 1e9
        );
        assert_eq!(dst_dev.read_buffer(dst, 0, bytes as u64).unwrap(), data);
        src_dev.destroy_buffer(src);
        dst_dev.destroy_buffer(dst);
    }
}
