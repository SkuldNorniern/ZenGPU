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
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferCopyOp, BufferDesc,
    BufferHandle, BufferUsage, CompletedReadback, CompletedSubmission, ComputeCaps, ComputeOp,
    ComputePipelineDesc, DType, DeviceLimits, DeviceRequest, DeviceType, DispatchOp, GpuAdapter,
    GpuDevice, GpuError, GpuInstance, GpuReadback, GpuSubmission, HalCapabilities, MemoryBudget,
    MemoryUsage, MmaShape, PipelineHandle, PipelineStats, PowerPreference, ReadSubmission, Result,
    SamplerDesc, SamplerHandle, Scalar, ShaderDesc, ShaderHandle, ShaderSource, SlotMap,
    Submission, SubmissionStatus, TextureDesc, TextureHandle, UsageError, marker,
};

mod hip_layout {
    include!(concat!(env!("OUT_DIR"), "/hip_layout.rs"));
}
use hip_layout::*;

pub mod version;
use version::{GfxFamily, HipCapabilities, RocmVersion};

pub mod workarounds;

mod code_object;
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
/// `HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`: static LDS bytes per block.
const HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: i32 = 1;
/// `HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`: private (spill) bytes per lane.
const HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: i32 = 3;
/// `HIP_FUNC_ATTRIBUTE_NUM_REGS`: vector registers per lane.
const HIP_FUNC_ATTRIBUTE_NUM_REGS: i32 = 4;

/// `None` if the runtime refuses the query.
fn func_attribute(func: HipFunction, attr: i32) -> Option<u32> {
    let mut value = 0i32;
    let rc = unsafe { hipFuncGetAttribute(&mut value, attr, func) };
    (rc == HIP_SUCCESS && value >= 0).then_some(value as u32)
}

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
    /// Workgroup-shared memory per block, in bytes (`hipDeviceAttributeMaxSharedMemoryPerBlock`).
    pub shared_mem_per_block: u32,
    pub capabilities: HipCapabilities,
}

/// WMMA shapes emitted: 16x16x16 f16 and i8, gfx12 wave32. gfx11 is gated on `__GFX12__` and is not
/// yet lowered or run.
fn hip_mma_shapes(family: GfxFamily, wave_size: u32) -> Vec<MmaShape> {
    if family != GfxFamily::Rdna4 || wave_size != 32 {
        return Vec::new();
    }
    vec![
        MmaShape {
            m: 16,
            n: 16,
            k: 16,
            a: DType::F16,
            b: DType::F16,
            acc: DType::F32,
            subgroup: 32,
        },
        MmaShape {
            m: 16,
            n: 16,
            k: 16,
            a: DType::I8,
            b: DType::I8,
            acc: DType::I32,
            subgroup: 32,
        },
    ]
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
            shared_mem_per_block: positive_attribute(HIP_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, ordinal)
                .unwrap_or(0),
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
    sgprs: HashMap<String, u32>,
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
    stats: PipelineStats,
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

    /// Blocks until queued work on `touched` is done, leaving unrelated work pending. Writes wait
    /// for every lane's last access; reads wait only for the last write.
    fn settle(&self, touched: &[(usize, bool)]) -> Result<()> {
        self.activate()?;
        let events: Vec<Arc<OwnedEvent>> = {
            let lifetime = lock(&self.lifetime)?;
            let mut events = Vec::new();
            for &(ptr, written) in touched {
                let Some(hazard) = lifetime.hazards.get(&ptr) else {
                    continue;
                };
                if written {
                    events.extend(hazard.last_access.iter().flatten().cloned());
                }
                events.extend(hazard.last_write.as_ref().map(|(_, e)| Arc::clone(e)));
            }
            events
        };
        for event in events {
            unsafe { self.check(hipEventSynchronize(event.0), "hipEventSynchronize")? };
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
        self.pinned.trim(0);
        alloc::wait_for_frees(Duration::from_secs(5));
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

    fn compute_caps(&self) -> ComputeCaps {
        let info = &self.core.info;
        ComputeCaps {
            subgroup_sizes: (info.wave_size, info.wave_size),
            default_subgroup: info.wave_size,
            shared_bytes: info.shared_mem_per_block,
            max_workgroup: info.max_threads_per_block,
            mma: hip_mma_shapes(info.gfx_family, info.wave_size),
        }
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
        self.submit_write_buffer(0, buffer, offset, data)?
            .wait(Duration::MAX)
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.submit_read_buffer(0, buffer, offset, len)?
            .into_bytes()
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.submit_read_buffer(0, buffer, offset, dst.len() as u64)?
            .read_into(dst)
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        self.submit_compute_ops(
            0,
            &[ComputeOp::CopyBuffer(BufferCopyOp {
                src,
                src_offset,
                dst,
                dst_offset,
                len,
            })],
        )?
        .wait(Duration::MAX)
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
        self.core.settle(&[(dst_buf.ptr as usize, true)])?;
        src_device.core.settle(&[(src_buf.ptr as usize, false)])?;
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
        let sgprs = code_object::sgpr_counts(&code);
        Ok(lock(&self.core.shaders)?.insert(Arc::new(HipModule { raw: module, sgprs })))
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

        let kernel_max =
            func_attribute(func, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap_or(0);
        if kernel_max > 0 && threads > u64::from(kernel_max) {
            return Err(GpuError::PipelineCreation(format!(
                "hip: block {block:?} has {threads} threads but kernel `{}` allows at most \
                 {kernel_max} (check its __launch_bounds__ / @workgroup_size)",
                desc.entry
            )));
        }

        let stats = PipelineStats {
            vgprs: func_attribute(func, HIP_FUNC_ATTRIBUTE_NUM_REGS).unwrap_or(0),
            sgprs: module.sgprs.get(desc.entry).copied().unwrap_or(0),
            scratch_bytes: func_attribute(func, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(0),
            lds_bytes: func_attribute(func, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES).unwrap_or(0),
            max_threads: kernel_max,
            waves_per_simd: None,
        };

        Ok(lock(&self.core.pipelines)?.insert(HipPipeline {
            function: func,
            block,
            stats,
            _module: module,
        }))
    }

    fn pipeline_stats(&self, pipeline: PipelineHandle) -> Option<PipelineStats> {
        let pipelines = lock(&self.core.pipelines).ok()?;
        pipelines.get(pipeline).map(|pipe| pipe.stats)
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
    use std::slice::{from_raw_parts, from_ref};
    use std::sync::Arc;
    use std::thread::spawn;
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
    fn pipeline_stats_order_tiny_heavy_and_spilling_kernels() {
        const KERNELS: &str = r#"
extern "C" __global__ __launch_bounds__(256) void tiny(float* x) {
    x[threadIdx.x] = 1.0f;
}
extern "C" __global__ __launch_bounds__(256) void heavy(const float* __restrict__ a, float* __restrict__ out) {
    float acc[64];
    for (int i = 0; i < 64; ++i) acc[i] = a[threadIdx.x * 64 + i];
    for (int r = 0; r < 4; ++r)
        for (int i = 0; i < 64; ++i) acc[i] = acc[i] * acc[(i + 7) & 63] + 0.5f;
    float s = 0;
    for (int i = 0; i < 64; ++i) s += acc[i];
    out[threadIdx.x] = s;
}
extern "C" __global__ __launch_bounds__(256) void spill(float* out) {
    float acc[512];
    for (int i = 0; i < 512; ++i) acc[i] = 0.0f;
    for (int i = 0; i < 512; ++i) acc[(i * (int)threadIdx.x + 3) & 511] += (float)i;
    out[threadIdx.x] = acc[(threadIdx.x * 7) & 511];
}
"#;
        let Some(inst) = try_instance() else { return };
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let device = adapter.open(DeviceRequest::default()).unwrap();
        let shader = device.create_shader(ShaderDesc::hip(KERNELS)).unwrap();
        let stats = |entry: &str| {
            let pipeline = device
                .create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry,
                    block: [256, 1, 1],
                })
                .unwrap();
            device.pipeline_stats(pipeline).unwrap()
        };
        let (tiny, heavy, spill) = (stats("tiny"), stats("heavy"), stats("spill"));
        assert!(tiny.sgprs > 0 && tiny.vgprs > 0, "{tiny:?}");
        assert!(heavy.vgprs > tiny.vgprs && heavy.sgprs > 0, "{heavy:?}");
        assert_eq!(tiny.scratch_bytes, 0);
        assert!(spill.scratch_bytes > 0, "{spill:?}");
        assert_eq!(heavy.max_threads, 256);
        println!("hip stats: tiny {tiny:?} heavy {heavy:?} spill {spill:?}");
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

    #[test]
    fn heavy_sgemm() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let device = adapters[0].open(DeviceRequest::default()).unwrap();

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

        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials: &[f32] = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "reduction sum wrong: got {sum}, expected {expected}, err={err:.2e}"
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
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

        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "wave reduce: got {sum}, expected {expected}, err={err:.2e}"
        );

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
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
            if info.capabilities.mfma {
                assert!(
                    info.gfx_family.full_fp64(),
                    "mfma set but not a CDNA device: {}",
                    info.gfx_target
                );
            }
            assert_eq!(
                info.capabilities.wgp_default,
                info.gfx_family.is_rdna(),
                "WGP mode mismatch for {}",
                info.gfx_target
            );
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
                shared_mem_per_block: 65536,
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
            let shader = device.create_shader(ShaderDesc::hip(&src)).unwrap();
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
        assert_eq!(dst_dev.read_buffer(dst, 0, bytes as u64).unwrap(), data);
        src_dev.destroy_buffer(src);
        dst_dev.destroy_buffer(dst);
    }

    fn open_device() -> Option<Box<dyn GpuDevice>> {
        let inst = try_instance()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    fn rw_buffer(device: &dyn GpuDevice, size: u64) -> BufferHandle {
        device
            .create_buffer(BufferDesc {
                size,
                usage: BufferUsage::STORAGE
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap()
    }

    fn kernel(device: &dyn GpuDevice, src: &str, entry: &str) -> (ShaderHandle, PipelineHandle) {
        let shader = device.create_shader(ShaderDesc::hip(src)).unwrap();
        let pipeline = device
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [64, 1, 1],
            })
            .unwrap();
        (shader, pipeline)
    }

    fn dispatch(
        device: &dyn GpuDevice,
        pipeline: PipelineHandle,
        buffers: &[BufferHandle],
        scalars: &[Scalar],
        n: u32,
    ) -> Submission {
        let slots: Vec<u32> = buffers.iter().map(|b| b.index()).collect();
        let op = DispatchOp {
            pipeline,
            bindings: Bindings {
                buffers: &slots,
                scalars,
                textures: &[],
                images: &[],
            },
            grid: [n.div_ceil(64), 1, 1],
        };
        device.submit_batch(0, from_ref(&op)).unwrap()
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|&w| f32::from_le_bytes(w)).collect()
    }

    fn float_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(f32::to_le_bytes).collect()
    }

    const COPY_SRC: &str = r#"
extern "C" __global__ void copy_add(const float* in, float* out, float v, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + v;
}
extern "C" __global__ void slow_copy(const float* in, float* out, unsigned int n, unsigned int cycles) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = clock64();
    volatile long long now = start;
    while (now - start < (long long)cycles) now = clock64();
    if (i < n) out[i] = in[i];
}
"#;

    #[test]
    fn async_write_dispatch_read_stay_ordered() {
        let Some(device) = open_device() else { return };
        let device = &*device;
        const N: usize = 1 << 19;
        let size = (N * 4) as u64;
        let (x, y) = (rw_buffer(device, size), rw_buffer(device, size));
        let (shader, copy) = kernel(device, COPY_SRC, "copy_add");
        let (slow_shader, slow) = kernel(device, COPY_SRC, "slow_copy");

        let first = float_bytes((0..N).map(|i| i as f32));
        let write = device.submit_write_buffer(1, x, 0, &first).unwrap();
        let add = dispatch(
            device,
            copy,
            &[x, y],
            &[Scalar::F32(2.0), Scalar::U32(N as u32)],
            N as u32,
        );
        let read = device.submit_read_buffer(2, y, 0, size).unwrap();
        let want: Vec<f32> = (0..N).map(|i| i as f32 + 2.0).collect();
        assert_eq!(floats(&read.into_bytes().unwrap()), want);

        let spin = dispatch(
            device,
            slow,
            &[x, y],
            &[Scalar::U32(N as u32), Scalar::U32(50_000_000)],
            N as u32,
        );
        let second = float_bytes((0..N).map(|i| -(i as f32)));
        let rewrite = device.submit_write_buffer(3, x, 0, &second).unwrap();
        let copied = device.submit_read_buffer(4, y, 0, size).unwrap();
        assert_eq!(floats(&copied.into_bytes().unwrap()), floats(&first));
        let now = device.submit_read_buffer(5, x, 0, size).unwrap();
        assert_eq!(now.into_bytes().unwrap(), second);
        for sub in [write, add, spin, rewrite] {
            sub.wait(Duration::from_secs(10)).unwrap();
        }
        for p in [copy, slow] {
            device.destroy_pipeline(p);
        }
        device.destroy_shader(shader);
        device.destroy_shader(slow_shader);
        device.destroy_buffer(x);
        device.destroy_buffer(y);
    }

    #[test]
    fn writes_to_busy_and_idle_buffers_keep_their_data() {
        let Some(device) = open_device() else { return };
        let device = &*device;
        const N: usize = 1 << 16;
        let size = (N * 4) as u64;
        let (busy, idle, scratch) = (
            rw_buffer(device, size),
            rw_buffer(device, size),
            rw_buffer(device, size),
        );
        let (shader, slow) = kernel(device, COPY_SRC, "slow_copy");
        let spin = dispatch(
            device,
            slow,
            &[busy, scratch],
            &[Scalar::U32(64), Scalar::U32(4_000_000_000)],
            64,
        );
        let idle_data = float_bytes((0..N).map(|i| i as f32 * 0.5));
        let busy_data = float_bytes((0..N).map(|i| -(i as f32)));
        device.write_buffer(idle, 0, &idle_data).unwrap();
        device.write_buffer(busy, 0, &busy_data).unwrap();
        spin.wait(Duration::from_secs(30)).unwrap();
        assert_eq!(device.read_buffer(idle, 0, size).unwrap(), idle_data);
        assert_eq!(device.read_buffer(busy, 0, size).unwrap(), busy_data);
        device.destroy_pipeline(slow);
        device.destroy_shader(shader);
        device.destroy_buffer(busy);
        device.destroy_buffer(idle);
        device.destroy_buffer(scratch);
    }

    const TOUCH_SRC: &str = r#"
extern "C" __global__ void touch(unsigned int* out, unsigned int v) {
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}
"#;

    /// Prints host-side overheads (compile, dispatch, batching, transfers). Asserts data only.
    #[test]
    #[ignore = "host overhead bench; run with --ignored --nocapture"]
    fn overheads() {
        let Some(dev) = open_device() else { return };
        let per_iter = |start: Instant, n: u32| start.elapsed() / n;

        let src = format!("{TOUCH_SRC}\n// {:?}\n", Instant::now());
        let t = Instant::now();
        let shader = dev.create_shader(ShaderDesc::hip(&src)).unwrap();
        println!("hip: cold compile + load        {:?}", t.elapsed());
        let t = Instant::now();
        let again = dev.create_shader(ShaderDesc::hip(&src)).unwrap();
        println!("hip: cached compile + load      {:?}", t.elapsed());
        dev.destroy_shader(again);

        let pipeline = dev
            .create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "touch",
                block: [64, 1, 1],
            })
            .unwrap();
        let out = buffer(
            dev.as_ref(),
            4096,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        let scalars = [Scalar::U32(5)];
        let bindings = Bindings {
            buffers: &[out.index()],
            textures: &[],
            images: &[],
            scalars: &scalars,
        };

        dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
        const N: u32 = 2000;
        let t = Instant::now();
        for _ in 0..N {
            dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
        }
        println!("hip: dispatch round trip        {:?}", per_iter(t, N));

        let ops: Vec<DispatchOp<'_>> = (0..N)
            .map(|_| DispatchOp {
                pipeline,
                bindings,
                grid: [1, 1, 1],
            })
            .collect();
        let t = Instant::now();
        dev.dispatch_batch(&ops).unwrap();
        println!("hip: batched launch             {:?}", per_iter(t, N));

        let t = Instant::now();
        let sub = dev.submit_batch(0, &ops).unwrap();
        println!("hip: enqueue N dispatches (CPU) {:?}", per_iter(t, N));
        sub.wait(Duration::from_secs(10)).unwrap();

        let t = Instant::now();
        for _ in 0..N {
            let sub = dev.submit(0, pipeline, bindings, [1, 1, 1]).unwrap();
            sub.wait(Duration::from_secs(10)).unwrap();
        }
        println!("hip: submit + wait              {:?}", per_iter(t, N));

        let mb = 256usize;
        let big = buffer(
            dev.as_ref(),
            (mb << 20) as u64,
            BufferUsage::STORAGE | BufferUsage::READBACK,
        );
        let host = vec![7u8; mb << 20];
        let mut back = vec![1u8; mb << 20];
        dev.write_buffer(big, 0, &host).unwrap();
        dev.read_buffer_into(big, 0, &mut back).unwrap();
        let best = |mut op: Box<dyn FnMut() + '_>| {
            (0..3)
                .map(|_| {
                    let t = Instant::now();
                    op();
                    t.elapsed()
                })
                .min()
                .unwrap()
        };
        let up = best(Box::new(|| dev.write_buffer(big, 0, &host).unwrap()));
        let down = best(Box::new(|| {
            dev.read_buffer_into(big, 0, &mut back).unwrap()
        }));
        assert_eq!(back[12345], 7);
        let gbps = |d: Duration| (mb << 20) as f64 / d.as_secs_f64() / 1e9;
        println!("hip: upload   {mb} MiB             {:.1} GB/s", gbps(up));
        println!("hip: readback {mb} MiB             {:.1} GB/s", gbps(down));

        let small = [1u8; 256];
        let t = Instant::now();
        for _ in 0..N {
            dev.write_buffer(out, 0, &small).unwrap();
        }
        println!("hip: 256 B write                {:?}", per_iter(t, N));

        for mib in [1u64, 64] {
            let t = Instant::now();
            for _ in 0..100 {
                let b = buffer(dev.as_ref(), mib << 20, BufferUsage::STORAGE);
                dev.destroy_buffer(b);
            }
            println!(
                "hip: create+destroy {mib:>2} MiB       {:?}",
                per_iter(t, 100)
            );
        }

        let first = dev.read_buffer(out, 0, 4).unwrap();
        assert_eq!(first, [1, 1, 1, 1]);

        dev.destroy_buffer(big);
        dev.destroy_buffer(out);
        dev.destroy_pipeline(pipeline);
        dev.destroy_shader(shader);
    }

    /// ZSL kernels lowered for HIP, checked against CPU references and Zetna kernels.
    mod zsl {
        use std::env;
        use std::fs;
        use std::path::Path;
        use std::sync::Arc;

        use zengpu_compute::{BufferPool, DeviceArray, OpList};
        use zengpu_hal::{AdapterRequest, DType, DeviceRequest, GpuDevice, GpuInstance};

        use crate::HipInstance;

        fn hip_device() -> Option<Arc<dyn GpuDevice>> {
            let inst = HipInstance::new().ok()?;
            let adapter = inst.request_adapter(AdapterRequest::default())?;
            Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
        }

        fn upload(pool: &BufferPool, shape: Vec<u32>, values: &[f32]) -> DeviceArray {
            let array = pool.alloc(shape, DType::F32).unwrap();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
            array
        }

        fn download(device: &dyn GpuDevice, array: &DeviceArray) -> Vec<f32> {
            device
                .read_buffer(array.buffer, 0, array.size_bytes())
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }

        fn close(a: f32, b: f32) -> bool {
            (a - b).abs() <= 1e-4 * b.abs().max(1.0)
        }

        /// Uploads f16 or bf16 bit patterns as little-endian bytes.
        fn upload_half(
            pool: &BufferPool,
            shape: Vec<u32>,
            dtype: DType,
            bits: &[u16],
        ) -> DeviceArray {
            let array = pool.alloc(shape, dtype).unwrap();
            let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
            pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
            array
        }

        /// Converts f32 to bf16, rounding to nearest even on the top 16 bits.
        fn f32_to_bf16_bits(x: f32) -> u16 {
            let b = x.to_bits();
            ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
        }

        fn bf16_bits_to_f32(h: u16) -> f32 {
            f32::from_bits(u32::from(h) << 16)
        }

        /// Converts f32 to binary16 with nearest even rounding. Subnormals are kept and overflow goes
        /// to inf.
        fn f32_to_f16_bits(x: f32) -> u16 {
            let b = x.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32;
            let man = b & 0x007f_ffff;
            if exp == 0xff {
                let payload = if man != 0 { 0x0200 } else { 0 };
                return sign | 0x7c00 | payload;
            }
            let e = exp - 112;
            if e >= 31 {
                return sign | 0x7c00;
            }
            if e <= 0 {
                if e < -10 {
                    return sign;
                }
                let full = man | 0x0080_0000;
                let shift = (14 - e) as u32;
                let q = full >> shift;
                let rem = full & ((1 << shift) - 1);
                let half = 1 << (shift - 1);
                let q = if rem > half || (rem == half && (q & 1) == 1) {
                    q + 1
                } else {
                    q
                };
                return sign | q as u16;
            }
            let m = man >> 13;
            let rem = man & 0x1fff;
            let h = ((e as u32) << 10) | m;
            let h = if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
                h + 1
            } else {
                h
            };
            sign | h as u16
        }

        /// Converts binary16 to f32 exactly, subnormals included.
        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((h >> 10) & 0x1f);
            let man = f32::from(h & 0x3ff);
            match exp {
                0 => sign * man * 2f32.powi(-24),
                31 if man == 0.0 => sign * f32::INFINITY,
                31 => f32::NAN,
                _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
            }
        }

        const HALF_IO_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel half_io(xh: device buffer<f16>, xb: device buffer<bf16>, yh: device mut buffer<f16>, yb: device mut buffer<bf16>, sum: device mut buffer<f32>, xf: device buffer<f32>, zh: device mut buffer<f16>, zb: device mut buffer<bf16>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    yh[i] = xh[i] * 2.0
                    yb[i] = xb[i] * 2.0
                    sum[i] = xh[i] + xb[i]
                    zh[i] = xf[i]
                    zb[i] = xf[i]
                }
            }
        );

        /// Rounds `x` to nearest even at `man` fraction bits. Assumes exp >= `min_exp` and a finite result
        /// <= `max`. `x` must not be NaN.
        fn round_ref(x: f32, man: i32, min_exp: i32, max: f64) -> f64 {
            let v = f64::from(x);
            if v == 0.0 || !v.is_finite() {
                return v;
            }
            let e = (v.abs().log2().floor() as i32).max(min_exp);
            let ulp = 2f64.powi(e - man);
            let q = (v / ulp).round_ties_even() * ulp;
            if q.abs() > max {
                f64::INFINITY.copysign(v)
            } else {
                q
            }
        }

        fn store_inputs(n: u32) -> Vec<f32> {
            let edge = [
                65504.0,
                65519.0,
                65520.0,
                1.0e30,
                -1.0e30,
                2f32.powi(-24),
                2f32.powi(-25),
                3.0 * 2f32.powi(-25),
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2f32.powi(-14) * (1.0 - 2f32.powi(-11)),
                3.0e38,
                f32::MAX,
                1.0e-40,
                -2.0 * 2f32.powi(-25),
                f32::NAN,
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0xffc0_0000),
            ];
            (0..n)
                .map(|i| {
                    if let Some(&e) = edge.get(i as usize) {
                        return e;
                    }
                    let v = if i % 2 == 0 {
                        (1.0 + ((i % 1024) as f32 + 0.5) * 2f32.powi(-10))
                            * 2f32.powi((i % 40) as i32 - 20)
                    } else {
                        (1.0 + ((i % 128) as f32 + 0.5) * 2f32.powi(-7))
                            * 2f32.powi((i % 40) as i32 - 20)
                    };
                    if i % 3 == 0 { -v } else { v }
                })
                .collect()
        }

        #[test]
        fn zsl_half_buffers_load_and_store() {
            let Some(device) = hip_device() else { return };
            let pool = BufferPool::new(device.clone());
            let (shader, entry) = HALF_IO_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let n = 1001u32;
            let x: Vec<f32> = (0..n).map(|i| ((i % 256) as f32 - 128.0) * 0.25).collect();
            let xh_bits: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();
            let xb_bits: Vec<u16> = x.iter().map(|&v| f32_to_bf16_bits(v)).collect();
            let xh = upload_half(&pool, vec![n], DType::F16, &xh_bits);
            let xb = upload_half(&pool, vec![n], DType::Bf16, &xb_bits);
            let yh = pool.alloc(vec![n], DType::F16).unwrap();
            let yb = pool.alloc(vec![n], DType::Bf16).unwrap();
            let sum = pool.alloc(vec![n], DType::F32).unwrap();
            let store_in = store_inputs(n);
            let xf = upload(&pool, vec![n], &store_in);
            let zh = pool.alloc(vec![n], DType::F16).unwrap();
            let zb = pool.alloc(vec![n], DType::Bf16).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    xh.buffer.index(),
                    xb.buffer.index(),
                    yh.buffer.index(),
                    yb.buffer.index(),
                    sum.buffer.index(),
                    xf.buffer.index(),
                    zh.buffer.index(),
                    zb.buffer.index(),
                ],
                &[zengpu_hal::Scalar::U32(n)],
                [n.div_ceil(64), 1, 1],
            );
            ops.run(&*device).unwrap();
            let halves = |a: &DeviceArray| -> Vec<u16> {
                device
                    .read_buffer(a.buffer, 0, a.size_bytes())
                    .unwrap()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            };
            let (out_h, out_b, out_sum) = (halves(&yh), halves(&yb), download(&*device, &sum));
            let (store_h, store_b) = (halves(&zh), halves(&zb));
            for i in 0..n as usize {
                assert_eq!(f16_bits_to_f32(out_h[i]), 2.0 * x[i], "f16 at {i}");
                assert_eq!(bf16_bits_to_f32(out_b[i]), 2.0 * x[i], "bf16 at {i}");
                assert_eq!(out_sum[i], 2.0 * x[i], "sum at {i}");
                let v = store_in[i];
                let got_h = f16_bits_to_f32(store_h[i]);
                let got_b = bf16_bits_to_f32(store_b[i]);
                if v.is_nan() {
                    assert!(got_h.is_nan(), "f16 store of NaN at {i}");
                    assert!(got_b.is_nan(), "bf16 store of NaN at {i}");
                    continue;
                }
                let want_h = round_ref(v, 10, -14, 65504.0);
                let want_b = round_ref(v, 7, -126, f64::from(f32::from_bits(0x7f7f_0000)));
                assert_eq!(
                    f64::from(got_h).to_bits(),
                    want_h.to_bits(),
                    "f16 store of {v:e} at {i}"
                );
                assert_eq!(
                    f64::from(got_b).to_bits(),
                    want_b.to_bits(),
                    "bf16 store of {v:e} at {i}"
                );
            }
            drop(ops);
            for a in [xh, xb, yh, yb, sum, xf, zh, zb] {
                pool.free(a);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        const TRIG_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
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

        #[test]
        fn zsl_trigonometry_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = TRIG_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = 128u32;
            let input: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 / n as f32).collect();
            let src = upload(&pool, vec![n], &input);
            let out = pool.alloc(vec![n], DType::F32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[src.buffer.index(), out.buffer.index()],
                &[zengpu_hal::Scalar::U32(n)],
                [n.div_ceil(64), 1, 1],
            );
            ops.run(&*device).unwrap();
            let got = download(&*device, &out);
            for (i, x) in input.iter().enumerate() {
                let want = x.sin() + x.cos() + x.tan();
                assert!(
                    (got[i] - want).abs() < 2e-5,
                    "out[{i}]: got {} want {want}",
                    got[i]
                );
            }
            drop(ops);
            pool.free(src);
            pool.free(out);
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        const FINITE_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
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
        fn zsl_finite_classification_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = FINITE_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let input = [0.0f32, -12.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
            let n = input.len() as u32;
            let src = upload(&pool, vec![n], &input);
            let out = pool.alloc(vec![n], DType::U32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[src.buffer.index(), out.buffer.index()],
                &[zengpu_hal::Scalar::U32(n)],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got: Vec<u32> = device
                .read_buffer(out.buffer, 0, out.size_bytes())
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            assert_eq!(got, [1, 1, 2, 3, 3]);
            drop(ops);
            pool.free(src);
            pool.free(out);
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        const DOT4_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel dot4(x: device buffer<i32>, y: device buffer<u32>, k: device buffer<i32>, z: device buffer<u32>, out_i: device mut buffer<i32>, out_iu: device mut buffer<i32>, out_u: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    out_i[i] = dot4_i8(x[i], x[i ^ 1], k[i])
                    out_iu[i] = dot4_iu8(x[i], y[i], k[i])
                    out_u[i] = dot4_u8(y[i], y[i ^ 1], z[i])
                }
            }
        );

        #[test]
        fn zsl_dot4_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = DOT4_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = 1000u32;
            const EDGE: [u8; 4] = [0x80, 0x7f, 0xff, 0x00];
            let word = |i: u32| {
                u32::from_le_bytes([
                    EDGE[(i % 4) as usize],
                    EDGE[((i / 4 + 1) % 4) as usize],
                    EDGE[((i / 2 + 2) % 4) as usize],
                    (i.wrapping_mul(0x9e37_79b9) >> 24) as u8,
                ])
            };
            let xs: Vec<u32> = (0..n).map(word).collect();
            let ys: Vec<u32> = (0..n).map(|i| word(i + 7)).collect();
            let ks: Vec<u32> = (0..n).map(|i| i32::MAX as u32 - i % 1000).collect();
            let zs: Vec<u32> = (0..n).map(|i| u32::MAX - i % 1000).collect();
            let mut bufs = Vec::new();
            for data in [&xs, &ys, &ks, &zs] {
                let b = pool.alloc(vec![n], DType::U32).unwrap();
                let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
                device.write_buffer(b.buffer, 0, &bytes).unwrap();
                bufs.push(b);
            }
            let outs: Vec<_> = (0..3)
                .map(|_| pool.alloc(vec![n], DType::U32).unwrap())
                .collect();
            let mut ops = OpList::new(&pool);
            let slots: Vec<u32> = bufs
                .iter()
                .chain(outs.iter())
                .map(|b| b.buffer.index())
                .collect();
            ops.dispatch(
                pipeline,
                &slots,
                &[zengpu_hal::Scalar::U32(n)],
                [n.div_ceil(64), 1, 1],
            );
            ops.run(&*device).unwrap();
            let read = |b: &DeviceArray| -> Vec<u32> {
                device
                    .read_buffer(b.buffer, 0, b.size_bytes())
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (got_i, got_iu, got_u) = (read(&outs[0]), read(&outs[1]), read(&outs[2]));
            let s8 = |w: u32, l: u32| ((w >> (8 * l)) as u8) as i8 as i32;
            let u8l = |w: u32, l: u32| ((w >> (8 * l)) & 0xff) as i32;
            for i in 0..n as usize {
                let (x, x2, y, y2) = (xs[i], xs[i ^ 1], ys[i], ys[i ^ 1]);
                let (k, z) = (ks[i] as i32, zs[i]);
                let mut want_i = k;
                let mut want_iu = k;
                let mut want_u = z;
                for l in 0..4 {
                    want_i = want_i.wrapping_add(s8(x, l).wrapping_mul(s8(x2, l)));
                    want_iu = want_iu.wrapping_add(s8(x, l).wrapping_mul(u8l(y, l)));
                    want_u = want_u.wrapping_add((u8l(y, l) * u8l(y2, l)) as u32);
                }
                assert_eq!(got_i[i] as i32, want_i, "dot4_i8 at {i}");
                assert_eq!(got_iu[i] as i32, want_iu, "dot4_iu8 at {i}");
                assert_eq!(got_u[i], want_u, "dot4_u8 at {i}");
            }
            drop(ops);
            for b in bufs.into_iter().chain(outs) {
                pool.free(b);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Reference for f16 operands: products and sums exact, rounded once at the end.
        mod packed_ref {
            #[derive(Clone, Copy)]
            enum Fin {
                Zero(bool),
                Num(bool, i128, i32),
                Inf(bool),
                Nan,
            }

            fn classify(h: u16) -> Fin {
                let neg = h >> 15 == 1;
                let exp = (h >> 10) & 0x1f;
                let frac = i128::from(h & 0x3ff);
                match exp {
                    0x1f if frac == 0 => Fin::Inf(neg),
                    0x1f => Fin::Nan,
                    0 if frac == 0 => Fin::Zero(neg),
                    0 => Fin::Num(neg, frac, -24),
                    _ => Fin::Num(neg, frac | 0x400, i32::from(exp) - 25),
                }
            }

            fn sign(f: Fin) -> bool {
                match f {
                    Fin::Zero(s) | Fin::Inf(s) | Fin::Num(s, ..) => s,
                    Fin::Nan => false,
                }
            }

            fn mul(a: Fin, b: Fin) -> Fin {
                let s = sign(a) ^ sign(b);
                match (a, b) {
                    (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
                    (Fin::Inf(_), Fin::Zero(_)) | (Fin::Zero(_), Fin::Inf(_)) => Fin::Nan,
                    (Fin::Inf(_), _) | (_, Fin::Inf(_)) => Fin::Inf(s),
                    (Fin::Zero(_), _) | (_, Fin::Zero(_)) => Fin::Zero(s),
                    (Fin::Num(_, m1, e1), Fin::Num(_, m2, e2)) => Fin::Num(s, m1 * m2, e1 + e2),
                }
            }

            fn add(a: Fin, b: Fin) -> Fin {
                match (a, b) {
                    (Fin::Nan, _) | (_, Fin::Nan) => Fin::Nan,
                    (Fin::Inf(x), Fin::Inf(y)) => {
                        if x == y {
                            Fin::Inf(x)
                        } else {
                            Fin::Nan
                        }
                    }
                    (Fin::Inf(x), _) | (_, Fin::Inf(x)) => Fin::Inf(x),
                    (Fin::Zero(x), Fin::Zero(y)) => Fin::Zero(x && y),
                    (Fin::Zero(_), o) | (o, Fin::Zero(_)) => o,
                    (Fin::Num(sa, ma, ea), Fin::Num(sb, mb, eb)) => {
                        let e = ea.min(eb);
                        let va = (if sa { -ma } else { ma }) << (ea - e);
                        let vb = (if sb { -mb } else { mb }) << (eb - e);
                        let s = va + vb;
                        if s == 0 {
                            Fin::Zero(false)
                        } else {
                            Fin::Num(s < 0, s.abs(), e)
                        }
                    }
                }
            }

            /// Rounds `m * 2^e` (m > 0) to nearest even.
            fn round(neg: bool, m: u128, e: i32) -> u16 {
                let bitlen = (128 - m.leading_zeros()) as i32;
                let drop = (bitlen - 11).max(-24 - e);
                let (mut q, mut qe) = if drop <= 0 {
                    (m << (-drop) as u32, e + drop)
                } else if drop >= 127 {
                    (0, -24)
                } else {
                    let d = drop as u32;
                    let mut q = m >> d;
                    let rem = m & ((1u128 << d) - 1);
                    let half = 1u128 << (d - 1);
                    if rem > half || (rem == half && q & 1 == 1) {
                        q += 1;
                    }
                    (q, e + drop)
                };
                let sign = u16::from(neg) << 15;
                if q == 0 {
                    return sign;
                }
                while q >= 2048 {
                    q >>= 1;
                    qe += 1;
                }
                if q >= 1024 {
                    let t = qe + 10;
                    if t > 15 {
                        return sign | 0x7c00;
                    }
                    sign | (((t + 15) as u16) << 10) | (q as u16 - 1024)
                } else {
                    sign | q as u16
                }
            }

            fn bits(f: Fin) -> u16 {
                match f {
                    Fin::Nan => 0x7e00,
                    Fin::Inf(s) => (u16::from(s) << 15) | 0x7c00,
                    Fin::Zero(s) => u16::from(s) << 15,
                    Fin::Num(s, m, e) => round(s, m as u128, e),
                }
            }

            pub fn add16(a: u16, b: u16) -> u16 {
                bits(add(classify(a), classify(b)))
            }

            pub fn mul16(a: u16, b: u16) -> u16 {
                bits(mul(classify(a), classify(b)))
            }

            pub fn fma16(a: u16, b: u16, c: u16) -> u16 {
                bits(add(mul(classify(a), classify(b)), classify(c)))
            }

            fn to_f32(h: u16) -> f32 {
                match classify(h) {
                    Fin::Nan => f32::NAN,
                    Fin::Inf(s) => {
                        if s {
                            f32::NEG_INFINITY
                        } else {
                            f32::INFINITY
                        }
                    }
                    Fin::Zero(s) => {
                        if s {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    Fin::Num(s, m, e) => {
                        let v = m as f32 * 2f32.powi(e);
                        if s { -v } else { v }
                    }
                }
            }

            /// Reference dot2 in f32: exact products, then two adds.
            pub fn dot2(a: u32, b: u32, acc: f32) -> f32 {
                let p0 = to_f32(a as u16) * to_f32(b as u16);
                let p1 = to_f32((a >> 16) as u16) * to_f32((b >> 16) as u16);
                (acc + p0) + p1
            }

            pub fn lanes(a: u32, b: u32, f: impl Fn(u16, u16) -> u16) -> u32 {
                u32::from(f(a as u16, b as u16))
                    | (u32::from(f((a >> 16) as u16, (b >> 16) as u16)) << 16)
            }

            pub fn lanes3(a: u32, b: u32, c: u32, f: impl Fn(u16, u16, u16) -> u16) -> u32 {
                u32::from(f(a as u16, b as u16, c as u16))
                    | (u32::from(f((a >> 16) as u16, (b >> 16) as u16, (c >> 16) as u16)) << 16)
            }

            /// Compares per f16 lane, bit-exact except that any NaN matches any NaN.
            pub fn same_f16x2(got: u32, want: u32) -> bool {
                let lane_ok = |g: u16, w: u16| {
                    let nan = |h: u16| (h & 0x7fff) > 0x7c00;
                    (nan(g) && nan(w)) || g == w
                };
                lane_ok(got as u16, want as u16) && lane_ok((got >> 16) as u16, (want >> 16) as u16)
            }

            pub fn pk_add_u16(a: u32, b: u32) -> u32 {
                lanes(a, b, |x, y| x.wrapping_add(y))
            }

            pub fn pk_mul_u16(a: u32, b: u32) -> u32 {
                lanes(a, b, |x, y| x.wrapping_mul(y))
            }

            pub fn pk_mad_u16(a: u32, b: u32, c: u32) -> u32 {
                lanes3(a, b, c, |x, y, z| x.wrapping_mul(y).wrapping_add(z))
            }

            pub fn funnel_shr(hi: u32, lo: u32, s: u32) -> u32 {
                ((u64::from(hi) << 32 | u64::from(lo)) >> (s & 31)) as u32
            }

            /// Reference for CUDA `__byte_perm` with the 3-bit selector and no sign mode.
            pub fn byte_perm(a: u32, b: u32, sel: u32) -> u32 {
                let src = u64::from(a) | (u64::from(b) << 32);
                (0..4).fold(0u32, |out, i| {
                    let k = (sel >> (4 * i)) & 7;
                    let byte = ((src >> (8 * k)) & 0xff) as u32;
                    out | (byte << (8 * i))
                })
            }
        }

        const PACKED_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32 }
            @workgroup_size(64)
            kernel packed(x: device buffer<u32>, y: device buffer<u32>, z: device buffer<u32>, f: device buffer<f32>, o1: device mut buffer<u32>, o2: device mut buffer<u32>, o3: device mut buffer<u32>, o4: device mut buffer<u32>, o5: device mut buffer<u32>, o6: device mut buffer<u32>, o7: device mut buffer<u32>, o8: device mut buffer<u32>, of: device mut buffer<f32>, o9: device mut buffer<u32>, o10: device mut buffer<u32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    o1[i] = pk_add_u16(x[i], y[i])
                    o2[i] = pk_mul_u16(x[i], y[i])
                    o3[i] = pk_mad_u16(x[i], y[i], z[i])
                    o4[i] = pk_add_f16(x[i], y[i])
                    o5[i] = pk_mul_f16(x[i], y[i])
                    o6[i] = pk_fma_f16(x[i], y[i], z[i])
                    of[i] = dot2_f16(x[i], y[i], f[i])
                    o7[i] = funnel_shr(x[i], y[i], z[i])
                    o8[i] = byte_perm(x[i], y[i], z[i])
                    o9[i] = pk_mul_f16(x[i], 0x3c003c00u32)
                    o10[i] = pk_add_u16(x[i], 0x3c003c00u32) + (-2147483648) + 0b1010 + 0o17 + 0xffffffffu32
                }
            }
        );

        #[test]
        fn zsl_packed_ops_match_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = PACKED_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = 1024u32;
            const F16_EDGE: [u16; 16] = [
                0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff,
                0xfbff, 0x3c00, 0xbc00, 0x3555, 0x4000, 0x0200,
            ];
            let half = |s: u32| -> u16 {
                if s.is_multiple_of(3) {
                    (s.wrapping_mul(0x9e37_79b9) >> 16) as u16
                } else {
                    F16_EDGE[(s % 16) as usize]
                }
            };
            let lanes = |s: u32| u32::from(half(s)) | (u32::from(half(s.wrapping_add(7919))) << 16);
            let mut xs: Vec<u32> = (0..n).map(lanes).collect();
            let mut ys: Vec<u32> = (0..n).map(|i| lanes(i + 4096)).collect();
            let mut zs: Vec<u32> = (0..n)
                .map(|i| match i % 5 {
                    0 => 0,
                    1 => 31,
                    2 => 32,
                    3 => 0x7654_3210,
                    _ => lanes(i + 8192),
                })
                .collect();
            xs[0] = (xs[0] & 0xffff_0000) | 0x3c01;
            ys[0] = (ys[0] & 0xffff_0000) | 0x3bff;
            zs[0] = (zs[0] & 0xffff_0000) | 0x0009;
            let fs: Vec<f32> = (0..n)
                .map(|i| match i % 4 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => i as f32 * 0.37 - 100.0,
                    _ => f32::from_bits(lanes(i + 123)),
                })
                .collect();
            let mut inputs = Vec::new();
            for data in [&xs, &ys, &zs] {
                let b = pool.alloc(vec![n], DType::U32).unwrap();
                let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
                device.write_buffer(b.buffer, 0, &bytes).unwrap();
                inputs.push(b);
            }
            let b = pool.alloc(vec![n], DType::F32).unwrap();
            let bytes: Vec<u8> = fs.iter().flat_map(|w| w.to_le_bytes()).collect();
            device.write_buffer(b.buffer, 0, &bytes).unwrap();
            inputs.push(b);
            let outs: Vec<_> = (0..11)
                .map(|k| {
                    let ty = if k == 8 { DType::F32 } else { DType::U32 };
                    pool.alloc(vec![n], ty).unwrap()
                })
                .collect();
            let mut ops = OpList::new(&pool);
            let slots: Vec<u32> = inputs
                .iter()
                .chain(outs.iter())
                .map(|b| b.buffer.index())
                .collect();
            ops.dispatch(
                pipeline,
                &slots,
                &[zengpu_hal::Scalar::U32(n)],
                [n.div_ceil(64), 1, 1],
            );
            ops.run(&*device).unwrap();
            let read = |b: &DeviceArray| -> Vec<u32> {
                device
                    .read_buffer(b.buffer, 0, b.size_bytes())
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let got: Vec<Vec<u32>> = outs.iter().map(read).collect();
            use packed_ref as r;
            for i in 0..n as usize {
                let (x, y, z) = (xs[i], ys[i], zs[i]);
                assert!(
                    r::same_f16x2(got[9][i], r::lanes(x, 0x3c00_3c00, r::mul16)),
                    "pk_mul_f16 with a hex literal at {i}: x={x:08x}"
                );
                assert_eq!(
                    got[10][i],
                    r::pk_add_u16(x, 0x3c00_3c00)
                        .wrapping_add(0x8000_0000)
                        .wrapping_add(10)
                        .wrapping_add(15)
                        .wrapping_add(0xffff_ffff),
                    "literal expression at {i}"
                );
                let want = [
                    r::pk_add_u16(x, y),
                    r::pk_mul_u16(x, y),
                    r::pk_mad_u16(x, y, z),
                    r::lanes(x, y, r::add16),
                    r::lanes(x, y, r::mul16),
                    r::lanes3(x, y, z, r::fma16),
                    r::funnel_shr(x, y, z),
                    r::byte_perm(x, y, z),
                ];
                for (k, w) in want.iter().enumerate() {
                    let f16_out = (3..=5).contains(&k);
                    let ok = if f16_out {
                        r::same_f16x2(got[k][i], *w)
                    } else {
                        got[k][i] == *w
                    };
                    assert!(
                        ok,
                        "output {k} at {i}: got {:08x} want {w:08x} x={x:08x} y={y:08x} z={z:08x}",
                        got[k][i]
                    );
                }
                let want_f = r::dot2(x, y, fs[i]);
                let got_f = f32::from_bits(got[8][i]);
                let p0 = r::dot2(x & 0xffff, y & 0xffff, 0.0);
                let p1 = r::dot2(x >> 16, y >> 16, 0.0);
                let tol = fs[i].abs().max(p0.abs()).max(p1.abs()) * 1e-4;
                assert!(
                    (want_f.is_nan() && got_f.is_nan())
                        || want_f.to_bits() == got_f.to_bits()
                        || (got_f - want_f).abs() <= tol,
                    "dot2_f16 at {i}: got {got_f:?} want {want_f:?} x={x:08x} y={y:08x} acc={:?}",
                    fs[i]
                );
            }
            drop(ops);
            for b in inputs.into_iter().chain(outs) {
                pool.free(b);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// gfx12 WMMA: f16 and packed-i8 `C = A*B + C`. SPIR-V has no `frag_row`/`frag_get`/`frag_set`: the
        /// lane probe is in `FRAG_PROBE`.
        const FRAG_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel frag_gemm(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, out: device mut buffer<f32>, ba: device buffer<u32>, bb: device buffer<u32>, ci: device buffer<i32>, oi: device mut buffer<i32>) {
                let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let d = mma(fa, fb, acc)
                frag_store(out, 0, 16, d)
                let ia: frag<i8, 16, 16, a> = frag_load_a(ba, 0, 16)
                let ib: frag<i8, 16, 16, b> = frag_load_b(bb, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                let id2 = mma(ia, ib, iacc)
                frag_store(oi, 0, 16, id2)
            }
        );

        /// Needs `frag_get`/`frag_set`, which SPIR-V cannot express, so HIP only.
        const FRAG_GET_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel frag_get_bump(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, out: device mut buffer<f32>) {
                let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let d = mma(fa, fb, acc)
                let d2 = frag_set(d, 0, frag_get(d, 0) + 1.0)
                frag_store(out, 0, 16, d2)
            }
        );

        const TYPED_SHARED_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel typed_shared(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, ia: device buffer<i8>, ib: device buffer<i8>, ci: device buffer<i32>, wide: device buffer<i32>, ui: device buffer<u8>, e: device buffer<f32>, out: device mut buffer<f32>, oi: device mut buffer<i32>, rh: device mut buffer<f32>, rq: device mut buffer<i32>, ru: device mut buffer<u32>) {
                shared sa: array<f16, 256>
                shared sb: array<f16, 256>
                shared qa: array<i8, 256>
                shared qb: array<i8, 256>
                shared qw: array<i8, 256>
                shared us: array<u8, 256>
                shared he: array<f16, 256>
                let lane = local_id().x
                for i in 0..8 {
                    sa[lane + i * 32] = a[lane + i * 32]
                    sb[lane + i * 32] = b[lane + i * 32]
                    qa[lane + i * 32] = ia[lane + i * 32]
                    qb[lane + i * 32] = ib[lane + i * 32]
                    qw[lane + i * 32] = wide[lane + i * 32]
                    us[lane + i * 32] = ui[lane + i * 32]
                    he[lane + i * 32] = e[lane + i * 32]
                }
                barrier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b(sb, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let d = mma(fa, fb, acc)
                frag_store(out, 0, 16, d)
                let ga: frag<i8, 16, 16, a> = frag_load_a(qa, 0, 16)
                let gb: frag<i8, 16, 16, b> = frag_load_b(qb, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                let id2 = mma(ga, gb, iacc)
                frag_store(oi, 0, 16, id2)
                for i in 0..8 {
                    rh[lane + i * 32] = he[lane + i * 32]
                    rq[lane + i * 32] = qw[lane + i * 32]
                    ru[lane + i * 32] = us[lane + i * 32]
                }
            }
        );

        #[test]
        fn zsl_typed_shared_round_trips_and_feeds_frags() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = TYPED_SHARED_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let fa: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let fb: Vec<f32> = (0..256)
                .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
                .collect();
            let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
            let ia: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
            let ib: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
            let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
            let wide: Vec<i32> = (0..256i32).map(|i| i * 37 - 4000).collect();
            let ui: Vec<u8> = (0..256u32).map(|i| (i * 53 % 256) as u8).collect();
            let edge: Vec<f32> = (0..256)
                .map(|i| {
                    [
                        0.1,
                        65504.0,
                        65520.0,
                        1.0 + 2f32.powi(-11),
                        1.0 + 3.0 * 2f32.powi(-11),
                        1e-5,
                        -2.5e-7,
                        123.456,
                        2.75,
                        -0.0,
                    ][i % 10]
                })
                .collect();
            let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let i8s = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
            let make = |dtype: DType, bytes: &[u8]| {
                let arr = pool.alloc(vec![256], dtype).unwrap();
                device.write_buffer(arr.buffer, 0, bytes).unwrap();
                arr
            };
            let a = make(DType::F32, &f32s(&fa));
            let b = make(DType::F32, &f32s(&fb));
            let c = make(DType::F32, &f32s(&fc));
            let ga = make(DType::I8, &i8s(&ia));
            let gb = make(DType::I8, &i8s(&ib));
            let gci = make(DType::I32, &i32s(&ci));
            let gw = make(DType::I32, &i32s(&wide));
            let gu = make(DType::U8, &ui);
            let ge = make(DType::F32, &f32s(&edge));
            let out = pool.alloc(vec![256], DType::F32).unwrap();
            let oi = pool.alloc(vec![256], DType::I32).unwrap();
            let rh = pool.alloc(vec![256], DType::F32).unwrap();
            let rq = pool.alloc(vec![256], DType::I32).unwrap();
            let ru = pool.alloc(vec![256], DType::U32).unwrap();
            let bufs = [a, b, c, ga, gb, gci, gw, gu, ge, out, oi, rh, rq, ru];
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &bufs.iter().map(|x| x.buffer.index()).collect::<Vec<_>>(),
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let read = |x: &DeviceArray| device.read_buffer(x.buffer, 0, x.size_bytes()).unwrap();
            let floats = |bytes: Vec<u8>| -> Vec<f32> {
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let ints = |bytes: Vec<u8>| -> Vec<i32> {
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (got_out, got_oi) = (floats(read(&bufs[9])), ints(read(&bufs[10])));
            let (got_rh, got_rq) = (floats(read(&bufs[11])), ints(read(&bufs[12])));
            let got_ru: Vec<u32> = read(&bufs[13])
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for m in 0..16 {
                for n in 0..16 {
                    let mut s = fc[m * 16 + n];
                    let mut si = ci[m * 16 + n];
                    for k in 0..16 {
                        s += fa[m * 16 + k] * fb[k * 16 + n];
                        si = si.wrapping_add(i32::from(ia[m * 16 + k]) * i32::from(ib[k * 16 + n]));
                    }
                    let at = m * 16 + n;
                    assert!(
                        close(got_out[at], s),
                        "f16 mma from shared at ({m}, {n}): {} vs {s}",
                        got_out[at]
                    );
                    assert_eq!(got_oi[at], si, "i8 mma from shared at ({m}, {n})");
                }
            }
            for i in 0..256 {
                let want = f16_bits_to_f32(f32_to_f16_bits(edge[i]));
                assert_eq!(
                    got_rh[i].to_bits(),
                    want.to_bits(),
                    "f16 round trip of {}",
                    edge[i]
                );
                assert_eq!(
                    got_rq[i],
                    i32::from(wide[i] as i8),
                    "i8 narrowing of {}",
                    wide[i]
                );
                assert_eq!(got_ru[i], u32::from(ui[i]), "u8 round trip");
            }
            drop(ops);
            for x in bufs {
                pool.free(x);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Loads B from an N x K row-major source with `frag_load_b_t`, with no transpose stage.
        const FRAG_BT_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel frag_bt(a: device buffer<f16>, b: device buffer<f16>, ai: device buffer<u32>, bi: device buffer<u32>, c: device buffer<f32>, ci: device buffer<i32>, out: device mut buffer<f32>, oi: device mut buffer<i32>) {
                let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b_t(b, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                frag_store(out, 0, 16, mma(fa, fb, acc))
                let ia: frag<i8, 16, 16, a> = frag_load_a(ai, 0, 16)
                let ib: frag<i8, 16, 16, b> = frag_load_b_t(bi, 0, 16)
                let iacc: frag<i32, 16, 16, acc> = frag_load_acc(ci, 0, 16)
                frag_store(oi, 0, 16, mma(ia, ib, iacc))
            }
        );

        #[test]
        fn zsl_frag_load_b_t_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = FRAG_BT_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let fa: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let fb: Vec<f32> = (0..256)
                .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
                .collect();
            let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
            let ia: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
            let ib: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
            let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
            let bits_a: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let bits_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let pack = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
            let a = upload_half(&pool, vec![256], DType::F16, &bits_a);
            let b = upload_half(&pool, vec![256], DType::F16, &bits_b);
            let ai = pool.alloc(vec![64], DType::U32).unwrap();
            let bi = pool.alloc(vec![64], DType::U32).unwrap();
            device.write_buffer(ai.buffer, 0, &pack(&ia)).unwrap();
            device.write_buffer(bi.buffer, 0, &pack(&ib)).unwrap();
            let c = upload(&pool, vec![256], &fc);
            let ci_buf = pool.alloc(vec![256], DType::I32).unwrap();
            let ci_bytes: Vec<u8> = ci.iter().flat_map(|x| x.to_le_bytes()).collect();
            device.write_buffer(ci_buf.buffer, 0, &ci_bytes).unwrap();
            let out = upload(&pool, vec![256], &[0.0; 256]);
            let oi = pool.alloc(vec![256], DType::I32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    a.buffer.index(),
                    b.buffer.index(),
                    ai.buffer.index(),
                    bi.buffer.index(),
                    c.buffer.index(),
                    ci_buf.buffer.index(),
                    out.buffer.index(),
                    oi.buffer.index(),
                ],
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got = download(&*device, &out);
            let got_i: Vec<i32> = device
                .read_buffer(oi.buffer, 0, 1024)
                .unwrap()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for m in 0..16 {
                for n in 0..16 {
                    let mut s = fc[m * 16 + n];
                    let mut si = ci[m * 16 + n];
                    for k in 0..16 {
                        s += fa[m * 16 + k] * fb[n * 16 + k];
                        si = si.wrapping_add(i32::from(ia[m * 16 + k]) * i32::from(ib[n * 16 + k]));
                    }
                    assert!(
                        close(got[m * 16 + n], s),
                        "f16 ({m}, {n}): {} vs {s}",
                        got[m * 16 + n]
                    );
                    assert_eq!(got_i[m * 16 + n], si, "i8 ({m}, {n})");
                }
            }
            drop(ops);
            for buf in [a, b, c, out] {
                pool.free(buf);
            }
            for buf in [ai, bi, ci_buf, oi] {
                pool.free(buf);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        const FEATURES_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel scalar_features(ua: device buffer<u32>, ia: device buffer<i32>, fa: device buffer<f32>, sb: device buffer<i8>, ub: device buffer<u8>, ou: device mut buffer<u32>, oi: device mut buffer<i32>, of: device mut buffer<f32>, ob: device mut buffer<i8>, oub: device mut buffer<u8>, oh: device mut buffer<i32>, oh2: device mut buffer<i32>, ofr: device mut buffer<f32>, osel: device mut buffer<i32>, id: global_id) {
                let i = id.x
                ou[i] = ua[i] % 7u
                oi[i] = ia[i] % 5
                of[i] = fa[i] % 2.5
                ob[i] = sb[i] + 1
                oub[i] = ub[i] + 1
                oh[i] = i32(fa[i] * 3.0)
                oh2[i] = i32(ua[i])
                ofr[i] = f16_round(fa[i])
                osel[i] = select(ia[i] < 0, ia[i] * 2, ia[i])
            }
        );

        #[test]
        fn zsl_scalar_features_match_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = FEATURES_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = 256usize;
            let ua: Vec<u32> = (0..n as u32).map(|i| i * 37 + 5).collect();
            let ia: Vec<i32> = (0..n as i32).map(|i| i * 13 - 1500).collect();
            let fa: Vec<f32> = (0..n)
                .map(|i| match i % 16 {
                    0 => 3.0e9,
                    1 => -3.0e9,
                    _ => i as f32 * 0.37 - 40.0,
                })
                .collect();
            let sb: Vec<i8> = (0..n as i32).map(|i| (i * 7 - 500) as i8).collect();
            let ub: Vec<u8> = (0..n).map(|i| (i * 11 % 256) as u8).collect();
            let u32_buf =
                |vals: &[u32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
            let i32_buf =
                |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
            let f32_buf =
                |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
            let sb_bytes: Vec<u8> = sb.iter().map(|&v| v as u8).collect();
            let mk = |bytes: &[u8]| {
                let buf = pool
                    .alloc(vec![(bytes.len() as u32).div_ceil(4)], DType::U32)
                    .unwrap();
                device.write_buffer(buf.buffer, 0, bytes).unwrap();
                buf
            };
            let bufs = [
                mk(&u32_buf(&ua)),
                mk(&i32_buf(&ia)),
                mk(&f32_buf(&fa)),
                mk(&sb_bytes),
                mk(&ub),
            ];
            let outs: Vec<_> = [1024usize, 1024, 1024, 256, 256, 1024, 1024, 1024, 1024]
                .iter()
                .map(|&size| {
                    pool.alloc(vec![(size as u32).div_ceil(4)], DType::U32)
                        .unwrap()
                })
                .collect();
            let mut ops = OpList::new(&pool);
            let slots: Vec<_> = bufs
                .iter()
                .chain(outs.iter())
                .map(|b| b.buffer.index())
                .collect();
            ops.dispatch(pipeline, &slots, &[], [8, 1, 1]);
            ops.run(&*device).unwrap();
            let read = |b: &DeviceArray, len: u64| device.read_buffer(b.buffer, 0, len).unwrap();
            let got_u32 = |b: &DeviceArray| -> Vec<u32> {
                read(b, 1024)
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let got_i32 = |b: &DeviceArray| -> Vec<i32> {
                read(b, 1024)
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let got_f32 = |b: &DeviceArray| -> Vec<f32> {
                read(b, 1024)
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (ou, oi, of) = (got_u32(&outs[0]), got_i32(&outs[1]), got_f32(&outs[2]));
            let ob = read(&outs[3], 256);
            let oub = read(&outs[4], 256);
            let (oh, oh2, ofr, osel) = (
                got_i32(&outs[5]),
                got_i32(&outs[6]),
                got_f32(&outs[7]),
                got_i32(&outs[8]),
            );
            for i in 0..n {
                assert_eq!(ou[i], ua[i] % 7, "u32 %% at {i}");
                assert_eq!(oi[i], ia[i] % 5, "i32 %% at {i}");
                assert!(
                    of[i] == fa[i] % 2.5,
                    "f32 % at {i}: {} vs {}",
                    of[i],
                    fa[i] % 2.5
                );
                assert_eq!(ob[i] as i8, (i32::from(sb[i]) + 1) as i8, "i8 store at {i}");
                assert_eq!(oub[i], ((u32::from(ub[i])) + 1) as u8, "u8 store at {i}");
                assert_eq!(oh[i], (fa[i] * 3.0) as i32, "i32(f32) at {i}");
                assert_eq!(oh2[i], ua[i] as i32, "i32(u32) at {i}");
                let want_round = f16_bits_to_f32(f32_to_f16_bits(fa[i]));
                assert!(
                    ofr[i] == want_round || (ofr[i].is_nan() && want_round.is_nan()),
                    "f16_round at {i}: {} vs {want_round}",
                    ofr[i]
                );
                let want_sel = if ia[i] < 0 { ia[i] * 2 } else { ia[i] };
                assert_eq!(osel[i], want_sel, "select at {i}");
            }
            drop(ops);
            for b in bufs.into_iter().chain(outs) {
                pool.free(b);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Lane probe, HIP only: SPIR-V's cooperative matrix mapping is opaque. Lowered with `zengpu-zslc`,
        /// not `zsl!`.
        const FRAG_PROBE: &str = r#"
    @workgroup_size(32)
    kernel frag_probe(a: device buffer<f16>, b: device buffer<f16>, c: device buffer<f32>, pr: device mut buffer<u32>, pc: device mut buffer<u32>, pv: device mut buffer<f32>, out2: device mut buffer<f32>, id: global_id) {
        let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
        let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
        let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
        let d = mma(fa, fb, acc)
        let lane = local_id().x
        for i in 0..8 {
            pr[lane * 8 + i] = frag_row(d, i)
            pc[lane * 8 + i] = frag_col(d, i)
            pv[lane * 8 + i] = frag_get(d, i)
        }
        let d2 = frag_set(d, 0, frag_get(d, 0) + 1.0)
        frag_store(out2, 0, 16, d2)
    }
"#;

        #[test]
        fn zsl_frag_mma_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = FRAG_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let fa: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let fb: Vec<f32> = (0..256)
                .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
                .collect();
            let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
            let ba8: Vec<i8> = (0..256).map(|i| ((i * 5 + 2) % 16) as i8 - 8).collect();
            let bb8: Vec<i8> = (0..256).map(|i| ((i * 11 + 7) % 16) as i8 - 8).collect();
            let ci: Vec<i32> = (0..256i32).map(|i| i % 13 - 6).collect();
            let words = |v: &[i8]| -> Vec<u8> {
                v.chunks_exact(4)
                    .flat_map(|c| {
                        u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])
                            .to_le_bytes()
                    })
                    .collect()
            };
            let halves: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let halves_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let a = upload_half(&pool, vec![256], DType::F16, &halves);
            let b = upload_half(&pool, vec![256], DType::F16, &halves_b);
            let c = upload(&pool, vec![256], &fc);
            let ba = pool.alloc(vec![64], DType::U32).unwrap();
            let bb = pool.alloc(vec![64], DType::U32).unwrap();
            device.write_buffer(ba.buffer, 0, &words(&ba8)).unwrap();
            device.write_buffer(bb.buffer, 0, &words(&bb8)).unwrap();
            let ci_buf = pool.alloc(vec![256], DType::I32).unwrap();
            let ci_bytes: Vec<u8> = ci.iter().flat_map(|v| v.to_le_bytes()).collect();
            device.write_buffer(ci_buf.buffer, 0, &ci_bytes).unwrap();
            let out = upload(&pool, vec![256], &[0.0; 256]);
            let oi = pool.alloc(vec![256], DType::I32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    a.buffer.index(),
                    b.buffer.index(),
                    c.buffer.index(),
                    out.buffer.index(),
                    ba.buffer.index(),
                    bb.buffer.index(),
                    ci_buf.buffer.index(),
                    oi.buffer.index(),
                ],
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got_f = download(&*device, &out);
            let got_i: Vec<i32> = device
                .read_buffer(oi.buffer, 0, oi.size_bytes())
                .unwrap()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for m in 0..16 {
                for n in 0..16 {
                    let mut s = fc[m * 16 + n];
                    let mut si = ci[m * 16 + n];
                    for k in 0..16 {
                        s += fa[m * 16 + k] * fb[k * 16 + n];
                        si = si
                            .wrapping_add(i32::from(ba8[m * 16 + k]) * i32::from(bb8[k * 16 + n]));
                    }
                    let at = m * 16 + n;
                    assert!(
                        close(got_f[at], s),
                        "f16 mma at ({m}, {n}): {} vs {s}",
                        got_f[at]
                    );
                    assert_eq!(got_i[at], si, "i8 mma at ({m}, {n})");
                }
            }
            drop(ops);
            for buf in [a, b, c, out, oi] {
                pool.free(buf);
            }
            for buf in [ba, bb, ci_buf] {
                pool.free(buf);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        #[test]
        fn zsl_frag_get_is_hip_only() {
            assert!(FRAG_GET_ZSL.spv.is_err(), "frag_get has no SPIR-V lowering");
            let err = FRAG_GET_ZSL.spirv_desc().unwrap_err().to_string();
            assert!(err.contains("no spirv artifact"), "{err}");
            assert!(FRAG_GET_ZSL.msl.is_err(), "frag_get has no MSL lowering");
        }

        #[test]
        fn zsl_frag_probe_maps_lanes() {
            use zengpu_zslc::{backend::hip::lower_compute, frontend::parser::parse_compute};
            let Some(device) = hip_device() else { return };
            let module = parse_compute(FRAG_PROBE).unwrap();
            let hip = lower_compute(&module).unwrap();
            let shader = device
                .create_shader(zengpu_hal::ShaderDesc::hip(&hip.source))
                .unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry: hip.entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let fa: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let fb: Vec<f32> = (0..256)
                .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
                .collect();
            let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
            let halves: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let halves_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let a = upload_half(&pool, vec![256], DType::F16, &halves);
            let b = upload_half(&pool, vec![256], DType::F16, &halves_b);
            let c = upload(&pool, vec![256], &fc);
            let pr = pool.alloc(vec![256], DType::U32).unwrap();
            let pc = pool.alloc(vec![256], DType::U32).unwrap();
            let pv = upload(&pool, vec![256], &[0.0; 256]);
            let out2 = upload(&pool, vec![256], &[0.0; 256]);
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    a.buffer.index(),
                    b.buffer.index(),
                    c.buffer.index(),
                    pr.buffer.index(),
                    pc.buffer.index(),
                    pv.buffer.index(),
                    out2.buffer.index(),
                ],
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let read_u32 = |arr: &DeviceArray| -> Vec<u32> {
                device
                    .read_buffer(arr.buffer, 0, arr.size_bytes())
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (got_r, got_c) = (read_u32(&pr), read_u32(&pc));
            let got_v = download(&*device, &pv);
            let got_out2 = download(&*device, &out2);

            let mut want = vec![0f32; 256];
            for m in 0..16 {
                for n in 0..16 {
                    let mut s = fc[m * 16 + n];
                    for k in 0..16 {
                        s += fa[m * 16 + k] * fb[k * 16 + n];
                    }
                    want[m * 16 + n] = s;
                }
            }
            let mut seen = [false; 256];
            for i in 0..256 {
                let (r, c) = (got_r[i] as usize, got_c[i] as usize);
                assert!(r < 16 && c < 16, "frag_row/frag_col out of range at {i}");
                assert!(!seen[r * 16 + c], "frag position ({r}, {c}) mapped twice");
                seen[r * 16 + c] = true;
                assert!(close(got_v[i], want[r * 16 + c]), "frag_get at ({r}, {c})");
            }
            for r in 0..16 {
                for c in 0..16 {
                    let bump = if r == 0 || r == 8 { 1.0 } else { 0.0 };
                    assert!(
                        close(got_out2[r * 16 + c], want[r * 16 + c] + bump),
                        "frag_set store at ({r}, {c})"
                    );
                }
            }
            drop(ops);
            for buf in [a, b, c, pv, out2] {
                pool.free(buf);
            }
            for buf in [pr, pc] {
                pool.free(buf);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Vector copy shader over `u32x4` elements, with a scalar tail past the last full vector.
        const VEC_COPY_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32, tail: u32 }
            @workgroup_size(64)
            kernel vec_copy(src: device buffer<u32x4>, out: device mut buffer<u32x4>, words: device buffer<u32>, out_words: device mut buffer<u32>, p: P, id: global_id) {
                if id.x < p.n {
                    let v = src[id.x]
                    out[id.x] = u32x4(v.w, v.z, v.y + v.x, v.x)
                }
                if id.x < p.tail {
                    out_words[p.n * 4 + id.x] = words[p.n * 4 + id.x]
                }
            }
        );

        #[test]
        fn zsl_u32x4_copy_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = VEC_COPY_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let (n, tail) = (5u32, 3u32);
            let words: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
            let src_words: Vec<u32> = (0..20u32)
                .map(|i| i.wrapping_mul(40_503) ^ 0x9e37)
                .collect();
            let as_bytes =
                |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let src = pool.alloc(vec![20], DType::U32).unwrap();
            device
                .write_buffer(src.buffer, 0, &as_bytes(&src_words))
                .unwrap();
            let wbuf = pool.alloc(vec![32], DType::U32).unwrap();
            device
                .write_buffer(wbuf.buffer, 0, &as_bytes(&words))
                .unwrap();
            let out = pool.alloc(vec![20], DType::U32).unwrap();
            device
                .write_buffer(out.buffer, 0, &as_bytes(&[0u32; 20]))
                .unwrap();
            let out_words = pool.alloc(vec![32], DType::U32).unwrap();
            device
                .write_buffer(out_words.buffer, 0, &as_bytes(&[0u32; 32]))
                .unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    src.buffer.index(),
                    out.buffer.index(),
                    wbuf.buffer.index(),
                    out_words.buffer.index(),
                ],
                &[zengpu_hal::Scalar::U32(n), zengpu_hal::Scalar::U32(tail)],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let read = |b: &DeviceArray, count: usize| -> Vec<u32> {
                device
                    .read_buffer(b.buffer, 0, (count * 4) as u64)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let got = read(&out, 20);
            for i in 0..n as usize {
                let v = &src_words[i * 4..i * 4 + 4];
                let want = [v[3], v[2], v[1].wrapping_add(v[0]), v[0]];
                assert_eq!(&got[i * 4..i * 4 + 4], &want, "vector {i}");
            }
            let got_words = read(&out_words, 32);
            let tail_start = (n * 4) as usize;
            for i in 0..tail as usize {
                assert_eq!(
                    got_words[tail_start + i],
                    words[tail_start + i],
                    "tail word {i}"
                );
            }
            assert!(got_words[..tail_start].iter().all(|&w| w == 0));
        }

        /// Expected f16 decode results: exact, with NaN staying NaN.
        fn f16_decode(h: u16) -> f32 {
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let e = i32::from((h >> 10) & 0x1f);
            let m = f32::from(h & 0x3ff);
            if e == 0 {
                sign * m * 2f32.powi(-24)
            } else if e == 31 {
                if m == 0.0 {
                    sign * f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
            }
        }

        /// Inputs for `pack_f16x2`: subnormal, inf and tie edges.
        const BIT_HALVES: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00,
            0x3c00, 0xbc00, 0x3555, 0x0200, 0x7d00,
        ];
        const BIT_F32: [f32; 16] = [
            0.0,
            -0.0,
            1.000_488_3,
            1.001_464_8,
            65520.0,
            2.980_232_2e-8,
            8.940_697e-8,
            1.0,
            -2.5,
            65504.0,
            f32::INFINITY,
            f32::NAN,
            1.0e-8,
            -1.0e-8,
            0.333_333_34,
            3.0e-5,
        ];
        /// Expected `pack_f16x2` bits for `BIT_F32`. 0x7e00 matches any NaN.
        const BIT_F32_F16: [u16; 16] = [
            0x0000, 0x8000, 0x3c00, 0x3c02, 0x7c00, 0x0000, 0x0002, 0x3c00, 0xc100, 0x7bff, 0x7c00,
            0x7e00, 0x0000, 0x8000, 0x3555, 0x01f7,
        ];

        const BITS_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel bits(x: device buffer<u32>, y: device buffer<f32>, z: device buffer<f32>, lo: device mut buffer<f32>, hi: device mut buffer<f32>, packed: device mut buffer<u32>, bf: device mut buffer<f32>, bu: device mut buffer<u32>, bi: device mut buffer<i32>, sb: device mut buffer<i32>, ub: device mut buffer<u32>, id: global_id) {
                lo[id.x] = f16_lo(x[id.x])
                hi[id.x] = f16_hi(x[id.x])
                packed[id.x] = pack_f16x2(y[id.x], z[id.x])
                bf[id.x] = bitcast_f32(x[id.x])
                bu[id.x] = bitcast_u32(y[id.x])
                bi[id.x] = bitcast_i32(x[id.x])
                sb[id.x] = sbyte(x[id.x], id.x)
                ub[id.x] = byte(x[id.x], id.x)
            }
        );

        #[test]
        fn zsl_bit_builtins_match_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = BITS_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = 256usize;
            let xs: Vec<u32> = (0..n)
                .map(|i| {
                    let lo = BIT_HALVES[i % 16];
                    let hi = BIT_HALVES[(i * 7 + 3) % 16];
                    u32::from(lo) | (u32::from(hi) << 16)
                })
                .collect();
            let ys: Vec<f32> = (0..n).map(|i| BIT_F32[i % 16]).collect();
            let zs: Vec<f32> = (0..n).map(|i| BIT_F32[(i * 5 + 1) % 16]).collect();
            let words = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
            let x = pool.alloc(vec![n as u32], DType::U32).unwrap();
            device.write_buffer(x.buffer, 0, &words(&xs)).unwrap();
            let y = upload(&pool, vec![n as u32], &ys);
            let z = upload(&pool, vec![n as u32], &zs);
            let lo = upload(&pool, vec![n as u32], &vec![0.0; n]);
            let hi = upload(&pool, vec![n as u32], &vec![0.0; n]);
            let packed = pool.alloc(vec![n as u32], DType::U32).unwrap();
            let bf = upload(&pool, vec![n as u32], &vec![0.0; n]);
            let bu = pool.alloc(vec![n as u32], DType::U32).unwrap();
            let bi = pool.alloc(vec![n as u32], DType::I32).unwrap();
            let sb = pool.alloc(vec![n as u32], DType::I32).unwrap();
            let ub = pool.alloc(vec![n as u32], DType::U32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    x.buffer.index(),
                    y.buffer.index(),
                    z.buffer.index(),
                    lo.buffer.index(),
                    hi.buffer.index(),
                    packed.buffer.index(),
                    bf.buffer.index(),
                    bu.buffer.index(),
                    bi.buffer.index(),
                    sb.buffer.index(),
                    ub.buffer.index(),
                ],
                &[],
                [(n / 32) as u32, 1, 1],
            );
            ops.run(&*device).unwrap();
            let read =
                |arr: &DeviceArray| device.read_buffer(arr.buffer, 0, arr.size_bytes()).unwrap();
            let as_f32 = |b: Vec<u8>| -> Vec<f32> {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let as_u32 = |b: Vec<u8>| -> Vec<u32> {
                b.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (got_lo, got_hi) = (as_f32(read(&lo)), as_f32(read(&hi)));
            let got_packed = as_u32(read(&packed));
            let got_bf = as_f32(read(&bf));
            let got_bu = as_u32(read(&bu));
            let got_bi = as_u32(read(&bi));
            let got_sb = as_u32(read(&sb));
            let got_ub = as_u32(read(&ub));
            let same_f32 = |got: f32, want: f32| {
                if want.is_nan() {
                    got.is_nan()
                } else {
                    got.to_bits() == want.to_bits()
                }
            };
            let same_f16 = |got: u16, want: u16| {
                if want == 0x7e00 {
                    got & 0x7fff > 0x7c00
                } else {
                    got == want
                }
            };
            for i in 0..n {
                let x_bits = xs[i];
                let (lo_h, hi_h) = ((x_bits & 0xffff) as u16, (x_bits >> 16) as u16);
                assert!(
                    same_f32(got_lo[i], f16_decode(lo_h)),
                    "f16_lo {lo_h:#06x}: {}",
                    got_lo[i]
                );
                assert!(
                    same_f32(got_hi[i], f16_decode(hi_h)),
                    "f16_hi {hi_h:#06x}: {}",
                    got_hi[i]
                );
                let k = i % 16;
                let k2 = (i * 5 + 1) % 16;
                let (g_lo, g_hi) = (
                    (got_packed[i] & 0xffff) as u16,
                    (got_packed[i] >> 16) as u16,
                );
                assert!(
                    same_f16(g_lo, BIT_F32_F16[k]),
                    "pack lo {}: {g_lo:#06x}",
                    BIT_F32[k]
                );
                assert!(
                    same_f16(g_hi, BIT_F32_F16[k2]),
                    "pack hi {}: {g_hi:#06x}",
                    BIT_F32[k2]
                );
                assert_eq!(got_bf[i].to_bits(), x_bits, "bitcast_f32 at {i}");
                assert_eq!(got_bu[i], ys[i].to_bits(), "bitcast_u32 at {i}");
                assert_eq!(got_bi[i], x_bits, "bitcast_i32 at {i}");
                let lane = (i % 4) as u32;
                let byte = (x_bits >> (8 * lane)) & 0xff;
                assert_eq!(got_ub[i], byte, "byte at {i}");
                assert_eq!(got_sb[i], (byte as u8 as i8 as i32) as u32, "sbyte at {i}");
            }
            drop(ops);
            for arr in [x, y, z, lo, hi, bf] {
                pool.free(arr);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Stores `u32x4`, runs a barrier, then loads f16 fragments and runs one mma.
        const VIEW_MMA_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel view_mma(a: device buffer<u32x4>, b: device buffer<u32x4>, c: device buffer<f32>, out: device mut buffer<f32>) {
                shared sa: array<f16, 256>
                shared sb: array<f16, 256>
                view sa4: array<u32x4> = sa
                view sb4: array<u32x4> = sb
                let lane = local_id().x
                sa4[lane] = a[lane]
                sb4[lane] = b[lane]
                barrier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(sa, 0, 16)
                let fb: frag<f16, 16, 16, b> = frag_load_b_t(sb, 0, 16)
                let acc: frag<f32, 16, 16, acc> = frag_load_acc(c, 0, 16)
                let done: frag<f32, 16, 16, acc> = mma(fa, fb, acc)
                frag_store(out, 0, 16, done)
            }
        );

        #[test]
        fn zsl_shared_view_mma_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = VIEW_MMA_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let fa: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let fb: Vec<f32> = (0..256)
                .map(|i| ((i * 5 + 1) % 9 - 4) as f32 / 4.0)
                .collect();
            let fc: Vec<f32> = (0..256).map(|i| ((i * 3) % 11) as f32 - 5.0).collect();
            let bits_a: Vec<u16> = fa.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let bits_b: Vec<u16> = fb.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let a = upload_half(&pool, vec![256], DType::F16, &bits_a);
            let b = upload_half(&pool, vec![256], DType::F16, &bits_b);
            let c = upload(&pool, vec![256], &fc);
            let out = upload(&pool, vec![256], &[0.0; 256]);
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    a.buffer.index(),
                    b.buffer.index(),
                    c.buffer.index(),
                    out.buffer.index(),
                ],
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got = download(&*device, &out);
            for m in 0..16 {
                for n in 0..16 {
                    let mut s = fc[m * 16 + n];
                    for k in 0..16 {
                        s += fa[m * 16 + k] * fb[n * 16 + k];
                    }
                    assert!(
                        close(got[m * 16 + n], s),
                        "({m}, {n}): {} vs {s}",
                        got[m * 16 + n]
                    );
                }
            }
            drop(ops);
            for arr in [a, b, c, out] {
                pool.free(arr);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// One buffer, two params: `x` written as u32x4, `xh` read as f16 after a barrier.
        const ALIAS_BUF_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(32)
            kernel alias_buf(src: device buffer<u32x4>, x: device mut buffer<u32x4>, xh: device buffer<f16>, out: device mut buffer<f32>, id: global_id) {
                x[id.x] = src[id.x]
                barrier()
                for j in 0..8 {
                    out[id.x * 8 + j] = xh[id.x * 8 + j]
                }
            }
        );

        #[test]
        fn zsl_one_buffer_two_element_types_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = ALIAS_BUF_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [32, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let vals: Vec<f32> = (0..256)
                .map(|i| ((i * 7 + 3) % 9 - 4) as f32 / 4.0)
                .collect();
            let bits: Vec<u16> = vals.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let src = upload_half(&pool, vec![256], DType::F16, &bits);
            let x = upload_half(&pool, vec![256], DType::F16, &[0; 256]);
            let out = upload(&pool, vec![256], &[0.0; 256]);
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[
                    src.buffer.index(),
                    x.buffer.index(),
                    x.buffer.index(),
                    out.buffer.index(),
                ],
                &[],
                [1, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got = download(&*device, &out);
            for (h, (&g, &v)) in got.iter().zip(&vals).enumerate() {
                assert_eq!(g, v, "half {h}");
            }
            drop(ops);
            for arr in [src, x, out] {
                pool.free(arr);
            }
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        /// Zetna kernel sources; comparison tests skip without `ZETNA_HIP_DIR`.
        fn zetna_hip_source(files: &[&str]) -> Option<String> {
            let dir = env::var_os("ZETNA_HIP_DIR")?;
            let mut src = String::new();
            for f in files {
                let path = Path::new(&dir).join(f);
                src.push_str(
                    &fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
                );
                src.push('\n');
            }
            Some(src)
        }

        /// Returns the ulp distance between two finite f32 values.
        fn ulp_gap(a: f32, b: f32) -> u32 {
            let key = |x: f32| {
                let b = x.to_bits() as i32;
                if b < 0 { i32::MIN.wrapping_sub(b) } else { b }
            };
            (key(a) as i64 - key(b) as i64).unsigned_abs() as u32
        }

        const GLU_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32, act: u32 }
            @workgroup_size(256)
            kernel glu(gate: device buffer<f32>, up: device buffer<f32>, y: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                if i < p.n {
                    let g = gate[i]
                    let silu = g / (1.0 + fast_exp(-g))
                    let gelu = 0.5 * g * (1.0 + tanh(0.7978845608 * (g + 0.044715 * g * g * g)))
                    y[i] = select(p.act != 0, gelu, silu) * up[i]
                }
            }
        );

        /// Silu and gelu over `n` elements, where `n` is not a block multiple.
        #[test]
        fn zsl_glu_matches_zetna() {
            let Some(device) = hip_device() else { return };
            let Some(src) = zetna_hip_source(&["common.hip", "ops.hip"]) else {
                return;
            };
            let zetna = device
                .create_shader(zengpu_hal::ShaderDesc::hip(&src))
                .unwrap();
            let zetna_pipe = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader: zetna,
                    entry: "glu",
                    block: [256, 1, 1],
                })
                .unwrap();
            let (shader, entry) = GLU_ZSL.create(&*device).unwrap();
            let zsl_pipe = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [256, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let n = (1u32 << 20) + 37;
            let mut seed = 0x2545_f491u32;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f32 / (1u32 << 24) as f32 * 12.0 - 6.0
            };
            let gate: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let up: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let g = upload(&pool, vec![n], &gate);
            let u = upload(&pool, vec![n], &up);
            let y = pool.alloc(vec![n], DType::F32).unwrap();
            let grid = [n.div_ceil(256), 1, 1];
            for act in [0u32, 1] {
                let slots = [g.buffer.index(), u.buffer.index(), y.buffer.index()];
                let scalars = [zengpu_hal::Scalar::U32(n), zengpu_hal::Scalar::U32(act)];
                let mut ops = OpList::new(&pool);
                ops.dispatch(zetna_pipe, &slots, &scalars, grid);
                ops.run(&*device).unwrap();
                let want = download(&*device, &y);
                let mut ops = OpList::new(&pool);
                ops.dispatch(zsl_pipe, &slots, &scalars, grid);
                ops.run(&*device).unwrap();
                let got = download(&*device, &y);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| ulp_gap(a, b))
                    .max()
                    .unwrap();
                println!("glu act {act}: max ulp vs zetna {worst}");
                assert!(worst <= 1, "glu act {act}: {worst} ulp");
            }
            device.destroy_pipeline(zetna_pipe);
            device.destroy_pipeline(zsl_pipe);
            device.destroy_shader(zetna);
            device.destroy_shader(shader);
        }

        const ROUND_EVEN_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            @workgroup_size(64)
            kernel round_even(src: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                let i = id.x
                out[i] = round_even(src[i])
            }
        );

        /// Rounds ties to even, as `__float2int_rn` does.
        #[test]
        fn zsl_round_even_matches_cpu() {
            let Some(device) = hip_device() else { return };
            let (shader, entry) = ROUND_EVEN_ZSL.create(&*device).unwrap();
            let pipeline = device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry,
                    block: [64, 1, 1],
                })
                .unwrap();
            let pool = BufferPool::new(device.clone());
            let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.25 + 1e-3).collect();
            let src = upload(&pool, vec![128], &input);
            let out = pool.alloc(vec![128], DType::F32).unwrap();
            let mut ops = OpList::new(&pool);
            ops.dispatch(
                pipeline,
                &[src.buffer.index(), out.buffer.index()],
                &[],
                [2, 1, 1],
            );
            ops.run(&*device).unwrap();
            let got = download(&*device, &out);
            for (i, (&g, &x)) in got.iter().zip(&input).enumerate() {
                assert_eq!(g, x.round_ties_even(), "at {i}: {x}");
            }
            drop(ops);
            pool.free(src);
            pool.free(out);
            device.destroy_pipeline(pipeline);
            device.destroy_shader(shader);
        }

        const QUANT_Q8_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { n: u32 }
            @workgroup_size(256)
            kernel quantize_q8(x: device buffer<f32>, xq: device mut buffer<i8>, xs: device mut buffer<f32>, p: P, id: global_id) {
                let i = id.x
                let v = x[i]
                let amax = subgroup_reduce_max(abs(v))
                let d = amax / 127.0
                let qi = i32(select(d > 0.0, round_even(v / d), 0.0))
                let qsum = subgroup_reduce_add(qi)
                xq[i] = qi
                if i % 32 == 0 {
                    xs[p.n / 4 + 2 * (i / 32)] = d
                    xs[p.n / 4 + 2 * (i / 32) + 1] = d * f32(qsum)
                }
            }
        );

        fn q4k_weights(seed: &mut u64, rows: usize, cols: usize) -> Vec<u8> {
            let mut next = || {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                *seed
            };
            let mut w = vec![0u8; rows * cols / 256 * 144];
            for blk in w.chunks_exact_mut(144) {
                let d = 0x1C00u16 | (next() as u16 & 0x3F);
                let dmin = 0x1C00u16 | (next() as u16 & 0x3F);
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                blk[2..4].copy_from_slice(&dmin.to_le_bytes());
                for byte in &mut blk[4..] {
                    *byte = next() as u8;
                }
            }
            w
        }

        fn hip_pipeline(
            device: &dyn zengpu_hal::GpuDevice,
            shader: zengpu_hal::ShaderHandle,
            entry: &str,
            block: u32,
        ) -> zengpu_hal::PipelineHandle {
            device
                .create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
                    shader,
                    entry: entry.to_string().leak(),
                    block: [block, 1, 1],
                })
                .unwrap()
        }

        fn words_buffer(pool: &BufferPool, bytes: &[u8]) -> DeviceArray {
            let arr = pool
                .alloc(vec![(bytes.len() / 4) as u32], DType::U32)
                .unwrap();
            pool.device().write_buffer(arr.buffer, 0, bytes).unwrap();
            arr
        }

        #[test]
        fn zsl_gemv_q8_q4k_matches_zetna() {
            let Some(device) = hip_device() else { return };
            let Some(src) = zetna_hip_source(&["common.hip", "quant.hip"]) else {
                return;
            };
            let zetna = device
                .create_shader(zengpu_hal::ShaderDesc::hip(&src))
                .unwrap();
            let z_quant = hip_pipeline(&*device, zetna, "quantize_q8", 256);
            let z_gemv = hip_pipeline(&*device, zetna, "gemv_q8_1_q4_k", 256);
            let (qs, qe) = QUANT_Q8_ZSL.create(&*device).unwrap();
            let s_quant = hip_pipeline(&*device, qs, qe, 256);
            let (gs, ge) = GEMV_Q4K_Q8_ZSL.create(&*device).unwrap();
            let s_gemv = hip_pipeline(&*device, gs, ge, 256);
            let pool = BufferPool::new(device.clone());
            let cols = 5120usize;
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i * 2_654_435_761usize) % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let x_buf = upload(&pool, vec![cols as u32], &x);
            let act_bytes = cols + cols / 32 * 8;
            let act = pool
                .alloc(vec![(act_bytes / 4) as u32], DType::U32)
                .unwrap();
            for rows in [5120usize, 17408] {
                let w_bytes = q4k_weights(&mut seed, rows, cols);
                let w = words_buffer(&pool, &w_bytes);
                let y_z = pool.alloc(vec![rows as u32], DType::F32).unwrap();
                let y_s = pool.alloc(vec![rows as u32], DType::F32).unwrap();
                let quant_grid = [(cols / 256) as u32, 1, 1];
                let gemv_grid = [(rows / 16) as u32, 1, 1];
                let zslots = [x_buf.buffer.index(), act.buffer.index()];
                let mut ops = OpList::new(&pool);
                ops.dispatch(
                    z_quant,
                    &zslots,
                    &[
                        zengpu_hal::Scalar::U32(1),
                        zengpu_hal::Scalar::U32(cols as u32),
                    ],
                    quant_grid,
                );
                ops.run(&*device).unwrap();
                let zq_bytes = device.read_buffer(act.buffer, 0, act_bytes as u64).unwrap();
                let zslots = [
                    w.buffer.index(),
                    w.buffer.index(),
                    w.buffer.index(),
                    act.buffer.index(),
                    y_z.buffer.index(),
                    y_z.buffer.index(),
                    y_z.buffer.index(),
                ];
                let zscal = [
                    zengpu_hal::Scalar::U32(rows as u32),
                    zengpu_hal::Scalar::U32(0),
                    zengpu_hal::Scalar::U32(0),
                    zengpu_hal::Scalar::U32(1),
                    zengpu_hal::Scalar::U32(cols as u32),
                ];
                let mut ops = OpList::new(&pool);
                ops.dispatch(z_gemv, &zslots, &zscal, gemv_grid);
                ops.run(&*device).unwrap();
                let want = download(&*device, &y_z);
                let mut ops = OpList::new(&pool);
                ops.dispatch(
                    s_quant,
                    &zslots[..0]
                        .iter()
                        .copied()
                        .chain([x_buf.buffer.index(), act.buffer.index(), act.buffer.index()])
                        .collect::<Vec<u32>>(),
                    &[zengpu_hal::Scalar::U32(cols as u32)],
                    quant_grid,
                );
                ops.run(&*device).unwrap();
                let s_q_bytes = device.read_buffer(act.buffer, 0, act_bytes as u64).unwrap();
                let mismatch = zq_bytes[..cols]
                    .iter()
                    .zip(&s_q_bytes[..cols])
                    .filter(|(a, b)| a != b)
                    .count();
                let scale_gap = zq_bytes[cols..]
                    .chunks_exact(4)
                    .zip(s_q_bytes[cols..].chunks_exact(4))
                    .map(|(a, b)| {
                        ulp_gap(
                            f32::from_le_bytes(a.try_into().unwrap()),
                            f32::from_le_bytes(b.try_into().unwrap()),
                        )
                    })
                    .max()
                    .unwrap_or(0);
                println!(
                    "quantize cols {cols}: {mismatch} int8 mismatches, scale max ulp {scale_gap}"
                );
                let gslots = [
                    w.buffer.index(),
                    act.buffer.index(),
                    act.buffer.index(),
                    y_s.buffer.index(),
                ];
                let sscal = [
                    zengpu_hal::Scalar::U32(rows as u32),
                    zengpu_hal::Scalar::U32((cols / 256) as u32),
                    zengpu_hal::Scalar::U32((cols / 64) as u32),
                    zengpu_hal::Scalar::U32(((cols / 64) as u32).div_ceil(32)),
                    zengpu_hal::Scalar::U32((cols / 4) as u32),
                ];
                let mut ops = OpList::new(&pool);
                ops.dispatch(s_gemv, &gslots, &sscal, gemv_grid);
                ops.run(&*device).unwrap();
                let got = download(&*device, &y_s);
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                    / scale;
                println!(
                    "gemv rows {rows}: max abs y {scale:e}, worst |zsl - zetna| / max|y| = {worst:e}"
                );
                assert!(worst < 1e-4, "gemv rows {rows}: {worst}");
                drop(ops);
                for b in [w, y_z, y_s] {
                    pool.free(b);
                }
            }
            pool.free(act);
            pool.free(x_buf);
            for p in [z_quant, z_gemv, s_quant, s_gemv] {
                device.destroy_pipeline(p);
            }
            device.destroy_shader(zetna);
            device.destroy_shader(qs);
            device.destroy_shader(gs);
        }

        const GEMV_Q4K_Q8_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { rows: u32, nb: u32, units: u32, iters: u32, xbase: u32 }
            @workgroup_size(256)
            kernel gemv_q8_q4k(w: device buffer<u32x4>, xq: device buffer<u32x4>, xs: device buffer<f32>, y: device mut buffer<f32>, p: P, id: global_id) {
                let lane = id.x % 32
                let r0 = id.x / 32 * 2
                let r1 = r0 + 1
                let rb0 = r0 * p.nb * 9
                let rb1 = r1 * p.nb * 9
                let accA = 0.0
                let accB = 0.0
                for it in 0..p.iters {
                    let u = lane + it * 32
                    if u < p.units {
                    let blk = (u / 4) * 9
                    let qq = u % 4
                    let x0 = xq[4 * u]
                    let x1 = xq[4 * u + 1]
                    let x2 = xq[4 * u + 2]
                    let x3 = xq[4 * u + 3]
                    let dx = xs[p.xbase + 4 * u]
                    let dy = xs[p.xbase + 4 * u + 1]
                    let dz = xs[p.xbase + 4 * u + 2]
                    let dw = xs[p.xbase + 4 * u + 3]
                    let hA = w[rb0 + blk]
                    let aA = w[rb0 + blk + 1 + 2 * qq]
                    let bA = w[rb0 + blk + 2 + 2 * qq]
                    let jA0 = 2 * qq + 0
                    let jA0p = jA0 + 4
                    let jA0m = jA0 - 4
                    let tA0a = byte(select((jA0) < 4, hA.y, select((jA0) < 8, hA.z, hA.w)), (jA0) % 4)
                    let tA0b = byte(select((jA0p) < 4, hA.y, select((jA0p) < 8, hA.z, hA.w)), (jA0p) % 4)
                    let tA0c = byte(select((jA0m) < 4, hA.y, select((jA0m) < 8, hA.z, hA.w)), (jA0m) % 4)
                    let scA0 = select(jA0 < 4, tA0a & 63, (tA0b & 15) | ((tA0c >> 6) << 4))
                    let mA0 = select(jA0 < 4, tA0b & 63, (tA0b >> 4) | ((tA0a >> 6) << 4))
                    let jA1 = 2 * qq + 1
                    let jA1p = jA1 + 4
                    let jA1m = jA1 - 4
                    let tA1a = byte(select((jA1) < 4, hA.y, select((jA1) < 8, hA.z, hA.w)), (jA1) % 4)
                    let tA1b = byte(select((jA1p) < 4, hA.y, select((jA1p) < 8, hA.z, hA.w)), (jA1p) % 4)
                    let tA1c = byte(select((jA1m) < 4, hA.y, select((jA1m) < 8, hA.z, hA.w)), (jA1m) % 4)
                    let scA1 = select(jA1 < 4, tA1a & 63, (tA1b & 15) | ((tA1c >> 6) << 4))
                    let mA1 = select(jA1 < 4, tA1b & 63, (tA1b >> 4) | ((tA1a >> 6) << 4))
                    let dA = f16_lo(hA.x)
                    let dmA = f16_hi(hA.x)
                    let loA0 = bitcast_i32(aA.x & 0x0F0F0F0F)
                    let hiA0 = bitcast_i32((aA.x >> 4) & 0x0F0F0F0F)
                    let loA1 = bitcast_i32(aA.y & 0x0F0F0F0F)
                    let hiA1 = bitcast_i32((aA.y >> 4) & 0x0F0F0F0F)
                    let loA2 = bitcast_i32(aA.z & 0x0F0F0F0F)
                    let hiA2 = bitcast_i32((aA.z >> 4) & 0x0F0F0F0F)
                    let loA3 = bitcast_i32(aA.w & 0x0F0F0F0F)
                    let hiA3 = bitcast_i32((aA.w >> 4) & 0x0F0F0F0F)
                    let loA4 = bitcast_i32(bA.x & 0x0F0F0F0F)
                    let hiA4 = bitcast_i32((bA.x >> 4) & 0x0F0F0F0F)
                    let loA5 = bitcast_i32(bA.y & 0x0F0F0F0F)
                    let hiA5 = bitcast_i32((bA.y >> 4) & 0x0F0F0F0F)
                    let loA6 = bitcast_i32(bA.z & 0x0F0F0F0F)
                    let hiA6 = bitcast_i32((bA.z >> 4) & 0x0F0F0F0F)
                    let loA7 = bitcast_i32(bA.w & 0x0F0F0F0F)
                    let hiA7 = bitcast_i32((bA.w >> 4) & 0x0F0F0F0F)
                    let pA0 = dot4_i8(loA0, bitcast_i32(x0.x), 0)
                    let pA1 = dot4_i8(loA1, bitcast_i32(x0.y), pA0)
                    let pA2 = dot4_i8(loA2, bitcast_i32(x0.z), pA1)
                    let pA3 = dot4_i8(loA3, bitcast_i32(x0.w), pA2)
                    let pA4 = dot4_i8(loA4, bitcast_i32(x1.x), pA3)
                    let pA5 = dot4_i8(loA5, bitcast_i32(x1.y), pA4)
                    let pA6 = dot4_i8(loA6, bitcast_i32(x1.z), pA5)
                    let pA7 = dot4_i8(loA7, bitcast_i32(x1.w), pA6)
                    let qA0 = dot4_i8(hiA0, bitcast_i32(x2.x), 0)
                    let qA1 = dot4_i8(hiA1, bitcast_i32(x2.y), qA0)
                    let qA2 = dot4_i8(hiA2, bitcast_i32(x2.z), qA1)
                    let qA3 = dot4_i8(hiA3, bitcast_i32(x2.w), qA2)
                    let qA4 = dot4_i8(hiA4, bitcast_i32(x3.x), qA3)
                    let qA5 = dot4_i8(hiA5, bitcast_i32(x3.y), qA4)
                    let qA6 = dot4_i8(hiA6, bitcast_i32(x3.z), qA5)
                    let qA7 = dot4_i8(hiA7, bitcast_i32(x3.w), qA6)
                    let addA = dA * scA0 * dx * f32(pA7) + dA * scA1 * dz * f32(qA7) - (dmA * mA0 * dy + dmA * mA1 * dw)
                    accA = accA + addA
                    let hB = w[rb1 + blk]
                    let aB = w[rb1 + blk + 1 + 2 * qq]
                    let bB = w[rb1 + blk + 2 + 2 * qq]
                    let jB0 = 2 * qq + 0
                    let jB0p = jB0 + 4
                    let jB0m = jB0 - 4
                    let tB0a = byte(select((jB0) < 4, hB.y, select((jB0) < 8, hB.z, hB.w)), (jB0) % 4)
                    let tB0b = byte(select((jB0p) < 4, hB.y, select((jB0p) < 8, hB.z, hB.w)), (jB0p) % 4)
                    let tB0c = byte(select((jB0m) < 4, hB.y, select((jB0m) < 8, hB.z, hB.w)), (jB0m) % 4)
                    let scB0 = select(jB0 < 4, tB0a & 63, (tB0b & 15) | ((tB0c >> 6) << 4))
                    let mB0 = select(jB0 < 4, tB0b & 63, (tB0b >> 4) | ((tB0a >> 6) << 4))
                    let jB1 = 2 * qq + 1
                    let jB1p = jB1 + 4
                    let jB1m = jB1 - 4
                    let tB1a = byte(select((jB1) < 4, hB.y, select((jB1) < 8, hB.z, hB.w)), (jB1) % 4)
                    let tB1b = byte(select((jB1p) < 4, hB.y, select((jB1p) < 8, hB.z, hB.w)), (jB1p) % 4)
                    let tB1c = byte(select((jB1m) < 4, hB.y, select((jB1m) < 8, hB.z, hB.w)), (jB1m) % 4)
                    let scB1 = select(jB1 < 4, tB1a & 63, (tB1b & 15) | ((tB1c >> 6) << 4))
                    let mB1 = select(jB1 < 4, tB1b & 63, (tB1b >> 4) | ((tB1a >> 6) << 4))
                    let dB = f16_lo(hB.x)
                    let dmB = f16_hi(hB.x)
                    let loB0 = bitcast_i32(aB.x & 0x0F0F0F0F)
                    let hiB0 = bitcast_i32((aB.x >> 4) & 0x0F0F0F0F)
                    let loB1 = bitcast_i32(aB.y & 0x0F0F0F0F)
                    let hiB1 = bitcast_i32((aB.y >> 4) & 0x0F0F0F0F)
                    let loB2 = bitcast_i32(aB.z & 0x0F0F0F0F)
                    let hiB2 = bitcast_i32((aB.z >> 4) & 0x0F0F0F0F)
                    let loB3 = bitcast_i32(aB.w & 0x0F0F0F0F)
                    let hiB3 = bitcast_i32((aB.w >> 4) & 0x0F0F0F0F)
                    let loB4 = bitcast_i32(bB.x & 0x0F0F0F0F)
                    let hiB4 = bitcast_i32((bB.x >> 4) & 0x0F0F0F0F)
                    let loB5 = bitcast_i32(bB.y & 0x0F0F0F0F)
                    let hiB5 = bitcast_i32((bB.y >> 4) & 0x0F0F0F0F)
                    let loB6 = bitcast_i32(bB.z & 0x0F0F0F0F)
                    let hiB6 = bitcast_i32((bB.z >> 4) & 0x0F0F0F0F)
                    let loB7 = bitcast_i32(bB.w & 0x0F0F0F0F)
                    let hiB7 = bitcast_i32((bB.w >> 4) & 0x0F0F0F0F)
                    let pB0 = dot4_i8(loB0, bitcast_i32(x0.x), 0)
                    let pB1 = dot4_i8(loB1, bitcast_i32(x0.y), pB0)
                    let pB2 = dot4_i8(loB2, bitcast_i32(x0.z), pB1)
                    let pB3 = dot4_i8(loB3, bitcast_i32(x0.w), pB2)
                    let pB4 = dot4_i8(loB4, bitcast_i32(x1.x), pB3)
                    let pB5 = dot4_i8(loB5, bitcast_i32(x1.y), pB4)
                    let pB6 = dot4_i8(loB6, bitcast_i32(x1.z), pB5)
                    let pB7 = dot4_i8(loB7, bitcast_i32(x1.w), pB6)
                    let qB0 = dot4_i8(hiB0, bitcast_i32(x2.x), 0)
                    let qB1 = dot4_i8(hiB1, bitcast_i32(x2.y), qB0)
                    let qB2 = dot4_i8(hiB2, bitcast_i32(x2.z), qB1)
                    let qB3 = dot4_i8(hiB3, bitcast_i32(x2.w), qB2)
                    let qB4 = dot4_i8(hiB4, bitcast_i32(x3.x), qB3)
                    let qB5 = dot4_i8(hiB5, bitcast_i32(x3.y), qB4)
                    let qB6 = dot4_i8(hiB6, bitcast_i32(x3.z), qB5)
                    let qB7 = dot4_i8(hiB7, bitcast_i32(x3.w), qB6)
                    let addB = dB * scB0 * dx * f32(pB7) + dB * scB1 * dz * f32(qB7) - (dmB * mB0 * dy + dmB * mB1 * dw)
                    accB = accB + addB
                    }
                }
                let tA = subgroup_reduce_add(accA)
                let tB = subgroup_reduce_add(accB)
                if lane == 0 {
                    y[r0] = tA
                    y[r1] = tB
                }
            }
        );

        const NORM_ROPE_ZSL: zengpu_spirv::ZslShader = zengpu_spirv::zsl!(
            push P { d: u32, heads: u32, rope: u32, dk: u32, base: f32, eps: f32, flags: u32 }
            @workgroup_size(128)
            kernel norm_rope(x: device mut buffer<f32>, w: device buffer<f32>, factors: device buffer<f32>, meta: device buffer<u32>, p: P, id: global_id) {
                shared t: array<f32, 512>
                let b = id.x / 128
                let tid = id.x % 128
                let row = b / p.heads
                let pos = f32(meta[2 * row])
                let hb = b * p.d
                let ss = 0.0
                for k in 0..p.dk {
                    let v = x[hb + tid + k * 128]
                    t[tid + k * 128] = v
                    ss = ss + v * v
                }
                let inv = rsqrt(workgroup_reduce_add(ss) / f32(p.d) + p.eps)
                for k in 0..p.dk {
                    let i = tid + k * 128
                    t[i] = t[i] * select((p.flags & 1) != 0, inv * w[i], inv)
                }
                barrier()
                let half = p.rope / 2
                for k in 0..p.dk {
                    let i = tid + k * 128
                    let pair = i % half
                    let theta0 = pos * pow(p.base, -2.0 * f32(pair) / f32(p.rope))
                    let theta = select((p.flags & 2) != 0, theta0 / factors[pair], theta0)
                    let c = cos(theta)
                    let s = sin(theta)
                    let rotated = select(i < half, t[i] * c - t[i + half] * s, t[i - half] * s + t[i] * c)
                    x[hb + i] = select(i < p.rope, rotated, t[i])
                }
            }
        );

        #[test]
        fn zsl_norm_rope_matches_zetna() {
            let Some(device) = hip_device() else { return };
            let Some(src) = zetna_hip_source(&["common.hip", "ops.hip"]) else {
                return;
            };
            let zetna = device
                .create_shader(zengpu_hal::ShaderDesc::hip(&src))
                .unwrap();
            let z_pipe = hip_pipeline(&*device, zetna, "norm_rope", 128);
            let (ns, ne) = NORM_ROPE_ZSL.create(&*device).unwrap();
            let s_pipe = hip_pipeline(&*device, ns, ne, 128);
            let pool = BufferPool::new(device.clone());
            let (rows, heads, d) = (3usize, 4usize, 128usize);
            let x: Vec<f32> = (0..rows * heads * d)
                .map(|i| ((i * 2_654_435_761usize) % 4001) as f32 / 1000.0 - 2.0)
                .collect();
            let wt: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
            let factors: Vec<f32> = (0..d / 2).map(|i| 1.0 + (i % 5) as f32 / 4.0).collect();
            let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let xb = pool.alloc(vec![x.len() as u32], DType::F32).unwrap();
            let wb = upload(&pool, vec![d as u32], &wt);
            let fb = upload(&pool, vec![(d / 2) as u32], &factors);
            let meta: Vec<u32> = (0..rows)
                .flat_map(|r| [17 + 5 * r as u32, r as u32])
                .collect();
            let meta_bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mb = pool.alloc(vec![meta.len() as u32], DType::U32).unwrap();
            device.write_buffer(mb.buffer, 0, &meta_bytes).unwrap();
            let slots = [
                xb.buffer.index(),
                wb.buffer.index(),
                fb.buffer.index(),
                mb.buffer.index(),
            ];
            let grid = [(rows * heads) as u32, 1, 1];
            for rope in [128usize, 64] {
                let scal = |dk: usize| {
                    [
                        zengpu_hal::Scalar::U32(d as u32),
                        zengpu_hal::Scalar::U32(heads as u32),
                        zengpu_hal::Scalar::U32(rope as u32),
                        zengpu_hal::Scalar::U32(dk as u32),
                        zengpu_hal::Scalar::F32(10000.0),
                        zengpu_hal::Scalar::F32(1e-6),
                        zengpu_hal::Scalar::U32(3),
                    ]
                };
                let zscal = [
                    zengpu_hal::Scalar::U32(d as u32),
                    zengpu_hal::Scalar::U32(heads as u32),
                    zengpu_hal::Scalar::U32(rope as u32),
                    zengpu_hal::Scalar::F32(10000.0),
                    zengpu_hal::Scalar::F32(1e-6),
                    zengpu_hal::Scalar::U32(3),
                ];
                device.write_buffer(xb.buffer, 0, &x_bytes).unwrap();
                let mut ops = OpList::new(&pool);
                ops.dispatch(z_pipe, &slots, &zscal, grid);
                ops.run(&*device).unwrap();
                let want = download(&*device, &xb);
                device.write_buffer(xb.buffer, 0, &x_bytes).unwrap();
                let mut ops = OpList::new(&pool);
                ops.dispatch(s_pipe, &slots, &scal(d / 128), grid);
                ops.run(&*device).unwrap();
                let got = download(&*device, &xb);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| ulp_gap(a, b))
                    .max()
                    .unwrap();
                println!("norm_rope rope {rope}: max ulp vs zetna {worst}");
                assert!(worst <= 2, "norm_rope rope {rope}: {worst} ulp");
            }
            device.destroy_pipeline(z_pipe);
            device.destroy_pipeline(s_pipe);
            device.destroy_shader(zetna);
            device.destroy_shader(ns);
        }
    }
}
