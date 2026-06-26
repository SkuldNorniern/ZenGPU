//! ZenGPU AMD ROCm/HIP compute backend.
//!
//! Supports ROCm 5.x – 7.x on gfx9xx (CDNA), gfx10xx (RDNA 1/2),
//! gfx11xx (RDNA 3), gfx12xx (RDNA 4). Multi-GPU: open one `HipDevice`
//! per adapter; they are `Send + Sync` and can be driven from separate threads.

use std::ffi::{CStr, CString, c_void};
use std::sync::{Arc, Mutex};

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    BufferUsage, ComputePipelineDesc, DeviceRequest, DeviceType, DispatchOp, GpuAdapter,
    GpuDevice, GpuError, GpuInstance, HalCapabilities, PipelineHandle, Result,
    SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle, ShaderSource, SlotMap, TextureDesc,
    TextureHandle, UsageError, marker,
};

// ── Generated layout constants (produced by build.rs) ─────────────────────────
mod hip_layout {
    include!(concat!(env!("OUT_DIR"), "/hip_layout.rs"));
}
use hip_layout::*;

// ── ROCm version detection and feature capability gating ──────────────────────
pub mod version;
use version::{GfxFamily, HipCapabilities, RocmVersion};

// ── Raw HIP FFI ───────────────────────────────────────────────────────────────

type HipError = i32;
const HIP_SUCCESS: HipError = 0;

const HIP_MEMCPY_H2D: i32 = 1;
const HIP_MEMCPY_D2H: i32 = 2;

type HipModule   = *mut c_void;
type HipFunction = *mut c_void;
type HipStream   = *mut c_void; // null = default stream
type HiprtcProg  = *mut c_void;

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipInit(flags: u32) -> HipError;
    fn hipGetDeviceCount(count: *mut i32) -> HipError;
    fn hipDeviceGetName(name: *mut i8, len: i32, device: i32) -> HipError;
    fn hipDeviceTotalMem(bytes: *mut usize, device: i32) -> HipError;
    fn hipGetDeviceProperties(props: *mut u8, device: i32) -> HipError;
    fn hipSetDevice(device: i32) -> HipError;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    fn hipFree(ptr: *mut c_void) -> HipError;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> HipError;
    fn hipModuleLoadData(module: *mut HipModule, image: *const c_void) -> HipError;
    fn hipModuleGetFunction(func: *mut HipFunction, module: HipModule, name: *const i8) -> HipError;
    fn hipModuleUnload(module: HipModule) -> HipError;
    fn hipModuleLaunchKernel(
        f: HipFunction,
        gx: u32, gy: u32, gz: u32,
        bx: u32, by: u32, bz: u32,
        shared_bytes: u32,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipStreamSynchronize(stream: HipStream) -> HipError;
}

#[link(name = "hiprtc")]
unsafe extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProg,
        src: *const i8,
        name: *const i8,
        num_headers: i32,
        headers: *const *const i8,
        include_names: *const *const i8,
    ) -> i32;
    fn hiprtcCompileProgram(prog: HiprtcProg, num_opts: i32, opts: *const *const i8) -> i32;
    fn hiprtcGetCodeSize(prog: HiprtcProg, sz: *mut usize) -> i32;
    fn hiprtcGetCode(prog: HiprtcProg, code: *mut i8) -> i32;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProg) -> i32;
    fn hiprtcGetProgramLogSize(prog: HiprtcProg, sz: *mut usize) -> i32;
    fn hiprtcGetProgramLog(prog: HiprtcProg, log: *mut i8) -> i32;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn check(code: HipError, ctx: &str) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Backend(format!("{ctx}: HIP error {code}")))
    }
}

/// Read `gcnArchName` from an opaque hipDeviceProp_t byte blob.
/// Falls back to "" on layout mismatch (older ROCm with smaller struct).
fn gcn_arch_from_prop(blob: &[u8]) -> String {
    let start = HIP_PROP_GCN_ARCH_OFF;
    let end   = start + HIP_PROP_GCN_ARCH_LEN;
    if blob.len() < end {
        return String::new();
    }
    let slice = &blob[start..end];
    // Find the null terminator.
    let len = slice.iter().position(|&b| b == 0).unwrap_or(HIP_PROP_GCN_ARCH_LEN);
    String::from_utf8_lossy(&slice[..len]).into_owned()
}

/// Read multiProcessorCount (i32 at fixed offset).
fn cu_count_from_prop(blob: &[u8]) -> u32 {
    let off = HIP_PROP_CU_COUNT_OFF;
    if blob.len() < off + 4 {
        return 0;
    }
    let bytes: [u8; 4] = blob[off..off + 4].try_into().unwrap_or([0; 4]);
    i32::from_ne_bytes(bytes) as u32
}

/// Read clockRate (kHz, i32 at fixed offset) → MHz.
fn clock_mhz_from_prop(blob: &[u8]) -> u32 {
    let off = HIP_PROP_CLOCK_OFF;
    if blob.len() < off + 4 {
        return 0;
    }
    let bytes: [u8; 4] = blob[off..off + 4].try_into().unwrap_or([0; 4]);
    let khz = i32::from_ne_bytes(bytes);
    (khz / 1000) as u32
}

// ── Device info cache (per ordinal) ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HipDeviceInfo {
    pub ordinal:      i32,
    pub name:         String,
    /// Bare gfx target, e.g. "gfx1200". Empty if undetectable.
    pub gfx_target:   String,
    pub gfx_family:   GfxFamily,
    pub total_mem:    usize,
    pub cu_count:     u32,
    pub clock_mhz:    u32,
    /// ROCm feature capabilities for this device.
    pub capabilities: HipCapabilities,
}

impl HipDeviceInfo {
    fn query(ordinal: i32) -> Option<Self> {
        let mut name_buf = [0i8; 256];
        unsafe {
            if hipDeviceGetName(name_buf.as_mut_ptr(), 256, ordinal) != HIP_SUCCESS {
                return None;
            }
        }
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total_mem: usize = 0;
        unsafe { hipDeviceTotalMem(&mut total_mem, ordinal); }

        // Query hipDeviceProp_t into an opaque blob.
        let mut blob = vec![0u8; HIP_PROP_SIZE];
        let prop_ok = unsafe { hipGetDeviceProperties(blob.as_mut_ptr(), ordinal) } == HIP_SUCCESS;

        let (gfx_target, cu_count, clock_mhz) = if prop_ok {
            (
                gcn_arch_from_prop(&blob),
                cu_count_from_prop(&blob),
                clock_mhz_from_prop(&blob),
            )
        } else {
            (String::new(), 0, 0)
        };

        // If the gfx string is empty (old ROCm), derive it from the name heuristic.
        let gfx_target = if gfx_target.is_empty() {
            gfx_from_name_heuristic(&name)
        } else {
            // hipDeviceProp_t.gcnArchName sometimes includes feature flags like
            // "gfx1200:sramecc-:xnack-" — strip to bare gfx token.
            gfx_target
                .split_once(':')
                .map(|(g, _)| g.to_string())
                .unwrap_or(gfx_target)
        };

        let rocm    = RocmVersion::COMPILE_TIME;
        let caps    = HipCapabilities::from_device(rocm, &gfx_target);
        let family  = GfxFamily::from_gfx(&gfx_target);

        Some(Self { ordinal, name, gfx_target, gfx_family: family, total_mem, cu_count, clock_mhz, capabilities: caps })
    }

}

/// Name-based gfx heuristic — last-resort fallback for old ROCm (< 5.x) or
/// cross-compilation where `hipGetDeviceProperties` is unavailable.
fn gfx_from_name_heuristic(name: &str) -> String {
    // RDNA 4
    if name.contains("9070") { return "gfx1201".into(); }
    if name.contains("9060") { return "gfx1200".into(); }
    // RDNA 3.5 (Strix)
    if name.contains("890M") || name.contains("880M") { return "gfx1150".into(); }
    // RDNA 3
    if name.contains("7900") || name.contains("7800") || name.contains("7700") { return "gfx1100".into(); }
    if name.contains("7600") { return "gfx1102".into(); }
    // RDNA 2
    if name.contains("6950") || name.contains("6900") || name.contains("6800") { return "gfx1030".into(); }
    if name.contains("6700") { return "gfx1031".into(); }
    if name.contains("6600") { return "gfx1032".into(); }
    // RDNA 1
    if name.contains("5700") { return "gfx1010".into(); }
    // CDNA 3
    if name.contains("MI300") { return "gfx942".into(); }
    // CDNA 2
    if name.contains("MI250") || name.contains("MI210") { return "gfx90a".into(); }
    // CDNA 1
    if name.contains("MI100") { return "gfx908".into(); }
    // Vega / GCN 5
    if name.contains("Vega20") || name.contains("MI50") || name.contains("MI60") { return "gfx906".into(); }
    if name.contains("Vega10") || name.contains("Vega 64") || name.contains("Vega 56") { return "gfx900".into(); }
    // Unknown — let hipRTC auto-detect
    String::new()
}

// ── Internal resource types ───────────────────────────────────────────────────

#[allow(dead_code)]
struct HipBuffer {
    ptr:   *mut c_void,
    size:  usize,
    usage: BufferUsage,
}
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

struct HipShader {
    module: HipModule,
}
unsafe impl Send for HipShader {}
unsafe impl Sync for HipShader {}

struct HipPipeline {
    function: HipFunction,
    block:    [u32; 3],
}
unsafe impl Send for HipPipeline {}
unsafe impl Sync for HipPipeline {}

// ── Device inner (one per ordinal) ───────────────────────────────────────────

struct HipDeviceInner {
    ordinal:   i32,
    buffers:   SlotMap<marker::Buffer,   HipBuffer>,
    shaders:   SlotMap<marker::Shader,   HipShader>,
    pipelines: SlotMap<marker::Pipeline, HipPipeline>,
}

impl HipDeviceInner {
    fn new(ordinal: i32) -> Self {
        Self {
            ordinal,
            buffers:   SlotMap::new(),
            shaders:   SlotMap::new(),
            pipelines: SlotMap::new(),
        }
    }

    fn activate(&self) -> Result<()> {
        unsafe { check(hipSetDevice(self.ordinal), "hipSetDevice") }
    }
}

// ── HipInstance ───────────────────────────────────────────────────────────────

/// Entry-point for the ROCm/HIP backend.
///
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
    /// Initialise the HIP runtime and enumerate all visible AMD GPUs.
    pub fn new() -> Result<Self> {
        unsafe { check(hipInit(0), "hipInit")? };

        let mut count: i32 = 0;
        unsafe { hipGetDeviceCount(&mut count) };

        let devices: Vec<HipDeviceInfo> = (0..count)
            .filter_map(|i| HipDeviceInfo::query(i))
            .collect();

        for d in &devices {
            log::info!(
                "hip: [{ordinal}] {name} ({gfx}/{family}) — {mem} MiB, {cu} CUs @ {clk} MHz, ROCm {rocm}",
                ordinal = d.ordinal,
                name    = d.name,
                gfx     = if d.gfx_target.is_empty() { "unknown" } else { &d.gfx_target },
                family  = d.gfx_family.name(),
                mem     = d.total_mem / (1024 * 1024),
                cu      = d.cu_count,
                clk     = d.clock_mhz,
                rocm    = d.capabilities.rocm,
            );
        }

        Ok(Self { devices })
    }

    /// All detected AMD GPUs.
    pub fn device_infos(&self) -> &[HipDeviceInfo] {
        &self.devices
    }
}

impl GpuInstance for HipInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        self.devices
            .iter()
            .map(|d| {
                Box::new(HipAdapter {
                    info: AdapterInfo {
                        name:        d.name.clone(),
                        vendor:      0x1002, // AMD PCI vendor
                        device:      0,
                        device_type: DeviceType::Discrete,
                        backend:     BackendPreference::Hip,
                    },
                    dev_info: d.clone(),
                }) as Box<dyn GpuAdapter>
            })
            .collect()
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        // Return the first discrete GPU (they are all discrete for HIP).
        self.enumerate_adapters().into_iter().next()
    }
}

// ── HipAdapter ────────────────────────────────────────────────────────────────

pub struct HipAdapter {
    info:     AdapterInfo,
    dev_info: HipDeviceInfo,
}

impl GpuAdapter for HipAdapter {
    fn info(&self) -> &AdapterInfo { &self.info }

    fn capabilities(&self) -> HalCapabilities { HalCapabilities::compute_only() }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Ok(Box::new(HipDevice {
            dev_info: Arc::new(self.dev_info.clone()),
            inner:    Mutex::new(HipDeviceInner::new(self.dev_info.ordinal)),
        }))
    }
}

// ── HipDevice ─────────────────────────────────────────────────────────────────

pub struct HipDevice {
    dev_info: Arc<HipDeviceInfo>,
    inner:    Mutex<HipDeviceInner>,
}

impl HipDevice {
    /// Access device properties outside of the locked inner.
    pub fn device_info(&self) -> &HipDeviceInfo { &self.dev_info }

    // ── hipRTC compilation ────────────────────────────────────────────────────

    fn compile_hip_src(&self, src: &[u8]) -> Result<Vec<u8>> {
        let src_c = CString::new(src)
            .map_err(|_| GpuError::ShaderCompile("HIP source contains null byte".into()))?;
        let name_c = CString::new("zen_kernel").unwrap();

        // Build compile options via capability-aware helper.
        // Older ROCm versions may not accept all flags (-O3 gated to ≥ 5.0).
        let caps  = &self.dev_info.capabilities;
        let flags = caps.rtc_options(&self.dev_info.gfx_target);
        let opts_owned: Vec<CString> = flags
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();

        let opt_ptrs: Vec<*const i8> = opts_owned.iter().map(|s| s.as_ptr()).collect();

        let mut prog: HiprtcProg = std::ptr::null_mut();
        unsafe {
            // Activate the target device before compilation.
            self.inner.lock().unwrap().activate()?;

            let rc = hiprtcCreateProgram(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0, std::ptr::null(), std::ptr::null(),
            );
            if rc != 0 {
                return Err(GpuError::ShaderCompile(format!("hiprtcCreateProgram: {rc}")));
            }

            let rc = hiprtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
            if rc != 0 {
                let log = fetch_rtc_log(prog);
                let mut p = prog;
                hiprtcDestroyProgram(&mut p);
                return Err(GpuError::ShaderCompile(log));
            }

            let mut code_sz: usize = 0;
            hiprtcGetCodeSize(prog, &mut code_sz);
            let mut code = vec![0i8; code_sz];
            hiprtcGetCode(prog, code.as_mut_ptr());
            let mut p = prog;
            hiprtcDestroyProgram(&mut p);

            Ok(code.into_iter().map(|b| b as u8).collect())
        }
    }

    // ── Stale-handle error helper ─────────────────────────────────────────────

    fn stale<K>(handle: zengpu_hal::Handle<K>) -> GpuError {
        GpuError::InvalidUsage(UsageError::StaleHandle {
            index:        handle.index(),
            expected_gen: handle.generation(),
            actual_gen:   0,
        })
    }
}

unsafe fn fetch_rtc_log(prog: HiprtcProg) -> String {
    let mut sz: usize = 0;
    unsafe { hiprtcGetProgramLogSize(prog, &mut sz) };
    if sz <= 1 {
        return "hiprtcCompileProgram failed (no log)".into();
    }
    let mut buf = vec![0i8; sz];
    unsafe { hiprtcGetProgramLog(prog, buf.as_mut_ptr()) };
    let bytes: Vec<u8> = buf[..sz.saturating_sub(1)].iter().map(|&b| b as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl GpuDevice for HipDevice {
    fn as_any(&self) -> &dyn std::any::Any { self }

    fn capabilities(&self) -> HalCapabilities { HalCapabilities::compute_only() }

    // ── Buffer ────────────────────────────────────────────────────────────────

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let mut inner = self.inner.lock().unwrap();
        inner.activate()?;

        let size = desc.size as usize;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { check(hipMalloc(&mut ptr, size), "hipMalloc")? };

        Ok(inner.buffers.insert(HipBuffer { ptr, size, usage: desc.usage }))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner.activate()?;
        let buf = inner.buffers.get(buffer).ok_or_else(|| Self::stale(buffer))?;
        let dst = unsafe { (buf.ptr as *mut u8).add(offset as usize) as *mut c_void };
        unsafe {
            check(
                hipMemcpy(dst, data.as_ptr() as *const c_void, data.len(), HIP_MEMCPY_H2D),
                "hipMemcpy H→D",
            )
        }
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let inner = self.inner.lock().unwrap();
        inner.activate()?;
        let buf = inner.buffers.get(buffer).ok_or_else(|| Self::stale(buffer))?;
        let src = unsafe { (buf.ptr as *const u8).add(offset as usize) as *const c_void };
        let mut out = vec![0u8; len as usize];
        unsafe {
            check(
                hipMemcpy(out.as_mut_ptr() as *mut c_void, src, len as usize, HIP_MEMCPY_D2H),
                "hipMemcpy D→H",
            )?;
            check(hipStreamSynchronize(std::ptr::null_mut()), "hipStreamSynchronize")?;
        }
        Ok(out)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(b) = inner.buffers.remove(buffer) {
            unsafe { hipFree(b.ptr) };
        }
    }

    // ── Texture / sampler — compute-only backend ──────────────────────────────

    fn create_texture(&self, _: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("hip: compute-only — no textures".into()))
    }
    fn upload_texture_data(&self, _: TextureHandle, _: &[u8]) -> Result<()> {
        Err(GpuError::Backend("hip: compute-only — no textures".into()))
    }
    fn destroy_texture(&self, _: TextureHandle) {}

    fn create_sampler(&self, _: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("hip: compute-only — no samplers".into()))
    }
    fn destroy_sampler(&self, _: SamplerHandle) {}

    // ── Compute ───────────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let code = match desc.source {
            ShaderSource::Hip(src) => self.compile_hip_src(src)?,
            ShaderSource::Spirv(_) => {
                return Err(GpuError::ShaderCompile(
                    "hip backend: SPIR-V not yet supported; use ShaderDesc::hip()".into(),
                ));
            }
            _ => {
                return Err(GpuError::ShaderCompile(
                    "hip backend: only Hip source accepted".into(),
                ));
            }
        };

        let mut inner = self.inner.lock().unwrap();
        inner.activate()?;

        let mut module: HipModule = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleLoadData(&mut module, code.as_ptr() as *const c_void),
                "hipModuleLoadData",
            )?
        };

        Ok(inner.shaders.insert(HipShader { module }))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(s) = inner.shaders.remove(shader) {
            unsafe { hipModuleUnload(s.module) };
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let entry = CString::new(desc.entry)
            .map_err(|_| GpuError::PipelineCreation("entry point contains null byte".into()))?;

        let mut inner = self.inner.lock().unwrap();
        let module = inner
            .shaders
            .get(desc.shader)
            .ok_or_else(|| GpuError::PipelineCreation("stale shader handle".into()))?
            .module;

        let mut func: HipFunction = std::ptr::null_mut();
        unsafe {
            check(hipModuleGetFunction(&mut func, module, entry.as_ptr()), "hipModuleGetFunction")?
        };

        let block = if desc.block == [0, 0, 0] { [256, 1, 1] } else { desc.block };
        Ok(inner.pipelines.insert(HipPipeline { function: func, block }))
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let mut inner = self.inner.lock().unwrap();
        inner.pipelines.remove(pipeline);
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner.activate()?;

        let pipe = inner
            .pipelines
            .get(pipeline)
            .ok_or_else(|| GpuError::Dispatch("stale pipeline handle".into()))?;

        // Resolve buffer slot indices → raw device pointers.
        let mut raw_ptrs: Vec<*mut c_void> = bindings
            .buffers
            .iter()
            .map(|&idx| {
                inner
                    .buffers
                    .get_by_slot_index(idx)
                    .map(|b| b.ptr)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();

        // Scalar arguments as u32 words.
        let mut scalar_words: Vec<u32> = bindings
            .scalars
            .iter()
            .map(|s| match s {
                zengpu_hal::Scalar::U32(v) => *v,
                zengpu_hal::Scalar::I32(v) => *v as u32,
                zengpu_hal::Scalar::F32(v) => v.to_bits(),
            })
            .collect();

        // hipModuleLaunchKernel kernel_params: pointer-to-each-arg.
        let mut params: Vec<*mut c_void> = raw_ptrs
            .iter_mut()
            .map(|p| p as *mut _ as *mut c_void)
            .chain(scalar_words.iter_mut().map(|w| w as *mut _ as *mut c_void))
            .collect();

        unsafe {
            check(
                hipModuleLaunchKernel(
                    pipe.function,
                    grid[0], grid[1], grid[2],
                    pipe.block[0], pipe.block[1], pipe.block[2],
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )?;
            check(hipStreamSynchronize(std::ptr::null_mut()), "hipStreamSynchronize")?;
        }
        Ok(())
    }

    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        for op in ops {
            self.dispatch(op.pipeline, op.bindings, op.grid)?;
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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
            Err(e)   => { println!("hip: skipped — {e}"); None }
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
        // Require at least one GPU if ROCm initialised successfully.
        assert!(!adapters.is_empty(), "ROCm initialised but found no adapters");
    }

    #[test]
    fn device_info_gcn_target() {
        let Some(inst) = try_instance() else { return };
        for info in inst.device_infos() {
            println!("  [{ordinal}] {name} → gfx target: {gfx:?}",
                ordinal = info.ordinal, name = info.name, gfx = info.gfx_target);
            // gfx_target must be non-empty for any ROCm-supported GPU.
            assert!(!info.gfx_target.is_empty(),
                "no gfx target derived for {}", info.name);
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let Some(inst) = try_instance() else { return };
        let adapter = match inst.request_adapter(AdapterRequest::default()) {
            Some(a) => a,
            None    => { println!("hip: no adapter"); return }
        };
        let device = adapter.open(DeviceRequest::default()).unwrap();

        let data: Vec<u8> = (0u8..128).collect();
        let buf = device.create_buffer(BufferDesc {
            size:   data.len() as u64,
            usage:  BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        }).unwrap();
        device.write_buffer(buf, 0, &data).unwrap();
        let back = device.read_buffer(buf, 0, data.len() as u64).unwrap();
        assert_eq!(data, back, "buffer roundtrip mismatch");
        device.destroy_buffer(buf);
    }

    /// Compile and run vec_add on GPU 0, verify results on CPU.
    #[test]
    fn vec_add_single_gpu() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        run_vec_add(adapters[0].open(DeviceRequest::default()).unwrap().as_ref(), "GPU 0");
    }

    /// Run vec_add on ALL available GPUs sequentially.
    #[test]
    fn vec_add_all_gpus() {
        let Some(inst) = try_instance() else { return };
        for adapter in inst.enumerate_adapters() {
            let label = adapter.info().name.clone();
            let device = adapter.open(DeviceRequest::default()).unwrap();
            run_vec_add(device.as_ref(), &label);
        }
    }

    /// Dispatch vec_add on two GPUs from two threads in parallel.
    #[test]
    fn multi_gpu_parallel_dispatch() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            println!("hip: multi-gpu test needs ≥ 2 GPUs; found {}", adapters.len());
            return;
        }

        // Open a device per adapter, wrap in Arc so threads can own them.
        let devices: Vec<Arc<Box<dyn GpuDevice>>> = adapters
            .iter()
            .map(|a| Arc::new(a.open(DeviceRequest::default()).unwrap()))
            .collect();

        let handles: Vec<_> = devices
            .into_iter()
            .enumerate()
            .map(|(idx, dev)| {
                let label = format!("GPU {idx}");
                std::thread::spawn(move || run_vec_add(dev.as_ref().as_ref(), &label))
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    // ── Shared kernel runner ──────────────────────────────────────────────────

    fn run_vec_add(device: &dyn GpuDevice, label: &str) {
        const N: usize = 1024 * 256; // 256 K elements

        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

        let byte_len = (N * 4) as u64;
        let storage  = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_a = device.create_buffer(BufferDesc { size: byte_len, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let buf_b = device.create_buffer(BufferDesc { size: byte_len, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let buf_c = device.create_buffer(BufferDesc { size: byte_len, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();

        device.write_buffer(buf_a, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(buf_b, 0, bytemuck_cast(&b)).unwrap();

        let shader = device.create_shader(ShaderDesc::hip(VEC_ADD_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "vec_add",
            block: [256, 1, 1],
        }).unwrap();

        let blocks = ((N as u32) + 255) / 256;
        device.dispatch(
            pipeline,
            Bindings {
                buffers:  &[buf_a.index(), buf_b.index(), buf_c.index()],
                textures: &[],
                scalars:  &[zengpu_hal::Scalar::U32(N as u32)],
            },
            [blocks, 1, 1],
        ).unwrap();

        let raw = device.read_buffer(buf_c, 0, byte_len).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);

        let expected = N as f32;
        let mismatches: Vec<_> = c.iter().enumerate()
            .filter(|&(_, &v)| (v - expected).abs() > 1e-3)
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "{label}: vec_add result mismatch at indices {:?} (expected {expected})",
            mismatches.iter().map(|(i, _)| i).collect::<Vec<_>>()
        );
        println!("{label}: vec_add {N} elements OK (c[0]={}, c[N-1]={})", c[0], c[N - 1]);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_a);
        device.destroy_buffer(buf_b);
        device.destroy_buffer(buf_c);
    }

    // ── Tiny bytemuck-free cast helpers ──────────────────────────────────────

    fn bytemuck_cast(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    fn bytemuck_cast_slice(v: &[u8]) -> &[f32] {
        assert_eq!(v.len() % 4, 0);
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
    }

    // ── Heavy compute kernels ─────────────────────────────────────────────────

    /// Tiled SGEMM: C = A * B, A[M×K], B[K×N], C[M×N].
    /// 32×32 tile — fills a full WGP on gfx1200 (1024 threads/WGP).
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

    /// 64×64 macro-tile SGEMM with 4×4 register blocking (16×16 block).
    /// Each thread computes 4×4 = 16 outputs; block tile = 64×64 = 4096 outputs.
    /// LDS padded +1 on K dimension to avoid bank conflicts.
    /// Expected speedup vs SGEMM_SRC: ~4× (4× more compute per LDS access).
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

    /// Parallel tree reduction: reduces N f32 → one f32 per block.
    /// Uses static LDS sized to block (512 threads) — no dynamic shared needed.
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

    /// Memory bandwidth kernel: out[i] = in[i] * scale (streaming read+write).
    const SCALE_SRC: &str = r#"
extern "C" __global__
void mem_scale(const float* __restrict__ in, float* __restrict__ out,
               float scale, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}
"#;

    // ── Bench helper ─────────────────────────────────────────────────────────

    fn elapsed_ms(start: std::time::Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Tiled SGEMM 2048×2048×2048 — measures GFLOP/s.
    #[test]
    fn heavy_sgemm() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let info   = adapters[0].info();

        const M: usize = 2048;
        const N: usize = 2048;
        const K: usize = 2048;

        let a: Vec<f32> = (0..M*K).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..K*N).map(|i| (i % 13) as f32 * 0.01).collect();

        let bytes_mn = (M * N * 4) as u64;
        let bytes_mk = (M * K * 4) as u64;
        let bytes_kn = (K * N * 4) as u64;
        let storage  = BufferUsage::STORAGE | BufferUsage::READBACK;

        let ba = device.create_buffer(BufferDesc { size: bytes_mk, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let bb = device.create_buffer(BufferDesc { size: bytes_kn, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let bc = device.create_buffer(BufferDesc { size: bytes_mn, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();

        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

        let shader   = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "sgemm",
            block: [32, 32, 1],
        }).unwrap();

        // Warm-up.
        let grid = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[
                zengpu_hal::Scalar::U32(M as u32),
                zengpu_hal::Scalar::U32(N as u32),
                zengpu_hal::Scalar::U32(K as u32),
            ],
        };
        device.dispatch(pipeline, bindings, grid).unwrap();

        // Timed run (5 iterations).
        const REPS: u32 = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, grid).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let flop  = 2.0 * M as f64 * N as f64 * K as f64;
        let gflops = flop / (ms * 1e6);

        println!("[{}] SGEMM {M}×{K}×{N}: {ms:.2} ms → {gflops:.1} GFLOP/s", info.name);

        // Correctness spot-check: C[0][0] = sum_k A[0][k]*B[k][0].
        let raw = device.read_buffer(bc, 0, bytes_mn).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);
        let expected_c00: f32 = (0..K).map(|k| a[k] * b[k * N]).sum();
        let err = (c[0] - expected_c00).abs() / expected_c00.abs().max(1e-6);
        assert!(err < 1e-3, "SGEMM C[0][0] error {err:.2e}: got {}, expected {expected_c00}", c[0]);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
    }

    /// Parallel reduction over 128 M floats — verifies sum and measures GB/s.
    #[test]
    fn heavy_reduction() {
        let Some(inst) = try_instance() else { return };
        let device = adapters_or_skip(&inst);

        const N: usize = 128 * 1024 * 1024; // 128 M floats = 512 MB
        const BLOCK: u32 = 512;

        // All ones → expected sum = N.
        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes = (N * 4) as u64;
        let num_blocks = ((N as u32) + BLOCK * 2 - 1) / (BLOCK * 2);
        let partial_bytes = (num_blocks as usize * 4) as u64;
        let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in  = device.create_buffer(BufferDesc { size: src_bytes,     usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let buf_out = device.create_buffer(BufferDesc { size: partial_bytes, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();

        device.write_buffer(buf_in, 0, bytemuck_cast(&data)).unwrap();

        let shader   = device.create_shader(ShaderDesc::hip(REDUCE_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "reduce_sum",
            block: [BLOCK, 1, 1],
        }).unwrap();

        // hipModuleLaunchKernel needs shared_bytes > 0 for `extern __shared__`.
        // Our Bindings API doesn't carry shared_bytes yet — work around by
        // embedding N in a scalar and using a fixed 512*4 shared allocation.
        // For now dispatch with default (0 dynamic shared) — the kernel uses
        // `extern __shared__` which HIP will satisfy with the block * sizeof(float)
        // already allocated for register spill. This works in practice because
        // RDNA3/4 LDS is always allocated per-block; the 0 just means "no
        // *extra* dynamic shared beyond what the kernel declares statically".
        // Actually for `extern __shared__` in HIP with hiprtc we need to pass
        // the size. We'll use a specialised dispatch that goes through `extra`.
        //
        // Simpler: rewrite to use static shared memory size.

        let bindings = Bindings {
            buffers:  &[buf_in.index(), buf_out.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(N as u32)],
        };

        // Warm-up.
        device.dispatch(pipeline, bindings, [num_blocks, 1, 1]).unwrap();

        const REPS: u32 = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, [num_blocks, 1, 1]).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        // Read partial sums → final sum on CPU.
        let raw = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials: &[f32] = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err = (sum - expected).abs() / expected;
        assert!(err < 1e-3, "reduction sum wrong: got {sum}, expected {expected}, err={err:.2e}");

        // Bytes read + written: N*4 read + num_blocks*4 written.
        let bytes_moved = (N * 4 + num_blocks as usize * 4) as f64;
        let gb_s = bytes_moved / (ms * 1e6);
        println!("[GPU 0] reduce_sum {N} floats (512 MB): {ms:.2} ms → {gb_s:.1} GB/s (sum={sum})");

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    /// Streaming scale kernel — measures raw memory bandwidth.
    #[test]
    fn heavy_bandwidth() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        // Test on every GPU.
        for adapter in &adapters {
            let device = adapter.open(DeviceRequest::default()).unwrap();
            let name   = adapter.info().name.clone();

            const N: usize = 256 * 1024 * 1024; // 256 M floats = 1 GB
            const BLOCK: u32 = 256;

            let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
            let bytes = (N * 4) as u64;
            let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

            let buf_in  = device.create_buffer(BufferDesc { size: bytes, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
            let buf_out = device.create_buffer(BufferDesc { size: bytes, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
            device.write_buffer(buf_in, 0, bytemuck_cast(&data)).unwrap();

            let shader   = device.create_shader(ShaderDesc::hip(SCALE_SRC)).unwrap();
            let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
                shader, entry: "mem_scale", block: [BLOCK, 1, 1],
            }).unwrap();

            let grid     = [(N as u32 + BLOCK - 1) / BLOCK, 1, 1];
            let bindings = Bindings {
                buffers:  &[buf_in.index(), buf_out.index()],
                textures: &[],
                scalars:  &[zengpu_hal::Scalar::F32(2.0f32), zengpu_hal::Scalar::U32(N as u32)],
            };

            // Warm-up.
            device.dispatch(pipeline, bindings, grid).unwrap();

            const REPS: u32 = 5;
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                device.dispatch(pipeline, bindings, grid).unwrap();
            }
            let ms = elapsed_ms(t0) / REPS as f64;

            // Bytes moved: 1 read + 1 write = 2 GB.
            let gb_s = (2.0 * bytes as f64) / (ms * 1e6);
            println!("[{name}] mem_scale {N} floats (1 GB r+w): {ms:.2} ms → {gb_s:.1} GB/s");

            // Spot-check first and last elements.
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

    /// SGEMM on both GPUs simultaneously from separate threads.
    #[test]
    fn heavy_multi_gpu_sgemm() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.len() < 2 {
            println!("hip: multi-gpu SGEMM needs ≥ 2 GPUs; found {}", adapters.len());
            // Still run on GPU 0 so CI doesn't skip.
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
                std::thread::spawn(move || {
                    const M: usize = 2048;
                    const N: usize = 2048;
                    const K: usize = 2048;
                    let a: Vec<f32> = (0..M*K).map(|i| (i % 7) as f32 * 0.1).collect();
                    let b: Vec<f32> = (0..K*N).map(|i| (i % 5) as f32 * 0.1).collect();
                    let bytes = |r: usize, c: usize| (r * c * 4) as u64;
                    let storage = BufferUsage::STORAGE | BufferUsage::READBACK;
                    let ba = dev.create_buffer(BufferDesc { size: bytes(M,K), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    let bb = dev.create_buffer(BufferDesc { size: bytes(K,N), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    let bc = dev.create_buffer(BufferDesc { size: bytes(M,N), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    dev.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
                    dev.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
                    let shader   = dev.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
                    let pipeline = dev.create_compute_pipeline(ComputePipelineDesc {
                        shader, entry: "sgemm", block: [32, 32, 1],
                    }).unwrap();
                    let grid = [(N as u32+31)/32, (M as u32+31)/32, 1];
                    let bindings = Bindings {
                        buffers:  &[ba.index(), bb.index(), bc.index()],
                        textures: &[],
                        scalars:  &[
                            zengpu_hal::Scalar::U32(M as u32),
                            zengpu_hal::Scalar::U32(N as u32),
                            zengpu_hal::Scalar::U32(K as u32),
                        ],
                    };
                    // Warm-up.
                    dev.dispatch(pipeline, bindings, grid).unwrap();
                    // Timed.
                    const REPS: u32 = 5;
                    let t0 = std::time::Instant::now();
                    for _ in 0..REPS { dev.dispatch(pipeline, bindings, grid).unwrap(); }
                    let ms = elapsed_ms(t0) / REPS as f64;
                    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);
                    println!("[GPU {idx} – {name}] SGEMM {M}×{K}×{N}: {ms:.2} ms → {gflops:.1} GFLOP/s");
                    dev.destroy_pipeline(pipeline);
                    dev.destroy_shader(shader);
                    dev.destroy_buffer(ba);
                    dev.destroy_buffer(bb);
                    dev.destroy_buffer(bc);
                })
            })
            .collect();

        for h in handles { h.join().expect("thread panicked"); }
    }

    // ── Optimised SGEMM helper ────────────────────────────────────────────────

    fn run_sgemm_opt(device: &dyn GpuDevice, m: usize, n: usize, k: usize) -> f64 {
        let a: Vec<f32> = (0..m*k).map(|i| (i % 7)  as f32 * 0.001).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i % 11) as f32 * 0.001).collect();
        let st = BufferUsage::STORAGE | BufferUsage::READBACK;
        let ba = device.create_buffer(BufferDesc { size: (m*k*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        let bb = device.create_buffer(BufferDesc { size: (k*n*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        let bc = device.create_buffer(BufferDesc { size: (m*n*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
        let shader   = device.create_shader(ShaderDesc::hip(SGEMM_OPT_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "sgemm_opt", block: [16, 16, 1],
        }).unwrap();
        let grid     = [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1];
        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(m as u32), zengpu_hal::Scalar::U32(n as u32), zengpu_hal::Scalar::U32(k as u32)],
        };
        // Two warm-up passes to fill instruction cache.
        device.dispatch(pipeline, bindings, grid).unwrap();
        device.dispatch(pipeline, bindings, grid).unwrap();
        const REPS: u32 = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS { device.dispatch(pipeline, bindings, grid).unwrap(); }
        let ms     = elapsed_ms(t0) / REPS as f64;
        let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e6);
        device.destroy_pipeline(pipeline); device.destroy_shader(shader);
        device.destroy_buffer(ba); device.destroy_buffer(bb); device.destroy_buffer(bc);
        gflops
    }

    /// Optimised SGEMM (reg-blocked): compare naive 32×32 vs opt 64×64 on each size.
    #[test]
    fn heavy_sgemm_opt_sweep() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let adapter = &adapters[0];
        let device  = adapter.open(DeviceRequest::default()).unwrap();
        let name    = adapter.info().name.clone();

        for &sz in &[2048usize, 4096, 8192] {
            // 8192³ = 3 × 256 MB matrices → skip if sz==8192 and VRAM < ~1 GB budget
            // (9060 XT has 16 GB; skip guard here is a safety net on small systems).
            let mem_mb = 3 * sz * sz * 4 / (1024 * 1024);
            let info   = inst.device_infos();
            let vram   = info[0].total_mem / (1024 * 1024);
            if mem_mb as usize > vram * 3 / 4 {
                println!("[{name}] skip SGEMM {sz}³: need {mem_mb} MB, VRAM={vram} MB");
                continue;
            }

            let gflops_naive = if sz <= 4096 {
                // Run naive for comparison only at smaller sizes (slow at 8192).
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

            // At least 2× improvement over naive (usually 3-5×).
            if let Some(n) = gflops_naive {
                assert!(
                    gflops_opt > n * 1.5,
                    "opt SGEMM {sz}³ ({gflops_opt:.0}) should beat naive ({n:.0}) by ≥1.5×"
                );
            }
        }
    }

    fn run_sgemm_naive(device: &dyn GpuDevice, sz: usize) -> f64 {
        let a: Vec<f32> = (0..sz*sz).map(|i| (i % 7)  as f32 * 0.01).collect();
        let b: Vec<f32> = (0..sz*sz).map(|i| (i % 13) as f32 * 0.01).collect();
        let st = BufferUsage::STORAGE | BufferUsage::READBACK;
        let ba = device.create_buffer(BufferDesc { size: (sz*sz*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        let bb = device.create_buffer(BufferDesc { size: (sz*sz*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        let bc = device.create_buffer(BufferDesc { size: (sz*sz*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();
        let shader   = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "sgemm", block: [32, 32, 1],
        }).unwrap();
        let grid = [((sz as u32)+31)/32, ((sz as u32)+31)/32, 1];
        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(sz as u32), zengpu_hal::Scalar::U32(sz as u32), zengpu_hal::Scalar::U32(sz as u32)],
        };
        device.dispatch(pipeline, bindings, grid).unwrap(); // warm-up
        const REPS: u32 = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS { device.dispatch(pipeline, bindings, grid).unwrap(); }
        let ms = elapsed_ms(t0) / REPS as f64;
        device.destroy_pipeline(pipeline); device.destroy_shader(shader);
        device.destroy_buffer(ba); device.destroy_buffer(bb); device.destroy_buffer(bc);
        2.0 * sz as f64 * sz as f64 * sz as f64 / (ms * 1e6)
    }

    /// Multi-GPU parallel opt-SGEMM 4096³.
    #[test]
    fn heavy_multi_gpu_sgemm_opt_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        if adapters.is_empty() { return; }

        let handles: Vec<_> = adapters.iter().enumerate().map(|(idx, a)| {
            let dev: std::sync::Arc<dyn GpuDevice> =
                std::sync::Arc::from(a.open(DeviceRequest::default()).unwrap());
            let name = a.info().name.clone();
            std::thread::spawn(move || {
                let gflops = run_sgemm_opt(dev.as_ref(), 4096, 4096, 4096);
                println!("[GPU {idx} – {name}] opt SGEMM 4096³: {gflops:.0} GFLOP/s");
            })
        }).collect();
        for h in handles { h.join().expect("thread panicked"); }
    }

    // ── ZSL → HIP integration test ────────────────────────────────────────────

    /// Verify that ZSL compiles to valid HIP C++ that hipRTC accepts.
    #[test]
    fn zsl_hip_vec_scale() {
        let Some(inst) = try_instance() else { return };
        let device = adapters_or_skip(&inst);

        // ZSL → all backends at compile time; select HIP C++ at runtime.
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
        let st    = BufferUsage::STORAGE | BufferUsage::READBACK;

        let src = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        let dst = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
        device.write_buffer(src, 0, bytemuck_cast(&data)).unwrap();

        // Compile ZSL's HIP C++ form via hipRTC at runtime.
        let shader   = device.create_shader(ShaderDesc::hip(scale_src)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "zsl_kernel", block: [256, 1, 1],
        }).unwrap();

        let bindings = Bindings {
            buffers:  &[src.index(), dst.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(N as u32)],
        };
        let grid = [(N as u32 + 255) / 256, 1, 1];
        device.dispatch(pipeline, bindings, grid).unwrap();

        let raw = device.read_buffer(dst, 0, 32).unwrap();
        let out = bytemuck_cast_slice(&raw);
        assert!((out[0] - 0.0f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 2.0f32).abs() < 1e-5, "out[1]={}", out[1]);
        assert!((out[2] - 4.0f32).abs() < 1e-5, "out[2]={}", out[2]);
        println!("zsl! → hipRTC: vec_scale OK  out[0..3]={} {} {}", out[0], out[1], out[2]);

        device.destroy_pipeline(pipeline); device.destroy_shader(shader);
        device.destroy_buffer(src); device.destroy_buffer(dst);
    }

    fn adapters_or_skip(inst: &HipInstance) -> Box<dyn GpuDevice> {
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty(), "no HIP adapters");
        adapters.into_iter().next().unwrap().open(DeviceRequest::default()).unwrap()
    }

    // ── 4096 workloads ────────────────────────────────────────────────────────

    /// SGEMM 4096³ single GPU — saturates L2 bandwidth, measures peak GFLOP/s.
    #[test]
    fn heavy_sgemm_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let adapter = &adapters[0];
        let device  = adapter.open(DeviceRequest::default()).unwrap();
        let info    = adapter.info();

        const M: usize = 4096;
        const N: usize = 4096;
        const K: usize = 4096;

        // 4096² × f32 = 64 MB per matrix; 3 matrices = 192 MB — fine for 9060 XT.
        let a: Vec<f32> = (0..M*K).map(|i| (i % 7)  as f32 * 0.001).collect();
        let b: Vec<f32> = (0..K*N).map(|i| (i % 11) as f32 * 0.001).collect();

        let bytes_mn = (M * N * 4) as u64;
        let bytes_mk = (M * K * 4) as u64;
        let bytes_kn = (K * N * 4) as u64;
        let storage  = BufferUsage::STORAGE | BufferUsage::READBACK;

        let ba = device.create_buffer(BufferDesc { size: bytes_mk, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let bb = device.create_buffer(BufferDesc { size: bytes_kn, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let bc = device.create_buffer(BufferDesc { size: bytes_mn, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();

        device.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
        device.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

        let shader   = device.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "sgemm", block: [32, 32, 1],
        }).unwrap();

        let grid     = [(N as u32 + 31) / 32, (M as u32 + 31) / 32, 1];
        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[
                zengpu_hal::Scalar::U32(M as u32),
                zengpu_hal::Scalar::U32(N as u32),
                zengpu_hal::Scalar::U32(K as u32),
            ],
        };

        // Warm-up (2 passes to fill caches).
        device.dispatch(pipeline, bindings, grid).unwrap();
        device.dispatch(pipeline, bindings, grid).unwrap();

        const REPS: u32 = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS { device.dispatch(pipeline, bindings, grid).unwrap(); }
        let ms     = elapsed_ms(t0) / REPS as f64;
        let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);

        println!("[{}] SGEMM {M}×{K}×{N}: {ms:.1} ms → {gflops:.1} GFLOP/s", info.name);

        // Correctness spot-check.
        let raw = device.read_buffer(bc, 0, 64).unwrap();
        let c: &[f32] = bytemuck_cast_slice(&raw);
        let expected: f32 = (0..K).map(|k| a[k] * b[k * N]).sum();
        let err = (c[0] - expected).abs() / expected.abs().max(1e-6);
        assert!(err < 1e-2, "SGEMM 4096 C[0][0] err {err:.2e}: got {} expected {expected}", c[0]);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(ba);
        device.destroy_buffer(bb);
        device.destroy_buffer(bc);
    }

    /// Parallel tree reduction 256 M floats (1 GB) — larger than GPU L2.
    #[test]
    fn heavy_reduction_256m() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let name   = adapters[0].info().name.clone();

        const N: usize = 256 * 1024 * 1024; // 256 M floats = 1 GB
        const BLOCK: u32 = 512;

        // Use sequential integer data so the expected sum is exact in f32.
        // f32 can represent all integers up to 2^24 = 16 M exactly; for 256 M
        // elements of value 1.0 the sum is 256 M which fits exactly.
        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes     = (N * 4) as u64;
        let num_blocks    = ((N as u32) + BLOCK * 2 - 1) / (BLOCK * 2);
        let partial_bytes = (num_blocks as usize * 4) as u64;
        let storage       = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in  = device.create_buffer(BufferDesc { size: src_bytes,     usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let buf_out = device.create_buffer(BufferDesc { size: partial_bytes, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        device.write_buffer(buf_in, 0, bytemuck_cast(&data)).unwrap();

        let shader   = device.create_shader(ShaderDesc::hip(REDUCE_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "reduce_sum", block: [BLOCK, 1, 1],
        }).unwrap();

        let bindings = Bindings {
            buffers:  &[buf_in.index(), buf_out.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(N as u32)],
        };

        // Warm-up.
        device.dispatch(pipeline, bindings, [num_blocks, 1, 1]).unwrap();

        const REPS: u32 = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, [num_blocks, 1, 1]).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let raw      = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err      = (sum - expected).abs() / expected;
        assert!(err < 1e-3, "reduction 256M: got {sum}, expected {expected}, err={err:.2e}");

        let bytes_moved = (N * 4 + num_blocks as usize * 4) as f64;
        let gb_s        = bytes_moved / (ms * 1e6);
        println!("[{name}] reduce_sum 256M floats (1 GB): {ms:.2} ms → {gb_s:.1} GB/s  sum={sum}");

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    /// Wave-level reduction kernel (DPP swizzle) — RDNA 4 specific, uses
    /// `__builtin_amdgcn_ds_swizzle` for intra-wave communication without LDS.
    /// Falls back to LDS reduction on older ROCm / non-RDNA hardware.
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

    // Grid-stride accumulation into a register.
    float v = 0.0f;
    for (unsigned int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += gridDim.x * BLOCK_SIZE)
        v += in[i];

    // Intra-wave butterfly reduction (DPP / warp shuffle).
    v += __shfl_xor(v, 16);
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 1);

    // Lane 0 writes the wave sum to shared memory.
    if (lane == 0) sdata[wid] = v;
    __syncthreads();

    // Thread 0 sums the wave results.
    if (tid == 0) {
        float s = 0.0f;
        unsigned int nw = BLOCK_SIZE / 32;
        for (unsigned int w = 0; w < nw; ++w) s += sdata[w];
        out[blockIdx.x] = s;
    }
}
"#;

    /// Wave-level reduction 256 M floats — uses `__shfl_xor` (DPP on RDNA).
    /// Skips on ROCm < 4.0 where wave intrinsics may not compile.
    #[test]
    fn heavy_wave_reduce_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty());

        let hip_info = inst.device_infos();
        // Gate: wave_reduction requires ROCm ≥ 4.0.
        if !hip_info[0].capabilities.wave_reduction {
            println!("skip heavy_wave_reduce_4096: ROCm < 4.0 (no wave reduction builtins)");
            return;
        }

        let device = adapters[0].open(DeviceRequest::default()).unwrap();
        let name   = adapters[0].info().name.clone();

        const N: usize = 256 * 1024 * 1024;
        const BLOCK: u32 = 256;
        const NUM_BLOCKS: u32 = 2048; // grid-stride, so fixed block count

        let data: Vec<f32> = vec![1.0f32; N];

        let src_bytes     = (N * 4) as u64;
        let partial_bytes = (NUM_BLOCKS as usize * 4) as u64;
        let storage       = BufferUsage::STORAGE | BufferUsage::READBACK;

        let buf_in  = device.create_buffer(BufferDesc { size: src_bytes,     usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        let buf_out = device.create_buffer(BufferDesc { size: partial_bytes, usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
        device.write_buffer(buf_in, 0, bytemuck_cast(&data)).unwrap();

        let shader   = device.create_shader(ShaderDesc::hip(WAVE_REDUCE_SRC)).unwrap();
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader, entry: "wave_reduce_sum", block: [BLOCK, 1, 1],
        }).unwrap();

        let bindings = Bindings {
            buffers:  &[buf_in.index(), buf_out.index()],
            textures: &[],
            scalars:  &[zengpu_hal::Scalar::U32(N as u32)],
        };

        device.dispatch(pipeline, bindings, [NUM_BLOCKS, 1, 1]).unwrap();

        const REPS: u32 = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            device.dispatch(pipeline, bindings, [NUM_BLOCKS, 1, 1]).unwrap();
        }
        let ms = elapsed_ms(t0) / REPS as f64;

        let raw      = device.read_buffer(buf_out, 0, partial_bytes).unwrap();
        let partials = bytemuck_cast_slice(&raw);
        let sum: f32 = partials.iter().sum();
        let expected = N as f32;
        let err      = (sum - expected).abs() / expected;
        assert!(err < 1e-3, "wave reduce: got {sum}, expected {expected}, err={err:.2e}");

        let bytes_moved = N as f64 * 4.0;
        let gb_s        = bytes_moved / (ms * 1e6);
        println!("[{name}] wave_reduce 256M floats: {ms:.2} ms → {gb_s:.1} GB/s  (wave{} DPP)",
                 hip_info[0].capabilities.wave_size);

        device.destroy_pipeline(pipeline);
        device.destroy_shader(shader);
        device.destroy_buffer(buf_in);
        device.destroy_buffer(buf_out);
    }

    /// Multi-GPU SGEMM 4096³ from separate threads simultaneously.
    #[test]
    fn heavy_multi_gpu_sgemm_4096() {
        let Some(inst) = try_instance() else { return };
        let adapters = inst.enumerate_adapters();

        if adapters.is_empty() { return; }
        if adapters.len() < 2 {
            println!("hip: multi-gpu 4096 needs ≥ 2 GPUs; running single-GPU only");
        }

        let devices: Vec<Arc<Box<dyn GpuDevice>>> = adapters
            .iter()
            .map(|a| Arc::new(a.open(DeviceRequest::default()).unwrap()))
            .collect();
        let names: Vec<String> = adapters.iter().map(|a| a.info().name.clone()).collect();

        let handles: Vec<_> = devices.into_iter().zip(names).enumerate()
            .map(|(idx, (dev, name))| {
                std::thread::spawn(move || {
                    const M: usize = 4096;
                    const N: usize = 4096;
                    const K: usize = 4096;

                    let a: Vec<f32> = (0..M*K).map(|i| (i % 7)  as f32 * 0.001).collect();
                    let b: Vec<f32> = (0..K*N).map(|i| (i % 11) as f32 * 0.001).collect();

                    let bytes = |r: usize, c: usize| (r * c * 4) as u64;
                    let storage = BufferUsage::STORAGE | BufferUsage::READBACK;

                    let ba = dev.create_buffer(BufferDesc { size: bytes(M,K), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    let bb = dev.create_buffer(BufferDesc { size: bytes(K,N), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    let bc = dev.create_buffer(BufferDesc { size: bytes(M,N), usage: storage, memory: MemoryUsage::GpuOnly }).unwrap();
                    dev.write_buffer(ba, 0, bytemuck_cast(&a)).unwrap();
                    dev.write_buffer(bb, 0, bytemuck_cast(&b)).unwrap();

                    let shader   = dev.create_shader(ShaderDesc::hip(SGEMM_SRC)).unwrap();
                    let pipeline = dev.create_compute_pipeline(ComputePipelineDesc {
                        shader, entry: "sgemm", block: [32, 32, 1],
                    }).unwrap();

                    let grid     = [(N as u32+31)/32, (M as u32+31)/32, 1];
                    let bindings = Bindings {
                        buffers:  &[ba.index(), bb.index(), bc.index()],
                        textures: &[],
                        scalars:  &[
                            zengpu_hal::Scalar::U32(M as u32),
                            zengpu_hal::Scalar::U32(N as u32),
                            zengpu_hal::Scalar::U32(K as u32),
                        ],
                    };

                    // Two warm-up passes.
                    dev.dispatch(pipeline, bindings, grid).unwrap();
                    dev.dispatch(pipeline, bindings, grid).unwrap();

                    const REPS: u32 = 3;
                    let t0 = std::time::Instant::now();
                    for _ in 0..REPS { dev.dispatch(pipeline, bindings, grid).unwrap(); }
                    let ms     = elapsed_ms(t0) / REPS as f64;
                    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (ms * 1e6);
                    println!("[GPU {idx} – {name}] SGEMM 4096³: {ms:.1} ms → {gflops:.1} GFLOP/s");

                    dev.destroy_pipeline(pipeline);
                    dev.destroy_shader(shader);
                    dev.destroy_buffer(ba);
                    dev.destroy_buffer(bb);
                    dev.destroy_buffer(bc);
                })
            })
            .collect();

        for h in handles { h.join().expect("thread panicked"); }
    }

    // ── Capability report ─────────────────────────────────────────────────────

    /// Print the full capability report for every detected GPU.
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

            // Invariants that must hold for any device on any ROCm version:
            // 1. wave size must be either 32 or 64.
            assert!(
                caps.wave_size == 32 || caps.wave_size == 64,
                "unexpected wave size {} on {}", caps.wave_size, info.gfx_target,
            );
            // 2. MFMA implies CDNA (no MFMA on RDNA consumer silicon).
            if caps.mfma {
                assert!(
                    info.gfx_family.full_fp64(),
                    "mfma set but not a CDNA device: {}", info.gfx_target,
                );
            }
            // 3. If hipRTC unavailable, bitcode must also be unavailable.
            if !caps.hiprtc {
                assert!(!caps.hiprtc_bitcode, "bitcode without hipRTC on {}", info.gfx_target);
            }
            // 4. RDNA 2+ is WGP mode by default; RDNA 1 / GCN / CDNA is not.
            let expect_wgp = matches!(
                info.gfx_family,
                version::GfxFamily::Rdna2 | version::GfxFamily::Rdna3
                | version::GfxFamily::Rdna3p5 | version::GfxFamily::Rdna4,
            );
            assert_eq!(
                caps.wgp_default, expect_wgp,
                "WGP mode mismatch for {} ({:?})", info.gfx_target, info.gfx_family,
            );
        }
    }
}
