//! ZenGPU AMD ROCm/HIP compute backend.
//!
//! Implements the compute slice of `GpuDevice` using the HIP runtime
//! (`libamdhip64`) and hipRTC for runtime kernel compilation.
//!
//! Graphics APIs (textures, samplers, surfaces) are intentionally unsupported:
//! this is a compute-only backend.

use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle, BufferUsage,
    ComputePipelineDesc, DeviceRequest, DeviceType, DispatchOp, GpuAdapter, GpuDevice, GpuError,
    GpuInstance, HalCapabilities, PipelineHandle, Result, SamplerDesc, SamplerHandle,
    ShaderDesc, ShaderHandle, ShaderSource, SlotMap, TextureDesc, TextureHandle, marker,
};

// ── Raw HIP / hipRTC FFI ──────────────────────────────────────────────────────

// Numeric error code returned by every HIP API call; 0 = hipSuccess.
type HipError = i32;

const HIP_SUCCESS: HipError = 0;

// Memory copy direction constants.
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipStream = *mut c_void; // null = default stream
type HiprtcProgram = *mut c_void;

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipInit(flags: u32) -> HipError;
    fn hipGetDeviceCount(count: *mut i32) -> HipError;
    fn hipDeviceGetName(name: *mut i8, len: i32, device: i32) -> HipError;
    fn hipDeviceTotalMem(bytes: *mut usize, device: i32) -> HipError;
    fn hipSetDevice(device: i32) -> HipError;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    fn hipFree(ptr: *mut c_void) -> HipError;
    fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: i32,
    ) -> HipError;
    fn hipModuleLoadData(module: *mut HipModule, image: *const c_void) -> HipError;
    fn hipModuleGetFunction(
        function: *mut HipFunction,
        module: HipModule,
        kname: *const i8,
    ) -> HipError;
    fn hipModuleUnload(module: HipModule) -> HipError;
    fn hipModuleLaunchKernel(
        f: HipFunction,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem_bytes: u32,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipStreamSynchronize(stream: HipStream) -> HipError;
}

#[link(name = "hiprtc")]
unsafe extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const i8,
        name: *const i8,
        num_headers: i32,
        headers: *const *const i8,
        include_names: *const *const i8,
    ) -> i32;
    fn hiprtcCompileProgram(prog: HiprtcProgram, num_options: i32, options: *const *const i8) -> i32;
    fn hiprtcGetCodeSize(prog: HiprtcProgram, code_size: *mut usize) -> i32;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut i8) -> i32;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> i32;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size: *mut usize) -> i32;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut i8) -> i32;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn check(code: HipError, ctx: &'static str) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Backend(format!("{ctx}: HIP error {code}")))
    }
}

// ── Internal resource types ───────────────────────────────────────────────────

#[allow(dead_code)]
struct HipBuffer {
    ptr: *mut c_void,
    size: usize,
    usage: BufferUsage,
    // Host-side shadow for readback buffers, avoiding a synchronous D→H copy
    // per write. None for GpuOnly buffers.
    host_shadow: Option<Vec<u8>>,
}

// SAFETY: The raw pointer is owned by this struct; Send/Sync are safe as long
// as the HipDeviceInner Mutex is held for all accesses.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

struct HipShader {
    module: HipModule,
}

unsafe impl Send for HipShader {}
unsafe impl Sync for HipShader {}

struct HipPipeline {
    function: HipFunction,
    block: [u32; 3],
}

unsafe impl Send for HipPipeline {}
unsafe impl Sync for HipPipeline {}

// ── Device inner state ────────────────────────────────────────────────────────

struct HipDeviceInner {
    ordinal: i32,
    buffers: SlotMap<marker::Buffer, HipBuffer>,
    shaders: SlotMap<marker::Shader, HipShader>,
    pipelines: SlotMap<marker::Pipeline, HipPipeline>,
}

impl HipDeviceInner {
    fn new(ordinal: i32) -> Self {
        Self {
            ordinal,
            buffers: SlotMap::new(),
            shaders: SlotMap::new(),
            pipelines: SlotMap::new(),
        }
    }

    fn set_device(&self) -> Result<()> {
        unsafe { check(hipSetDevice(self.ordinal), "hipSetDevice") }
    }
}

impl Drop for HipDeviceInner {
    fn drop(&mut self) {
        // Drain remaining GPU resources.
        // SlotMap doesn't expose drain, so we rely on destroy calls having
        // freed things already; modules at least need explicit unload.
        // In practice the caller should have destroyed everything first.
    }
}

// ── HipInstance ───────────────────────────────────────────────────────────────

/// Entry-point for the ROCm/HIP backend.
pub struct HipInstance;

impl HipInstance {
    /// Initialise the HIP runtime and return an instance.
    ///
    /// Returns `Err` if `hipInit` fails (no AMD driver / ROCm not installed).
    pub fn new() -> Result<Self> {
        unsafe { check(hipInit(0), "hipInit")? };
        Ok(Self)
    }
}

impl GpuInstance for HipInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        let mut count: i32 = 0;
        if unsafe { hipGetDeviceCount(&mut count) } != HIP_SUCCESS || count == 0 {
            log::debug!("hip: no devices found");
            return Vec::new();
        }

        (0..count)
            .filter_map(|i| {
                let mut name_buf = [0i8; 256];
                let mut total_mem: usize = 0;
                unsafe {
                    if hipDeviceGetName(name_buf.as_mut_ptr(), 256, i) != HIP_SUCCESS {
                        return None;
                    }
                    let _ = hipDeviceTotalMem(&mut total_mem, i);
                }
                let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                log::debug!("hip: adapter {i}: {name} ({} MiB)", total_mem / (1024 * 1024));
                Some(Box::new(HipAdapter {
                    info: AdapterInfo {
                        name,
                        vendor: 0x1002, // AMD PCI vendor ID
                        device: 0,
                        device_type: DeviceType::Discrete,
                        backend: BackendPreference::Hip,
                    },
                    ordinal: i,
                    total_mem,
                }) as Box<dyn GpuAdapter>)
            })
            .collect()
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        let adapters = self.enumerate_adapters();
        if adapters.is_empty() {
            return None;
        }
        // Prefer discrete; fall back to first.
        let _ = req;
        adapters.into_iter().next()
    }
}

// ── HipAdapter ────────────────────────────────────────────────────────────────

pub struct HipAdapter {
    info: AdapterInfo,
    ordinal: i32,
    total_mem: usize,
}

impl GpuAdapter for HipAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Ok(Box::new(HipDevice {
            info: self.info.clone(),
            total_mem: self.total_mem,
            inner: Mutex::new(HipDeviceInner::new(self.ordinal)),
        }))
    }
}

// ── HipDevice ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct HipDevice {
    info: AdapterInfo,
    total_mem: usize,
    inner: Mutex<HipDeviceInner>,
}

impl HipDevice {
    fn compile_hip_src(&self, src: &[u8]) -> Result<Vec<u8>> {
        let src_c = CString::new(src)
            .map_err(|_| GpuError::ShaderCompile("HIP source contains null byte".into()))?;
        let name_c = CString::new("zen_kernel").unwrap();

        // Determine the gfx target from the device name if possible, fall back
        // to a safe default that covers RDNA 3/4.
        let ordinal = self.inner.lock().unwrap().ordinal;
        let mut name_buf = [0i8; 256];
        let gfx_arg = unsafe {
            hipDeviceGetName(name_buf.as_mut_ptr(), 256, ordinal);
            // Map well-known marketing names → gfx targets. Extend as needed.
            let name = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy();
            if name.contains("9060") || name.contains("9070") || name.contains("9080") {
                // RDNA 4 family
                "--gpu-architecture=gfx1200".to_string()
            } else if name.contains("7900") || name.contains("7800") || name.contains("7600") {
                // RDNA 3 family
                "--gpu-architecture=gfx1100".to_string()
            } else {
                // Safe cross-generation fallback
                "--gpu-architecture=gfx906".to_string()
            }
        };
        let gfx_c = CString::new(gfx_arg).unwrap();
        let options = [gfx_c.as_ptr()];

        let mut prog: HiprtcProgram = std::ptr::null_mut();
        unsafe {
            let rc = hiprtcCreateProgram(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0, std::ptr::null(), std::ptr::null(),
            );
            if rc != 0 {
                return Err(GpuError::ShaderCompile(format!("hiprtcCreateProgram: {rc}")));
            }

            let rc = hiprtcCompileProgram(prog, options.len() as i32, options.as_ptr());
            if rc != 0 {
                // Retrieve the log before destroying the program.
                let mut log_size: usize = 0;
                hiprtcGetProgramLogSize(prog, &mut log_size);
                let log = if log_size > 1 {
                    let mut buf = vec![0i8; log_size];
                    hiprtcGetProgramLog(prog, buf.as_mut_ptr());
                    String::from_utf8_lossy(
                        &buf[..log_size.saturating_sub(1)]
                            .iter()
                            .map(|&b| b as u8)
                            .collect::<Vec<_>>(),
                    )
                    .into_owned()
                } else {
                    format!("hiprtcCompileProgram error {rc}")
                };
                let mut p = prog;
                hiprtcDestroyProgram(&mut p);
                return Err(GpuError::ShaderCompile(log));
            }

            let mut code_size: usize = 0;
            hiprtcGetCodeSize(prog, &mut code_size);
            let mut code = vec![0i8; code_size];
            hiprtcGetCode(prog, code.as_mut_ptr());
            let mut p = prog;
            hiprtcDestroyProgram(&mut p);

            Ok(code.into_iter().map(|b| b as u8).collect())
        }
    }
}

impl GpuDevice for HipDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    // ── Buffer API ────────────────────────────────────────────────────────────

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let mut inner = self.inner.lock().unwrap();
        inner.set_device()?;

        let size = desc.size as usize;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { check(hipMalloc(&mut ptr, size), "hipMalloc")? };

        let host_shadow = if desc.usage.contains(BufferUsage::READBACK) {
            Some(vec![0u8; size])
        } else {
            None
        };

        let handle = inner.buffers.insert(HipBuffer {
            ptr,
            size,
            usage: desc.usage,
            host_shadow,
        });
        Ok(handle)
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner.set_device()?;

        let buf = inner.buffers.get(buffer).ok_or(GpuError::InvalidUsage(
            zengpu_hal::UsageError::StaleHandle {
                index: buffer.index(),
                expected_gen: buffer.generation(),
                actual_gen: 0,
            },
        ))?;

        let dst = unsafe { (buf.ptr as *mut u8).add(offset as usize) };
        unsafe {
            check(
                hipMemcpy(
                    dst as *mut c_void,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "hipMemcpy H→D",
            )?
        };
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let inner = self.inner.lock().unwrap();
        inner.set_device()?;

        let buf = inner.buffers.get(buffer).ok_or(GpuError::InvalidUsage(
            zengpu_hal::UsageError::StaleHandle {
                index: buffer.index(),
                expected_gen: buffer.generation(),
                actual_gen: 0,
            },
        ))?;

        let mut out = vec![0u8; len as usize];
        let src = unsafe { (buf.ptr as *const u8).add(offset as usize) };
        unsafe {
            check(
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    src as *const c_void,
                    len as usize,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                ),
                "hipMemcpy D→H",
            )?;
            check(hipStreamSynchronize(std::ptr::null_mut()), "hipStreamSynchronize")?;
        }
        Ok(out)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(buf) = inner.buffers.remove(buffer) {
            unsafe { hipFree(buf.ptr) };
        }
    }

    // ── Texture / sampler (unsupported — compute-only) ────────────────────────

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("hip: compute-only — no textures".into()))
    }
    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("hip: compute-only — no textures".into()))
    }
    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("hip: compute-only — no samplers".into()))
    }
    fn destroy_sampler(&self, _sampler: SamplerHandle) {}

    // ── Compute API ───────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let code = match desc.source {
            ShaderSource::Hip(src) => self.compile_hip_src(src)?,
            ShaderSource::Spirv(_) => {
                return Err(GpuError::ShaderCompile(
                    "hip backend: SPIR-V not yet supported; supply HIP C++ source via ShaderDesc::hip()".into(),
                ));
            }
            _ => {
                return Err(GpuError::ShaderCompile(
                    "hip backend: only Hip and Spirv sources are accepted".into(),
                ));
            }
        };

        let mut inner = self.inner.lock().unwrap();
        inner.set_device()?;

        let mut module: HipModule = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleLoadData(&mut module, code.as_ptr() as *const c_void),
                "hipModuleLoadData",
            )?
        };

        let handle = inner.shaders.insert(HipShader { module });
        Ok(handle)
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
            .ok_or_else(|| GpuError::PipelineCreation("shader handle is stale".into()))?
            .module;

        let mut function: HipFunction = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleGetFunction(&mut function, module, entry.as_ptr()),
                "hipModuleGetFunction",
            )?
        };

        let block = if desc.block == [0, 0, 0] {
            [256, 1, 1]
        } else {
            desc.block
        };

        let handle = inner.pipelines.insert(HipPipeline { function, block });
        Ok(handle)
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let mut inner = self.inner.lock().unwrap();
        inner.pipelines.remove(pipeline);
        // HIP functions are owned by the module; no explicit release needed.
    }

    fn dispatch(&self, pipeline: PipelineHandle, bindings: Bindings<'_>, grid: [u32; 3]) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner.set_device()?;

        let pipe = inner
            .pipelines
            .get(pipeline)
            .ok_or_else(|| GpuError::Dispatch("pipeline handle is stale".into()))?;

        // Build kernel_params: buffer pointers, then scalars.
        // Each buffer binding is an index into the slot map; we pass the raw
        // device pointer to the kernel.
        let mut raw_buf_ptrs: Vec<*mut c_void> = bindings
            .buffers
            .iter()
            .map(|&idx| {
                // SlotMap index → raw device pointer.
                // We reconstruct a handle from the index at generation 0; if the
                // slot has been recycled the pointer will be null-ish, but that is
                // a caller bug, not ours.
                inner
                    .buffers
                    .get_by_slot_index(idx)
                    .map(|b| b.ptr)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();

        // Scalar arguments as u32/f32 words.
        let mut scalar_words: Vec<u32> = bindings
            .scalars
            .iter()
            .map(|s| match s {
                zengpu_hal::Scalar::U32(v) => *v,
                zengpu_hal::Scalar::I32(v) => *v as u32,
                zengpu_hal::Scalar::F32(v) => v.to_bits(),
            })
            .collect();

        // hipModuleLaunchKernel kernel_params: array of pointers to each arg.
        let mut params: Vec<*mut c_void> = raw_buf_ptrs
            .iter_mut()
            .map(|p| p as *mut _ as *mut c_void)
            .chain(
                scalar_words
                    .iter_mut()
                    .map(|w| w as *mut _ as *mut c_void),
            )
            .collect();
        params.push(std::ptr::null_mut());

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
        let inner = self.inner.lock().unwrap();
        inner.set_device()?;
        drop(inner); // release lock before calling dispatch in a loop

        for op in ops {
            self.dispatch(op.pipeline, op.bindings, op.grid)?;
        }
        // Single sync after the batch — override the per-dispatch sync.
        // Currently dispatch already syncs; a future async path can batch here.
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_and_enumerates() {
        match HipInstance::new() {
            Ok(inst) => {
                let adapters = inst.enumerate_adapters();
                println!("hip: found {} adapter(s)", adapters.len());
                for a in &adapters {
                    println!("  {}", a.info().name);
                }
            }
            Err(e) => {
                println!("hip: skipped — {e}");
            }
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let inst = match HipInstance::new() {
            Ok(i) => i,
            Err(e) => { println!("hip: skipped — {e}"); return; }
        };
        let adapter = match inst.request_adapter(AdapterRequest::default()) {
            Some(a) => a,
            None => { println!("hip: no adapter"); return; }
        };
        let device = adapter.open(DeviceRequest::default()).unwrap();

        let desc = BufferDesc {
            size: 256,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: zengpu_hal::MemoryUsage::GpuOnly,
        };
        let buf = device.create_buffer(desc).unwrap();
        let data: Vec<u8> = (0u8..64).collect();
        device.write_buffer(buf, 0, &data).unwrap();
        let back = device.read_buffer(buf, 0, 64).unwrap();
        assert_eq!(data, back);
        device.destroy_buffer(buf);
    }
}
