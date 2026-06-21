//! Raw CUDA Driver API bindings loaded at runtime via libloading.
//!
//! All `CU*` types and raw function pointers are `pub(crate)`. None cross the
//! public ZenGPU facade (D17 / D10 policy).

use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;

// ── Raw Driver API types ──────────────────────────────────────────────────────

/// Ordinal handle for a CUDA physical device. An integer in [0, device_count).
pub(crate) type CUdevice = c_int;

/// CUDA Driver API result code. 0 = CUDA_SUCCESS.
pub(crate) type CUresult = c_uint;

pub(crate) const CUDA_SUCCESS: CUresult = 0;

// ── Function pointer types (cdecl on all platforms) ───────────────────────────

pub(crate) type FnCuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
pub(crate) type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
pub(crate) type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
pub(crate) type FnCuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;

// ── Loaded Driver API ─────────────────────────────────────────────────────────

/// Loaded CUDA Driver API symbols. The library is kept alive by `_lib`; raw
/// function pointers remain valid as long as this struct is live.
pub(crate) struct CudaApi {
    pub(crate) cu_init: FnCuInit,
    pub(crate) cu_device_get_count: FnCuDeviceGetCount,
    pub(crate) cu_device_get: FnCuDeviceGet,
    pub(crate) cu_device_get_name: FnCuDeviceGetName,
    _lib: libloading::Library, // keep library resident; fields drop in order — _lib last
}

impl CudaApi {
    /// Try to load the CUDA Driver API from the platform system library.
    /// Returns `None` when CUDA is absent or a required symbol is missing.
    pub(crate) fn load() -> Option<Arc<Self>> {
        let lib_name = if cfg!(target_os = "windows") {
            "nvcuda.dll"
        } else if cfg!(target_os = "macos") {
            "libcuda.dylib"
        } else {
            "libcuda.so.1"
        };

        let lib = match unsafe { libloading::Library::new(lib_name) } {
            Ok(l) => l,
            Err(e) => {
                log::debug!("cuda: library '{lib_name}' not found ({e}); no CUDA adapters");
                return None;
            }
        };

        let cu_init: FnCuInit = match unsafe { lib.get(b"cuInit\0") } {
            Ok(s) => *s,
            Err(e) => {
                log::warn!("cuda: symbol 'cuInit' missing: {e}");
                return None;
            }
        };
        let cu_device_get_count: FnCuDeviceGetCount =
            match unsafe { lib.get(b"cuDeviceGetCount\0") } {
                Ok(s) => *s,
                Err(e) => {
                    log::warn!("cuda: symbol 'cuDeviceGetCount' missing: {e}");
                    return None;
                }
            };
        let cu_device_get: FnCuDeviceGet = match unsafe { lib.get(b"cuDeviceGet\0") } {
            Ok(s) => *s,
            Err(e) => {
                log::warn!("cuda: symbol 'cuDeviceGet' missing: {e}");
                return None;
            }
        };
        let cu_device_get_name: FnCuDeviceGetName =
            match unsafe { lib.get(b"cuDeviceGetName\0") } {
                Ok(s) => *s,
                Err(e) => {
                    log::warn!("cuda: symbol 'cuDeviceGetName' missing: {e}");
                    return None;
                }
            };

        Some(Arc::new(Self {
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_device_get_name,
            _lib: lib,
        }))
    }
}
