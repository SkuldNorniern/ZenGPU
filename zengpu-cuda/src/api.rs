//! Raw CUDA Driver API, loaded at runtime via libloading.
//!
//! `CU*` types and function pointers are `pub(crate)`. Nothing here crosses the public facade.

use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;

/// Ordinal of a CUDA device, in [0, device_count).
pub(crate) type CUdevice = c_int;

/// `CUresult` code. 0 is `CUDA_SUCCESS`.
pub(crate) type CUresult = c_uint;

pub(crate) const CUDA_SUCCESS: CUresult = 0;

pub(crate) type FnCuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
pub(crate) type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
pub(crate) type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
pub(crate) type FnCuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;

/// Function pointers are valid only while `_lib` is alive, i.e. while this struct lives.
pub(crate) struct CudaApi {
    pub(crate) cu_init: FnCuInit,
    pub(crate) cu_device_get_count: FnCuDeviceGetCount,
    pub(crate) cu_device_get: FnCuDeviceGet,
    pub(crate) cu_device_get_name: FnCuDeviceGetName,
    _lib: libloading::Library,
}

impl CudaApi {
    /// Returns `None` if CUDA is absent or a required symbol is missing.
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
