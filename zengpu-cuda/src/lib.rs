//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Loads the Driver API at runtime via libloading; absent CUDA yields
//! an empty adapter list rather than a build or link error.
//!
//! Commit 1: instance + adapter enumeration. Device memory / dispatch / BLAS
//! come in subsequent commits as per the bring-up plan.

mod api;
mod error;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Arc;

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, DeviceRequest,
    DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, Result,
    SamplerDesc, SamplerHandle, TextureDesc, TextureHandle,
};

use api::{CUDA_SUCCESS, CUdevice, CudaApi};

// ── CudaInstance ──────────────────────────────────────────────────────────────

/// Entry-point for the CUDA backend. Holds the loaded Driver API; `None` when
/// CUDA is absent on the current machine.
pub struct CudaInstance {
    api: Option<Arc<CudaApi>>,
}

impl CudaInstance {
    pub fn new() -> Self {
        let api = CudaApi::load().and_then(|api| {
            let r = unsafe { (api.cu_init)(0) };
            if r == CUDA_SUCCESS {
                Some(api)
            } else {
                log::debug!("cuda: cuInit failed (CUresult={r}); no CUDA adapters available");
                None
            }
        });
        Self { api }
    }
}

impl Default for CudaInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for CudaInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        let Some(api) = &self.api else {
            return Vec::new();
        };

        let mut count: i32 = 0;
        let r = unsafe { (api.cu_device_get_count)(&mut count) };
        if r != CUDA_SUCCESS || count <= 0 {
            return Vec::new();
        }

        (0..count)
            .filter_map(|ordinal| {
                let mut dev: CUdevice = 0;
                if unsafe { (api.cu_device_get)(&mut dev, ordinal) } != CUDA_SUCCESS {
                    return None;
                }

                let mut name_buf = [0i8; 256];
                let r = unsafe {
                    (api.cu_device_get_name)(name_buf.as_mut_ptr() as *mut c_char, 256, dev)
                };
                let name = if r == CUDA_SUCCESS {
                    unsafe { CStr::from_ptr(name_buf.as_ptr() as *const c_char) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("CUDA Device {ordinal}")
                };

                let info = AdapterInfo {
                    name,
                    vendor: 0x10de, // NVIDIA PCI vendor ID
                    device: 0,
                    device_type: DeviceType::Discrete,
                    backend: BackendPreference::Cuda,
                };
                Some(Box::new(CudaAdapter { api: Arc::clone(api), device: dev, info })
                    as Box<dyn GpuAdapter>)
            })
            .collect()
    }

    fn request_adapter(&self, req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        // Ordinal 0 is the primary GPU; future: honour req.power for multi-GPU.
        let _ = req;
        self.enumerate_adapters().into_iter().next()
    }
}

// ── CudaAdapter ───────────────────────────────────────────────────────────────

pub struct CudaAdapter {
    #[allow(dead_code)]
    api: Arc<CudaApi>,
    #[allow(dead_code)]
    device: CUdevice,
    info: AdapterInfo,
}

impl GpuAdapter for CudaAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        // Context / stream / buffer allocation lands in the next commit.
        Err(GpuError::Backend(
            "cuda: device not yet implemented; adapter enumeration only".into(),
        ))
    }
}

// ── CudaDevice (stub) ─────────────────────────────────────────────────────────
// Defined here as a skeleton; open() will construct it once context + streams
// are wired up in the next commit.

#[allow(dead_code)]
pub(crate) struct CudaDevice {
    api: Arc<CudaApi>,
    device: CUdevice,
}

impl GpuDevice for CudaDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, _desc: BufferDesc) -> Result<BufferHandle> {
        Err(GpuError::Backend("cuda: not yet implemented".into()))
    }

    fn write_buffer(&self, _buffer: BufferHandle, _offset: u64, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("cuda: not yet implemented".into()))
    }

    fn read_buffer(&self, _buffer: BufferHandle, _offset: u64, _len: u64) -> Result<Vec<u8>> {
        Err(GpuError::Backend("cuda: not yet implemented".into()))
    }

    fn destroy_buffer(&self, _buffer: BufferHandle) {}

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("cuda: compute-only; no texture support".into()))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("cuda: compute-only; no texture support".into()))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("cuda: compute-only; no sampler support".into()))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        // Must not panic on machines without CUDA installed.
        let inst = CudaInstance::new();
        let _ = inst.enumerate_adapters(); // empty on non-NVIDIA machines; that's fine
    }

    #[test]
    fn adapter_capabilities_are_compute_only() {
        // If CUDA is present, verify adapters report compute-only.
        let inst = CudaInstance::new();
        for adapter in inst.enumerate_adapters() {
            assert!(adapter.capabilities().compute);
            assert!(!adapter.capabilities().graphics);
        }
    }
}
