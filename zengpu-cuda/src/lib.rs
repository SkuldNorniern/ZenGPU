//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Uses cuda-oxide for Driver API access; absent CUDA yields an empty
//! adapter list rather than a build or link error (the stub library path
//! returns ErrorCode::StubLibrary from cuInit).
//!
//! Commit 1: instance + adapter enumeration. Device memory / dispatch / BLAS
//! come in subsequent commits as per the bring-up plan.

mod error;

use cuda_oxide::{Cuda, device::Device};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, DeviceRequest,
    DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities, Result,
    SamplerDesc, SamplerHandle, TextureDesc, TextureHandle,
};

#[allow(unused_imports)]
use error::from_cuda;

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
                    let name = dev
                        .name()
                        .unwrap_or_else(|_| "Unknown CUDA Device".into());
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
    #[allow(dead_code)]
    dev: Device,
    info: AdapterInfo,
}

// SAFETY: Device is a newtype over a CUdevice (c_int ordinal); it is safe to
// send across threads.
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
        // Context / stream / buffer allocation lands in the next commit.
        Err(GpuError::Backend(
            "cuda: device not yet implemented; adapter enumeration only".into(),
        ))
    }
}

// ── CudaDevice (stub) ─────────────────────────────────────────────────────────
// Skeleton for the next commit; open() will construct it once context + streams
// are wired up.

#[allow(dead_code)]
pub(crate) struct CudaDevice {
    dev: Device,
}

// SAFETY: CudaDevice wraps a CUdevice ordinal (c_int); safe to send.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

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
}
