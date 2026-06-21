//! ZenGPU CUDA backend — compute HAL only (no graphics surfaces or render
//! passes). Uses cuda-oxide for Driver API access; absent CUDA yields an empty
//! adapter list rather than a build or link error (the stub library path
//! returns ErrorCode::StubLibrary from cuInit).

mod error;

use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::Mutex;

use cuda_oxide::{
    Cuda,
    context::{Context, Handle},
    device::Device,
    mem::{DeviceBox, DevicePtr},
};
use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferHandle,
    ComputePipelineDesc, DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance,
    HalCapabilities, PipelineHandle, Result, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    SlotMap, TextureDesc, TextureHandle, marker,
};

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
        let ctx = from_cuda(Context::new(&self.dev))?;
        Ok(Box::new(CudaDevice {
            ctx: UnsafeCell::new(ctx),
            ctx_lock: Mutex::new(()),
            buffers: Mutex::new(SlotMap::new()),
        }))
    }
}

// ── CudaBuffer ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CudaBuffer {
    ptr: u64,
    len: u64,
}

// ── CudaDevice ────────────────────────────────────────────────────────────────

/// An opened CUDA device. Provides compute-only buffer operations via the
/// CUDA Driver API; graphics (textures, render passes) are not supported.
///
/// # Thread safety
///
/// `Context` in cuda-oxide is `!Send` because `Handle` uses an `Rc`. We
/// compensate by holding `ctx_lock` across every operation that enters the
/// context: only one thread at a time ever touches `ctx`, and the
/// `Rc<Handle>` is always created and destroyed within the same locked
/// method call — it never crosses a thread boundary.
pub struct CudaDevice {
    ctx: UnsafeCell<Context>,
    ctx_lock: Mutex<()>,
    buffers: Mutex<SlotMap<marker::Buffer, CudaBuffer>>,
}

// SAFETY: see the CudaDevice doc comment above.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Enter the CUDA context, run `f`, exit the context. `ctx_lock` must be
    /// held for the lifetime of `f` to serialise cross-thread access.
    fn with_context<F, T>(&self, f: F) -> Result<T>
    where
        F: for<'h> FnOnce(Rc<Handle<'h>>) -> Result<T>,
    {
        let _guard = self.ctx_lock.lock().map_err(|_| GpuError::DeviceLost)?;
        // SAFETY: `ctx_lock` is held; no other thread can reach this point.
        let handle = from_cuda(unsafe { &mut *self.ctx.get() }.enter())?;
        f(handle)
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // `&mut self` provides exclusive access — ctx_lock not needed.
        let buffers: Vec<CudaBuffer> = self.buffers.get_mut().unwrap().drain().collect();
        if buffers.is_empty() {
            return;
        }
        // UnsafeCell::get_mut is safe here because of the exclusive &mut self.
        if let Ok(handle) = self.ctx.get_mut().enter() {
            for cb in buffers {
                // SAFETY: ptr/len came from a DeviceBox we explicitly leaked;
                // we are the sole owner and the context is current.
                let dp = unsafe { DevicePtr::from_raw_parts(handle.clone(), cb.ptr, cb.len) };
                let db = unsafe { DeviceBox::from_raw(dp) };
                drop(db); // calls cuMemFree while context is current
            }
        }
        // If enter() fails the device is already dead; allocations are leaked.
    }
}

impl GpuDevice for CudaDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        self.with_context(|handle| {
            let db = from_cuda(DeviceBox::alloc(&handle, desc.size))?;
            // Extract the raw device pointer before leaking so we can track it.
            let ptr = db.as_raw();
            let len = db.len();
            // Prevent Drop from calling cuMemFree; we manage the allocation manually.
            db.leak();
            let bh = self
                .buffers
                .lock()
                .map_err(|_| GpuError::DeviceLost)?
                .insert(CudaBuffer { ptr, len });
            Ok(bh)
        })
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let cb = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;

        if offset + data.len() as u64 > cb.len {
            return Err(GpuError::Backend("cuda: write out of bounds".into()));
        }

        self.with_context(|handle| {
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + data.len() as u64);
            from_cuda(view.store(data))
        })
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let cb = self
            .buffers
            .lock()
            .map_err(|_| GpuError::DeviceLost)?
            .get(buffer)
            .copied()
            .ok_or_else(|| GpuError::Backend("cuda: invalid buffer handle".into()))?;

        if offset + len > cb.len {
            return Err(GpuError::Backend("cuda: read out of bounds".into()));
        }

        self.with_context(|handle| {
            // SAFETY: ptr/len from our buffer table; context is current.
            let dp = unsafe { DevicePtr::from_raw_parts(handle, cb.ptr, cb.len) };
            let view = dp.subslice(offset, offset + len);
            from_cuda(view.load())
        })
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let cb = match self.buffers.lock() {
            Ok(mut g) => g.remove(buffer),
            Err(_) => return,
        };
        if let Some(cb) = cb {
            let _ = self.with_context(|handle| {
                // SAFETY: ptr/len from our buffer table; context is current.
                let dp = unsafe { DevicePtr::from_raw_parts(handle.clone(), cb.ptr, cb.len) };
                let db = unsafe { DeviceBox::from_raw(dp) };
                drop(db); // calls cuMemFree while context is current
                Ok(())
            });
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
        // PTX module loading (cuModuleLoadDataEx) — next commit.
        let _ = desc;
        Err(GpuError::Backend(
            "cuda: shader/PTX loading not yet implemented".into(),
        ))
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        // cuModuleGetFunction — next commit (after create_shader).
        let _ = desc;
        Err(GpuError::Backend(
            "cuda: compute pipelines not yet implemented".into(),
        ))
    }

    fn dispatch(
        &self,
        _pipeline: PipelineHandle,
        _bindings: Bindings<'_>,
        _grid: [u32; 3],
    ) -> Result<()> {
        // cuLaunchKernel — next commit.
        Err(GpuError::Backend(
            "cuda: kernel dispatch not yet implemented".into(),
        ))
    }
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

    #[test]
    fn open_and_buffer_round_trip() {
        let inst = CudaInstance::new();
        let Some(adapter) = inst.enumerate_adapters().into_iter().next() else {
            return; // no CUDA device in CI; skip gracefully
        };
        let device = adapter
            .open(DeviceRequest::default())
            .expect("open failed");
        let buf = device
            .create_buffer(BufferDesc {
                size: 256,
                usage: zengpu_hal::BufferUsage::STORAGE,
                memory: zengpu_hal::MemoryUsage::GpuOnly,
            })
            .expect("create_buffer failed");
        let data: Vec<u8> = (0u8..=255).collect();
        device.write_buffer(buf, 0, &data).expect("write failed");
        let read_back = device.read_buffer(buf, 0, 256).expect("read failed");
        assert_eq!(read_back, data);
        device.destroy_buffer(buf);
    }
}
