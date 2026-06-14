//! ZenGPU CPU reference backend — the conformance oracle (plan D7).
//!
//! Plain Rust over `Vec<u8>` buffers: correctness and determinism over speed.
//! It is the reference the GPU backends are validated against (plan §18), **not**
//! a product fallback (consumers like aurea keep their own CPU paths).

#![forbid(unsafe_code)]

use std::sync::Mutex;

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BackendPreference, BufferDesc, BufferHandle, BufferUsage,
    DeviceRequest, DeviceType, GpuAdapter, GpuDevice, GpuError, GpuInstance, HalCapabilities,
    Result, SlotMap, UsageError, marker,
};

struct CpuBuffer {
    data: Vec<u8>,
    usage: BufferUsage,
}

/// Buffers are stored under the public buffer-handle tag.
type BufferMap = SlotMap<marker::Buffer, CpuBuffer>;

/// A CPU-backed [`GpuDevice`]. All buffers live in host memory behind a mutex,
/// so the device is `Send + Sync` (plan D5).
pub struct CpuDevice {
    buffers: Mutex<BufferMap>,
}

impl CpuDevice {
    /// Create a fresh CPU device.
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(SlotMap::new()),
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a precise stale-handle error for diagnostics (plan §9).
fn stale(handle: BufferHandle, buffers: &BufferMap) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: buffers.generation_at(handle.index()).unwrap_or(u32::MAX),
    })
}

/// Build an out-of-bounds error for a buffer range.
fn out_of_bounds(start: usize, end: usize, len: usize) -> GpuError {
    GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
        "range {start}..{end} exceeds buffer size {len}"
    )))
}

impl GpuDevice for CpuDevice {
    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let buffer = CpuBuffer {
            data: vec![0u8; desc.size as usize],
            usage: desc.usage,
        };
        Ok(self.buffers.lock().unwrap().insert(buffer))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.get(buffer).is_none() {
            return Err(stale(buffer, &buffers));
        }
        let buf = buffers.get_mut(buffer).unwrap();
        let start = offset as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| out_of_bounds(start, usize::MAX, buf.data.len()))?;
        if end > buf.data.len() {
            return Err(out_of_bounds(start, end, buf.data.len()));
        }
        buf.data[start..end].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or_else(|| out_of_bounds(start, usize::MAX, buf.data.len()))?;
        if end > buf.data.len() {
            return Err(out_of_bounds(start, end, buf.data.len()));
        }
        Ok(buf.data[start..end].to_vec())
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        self.buffers.lock().unwrap().remove(buffer);
    }
}

/// A CPU adapter — wraps the single CPU entry in the HAL adapter chain.
pub struct CpuAdapter {
    info: AdapterInfo,
}

impl CpuAdapter {
    pub fn new() -> Self {
        Self {
            info: AdapterInfo {
                name: "ZenGPU CPU Reference".to_string(),
                vendor: 0,
                device: 0,
                device_type: DeviceType::Cpu,
                backend: BackendPreference::Cpu,
            },
        }
    }
}

impl Default for CpuAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuAdapter for CpuAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::compute_only()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        Ok(Box::new(CpuDevice::new()))
    }
}

/// Instance for the CPU backend — always available, always returns the single
/// CPU adapter regardless of the adapter request.
pub struct CpuInstance;

impl GpuInstance for CpuInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        vec![Box::new(CpuAdapter::new())]
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        Some(Box::new(CpuAdapter::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_hal::MemoryUsage;

    fn rw_desc(size: u64) -> BufferDesc {
        BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), vec![1, 2, 3, 4]);
        // Offset read.
        assert_eq!(dev.read_buffer(h, 2, 2).unwrap(), vec![3, 4]);
    }

    #[test]
    fn read_without_readback_usage_fails() {
        let dev = CpuDevice::new();
        let h = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::MissingUsage { needed: "READBACK", .. })
        ));
    }

    #[test]
    fn use_after_destroy_is_stale() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.destroy_buffer(h);
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        match err {
            GpuError::InvalidUsage(UsageError::StaleHandle {
                expected_gen,
                actual_gen,
                ..
            }) => assert_ne!(expected_gen, actual_gen),
            other => panic!("expected StaleHandle, got {other}"),
        }
    }

    #[test]
    fn out_of_bounds_write_fails() {
        let dev = CpuDevice::new();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));
    }

    #[test]
    fn reports_compute_only_capabilities() {
        let dev = CpuDevice::new();
        assert!(dev.capabilities().compute);
        assert!(!dev.capabilities().graphics);
    }

    #[test]
    fn usable_as_dyn_device() {
        let dev: Box<dyn GpuDevice> = Box::new(CpuDevice::new());
        let h = dev.create_buffer(rw_desc(2)).unwrap();
        dev.write_buffer(h, 0, &[9, 8]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 2).unwrap(), vec![9, 8]);
    }

    #[test]
    fn adapter_opens_cpu_device() {
        let adapter = CpuAdapter::new();
        assert_eq!(adapter.info().name, "ZenGPU CPU Reference");
        assert!(!adapter.capabilities().graphics);
        assert!(adapter.capabilities().compute);
        let dev = adapter.open(DeviceRequest::default()).unwrap();
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn instance_enumerates_one_adapter() {
        let inst = CpuInstance;
        let adapters = inst.enumerate_adapters();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].info().name, "ZenGPU CPU Reference");
    }

    #[test]
    fn instance_request_adapter_always_returns_cpu() {
        let inst = CpuInstance;
        let adapter = inst.request_adapter(AdapterRequest::default()).unwrap();
        let dev = adapter.open(DeviceRequest::default()).unwrap();
        assert!(dev.capabilities().compute);
    }
}
