//! ZenGPU Apple Metal backend — graphics + compute on macOS/iOS.
//!
//! On macOS the instance enumerates all `MTLDevice` objects (including eGPUs).
//! On non-Apple platforms the instance compiles but returns no adapters.
//! Device open (`MTLDevice` creation, command queues, buffers) lands in the
//! next commit once the surface extension story is settled.

use zengpu_hal::{
    AdapterInfo, AdapterRequest, BufferDesc, BufferHandle, DeviceRequest, GpuAdapter, GpuDevice,
    GpuError, GpuInstance, HalCapabilities, Result, SamplerDesc, SamplerHandle, TextureDesc,
    TextureHandle,
};

#[cfg(target_os = "macos")]
use zengpu_hal::{BackendPreference, DeviceType, SlotMap, marker};

// ── MetalInstance ─────────────────────────────────────────────────────────────

/// Entry-point for the Metal backend.
///
/// On macOS, [`enumerate_adapters`] returns one entry per `MTLDevice` —
/// including Apple Silicon integrated GPUs and any connected eGPUs.
/// On other platforms it always returns empty.
///
/// [`enumerate_adapters`]: MetalInstance::enumerate_adapters
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetalInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInstance for MetalInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        #[cfg(target_os = "macos")]
        {
            metal::Device::all()
                .into_iter()
                .map(|dev| {
                    let device_type = if dev.is_low_power() {
                        DeviceType::Integrated
                    } else {
                        DeviceType::Discrete
                    };
                    let info = AdapterInfo {
                        name: dev.name().to_string(),
                        vendor: 0x106b, // Apple PCI vendor ID
                        device: 0,
                        device_type,
                        backend: BackendPreference::Metal,
                    };
                    Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Vec::new()
        }
    }

    fn request_adapter(&self, _req: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        // On macOS, prefer the non-low-power device if multiple are present.
        #[cfg(target_os = "macos")]
        {
            let all = self.enumerate_adapters();
            // system_default() is the OS-preferred device; use it.
            metal::Device::system_default().map(|dev| {
                let device_type = if dev.is_low_power() {
                    DeviceType::Integrated
                } else {
                    DeviceType::Discrete
                };
                let info = AdapterInfo {
                    name: dev.name().to_string(),
                    vendor: 0x106b,
                    device: 0,
                    device_type,
                    backend: BackendPreference::Metal,
                };
                Box::new(MetalAdapter { info }) as Box<dyn GpuAdapter>
            })
            // Fall back to first enumerated device if system_default is None.
            .or_else(|| all.into_iter().next())
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

// ── MetalAdapter ──────────────────────────────────────────────────────────────

pub struct MetalAdapter {
    info: AdapterInfo,
}

impl GpuAdapter for MetalAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn open(&self, _req: DeviceRequest) -> Result<Box<dyn GpuDevice>> {
        #[cfg(target_os = "macos")]
        {
            let device = metal::Device::system_default()
                .ok_or_else(|| GpuError::Backend("metal: no MTLDevice available".into()))?;
            let queue = device.new_command_queue();
            Ok(Box::new(MetalDevice {
                inner: MacDevice {
                    device,
                    queue,
                    buffers: std::sync::Mutex::new(SlotMap::default()),
                },
            }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(GpuError::Backend(
                "metal: unavailable on this platform".into(),
            ))
        }
    }
}

// ── MetalDevice ───────────────────────────────────────────────────────────────

/// A GPU-resident buffer. On Apple Silicon, `Shared` storage is host-visible and
/// device-visible (unified memory), so reads/writes are plain `memcpy`.
#[cfg(target_os = "macos")]
struct MetalBuffer {
    buf: metal::Buffer,
    size: u64,
}

#[cfg(target_os = "macos")]
struct MacDevice {
    device: metal::Device,
    #[allow(dead_code)] // used once compute/graphics submission lands
    queue: metal::CommandQueue,
    buffers: std::sync::Mutex<SlotMap<marker::Buffer, MetalBuffer>>,
}

/// An opened Metal device. Buffers today; compute/graphics submission and the
/// ZSL→MSL shader path follow.
pub struct MetalDevice {
    #[cfg(target_os = "macos")]
    inner: MacDevice,
}

// SAFETY: the contained Metal objects are reference-counted Obj-C handles; all
// mutable state (the buffer slot map) is guarded by a Mutex, and no raw pointer
// is shared across threads.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalDevice {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalDevice {}

impl GpuDevice for MetalDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        #[cfg(target_os = "macos")]
        {
            // Metal rejects zero-length buffers; round up to 1 byte.
            let len = desc.size.max(1);
            let buf = self
                .inner
                .device
                .new_buffer(len, metal::MTLResourceOptions::StorageModeShared);
            let mut buffers = self.inner.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            Ok(buffers.insert(MetalBuffer { buf, size: desc.size }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = desc;
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self.inner.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            if offset + data.len() as u64 > b.size {
                return Err(GpuError::Backend("metal: write out of bounds".into()));
            }
            // SAFETY: Shared-storage contents() is a valid host pointer for `size`
            // bytes; the bounds check above keeps the copy in range.
            unsafe {
                let ptr = (b.buf.contents() as *mut u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (buffer, offset, data);
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self.inner.buffers.lock().map_err(|_| GpuError::DeviceLost)?;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| GpuError::Backend("metal: invalid buffer handle".into()))?;
            if offset + len > b.size {
                return Err(GpuError::Backend("metal: read out of bounds".into()));
            }
            let mut out = vec![0u8; len as usize];
            // SAFETY: as above; the bounds check keeps the copy within `size`.
            unsafe {
                let ptr = (b.buf.contents() as *const u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len as usize);
            }
            Ok(out)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (buffer, offset, len);
            Err(GpuError::Backend("metal: unavailable".into()))
        }
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut buffers) = self.inner.buffers.lock() {
                buffers.remove(buffer);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = buffer;
        }
    }

    fn create_texture(&self, _desc: TextureDesc) -> Result<TextureHandle> {
        Err(GpuError::Backend("metal: textures not yet implemented".into()))
    }

    fn upload_texture_data(&self, _texture: TextureHandle, _data: &[u8]) -> Result<()> {
        Err(GpuError::Backend("metal: textures not yet implemented".into()))
    }

    fn destroy_texture(&self, _texture: TextureHandle) {}

    fn create_sampler(&self, _desc: SamplerDesc) -> Result<SamplerHandle> {
        Err(GpuError::Backend("metal: samplers not yet implemented".into()))
    }

    fn destroy_sampler(&self, _sampler: SamplerHandle) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_constructs_without_panic() {
        let inst = MetalInstance::new();
        let _ = inst.enumerate_adapters();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn enumerates_at_least_one_adapter_on_macos() {
        let inst = MetalInstance::new();
        let adapters = inst.enumerate_adapters();
        assert!(!adapters.is_empty(), "expected at least one Metal adapter on macOS");
        for a in &adapters {
            assert!(!a.info().name.is_empty());
            assert!(a.capabilities().graphics);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn buffer_write_read_round_trip() {
        let inst = MetalInstance::new();
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return; // no Metal device in this environment
        };
        let device = adapter.open(DeviceRequest::default()).expect("open MTLDevice");

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes = bytemuck_cast(&data);
        let buf = device
            .create_buffer(BufferDesc {
                size: bytes.len() as u64,
                usage: zengpu_hal::BufferUsage::STORAGE | zengpu_hal::BufferUsage::READBACK,
                memory: zengpu_hal::MemoryUsage::Upload,
            })
            .expect("create buffer");
        device.write_buffer(buf, 0, bytes).expect("write");
        let out = device.read_buffer(buf, 0, bytes.len() as u64).expect("read");
        assert_eq!(out, bytes);

        // Out-of-bounds write is rejected.
        assert!(device.write_buffer(buf, bytes.len() as u64, &[0u8; 4]).is_err());
        device.destroy_buffer(buf);
    }

    #[cfg(target_os = "macos")]
    fn bytemuck_cast(data: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding/invalid bit patterns; viewing as bytes is sound.
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
    }
}
