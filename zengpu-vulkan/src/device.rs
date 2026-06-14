//! Vulkan logical device and buffer operations (plan M1 / G2).

use std::sync::{Arc, Mutex};

use ash::{Device, vk};
use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, DeviceRequest, GpuDevice, GpuError, HalCapabilities,
    MemoryUsage, Result, SlotMap, UsageError, marker,
};

use crate::instance::VulkanShared;

/// Shared logical device state — owned by `Arc` so swapchains can hold a ref.
pub(crate) struct VulkanDeviceInner {
    pub shared: Arc<VulkanShared>,
    pub device: Device,
    pub physical: vk::PhysicalDevice,
    pub queue_family: u32,
    pub queue: vk::Queue,
}

// ash::Device is Send + Sync; vk::PhysicalDevice is a u64.
unsafe impl Send for VulkanDeviceInner {}
unsafe impl Sync for VulkanDeviceInner {}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    usage: BufferUsage,
    mapped: *mut u8,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

/// Vulkan logical device implementing [`GpuDevice`].
pub struct VulkanDevice {
    pub(crate) inner: Arc<VulkanDeviceInner>,
    buffers: Mutex<SlotMap<marker::Buffer, VulkanBuffer>>,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub(crate) fn create(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        _req: DeviceRequest,
        extra_extensions: &[*const i8],
    ) -> Result<Self> {
        let queue_family = compute_queue_family(&shared.instance, physical)
            .ok_or_else(|| GpuError::Backend("no compute queue family".to_string()))?;

        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extra_extensions.len() as u32,
            pp_enabled_extension_names: if extra_extensions.is_empty() {
                std::ptr::null()
            } else {
                extra_extensions.as_ptr()
            },
            ..Default::default()
        };

        let device = unsafe {
            shared
                .instance
                .create_device(physical, &device_create_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateDevice: {e}")))?
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            inner: Arc::new(VulkanDeviceInner {
                shared,
                device,
                physical,
                queue_family,
                queue,
            }),
            buffers: Mutex::new(SlotMap::new()),
        })
    }

    pub(crate) fn new(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(shared, physical, req, &[])
    }

    pub(crate) fn new_with_swapchain(
        shared: Arc<VulkanShared>,
        physical: vk::PhysicalDevice,
        req: DeviceRequest,
    ) -> Result<Self> {
        Self::create(
            shared,
            physical,
            req,
            &[ash::khr::swapchain::NAME.as_ptr()],
        )
    }

    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        let mem_props = unsafe {
            self.inner
                .shared
                .instance
                .get_physical_device_memory_properties(self.inner.physical)
        };
        (0..mem_props.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
        })
    }
}

fn compute_queue_family(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(physical) }
        .into_iter()
        .enumerate()
        .find_map(|(i, f)| {
            if f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                Some(i as u32)
            } else {
                None
            }
        })
}

fn buffer_usage_to_vk(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsage::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(BufferUsage::TRANSFER_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::TRANSFER_DST) || usage.contains(BufferUsage::READBACK) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if flags.is_empty() {
        flags = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    }
    flags
}

fn memory_usage_to_vk(usage: MemoryUsage) -> vk::MemoryPropertyFlags {
    match usage {
        MemoryUsage::GpuOnly | MemoryUsage::Pooled => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryUsage::Upload
        | MemoryUsage::CpuToGpu
        | MemoryUsage::Transient
        | MemoryUsage::Persistent => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUsage::Readback => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
        }
    }
}

fn stale(handle: BufferHandle, buffers: &SlotMap<marker::Buffer, VulkanBuffer>) -> GpuError {
    GpuError::InvalidUsage(UsageError::StaleHandle {
        index: handle.index(),
        expected_gen: handle.generation(),
        actual_gen: buffers.generation_at(handle.index()).unwrap_or(u32::MAX),
    })
}

impl GpuDevice for VulkanDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities::all()
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        let vk_usage = buffer_usage_to_vk(desc.usage);
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size,
            usage: vk_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            self.inner
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateBuffer: {e}")))?
        };

        let mem_reqs =
            unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };

        let preferred_flags = memory_usage_to_vk(desc.memory);
        let type_index = self
            .find_memory_type(mem_reqs.memory_type_bits, preferred_flags)
            .or_else(|| {
                if preferred_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
                    self.find_memory_type(
                        mem_reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                } else {
                    None
                }
            });

        let type_index = match type_index {
            Some(i) => i,
            None => {
                unsafe { self.inner.device.destroy_buffer(buffer, None) };
                return Err(GpuError::OutOfMemory(desc.memory));
            }
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index: type_index,
            ..Default::default()
        };

        let memory = unsafe {
            match self.inner.device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(_) => {
                    self.inner.device.destroy_buffer(buffer, None);
                    return Err(GpuError::OutOfMemory(desc.memory));
                }
            }
        };

        if let Err(e) = unsafe {
            self.inner.device.bind_buffer_memory(buffer, memory, 0)
        } {
            unsafe {
                self.inner.device.destroy_buffer(buffer, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::Backend(format!("vkBindBufferMemory: {e}")));
        }

        let is_host_visible = preferred_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            || matches!(desc.memory, MemoryUsage::Readback);

        let mapped = if is_host_visible {
            match unsafe {
                self.inner.device.map_memory(
                    memory,
                    0,
                    desc.size,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(ptr) => ptr as *mut u8,
                Err(e) => {
                    unsafe {
                        self.inner.device.destroy_buffer(buffer, None);
                        self.inner.device.free_memory(memory, None);
                    }
                    return Err(GpuError::Backend(format!("vkMapMemory: {e}")));
                }
            }
        } else {
            std::ptr::null_mut()
        };

        Ok(self.buffers.lock().unwrap().insert(VulkanBuffer {
            buffer,
            memory,
            size: desc.size,
            usage: desc.usage,
            mapped,
        }))
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;

        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "write_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(data.len()).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..{end} exceeds buffer size {}",
                buf.size
            ))));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.mapped.add(start), data.len());
        }
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
        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "read_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(len as usize).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..{end} exceeds buffer size {}",
                buf.size
            ))));
        }
        let mut out = vec![0u8; len as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.mapped.add(start), out.as_mut_ptr(), len as usize);
        }
        Ok(out)
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buf) = buffers.remove(buffer) {
            unsafe {
                if !buf.mapped.is_null() {
                    self.inner.device.unmap_memory(buf.memory);
                }
                self.inner.device.destroy_buffer(buf.buffer, None);
                self.inner.device.free_memory(buf.memory, None);
            }
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let mut buffers = self.buffers.lock().unwrap();
        for buf in buffers.drain() {
            unsafe {
                if !buf.mapped.is_null() {
                    self.inner.device.unmap_memory(buf.memory);
                }
                self.inner.device.destroy_buffer(buf.buffer, None);
                self.inner.device.free_memory(buf.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};
    use crate::instance::VulkanInstance;

    fn try_device() -> Option<Box<dyn GpuDevice>> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    fn rw_desc(size: u64) -> BufferDesc {
        BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            memory: MemoryUsage::Upload,
        }
    }

    #[test]
    fn buffer_roundtrip() {
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.write_buffer(h, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(dev.read_buffer(h, 0, 4).unwrap(), [1, 2, 3, 4]);
        assert_eq!(dev.read_buffer(h, 2, 2).unwrap(), [3, 4]);
    }

    #[test]
    fn read_without_readback_usage_fails() {
        let Some(dev) = try_device() else { return };
        let h = dev
            .create_buffer(BufferDesc {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::Upload,
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
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        dev.destroy_buffer(h);
        let err = dev.read_buffer(h, 0, 4).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::StaleHandle { .. })
        ));
    }

    #[test]
    fn out_of_bounds_write_fails() {
        let Some(dev) = try_device() else { return };
        let h = dev.create_buffer(rw_desc(4)).unwrap();
        let err = dev.write_buffer(h, 2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            GpuError::InvalidUsage(UsageError::BindingMismatch(_))
        ));
    }

    #[test]
    fn reports_graphics_and_compute() {
        let Some(dev) = try_device() else { return };
        assert!(dev.capabilities().graphics);
        assert!(dev.capabilities().compute);
    }
}
