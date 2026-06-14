//! Cross-backend conformance: CPU oracle vs Vulkan (plan M1.5).
//!
//! Each test gracefully skips if no Vulkan driver is present.

use zengpu_conformance::{compare_full, run_buffer_suite};
use zengpu_cpu::CpuDevice;
use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};
use zengpu_vulkan::VulkanInstance;

fn cpu_device() -> Box<dyn zengpu_hal::GpuDevice> {
    Box::new(CpuDevice::new())
}

fn vulkan_device() -> Option<Box<dyn zengpu_hal::GpuDevice>> {
    let inst = VulkanInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    adapter.open(DeviceRequest::default()).ok()
}

#[test]
fn cpu_buffer_suite() {
    run_buffer_suite("cpu", &*cpu_device());
}

#[test]
fn vulkan_buffer_suite() {
    let Some(dev) = vulkan_device() else { return };
    run_buffer_suite("vulkan", &*dev);
}

#[test]
fn cpu_vs_vulkan() {
    let Some(vk) = vulkan_device() else { return };
    compare_full("cpu", &*cpu_device(), "vulkan", &*vk);
}
