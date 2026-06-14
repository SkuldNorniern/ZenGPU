//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+ (plan D15).
//!
//! Implements the split-HAL traits ([`GpuInstance`], [`GpuAdapter`],
//! [`GpuDevice`]) against `ash`. M1 scope: buffer create/upload/read/destroy
//! on host-visible memory. Graphics and async-compute follow in the G-track.

mod adapter;
mod device;
mod instance;

pub use adapter::VulkanAdapter;
pub use device::VulkanDevice;
pub use instance::VulkanInstance;
