//! ZenGPU Vulkan backend, Vulkan 1.2+.
//!
//! Implements the split-HAL traits ([`GpuInstance`], [`GpuAdapter`], [`GpuDevice`]) against `ash`.
//! Host-visible buffer create, upload, read and destroy.

mod adapter;
mod device;
mod instance;

pub use adapter::VulkanAdapter;
pub use device::VulkanDevice;
pub use instance::VulkanInstance;
