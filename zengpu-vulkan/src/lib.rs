//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+ (plan D15).

mod adapter;
mod device;
mod instance;
mod swapchain;

pub use adapter::VulkanAdapter;
pub use device::VulkanDevice;
pub use instance::VulkanInstance;
pub use swapchain::VulkanSwapchain;
