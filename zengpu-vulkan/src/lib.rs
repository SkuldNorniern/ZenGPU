//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+ (plan D15).

pub mod adapter;
pub mod device;
pub mod instance;
pub mod swapchain;
pub mod swapchain_2d;
pub mod swapchain_textured;

pub use adapter::VulkanAdapter;
pub use device::VulkanDevice;
pub use instance::VulkanInstance;
pub use swapchain::VulkanSwapchain;
pub use swapchain_2d::{RectInstance, Vulkan2dSurface};
pub use swapchain_textured::{BINDLESS_CAPACITY, VulkanTexturedSwapchain};
