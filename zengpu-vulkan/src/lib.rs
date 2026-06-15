//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+ (plan D15).

pub mod adapter;
pub mod device;
pub mod frame_graph;
pub mod instance;
pub mod offscreen;
pub mod swapchain;
pub mod swapchain_2d;

pub use ash;
pub use ash::vk;
pub use adapter::VulkanAdapter;
pub use device::VulkanDevice;
pub use instance::VulkanInstance;
pub use frame_graph::{AttachmentUsage, FrameGraph, ResourceId};
pub use offscreen::OffscreenTarget;
pub use swapchain::{BeginFrame, DeviceContext, Swapchain};
pub use swapchain_2d::{
    CircleInstance, DrawRef, Frame2d, GradientInstance, IMAGE_SLOTS, ImageInstance, RectInstance,
    TextInstance, Vulkan2dSurface,
};
