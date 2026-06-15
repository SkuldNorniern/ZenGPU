//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+.

pub mod adapter;
pub mod depth_target;
pub mod device;
pub mod frame_graph;
pub mod instance;
pub mod offscreen;
pub mod swapchain;

pub use adapter::VulkanAdapter;
pub use ash;
pub use ash::vk;
pub use depth_target::{DepthTarget, DEPTH_FORMAT};
pub use device::VulkanDevice;
pub use frame_graph::{AttachmentUsage, FrameGraph, ResourceId};
pub use instance::VulkanInstance;
pub use offscreen::{OffscreenTarget, SampledImageView};
pub use swapchain::{BeginFrame, DeviceContext, Swapchain};
