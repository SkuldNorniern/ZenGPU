//! ZenGPU Vulkan backend — native-first GPU runtime on Vulkan 1.2+.

pub mod adapter;
pub(crate) mod command_list;
pub mod depth_target;
pub mod device;
pub mod frame_graph;
pub mod instance;
pub mod offscreen;
pub(crate) mod surface;
pub mod swapchain;

pub use adapter::VulkanAdapter;
pub use command_list::VulkanCommandList;
pub use depth_target::{DEPTH_FORMAT, DepthTarget};
pub use device::VulkanDevice;
pub use frame_graph::{AttachmentUsage, FrameGraph, ResourceId};
pub use instance::VulkanInstance;
pub use offscreen::OffscreenTarget;
pub use surface::{VulkanFrame, VulkanSurface};
pub use swapchain::{BeginFrame, DeviceContext, Swapchain};

use ash::vk;

#[cfg(test)]
pub(crate) fn test_gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    // Most unit tests need their own logical device. Keep those heavyweight
    // fixtures sequential inside one test process; explicit same-device tests
    // still exercise queue and command-recording concurrency directly.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ── Format conversion ─────────────────────────────────────────────────────────

/// Convert a HAL [`zengpu_hal::Format`] to the matching `vk::Format`.
pub(crate) fn to_vk_format(f: zengpu_hal::Format) -> vk::Format {
    use zengpu_hal::Format;
    match f {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
        Format::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
    }
}

/// Convert a `vk::Format` back to the HAL [`zengpu_hal::Format`], or `None`
/// for Vulkan formats that have no HAL counterpart.
pub(crate) fn from_vk_format(f: vk::Format) -> Option<zengpu_hal::Format> {
    use zengpu_hal::Format;
    Some(match f {
        vk::Format::R8G8B8A8_UNORM => Format::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => Format::Rgba8UnormSrgb,
        vk::Format::B8G8R8A8_UNORM => Format::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => Format::Bgra8UnormSrgb,
        vk::Format::R32_SFLOAT => Format::R32Float,
        vk::Format::D32_SFLOAT => Format::Depth32Float,
        vk::Format::D24_UNORM_S8_UINT => Format::Depth24PlusStencil8,
        _ => return None,
    })
}
