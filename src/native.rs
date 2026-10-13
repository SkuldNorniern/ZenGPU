//! The platform's own graphics backend, picked at compile time: Metal on
//! macOS, Vulkan everywhere else.
//!
//! The aliases name concrete types, so graphics code written against them
//! (and the [`GraphicsDevice`](crate::GraphicsDevice) traits) builds for
//! whichever backend the platform prefers, with no runtime dispatch. Both
//! backends answer to the same calls: `NativeInstance::new_with_surface()`,
//! `graphics_adapters()`, `request_graphics_adapter()`,
//! `graphics_adapter_by_uuid()`, `NativeAdapter::uuid()` and
//! `open_graphics()`, then on the device `bind_texture`,
//! `register_color_target`, `create_depth_target`, `register_depth_target`,
//! `unregister_render_target`, `wait_idle` and `submit_and_wait`. Shaders come from one `zsl!` through [`shader`].

use zengpu_hal::{BackendPreference, ShaderDesc};
use zengpu_spirv::ZslShader;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use zengpu_metal::{
    DepthTarget as NativeDepthTarget, MetalAdapter as NativeAdapter,
    MetalCommandList as NativeCommandList, MetalDevice as NativeDevice, MetalFrame as NativeFrame,
    MetalInstance as NativeInstance, MetalSurface as NativeSurface,
    OffscreenTarget as NativeOffscreenTarget,
};

/// The backend the aliases stand for.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const BACKEND: BackendPreference = BackendPreference::Metal;

#[cfg(all(not(all(target_os = "macos", feature = "metal")), feature = "vulkan"))]
pub use zengpu_vulkan::{
    DepthTarget as NativeDepthTarget, OffscreenTarget as NativeOffscreenTarget,
    VulkanAdapter as NativeAdapter, VulkanCommandList as NativeCommandList,
    VulkanDevice as NativeDevice, VulkanFrame as NativeFrame, VulkanInstance as NativeInstance,
    VulkanSurface as NativeSurface,
};

/// The backend the aliases stand for.
#[cfg(all(not(all(target_os = "macos", feature = "metal")), feature = "vulkan"))]
pub const BACKEND: BackendPreference = BackendPreference::Vulkan;

/// `shader`'s artifact for the native backend: MSL on Metal, SPIR-V on
/// Vulkan.
pub fn shader(shader: &ZslShader) -> ShaderDesc<'_> {
    shader.for_backend(BACKEND).0
}
