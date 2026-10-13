//! Platform graphics backend, picked at compile time: Metal on macOS, Vulkan elsewhere.
//!
//! No runtime dispatch: both backends expose the same calls. Shaders: one `zsl!` via [`shader`].

use zengpu_hal::{BackendPreference, ShaderDesc};
use zengpu_spirv::ZslShader;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use zengpu_metal::{
    DepthTarget as NativeDepthTarget, MetalAdapter as NativeAdapter,
    MetalCommandList as NativeCommandList, MetalDevice as NativeDevice, MetalFrame as NativeFrame,
    MetalInstance as NativeInstance, MetalSurface as NativeSurface,
};

/// The backend the aliases stand for.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const BACKEND: BackendPreference = BackendPreference::Metal;

#[cfg(all(not(all(target_os = "macos", feature = "metal")), feature = "vulkan"))]
pub use zengpu_vulkan::{
    DepthTarget as NativeDepthTarget, VulkanAdapter as NativeAdapter,
    VulkanCommandList as NativeCommandList, VulkanDevice as NativeDevice,
    VulkanFrame as NativeFrame, VulkanInstance as NativeInstance, VulkanSurface as NativeSurface,
};

/// The backend the aliases stand for.
#[cfg(all(not(all(target_os = "macos", feature = "metal")), feature = "vulkan"))]
pub const BACKEND: BackendPreference = BackendPreference::Vulkan;

/// Returns the native backend's artifact for `shader`: MSL on Metal, SPIR-V on Vulkan.
pub fn shader(shader: &ZslShader) -> ShaderDesc<'_> {
    shader.for_backend(BACKEND).0
}
