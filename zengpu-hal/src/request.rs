//! Adapter/device request and capability descriptors.

use crate::types::{BackendPreference, Features, PowerPreference};

/// What to ask an adapter for during selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdapterRequest {
    /// Preferred backend; `Auto` picks the best available native one.
    pub backend: BackendPreference,
    pub power: PowerPreference,
}

/// What to ask a device for at creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceRequest {
    pub backend: BackendPreference,
    pub power: PowerPreference,
    /// Features that must be present, or device creation fails.
    pub required: Features,
    /// Features to enable if available, ignored if not.
    pub optional: Features,
}

/// Which HAL capabilities a backend implements. A backend may provide
/// graphics, compute, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HalCapabilities {
    pub graphics: bool,
    pub compute: bool,
    /// Feature set implemented and enabled on this adapter/device path.
    pub features: Features,
}

impl HalCapabilities {
    /// Capabilities for a graphics + compute backend (e.g. Vulkan).
    pub fn all() -> Self {
        Self {
            graphics: true,
            compute: true,
            features: Features::COMPUTE | Features::GRAPHICS,
        }
    }

    /// Capabilities for a compute-only backend (e.g. CUDA, the CPU oracle).
    pub fn compute_only() -> Self {
        Self {
            graphics: false,
            compute: true,
            features: Features::COMPUTE,
        }
    }

    /// Add backend-specific feature flags to this capability description.
    pub fn with_features(mut self, features: Features) -> Self {
        self.features |= features;
        self
    }
}

/// Portable device limits relevant to compute workloads and descriptor sizing.
/// Unsupported or unavailable values are zero.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeviceLimits {
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_dispatch_size: [u32; 3],
    pub max_storage_buffer_range: u64,
    pub max_push_constant_size: u32,
    pub max_storage_buffers: u32,
    pub max_sampled_textures: u32,
    pub max_update_after_bind_descriptors: u32,
    pub max_memory_allocations: u32,
    pub timestamp_supported: bool,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period_ns: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_constructors() {
        assert!(HalCapabilities::all().graphics);
        assert!(HalCapabilities::all().compute);
        assert!(!HalCapabilities::compute_only().graphics);
        assert!(HalCapabilities::compute_only().compute);
        assert!(
            HalCapabilities::all()
                .features
                .contains(Features::GRAPHICS | Features::COMPUTE)
        );
    }

    #[test]
    fn device_request_default_requires_nothing() {
        let r = DeviceRequest::default();
        assert!(r.required.is_empty());
        assert_eq!(r.backend, BackendPreference::Auto);
    }
}
