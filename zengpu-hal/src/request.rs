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
}

impl HalCapabilities {
    /// Capabilities for a graphics + compute backend (e.g. Vulkan).
    pub const fn all() -> Self {
        Self {
            graphics: true,
            compute: true,
        }
    }

    /// Capabilities for a compute-only backend (e.g. CUDA, the CPU oracle).
    pub const fn compute_only() -> Self {
        Self {
            graphics: false,
            compute: true,
        }
    }
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
    }

    #[test]
    fn device_request_default_requires_nothing() {
        let r = DeviceRequest::default();
        assert!(r.required.is_empty());
        assert_eq!(r.backend, BackendPreference::Auto);
    }
}
