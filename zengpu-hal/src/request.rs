
use crate::types::{BackendPreference, Features, PowerPreference};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdapterRequest {
    /// Preferred backend. `Auto` picks the best available native one.
    pub backend: BackendPreference,
    pub power: PowerPreference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceRequest {
    pub backend: BackendPreference,
    pub power: PowerPreference,
    /// Features that must be present, or device creation fails.
    pub required: Features,
    /// Features to enable if available, ignored if not.
    pub optional: Features,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HalCapabilities {
    pub graphics: bool,
    pub compute: bool,
}

impl HalCapabilities {
    pub const fn all() -> Self {
        Self {
            graphics: true,
            compute: true,
        }
    }

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
