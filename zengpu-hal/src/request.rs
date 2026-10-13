
use std::path::PathBuf;

use crate::types::{BackendPreference, Features, PowerPreference};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdapterRequest {
    /// Preferred backend. `Auto` picks the best available native one.
    pub backend: BackendPreference,
    pub power: PowerPreference,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceRequest {
    pub backend: BackendPreference,
    pub power: PowerPreference,
    /// Features that must be present, or device creation fails.
    pub required: Features,
    /// Features to enable if available, ignored if not.
    pub optional: Features,
    /// Optional pipeline cache file, loaded and saved across device lifetimes.
    pub pipeline_cache_path: Option<PathBuf>,
}

/// Transfer paths a backend may implement.
///
/// The default of each refuses with `Unsupported`, so check before relying on one and fall back
/// when it is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransferCaps {
    pub texture_region_write: bool,
    pub buffer_texture_copy: bool,
    /// Uploads reuse staging memory instead of allocating per call.
    pub staged_uploads: bool,
    pub async_transfers: bool,
}

/// `false` means refused or undefined. Check before relying on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FormatCaps {
    pub sampled: bool,
    /// Sampled with linear filtering, not only nearest.
    pub filterable: bool,
    pub render_target: bool,
    pub blendable: bool,
    pub storage: bool,
}

/// Drivers move `budget` and `usage` over time. Read them again rather than caching them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Device-local memory in total (all device-local heaps).
    pub device_local: u64,
    /// Budget, in bytes, before the driver starts evicting or failing.
    pub budget: u64,
    pub usage: u64,
}

impl MemoryBudget {
    pub fn headroom(&self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HalCapabilities {
    pub graphics: bool,
    pub compute: bool,
    pub features: Features,
    pub transfers: TransferCaps,
}

impl HalCapabilities {
    pub fn all() -> Self {
        Self {
            graphics: true,
            compute: true,
            features: Features::COMPUTE | Features::GRAPHICS,
            transfers: TransferCaps {
                texture_region_write: false,
                buffer_texture_copy: false,
                staged_uploads: false,
                async_transfers: false,
            },
        }
    }

    pub fn compute_only() -> Self {
        Self {
            graphics: false,
            compute: true,
            features: Features::COMPUTE,
            transfers: TransferCaps {
                texture_region_write: false,
                buffer_texture_copy: false,
                staged_uploads: false,
                async_transfers: false,
            },
        }
    }

    pub fn with_features(mut self, features: Features) -> Self {
        self.features |= features;
        self
    }

    pub fn with_transfers(mut self, transfers: TransferCaps) -> Self {
        self.transfers = transfers;
        self
    }
}

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
