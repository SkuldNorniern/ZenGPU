//! Adapter/device request and capability descriptors.

use std::path::PathBuf;

use crate::types::{BackendPreference, Features, PowerPreference};

/// What to ask an adapter for during selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdapterRequest {
    /// Preferred backend; `Auto` picks the best available native one.
    pub backend: BackendPreference,
    pub power: PowerPreference,
}

/// What to ask a device for at creation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceRequest {
    pub backend: BackendPreference,
    pub power: PowerPreference,
    /// Features that must be present, or device creation fails.
    pub required: Features,
    /// Features to enable if available, ignored if not.
    pub optional: Features,
    /// Optional path used by supporting backends to load and persist compiled
    /// pipeline cache data across device lifetimes.
    pub pipeline_cache_path: Option<PathBuf>,
}

/// Which ways of moving data a backend implements itself. The trait has a
/// default for every one, but the default of these refuses with
/// `Unsupported`: check before relying on one, and fall back where it is
/// missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransferCaps {
    /// [`crate::GpuDevice::write_textures`] updates a rectangle of a
    /// texture and keeps the rest.
    pub texture_region_write: bool,
    /// [`crate::GpuDevice::copy_buffer_to_texture`] and
    /// [`crate::GpuDevice::copy_texture_region_to_buffer`] copy on the
    /// device, with no trip through the host.
    pub buffer_texture_copy: bool,
    /// Uploads reuse staging memory instead of allocating per call.
    pub staged_uploads: bool,
    /// Transfers return before the copy is done, with a submission to poll.
    pub async_transfers: bool,
}

/// What a texture of one format can be used for on a device, from
/// [`crate::GpuDevice::format_caps`]. A use reported `false` is refused or
/// undefined there: check before relying on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FormatCaps {
    /// Read by shaders.
    pub sampled: bool,
    /// Sampled with linear filtering, not only nearest.
    pub filterable: bool,
    /// Drawn into as a colour (or depth) attachment.
    pub render_target: bool,
    /// Blended into as a colour attachment.
    pub blendable: bool,
    /// Written by compute shaders as a storage texture.
    pub storage: bool,
}

/// Device-local memory and this process's share of it, from
/// [`crate::GpuDevice::memory_budget`]. Drivers move `budget` and `usage`
/// over time, so read it again rather than caching it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Device-local memory in total (all device-local heaps).
    pub device_local: u64,
    /// What this process may use of it before the driver starts evicting or
    /// failing: the budget.
    pub budget: u64,
    /// What this process uses now.
    pub usage: u64,
}

impl MemoryBudget {
    /// Bytes this process can still allocate before reaching its budget.
    pub fn headroom(&self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }
}

/// Which HAL capabilities a backend implements. A backend may provide
/// graphics, compute, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HalCapabilities {
    pub graphics: bool,
    pub compute: bool,
    /// Feature set implemented and enabled on this adapter/device path.
    pub features: Features,
    /// The transfer paths it implements.
    pub transfers: TransferCaps,
}

impl HalCapabilities {
    /// Capabilities for a graphics + compute backend (e.g. Vulkan).
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

    /// Capabilities for a compute-only backend (e.g. CUDA, the CPU oracle).
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

    /// Add backend-specific feature flags to this capability description.
    pub fn with_features(mut self, features: Features) -> Self {
        self.features |= features;
        self
    }

    /// These capabilities with `transfers` as the transfer paths.
    pub fn with_transfers(mut self, transfers: TransferCaps) -> Self {
        self.transfers = transfers;
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
    /// Slots for storage textures, the ones compute kernels store to through
    /// [`crate::Bindings::images`]; 0 when compute cannot store to textures.
    pub max_storage_textures: u32,
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
