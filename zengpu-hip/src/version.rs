//! ROCm version detection and feature capability gating.
//!
//! `RocmVersion` is constructed once at startup from the compile-time
//! constants emitted by build.rs (which read `hip_version.h`).
//! Runtime queries are also available via `hipRuntimeGetVersion`.
//!
//! Feature matrix (conservative — older ROCm may support some earlier):
//!
//! | Feature                        | Min ROCm |
//! |--------------------------------|----------|
//! | hipRTC compile                 | 3.10     |
//! | hipRTC get bitcode             | 5.3      |
//! | hipGraph                       | 5.0      |
//! | hipDeviceProp_t.gcnArchName    | 4.0      |
//! | gfx10xx (RDNA 1/2) targets     | 4.0      |
//! | gfx1100 (RDNA 3) targets       | 5.3      |
//! | gfx1150 (RDNA 3.5) targets     | 6.3      |
//! | gfx1200 (RDNA 4) targets       | 7.0      |
//! | gfx942 (CDNA 3 / MI300) target | 6.0      |
//! | hipMemPool / hipMallocAsync    | 5.2      |
//! | cooperative groups             | 4.5      |

use crate::hip_layout::{ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR};

// Re-export so callers don't need to know the module name.
pub use crate::hip_layout::{ROCM_VERSION_MAJOR as COMPILE_ROCM_MAJOR,
                             ROCM_VERSION_MINOR as COMPILE_ROCM_MINOR};

/// ROCm version at the time the crate was compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RocmVersion {
    pub major: u32,
    pub minor: u32,
}

impl RocmVersion {
    /// Version detected at compile time from `hip_version.h`.
    pub const COMPILE_TIME: Self = Self {
        major: ROCM_VERSION_MAJOR,
        minor: ROCM_VERSION_MINOR,
    };

    pub const fn new(major: u32, minor: u32) -> Self { Self { major, minor } }

    pub const fn encode(&self) -> u64 { self.major as u64 * 10_000 + self.minor as u64 }

    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        self.encode() >= Self::new(major, minor).encode()
    }
}

impl std::fmt::Display for RocmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

// ── Feature capability set ────────────────────────────────────────────────────

/// Per-device feature flags derived from gfx target + ROCm version.
/// Constructed once after device enumeration; consulted at dispatch time
/// to choose the best code path or emit a clear error.
#[derive(Debug, Clone)]
pub struct HipCapabilities {
    pub rocm:            RocmVersion,
    pub gfx_family:      GfxFamily,
    /// hipRTC is available (ROCm ≥ 3.10).
    pub hiprtc:          bool,
    /// hipRTC can emit LLVM bitcode for later linking (ROCm ≥ 5.3).
    pub hiprtc_bitcode:  bool,
    /// hipGraph API available (ROCm ≥ 5.0) — enables batched submission.
    pub hip_graph:       bool,
    /// hipMallocAsync / hipMemPool API (ROCm ≥ 5.2).
    pub mem_pool:        bool,
    /// Device supports gcnArchName in hipDeviceProp_t (ROCm ≥ 4.0).
    pub gcn_arch_name:   bool,
    /// WGP mode is default for this gfx family (RDNA 2+); can be overridden
    /// with `-mcumode` in hipRTC options.
    pub wgp_default:     bool,
    /// FP64 throughput is full (CDNA) vs 1/64 (RDNA consumer).
    pub full_fp64:       bool,
}

impl HipCapabilities {
    pub fn from_device(rocm: RocmVersion, gfx: &str) -> Self {
        let family = GfxFamily::from_gfx(gfx);
        Self {
            gfx_family:     family,
            hiprtc:         rocm.at_least(3, 10),
            hiprtc_bitcode: rocm.at_least(5, 3),
            hip_graph:      rocm.at_least(5, 0),
            mem_pool:       rocm.at_least(5, 2),
            gcn_arch_name:  rocm.at_least(4, 0),
            wgp_default:    family.wgp_default(),
            full_fp64:      family.full_fp64(),
            rocm,
        }
    }

    /// Returns the recommended hipRTC options for this device + ROCm version.
    pub fn rtc_options(&self, gfx_target: &str) -> Vec<String> {
        let mut opts = Vec::new();
        if !gfx_target.is_empty() {
            opts.push(format!("--gpu-architecture={gfx_target}"));
        }
        opts.push("-ffast-math".into());
        // RDNA 2+ defaults to WGP; CU mode can help register-heavy kernels.
        // Leave as WGP (default) — users can override via compile options.
        // On ROCm < 5, some flags weren't recognised; gate accordingly.
        if self.rocm.at_least(5, 0) {
            opts.push("-O3".into());
        }
        opts
    }
}

// ── GFX family classification ─────────────────────────────────────────────────

/// Coarse GPU family, used to gate architecture-specific optimisations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxFamily {
    /// GCN 5 (gfx906 Vega20, gfx900 Vega10).
    Gcn5,
    /// CDNA 1 (gfx908 MI100).
    Cdna1,
    /// CDNA 2 (gfx90a MI200).
    Cdna2,
    /// CDNA 3 (gfx940/941/942 MI300).
    Cdna3,
    /// RDNA 1 (gfx1010–1012, RX 5xxx).
    Rdna1,
    /// RDNA 2 (gfx1030–1036, RX 6xxx).
    Rdna2,
    /// RDNA 3 (gfx1100–1103, RX 7xxx).
    Rdna3,
    /// RDNA 3.5 (gfx1150–1153, Strix APU).
    Rdna3p5,
    /// RDNA 4 (gfx1200–1201, RX 9xxx).
    Rdna4,
    /// Unknown / unsupported target.
    Unknown,
}

impl GfxFamily {
    pub fn from_gfx(gfx: &str) -> Self {
        // Strip feature flags ("gfx1200:sramecc-:xnack-" → "gfx1200").
        let base = gfx.split(':').next().unwrap_or(gfx).trim();
        match base {
            "gfx900" | "gfx901"                          => Self::Gcn5,
            "gfx906" | "gfx907"                          => Self::Gcn5,
            "gfx908"                                      => Self::Cdna1,
            "gfx90a" | "gfx90c"                          => Self::Cdna2,
            "gfx940" | "gfx941" | "gfx942"               => Self::Cdna3,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" => Self::Rdna1,
            "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1033" | "gfx1034" | "gfx1035" | "gfx1036" => Self::Rdna2,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Self::Rdna3,
            "gfx1150" | "gfx1151" | "gfx1152" | "gfx1153" => Self::Rdna3p5,
            "gfx1200" | "gfx1201"                         => Self::Rdna4,
            _                                             => Self::Unknown,
        }
    }

    /// RDNA 2+ defaults to WGP mode (two CUs share one WGP).
    pub const fn wgp_default(self) -> bool {
        matches!(self, Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4)
    }

    /// CDNA has full-rate FP64; RDNA consumer is 1/64 rate.
    pub const fn full_fp64(self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3)
    }

    /// Recommended block size for generic compute kernels on this family.
    pub const fn default_block_size(self) -> u32 {
        match self {
            // CDNA: 64-wide warps, 4 warps/CU → 256 good default.
            Self::Cdna1 | Self::Cdna2 | Self::Cdna3 => 256,
            // RDNA: 32-wide waves, WGP = 2 CUs × 20 waves = 1280 threads.
            // 256 fills one CU comfortably.
            Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4 => 256,
            // RDNA 1 / GCN 5: 64-wide waves.
            _ => 256,
        }
    }

    /// Wave width (warp size) for this family.
    pub const fn wave_size(self) -> u32 {
        match self {
            // RDNA uses wave32 by default.
            Self::Rdna1 | Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4 => 32,
            // GCN / CDNA use wave64.
            _ => 64,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gcn5    => "GCN5",
            Self::Cdna1   => "CDNA1",
            Self::Cdna2   => "CDNA2",
            Self::Cdna3   => "CDNA3",
            Self::Rdna1   => "RDNA1",
            Self::Rdna2   => "RDNA2",
            Self::Rdna3   => "RDNA3",
            Self::Rdna3p5 => "RDNA3.5",
            Self::Rdna4   => "RDNA4",
            Self::Unknown => "Unknown",
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx_family_classification() {
        assert_eq!(GfxFamily::from_gfx("gfx1200"),        GfxFamily::Rdna4);
        assert_eq!(GfxFamily::from_gfx("gfx1200:sramecc-:xnack-"), GfxFamily::Rdna4);
        assert_eq!(GfxFamily::from_gfx("gfx1100"),        GfxFamily::Rdna3);
        assert_eq!(GfxFamily::from_gfx("gfx1030"),        GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_gfx("gfx1010"),        GfxFamily::Rdna1);
        assert_eq!(GfxFamily::from_gfx("gfx942"),         GfxFamily::Cdna3);
        assert_eq!(GfxFamily::from_gfx("gfx90a"),         GfxFamily::Cdna2);
        assert_eq!(GfxFamily::from_gfx("gfx908"),         GfxFamily::Cdna1);
        assert_eq!(GfxFamily::from_gfx("gfx906"),         GfxFamily::Gcn5);
        assert_eq!(GfxFamily::from_gfx("gfx9999"),        GfxFamily::Unknown);
    }

    #[test]
    fn wgp_and_wave_sizes() {
        assert!(GfxFamily::Rdna4.wgp_default());
        assert!(GfxFamily::Rdna3.wgp_default());
        assert!(GfxFamily::Rdna2.wgp_default());
        assert!(!GfxFamily::Rdna1.wgp_default());
        assert!(!GfxFamily::Cdna2.wgp_default());

        assert_eq!(GfxFamily::Rdna4.wave_size(), 32);
        assert_eq!(GfxFamily::Cdna2.wave_size(), 64);
        assert_eq!(GfxFamily::Gcn5.wave_size(),  64);
    }

    #[test]
    fn version_ordering() {
        let v60 = RocmVersion::new(6, 0);
        let v72 = RocmVersion::new(7, 2);
        assert!(v72 > v60);
        assert!(v72.at_least(7, 0));
        assert!(v72.at_least(6, 0));
        assert!(!v60.at_least(7, 0));
    }

    #[test]
    fn compile_time_version_is_sane() {
        // Must be at least ROCm 5 on any remotely modern system.
        assert!(
            RocmVersion::COMPILE_TIME.at_least(5, 0),
            "compiled against ROCm {}: expected ≥ 5.0",
            RocmVersion::COMPILE_TIME,
        );
    }

    #[test]
    fn rtc_options_non_empty_for_known_target() {
        let rocm = RocmVersion::new(7, 2);
        let caps = HipCapabilities::from_device(rocm, "gfx1200");
        let opts = caps.rtc_options("gfx1200");
        assert!(opts.iter().any(|o| o.contains("gfx1200")));
        assert!(caps.hiprtc);
        assert!(caps.hiprtc_bitcode);
        assert!(caps.hip_graph);
    }
}
