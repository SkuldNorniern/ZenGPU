//! ROCm version detection and feature capability gating.
//!
//! Feature matrix (conservative lower bounds — actual support may be earlier):
//!
//! | Feature                             | Min ROCm |
//! |-------------------------------------|----------|
//! | hipRTC compile                      | 3.10     |
//! | wave-level builtins (DPP/swizzle)   | 4.0      |
//! | hipDeviceProp_t.gcnArchName         | 4.0      |
//! | gfx10xx (RDNA 1/2) targets          | 4.0      |
//! | cooperative groups                  | 4.5      |
//! | MFMA matrix instructions (CDNA)     | 4.0      |
//! | hipGraph API                        | 5.0      |
//! | hipMallocAsync / hipMemPool         | 5.2      |
//! | hipRTC emit LLVM bitcode            | 5.3      |
//! | gfx1100 (RDNA 3) targets            | 5.3      |
//! | gfx942 (CDNA 3 / MI300) target      | 6.0      |
//! | hipStreamGetCaptureInfo_v2          | 6.0      |
//! | gfx1150 (RDNA 3.5) targets          | 6.3      |
//! | gfx1200 (RDNA 4) targets            | 7.0      |
//! | WMMAv2 matrix instructions (RDNA 3) | 5.4      |

use crate::hip_layout::{ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR};

pub use crate::hip_layout::{
    ROCM_VERSION_MAJOR as COMPILE_ROCM_MAJOR, ROCM_VERSION_MINOR as COMPILE_ROCM_MINOR,
};

// ── RocmVersion ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RocmVersion {
    pub major: u32,
    pub minor: u32,
}

impl RocmVersion {
    pub const COMPILE_TIME: Self = Self {
        major: ROCM_VERSION_MAJOR,
        minor: ROCM_VERSION_MINOR,
    };

    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    pub const fn encode(self) -> u64 {
        self.major as u64 * 10_000 + self.minor as u64
    }

    pub fn at_least(self, major: u32, minor: u32) -> bool {
        self.encode() >= Self::new(major, minor).encode()
    }

    /// Query the ROCm runtime version at runtime.
    /// Falls back to compile-time version if the call fails.
    pub fn runtime() -> Self {
        // hipRuntimeGetVersion returns (major*10_000_000 + minor*100_000 + patch).
        let mut ver: i32 = 0;
        let ok = unsafe { hipRuntimeGetVersion(&mut ver) };
        if ok == 0 && ver > 0 {
            let major = (ver / 10_000_000) as u32;
            let minor = ((ver % 10_000_000) / 100_000) as u32;
            Self { major, minor }
        } else {
            Self::COMPILE_TIME
        }
    }
}

impl std::fmt::Display for RocmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipRuntimeGetVersion(ver: *mut i32) -> i32;
}

// ── HipCapabilities ──────────────────────────────────────────────────────────

/// Complete feature capability set for a single device.
/// Constructed once during device enumeration from gfx target + ROCm version.
#[derive(Debug, Clone)]
pub struct HipCapabilities {
    // ── version ──
    pub rocm: RocmVersion,
    pub runtime_rocm: RocmVersion,

    // ── arch ──
    pub gfx_family: GfxFamily,
    pub wave_size: u32,
    pub wgp_default: bool,
    pub full_fp64: bool,

    // ── hipRTC ──
    /// Basic hipRTC kernel compilation (ROCm ≥ 3.10).
    pub hiprtc: bool,
    /// hipRTC -O3 optimisation flag recognised (ROCm ≥ 5.0).
    pub hiprtc_opt3: bool,
    /// hipRTC can emit LLVM bitcode for offline linking (ROCm ≥ 5.3).
    pub hiprtc_bitcode: bool,

    // ── compute intrinsics ──
    /// Wave-level DPP / swizzle builtins usable in hipRTC kernels (ROCm ≥ 4.0).
    pub wave_reduction: bool,
    /// `__hip_cooperative_groups` header available (ROCm ≥ 4.5).
    pub cooperative_groups: bool,
    /// MFMA matrix-fused-multiply-add instructions (CDNA only, ROCm ≥ 4.0).
    pub mfma: bool,
    /// WMMA v2 instructions usable in kernels (RDNA 3+, ROCm ≥ 5.4).
    pub wmma: bool,

    // ── memory ──
    /// hipMallocAsync / hipMemPool (ROCm ≥ 5.2).
    pub mem_pool: bool,

    // ── dispatch ──
    /// hipGraph stream capture + replay (ROCm ≥ 5.0).
    pub hip_graph: bool,
    /// hipStreamGetCaptureInfo_v2 (ROCm ≥ 6.0).
    pub hip_graph_v2: bool,

    // ── props ──
    /// gcnArchName field valid in hipDeviceProp_t (ROCm ≥ 4.0).
    pub gcn_arch_name: bool,

    // ── gfx target availability ──
    /// gfx10xx (RDNA 1/2) compile target supported (ROCm ≥ 4.0).
    pub target_rdna12: bool,
    /// gfx11xx (RDNA 3) compile target supported (ROCm ≥ 5.3).
    pub target_rdna3: bool,
    /// gfx115x (RDNA 3.5) compile target supported (ROCm ≥ 6.3).
    pub target_rdna3p5: bool,
    /// gfx12xx (RDNA 4) compile target supported (ROCm ≥ 7.0).
    pub target_rdna4: bool,
    /// gfx942 (CDNA 3 / MI300) compile target supported (ROCm ≥ 6.0).
    pub target_cdna3: bool,
}

impl HipCapabilities {
    pub fn from_device(rocm: RocmVersion, gfx: &str) -> Self {
        let family = GfxFamily::from_gfx(gfx);
        let rt = RocmVersion::runtime();
        Self {
            rocm,
            runtime_rocm: rt,
            gfx_family: family,
            wave_size: family.wave_size(),
            wgp_default: family.wgp_default(),
            full_fp64: family.full_fp64(),

            hiprtc: rocm.at_least(3, 10),
            hiprtc_opt3: rocm.at_least(5, 0),
            hiprtc_bitcode: rocm.at_least(5, 3),

            wave_reduction: rocm.at_least(4, 0),
            cooperative_groups: rocm.at_least(4, 5),
            mfma: rocm.at_least(4, 0) && family.has_mfma(),
            wmma: rocm.at_least(5, 4) && family.has_wmma(),

            mem_pool: rocm.at_least(5, 2),
            hip_graph: rocm.at_least(5, 0),
            hip_graph_v2: rocm.at_least(6, 0),
            gcn_arch_name: rocm.at_least(4, 0),

            target_rdna12: rocm.at_least(4, 0),
            target_rdna3: rocm.at_least(5, 3),
            target_rdna3p5: rocm.at_least(6, 3),
            target_rdna4: rocm.at_least(7, 0),
            target_cdna3: rocm.at_least(6, 0),
        }
    }

    /// hipRTC compile options appropriate for this device + ROCm version.
    pub fn rtc_options(&self, gfx_target: &str) -> Vec<String> {
        let mut opts = Vec::new();
        if !gfx_target.is_empty() {
            opts.push(format!("--gpu-architecture={gfx_target}"));
        }
        opts.push("-ffast-math".into());
        if self.hiprtc_opt3 {
            opts.push("-O3".into());
        }
        opts
    }

    /// hipRTC options for wave-level reduction kernels.
    /// Only include wave-reduction builtins if available.
    pub fn rtc_options_wave(&self, gfx_target: &str) -> Vec<String> {
        let mut opts = self.rtc_options(gfx_target);
        if self.wave_reduction {
            // DPP is enabled by default; no extra flag needed.
            // Some versions need explicit wave size.
            if self.wave_size == 32 {
                opts.push("-mwavefrontsize32".into());
            } else {
                opts.push("-mwavefrontsize64".into());
            }
        }
        opts
    }

    /// Human-readable capability report. Used in tests and diagnostics.
    pub fn report(&self, device_name: &str, gfx: &str) -> String {
        let check = |b: bool| if b { "✓" } else { "✗" };
        format!(
            "Device : {device_name} ({gfx} / {})\n\
             ROCm   : compile={} runtime={}\n\
             Arch   : wave{} WGP={} fp64={}\n\
             hipRTC : compile={} opt3={} bitcode={}\n\
             Compute: wave_reduction={} coop_groups={} mfma={} wmma={}\n\
             Memory : mem_pool={}\n\
             Dispatch: hip_graph={} hip_graph_v2={}\n\
             Targets: rdna12={} rdna3={} rdna3.5={} rdna4={} cdna3={}",
            self.gfx_family.name(),
            self.rocm,
            self.runtime_rocm,
            self.wave_size,
            self.wgp_default,
            self.full_fp64,
            check(self.hiprtc),
            check(self.hiprtc_opt3),
            check(self.hiprtc_bitcode),
            check(self.wave_reduction),
            check(self.cooperative_groups),
            check(self.mfma),
            check(self.wmma),
            check(self.mem_pool),
            check(self.hip_graph),
            check(self.hip_graph_v2),
            check(self.target_rdna12),
            check(self.target_rdna3),
            check(self.target_rdna3p5),
            check(self.target_rdna4),
            check(self.target_cdna3),
        )
    }
}

// ── GfxFamily ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxFamily {
    Gcn5,
    Cdna1,
    Cdna2,
    Cdna3,
    Rdna1,
    Rdna2,
    Rdna3,
    Rdna3p5,
    Rdna4,
    Unknown,
}

impl GfxFamily {
    pub fn from_gfx(gfx: &str) -> Self {
        let base = gfx.split(':').next().unwrap_or(gfx).trim();
        match base {
            "gfx900" | "gfx901" => Self::Gcn5,
            "gfx906" | "gfx907" => Self::Gcn5,
            "gfx908" => Self::Cdna1,
            "gfx90a" | "gfx90c" => Self::Cdna2,
            "gfx940" | "gfx941" | "gfx942" => Self::Cdna3,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" => Self::Rdna1,
            "gfx1030" | "gfx1031" | "gfx1032" | "gfx1033" | "gfx1034" | "gfx1035" | "gfx1036" => {
                Self::Rdna2
            }
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Self::Rdna3,
            "gfx1150" | "gfx1151" | "gfx1152" | "gfx1153" => Self::Rdna3p5,
            "gfx1200" | "gfx1201" => Self::Rdna4,
            _ => Self::Unknown,
        }
    }

    /// RDNA 2+ uses WGP mode by default (two CUs share one WGP).
    pub const fn wgp_default(self) -> bool {
        matches!(
            self,
            Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4
        )
    }

    /// CDNA has full-rate FP64; RDNA consumer is 1/64.
    pub const fn full_fp64(self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3)
    }

    /// MFMA (matrix fused multiply-add) — CDNA only.
    pub const fn has_mfma(self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3)
    }

    /// WMMA (wave matrix multiply-accumulate) — RDNA 3+ (hardware WMMA).
    pub const fn has_wmma(self) -> bool {
        matches!(self, Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4)
    }

    /// Wave width (warp size) for this family.
    pub const fn wave_size(self) -> u32 {
        match self {
            Self::Rdna1 | Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4 => 32,
            _ => 64,
        }
    }

    /// Recommended block size for generic compute kernels.
    pub const fn default_block_size(self) -> u32 {
        match self {
            // CDNA: 64-wide waves, typical 4 waves/CU → 256.
            Self::Cdna1 | Self::Cdna2 | Self::Cdna3 => 256,
            // RDNA: 32-wide waves, 256 fills one CU (8 waves × 32).
            Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4 => 256,
            _ => 256,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gcn5 => "GCN5",
            Self::Cdna1 => "CDNA1",
            Self::Cdna2 => "CDNA2",
            Self::Cdna3 => "CDNA3",
            Self::Rdna1 => "RDNA1",
            Self::Rdna2 => "RDNA2",
            Self::Rdna3 => "RDNA3",
            Self::Rdna3p5 => "RDNA3.5",
            Self::Rdna4 => "RDNA4",
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
        assert_eq!(GfxFamily::from_gfx("gfx1200"), GfxFamily::Rdna4);
        assert_eq!(
            GfxFamily::from_gfx("gfx1200:sramecc-:xnack-"),
            GfxFamily::Rdna4
        );
        assert_eq!(GfxFamily::from_gfx("gfx1201"), GfxFamily::Rdna4);
        assert_eq!(GfxFamily::from_gfx("gfx1150"), GfxFamily::Rdna3p5);
        assert_eq!(GfxFamily::from_gfx("gfx1100"), GfxFamily::Rdna3);
        assert_eq!(GfxFamily::from_gfx("gfx1102"), GfxFamily::Rdna3);
        assert_eq!(GfxFamily::from_gfx("gfx1030"), GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_gfx("gfx1036"), GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_gfx("gfx1010"), GfxFamily::Rdna1);
        assert_eq!(GfxFamily::from_gfx("gfx942"), GfxFamily::Cdna3);
        assert_eq!(GfxFamily::from_gfx("gfx940"), GfxFamily::Cdna3);
        assert_eq!(GfxFamily::from_gfx("gfx90a"), GfxFamily::Cdna2);
        assert_eq!(GfxFamily::from_gfx("gfx908"), GfxFamily::Cdna1);
        assert_eq!(GfxFamily::from_gfx("gfx906"), GfxFamily::Gcn5);
        assert_eq!(GfxFamily::from_gfx("gfx9999"), GfxFamily::Unknown);
    }

    #[test]
    fn arch_properties() {
        assert!(GfxFamily::Rdna4.wgp_default());
        assert!(GfxFamily::Rdna3.wgp_default());
        assert!(GfxFamily::Rdna2.wgp_default());
        assert!(!GfxFamily::Rdna1.wgp_default());
        assert!(!GfxFamily::Cdna2.wgp_default());

        assert_eq!(GfxFamily::Rdna4.wave_size(), 32);
        assert_eq!(GfxFamily::Rdna1.wave_size(), 32);
        assert_eq!(GfxFamily::Cdna2.wave_size(), 64);
        assert_eq!(GfxFamily::Gcn5.wave_size(), 64);

        assert!(GfxFamily::Cdna3.has_mfma());
        assert!(GfxFamily::Cdna2.has_mfma());
        assert!(!GfxFamily::Rdna4.has_mfma());

        assert!(GfxFamily::Rdna4.has_wmma());
        assert!(GfxFamily::Rdna3.has_wmma());
        assert!(!GfxFamily::Rdna2.has_wmma());
        assert!(!GfxFamily::Cdna3.has_wmma());

        assert!(GfxFamily::Cdna1.full_fp64());
        assert!(!GfxFamily::Rdna4.full_fp64());
    }

    #[test]
    fn version_ordering() {
        let v60 = RocmVersion::new(6, 0);
        let v72 = RocmVersion::new(7, 2);
        let v53 = RocmVersion::new(5, 3);
        assert!(v72 > v60);
        assert!(v72.at_least(7, 0));
        assert!(v72.at_least(6, 0));
        assert!(!v60.at_least(7, 0));
        assert!(!v53.at_least(6, 0));
    }

    #[test]
    fn compile_time_version_is_sane() {
        assert!(
            RocmVersion::COMPILE_TIME.at_least(5, 0),
            "compiled against ROCm {}: expected ≥ 5.0",
            RocmVersion::COMPILE_TIME,
        );
    }

    #[test]
    fn rdna4_capabilities_on_rocm72() {
        let rocm = RocmVersion::new(7, 2);
        let caps = HipCapabilities::from_device(rocm, "gfx1200");

        // All modern features should be present.
        assert!(caps.hiprtc, "hipRTC");
        assert!(caps.hiprtc_opt3, "hipRTC -O3");
        assert!(caps.hiprtc_bitcode, "hipRTC bitcode");
        assert!(caps.wave_reduction, "wave reduction builtins");
        assert!(caps.cooperative_groups, "cooperative groups");
        assert!(caps.mem_pool, "mem pool");
        assert!(caps.hip_graph, "hip graph");
        assert!(caps.hip_graph_v2, "hip graph v2");
        assert!(caps.gcn_arch_name, "gcnArchName");
        assert!(caps.target_rdna4, "target rdna4");
        assert!(caps.wgp_default, "WGP mode default");
        assert_eq!(caps.wave_size, 32, "wave32");

        // RDNA 4 does not have MFMA or full FP64.
        assert!(!caps.mfma, "no MFMA on RDNA 4");
        assert!(!caps.full_fp64, "no full FP64 on RDNA 4");

        // WMMA is present (RDNA 3+, ROCm ≥ 5.4 ✓).
        assert!(caps.wmma, "WMMA on RDNA 4");
    }

    #[test]
    fn old_rocm_feature_gating() {
        let rocm40 = RocmVersion::new(4, 0);
        let caps = HipCapabilities::from_device(rocm40, "gfx1030");

        assert!(caps.hiprtc);
        assert!(!caps.hiprtc_opt3, "no -O3 on ROCm 4.0");
        assert!(!caps.hiprtc_bitcode, "no bitcode on ROCm 4.0");
        assert!(!caps.cooperative_groups, "no coop groups on ROCm 4.0");
        assert!(!caps.hip_graph, "no hip_graph on ROCm 4.0");
        assert!(!caps.mem_pool, "no mem_pool on ROCm 4.0");
        assert!(!caps.target_rdna3, "no rdna3 target on ROCm 4.0");
        assert!(!caps.target_rdna4, "no rdna4 target on ROCm 4.0");
    }

    #[test]
    fn rtc_options_version_gated() {
        let old = HipCapabilities::from_device(RocmVersion::new(4, 2), "gfx1030");
        let new = HipCapabilities::from_device(RocmVersion::new(7, 2), "gfx1200");

        let opts_old = old.rtc_options("gfx1030");
        let opts_new = new.rtc_options("gfx1200");

        assert!(!opts_old.iter().any(|o| o == "-O3"), "no -O3 on ROCm 4");
        assert!(opts_new.iter().any(|o| o == "-O3"), "-O3 on ROCm 7");
        assert!(opts_new.iter().any(|o| o.contains("gfx1200")));
    }

    #[test]
    fn runtime_version_available() {
        let rt = RocmVersion::runtime();
        // Must be at least ROCm 5 on any modern system.
        assert!(
            rt.at_least(5, 0),
            "runtime ROCm version {rt} is unexpectedly old",
        );
    }

    #[test]
    fn capability_report_format() {
        let caps = HipCapabilities::from_device(RocmVersion::new(7, 2), "gfx1200");
        let report = caps.report("AMD Radeon RX 9060 XT", "gfx1200");
        // Spot-check a few required lines.
        assert!(report.contains("gfx1200"));
        assert!(report.contains("RDNA4"));
        assert!(report.contains("wave32"));
        assert!(report.contains("compile=7.2"));
    }
}
