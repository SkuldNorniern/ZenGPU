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
//! | gfx10-1/10-3/11/12-generic targets  | 6.2      |
//! | cooperative groups                  | 4.5      |
//! | MFMA matrix instructions (CDNA)     | 4.0      |
//! | hipGraph API                        | 5.0      |
//! | hipMallocAsync / hipMemPool         | 5.2      |
//! | hipRTC emit LLVM bitcode            | 5.3      |
//! | gfx1100 (RDNA 3) targets            | 5.3      |
//! | gfx942 (CDNA 3 / MI300) target      | 6.0      |
//! | hipStreamGetCaptureInfo_v2          | 6.0      |
//! | gfx1150 (RDNA 3.5) targets          | 6.3      |
//! | gfx1200 (RDNA 4) targets            | 6.4      |
//! | WMMAv2 matrix instructions (RDNA 3) | 5.4      |

use std::fmt;

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

impl fmt::Display for RocmVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    /// gfx12xx (RDNA 4) compile target supported (ROCm ≥ 6.4).
    pub target_rdna4: bool,
    /// gfx942 (CDNA 3 / MI300) compile target supported (ROCm ≥ 6.0).
    pub target_cdna3: bool,
}

impl HipCapabilities {
    /// Capabilities for `gfx` given the ROCm headers the crate was built
    /// against. A feature is reported only when both the build headers and the
    /// loaded runtime are new enough, since an older `libamdhip64` at run time
    /// lacks what newer headers advertise.
    pub fn from_device(rocm: RocmVersion, gfx: &str) -> Self {
        let family = GfxFamily::from_gfx(gfx);
        Self::for_family(rocm, RocmVersion::runtime(), family)
    }

    pub fn for_family(compile: RocmVersion, runtime: RocmVersion, family: GfxFamily) -> Self {
        let rocm = compile.min(runtime);
        let rt = runtime;
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
            target_rdna4: rocm.at_least(6, 4),
            target_cdna3: rocm.at_least(6, 0),
        }
    }

    /// hipRTC compile options appropriate for this device + ROCm version.
    ///
    /// No `-ffast-math`: it lets the compiler fold `isnan`/`isinf` to false
    /// and reassociate reductions, so results would differ from the CUDA and
    /// Vulkan backends, which compile with IEEE semantics.
    ///
    /// An empty `gfx_target` leaves the architecture to hipRTC, which then
    /// compiles for the current device.
    pub fn rtc_options(&self, gfx_target: &str) -> Vec<String> {
        let mut opts = Vec::new();
        if !gfx_target.is_empty() {
            opts.push(format!("--gpu-architecture={gfx_target}"));
        }
        if self.hiprtc_opt3 {
            opts.push("-O3".into());
        }
        opts
    }

    /// Whether the ROCm in use can compile for this device's family. `false`
    /// means hipRTC will most likely reject the target; callers use it to
    /// explain the failure rather than to refuse up front.
    pub fn target_supported(&self) -> bool {
        match self.gfx_family {
            GfxFamily::Rdna1 | GfxFamily::Rdna2 => self.target_rdna12,
            GfxFamily::Rdna3 => self.target_rdna3,
            GfxFamily::Rdna3p5 => self.target_rdna3p5,
            GfxFamily::Rdna4 => self.target_rdna4,
            GfxFamily::Cdna3 => self.target_cdna3,
            GfxFamily::Cdna4 => self.rocm.at_least(7, 0),
            _ => true,
        }
    }

    /// hipRTC options for wave-level reduction kernels.
    /// Only include wave-reduction builtins if available.
    pub fn rtc_options_wave(&self, gfx_target: &str) -> Vec<String> {
        let mut opts = self.rtc_options(gfx_target);
        if self.wave_reduction {
            // DPP is enabled by default; no extra flag needed.
            // Some versions need explicit wave size.
            // Only RDNA can run either width; GCN/CDNA reject the flag.
            if self.gfx_family.is_rdna() {
                if self.wave_size == 32 {
                    opts.push("-mwavefrontsize32".into());
                } else {
                    opts.push("-mwavefrontsize64".into());
                }
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
    /// Polaris / Fiji (gfx803). Dropped from official ROCm support after
    /// 4.x; hipRTC can still target it where the kernel driver runs it.
    Gcn4,
    Gcn5,
    Cdna1,
    Cdna2,
    Cdna3,
    /// MI350 / MI355 (gfx950).
    Cdna4,
    /// Navi 1x (RX 5000 series). Never officially supported by ROCm, but
    /// hipRTC compiles gfx101x natively, so no `HSA_OVERRIDE_GFX_VERSION`
    /// is needed (and overriding to 10.3.0 runs RDNA 2 code on RDNA 1
    /// hardware, which can hang).
    Rdna1,
    Rdna2,
    Rdna3,
    Rdna3p5,
    Rdna4,
    Unknown,
}

impl GfxFamily {
    /// Classify a gfx target such as `gfx1010`, `gfx90a:xnack-` or
    /// `gfx10-1-generic`. Unlisted steppings fall back to their generation, so
    /// a new part in a known family is not treated as unknown.
    pub fn from_gfx(gfx: &str) -> Self {
        let base = gfx.split(':').next().unwrap_or(gfx).trim();
        match base {
            "gfx801" | "gfx802" | "gfx803" | "gfx805" | "gfx810" => return Self::Gcn4,
            "gfx900" | "gfx902" | "gfx904" | "gfx906" | "gfx909" | "gfx90c" => return Self::Gcn5,
            "gfx9-generic" => return Self::Gcn5,
            "gfx908" => return Self::Cdna1,
            "gfx90a" => return Self::Cdna2,
            "gfx940" | "gfx941" | "gfx942" | "gfx9-4-generic" => return Self::Cdna3,
            "gfx950" => return Self::Cdna4,
            "gfx10-1-generic" => return Self::Rdna1,
            "gfx10-3-generic" => return Self::Rdna2,
            "gfx11-generic" => return Self::Rdna3,
            "gfx12-generic" => return Self::Rdna4,
            _ => {}
        }
        let Some(digits) = base.strip_prefix("gfx") else {
            return Self::Unknown;
        };
        if digits.starts_with("115") {
            Self::Rdna3p5
        } else if digits.starts_with("12") {
            Self::Rdna4
        } else if digits.starts_with("11") {
            Self::Rdna3
        } else if digits.starts_with("103") {
            Self::Rdna2
        } else if digits.starts_with("101") {
            Self::Rdna1
        } else {
            Self::Unknown
        }
    }

    /// Family from the compute-capability pair HIP reports for AMD devices
    /// (`major` = gfx major, `minor` = gfx minor). Used when the gfx name is
    /// unavailable; it cannot tell gfx906 from gfx90a, so CDNA 1/2 read as
    /// GCN 5.
    pub const fn from_compute_capability(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (8, _) => Self::Gcn4,
            (9, 4) => Self::Cdna3,
            (9, 5) => Self::Cdna4,
            (9, _) => Self::Gcn5,
            (10, 1) => Self::Rdna1,
            (10, 3) => Self::Rdna2,
            (11, 5) => Self::Rdna3p5,
            (11, _) => Self::Rdna3,
            (12, _) => Self::Rdna4,
            _ => Self::Unknown,
        }
    }

    /// RDNA family (any generation).
    pub const fn is_rdna(self) -> bool {
        matches!(
            self,
            Self::Rdna1 | Self::Rdna2 | Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4
        )
    }

    /// RDNA compiles in WGP mode by default (two CUs share one WGP); LLVM's
    /// `cumode` is off for every gfx10+ target, RDNA 1 included.
    pub const fn wgp_default(self) -> bool {
        self.is_rdna()
    }

    /// CDNA has full-rate FP64; RDNA consumer is 1/64.
    pub const fn full_fp64(self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4)
    }

    /// MFMA (matrix fused multiply-add) — CDNA only.
    pub const fn has_mfma(self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4)
    }

    /// WMMA (wave matrix multiply-accumulate) — RDNA 3+ (hardware WMMA).
    pub const fn has_wmma(self) -> bool {
        matches!(self, Self::Rdna3 | Self::Rdna3p5 | Self::Rdna4)
    }

    /// Wave width (warp size) for this family.
    /// HIP's default `warpSize` for this family. The device's own
    /// `hipDeviceAttributeWarpSize` overrides it when available.
    pub const fn wave_size(self) -> u32 {
        if self.is_rdna() { 32 } else { 64 }
    }

    /// Recommended block size for generic compute kernels: four 64-wide
    /// waves on GCN/CDNA, eight 32-wide waves on RDNA.
    pub const fn default_block_size(self) -> u32 {
        256
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gcn4 => "GCN4",
            Self::Gcn5 => "GCN5",
            Self::Cdna1 => "CDNA1",
            Self::Cdna2 => "CDNA2",
            Self::Cdna3 => "CDNA3",
            Self::Cdna4 => "CDNA4",
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
        assert_eq!(GfxFamily::from_gfx("gfx950"), GfxFamily::Cdna4);
        assert_eq!(GfxFamily::from_gfx("gfx803"), GfxFamily::Gcn4);
        assert_eq!(GfxFamily::from_gfx("gfx9999"), GfxFamily::Unknown);
        assert_eq!(GfxFamily::from_gfx("navi10"), GfxFamily::Unknown);
        assert_eq!(GfxFamily::from_gfx(""), GfxFamily::Unknown);
    }

    #[test]
    fn rdna1_targets_classify() {
        for gfx in ["gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1010:xnack-"] {
            assert_eq!(GfxFamily::from_gfx(gfx), GfxFamily::Rdna1, "{gfx}");
        }
        assert_eq!(GfxFamily::from_gfx("gfx10-1-generic"), GfxFamily::Rdna1);
        assert_eq!(GfxFamily::from_gfx("gfx10-3-generic"), GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_gfx("gfx11-generic"), GfxFamily::Rdna3);
        assert_eq!(GfxFamily::from_gfx("gfx12-generic"), GfxFamily::Rdna4);
        // Unlisted steppings fall back to their generation.
        assert_eq!(GfxFamily::from_gfx("gfx1037"), GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_gfx("gfx1103"), GfxFamily::Rdna3);
        assert_eq!(GfxFamily::from_gfx("gfx1153"), GfxFamily::Rdna3p5);
        assert_eq!(GfxFamily::from_gfx("gfx1250"), GfxFamily::Rdna4);
    }

    #[test]
    fn compute_capability_fallback() {
        assert_eq!(GfxFamily::from_compute_capability(10, 1), GfxFamily::Rdna1);
        assert_eq!(GfxFamily::from_compute_capability(10, 3), GfxFamily::Rdna2);
        assert_eq!(GfxFamily::from_compute_capability(11, 0), GfxFamily::Rdna3);
        assert_eq!(
            GfxFamily::from_compute_capability(11, 5),
            GfxFamily::Rdna3p5
        );
        assert_eq!(GfxFamily::from_compute_capability(12, 0), GfxFamily::Rdna4);
        assert_eq!(GfxFamily::from_compute_capability(9, 4), GfxFamily::Cdna3);
        assert_eq!(GfxFamily::from_compute_capability(0, 0), GfxFamily::Unknown);
    }

    #[test]
    fn rdna1_capabilities_on_old_and_new_rocm() {
        for rocm in [RocmVersion::new(5, 7), RocmVersion::new(7, 2)] {
            let caps = HipCapabilities::for_family(rocm, rocm, GfxFamily::Rdna1);
            assert_eq!(caps.wave_size, 32);
            assert!(caps.target_supported(), "gfx101x compiles on ROCm {rocm}");
            assert!(!caps.wmma, "RDNA 1 has no WMMA");
            assert!(!caps.mfma);
            let opts = caps.rtc_options("gfx1010");
            assert!(opts.iter().any(|o| o == "--gpu-architecture=gfx1010"));
        }
    }

    #[test]
    fn runtime_older_than_headers_limits_features() {
        let caps = HipCapabilities::for_family(
            RocmVersion::new(7, 2),
            RocmVersion::new(6, 1),
            GfxFamily::Rdna4,
        );
        assert!(!caps.target_rdna4, "ROCm 6.1 runtime cannot target gfx12");
        assert!(!caps.target_supported());
    }

    #[test]
    fn arch_properties() {
        assert!(GfxFamily::Rdna4.wgp_default());
        assert!(GfxFamily::Rdna3.wgp_default());
        assert!(GfxFamily::Rdna2.wgp_default());
        assert!(GfxFamily::Rdna1.wgp_default());
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
        let caps = HipCapabilities::for_family(rocm, rocm, GfxFamily::Rdna4);

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
        let caps = HipCapabilities::for_family(rocm40, rocm40, GfxFamily::Rdna2);

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
        let v42 = RocmVersion::new(4, 2);
        let v72 = RocmVersion::new(7, 2);
        let old = HipCapabilities::for_family(v42, v42, GfxFamily::Rdna2);
        let new = HipCapabilities::for_family(v72, v72, GfxFamily::Rdna4);

        let opts_old = old.rtc_options("gfx1030");
        let opts_new = new.rtc_options("gfx1200");

        assert!(!opts_old.iter().any(|o| o == "-O3"), "no -O3 on ROCm 4");
        assert!(opts_new.iter().any(|o| o == "-O3"), "-O3 on ROCm 7");
        assert!(opts_new.iter().any(|o| o.contains("gfx1200")));
        assert!(
            !opts_new.iter().any(|o| o.contains("fast-math")),
            "fast-math changes NaN/Inf semantics vs the other backends"
        );
        assert!(
            !new.rtc_options("")
                .iter()
                .any(|o| o.contains("gpu-architecture"))
        );
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
        let v72 = RocmVersion::new(7, 2);
        let caps = HipCapabilities::for_family(v72, v72, GfxFamily::Rdna4);
        let report = caps.report("AMD Radeon RX 9060 XT", "gfx1200");
        // Spot-check a few required lines.
        assert!(report.contains("gfx1200"));
        assert!(report.contains("RDNA4"));
        assert!(report.contains("wave32"));
        assert!(report.contains("compile=7.2"));
    }
}
