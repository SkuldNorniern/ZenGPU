//! Per-GPU and per-ROCm workarounds.
//!
//! | Problem                                                         | Mitigation                                  |
//! |-----------------------------------------------------------------|---------------------------------------------|
//! | SDMA copy engine hangs/corruption on RDNA 1/2 consumer cards    | `HSA_ENABLE_SDMA=0` before `hipInit` (blit  |
//! | (commonly reported on Navi 1x/2x, worst with multiple GPUs)     | kernels do the copies instead)              |
//! | `HSA_OVERRIDE_GFX_VERSION=10.3.0` on RDNA 1 runs RDNA 2 code on | warn; this crate compiles native gfx101x,   |
//! | RDNA 1 silicon and can hang the GPU                             | so the override is unnecessary              |
//! | PCIe peer-to-peer on consumer RDNA boards reported usable but   | P2P only on CDNA by default; RDNA opts in   |
//! | faulting or hanging without large BAR / with IOMMU remapping    | with `ZENGPU_HIP_P2P=1`; host staging else  |
//! | APUs carve out a small VRAM region (often 512 MiB), so          | integrated GPUs fall back to pinned mapped  |
//! | `hipMalloc` fails long before memory is exhausted               | host memory on out-of-memory                |
//! | hipRTC/comgr not thread-safe in older ROCm releases             | every compile is serialized (and cached)    |
//! | `hipInit` fails with no hint when `/dev/kfd` is missing or the  | the error names the missing driver or the   |
//! | user is not in the `render` group                               | group to join                               |
//! | Out-of-bounds GPU access makes ROCr `abort()` the process       | host-side bounds, slot, and launch-size     |
//! | ("Memory access fault by GPU node-N")                           | validation before anything reaches the GPU  |
//!
//! Environment changes are skipped when `ZENGPU_HIP_NO_ENV_WORKAROUNDS` is set or the variable is
//! already set.

use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Once;

use crate::version::GfxFamily;

/// Environment variable that disables every workaround in this module when set.
pub const NO_ENV_WORKAROUNDS: &str = "ZENGPU_HIP_NO_ENV_WORKAROUNDS";
/// Value of `ZENGPU_HIP_P2P`: `1` enables PCIe peer-to-peer on consumer RDNA, `0`
/// disables it everywhere.
pub const P2P_ENV: &str = "ZENGPU_HIP_P2P";

const KFD_TOPOLOGY: &str = "/sys/class/kfd/kfd/topology/nodes";

/// GPU families from KFD topology, read before HIP init. Empty off Linux or without amdgpu KFD.
/// This is the real silicon: `HSA_OVERRIDE_GFX_VERSION` does not change it.
pub fn kfd_families() -> Vec<GfxFamily> {
    let Ok(nodes) = fs::read_dir(KFD_TOPOLOGY) else {
        return Vec::new();
    };
    let mut families = Vec::new();
    for node in nodes.flatten() {
        let Ok(props) = fs::read_to_string(node.path().join("properties")) else {
            continue;
        };
        if let Some(version) = parse_gfx_target_version(&props)
            && version != 0
        {
            families.push(family_from_target_version(version));
        }
    }
    families
}

/// `gfx_target_version` from a KFD node's `properties` file; 0 for CPU nodes.
fn parse_gfx_target_version(props: &str) -> Option<u32> {
    props.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        (parts.next() == Some("gfx_target_version"))
            .then(|| parts.next()?.parse().ok())
            .flatten()
    })
}

/// KFD encodes gfx targets as `major * 10000 + minor * 100 + stepping`.
fn family_from_target_version(version: u32) -> GfxFamily {
    let major = (version / 10_000) as i32;
    let minor = ((version / 100) % 100) as i32;
    let stepping = version % 100;
    match (major, minor, stepping) {
        (9, 0, 8) => GfxFamily::Cdna1,
        (9, 0, 10) => GfxFamily::Cdna2,
        _ => GfxFamily::from_compute_capability(major, minor),
    }
}

/// Runs once, before the first `hipInit`: ROCr reads these variables only at init.
pub fn apply_process_env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if env::var_os(NO_ENV_WORKAROUNDS).is_some() {
            return;
        }
        let families = kfd_families();

        if families.contains(&GfxFamily::Rdna1)
            && let Ok(over) = env::var("HSA_OVERRIDE_GFX_VERSION")
        {
            log::warn!(
                "hip: HSA_OVERRIDE_GFX_VERSION={over} is set on RDNA 1 hardware; \
                 running another generation's code on Navi 1x can hang the GPU. \
                 zengpu compiles native gfx101x kernels, so unset it."
            );
        }

        if needs_sdma_off(&families) && env::var_os("HSA_ENABLE_SDMA").is_none() {
            log::info!(
                "hip: setting HSA_ENABLE_SDMA=0 for RDNA 1/2 copy-engine stability \
                 (set {NO_ENV_WORKAROUNDS}=1 or HSA_ENABLE_SDMA yourself to skip)"
            );
            // SAFETY: runs before hipInit, so no HIP thread reads the environment yet. App threads
            unsafe { env::set_var("HSA_ENABLE_SDMA", "0") };
        }
    });
}

fn needs_sdma_off(families: &[GfxFamily]) -> bool {
    families
        .iter()
        .any(|f| matches!(f, GfxFamily::Rdna1 | GfxFamily::Rdna2))
}

/// Returns whether to use direct peer copies between two device families, given `ZENGPU_HIP_P2P`
/// (`None` when unset).
pub fn peer_copy_allowed(a: GfxFamily, b: GfxFamily, env_value: Option<&str>) -> bool {
    match env_value.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => !a.is_rdna() && !b.is_rdna() && a != GfxFamily::Unknown && b != GfxFamily::Unknown,
    }
}

/// Why HIP could not start, when the cause is visible from user space.
pub fn kfd_diagnostic() -> Option<String> {
    kfd_diagnostic_at(Path::new("/dev/kfd"))
}

fn kfd_diagnostic_at(path: &Path) -> Option<String> {
    if cfg!(not(target_os = "linux")) {
        return None;
    }
    match fs::OpenOptions::new().read(true).write(true).open(path) {
        Ok(_) => None,
        Err(e) if e.kind() == ErrorKind::NotFound => Some(format!(
            "{} not found: the amdgpu kernel driver with KFD is not loaded",
            path.display()
        )),
        Err(e) if e.kind() == ErrorKind::PermissionDenied => Some(format!(
            "no permission to open {}: add the user to the `render` and `video` \
             groups and log in again",
            path.display()
        )),
        Err(e) => Some(format!("cannot open {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kfd_properties() {
        let props = "cpu_cores_count 0\nsimd_count 64\ngfx_target_version 120000\n";
        assert_eq!(parse_gfx_target_version(props), Some(120000));
        assert_eq!(parse_gfx_target_version("simd_count 0\n"), None);
    }

    #[test]
    fn kfd_versions_map_to_families() {
        assert_eq!(family_from_target_version(100100), GfxFamily::Rdna1);
        assert_eq!(family_from_target_version(100102), GfxFamily::Rdna1);
        assert_eq!(family_from_target_version(100300), GfxFamily::Rdna2);
        assert_eq!(family_from_target_version(110000), GfxFamily::Rdna3);
        assert_eq!(family_from_target_version(110501), GfxFamily::Rdna3p5);
        assert_eq!(family_from_target_version(120001), GfxFamily::Rdna4);
        assert_eq!(family_from_target_version(90010), GfxFamily::Cdna2);
        assert_eq!(family_from_target_version(90008), GfxFamily::Cdna1);
        assert_eq!(family_from_target_version(90402), GfxFamily::Cdna3);
        assert_eq!(family_from_target_version(90006), GfxFamily::Gcn5);
        assert_eq!(family_from_target_version(80003), GfxFamily::Gcn4);
    }

    #[test]
    fn sdma_disabled_only_for_rdna12() {
        assert!(needs_sdma_off(&[GfxFamily::Rdna1]));
        assert!(needs_sdma_off(&[GfxFamily::Rdna4, GfxFamily::Rdna2]));
        assert!(!needs_sdma_off(&[GfxFamily::Rdna3, GfxFamily::Rdna4]));
        assert!(!needs_sdma_off(&[GfxFamily::Cdna3]));
        assert!(!needs_sdma_off(&[]));
    }

    #[test]
    fn peer_policy() {
        use GfxFamily::*;
        assert!(peer_copy_allowed(Cdna3, Cdna3, None));
        assert!(!peer_copy_allowed(Rdna4, Rdna4, None));
        assert!(peer_copy_allowed(Rdna4, Rdna4, Some("1")));
        assert!(!peer_copy_allowed(Cdna3, Cdna3, Some("0")));
        assert!(!peer_copy_allowed(Unknown, Cdna3, None));
    }

    #[test]
    fn missing_kfd_is_explained() {
        let msg = kfd_diagnostic_at(Path::new("/nonexistent/kfd"));
        if cfg!(target_os = "linux") {
            assert!(msg.unwrap().contains("not loaded"));
        }
    }
}
