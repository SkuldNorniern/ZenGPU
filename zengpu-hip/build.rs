// A build script failing is meant to abort the build loudly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

/// Build script for zengpu-hip.
///
/// Nothing is linked: the HIP runtime and hipRTC are loaded at run time
/// (see `src/ffi.rs`), so ROCm is optional both here and on the target.
///
/// 1. Locates the ROCm headers (ROCM_PATH env → /opt/rocm scan).
/// 2. Probes `hip_version.h` → emits `cfg(rocm_major="N")` / `cfg(rocm_minor="N")`
///    so the crate can gate features on ROCm version at compile time.
/// 3. Compiles a tiny C probe to measure the `hipDeviceProp_t` size, the
///    `gcnArchName` offset, and the `hipDeviceAttribute_t` values the crate
///    queries, all of which shifted between major ROCm releases. Falls back to
///    ROCm 6/7 x86-64 defaults when cross-compiling or when no AMD compiler is
///    present.
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| find_rocm());

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed={rocm_path}/include/hip/hip_version.h");
    println!("cargo:rerun-if-changed=build.rs");

    // ── ROCm version detection ────────────────────────────────────────────────
    let (major, minor) = detect_rocm_version(&rocm_path);
    if major < 5 {
        println!(
            "cargo:warning=zengpu-hip: ROCm {major}.{minor} predates the 5.0 device-attribute \
             renumbering; ROCm 5.0 or newer is required for correct device queries"
        );
    }
    println!("cargo:rustc-cfg=rocm_major=\"{major}\"");
    println!("cargo:rustc-cfg=rocm_minor=\"{minor}\"");
    // The generated file `hip_layout.rs` carries ROCM_VERSION_MAJOR/MINOR for
    // numeric comparisons.

    // ── hipDeviceProp_t layout + enum probe ─────────────────────────────────
    emit_hip_layout(&rocm_path, major, minor);
}

// ── ROCm finder ──────────────────────────────────────────────────────────────

fn find_rocm() -> String {
    // Prefer versioned symlink (e.g. /opt/rocm-7.2.1 → /opt/rocm).
    for candidate in &["/opt/rocm", "/usr/local/rocm", "/usr"] {
        if PathBuf::from(format!("{candidate}/lib/libamdhip64.so")).exists()
            || PathBuf::from(format!("{candidate}/lib64/libamdhip64.so")).exists()
        {
            return candidate.to_string();
        }
    }
    "/opt/rocm".to_string()
}

// ── Version detection ─────────────────────────────────────────────────────────

fn detect_rocm_version(rocm_path: &str) -> (u32, u32) {
    // Primary: parse hip_version.h (works offline, no compilation needed).
    let version_h = format!("{rocm_path}/include/hip/hip_version.h");
    if let Ok(text) = fs::read_to_string(&version_h) {
        let major = parse_define(&text, "HIP_VERSION_MAJOR");
        let minor = parse_define(&text, "HIP_VERSION_MINOR");
        if let (Some(maj), Some(min)) = (major, minor) {
            println!("cargo:warning=zengpu-hip: detected ROCm {maj}.{min} from {version_h}");
            return (maj, min);
        }
    }

    // Secondary: `hipconfig --version` → "HIP version: 7.2.53211"
    if let Ok(out) = Command::new(format!("{rocm_path}/bin/hipconfig"))
        .arg("--version")
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(ver) = s.split("HIP version:").nth(1) {
            let parts: Vec<&str> = ver.trim().splitn(3, '.').collect();
            if parts.len() >= 2
                && let (Ok(maj), Ok(min)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
            {
                println!("cargo:warning=zengpu-hip: detected ROCm {maj}.{min} via hipconfig");
                return (maj, min);
            }
        }
    }

    println!("cargo:warning=zengpu-hip: ROCm version unknown; assuming 6.0");
    (6, 0)
}

fn parse_define(text: &str, name: &str) -> Option<u32> {
    text.lines()
        .find(|l| l.contains(&format!("#define {name} ")))?
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

// ── hipDeviceProp_t layout + enum probe ──────────────────────────────────────

/// Values the probe measures. Struct offsets shift between ROCm majors and the
/// `hipDeviceAttribute_t` enum was renumbered in ROCm 5.0, so neither is
/// hard-coded when a compiler can tell us.
struct HipLayout {
    prop_size: usize,
    gcn_arch_off: usize,
    attrs: [i32; ATTR_NAMES.len()],
}

/// `hipDeviceAttribute_t` members the crate queries, in the order the generated
/// constants are written.
const ATTR_NAMES: [&str; 11] = [
    "MultiprocessorCount",
    "ClockRate",
    "Integrated",
    "WarpSize",
    "MaxThreadsPerBlock",
    "MaxBlockDimX",
    "MaxBlockDimY",
    "MaxBlockDimZ",
    "MaxGridDimX",
    "MaxGridDimY",
    "MaxGridDimZ",
];

/// ROCm 5.0 – 7.2 values, used only when the probe cannot run.
const ATTR_DEFAULTS: [i32; ATTR_NAMES.len()] = [63, 5, 16, 87, 56, 26, 27, 28, 29, 30, 31];

fn emit_hip_layout(rocm_path: &str, rocm_major: u32, rocm_minor: u32) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let layout_rs = out_dir.join("hip_layout.rs");

    println!("cargo:rustc-check-cfg=cfg(rocm_major, values(any()))");
    println!("cargo:rustc-check-cfg=cfg(rocm_minor, values(any()))");

    // Fall-back offsets by ROCm major version (x86-64 measurements).
    let (def_size, def_gcn) = match rocm_major {
        ..=4 => (1184, 944),
        5 => (1344, 1096),
        _ => (1472, 1160),
    };

    let layout = match probe_layout(rocm_path) {
        Ok(layout) => layout,
        Err(e) => {
            println!(
                "cargo:warning=zengpu-hip: layout probe failed ({e}); \
                 using ROCm {rocm_major}.{rocm_minor} defaults"
            );
            HipLayout {
                prop_size: def_size,
                gcn_arch_off: def_gcn,
                attrs: ATTR_DEFAULTS,
            }
        }
    };

    let mut content = format!(
        "// Auto-generated by zengpu-hip build.rs — do not edit.\n\
         pub const ROCM_VERSION_MAJOR:      u32   = {rocm_major};\n\
         pub const ROCM_VERSION_MINOR:      u32   = {rocm_minor};\n\
         pub const HIP_PROP_SIZE:           usize = {};\n\
         pub const HIP_PROP_GCN_ARCH_OFF:   usize = {};\n\
         pub const HIP_PROP_GCN_ARCH_LEN:   usize = 256;\n",
        layout.prop_size, layout.gcn_arch_off,
    );
    for (name, value) in ATTR_NAMES.iter().zip(layout.attrs) {
        content.push_str(&format!(
            "pub const HIP_ATTR_{}: i32 = {value};\n",
            screaming(name)
        ));
    }
    fs::write(&layout_rs, content).expect("write hip_layout.rs");
}

fn screaming(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i != 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

fn probe_layout(rocm_path: &str) -> Result<HipLayout, String> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = out_dir.join("hip_probe.cpp");
    let bin = out_dir.join("hip_probe");

    let mut program = String::from(
        "#define __HIP_PLATFORM_AMD__\n\
         #include <hip/hip_runtime_api.h>\n\
         #include <stdio.h>\n\
         #include <stddef.h>\n\
         int main(void) {\n\
         printf(\"%zu %zu\", sizeof(hipDeviceProp_t), offsetof(hipDeviceProp_t, gcnArchName));\n",
    );
    for name in ATTR_NAMES {
        program.push_str(&format!(
            "printf(\" %d\", (int)hipDeviceAttribute{name});\n"
        ));
    }
    program.push_str("printf(\"\\n\");\nreturn 0;\n}\n");
    fs::write(&src, program).map_err(|e| e.to_string())?;

    let compilers = [
        format!("{rocm_path}/bin/hipcc"),
        format!("{rocm_path}/lib/llvm/bin/clang++"),
        "clang++".to_string(),
        "g++".to_string(),
    ];

    for compiler in &compilers {
        if !PathBuf::from(compiler).exists() && compiler != "clang++" && compiler != "g++" {
            continue;
        }
        let mut cmd = Command::new(compiler);
        cmd.arg(&src).arg("-o").arg(&bin);
        if !compiler.ends_with("hipcc") {
            cmd.arg(format!("-I{rocm_path}/include"))
                .arg("-D__HIP_PLATFORM_AMD__");
        }
        if let Ok(o) = cmd.output()
            && o.status.success()
            && let Ok(run) = Command::new(&bin).output()
            && run.status.success()
        {
            let s = String::from_utf8_lossy(&run.stdout);
            let n: Vec<i64> = s
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            if n.len() == 2 + ATTR_NAMES.len() {
                let mut attrs = [0i32; ATTR_NAMES.len()];
                for (slot, value) in attrs.iter_mut().zip(&n[2..]) {
                    *slot = *value as i32;
                }
                return Ok(HipLayout {
                    prop_size: n[0] as usize,
                    gcn_arch_off: n[1] as usize,
                    attrs,
                });
            }
        }
    }
    Err("no suitable compiler found".into())
}
