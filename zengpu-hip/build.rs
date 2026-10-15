#![allow(clippy::unwrap_used, clippy::expect_used)]

/// Build script for zengpu-hip.
///
/// 1. Finds ROCm (`ROCM_PATH`, else a /opt/rocm scan). Without libamdhip64 and libhiprtc, it links
///    stubs so the crate still builds.
/// 2. Probes `hip_version.h` and emits `cfg(rocm_major="N")` and `cfg(rocm_minor="N")`.
/// 3. Compiles a C probe for the `hipDeviceProp_t` size, the `gcnArchName` offset and the
///    `hipDeviceAttribute_t` values, which shift between ROCm majors. Without an AMD compiler, or
///    when cross-compiling, it falls back to ROCm 6/7 x86-64 defaults.
use cc::Build;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| find_rocm());

    let lib_dirs = [
        PathBuf::from(format!("{rocm_path}/lib")),
        PathBuf::from(format!("{rocm_path}/lib/x64")),
        PathBuf::from(format!("{rocm_path}/lib64")),
    ];
    for dir in &lib_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    let has_hip = lib_dirs.iter().any(|dir| {
        has_library(
            dir,
            &["libamdhip64.so", "libamdhip64.dylib", "amdhip64.lib"],
        )
    });
    if !has_hip {
        Build::new().file("src/hip_stub.c").compile("amdhip64");
    }

    let has_hiprtc = lib_dirs
        .iter()
        .any(|dir| has_library(dir, &["libhiprtc.so", "libhiprtc.dylib", "hiprtc.lib"]));
    if !has_hiprtc {
        Build::new().file("src/hiprtc_stub.c").compile("hiprtc");
    }

    println!("cargo:rustc-link-lib=amdhip64");
    println!("cargo:rustc-link-lib=hiprtc");

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed={rocm_path}/include/hip/hip_version.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/hip_stub.c");
    println!("cargo:rerun-if-changed=src/hiprtc_stub.c");

    let (major, minor) = detect_rocm_version(&rocm_path);
    if major < 5 {
        println!(
            "cargo:warning=zengpu-hip: ROCm {major}.{minor} predates the 5.0 device-attribute \
             renumbering; ROCm 5.0 or newer is required for correct device queries"
        );
    }
    println!("cargo:rustc-cfg=rocm_major=\"{major}\"");
    println!("cargo:rustc-cfg=rocm_minor=\"{minor}\"");

    emit_hip_layout(&rocm_path, major, minor);
}

fn find_rocm() -> String {
    for candidate in &["/opt/rocm", "/usr/local/rocm", "/usr"] {
        if PathBuf::from(format!("{candidate}/lib/libamdhip64.so")).exists()
            || PathBuf::from(format!("{candidate}/lib64/libamdhip64.so")).exists()
        {
            return candidate.to_string();
        }
    }
    "/opt/rocm".to_string()
}

fn detect_rocm_version(rocm_path: &str) -> (u32, u32) {
    let version_h = format!("{rocm_path}/include/hip/hip_version.h");
    if let Ok(text) = fs::read_to_string(&version_h) {
        let major = parse_define(&text, "HIP_VERSION_MAJOR");
        let minor = parse_define(&text, "HIP_VERSION_MINOR");
        if let (Some(maj), Some(min)) = (major, minor) {
            println!("cargo:warning=zengpu-hip: detected ROCm {maj}.{min} from {version_h}");
            return (maj, min);
        }
    }

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

/// Measured by the probe. Struct offsets and the `hipDeviceAttribute_t` values differ across
/// releases, and the values were renumbered in ROCm 5.0.
struct HipLayout {
    prop_size: usize,
    gcn_arch_off: usize,
    attrs: [i32; ATTR_NAMES.len()],
}

/// `hipDeviceAttribute_t` members queried, in the order of the generated constants.
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

/// ROCm 5.0 to 7.2 values, used only when the probe cannot run.
const ATTR_DEFAULTS: [i32; ATTR_NAMES.len()] = [63, 5, 16, 87, 56, 26, 27, 28, 29, 30, 31];

fn emit_hip_layout(rocm_path: &str, rocm_major: u32, rocm_minor: u32) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let layout_rs = out_dir.join("hip_layout.rs");

    println!("cargo:rustc-check-cfg=cfg(hip_props_r0600)");
    println!("cargo:rustc-check-cfg=cfg(rocm_major, values(any()))");
    println!("cargo:rustc-check-cfg=cfg(rocm_minor, values(any()))");
    if rocm_major >= 6 {
        println!("cargo:rustc-cfg=hip_props_r0600");
    }

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
        "// Auto-generated by zengpu-hip build.rs: do not edit.\n\
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

fn has_library(dir: &Path, names: &[&str]) -> bool {
    names.iter().any(|name| dir.join(name).exists())
}
