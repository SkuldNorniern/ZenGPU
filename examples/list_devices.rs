use std::process::exit;

use zengpu::{DeviceType, detect_backends};

fn main() {
    let backends = detect_backends();

    if backends.is_empty() {
        eprintln!("No GPU backends available.");
        exit(1);
    }

    // Vulkan on macOS needs MoltenVK; without it, say so instead of leaving Vulkan out.
    #[cfg(all(target_os = "macos", feature = "vulkan"))]
    let vulkan_missing = !backends.iter().any(|b| b.name == "vulkan");

    for b in &backends {
        println!("[{}]  {} device(s)", b.name, b.adapters.len());
        for (i, a) in b.adapters.iter().enumerate() {
            let kind = match a.device_type {
                DeviceType::Discrete => "discrete",
                DeviceType::Integrated => "integrated",
                DeviceType::Virtual => "virtual",
                DeviceType::Cpu => "cpu",
                DeviceType::Unknown => "unknown",
            };
            println!(
                "  [{}] {}  ({})  vendor={:#06x}  device={:#06x}",
                i, a.name, kind, a.vendor, a.device
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "vulkan"))]
    if vulkan_missing {
        println!("[vulkan]  no device (MoltenVK not found)");
    }
}
