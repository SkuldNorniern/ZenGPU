use zengpu::{DeviceType, detect_backends};

fn main() {
    let backends = detect_backends();

    if backends.is_empty() {
        eprintln!("No GPU backends available.");
        std::process::exit(1);
    }

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
}
