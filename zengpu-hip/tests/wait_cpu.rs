//! Checks that waiting does not burn a core. Linux only (reads /proc/self/stat).
//!
//! Runs in its own binary: threads from other tests would count.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(target_os = "linux")]

use std::fs;
use std::time::{Duration, Instant};

use zengpu_hal::{
    AdapterRequest, Bindings, ComputePipelineDesc, DeviceRequest, GpuInstance, Scalar, ShaderDesc,
};
use zengpu_hip::HipInstance;

const SPIN: &str = r#"
extern "C" __global__ void spin(unsigned int ticks) {
    unsigned long long start = wall_clock64();
    while (wall_clock64() - start < ticks) {}
}
"#;

/// user + system, seconds. Assumes the usual 100 Hz tick.
fn cpu_seconds() -> f64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap();
    let rest = &stat[stat.rfind(')').unwrap() + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks: u64 = fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap();
    ticks as f64 / 100.0
}

#[test]
fn waiting_for_a_long_kernel_does_not_spin() {
    let Ok(inst) = HipInstance::new() else { return };
    let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
        return;
    };
    let dev = adapter.open(DeviceRequest::default()).unwrap();
    let shader = dev.create_shader(ShaderDesc::hip(SPIN)).unwrap();
    let pipeline = dev
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "spin",
            block: [1, 1, 1],
        })
        .unwrap();
    let bindings = Bindings {
        buffers: &[],
        textures: &[],
        images: &[],
        scalars: &[Scalar::U32(50_000_000)],
    };

    let (wall, cpu) = (Instant::now(), cpu_seconds());
    dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
    let blocking = (wall.elapsed().as_secs_f64(), cpu_seconds() - cpu);

    let (wall, cpu) = (Instant::now(), cpu_seconds());
    let sub = dev.submit(0, pipeline, bindings, [1, 1, 1]).unwrap();
    sub.wait(Duration::from_secs(30)).unwrap();
    let bounded = (wall.elapsed().as_secs_f64(), cpu_seconds() - cpu);

    println!(
        "hip: dispatch wait  wall {:.2}s cpu {:.2}s",
        blocking.0, blocking.1
    );
    println!(
        "hip: bounded wait   wall {:.2}s cpu {:.2}s",
        bounded.0, bounded.1
    );
    assert!(
        blocking.0 > 0.4 && bounded.0 > 0.4,
        "kernel too short to judge"
    );
    assert!(blocking.1 < 0.5 * blocking.0, "dispatch spun: {blocking:?}");
    assert!(
        bounded.1 < 0.5 * bounded.0,
        "bounded wait spun: {bounded:?}"
    );

    dev.destroy_pipeline(pipeline);
    dev.destroy_shader(shader);
}
