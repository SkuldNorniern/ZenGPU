//! Prints host-side overheads (compile, dispatch, batching, transfers). Asserts correctness only.
//! Skips without ROCm.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use zengpu_hal::{
    AdapterRequest, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, DeviceRequest,
    DispatchOp, GpuDevice, GpuInstance, MemoryUsage, Scalar, ShaderDesc,
};
use zengpu_hip::HipInstance;

const TOUCH: &str = r#"
extern "C" __global__ void touch(unsigned int* out, unsigned int v) {
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}
"#;

fn device() -> Option<Box<dyn GpuDevice>> {
    let inst = HipInstance::new().ok()?;
    inst.request_adapter(AdapterRequest::default())?
        .open(DeviceRequest::default())
        .ok()
}

fn per_iter(start: Instant, n: u32) -> Duration {
    start.elapsed() / n
}

#[test]
fn overheads() {
    let Some(dev) = device() else { return };

    let src = format!("{TOUCH}\n// {:?}\n", Instant::now());
    let t = Instant::now();
    let shader = dev.create_shader(ShaderDesc::hip(&src)).unwrap();
    println!("hip: cold compile + load        {:?}", t.elapsed());
    let t = Instant::now();
    let again = dev.create_shader(ShaderDesc::hip(&src)).unwrap();
    println!("hip: cached compile + load      {:?}", t.elapsed());
    dev.destroy_shader(again);

    let pipeline = dev
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "touch",
            block: [64, 1, 1],
        })
        .unwrap();
    let out = dev
        .create_buffer(BufferDesc {
            size: 4096,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        })
        .unwrap();
    let scalars = [Scalar::U32(5)];
    let bindings = Bindings {
        buffers: &[out.index()],
        textures: &[],
        images: &[],
        scalars: &scalars,
    };

    dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
    const N: u32 = 2000;
    let t = Instant::now();
    for _ in 0..N {
        dev.dispatch(pipeline, bindings, [1, 1, 1]).unwrap();
    }
    println!("hip: dispatch round trip        {:?}", per_iter(t, N));

    let ops: Vec<DispatchOp<'_>> = (0..N)
        .map(|_| DispatchOp {
            pipeline,
            bindings,
            grid: [1, 1, 1],
        })
        .collect();
    let t = Instant::now();
    dev.dispatch_batch(&ops).unwrap();
    println!("hip: batched launch             {:?}", per_iter(t, N));

    let t = Instant::now();
    for _ in 0..N {
        let sub = dev.submit(0, pipeline, bindings, [1, 1, 1]).unwrap();
        sub.wait(Duration::from_secs(10)).unwrap();
    }
    println!("hip: submit + wait              {:?}", per_iter(t, N));

    let mb = 256usize;
    let big = dev
        .create_buffer(BufferDesc {
            size: (mb << 20) as u64,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        })
        .unwrap();
    let host = vec![7u8; mb << 20];
    dev.write_buffer(big, 0, &host).unwrap();
    let t = Instant::now();
    dev.write_buffer(big, 0, &host).unwrap();
    let up = t.elapsed();
    let mut back = vec![1u8; mb << 20];
    dev.read_buffer_into(big, 0, &mut back).unwrap();
    let t = Instant::now();
    dev.read_buffer_into(big, 0, &mut back).unwrap();
    let down = t.elapsed();
    assert_eq!(back[12345], 7);
    let gbps = |d: Duration| (mb as f64 / 1024.0) / d.as_secs_f64();
    println!("hip: upload   {mb} MiB             {:.1} GB/s", gbps(up));
    println!("hip: readback {mb} MiB             {:.1} GB/s", gbps(down));

    let small = [1u8; 256];
    let t = Instant::now();
    for _ in 0..N {
        dev.write_buffer(out, 0, &small).unwrap();
    }
    println!("hip: 256 B write                {:?}", per_iter(t, N));

    for mib in [1usize, 64] {
        let desc = BufferDesc {
            size: (mib << 20) as u64,
            usage: BufferUsage::STORAGE,
            memory: MemoryUsage::GpuOnly,
        };
        let t = Instant::now();
        for _ in 0..100 {
            let b = dev.create_buffer(desc).unwrap();
            dev.destroy_buffer(b);
        }
        println!("hip: create+destroy {mib:>2} MiB       {:?}", per_iter(t, 100));
    }

    let first = dev.read_buffer(out, 0, 4).unwrap();
    assert_eq!(first, [1, 1, 1, 1]);

    dev.destroy_buffer(big);
    dev.destroy_buffer(out);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader(shader);
}
