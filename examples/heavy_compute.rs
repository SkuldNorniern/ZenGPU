use std::time::Instant;

use zengpu::{
    AdapterRequest, BackendPreference, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc,
    DeviceRequest, Instance, MemoryUsage, PowerPreference, Scalar, ZslShader, zsl,
};

/// Naive f32 SGEMM: C[row,col] = Σ_k A[row,k] * B[k,col].
///
/// 16×16 workgroup — one thread per output element. Correct for all sizes;
/// not a performance target. Use this as the reference for verifying
/// backend-specific tiled kernels.
const SGEMM: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32 }
    @workgroup_size(16, 16, 1)
    kernel sgemm(
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let row = id.y
        let col = id.x
        if row < p.m && col < p.n {
            let acc: f32 = 0.0
            for ki in 0..p.k {
                acc = acc + a[row * p.k + ki] * b[ki * p.n + col]
            }
            c[row * p.n + col] = acc
        }
    }
);

fn as_bytes_f32(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v != 0)
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = env_u32("ZENGPU_HEAVY_DIM", 1024);
    let reps = env_u32("ZENGPU_HEAVY_REPS", 4);

    // Build an instance with all compiled-in backends.
    let inst = {
        let b = Instance::builder();
        let b = b.try_vulkan().unwrap_or_else(|b| b);
        #[cfg(feature = "cuda")]
        let b = b.cuda();
        let b = b.try_hip().unwrap_or_else(|b| b);
        b.build()
    };

    // Select backend from ZENGPU_BACKEND env var, default to Auto.
    let backend_pref = match std::env::var("ZENGPU_BACKEND")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "vulkan" => BackendPreference::Vulkan,
        "cuda" => BackendPreference::Cuda,
        "hip" => BackendPreference::Hip,
        "metal" => BackendPreference::Metal,
        _ => BackendPreference::Auto,
    };

    let adapter = inst
        .request_adapter(AdapterRequest {
            backend: backend_pref,
            power: PowerPreference::HighPerformance,
        })
        .ok_or("no GPU adapter found — enable at least one backend feature")?;

    let info = adapter.info();
    eprintln!("ZenGPU heavy_compute: {} ({:?})", info.name, info.backend);

    let device = adapter.open(DeviceRequest::default())?;

    let m = dim as usize;
    let n = dim as usize;
    let k = dim as usize;
    let gflops_total = 2.0 * m as f64 * n as f64 * k as f64 * reps as f64 / 1e9;
    eprintln!("SGEMM {m}×{k} @ {k}×{n}, {reps} rep(s)  ~{gflops_total:.1} GFLOP");

    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 + 3) % 31) as f32 * 0.125 - 2.0)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1 - 1.4)
        .collect();

    let st = BufferUsage::STORAGE | BufferUsage::READBACK;
    let ba = device.create_buffer(BufferDesc {
        size: (m * k * 4) as u64,
        usage: st,
        memory: MemoryUsage::GpuOnly,
    })?;
    let bb = device.create_buffer(BufferDesc {
        size: (k * n * 4) as u64,
        usage: st,
        memory: MemoryUsage::GpuOnly,
    })?;
    let bc = device.create_buffer(BufferDesc {
        size: (m * n * 4) as u64,
        usage: st,
        memory: MemoryUsage::GpuOnly,
    })?;
    device.write_buffer(ba, 0, as_bytes_f32(&a_data))?;
    device.write_buffer(bb, 0, as_bytes_f32(&b_data))?;

    // Select the right compiled form for the active backend, then create pipeline.
    let (shader_desc, entry) = SGEMM.for_backend(info.backend);
    let shader = device.create_shader(shader_desc)?;
    let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
        shader,
        entry,
        block: [16, 16, 1],
    })?;

    let grid = [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1];
    let bindings = Bindings {
        buffers: &[ba.index(), bb.index(), bc.index()],
        textures: &[],
        scalars: &[
            Scalar::U32(m as u32),
            Scalar::U32(n as u32),
            Scalar::U32(k as u32),
        ],
    };

    // Warmup.
    device.dispatch(pipeline, bindings, grid)?;

    let start = Instant::now();
    for rep in 0..reps {
        let t = Instant::now();
        device.dispatch(pipeline, bindings, grid)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e6);
        eprintln!(
            "pass {:>3}/{reps}  {ms:7.1} ms  {gflops:8.1} GFLOP/s",
            rep + 1
        );
    }
    let elapsed = start.elapsed();

    let out = from_bytes_f32(&device.read_buffer(bc, 0, (m * n * 4) as u64)?);

    let samples = [(0, 0), (m / 3, n / 3), (m / 2, n / 2), (m - 1, n - 1)];
    for (row, col) in samples {
        let expected: f32 = (0..k)
            .map(|i| a_data[row * k + i] * b_data[i * n + col])
            .sum();
        let got = out[row * n + col];
        if (got - expected).abs() / expected.abs().max(1.0) > 0.01 {
            return Err(
                format!("SGEMM mismatch [{row},{col}]: got {got}, expected {expected}").into(),
            );
        }
    }

    let total_gflops = gflops_total / elapsed.as_secs_f64();
    eprintln!(
        "SGEMM OK  {reps}×({m}×{k} @ {k}×{n}) in {elapsed:?}  ({total_gflops:.1} GFLOP/s avg)"
    );

    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
    device.destroy_buffer(ba);
    device.destroy_buffer(bb);
    device.destroy_buffer(bc);

    Ok(())
}
