//! Heavy GEMM smoke test. Runs repeated large GEMMs and validates output.
//!
//! Defaults to 64 × (4096×4096 @ 4096×4096). Set `ZENGPU_HEAVY_DIM` and
//! `ZENGPU_HEAVY_REPS` to scale the workload.
//!
//! Backend selection: set `ZENGPU_BACKEND=cuda` to use the CUDA backend
//! (requires the `cuda` feature). Default is Vulkan.

use std::sync::Arc;
use std::time::Instant;

use zengpu::{AdapterRequest, BufferPool, DType, DeviceRequest, GemmKernel, GpuDevice, GpuInstance};

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

/// Returns `(backend_label, adapter_name, device)`.
#[allow(clippy::type_complexity)]
fn open_device() -> Result<(&'static str, String, Arc<dyn GpuDevice>), Box<dyn std::error::Error>> {
    let backend = std::env::var("ZENGPU_BACKEND").unwrap_or_default();
    match backend.to_ascii_lowercase().as_str() {
        #[cfg(feature = "cuda")]
        "cuda" => {
            use zengpu::cuda::CudaInstance;
            let inst = CudaInstance::new();
            let adapter = inst
                .request_adapter(AdapterRequest::default())
                .ok_or("no CUDA adapter found")?;
            let name = adapter.info().name.clone();
            let device: Arc<dyn GpuDevice> = Arc::from(adapter.open(DeviceRequest::default())?);
            Ok(("cuda", name, device))
        }
        _ => {
            #[cfg(not(feature = "vulkan"))]
            return Err("vulkan feature not enabled; set ZENGPU_BACKEND=cuda".into());
            #[cfg(feature = "vulkan")]
            {
                use zengpu::VulkanInstance;
                let inst = VulkanInstance::new()?;
                let adapter = inst
                    .request_adapter(AdapterRequest::default())
                    .ok_or("no Vulkan adapter found")?;
                let name = adapter.info().name.clone();
                let device: Arc<dyn GpuDevice> = Arc::from(adapter.open(DeviceRequest::default())?);
                Ok(("vulkan", name, device))
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, adapter_name, device) = open_device()?;
    eprintln!("ZenGPU heavy compute [{backend}]: {adapter_name}");

    let pool = BufferPool::new(device.clone());
    let gemm = GemmKernel::new(&*device)?;

    let dim  = env_u32("ZENGPU_HEAVY_DIM", 4096);
    let reps = env_u32("ZENGPU_HEAVY_REPS", 64);
    let (m, k, n) = (dim, dim, dim);
    let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (reps as f64) / 1.0e9;
    eprintln!("running GEMM workload: {reps} x ({m}x{k} @ {k}x{n}) ~= {gflops:.1} GFLOP");

    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 + 3) % 31) as f32 * 0.125 - 2.0)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1 - 1.4)
        .collect();

    let a = pool.alloc(vec![m, k], DType::F32)?;
    let b = pool.alloc(vec![k, n], DType::F32)?;
    device.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))?;
    device.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))?;

    let start = Instant::now();
    let mut final_c = None;
    for rep in 0..reps {
        if let Some(prev) = final_c.take() {
            pool.free(prev);
        }
        let c = gemm.gemm(&*device, &pool, &a, &b)?;
        if rep + 1 == reps || rep == 0 || (rep + 1) % 4 == 0 {
            eprintln!("completed GEMM pass {}/{}", rep + 1, reps);
        }
        final_c = Some(c);
    }
    let c = final_c.ok_or("no GEMM passes ran")?;
    let out = from_bytes_f32(&device.read_buffer(c.buffer, 0, c.size_bytes())?);
    let elapsed = start.elapsed();

    let samples = [
        (0usize, 0usize),
        (0, n as usize - 1),
        (m as usize / 3, n as usize / 3),
        (m as usize / 2, n as usize / 2),
        (m as usize - 1, 0),
        (m as usize - 1, n as usize - 1),
    ];
    for (row, col) in samples {
        let mut expected = 0.0f32;
        for i in 0..k as usize {
            expected += a_data[row * k as usize + i] * b_data[i * n as usize + col];
        }
        let got = out[row * n as usize + col];
        if (got - expected).abs() > 1e-2 {
            return Err(
                format!("GEMM mismatch at [{row},{col}]: got {got}, expected {expected}").into(),
            );
        }
    }

    let achieved = gflops / elapsed.as_secs_f64();
    eprintln!(
        "heavy GEMM OK: {reps} x ({m}x{k} @ {k}x{n}) in {elapsed:?} ({achieved:.1} GFLOP/s)"
    );

    pool.free(a);
    pool.free(b);
    pool.free(c);
    gemm.destroy(&*device);
    Ok(())
}
