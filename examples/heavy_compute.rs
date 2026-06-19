//! Heavier compute smoke test for the Vulkan backend.
//!
//! Runs a large GEMM through the public `DeviceArray`/`GemmKernel` path and
//! validates a sample of output cells against CPU-computed expected values.
//!
//! Set `ZENGPU_HEAVY_DIM=2048` (or another square dimension) to scale the
//! workload beyond the default 1024x1024 GEMM.

use std::sync::Arc;
use std::time::Instant;

use zengpu::{
    AdapterRequest, BufferPool, DType, DeviceRequest, GemmKernel, GpuDevice, GpuInstance,
    VulkanInstance,
};

fn as_bytes_f32(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inst = VulkanInstance::new()?;
    let adapter = inst
        .request_adapter(AdapterRequest::default())
        .ok_or("no Vulkan adapter")?;
    eprintln!("ZenGPU heavy compute: {}", adapter.info().name);

    let device: Arc<dyn GpuDevice> = Arc::from(adapter.open(DeviceRequest::default())?);
    let pool = BufferPool::new(device.clone());
    let gemm = GemmKernel::new(&*device)?;

    let dim = std::env::var("ZENGPU_HEAVY_DIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v != 0)
        .unwrap_or(1024);
    let (m, k, n) = (dim, dim, dim);
    eprintln!("running GEMM workload: {m}x{k} @ {k}x{n}");

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
    let c = gemm.gemm(&*device, &pool, &a, &b)?;
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

    eprintln!("heavy GEMM OK: {m}x{k} @ {k}x{n} -> {m}x{n} in {elapsed:?}");

    pool.free(a);
    pool.free(b);
    pool.free(c);
    gemm.destroy(&*device);
    Ok(())
}
