use std::sync::Arc;
use std::time::Instant;

use zengpu::{
    AdapterRequest, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, DeviceRequest,
    GpuDevice, GpuInstance, MemoryUsage, Scalar, ShaderDesc,
};

#[cfg(feature = "blas")]
use zengpu::{BufferPool, DType, GemmKernel};

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

#[cfg(feature = "hip")]
const SGEMM_HIP: &str = r#"
#define TILE 64
#define TK   16
#define DIM  16
extern "C" __global__ __launch_bounds__(DIM * DIM)
void sgemm(const float* __restrict__ A,
           const float* __restrict__ B,
           float* __restrict__ C,
           unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TK + 1];
    __shared__ float Bs[TILE][TK + 1];
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int bx = (int)blockIdx.x;
    int by = (int)blockIdx.y;
    int tid = ty * DIM + tx;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < (int)K; k0 += TK) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int mi = idx / TK; int ki = idx % TK;
            int gm = by * TILE + mi; int gk = k0 + ki;
            As[mi][ki] = (gm < (int)M && gk < (int)K) ? A[gm * (int)K + gk] : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int ni = idx % TILE; int ki = idx / TILE;
            int gn = bx * TILE + ni; int gk = k0 + ki;
            Bs[ni][ki] = (gk < (int)K && gn < (int)N) ? B[gk * (int)N + gn] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int ki = 0; ki < TK; ki++) {
            float a0 = As[ty][ki]; float a1 = As[ty+DIM][ki];
            float a2 = As[ty+2*DIM][ki]; float a3 = As[ty+3*DIM][ki];
            float b0 = Bs[tx][ki]; float b1 = Bs[tx+DIM][ki];
            float b2 = Bs[tx+2*DIM][ki]; float b3 = Bs[tx+3*DIM][ki];
            acc[0][0]+=a0*b0; acc[0][1]+=a0*b1; acc[0][2]+=a0*b2; acc[0][3]+=a0*b3;
            acc[1][0]+=a1*b0; acc[1][1]+=a1*b1; acc[1][2]+=a1*b2; acc[1][3]+=a1*b3;
            acc[2][0]+=a2*b0; acc[2][1]+=a2*b1; acc[2][2]+=a2*b2; acc[2][3]+=a2*b3;
            acc[3][0]+=a3*b0; acc[3][1]+=a3*b1; acc[3][2]+=a3*b2; acc[3][3]+=a3*b3;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = by * TILE + ty + i * DIM;
        if (gm < (int)M) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int gn = bx * TILE + tx + j * DIM;
                if (gn < (int)N) C[gm * (int)N + gn] = acc[i][j];
            }
        }
    }
}
"#;

#[cfg(feature = "hip")]
fn run_hip(dim: u32, reps: u32) -> Result<(), Box<dyn std::error::Error>> {
    use zengpu::hip::HipInstance;

    let inst = HipInstance::new().map_err(|e| format!("HIP init failed: {e}"))?;
    let adapter = inst
        .request_adapter(AdapterRequest::default())
        .ok_or("no HIP adapter")?;
    let name = adapter.info().name.clone();
    let device: Arc<dyn GpuDevice> = Arc::from(adapter.open(DeviceRequest::default())?);

    eprintln!("ZenGPU heavy compute [hip]: {name}");

    let m = dim as usize;
    let n = dim as usize;
    let k = dim as usize;
    let bytes = (m * k * 4) as u64;
    let gflops_total = 2.0 * m as f64 * n as f64 * k as f64 * reps as f64 / 1e9;
    eprintln!("running SGEMM workload: {reps} x ({m}×{k} @ {k}×{n})  ~{gflops_total:.1} GFLOP");

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 17 + 3) % 31) as f32 * 0.125 - 2.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 29) as f32 * 0.1 - 1.4).collect();

    let st = BufferUsage::STORAGE | BufferUsage::READBACK;
    let ba = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly })?;
    let bb = device.create_buffer(BufferDesc { size: (k * n * 4) as u64, usage: st, memory: MemoryUsage::GpuOnly })?;
    let bc = device.create_buffer(BufferDesc { size: (m * n * 4) as u64, usage: st, memory: MemoryUsage::GpuOnly })?;
    device.write_buffer(ba, 0, as_bytes_f32(&a_data))?;
    device.write_buffer(bb, 0, as_bytes_f32(&b_data))?;

    let shader   = device.create_shader(ShaderDesc::hip(SGEMM_HIP))?;
    let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
        shader, entry: "sgemm", block: [16, 16, 1],
    })?;
    let grid = [((n as u32 + 63) / 64), ((m as u32 + 63) / 64), 1];
    let bindings = Bindings {
        buffers:  &[ba.index(), bb.index(), bc.index()],
        textures: &[],
        scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
    };

    device.dispatch(pipeline, bindings, grid)?;

    let start = Instant::now();
    for rep in 0..reps {
        device.dispatch(pipeline, bindings, grid)?;
        if rep == 0 || (rep + 1) % 4 == 0 || rep + 1 == reps {
            eprintln!("completed SGEMM pass {}/{}", rep + 1, reps);
        }
    }
    let elapsed = start.elapsed();

    let out = from_bytes_f32(&device.read_buffer(bc, 0, (m * n * 4) as u64)?);

    let samples = [(0usize, 0usize), (m / 3, n / 3), (m / 2, n / 2), (m - 1, n - 1)];
    for (row, col) in samples {
        let expected: f32 = (0..k).map(|i| a_data[row * k + i] * b_data[i * n + col]).sum();
        let got = out[row * n + col];
        if (got - expected).abs() / expected.abs().max(1.0) > 0.01 {
            return Err(format!("SGEMM mismatch [{row},{col}]: got {got}, expected {expected}").into());
        }
    }

    let gflops_s = gflops_total / elapsed.as_secs_f64();
    eprintln!("heavy SGEMM OK: {reps} x ({m}×{k} @ {k}×{n}) in {elapsed:?}  ({gflops_s:.1} GFLOP/s)");

    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
    device.destroy_buffer(ba);
    device.destroy_buffer(bb);
    device.destroy_buffer(bc);
    Ok(())
}

#[cfg(feature = "blas")]
fn run_blas(dim: u32, reps: u32) -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::var("ZENGPU_BACKEND").unwrap_or_default();
    let device: Arc<dyn GpuDevice> = match backend.to_ascii_lowercase().as_str() {
        #[cfg(feature = "cuda")]
        "cuda" => {
            use zengpu::cuda::CudaInstance;
            let inst = CudaInstance::new();
            let adapter = inst.request_adapter(AdapterRequest::default()).ok_or("no CUDA adapter")?;
            eprintln!("ZenGPU heavy compute [cuda]: {}", adapter.info().name);
            Arc::from(adapter.open(DeviceRequest::default())?)
        }
        _ => {
            #[cfg(not(feature = "vulkan"))]
            return Err("set ZENGPU_BACKEND=hip (or cuda with cuda feature)".into());
            #[cfg(feature = "vulkan")]
            {
                use zengpu::VulkanInstance;
                let inst = VulkanInstance::new()?;
                let adapter = inst.request_adapter(AdapterRequest::default()).ok_or("no Vulkan adapter")?;
                eprintln!("ZenGPU heavy compute [vulkan]: {}", adapter.info().name);
                Arc::from(adapter.open(DeviceRequest::default())?)
            }
        }
    };

    let pool = BufferPool::new(device.clone());
    let gemm = GemmKernel::new(&*device)?;

    let m = dim as usize; let k = dim as usize; let n = dim as usize;
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 * reps as f64 / 1e9;
    eprintln!("running GEMM workload: {reps} x ({m}×{k} @ {k}×{n})  ~{gflops:.1} GFLOP");

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 17 + 3) % 31) as f32 * 0.125 - 2.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 29) as f32 * 0.1 - 1.4).collect();

    let a = pool.alloc(vec![m as u32, k as u32], DType::F32)?;
    let b = pool.alloc(vec![k as u32, n as u32], DType::F32)?;
    device.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))?;
    device.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))?;

    let start = Instant::now();
    let mut final_c = None;
    for rep in 0..reps {
        if let Some(prev) = final_c.take() { pool.free(prev); }
        let c = gemm.gemm(&*device, &pool, &a, &b)?;
        if rep == 0 || (rep + 1) % 4 == 0 || rep + 1 == reps {
            eprintln!("completed GEMM pass {}/{}", rep + 1, reps);
        }
        final_c = Some(c);
    }
    let c = final_c.ok_or("no GEMM passes ran")?;
    let out = from_bytes_f32(&device.read_buffer(c.buffer, 0, c.size_bytes())?);
    let elapsed = start.elapsed();

    let samples = [(0usize, 0usize), (m / 3, n / 3), (m / 2, n / 2), (m - 1, n - 1)];
    for (row, col) in samples {
        let expected: f32 = (0..k).map(|i| a_data[row * k + i] * b_data[i * n + col]).sum();
        let got = out[row * n + col];
        if (got - expected).abs() > 1e-2 {
            return Err(format!("GEMM mismatch [{row},{col}]: got {got}, expected {expected}").into());
        }
    }

    let gflops_s = gflops / elapsed.as_secs_f64();
    eprintln!("heavy GEMM OK: {reps} x ({m}×{k} @ {k}×{n}) in {elapsed:?}  ({gflops_s:.1} GFLOP/s)");

    pool.free(a); pool.free(b); pool.free(c);
    gemm.destroy(&*device);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim  = env_u32("ZENGPU_HEAVY_DIM", 4096);
    let reps = env_u32("ZENGPU_HEAVY_REPS", 64);
    let backend = std::env::var("ZENGPU_BACKEND").unwrap_or_default();

    match backend.to_ascii_lowercase().as_str() {
        #[cfg(feature = "hip")]
        "hip" => run_hip(dim, reps),
        #[cfg(feature = "blas")]
        _ => run_blas(dim, reps),
        #[cfg(not(feature = "blas"))]
        _ => Err("no backend: enable blas feature or set ZENGPU_BACKEND=hip".into()),
    }
}
