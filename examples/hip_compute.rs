//! AMD ROCm/HIP compute example.
//!
//! Run with:
//!   cargo run --example hip_compute --features hip
//!
//! Demonstrates: device detection, buffer round-trip, register-blocked SGEMM,
//! wave-level reduction, memory bandwidth measurement, multi-GPU dispatch.

use zengpu::{
    Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, DeviceRequest,
    GpuAdapter, GpuDevice, GpuInstance, MemoryUsage, Scalar, ShaderDesc,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // ── 1. Build instance and enumerate HIP adapters ──────────────────────────

    let instance = zengpu::Instance::builder()
        .hip()
        .expect("ROCm/HIP unavailable — is libamdhip64.so on LD_LIBRARY_PATH?")
        .build();

    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No AMD GPUs found via HIP. Exiting.");
        return;
    }

    println!("\n=== ZenGPU HIP backend — {} device(s) ===\n", adapters.len());
    for a in &adapters {
        let info = a.info();
        println!("  GPU: {} [{:?}]", info.name, info.device_type);
    }
    println!();

    // ── 2. Capability report via hip sub-crate ────────────────────────────────

    #[cfg(feature = "hip")]
    {
        use zengpu::hip::HipInstance;
        if let Ok(hip_inst) = HipInstance::new() {
            for d in hip_inst.device_infos() {
                let report = d.capabilities.report(&d.name, &d.gfx_target);
                println!("─── Capability Report ──────────────────────────────");
                println!("{report}");
                println!("────────────────────────────────────────────────────\n");
            }
        }
    }

    // ── 3. Vec-add smoke test on GPU 0 ────────────────────────────────────────

    let device = adapters[0].open(DeviceRequest::default()).expect("open device");

    const VEC_ADD: &str = r#"
extern "C" __global__
void vec_add(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

    let n: usize = 1 << 20; // 1 M elements
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|_| 1.0f32).collect();
    let bytes = (n * 4) as u64;
    let st = BufferUsage::STORAGE | BufferUsage::READBACK;

    let ba = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    let bb = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    let bc = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();

    device.write_buffer(ba, 0, as_bytes(&a_data)).unwrap();
    device.write_buffer(bb, 0, as_bytes(&b_data)).unwrap();

    let shader   = device.create_shader(ShaderDesc::hip(VEC_ADD)).unwrap();
    let pipeline = device.create_compute_pipeline(ComputePipelineDesc { shader, entry: "vec_add", block: [256, 1, 1] }).unwrap();

    let bindings = Bindings {
        buffers:  &[ba.index(), bb.index(), bc.index()],
        textures: &[],
        scalars:  &[Scalar::U32(n as u32)],
    };
    device.dispatch(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1]).unwrap();

    let raw = device.read_buffer(bc, 0, bytes).unwrap();
    let c: &[f32] = from_bytes(&raw);
    assert!((c[0]         - 1.0f32).abs() < 1e-5, "c[0] = {}", c[0]);
    assert!((c[n - 1]     - (n as f32)).abs() < 1e-2, "c[n-1] = {}", c[n - 1]);
    println!("[GPU 0] vec_add {}M elements: OK  (c[0]={} c[N-1]={})", n / (1 << 20), c[0], c[n - 1]);

    device.destroy_pipeline(pipeline); device.destroy_shader(shader);
    device.destroy_buffer(ba); device.destroy_buffer(bb); device.destroy_buffer(bc);

    // ── 4. Register-blocked SGEMM benchmark ──────────────────────────────────

    println!();
    for (i, adapter) in adapters.iter().enumerate() {
        let dev  = adapter.open(DeviceRequest::default()).unwrap();
        let name = adapter.info().name.clone();

        for &sz in &[2048usize, 4096] {
            let gflops = run_sgemm_opt(&dev, sz, sz, sz);
            println!("[GPU {i} – {name}] SGEMM {sz}³: {gflops:.0} GFLOP/s");
        }
    }

    // ── 5. Memory bandwidth ───────────────────────────────────────────────────

    println!();
    for (i, adapter) in adapters.iter().enumerate() {
        let dev  = adapter.open(DeviceRequest::default()).unwrap();
        let name = adapter.info().name.clone();
        let gb_s = run_bandwidth(&dev);
        println!("[GPU {i} – {name}] bandwidth (1 GB r+w): {gb_s:.1} GB/s");
    }

    // ── 6. Multi-GPU parallel SGEMM 4096³ ────────────────────────────────────

    if adapters.len() >= 2 {
        println!();
        let handles: Vec<_> = adapters.iter().enumerate().map(|(i, a)| {
            let dev  = std::sync::Arc::new(a.open(DeviceRequest::default()).unwrap());
            let name = a.info().name.clone();
            std::thread::spawn(move || {
                let gflops = run_sgemm_opt(&dev, 4096, 4096, 4096);
                println!("[GPU {i} – {name}] parallel SGEMM 4096³: {gflops:.0} GFLOP/s");
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    }

    println!("\nAll done.");
}

// ── Optimised SGEMM (64×64 tile, 4×4 register blocking) ─────────────────────

const SGEMM_OPT: &str = r#"
/* 64×64 macro-tile, 16×16 block (256 threads), 4×4 register tile per thread.
   Each thread computes rows {ty, ty+16, ty+32, ty+48} × cols {tx, tx+16, tx+32, tx+48}.
   LDS padded by +1 in the K dimension to eliminate bank conflicts.            */
#define TILE 64
#define TK   16
#define DIM  16

extern "C" __global__
__attribute__((reqd_work_group_size(DIM, DIM, 1)))
void sgemm_opt(const float* __restrict__ A, const float* __restrict__ B,
               float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TK + 1];  /* [m][k] row-major, +1 pad avoids bank conflict */
    __shared__ float Bs[TILE][TK + 1];  /* [n][k] row-major, same pad                   */

    int tx  = (int)threadIdx.x;
    int ty  = (int)threadIdx.y;
    int bx  = (int)blockIdx.x;
    int by  = (int)blockIdx.y;
    int tid = ty * DIM + tx;  /* 0..255 */

    float acc[4][4] = {};

    for (int k0 = 0; k0 < (int)K; k0 += TK) {
        /* ── Load As[TILE][TK] from global (1024 elements, 4 per thread) ── */
        /* idx encodes (mi, ki): mi = idx/TK, ki = idx%TK.
           16 consecutive tids share the same mi and write to adjacent ki → coalesced. */
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int mi  = idx / TK;
            int ki  = idx % TK;
            int gm  = by * TILE + mi;
            int gk  = k0 + ki;
            As[mi][ki] = (gm < (int)M && gk < (int)K) ? A[gm * (int)K + gk] : 0.0f;
        }
        /* ── Load Bs[TILE][TK] from global ── */
        /* ni = idx%TILE, ki = idx/TILE: 64 consecutive tids share ki and load
           consecutive N positions → full 256-byte coalesced transaction.       */
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int idx = tid + s * 256;
            int ni  = idx % TILE;
            int ki  = idx / TILE;
            int gn  = bx * TILE + ni;
            int gk  = k0 + ki;
            Bs[ni][ki] = (gk < (int)K && gn < (int)N) ? B[gk * (int)N + gn] : 0.0f;
        }
        __syncthreads();

        /* ── 4×4 register-blocked dot-product ── */
        #pragma unroll
        for (int ki = 0; ki < TK; ki++) {
            float a0 = As[ty          ][ki];
            float a1 = As[ty + DIM    ][ki];
            float a2 = As[ty + 2*DIM  ][ki];
            float a3 = As[ty + 3*DIM  ][ki];
            float b0 = Bs[tx          ][ki];
            float b1 = Bs[tx + DIM    ][ki];
            float b2 = Bs[tx + 2*DIM  ][ki];
            float b3 = Bs[tx + 3*DIM  ][ki];
            acc[0][0] += a0*b0; acc[0][1] += a0*b1; acc[0][2] += a0*b2; acc[0][3] += a0*b3;
            acc[1][0] += a1*b0; acc[1][1] += a1*b1; acc[1][2] += a1*b2; acc[1][3] += a1*b3;
            acc[2][0] += a2*b0; acc[2][1] += a2*b1; acc[2][2] += a2*b2; acc[2][3] += a2*b3;
            acc[3][0] += a3*b0; acc[3][1] += a3*b1; acc[3][2] += a3*b2; acc[3][3] += a3*b3;
        }
        __syncthreads();
    }

    /* ── Write 4×4 output block ── */
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

fn run_sgemm_opt(device: &dyn GpuDevice, m: usize, n: usize, k: usize) -> f64 {
    let a: Vec<f32> = (0..m*k).map(|i| (i % 7)  as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k*n).map(|i| (i % 11) as f32 * 0.01).collect();
    let st = BufferUsage::STORAGE | BufferUsage::READBACK;

    let ba = device.create_buffer(BufferDesc { size: (m*k*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    let bb = device.create_buffer(BufferDesc { size: (k*n*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    let bc = device.create_buffer(BufferDesc { size: (m*n*4) as u64, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();

    device.write_buffer(ba, 0, as_bytes(&a)).unwrap();
    device.write_buffer(bb, 0, as_bytes(&b)).unwrap();

    let shader   = device.create_shader(ShaderDesc::hip(SGEMM_OPT)).unwrap();
    let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
        shader, entry: "sgemm_opt", block: [16, 16, 1],
    }).unwrap();

    let grid     = [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1];
    let bindings = Bindings {
        buffers:  &[ba.index(), bb.index(), bc.index()],
        textures: &[],
        scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
    };

    // Two warm-up passes.
    device.dispatch(pipeline, bindings, grid).unwrap();
    device.dispatch(pipeline, bindings, grid).unwrap();

    const REPS: u32 = 3;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS { device.dispatch(pipeline, bindings, grid).unwrap(); }
    let ms     = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e6);

    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
    device.destroy_buffer(ba);
    device.destroy_buffer(bb);
    device.destroy_buffer(bc);
    gflops
}

// ── Bandwidth benchmark ───────────────────────────────────────────────────────

const SCALE_F4: &str = r#"
extern "C" __global__
void scale_f4(const float4* __restrict__ in, float4* __restrict__ out,
              float scale, unsigned int n4) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        out[i] = v;
    }
}
"#;

fn run_bandwidth(device: &dyn GpuDevice) -> f64 {
    const N: usize = 256 * 1024 * 1024; // 1 GB
    const BLOCK: u32 = 256;
    let n4 = (N / 4) as u32;

    let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
    let bytes = (N * 4) as u64;
    let st    = BufferUsage::STORAGE | BufferUsage::READBACK;

    let src = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    let dst = device.create_buffer(BufferDesc { size: bytes, usage: st, memory: MemoryUsage::GpuOnly }).unwrap();
    device.write_buffer(src, 0, as_bytes(&data)).unwrap();

    let shader   = device.create_shader(ShaderDesc::hip(SCALE_F4)).unwrap();
    let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
        shader, entry: "scale_f4", block: [BLOCK, 1, 1],
    }).unwrap();

    let grid     = [(n4 + BLOCK - 1) / BLOCK, 1, 1];
    let bindings = Bindings {
        buffers:  &[src.index(), dst.index()],
        textures: &[],
        scalars:  &[Scalar::F32(2.0), Scalar::U32(n4)],
    };

    device.dispatch(pipeline, bindings, grid).unwrap(); // warm-up

    const REPS: u32 = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS { device.dispatch(pipeline, bindings, grid).unwrap(); }
    let ms   = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    let gb_s = (2.0 * bytes as f64) / (ms * 1e6);

    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
    device.destroy_buffer(src);
    device.destroy_buffer(dst);
    gb_s
}

// ── cast helpers ─────────────────────────────────────────────────────────────

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn from_bytes(v: &[u8]) -> &[f32] {
    assert_eq!(v.len() % 4, 0);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
}
