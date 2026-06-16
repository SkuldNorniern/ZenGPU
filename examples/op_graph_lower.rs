//! Example of lowering an ML compute graph into ZenGPU operations.
//!
//! `Graph`/`Op`/`lower_and_run` below stand in for an ML compute-graph
//! compiler (Laminax or otherwise): a tiny tensor graph that lowers each node
//! to a single ZenGPU dispatch/BLAS call over resident [`DeviceArray`]s,
//! reading back only the final result. None of this lives in ZenGPU — it owns
//! no op-graph; ZenGPU only supplies `DeviceArray`, `BufferPool`,
//! `ElementwiseKernels`, and `GemmKernel`.
//!
//! Graph computed: `relu(A @ B + C)`.

use std::sync::Arc;

use zengpu::{
    AdapterRequest, BufferPool, DType, DeviceArray, DeviceRequest, ElementwiseKernels, GemmKernel,
    GpuDevice, GpuInstance, VulkanInstance,
};

// ── Stand-in for a compute-graph compiler's op-graph (not part of ZenGPU) ─────

/// A node in the toy graph. `Input` references an array passed in by the
/// caller; the rest reference earlier node ids (so the graph is already
/// topologically ordered — the compiler's DAG scheduler would produce this
/// order).
enum Op {
    Input(usize),
    Matmul(usize, usize),
    Add(usize, usize),
    Relu(usize),
}

struct Graph {
    nodes: Vec<Op>,
}

/// Lower `graph` to ZenGPU dispatch/BLAS calls, one per node, over resident
/// arrays. No host round-trips between nodes — the caller reads back only
/// the final value.
fn lower_and_run(
    device: &dyn GpuDevice,
    pool: &BufferPool,
    gemm: &GemmKernel,
    ew: &ElementwiseKernels,
    graph: &Graph,
    inputs: &[DeviceArray],
) -> zengpu_hal::Result<DeviceArray> {
    let mut values: Vec<Option<DeviceArray>> = (0..graph.nodes.len()).map(|_| None).collect();
    for (id, op) in graph.nodes.iter().enumerate() {
        let value = match *op {
            Op::Input(i) => inputs[i].clone(),
            Op::Matmul(a, b) => gemm.gemm(
                device,
                pool,
                values[a].as_ref().unwrap(),
                values[b].as_ref().unwrap(),
            )?,
            Op::Add(a, b) => ew.add(
                device,
                pool,
                values[a].as_ref().unwrap(),
                values[b].as_ref().unwrap(),
            )?,
            Op::Relu(a) => ew.relu(device, pool, values[a].as_ref().unwrap())?,
        };
        values[id] = Some(value);
    }
    Ok(values.pop().unwrap().unwrap())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn as_bytes_f32(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let inst = VulkanInstance::new()?;
    let adapter = inst
        .request_adapter(AdapterRequest::default())
        .ok_or("no Vulkan adapter")?;
    eprintln!("ZenGPU compute: {}", adapter.info().name);

    let device: Arc<dyn GpuDevice> = Arc::from(adapter.open(DeviceRequest::default())?);
    let pool = BufferPool::new(device.clone());
    let gemm = GemmKernel::new(&*device)?;
    let ew = ElementwiseKernels::new(&*device)?;

    // A: [M,K], B: [K,N], C: [M,N] -> relu(A @ B + C): [M,N]
    const M: u32 = 8;
    const K: u32 = 4;
    const N: u32 = 6;
    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 5) as f32 - 2.0).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i % 3) as f32 - 1.0).collect();
    let c_data: Vec<f32> = (0..M * N).map(|i| (i % 7) as f32 - 3.0).collect();

    let a = pool.alloc(vec![M, K], DType::F32)?;
    let b = pool.alloc(vec![K, N], DType::F32)?;
    let c = pool.alloc(vec![M, N], DType::F32)?;
    device.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))?;
    device.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))?;
    device.write_buffer(c.buffer, 0, as_bytes_f32(&c_data))?;

    // node 0..2: inputs A, B, C; node 3: A @ B; node 4: + C; node 5: relu
    let graph = Graph {
        nodes: vec![
            Op::Input(0),
            Op::Input(1),
            Op::Input(2),
            Op::Matmul(0, 1),
            Op::Add(3, 2),
            Op::Relu(4),
        ],
    };

    let result = lower_and_run(&*device, &pool, &gemm, &ew, &graph, &[a, b, c])?;
    let out = from_bytes_f32(&device.read_buffer(result.buffer, 0, result.size_bytes())?);

    // CPU reference for relu(A @ B + C).
    let mut errors = 0usize;
    for row in 0..M as usize {
        for col in 0..N as usize {
            let mut sum = 0.0f32;
            for i in 0..K as usize {
                sum += a_data[row * K as usize + i] * b_data[i * N as usize + col];
            }
            let expected = (sum + c_data[row * N as usize + col]).max(0.0);
            let got = out[row * N as usize + col];
            if (got - expected).abs() > 1e-4 {
                eprintln!("MISMATCH at [{row},{col}]: got {got}, expected {expected}");
                errors += 1;
            }
        }
    }

    if errors == 0 {
        eprintln!("op_graph_lower OK: relu(A @ B + C) correct for {M}x{K} @ {K}x{N} + {M}x{N}");
    } else {
        return Err(format!("{errors} mismatches").into());
    }

    gemm.destroy(&*device);
    ew.destroy(&*device);
    Ok(())
}
