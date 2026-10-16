//! Metal overheads and kernel throughput: the HIP `overheads` and `library_kernels` tests minus
//! BLAS. Prints numbers, asserts correctness only. Skips without Metal.

#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use zengpu_compute::elementwise::ElementwiseKernels;
use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    AdapterRequest, Bindings, BufferCopyOp, BufferDesc, BufferUsage, ComputeOp,
    ComputePipelineDesc, DType, DeviceRequest, DispatchOp, GpuDevice, GpuInstance, MemoryUsage,
    Scalar,
};
use zengpu_metal::MetalInstance;
use zengpu_spirv::{ZslShader, zsl};

const TOUCH: ZslShader = zsl!(
    push P { v: u32 }
    @workgroup_size(64)
    kernel touch(out: device mut buffer<u32>, p: P, id: global_id) {
        if id.x == 0 {
            out[0] = p.v
        }
    }
);

const INC: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(64)
    kernel inc(data: device mut buffer<u32>, p: P, id: global_id) {
        if id.x < p.n {
            data[id.x] = data[id.x] + 1
        }
    }
);

fn device() -> Option<Arc<dyn GpuDevice>> {
    let inst = MetalInstance::new();
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

fn per_iter(start: Instant, n: u32) -> Duration {
    start.elapsed() / n
}

#[test]
fn overheads() {
    let Some(dev) = device() else { return };

    let t = Instant::now();
    let (shader, entry) = TOUCH.create(&*dev).unwrap();
    println!("metal: compile + load           {:?}", t.elapsed());
    let t = Instant::now();
    let (again, _) = TOUCH.create(&*dev).unwrap();
    println!("metal: second compile + load    {:?}", t.elapsed());
    dev.destroy_shader(again);

    let pipeline = dev
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
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
    println!("metal: dispatch round trip      {:?}", per_iter(t, N));

    let ops: Vec<DispatchOp<'_>> = (0..N)
        .map(|_| DispatchOp {
            pipeline,
            bindings,
            grid: [1, 1, 1],
        })
        .collect();
    let t = Instant::now();
    dev.dispatch_batch(&ops).unwrap();
    println!("metal: batched launch           {:?}", per_iter(t, N));

    let t = Instant::now();
    for _ in 0..N {
        let sub = dev.submit(0, pipeline, bindings, [1, 1, 1]).unwrap();
        sub.wait(Duration::from_secs(10)).unwrap();
    }
    println!("metal: submit + wait            {:?}", per_iter(t, N));

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
    println!("metal: upload   {mb} MiB           {:.1} GB/s", gbps(up));
    println!("metal: readback {mb} MiB           {:.1} GB/s", gbps(down));

    let small = [1u8; 256];
    let t = Instant::now();
    for _ in 0..N {
        dev.write_buffer(out, 0, &small).unwrap();
    }
    println!("metal: 256 B write              {:?}", per_iter(t, N));

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
        println!(
            "metal: create+destroy {mib:>2} MiB     {:?}",
            per_iter(t, 100)
        );
    }

    let first = dev.read_buffer(out, 0, 4).unwrap();
    assert_eq!(first, [1, 1, 1, 1]);

    dev.destroy_buffer(big);
    dev.destroy_buffer(out);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader(shader);
}

fn upload(pool: &BufferPool, shape: Vec<u32>, values: &[f32]) -> DeviceArray {
    let array = pool.alloc(shape, DType::F32).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    pool.device().write_buffer(array.buffer, 0, &bytes).unwrap();
    array
}

fn download(device: &dyn GpuDevice, array: &DeviceArray) -> Vec<f32> {
    device
        .read_buffer(array.buffer, 0, array.size_bytes())
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Each dispatch reads the one before; the final copy sees them all.
#[test]
fn batched_ops_run_in_order() {
    let Some(dev) = device() else { return };
    let (shader, entry) = INC.create(&*dev).unwrap();
    let pipeline = dev
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
            block: [64, 1, 1],
        })
        .unwrap();
    let n = 1000u32;
    let desc = BufferDesc {
        size: u64::from(n) * 4,
        usage: BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::GpuOnly,
    };
    let (data, copy) = (
        dev.create_buffer(desc).unwrap(),
        dev.create_buffer(desc).unwrap(),
    );
    dev.write_buffer(data, 0, &vec![0; n as usize * 4]).unwrap();
    let scalars = [Scalar::U32(n)];
    let bindings = Bindings {
        buffers: &[data.index()],
        textures: &[],
        images: &[],
        scalars: &scalars,
    };
    let dispatch = DispatchOp {
        pipeline,
        bindings,
        grid: [n.div_ceil(64), 1, 1],
    };
    dev.dispatch_batch(&[dispatch; 100]).unwrap();
    let mut ops = vec![ComputeOp::Dispatch(dispatch); 50];
    ops.push(ComputeOp::CopyBuffer(BufferCopyOp {
        src: data,
        src_offset: 0,
        dst: copy,
        dst_offset: 0,
        len: desc.size,
    }));
    let sub = dev.submit_compute_ops(7, &ops).unwrap();
    assert_eq!(sub.cycle_id(), 7);
    sub.wait(Duration::from_secs(10)).unwrap();
    let words = |b| {
        dev.read_buffer(b, 0, desc.size)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert!(words(data).iter().all(|&v| v == 150));
    assert!(words(copy).iter().all(|&v| v == 150));
    dev.destroy_buffer(data);
    dev.destroy_buffer(copy);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader(shader);
}

#[test]
fn memory_bound_kernel_bandwidth() {
    let Some(device) = device() else { return };
    let pool = BufferPool::new(device.clone());
    let kernels = ElementwiseKernels::new(&*device).unwrap();
    let n = 64u32 << 20;
    let a = pool.alloc(vec![n], DType::F32).unwrap();
    let b = pool.alloc(vec![n], DType::F32).unwrap();
    let (x, y) = (
        upload(&pool, vec![3], &[1.0, -2.0, 0.5]),
        upload(&pool, vec![3], &[2.0, 1.0, 0.25]),
    );
    let sum = kernels.add(&*device, &pool, &x, &y).unwrap();
    assert_eq!(download(&*device, &sum), [3.0, -1.0, 0.75]);
    for array in [x, y, sum] {
        pool.free(array);
    }
    let gb = |arrays: f64, secs: f64| arrays * f64::from(n) * 4.0 / secs / 1e9;

    let out = kernels.add(&*device, &pool, &a, &b).unwrap();
    pool.free(out);
    let t = Instant::now();
    for _ in 0..5 {
        let out = kernels.add(&*device, &pool, &a, &b).unwrap();
        pool.free(out);
    }
    println!(
        "metal: elementwise add  {:.0} GB/s",
        gb(3.0, t.elapsed().as_secs_f64() / 5.0)
    );

    pool.free(a);
    pool.free(b);
    kernels.destroy(&*device);
}
