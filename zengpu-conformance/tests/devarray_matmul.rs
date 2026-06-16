//! Cross-backend conformance for `zengpu-blas`'s f32 GEMM.
//!
//! Skips if no Vulkan driver is present.

use std::sync::Arc;

use zengpu_blas::GemmKernel;
use zengpu_compute::BufferPool;
use zengpu_conformance::{as_bytes_f32, from_bytes_f32};
use zengpu_cpu::{CpuDevice, CpuKernelCtx};
use zengpu_hal::{AdapterRequest, DType, DeviceRequest, GpuDevice, GpuInstance, Scalar};
use zengpu_vulkan::VulkanInstance;

fn vulkan_device() -> Option<Arc<dyn GpuDevice>> {
    let inst = VulkanInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

/// CPU oracle for [`GEMM_SPV`](zengpu_blas): `C[m,n] = sum_k A[m,k] * B[k,n]`,
/// matching the shader's loop order exactly.
fn register_cpu_gemm_kernel(cpu: &CpuDevice, pipeline: zengpu_hal::PipelineHandle) {
    cpu.register_kernel(
        pipeline,
        Box::new(|ctx: &mut CpuKernelCtx| {
            let (m, n, k) = match ctx.scalars[..] {
                [Scalar::U32(m), Scalar::U32(n), Scalar::U32(k)] => {
                    (m as usize, n as usize, k as usize)
                }
                _ => return,
            };
            let read = |buf: &[u8], i: usize| {
                f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
            };
            let a = &ctx.buffers[0];
            let b = &ctx.buffers[1];
            let mut c = vec![0u8; m * n * 4];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for i in 0..k {
                        sum += read(a, row * k + i) * read(b, i * n + col);
                    }
                    c[(row * n + col) * 4..(row * n + col) * 4 + 4]
                        .copy_from_slice(&sum.to_le_bytes());
                }
            }
            ctx.buffers[2] = c;
        }),
    );
}

#[test]
fn devarray_matmul_cpu_vs_vulkan() {
    let Some(vk) = vulkan_device() else { return };
    let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());

    let cpu_gemm = GemmKernel::new(&*cpu).unwrap();
    register_cpu_gemm_kernel(
        cpu.as_any().downcast_ref::<CpuDevice>().unwrap(),
        cpu_gemm.pipeline,
    );
    let vk_gemm = GemmKernel::new(&*vk).unwrap();

    let cpu_pool = BufferPool::new(cpu.clone());
    let vk_pool = BufferPool::new(vk.clone());

    // Non-multiples of the 16x16 workgroup to exercise the bounds check.
    const M: u32 = 17;
    const K: u32 = 33;
    const N: u32 = 9;
    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 - 3.0).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 - 2.0).collect();

    let mut expected = vec![0f32; (M * N) as usize];
    for row in 0..M as usize {
        for col in 0..N as usize {
            let mut sum = 0.0f32;
            for i in 0..K as usize {
                sum += a_data[row * K as usize + i] * b_data[i * N as usize + col];
            }
            expected[row * N as usize + col] = sum;
        }
    }

    let mut results = Vec::new();
    for (label, dev, pool, gemm) in [
        ("cpu", &cpu, &cpu_pool, &cpu_gemm),
        ("vulkan", &vk, &vk_pool, &vk_gemm),
    ] {
        let a = pool.alloc(vec![M, K], DType::F32).unwrap();
        let b = pool.alloc(vec![K, N], DType::F32).unwrap();
        dev.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))
            .unwrap();
        dev.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))
            .unwrap();

        let c = gemm.gemm(&**dev, pool, &a, &b).unwrap();
        assert_eq!(c.shape, vec![M, N], "[{label}] gemm output shape");
        let out = from_bytes_f32(&dev.read_buffer(c.buffer, 0, c.size_bytes()).unwrap());

        for i in 0..(M * N) as usize {
            assert!(
                (out[i] - expected[i]).abs() < 1e-3,
                "[{label}] gemm[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }

        pool.free(a);
        pool.free(b);
        pool.free(c);
        results.push(out);
    }

    for (i, (cpu_val, vk_val)) in results[0].iter().zip(&results[1]).enumerate() {
        assert!(
            (cpu_val - vk_val).abs() < 1e-3,
            "gemm[{i}]: cpu={cpu_val} != vulkan={vk_val}"
        );
    }

    cpu_gemm.destroy(&*cpu);
    vk_gemm.destroy(&*vk);
}
