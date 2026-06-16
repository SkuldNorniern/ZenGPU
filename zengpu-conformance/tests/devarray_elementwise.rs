//! Cross-backend conformance for `zengpu-compute`'s elementwise operations:
//! device-array addition and ReLU remain resident and conformant.
//!
//! Skips if no Vulkan driver is present.

use std::sync::Arc;

use zengpu_compute::BufferPool;
use zengpu_compute::elementwise::ElementwiseKernels;
use zengpu_conformance::{as_bytes_f32, from_bytes_f32};
use zengpu_cpu::{CpuDevice, CpuKernelCtx};
use zengpu_hal::{AdapterRequest, DType, DeviceRequest, GpuDevice, GpuInstance, Scalar};
use zengpu_vulkan::VulkanInstance;

fn vulkan_device() -> Option<Arc<dyn GpuDevice>> {
    let inst = VulkanInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    Some(Arc::from(adapter.open(DeviceRequest::default()).ok()?))
}

fn read_f32(buf: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
}

fn write_f32(buf: &mut [u8], i: usize, v: f32) {
    buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

fn register_cpu_kernels(cpu: &CpuDevice, kernels: &ElementwiseKernels) {
    cpu.register_kernel(
        kernels.add_pipeline,
        Box::new(|ctx: &mut CpuKernelCtx| {
            let len = match ctx.scalars.first() {
                Some(&Scalar::U32(n)) => n as usize,
                _ => return,
            };
            let a: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[0], i)).collect();
            let b: Vec<f32> = (0..len).map(|i| read_f32(&ctx.buffers[1], i)).collect();
            for i in 0..len {
                write_f32(&mut ctx.buffers[2], i, a[i] + b[i]);
            }
        }),
    );
    cpu.register_kernel(
        kernels.relu_pipeline,
        Box::new(|ctx: &mut CpuKernelCtx| {
            let len = match ctx.scalars.first() {
                Some(&Scalar::U32(n)) => n as usize,
                _ => return,
            };
            for i in 0..len {
                let v = read_f32(&ctx.buffers[0], i);
                write_f32(&mut ctx.buffers[1], i, v.max(0.0));
            }
        }),
    );
}

#[test]
fn devarray_add_relu_cpu_vs_vulkan() {
    let Some(vk) = vulkan_device() else { return };
    let cpu: Arc<dyn GpuDevice> = Arc::new(CpuDevice::new());

    let cpu_kernels = ElementwiseKernels::new(&*cpu).unwrap();
    register_cpu_kernels(
        cpu.as_any().downcast_ref::<CpuDevice>().unwrap(),
        &cpu_kernels,
    );
    let vk_kernels = ElementwiseKernels::new(&*vk).unwrap();

    let cpu_pool = BufferPool::new(cpu.clone());
    let vk_pool = BufferPool::new(vk.clone());

    const N: u32 = 256;
    let shape = vec![N];
    // Mix of negative and positive values to exercise relu's branch.
    let a_data: Vec<f32> = (0..N).map(|i| i as f32 - 128.0).collect();
    let b_data: Vec<f32> = (0..N).map(|i| 100.0 - i as f32).collect();

    let mut results: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for (label, dev, pool, kernels) in [
        ("cpu", &cpu, &cpu_pool, &cpu_kernels),
        ("vulkan", &vk, &vk_pool, &vk_kernels),
    ] {
        let a = pool.alloc(shape.clone(), DType::F32).unwrap();
        let b = pool.alloc(shape.clone(), DType::F32).unwrap();
        dev.write_buffer(a.buffer, 0, as_bytes_f32(&a_data))
            .unwrap();
        dev.write_buffer(b.buffer, 0, as_bytes_f32(&b_data))
            .unwrap();

        let sum = kernels.add(&**dev, pool, &a, &b).unwrap();
        let relu_out = kernels.relu(&**dev, pool, &sum).unwrap();

        let sum_bytes = dev.read_buffer(sum.buffer, 0, sum.size_bytes()).unwrap();
        let relu_bytes = dev
            .read_buffer(relu_out.buffer, 0, relu_out.size_bytes())
            .unwrap();
        let sum_out = from_bytes_f32(&sum_bytes);
        let relu_result = from_bytes_f32(&relu_bytes);

        for i in 0..N as usize {
            let expected_sum = a_data[i] + b_data[i];
            assert!(
                (sum_out[i] - expected_sum).abs() < 1e-4,
                "[{label}] add[{i}] = {}, expected {expected_sum}",
                sum_out[i]
            );
            let expected_relu = expected_sum.max(0.0);
            assert!(
                (relu_result[i] - expected_relu).abs() < 1e-4,
                "[{label}] relu[{i}] = {}, expected {expected_relu}",
                relu_result[i]
            );
        }

        pool.free(a);
        pool.free(b);
        pool.free(sum);
        pool.free(relu_out);
        results.push((sum_out, relu_result));
    }

    let (cpu_sum, cpu_relu) = &results[0];
    let (vk_sum, vk_relu) = &results[1];
    for i in 0..N as usize {
        assert!(
            (cpu_sum[i] - vk_sum[i]).abs() < 1e-4,
            "add[{i}]: cpu={} != vulkan={}",
            cpu_sum[i],
            vk_sum[i]
        );
        assert!(
            (cpu_relu[i] - vk_relu[i]).abs() < 1e-4,
            "relu[{i}]: cpu={} != vulkan={}",
            cpu_relu[i],
            vk_relu[i]
        );
    }

    cpu_kernels.destroy(&*cpu);
    vk_kernels.destroy(&*vk);
}
