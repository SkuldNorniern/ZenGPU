//! Cross-backend conformance: CPU oracle vs Vulkan.
//!
//! Each test gracefully skips if no Vulkan driver is present.

use zengpu_conformance::{compare_full, compare_vec_add, run_buffer_suite};
use zengpu_cpu::CpuDevice;
use zengpu_hal::{
    AdapterRequest, ComputePipelineDesc, DeviceRequest, GpuDevice, GpuInstance, ShaderDesc,
};
use zengpu_spirv::{ZslShader, zsl};
use zengpu_vulkan::VulkanInstance;

/// vec_add: out[i] = a[i] + b[i] for i in 0..len (matches `ZenGPU/examples/vec_add.rs`).
const VEC_ADD_SHADER: ZslShader = zsl!(
    push P { len: u32 }
    @workgroup_size(256)
    kernel add(a: device buffer<f32>, b: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.len {
            out[i] = a[i] + b[i]
        }
    }
);

fn vec_add_spv_bytes() -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            VEC_ADD_SHADER.spv.as_ptr() as *const u8,
            std::mem::size_of_val(VEC_ADD_SHADER.spv),
        )
    }
}

fn register_vec_add_kernel(dev: &CpuDevice, pipeline: zengpu_hal::PipelineHandle) {
    dev.register_kernel(
        pipeline,
        Box::new(|ctx| {
            let a: Vec<f32> = ctx.buffers[0]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let b: Vec<f32> = ctx.buffers[1]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let out: Vec<u8> = a
                .iter()
                .zip(&b)
                .flat_map(|(x, y)| (x + y).to_le_bytes())
                .collect();
            ctx.buffers[2] = out;
        }),
    );
}

fn cpu_device() -> Box<dyn zengpu_hal::GpuDevice> {
    Box::new(CpuDevice::new())
}

fn vulkan_device() -> Option<Box<dyn zengpu_hal::GpuDevice>> {
    let inst = VulkanInstance::new().ok()?;
    let adapter = inst.request_adapter(AdapterRequest::default())?;
    adapter.open(DeviceRequest::default()).ok()
}

#[test]
fn cpu_buffer_suite() {
    run_buffer_suite("cpu", &*cpu_device());
}

#[test]
fn vulkan_buffer_suite() {
    let Some(dev) = vulkan_device() else { return };
    run_buffer_suite("vulkan", &*dev);
}

#[test]
fn cpu_vs_vulkan() {
    let Some(vk) = vulkan_device() else { return };
    compare_full("cpu", &*cpu_device(), "vulkan", &*vk);
}

#[test]
fn cpu_vs_vulkan_vec_add() {
    let Some(vk) = vulkan_device() else { return };

    let cpu = CpuDevice::new();
    let cpu_shader = cpu
        .create_shader(ShaderDesc::spirv(vec_add_spv_bytes()))
        .unwrap();
    let cpu_pipeline = cpu
        .create_compute_pipeline(ComputePipelineDesc {
            shader: cpu_shader,
            entry: "main",
            block: [256, 1, 1],
        })
        .unwrap();
    register_vec_add_kernel(&cpu, cpu_pipeline);

    let vk_shader = vk
        .create_shader(ShaderDesc::spirv(vec_add_spv_bytes()))
        .unwrap();
    let vk_pipeline = vk
        .create_compute_pipeline(ComputePipelineDesc {
            shader: vk_shader,
            entry: "main",
            block: [256, 1, 1],
        })
        .unwrap();

    compare_vec_add("cpu", &cpu, cpu_pipeline, "vulkan", &*vk, vk_pipeline);

    cpu.destroy_pipeline(cpu_pipeline);
    cpu.destroy_shader(cpu_shader);
    vk.destroy_pipeline(vk_pipeline);
    vk.destroy_shader(vk_shader);
}
