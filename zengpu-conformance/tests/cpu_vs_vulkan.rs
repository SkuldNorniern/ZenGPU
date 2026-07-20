//! Cross-backend conformance: CPU oracle vs Vulkan.
//!
//! Each test gracefully skips if no Vulkan driver is present.

use zengpu_conformance::{compare_full, compare_vec_add, run_buffer_suite, run_dispatch};
use zengpu_cpu::CpuDevice;
use zengpu_hal::{
    AdapterRequest, ComputePipelineDesc, DeviceRequest, GpuDevice, GpuInstance, Scalar, ShaderDesc,
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

// ── ZSL `fn` (inlined helper functions) + bitwise operators ────────────────
//
// A real 64-bit SplitMix64-style hash (the one `xytron_mppi`'s CPU noise
// generator uses, split into `(hi, lo)` u32 pairs since ZSL has no native
// u64), implemented entirely with `fn`/`out` params/bitwise ops and run on
// real Vulkan hardware, checked bit-exact against a plain Rust reference.

// hi/lo splits of the exact constants xytron_mppi's `core::noise::mix64` uses.
const MIX64_C1_HI: u32 = 3210233709; // 0xbf58_476d
const MIX64_C1_LO: u32 = 484763065; //  0x1ce4_e5b9
const MIX64_C2_HI: u32 = 2496678331; // 0x94d0_49bb
const MIX64_C2_LO: u32 = 321982955; //  0x1331_11eb

const MIX64_SHADER: ZslShader = zsl!(
    push P { n: u32 }

    fn mulhi_u32(u: u32, v: u32, out result: u32) {
        let u0 = u & 65535
        let u1 = u >> 16
        let v0 = v & 65535
        let v1 = v >> 16

        let t = u0 * v0
        let k = t >> 16

        t = u1 * v0 + k
        let w1 = t & 65535
        let w2 = t >> 16

        t = u0 * v1 + w1
        k = t >> 16

        result = u1 * v1 + w2 + k
    }

    // Truncated 64x64->64 unsigned multiply (only the low 64 bits of the
    // true 128-bit product survive a `wrapping_mul`, so `a_hi*b_hi`, which
    // only ever contributes to bits >=64, is correctly never computed).
    fn mul64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32, out r_hi: u32, out r_lo: u32) {
        let cross: u32 = 0
        mulhi_u32(a_lo, b_lo, cross)
        r_lo = a_lo * b_lo
        r_hi = a_hi * b_lo + a_lo * b_hi + cross
    }

    // Logical right shift of a 64-bit (hi,lo) pair by 1..=31 bits.
    fn shr64(hi: u32, lo: u32, n: u32, out r_hi: u32, out r_lo: u32) {
        r_lo = (lo >> n) | (hi << (32 - n))
        r_hi = hi >> n
    }

    fn mix64(hi: u32, lo: u32, out out_hi: u32, out out_lo: u32) {
        let s1_hi: u32 = 0
        let s1_lo: u32 = 0
        shr64(hi, lo, 30, s1_hi, s1_lo)
        let v1_hi = hi ^ s1_hi
        let v1_lo = lo ^ s1_lo

        let m1_hi: u32 = 0
        let m1_lo: u32 = 0
        mul64(v1_hi, v1_lo, 3210233709, 484763065, m1_hi, m1_lo)

        let s2_hi: u32 = 0
        let s2_lo: u32 = 0
        shr64(m1_hi, m1_lo, 27, s2_hi, s2_lo)
        let v2_hi = m1_hi ^ s2_hi
        let v2_lo = m1_lo ^ s2_lo

        let m2_hi: u32 = 0
        let m2_lo: u32 = 0
        mul64(v2_hi, v2_lo, 2496678331, 321982955, m2_hi, m2_lo)

        let s3_hi: u32 = 0
        let s3_lo: u32 = 0
        shr64(m2_hi, m2_lo, 31, s3_hi, s3_lo)
        out_hi = m2_hi ^ s3_hi
        out_lo = m2_lo ^ s3_lo
    }

    @workgroup_size(64)
    kernel hash(
        in_hi: device buffer<u32>, in_lo: device buffer<u32>,
        out_hi: device mut buffer<u32>, out_lo: device mut buffer<u32>,
        p: P, id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            let h: u32 = 0
            let l: u32 = 0
            mix64(in_hi[i], in_lo[i], h, l)
            out_hi[i] = h
            out_lo[i] = l
        }
    }
);

/// `xytron_mppi::core::noise::mix64`, reproduced exactly (not imported: this
/// crate has no dependency on xytron_mppi, and the point is to compare
/// against a plain, independently-obviously-correct Rust implementation).
fn mix64_reference(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    debug_assert_eq!((0xbf58_476d_1ce4_e5b9u64 >> 32) as u32, MIX64_C1_HI);
    debug_assert_eq!(0xbf58_476d_1ce4_e5b9u64 as u32, MIX64_C1_LO);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    debug_assert_eq!((0x94d0_49bb_1331_11ebu64 >> 32) as u32, MIX64_C2_HI);
    debug_assert_eq!(0x94d0_49bb_1331_11ebu64 as u32, MIX64_C2_LO);
    value ^ (value >> 31)
}

fn mix64_spv_bytes() -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            MIX64_SHADER.spv.as_ptr() as *const u8,
            std::mem::size_of_val(MIX64_SHADER.spv),
        )
    }
}

#[test]
fn mix64_matches_rust_reference_on_vulkan() {
    let Some(vk) = vulkan_device() else { return };

    let shader = vk
        .create_shader(ShaderDesc::spirv(mix64_spv_bytes()))
        .unwrap();
    let pipeline = vk
        .create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry: "main",
            block: [64, 1, 1],
        })
        .unwrap();

    let inputs: Vec<u64> = vec![
        0,
        1,
        u64::MAX,
        0x9e37_79b9_7f4a_7c15,
        0x1234_5678_9abc_def0,
        0xffff_ffff_0000_0000,
        0x0000_0000_ffff_ffff,
        0x8000_0000_0000_0001,
        123456789,
        u64::MAX - 1,
    ];
    let in_hi: Vec<u32> = inputs.iter().map(|v| (v >> 32) as u32).collect();
    let in_lo: Vec<u32> = inputs.iter().map(|v| *v as u32).collect();
    let n = inputs.len() as u32;

    let as_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let in_hi_bytes = as_bytes(&in_hi);
    let in_lo_bytes = as_bytes(&in_lo);

    let outputs = run_dispatch(
        &*vk,
        pipeline,
        &[&in_hi_bytes, &in_lo_bytes],
        &[(n as u64) * 4, (n as u64) * 4],
        &[Scalar::U32(n)],
        [n.div_ceil(64), 1, 1],
    );
    let out_hi: Vec<u32> = outputs[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let out_lo: Vec<u32> = outputs[1]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    for (i, &value) in inputs.iter().enumerate() {
        let expected = mix64_reference(value);
        let actual = ((out_hi[i] as u64) << 32) | out_lo[i] as u64;
        assert_eq!(
            actual, expected,
            "mix64({value:#x}): GPU={actual:#x}, expected={expected:#x}"
        );
    }

    vk.destroy_pipeline(pipeline);
    vk.destroy_shader(shader);
}
