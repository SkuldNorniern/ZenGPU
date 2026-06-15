//! Vector addition on the GPU via ZenGPU bindless compute.
//!
//! Demonstrates the minimal path: upload data → dispatch a compute shader →
//! read back the result. No graphics, no window, no ML — pure general-purpose
//! compute.
//!
//! The GLSL shader receives the three buffer indices and the element count as
//! push constants. Bindings.buffers[0..2] = [a_idx, b_idx, out_idx] packed
//! into the first three push-constant u32s; Bindings.scalars[0] = len.

use inline_spirv::inline_spirv;
use zengpu_hal::{
    AdapterRequest, Bindings, BufferDesc, BufferUsage, DeviceRequest, GpuInstance, MemoryUsage,
    Scalar,
};
use zengpu_vulkan::VulkanInstance;

// ── Shader ────────────────────────────────────────────────────────────────────

/// vec_add: out[i] = a[i] + b[i] for i in 0..len.
///
/// Push constant layout (matches Bindings packing in `dispatch`):
///   offset 0: uint a_idx   = bindings.buffers[0]
///   offset 4: uint b_idx   = bindings.buffers[1]
///   offset 8: uint out_idx = bindings.buffers[2]
///   offset 12: uint len    = bindings.scalars[0]
const SHADER_SPV: &[u32] = inline_spirv!(
    r#"
    #version 450
    #extension GL_EXT_nonuniform_qualifier : require

    layout(set = 0, binding = 0) buffer Buf { float data[]; } g_bufs[];

    layout(push_constant) uniform PC {
        uint a_idx;
        uint b_idx;
        uint out_idx;
        uint len;
    } pc;

    layout(local_size_x = 256) in;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < pc.len) {
            g_bufs[pc.out_idx].data[i] =
                g_bufs[pc.a_idx].data[i] + g_bufs[pc.b_idx].data[i];
        }
    }
    "#,
    comp, vulkan1_2
);

// ── Helpers ───────────────────────────────────────────────────────────────────

fn as_bytes<T>(s: &[T]) -> &[u8] {
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

    let device = adapter.open(DeviceRequest::default())?;

    // ── Upload input data ───────────────────────────────────────────────────
    const N: u32 = 1024;
    let size = (N as u64) * 4; // 4 bytes per f32
    let buf_desc = BufferDesc {
        size,
        usage: BufferUsage::STORAGE | BufferUsage::READBACK,
        memory: MemoryUsage::Upload, // host-visible so we can read back directly
    };

    let ha = device.create_buffer(buf_desc)?;
    let hb = device.create_buffer(buf_desc)?;
    let hout = device.create_buffer(buf_desc)?;

    let a_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..N).map(|i| 100.0 * i as f32).collect();
    device.write_buffer(ha, 0, as_bytes(&a_data))?;
    device.write_buffer(hb, 0, as_bytes(&b_data))?;

    // ── Create compute pipeline ─────────────────────────────────────────────
    let spv_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(SHADER_SPV.as_ptr() as *const u8, SHADER_SPV.len() * 4)
    };
    let shader = device.create_shader(zengpu_hal::ShaderDesc { spirv: spv_bytes })?;
    let pipeline = device.create_compute_pipeline(zengpu_hal::ComputePipelineDesc {
        shader,
        entry: "main",
    })?;

    // ── Dispatch ────────────────────────────────────────────────────────────
    let groups = N.div_ceil(256);
    device.dispatch(
        pipeline,
        Bindings {
            buffers: &[ha.index(), hb.index(), hout.index()],
            scalars: &[Scalar::U32(N)],
            textures: &[],
        },
        [groups, 1, 1],
    )?;

    // ── Verify ──────────────────────────────────────────────────────────────
    let out_bytes = device.read_buffer(hout, 0, size)?;
    let out = from_bytes_f32(&out_bytes);

    let mut errors = 0usize;
    for i in 0..N as usize {
        let expected = a_data[i] + b_data[i];
        let got = out[i];
        if (got - expected).abs() > 1e-4 {
            eprintln!("MISMATCH at [{i}]: got {got}, expected {expected}");
            errors += 1;
            if errors > 10 {
                eprintln!("(more mismatches elided)");
                break;
            }
        }
    }

    if errors == 0 {
        eprintln!("vec_add OK: {N} elements correct");
    } else {
        return Err(format!("{errors} mismatches").into());
    }

    device.destroy_pipeline(pipeline);
    device.destroy_shader(shader);
    device.destroy_buffer(ha);
    device.destroy_buffer(hb);
    device.destroy_buffer(hout);

    Ok(())
}
