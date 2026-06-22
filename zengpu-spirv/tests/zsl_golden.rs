//! Golden byte-for-byte snapshot of ZSL → SPIR-V output.
//!
//! The other ZSL tests only check that the emitted SPIR-V is *valid*. This one
//! pins the exact words, so a refactor (e.g. routing lowering through a
//! backend-neutral IR) is proven to not change the generated code. If ZSL
//! output legitimately changes, regenerate the digests below.

use zengpu_spirv::zengpu_spirv;

/// FNV-1a 64-bit over the SPIR-V word stream (length-sensitive).
fn digest(words: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h ^= words.len() as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    for &w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

macro_rules! cases {
    () => {{
        let mut v: Vec<(&'static str, u64)> = Vec::new();

        v.push(("compute_copy", digest(zengpu_spirv!(
            #[compute]
            fn cs_copy(buf: Buf<f32>, out: BufMut<f32>, len: u32) {
                let i: u32 = global_id().x;
                if i < len { out[i] = buf[i]; }
            }
        ))));

        v.push(("compute_2d_scale", digest(zengpu_spirv!(
            #[compute(local_size_x = 8, local_size_y = 8)]
            fn cs_2d(out: BufMut<f32>, width: u32) {
                let x: u32 = global_id().x;
                let y: u32 = global_id().y;
                let idx: u32 = y * width + x;
                out[idx] = 1.0;
            }
        ))));

        v.push(("compute_scale", digest(zengpu_spirv!(
            #[compute(local_size_x = 64)]
            fn cs_scale(src: Buf<f32>, dst: BufMut<f32>, len: u32, scale: f32) {
                let i: u32 = global_id().x;
                if i < len { dst[i] = src[i] * scale; }
            }
        ))));

        v.push(("compute_negate", digest(zengpu_spirv!(
            #[compute]
            fn cs_neg(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
                let i: u32 = global_id().x;
                if i < len { dst[i] = -src[i]; }
            }
        ))));

        v.push(("vertex_passthrough", digest(zengpu_spirv!(
            #[vertex]
            fn vs(#[location(0)] pos: Vec4) -> Vec4 { pos }
        ))));

        v.push(("vertex_mat4_push", digest(zengpu_spirv!(
            #[vertex]
            fn vs(#[location(0)] pos: Vec3, mvp: Mat4) -> Vec4 {
                mvp * pos.extend(1.0)
            }
        ))));

        v.push(("fragment_const", digest(zengpu_spirv!(
            #[fragment]
            fn fs() -> Vec4 { Vec4(1.0, 0.0, 0.0, 1.0) }
        ))));

        v
    }};
}

/// Expected digests, captured from known-good output. Regenerate (see below)
/// only when ZSL output is *intended* to change.
const GOLDEN: &[(&str, u64)] = &[
    ("compute_copy", 0xcebd_8729_3ba8_f815),
    ("compute_2d_scale", 0x8536_b96c_a0cb_720d),
    ("compute_scale", 0x1c48_e4e1_f47b_eee1),
    ("compute_negate", 0x6fd5_4da4_a04b_7232),
    ("vertex_passthrough", 0x8d43_ae45_7268_6b77),
    ("vertex_mat4_push", 0xc110_3860_017a_6700),
    ("fragment_const", 0x3498_6854_d982_0394),
];

#[test]
fn spirv_output_is_byte_stable() {
    let actual = cases!();
    assert_eq!(actual.len(), GOLDEN.len(), "case count drifted");
    for ((name, got), (exp_name, exp)) in actual.iter().zip(GOLDEN) {
        assert_eq!(name, exp_name, "case order drifted");
        assert_eq!(
            got, exp,
            "ZSL→SPIR-V output changed for `{name}` (got {got:#018x}, expected {exp:#018x}). \
             If intentional, update GOLDEN."
        );
    }
}
