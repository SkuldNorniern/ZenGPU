use zengpu_spirv::zengpu_spirv;

// ── Compute shaders ───────────────────────────────────────────────────────────

#[test]
fn compute_global_id_x() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_copy(buf: Buf<f32>, out: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                out[i] = buf[i];
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_global_id_y() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute(local_size_x = 8, local_size_y = 8)]
        fn cs_2d(out: BufMut<f32>, width: u32) {
            let x: u32 = global_id().x;
            let y: u32 = global_id().y;
            let idx: u32 = y * width + x;
            out[idx] = 1.0;
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_global_id_z() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_3d(out: BufMut<f32>) {
            let z: u32 = global_id().z;
            out[z] = 0.0;
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_scale_buffer() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute(local_size_x = 64)]
        fn cs_scale(src: Buf<f32>, dst: BufMut<f32>, len: u32, scale: f32) {
            let i: u32 = global_id().x;
            if i < len {
                dst[i] = src[i] * scale;
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_negate() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_neg(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                dst[i] = -src[i];
            }
        }
    );
    assert!(spv_valid(SPV));
}

// ── SPIR-V header check ───────────────────────────────────────────────────────

fn spv_valid(spv: &[u32]) -> bool {
    spv.len() >= 5 && spv[0] == 0x07230203
}
