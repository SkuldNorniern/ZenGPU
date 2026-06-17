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

#[test]
fn compute_comparison_gt_le() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_gt(src: Buf<f32>, dst: BufMut<f32>, len: u32, thresh: f32) {
            let i: u32 = global_id().x;
            if i < len {
                let v: f32 = src[i];
                if v > thresh {
                    dst[i] = 1.0;
                }
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_comparison_eq_ne_u32() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_eq(out: BufMut<f32>, target: u32) {
            let i: u32 = global_id().x;
            if i == target {
                out[i] = 1.0;
            }
            if i != target {
                out[i] = 0.0;
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_comparison_eq_ne_f32() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_feq(src: Buf<f32>, dst: BufMut<f32>) {
            let i: u32 = global_id().x;
            let v: f32 = src[i];
            if v == 0.0 {
                dst[i] = -1.0;
            }
            if v != 1.0 {
                dst[i] = 2.0;
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_logical_and_or() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_logic(src: Buf<f32>, dst: BufMut<f32>, lo: u32, hi: u32) {
            let i: u32 = global_id().x;
            if i >= lo && i < hi {
                dst[i] = src[i];
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_if_else() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_abs_clamped(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                let v: f32 = src[i];
                if v > 0.0 {
                    dst[i] = v;
                } else {
                    dst[i] = -v;
                }
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_builtin_abs_sqrt() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_math(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                dst[i] = sqrt(abs(src[i]));
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_builtin_floor_ceil_fract() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_rounding(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                let v: f32 = src[i];
                dst[i] = floor(v) + ceil(v) + fract(v);
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_builtin_clamp_mix() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_interp(src: Buf<f32>, dst: BufMut<f32>, len: u32) {
            let i: u32 = global_id().x;
            if i < len {
                let v: f32 = src[i];
                let c: f32 = clamp(v, 0.0, 1.0);
                dst[i] = mix(0.0, 1.0, c);
            }
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn compute_builtin_min_max_pow() {
    const SPV: &[u32] = zengpu_spirv!(
        #[compute]
        fn cs_minmax(src: Buf<f32>, dst: BufMut<f32>, len: u32, scale: f32) {
            let i: u32 = global_id().x;
            if i < len {
                let v: f32 = src[i];
                dst[i] = min(max(v, 0.0), pow(scale, 2.0));
            }
        }
    );
    assert!(spv_valid(SPV));
}

// ── SPIR-V header check ───────────────────────────────────────────────────────

fn spv_valid(spv: &[u32]) -> bool {
    spv.len() >= 5 && spv[0] == 0x07230203
}
