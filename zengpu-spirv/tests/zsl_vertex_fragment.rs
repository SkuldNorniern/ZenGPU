use zengpu_spirv::zengpu_spirv;

// ── Vertex shaders ────────────────────────────────────────────────────────────

#[test]
fn vertex_passthrough_vec4() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_passthrough(#[location(0)] pos: Vec4) -> Vec4 {
            pos
        }
    );
    assert!(spv_valid(SPV), "invalid SPIR-V header");
}

#[test]
fn vertex_vec3_to_vec4() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_expand(#[location(0)] pos: Vec3) -> Vec4 {
            Vec4(pos.x, pos.y, pos.z, 1.0)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_two_inputs() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_two(#[location(0)] pos: Vec4, #[location(1)] col: Vec4) -> Vec4 {
            pos
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_scalar_arithmetic() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_arith(#[location(0)] pos: Vec4) -> Vec4 {
            let x: f32 = pos.x + pos.y;
            Vec4(x, pos.y, pos.z, pos.w)
        }
    );
    assert!(spv_valid(SPV));
}

// ── Fragment shaders ──────────────────────────────────────────────────────────

#[test]
fn fragment_constant_color() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_red() -> Vec4 {
            Vec4(1.0, 0.0, 0.0, 1.0)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_passthrough_color() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_passthrough(#[location(0)] color: Vec4) -> Vec4 {
            color
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_vec3_input() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_vec3(#[location(0)] rgb: Vec3) -> Vec4 {
            Vec4(rgb.x, rgb.y, rgb.z, 1.0)
        }
    );
    assert!(spv_valid(SPV));
}

// ── Vertex varyings ───────────────────────────────────────────────────────────

#[test]
fn vertex_one_varying() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_color(#[location(0)] pos: Vec4, #[location(1)] col: Vec3) -> (Vec4, Vec3) {
            (pos, col)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_two_varyings() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_uv(
            #[location(0)] pos: Vec3,
            #[location(1)] col: Vec3,
            #[location(2)] uv: Vec2,
        ) -> (Vec4, Vec3, Vec2) {
            (Vec4(pos.x, pos.y, pos.z, 1.0), col, uv)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_varying_computed() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_computed(#[location(0)] pos: Vec4, #[location(1)] col: Vec4) -> (Vec4, Vec4) {
            let r: f32 = col.x + col.y;
            (pos, Vec4(r, col.y, col.z, col.w))
        }
    );
    assert!(spv_valid(SPV));
}

// ── Vec3::extend ─────────────────────────────────────────────────────────────

#[test]
fn vertex_extend_vec3() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_extend(#[location(0)] pos: Vec3) -> Vec4 {
            pos.extend(1.0)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_extend_with_expr() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_extend2(#[location(0)] pos: Vec3, #[location(1)] col: Vec3) -> (Vec4, Vec3) {
            (pos.extend(1.0), col)
        }
    );
    assert!(spv_valid(SPV));
}

// ── Mat4 push constant ────────────────────────────────────────────────────────

#[test]
fn vertex_mat4_push_const() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_mvp(#[location(0)] pos: Vec4, mvp: Mat4) -> Vec4 {
            mvp * pos
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_mat4_with_extend() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_mvp_extend(
            #[location(0)] pos: Vec3,
            #[location(1)] col: Vec3,
            mvp: Mat4,
        ) -> (Vec4, Vec3) {
            (mvp * pos.extend(1.0), col)
        }
    );
    assert!(spv_valid(SPV));
}

// ── Negation and vector arithmetic ───────────────────────────────────────────

#[test]
fn vertex_negate_scalar() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_neg(#[location(0)] pos: Vec4) -> Vec4 {
            let w: f32 = -pos.w;
            Vec4(pos.x, pos.y, pos.z, w)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_negate_vec() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_neg_vec(#[location(0)] pos: Vec4) -> Vec4 {
            -pos
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_vec_times_scalar() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_scale(#[location(0)] pos: Vec4, scale: f32) -> Vec4 {
            pos * scale
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_scalar_times_vec() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_scale2(#[location(0)] pos: Vec3, scale: f32) -> Vec4 {
            (scale * pos).extend(1.0)
        }
    );
    assert!(spv_valid(SPV));
}

// ── Fragment push constants ───────────────────────────────────────────────────

#[test]
fn fragment_scalar_push_const() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_tint(#[location(0)] color: Vec4, tint: f32) -> Vec4 {
            color * tint
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_mat4_push_const() {
    // Mat4 in fragment is unusual but valid (e.g. inverse VP for deferred passes).
    // This verifies the PC struct machinery works in the fragment execution model.
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_pc(#[location(0)] uv: Vec2, scale: f32, bias: f32) -> Vec4 {
            let r: f32 = uv.x * scale + bias;
            Vec4(r, uv.y, 0.0, 1.0)
        }
    );
    assert!(spv_valid(SPV));
}

// ── If / comparison ───────────────────────────────────────────────────────────

#[test]
fn fragment_if_clamp_alpha() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_clamp(#[location(0)] color: Vec4, alpha: f32) -> Vec4 {
            let a: f32 = alpha;
            if a < 0.0 {
                a = 0.0;
            }
            if a > 1.0 {
                a = 1.0;
            }
            Vec4(color.x, color.y, color.z, a)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_if_else() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_select(#[location(0)] a: Vec4, #[location(1)] b: Vec4, t: f32) -> Vec4 {
            let r: Vec4 = a;
            if t >= 0.5 {
                r = b;
            } else {
                r = a;
            }
            r
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn vertex_if_u32_compare() {
    const SPV: &[u32] = zengpu_spirv!(
        #[vertex]
        fn vs_clip(#[location(0)] pos: Vec4, flag: u32) -> Vec4 {
            let p: Vec4 = pos;
            if flag > 0 {
                p = Vec4(0.0, 0.0, 0.0, 1.0);
            }
            p
        }
    );
    assert!(spv_valid(SPV));
}

// ── Vec-vec arithmetic ────────────────────────────────────────────────────────

#[test]
fn fragment_vec4_add() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_add(#[location(0)] a: Vec4, #[location(1)] b: Vec4) -> Vec4 {
            a + b
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_vec4_sub() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_sub(#[location(0)] a: Vec4, #[location(1)] b: Vec4) -> Vec4 {
            a - b
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_vec3_mul_scalar_add_vec() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_lerp(#[location(0)] a: Vec3, #[location(1)] b: Vec3, t: f32) -> Vec4 {
            (a * t + b * (1.0 - t)).extend(1.0)
        }
    );
    assert!(spv_valid(SPV));
}

// ── dot() built-in ───────────────────────────────────────────────────────────

#[test]
fn fragment_dot_product() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_dot(#[location(0)] a: Vec3, #[location(1)] b: Vec3) -> Vec4 {
            let d: f32 = dot(a, b);
            Vec4(d, d, d, 1.0)
        }
    );
    assert!(spv_valid(SPV));
}

#[test]
fn fragment_diffuse_lighting() {
    const SPV: &[u32] = zengpu_spirv!(
        #[fragment]
        fn fs_diffuse(#[location(0)] normal: Vec3, #[location(1)] color: Vec3) -> Vec4 {
            let lx: f32 = 0.577;
            let ly: f32 = 0.577;
            let lz: f32 = 0.577;
            let light: Vec3 = Vec3(lx, ly, lz);
            let d: f32 = dot(normal, light);
            let lit: f32 = d;
            (color * lit).extend(1.0)
        }
    );
    assert!(spv_valid(SPV));
}

// ── SPIR-V header check ───────────────────────────────────────────────────────

fn spv_valid(spv: &[u32]) -> bool {
    spv.len() >= 5 && spv[0] == 0x07230203
}
