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

// ── SPIR-V header check ───────────────────────────────────────────────────────

fn spv_valid(spv: &[u32]) -> bool {
    spv.len() >= 5 && spv[0] == 0x07230203
}
