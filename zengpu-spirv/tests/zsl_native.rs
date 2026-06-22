//! Native ZSL syntax through the `zsl!` macro (no syn/quote in the pipeline).

use zengpu_spirv::zsl;

const SPIRV_MAGIC: u32 = 0x0723_0203;

#[test]
fn native_saxpy_compiles_to_spirv() {
    const SPV: &[u32] = zsl!(
        push SaxpyPush { n: u32, alpha: f32 }
        @workgroup_size(256)
        kernel saxpy(
            push: SaxpyPush,
            x: device buffer<f32>,
            y: device mut buffer<f32>,
            id: global_id,
        ) {
            let i = id.x
            if i < push.n {
                y[i] = push.alpha * x[i] + y[i]
            }
        }
    );
    assert!(!SPV.is_empty());
    assert_eq!(SPV[0], SPIRV_MAGIC, "valid SPIR-V header");
}

#[test]
fn native_vertex_mvp() {
    const SPV: &[u32] = zsl!(
        push Mvp { mvp: mat4x4<f32> }
        vertex vs(@location(0) pos: f32x3, m: Mvp) -> f32x4 {
            m.mvp * pos.extend(1.0)
        }
    );
    assert_eq!(SPV[0], SPIRV_MAGIC);
}

#[test]
fn native_vertex_with_varying() {
    const SPV: &[u32] = zsl!(
        vertex vs(@location(0) pos: f32x4, @location(1) uv: f32x2) -> (f32x4, f32x2) {
            (pos, uv)
        }
    );
    assert_eq!(SPV[0], SPIRV_MAGIC);
}

#[test]
fn native_fragment_color() {
    const SPV: &[u32] = zsl!(
        fragment fs(@location(0) color: f32x4) -> f32x4 {
            color * 0.5
        }
    );
    assert_eq!(SPV[0], SPIRV_MAGIC);
}

#[test]
fn native_copy_with_module_and_workgroup() {
    const SPV: &[u32] = zsl!(
        module zengpu.examples.copy
        @workgroup_size(64)
        kernel copy(
            src: device buffer<f32>,
            dst: device mut buffer<f32>,
            id: global_id,
        ) {
            let i = id.x
            dst[i] = src[i]
        }
    );
    assert_eq!(SPV[0], SPIRV_MAGIC);
}
