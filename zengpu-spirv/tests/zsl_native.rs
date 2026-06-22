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
