//! Bindless SSBO table per buffer param. Tables are `SSBO_BINDING_*` in
//! `zengpu-vulkan/src/device/mod.rs`. Shared with `backend::spirv::compute`.

use crate::ir::types::TypeId;
use crate::ir::{GraphicsEntry, GraphicsModule, Module, ParamKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingInfo {
    /// `set=0` binding in `zengpu-vulkan`'s bindless layout: f32=0, u32=2, i32=3.
    pub binding: u32,
}

/// Bindless SSBO binding index for a buffer element's `TypeId`. Must match `SSBO_BINDING_F32`,
/// `SSBO_BINDING_U32` and `SSBO_BINDING_I32` in `zengpu-vulkan/src/device/mod.rs` (f32 to 0, u32 to
/// 2, i32 to 3). Kept in sync by hand.
pub fn ssbo_binding_for(elem: TypeId, f32_id: TypeId, u32_id: TypeId, i32_id: TypeId) -> u32 {
    if elem == f32_id {
        0
    } else if elem == u32_id {
        2
    } else if elem == i32_id {
        3
    } else {
        unreachable!("unsupported compute buffer element")
    }
}

/// Bindless table assignment for each buffer param, in declaration order.
pub fn compute_bindings(module: &Module) -> Vec<(String, BindingInfo)> {
    let mut types = module.types.clone();
    let f32_id = types.f32_id();
    let u32_id = types.u32_id();
    let i32_id = types.i32_id();
    module
        .entry
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some((
                p.name.clone(),
                BindingInfo {
                    binding: ssbo_binding_for(*elem, f32_id, u32_id, i32_id),
                },
            )),
            _ => None,
        })
        .collect()
}

/// Graphics buffers are always `f32` (the parser rejects other element types). They use the f32
/// table (`binding=0`).
pub fn graphics_bindings(module: &GraphicsModule) -> Vec<(String, BindingInfo)> {
    let e: &GraphicsEntry = &module.entry;
    e.buf_params
        .iter()
        .map(|name| (name.clone(), BindingInfo { binding: 0 }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::{Shader, parse_zsl};

    fn compile(src: &str) -> Module {
        match parse_zsl(src).expect("parse") {
            Shader::Compute(m) => m,
            Shader::Graphics(_) => panic!("expected compute"),
        }
    }

    fn compile_graphics(src: &str) -> GraphicsModule {
        match parse_zsl(src).expect("parse") {
            Shader::Graphics(m) => m,
            Shader::Compute(_) => panic!("expected graphics"),
        }
    }

    #[test]
    fn typed_buffers_route_to_the_matching_table() {
        let src = r#"
            @workgroup_size(64)
            kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, id: global_id) {
                out_u[id.x] = src_u[id.x]
                out_i[id.x] = src_i[id.x]
            }
        "#;
        let module = compile(src);
        let bindings = compute_bindings(&module);
        let table: Vec<(String, u32)> = bindings
            .iter()
            .map(|(n, b)| (n.clone(), b.binding))
            .collect();
        assert_eq!(
            table,
            vec![
                ("src_u".to_string(), 2),
                ("src_i".to_string(), 3),
                ("out_u".to_string(), 2),
                ("out_i".to_string(), 3),
            ]
        );
    }

    #[test]
    fn f32_buffers_route_to_binding_zero() {
        let src = r#"
            @workgroup_size(64)
            kernel copy(src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
                dst[id.x] = src[id.x]
            }
        "#;
        let module = compile(src);
        let bindings = compute_bindings(&module);
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|(_, b)| b.binding == 0));
    }

    #[test]
    fn graphics_buffers_route_to_binding_zero() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, id: global_id) {
                dst[id.x] = 1.0
            }
        "#;
        let module = compile(src);
        assert_eq!(compute_bindings(&module)[0].1.binding, 0);

        let gsrc = r#"
            vertex vs(@location(0) pos: f32x4, buf: device buffer<f32>) -> f32x4 {
                pos
            }
        "#;
        let gmodule = compile_graphics(gsrc);
        let gbindings = graphics_bindings(&gmodule);
        assert_eq!(gbindings.len(), 1);
        assert_eq!(gbindings[0].1.binding, 0);
    }
}
