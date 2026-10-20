//! Bindless SSBO table per buffer param. Tables are `SSBO_BINDING_*` in
//! `zengpu-vulkan/src/device/mod.rs`. Shared with `backend::spirv::compute`.

use crate::ir::arena::TypeArena;
use crate::ir::typeck::{i8_type, u8_type, u32_vec_lanes};
use crate::ir::types::TypeId;
use crate::ir::{GraphicsEntry, GraphicsModule, Module, ParamKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingInfo {
    /// `set=0` binding in `zengpu-vulkan`'s bindless layout: f32=0, u32=2, i32=3.
    pub binding: u32,
}

/// Binding numbers match `zengpu-vulkan` by convention, with no shared crate constant.
///
/// `f16` and `bf16` buffers are read through the u32 table, two halves per word.
pub fn ssbo_binding_for(elem: TypeId, types: &mut TypeArena) -> u32 {
    if elem == types.f32_id() {
        0
    } else if elem == types.u32_id()
        || elem == types.f16_id()
        || elem == types.bf16_id()
        || elem == types.intern(i8_type())
        || elem == types.intern(u8_type())
    {
        2
    } else if elem == types.i32_id() {
        3
    } else if u32_vec_lanes(types, elem).is_some() {
        2
    } else {
        unreachable!("unsupported compute buffer element")
    }
}

pub const TEXTURE_BINDING: u32 = 1;
pub const IMAGE_BINDING: u32 = 4;

/// Slots are assigned in param declaration order.
pub fn compute_bindings(module: &Module) -> Vec<(String, BindingInfo)> {
    let mut types = module.types.clone();
    module
        .entry
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some((
                p.name.clone(),
                BindingInfo {
                    binding: ssbo_binding_for(*elem, &mut types),
                },
            )),
            ParamKind::Texture => Some((
                p.name.clone(),
                BindingInfo {
                    binding: TEXTURE_BINDING,
                },
            )),
            ParamKind::Image => Some((
                p.name.clone(),
                BindingInfo {
                    binding: IMAGE_BINDING,
                },
            )),
            ParamKind::Scalar(_) => None,
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
    fn untyped_buffers_route_to_binding_zero() {
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
