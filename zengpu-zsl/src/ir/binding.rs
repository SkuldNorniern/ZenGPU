//! Bindless SSBO table assignment for buffer params — which of
//! `zengpu-vulkan`'s typed tables (`SSBO_BINDING_F32`/`SSBO_BINDING_U32`/
//! `SSBO_BINDING_I32` in `zengpu-vulkan/src/device/mod.rs`) a buffer element
//! routes to.
//!
//! `backend::spirv::compute` computes this exact mapping itself (its own
//! `ssbo_binding_for`) when declaring each kernel's typed SSBO tables; this
//! module is the shared, authoritative version — the backend calls
//! [`ssbo_binding_for`] here instead of keeping a private copy.

use crate::ir::arena::TypeArena;
use crate::ir::types::TypeId;
use crate::ir::{GraphicsEntry, GraphicsModule, Module, ParamKind};

/// Which bindless SSBO table a buffer param routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingInfo {
    /// The `set=0` binding index in `zengpu-vulkan`'s bindless descriptor set
    /// layout (`SSBO_BINDING_F32`=0, `SSBO_BINDING_U32`=2,
    /// `SSBO_BINDING_I32`=3).
    pub binding: u32,
}

/// The bindless SSBO binding index for a buffer element's `TypeId`. Matches
/// `zengpu-vulkan/src/device/mod.rs`'s `SSBO_BINDING_F32`/`SSBO_BINDING_U32`/
/// `SSBO_BINDING_I32` constants exactly (f32→0, u32→2, i32→3) — kept in sync
/// by convention (not a shared crate constant across `zengpu-zsl` and
/// `zengpu-vulkan` — there are only three, and they rarely change). `f16` and
/// `bf16` buffers are read through the u32 table, two halves per word.
pub fn ssbo_binding_for(elem: TypeId, types: &mut TypeArena) -> u32 {
    if elem == types.f32_id() {
        0
    } else if elem == types.u32_id() || elem == types.f16_id() || elem == types.bf16_id() {
        2
    } else if elem == types.i32_id() {
        3
    } else {
        unreachable!("unsupported compute buffer element")
    }
}

/// The sampled-texture table's binding (`SSBO_BINDING_TEXTURES` in
/// `zengpu-vulkan`).
pub const TEXTURE_BINDING: u32 = 1;
/// The storage-image table's binding (`STORAGE_IMAGE_BINDING` in
/// `zengpu-vulkan`).
pub const IMAGE_BINDING: u32 = 4;

/// Compute each compute buffer, texture and image param's bindless table
/// assignment, in declaration order.
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

/// Graphics buffers are always `f32` (grammar-restricted to `buffer<f32>` —
/// the parser rejects any other element type), so every graphics buffer
/// param routes to the `f32` table (`binding=0`).
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
        // Mirrors `zengpu-zsl/src/lib.rs`'s `TYPED_BUFFERS` kernel.
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
        // Not a graphics kernel, but exercises the same shared computation
        // path via compute_bindings for a plain f32 buffer.
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
