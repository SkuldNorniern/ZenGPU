//! Host-visible push-constant layout for a compiled entry point.
//!
//! Buffer slots first (u32 each), then scalars, tightly packed. `zengpu-vulkan`'s dispatch packs
//! `Bindings` in this order. The layout is spelled out here so callers can check it.

use crate::ir::arena::TypeArena;
use crate::ir::types::TypeId;
use crate::ir::{Entry, GraphicsEntry, GraphicsModule, Module, ParamKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutInfo {
    pub offset: u32,
    /// Size in bytes: 4 for a slot index or scalar, 64 for a graphics `mat4x4<f32>`.
    pub size: u32,
    pub kind: LayoutKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    /// A `device buffer<elem>` param. The member is a `u32` slot index, and
    /// `elem` is informational.
    BufferIndex { elem: TypeId },
    /// Slot in the sampled-texture table, as a `u32`.
    TextureIndex,
    /// Slot in the storage-image table, as a `u32`.
    ImageIndex,
    Scalar { ty: TypeId },
}

/// Members come in this order: buffer slots, texture slots, image slots, then scalars.
///
/// Each group keeps declaration order. Every member is 4 bytes and tightly packed, as
/// in `backend::spirv::compute`.
pub fn compute_layout(module: &Module) -> Vec<(String, LayoutInfo)> {
    entry_layout(&module.entry)
}

fn entry_layout(entry: &Entry) -> Vec<(String, LayoutInfo)> {
    let mut out = Vec::new();
    let mut offset = 0u32;
    for p in &entry.params {
        if let ParamKind::Buffer { elem, .. } = &p.kind {
            out.push((
                p.name.clone(),
                LayoutInfo {
                    offset,
                    size: 4,
                    kind: LayoutKind::BufferIndex { elem: *elem },
                },
            ));
            offset += 4;
        }
    }
    for (image, kind) in [
        (false, LayoutKind::TextureIndex),
        (true, LayoutKind::ImageIndex),
    ] {
        for p in &entry.params {
            let wanted = match p.kind {
                ParamKind::Texture => !image,
                ParamKind::Image => image,
                _ => false,
            };
            if wanted {
                out.push((
                    p.name.clone(),
                    LayoutInfo {
                        offset,
                        size: 4,
                        kind,
                    },
                ));
                offset += 4;
            }
        }
    }
    for p in &entry.params {
        if let ParamKind::Scalar(ty) = &p.kind {
            out.push((
                p.name.clone(),
                LayoutInfo {
                    offset,
                    size: 4,
                    kind: LayoutKind::Scalar { ty: *ty },
                },
            ));
            offset += 4;
        }
    }
    out
}

/// Graphics layout: buffer slots, then scalars. `mat4x4<f32>` is 16-byte aligned, 64
/// bytes, column-major.
///
/// Graphics buffers are always `f32`, so `elem` is the f32 `TypeId`.
pub fn graphics_layout(module: &GraphicsModule) -> Vec<(String, LayoutInfo)> {
    let e: &GraphicsEntry = &module.entry;
    let mut types = module.types.clone();
    let f32_id = types.f32_id();

    let mut out = Vec::new();
    let mut offset = 0u32;
    for name in &e.buf_params {
        out.push((
            name.clone(),
            LayoutInfo {
                offset,
                size: 4,
                kind: LayoutKind::BufferIndex { elem: f32_id },
            },
        ));
        offset += 4;
    }
    for s in &e.scalar_params {
        let (align, size) = graphics_member(s.ty, &mut types);
        offset = offset.next_multiple_of(align);
        out.push((
            s.name.clone(),
            LayoutInfo {
                offset,
                size,
                kind: LayoutKind::Scalar { ty: s.ty },
            },
        ));
        offset += size;
    }
    out
}

/// std430 alignment and size of a graphics push-constant member.
pub fn graphics_member(ty: TypeId, types: &mut TypeArena) -> (u32, u32) {
    if ty == types.mat4_id() {
        (16, 64)
    } else if ty == types.vec_id(4) {
        (16, 16)
    } else if ty == types.vec_id(2) {
        (8, 8)
    } else {
        (4, 4)
    }
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
    fn typed_buffers_layout_matches_spirv_backend_offsets() {
        let src = r#"
            @workgroup_size(64)
            kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, id: global_id) {
                out_u[id.x] = src_u[id.x]
                out_i[id.x] = src_i[id.x]
            }
        "#;
        let module = compile(src);
        let layout = compute_layout(&module);
        let offsets: Vec<(String, u32)> = layout.iter().map(|(n, l)| (n.clone(), l.offset)).collect();
        assert_eq!(
            offsets,
            vec![
                ("src_u".to_string(), 0),
                ("src_i".to_string(), 4),
                ("out_u".to_string(), 8),
                ("out_i".to_string(), 12),
            ]
        );
        for (_, info) in &layout {
            assert_eq!(info.size, 4);
            assert!(matches!(info.kind, LayoutKind::BufferIndex { .. }));
        }
    }

    #[test]
    fn scalar_params_follow_buffer_indices() {
        let src = r#"
            push P { n: u32, alpha: f32 }
            @workgroup_size(64)
            kernel saxpy(push: P, x: device buffer<f32>, y: device mut buffer<f32>, id: global_id) {
                let i = id.x
                if i < push.n {
                    y[i] = push.alpha * x[i] + y[i]
                }
            }
        "#;
        let module = compile(src);
        let layout = compute_layout(&module);
        let offsets: Vec<(String, u32)> = layout.iter().map(|(n, l)| (n.clone(), l.offset)).collect();
        assert_eq!(
            offsets,
            vec![
                ("x".to_string(), 0),
                ("y".to_string(), 4),
                ("n".to_string(), 8),
                ("alpha".to_string(), 12),
            ]
        );
        assert!(matches!(layout[2].1.kind, LayoutKind::Scalar { .. }));
    }

    #[test]
    fn graphics_mat4_push_constant_is_16_byte_aligned() {
        let src = r#"
            push P { flag: u32, mvp: mat4x4<f32> }
            vertex vs(@location(0) pos: f32x3, p: P) -> f32x4 {
                p.mvp * pos.extend(1.0)
            }
        "#;
        let module = compile_graphics(src);
        let layout = graphics_layout(&module);
        let offsets: Vec<(String, u32, u32)> = layout
            .iter()
            .map(|(n, l)| (n.clone(), l.offset, l.size))
            .collect();
        assert_eq!(
            offsets,
            vec![
                ("flag".to_string(), 0, 4),
                ("mvp".to_string(), 16, 64),
            ]
        );
    }
}
