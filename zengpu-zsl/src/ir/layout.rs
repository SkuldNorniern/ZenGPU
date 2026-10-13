//! Host-visible push-constant layout metadata for a compiled entry point.
//!
//! `backend::spirv::compute`/`backend::spirv::graphics_ir` each independently
//! compute a push-constant-block byte layout when lowering — buffer params
//! first (each a `u32` bindless-table slot index), then scalar params, tightly
//! packed. `zengpu-vulkan/src/device/dispatch.rs`'s dispatch path assumes this
//! exact same layout when packing a caller's `Bindings::{buffers,scalars}`
//! into push-constant bytes (buffer indices then scalars, 4 bytes each) — but
//! today that assumption is implicit and unchecked on both sides. This module
//! makes the layout an explicit, computable, comparable value so a caller
//! holding a `Module`/`GraphicsModule` can inspect or validate against it
//! instead of relying on the two sides staying in sync by convention.

use crate::ir::arena::TypeArena;
use crate::ir::types::TypeId;
use crate::ir::{Entry, GraphicsEntry, GraphicsModule, Module, ParamKind};

/// One push-constant-block member's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutInfo {
    /// Byte offset within the push-constant block.
    pub offset: u32,
    /// Size in bytes (4 for a bindless buffer-slot index or scalar; 64 for a
    /// graphics `mat4x4<f32>` push-constant member).
    pub size: u32,
    pub kind: LayoutKind,
}

/// What a push-constant member actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    /// A `device buffer<elem>` param. The push-constant member itself always
    /// holds a bindless-table `u32` slot index, not `elem` — `elem` is
    /// carried here so a caller can still see what the buffer holds.
    BufferIndex { elem: TypeId },
    /// A `texture` param: a `u32` slot in the sampled-texture table.
    TextureIndex,
    /// An `image` param: a `u32` slot in the storage-image table.
    ImageIndex,
    /// A push-constant scalar (or, for graphics, `mat4x4<f32>`) param, stored
    /// as its own declared type.
    Scalar { ty: TypeId },
}

/// Compute a compute entry point's push-constant layout: buffer-index members
/// first, then texture-index members, then image-index members, then scalar
/// members (each group in param declaration order), each exactly 4 bytes,
/// tightly packed — matching
/// `backend::spirv::compute`'s `pc_members`/`member_decorate` construction.
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

/// Compute a graphics entry point's push-constant layout: buffer-index
/// members first, then scalar members — `mat4x4<f32>` members are 16-byte
/// aligned and 64 bytes wide (column-major, matrix stride 16), matching
/// `backend::spirv::graphics_ir`'s `pc_members`/`member_decorate`
/// construction exactly. Graphics buffers are always `f32` (grammar-
/// restricted to `buffer<f32>` — the parser rejects any other element type),
/// so `elem` is always the module's `f32` `TypeId`.
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

/// The alignment and size of a graphics push-constant member, by the std430
/// rules Vulkan applies to push-constant blocks.
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
        // Mirrors `zengpu-zsl/src/lib.rs`'s `TYPED_BUFFERS` kernel: two buffer
        // params (u32 index slots, offsets 0 and 4) and no scalar params —
        // cross-checked against `backend::spirv::compute`'s own
        // `member_decorate(..., OFFSET, [(i * 4)...])` construction, which
        // assigns offsets in exactly this declaration order.
        let src = r#"
            @workgroup_size(64)
            kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, id: global_id) {
                out_u[id.x] = src_u[id.x]
                out_i[id.x] = src_i[id.x]
            }
        "#;
        let module = compile(src);
        let layout = compute_layout(&module);
        let offsets: Vec<(String, u32)> =
            layout.iter().map(|(n, l)| (n.clone(), l.offset)).collect();
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
        let offsets: Vec<(String, u32)> =
            layout.iter().map(|(n, l)| (n.clone(), l.offset)).collect();
        // Buffers (x, y) get the first two u32 slots; push-struct scalars
        // (n, alpha) follow, in declaration order.
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
        // Mirrors `backend::spirv::graphics_ir`'s own offset/alignment
        // construction: a leading u32 scalar forces the following mat4 to pad
        // up to the next 16-byte boundary (4 -> 16), then occupies 64 bytes.
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
            vec![("flag".to_string(), 0, 4), ("mvp".to_string(), 16, 64),]
        );
    }
}
