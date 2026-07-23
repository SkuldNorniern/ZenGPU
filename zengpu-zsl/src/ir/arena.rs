//! Interns [`Type`] values behind [`TypeId`], and adapts today's three
//! separate type enums (`ScalarTy`, `GfxTy`, `BufElem`) into it.
//!
//! These adapters are inherent `TypeArena` methods rather than plain `impl
//! From<ScalarTy> for Type` — recursive `Type` variants (`Vector`, `Matrix`)
//! reference their element via an interned `TypeId`, so producing one
//! requires access to the arena doing the interning, not just the source
//! enum value. Nothing outside this module's own tests calls these adapters
//! yet; `ScalarTy`/`GfxTy`/`BufElem` and every backend are unaffected.

use std::collections::HashMap;

use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type, TypeId};
use crate::ir::{BufElem, GfxTy, ScalarTy};

/// An interning arena for [`Type`]. Structurally identical types (e.g. two
/// separate requests to intern `f32`) receive the same [`TypeId`].
#[derive(Debug, Default)]
pub struct TypeArena {
    types: Vec<Type>,
    by_value: HashMap<Type, TypeId>,
}

impl TypeArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `ty`, returning its (possibly pre-existing) [`TypeId`].
    pub fn intern(&mut self, ty: Type) -> TypeId {
        if let Some(id) = self.by_value.get(&ty) {
            return *id;
        }
        let id = TypeId(self.types.len() as u32);
        self.by_value.insert(ty.clone(), id);
        self.types.push(ty);
        id
    }

    /// Look up a previously interned type.
    pub fn get(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    /// Lossless conversion from today's compute-only [`ScalarTy`].
    pub fn intern_scalar(&mut self, ty: ScalarTy) -> TypeId {
        let t = match ty {
            ScalarTy::Bool => Type::Bool,
            ScalarTy::U32 => Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned,
            },
            ScalarTy::I32 => Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Signed,
            },
            ScalarTy::F32 => Type::Float {
                format: FloatFormat::F32,
            },
        };
        self.intern(t)
    }

    /// The reverse of [`TypeArena::intern_scalar`]: recover the `ScalarTy` a
    /// previously-interned `TypeId` came from. Used at backend boundaries
    /// that still work in terms of `ScalarTy` (e.g. every compute backend's
    /// `emit_stmt`/`emit_expr`), so `Entry.locals`/`SharedDecl.elem` can hold
    /// `TypeId` without requiring every backend to be rewritten in the same
    /// commit. Panics if `id` does not resolve to a scalar type produced by
    /// `intern_scalar` — every `TypeId` reachable from `Entry.locals` today
    /// is guaranteed to be one, since only `intern_scalar` is used to build
    /// `Entry.locals`/`SharedDecl.elem` in `frontend::parser`.
    pub fn scalar_ty_of(&self, id: TypeId) -> ScalarTy {
        match self.get(id) {
            Type::Bool => ScalarTy::Bool,
            Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned,
            } => ScalarTy::U32,
            Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Signed,
            } => ScalarTy::I32,
            Type::Float {
                format: FloatFormat::F32,
            } => ScalarTy::F32,
            other => panic!("TypeArena::scalar_ty_of: {other:?} is not a compute scalar type"),
        }
    }

    /// Lossless conversion from today's graphics-only [`GfxTy`]. Vector/matrix
    /// variants intern their `f32`/`u32` element first.
    pub fn intern_gfx_ty(&mut self, ty: GfxTy) -> TypeId {
        match ty {
            GfxTy::F32 => self.intern(Type::Float {
                format: FloatFormat::F32,
            }),
            GfxTy::U32 => self.intern(Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned,
            }),
            GfxTy::Vec2 | GfxTy::Vec3 | GfxTy::Vec4 => {
                let element = self.intern(Type::Float {
                    format: FloatFormat::F32,
                });
                let lanes = match ty {
                    GfxTy::Vec2 => 2,
                    GfxTy::Vec3 => 3,
                    GfxTy::Vec4 => 4,
                    _ => unreachable!("matched above"),
                };
                self.intern(Type::Vector { element, lanes })
            }
            GfxTy::Mat4 => {
                let element = self.intern(Type::Float {
                    format: FloatFormat::F32,
                });
                self.intern(Type::Matrix {
                    element,
                    rows: 4,
                    columns: 4,
                    major: crate::ir::types::MatrixMajor::Column,
                })
            }
        }
    }

    /// Lossless conversion from today's [`BufElem`] (the scalar element of a
    /// `device buffer<T>`). Vector/matrix buffer elements intern their `f32`
    /// element first, matching `intern_gfx_ty`.
    pub fn intern_buf_elem(&mut self, elem: BufElem) -> TypeId {
        match elem {
            BufElem::F32 => self.intern(Type::Float {
                format: FloatFormat::F32,
            }),
            BufElem::U32 => self.intern(Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned,
            }),
            BufElem::I32 => self.intern(Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Signed,
            }),
            BufElem::Vec2 | BufElem::Vec3 | BufElem::Vec4 => {
                let element = self.intern(Type::Float {
                    format: FloatFormat::F32,
                });
                let lanes = match elem {
                    BufElem::Vec2 => 2,
                    BufElem::Vec3 => 3,
                    BufElem::Vec4 => 4,
                    _ => unreachable!("matched above"),
                };
                self.intern(Type::Vector { element, lanes })
            }
            BufElem::Mat2 | BufElem::Mat3 | BufElem::Mat4 => {
                let element = self.intern(Type::Float {
                    format: FloatFormat::F32,
                });
                let n = match elem {
                    BufElem::Mat2 => 2,
                    BufElem::Mat3 => 3,
                    BufElem::Mat4 => 4,
                    _ => unreachable!("matched above"),
                };
                self.intern(Type::Matrix {
                    element,
                    rows: n,
                    columns: n,
                    major: crate::ir::types::MatrixMajor::Column,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{FloatFormat, IntWidth, Signedness};

    #[test]
    fn interning_the_same_scalar_type_twice_yields_the_same_id() {
        let mut arena = TypeArena::new();
        let a = arena.intern_scalar(ScalarTy::F32);
        let b = arena.intern_scalar(ScalarTy::F32);
        assert_eq!(a, b);
        assert_eq!(
            *arena.get(a),
            Type::Float {
                format: FloatFormat::F32
            }
        );
    }

    #[test]
    fn distinct_scalar_types_get_distinct_ids() {
        let mut arena = TypeArena::new();
        let f32_id = arena.intern_scalar(ScalarTy::F32);
        let u32_id = arena.intern_scalar(ScalarTy::U32);
        let i32_id = arena.intern_scalar(ScalarTy::I32);
        let bool_id = arena.intern_scalar(ScalarTy::Bool);
        assert_ne!(f32_id, u32_id);
        assert_ne!(u32_id, i32_id);
        assert_ne!(i32_id, bool_id);
        assert_eq!(
            *arena.get(u32_id),
            Type::Int {
                width: IntWidth::W32,
                signedness: Signedness::Unsigned
            }
        );
    }

    #[test]
    fn gfx_vector_interns_its_element_and_shares_it_with_the_scalar() {
        let mut arena = TypeArena::new();
        let f32_scalar = arena.intern_scalar(ScalarTy::F32);
        let vec4 = arena.intern_gfx_ty(GfxTy::Vec4);
        match arena.get(vec4) {
            Type::Vector { element, lanes } => {
                assert_eq!(*lanes, 4);
                // The f32 element inside Vec4 is the SAME interned id as the
                // plain f32 scalar — proving dedup works across adapters.
                assert_eq!(*element, f32_scalar);
            }
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn gfx_mat4_interns_a_column_major_4x4_f32_matrix() {
        let mut arena = TypeArena::new();
        let mat4 = arena.intern_gfx_ty(GfxTy::Mat4);
        match arena.get(mat4) {
            Type::Matrix {
                rows,
                columns,
                major,
                ..
            } => {
                assert_eq!(*rows, 4);
                assert_eq!(*columns, 4);
                assert_eq!(*major, crate::ir::types::MatrixMajor::Column);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    #[test]
    fn buf_elem_scalars_match_the_equivalent_scalar_ty_interning() {
        let mut arena = TypeArena::new();
        let via_scalar = arena.intern_scalar(ScalarTy::F32);
        let via_buf_elem = arena.intern_buf_elem(BufElem::F32);
        assert_eq!(via_scalar, via_buf_elem);
    }

    #[test]
    fn buf_elem_mat3_interns_a_3x3_matrix() {
        let mut arena = TypeArena::new();
        let mat3 = arena.intern_buf_elem(BufElem::Mat3);
        match arena.get(mat3) {
            Type::Matrix { rows, columns, .. } => {
                assert_eq!(*rows, 3);
                assert_eq!(*columns, 3);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    #[test]
    fn scalar_ty_of_round_trips_every_scalar_ty() {
        let mut arena = TypeArena::new();
        for ty in [ScalarTy::Bool, ScalarTy::U32, ScalarTy::I32, ScalarTy::F32] {
            let id = arena.intern_scalar(ty);
            assert_eq!(arena.scalar_ty_of(id), ty);
        }
    }
}
