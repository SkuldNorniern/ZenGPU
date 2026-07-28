
use std::collections::HashMap;

use crate::ir::types::{FloatFormat, IntWidth, MatrixMajor, Signedness, Type, TypeId};

/// structurally equal types share one [`TypeId`].
#[derive(Debug, Default, Clone)]
pub struct TypeArena {
    types: Vec<Type>,
    by_value: HashMap<Type, TypeId>,
}

impl TypeArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, ty: Type) -> TypeId {
        if let Some(id) = self.by_value.get(&ty) {
            return *id;
        }
        let id = TypeId(self.types.len() as u32);
        self.by_value.insert(ty.clone(), id);
        self.types.push(ty);
        id
    }

    pub fn get(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    pub fn u32_id(&mut self) -> TypeId {
        self.intern(Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        })
    }

    pub fn i32_id(&mut self) -> TypeId {
        self.intern(Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        })
    }

    pub fn f32_id(&mut self) -> TypeId {
        self.intern(Type::Float {
            format: FloatFormat::F32,
        })
    }

    pub fn bool_id(&mut self) -> TypeId {
        self.intern(Type::Bool)
    }

    /// Storage-only: no expression has type bf16, and loads narrow to `f32`.
    pub fn bf16_id(&mut self) -> TypeId {
        self.intern(Type::Float {
            format: FloatFormat::Bf16,
        })
    }

    pub fn vec_id(&mut self, lanes: u8) -> TypeId {
        let element = self.f32_id();
        self.intern(Type::Vector { element, lanes })
    }

    /// Column-major storage.
    pub fn mat4_id(&mut self) -> TypeId {
        let element = self.f32_id();
        self.intern(Type::Matrix {
            element,
            rows: 4,
            columns: 4,
            major: MatrixMajor::Column,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{FloatFormat, IntWidth, Signedness};

    #[test]
    fn interning_the_same_scalar_type_twice_yields_the_same_id() {
        let mut arena = TypeArena::new();
        let a = arena.f32_id();
        let b = arena.f32_id();
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
        let f32_id = arena.f32_id();
        let u32_id = arena.u32_id();
        let i32_id = arena.i32_id();
        let bool_id = arena.bool_id();
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
    fn bf16_is_distinct_from_f32_and_idempotent() {
        let mut arena = TypeArena::new();
        let bf16_a = arena.bf16_id();
        let bf16_b = arena.bf16_id();
        let f32_id = arena.f32_id();
        assert_eq!(bf16_a, bf16_b);
        assert_ne!(bf16_a, f32_id);
        assert_eq!(
            *arena.get(bf16_a),
            Type::Float {
                format: FloatFormat::Bf16
            }
        );
    }

    #[test]
    fn vector_interns_its_element_and_shares_it_with_the_scalar() {
        let mut arena = TypeArena::new();
        let f32_scalar = arena.f32_id();
        let vec4 = arena.vec_id(4);
        match arena.get(vec4) {
            Type::Vector { element, lanes } => {
                assert_eq!(*lanes, 4);
                assert_eq!(*element, f32_scalar);
            }
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn mat4_interns_a_column_major_4x4_f32_matrix() {
        let mut arena = TypeArena::new();
        let mat4 = arena.mat4_id();
        match arena.get(mat4) {
            Type::Matrix {
                rows,
                columns,
                major,
                ..
            } => {
                assert_eq!(*rows, 4);
                assert_eq!(*columns, 4);
                assert_eq!(*major, MatrixMajor::Column);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }
}
