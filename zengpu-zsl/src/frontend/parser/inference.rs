use super::*;

pub(super) fn builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "u32" => BuiltinFn::U32,
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
        "tanh" => BuiltinFn::Tanh,
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        "tan" => BuiltinFn::Tan,
        "isnan" => BuiltinFn::IsNan,
        "isinf" => BuiltinFn::IsInf,
        "isfinite" => BuiltinFn::IsFinite,
        "log" => BuiltinFn::Log,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        "mul_add_f32" => BuiltinFn::MulAddF32,
        "subgroup_id" => BuiltinFn::SubgroupId,
        "subgroup_local_id" => BuiltinFn::SubgroupLocalId,
        "subgroup_size" => BuiltinFn::SubgroupSize,
        _ => return None,
    })
}

pub(super) fn buffer_elem_name(elem: TypeId, arena: &crate::ir::arena::TypeArena) -> &'static str {
    use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type};
    match arena.get(elem) {
        Type::Float {
            format: FloatFormat::F32,
        } => "f32",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => "u32",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => "i32",
        Type::Float {
            format: FloatFormat::Bf16,
        } => "bf16",
        Type::Float {
            format: FloatFormat::F16,
        } => "f16",
        _ => "unsupported",
    }
}

pub(super) fn gfx_builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
        "tanh" => BuiltinFn::Tanh,
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        "tan" => BuiltinFn::Tan,
        "log" => BuiltinFn::Log,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        "normalize" => BuiltinFn::Normalize,
        "length" => BuiltinFn::Length,
        _ => return None,
    })
}
