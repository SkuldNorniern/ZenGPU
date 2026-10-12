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
        "subgroup_any" => BuiltinFn::SubgroupAny,
        "subgroup_all" => BuiltinFn::SubgroupAll,
        "subgroup_ballot" => BuiltinFn::SubgroupBallot,
        "subgroup_broadcast" => BuiltinFn::SubgroupBroadcast,
        "subgroup_shuffle" => BuiltinFn::SubgroupShuffle,
        "subgroup_shuffle_xor" => BuiltinFn::SubgroupShuffleXor,
        "subgroup_shuffle_up" => BuiltinFn::SubgroupShuffleUp,
        "subgroup_shuffle_down" => BuiltinFn::SubgroupShuffleDown,
        "subgroup_reduce_add" => BuiltinFn::SubgroupReduceAdd,
        "subgroup_reduce_min" => BuiltinFn::SubgroupReduceMin,
        "subgroup_reduce_max" => BuiltinFn::SubgroupReduceMax,
        "subgroup_reduce_and" => BuiltinFn::SubgroupReduceAnd,
        "subgroup_reduce_or" => BuiltinFn::SubgroupReduceOr,
        "subgroup_scan_inclusive_add" => BuiltinFn::SubgroupScanInclusiveAdd,
        "subgroup_scan_exclusive_add" => BuiltinFn::SubgroupScanExclusiveAdd,
        "workgroup_reduce_add" => BuiltinFn::WorkgroupReduceAdd,
        "workgroup_reduce_min" => BuiltinFn::WorkgroupReduceMin,
        "workgroup_reduce_max" => BuiltinFn::WorkgroupReduceMax,
        _ => return None,
    })
}

pub(super) fn buffer_elem_name(elem: TypeId, arena: &TypeArena) -> &'static str {
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
        "u32" => BuiltinFn::U32,
        "f32" => BuiltinFn::F32,
        "step" => BuiltinFn::Step,
        "smoothstep" => BuiltinFn::Smoothstep,
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
