//! Conservative subgroup-uniformity validation: collectives need uniform control flow.
//!
//! Uniformity is tracked only as far as the one-workgroup-per-row kernels need. Buffer and shared
//! loads count as divergent. There is no dataflow or alias analysis.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ir::Entry;
use crate::ir::node::{BuiltinFn, IrExpr, IrStmt};

fn expr_is_uniform(expr: &IrExpr, locals: &HashMap<String, bool>) -> bool {
    match expr {
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_) => true,
        IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::BufferLoad { .. }
        | IrExpr::SharedLoad { .. } => false,
        IrExpr::Local(name) => locals.get(name).copied().unwrap_or(false),
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } | IrExpr::Neg(base) => {
            expr_is_uniform(base, locals)
        }
        IrExpr::Binary { lhs, rhs, .. } => {
            expr_is_uniform(lhs, locals) && expr_is_uniform(rhs, locals)
        }
        IrExpr::Construct { args, .. } | IrExpr::Builtin { args, .. } => {
            args.iter().all(|arg| expr_is_uniform(arg, locals))
        }
        IrExpr::Sample { .. } => false,
        IrExpr::ArrayLit(elems) => elems.iter().all(|e| expr_is_uniform(e, locals)),
        IrExpr::Index { array, index } => {
            locals.get(array).copied().unwrap_or(false) && expr_is_uniform(index, locals)
        }
    }
}

/// workgroup reductions need the same uniform flow, so this predicate matches them too.
fn expr_contains_subgroup_collective(expr: &IrExpr) -> bool {
    match expr {
        IrExpr::Builtin { func, args } => {
            matches!(
                func,
                BuiltinFn::SubgroupAny
                    | BuiltinFn::SubgroupAll
                    | BuiltinFn::SubgroupBallot
                    | BuiltinFn::SubgroupBroadcast
                    | BuiltinFn::SubgroupShuffle
                    | BuiltinFn::SubgroupShuffleXor
                    | BuiltinFn::SubgroupShuffleUp
                    | BuiltinFn::SubgroupShuffleDown
                    | BuiltinFn::SubgroupReduceAdd
                    | BuiltinFn::SubgroupReduceMin
                    | BuiltinFn::SubgroupReduceMax
                    | BuiltinFn::SubgroupReduceAnd
                    | BuiltinFn::SubgroupReduceOr
                    | BuiltinFn::SubgroupScanInclusiveAdd
                    | BuiltinFn::SubgroupScanExclusiveAdd
                    | BuiltinFn::WorkgroupReduceAdd
                    | BuiltinFn::WorkgroupReduceMin
                    | BuiltinFn::WorkgroupReduceMax
            ) || args.iter().any(expr_contains_subgroup_collective)
        }
        IrExpr::BufferLoad { index, .. } | IrExpr::SharedLoad { index, .. } => {
            expr_contains_subgroup_collective(index)
        }
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } => {
            expr_contains_subgroup_collective(base)
        }
        IrExpr::Construct { args, .. } | IrExpr::ArrayLit(args) => {
            args.iter().any(expr_contains_subgroup_collective)
        }
        IrExpr::Index { index, .. } => expr_contains_subgroup_collective(index),
        IrExpr::Sample { index, uv, lod } => {
            expr_contains_subgroup_collective(index)
                || expr_contains_subgroup_collective(uv)
                || lod
                    .as_deref()
                    .is_some_and(expr_contains_subgroup_collective)
        }
        IrExpr::Neg(inner) => expr_contains_subgroup_collective(inner),
        IrExpr::Binary { lhs, rhs, .. } => {
            expr_contains_subgroup_collective(lhs) || expr_contains_subgroup_collective(rhs)
        }
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::Local(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_) => false,
    }
}

pub fn check_compute_uniformity(entry: &Entry) -> Result<(), String> {
    let mut locals = HashMap::new();
    check_stmts(&entry.body, &mut locals, false)
}

fn check_stmts(
    stmts: &[IrStmt],
    locals: &mut HashMap<String, bool>,
    divergent: bool,
) -> Result<(), String> {
    for stmt in stmts {
        check_stmt(stmt, locals, divergent)?;
    }
    Ok(())
}

fn reject_if_divergent(expr: &IrExpr, divergent: bool) -> Result<(), String> {
    if divergent && expr_contains_subgroup_collective(expr) {
        return Err(
            "ZSL: subgroup collectives must occur in subgroup-uniform control flow".to_string(),
        );
    }
    Ok(())
}

fn check_stmt(
    stmt: &IrStmt,
    locals: &mut HashMap<String, bool>,
    divergent: bool,
) -> Result<(), String> {
    match stmt {
        IrStmt::Let { name, init } => {
            reject_if_divergent(init, divergent)?;
            let uniform = expr_is_uniform(init, locals);
            locals.insert(name.clone(), uniform);
            Ok(())
        }
        IrStmt::AssignLocal { name, value } => {
            reject_if_divergent(value, divergent)?;
            let uniform = expr_is_uniform(value, locals);
            locals.insert(name.clone(), uniform);
            Ok(())
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => {
            reject_if_divergent(index, divergent)?;
            reject_if_divergent(value, divergent)
        }
        IrStmt::Barrier | IrStmt::Discard => Ok(()),
        IrStmt::If { cond, then, else_ } => {
            reject_if_divergent(cond, divergent)?;
            let cond_uniform = expr_is_uniform(cond, locals);
            let inner_divergent = divergent || !cond_uniform;
            check_stmts(then, locals, inner_divergent)?;
            if let Some(else_block) = else_ {
                check_stmts(else_block, locals, inner_divergent)?;
            }
            Ok(())
        }
        IrStmt::For { lo, hi, body, .. } => {
            reject_if_divergent(lo, divergent)?;
            reject_if_divergent(hi, divergent)?;
            let bounds_uniform = expr_is_uniform(lo, locals) && expr_is_uniform(hi, locals);
            check_stmts(body, locals, divergent || !bounds_uniform)
        }
        IrStmt::Eval(expr) => reject_if_divergent(expr, divergent),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::frontend::parser::parse_compute;

    fn check(source: &str) -> Result<(), String> {
        let module = parse_compute(source).expect("parse");
        check_compute_uniformity(&module.entry)
    }

    #[test]
    fn collective_inside_group_id_guard_is_uniform() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, push: P, id: global_id) {
                let r = group_id().x
                if r < push.n {
                    subgroup_any(id.x < 10u32)
                }
            }
        "#,
        )
        .expect("group-id guard is uniform");
    }

    #[test]
    fn subgroup_any_inside_local_id_guard_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, id: global_id) {
                if local_id().x < 5u32 {
                    subgroup_any(id.x < 10u32)
                }
            }
        "#,
        )
        .expect_err("local-id guard is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn subgroup_ballot_inside_local_id_guard_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, id: global_id) {
                if local_id().x < 5u32 {
                    subgroup_ballot(id.x < 10u32)
                }
            }
        "#,
        )
        .expect_err("local-id guard is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn subgroup_all_inside_group_id_guard_is_uniform() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, push: P, id: global_id) {
                if group_id().x < push.n {
                    subgroup_all(id.x < 10u32)
                }
            }
        "#,
        )
        .expect("group-id guard is uniform");
    }

    #[test]
    fn subgroup_shuffle_xor_inside_group_id_guard_is_uniform() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {
                if group_id().x < push.n {
                    out[id.x] = subgroup_shuffle_xor(id.x * 1.0, 1u32)
                }
            }
        "#,
        )
        .expect("group-id guard is uniform");
    }

    #[test]
    fn subgroup_shuffle_xor_inside_local_id_guard_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, id: global_id) {
                if local_id().x < 5u32 {
                    out[id.x] = subgroup_shuffle_xor(id.x * 1.0, 1u32)
                }
            }
        "#,
        )
        .expect_err("local-id guard is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn subgroup_reduce_add_inside_group_id_guard_is_uniform() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, push: P, id: global_id) {
                if group_id().x < push.n {
                    out[id.x] = subgroup_reduce_add(id.x)
                }
            }
        "#,
        )
        .expect("group-id guard is uniform");
    }

    #[test]
    fn subgroup_reduce_add_inside_local_id_guard_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, id: global_id) {
                if local_id().x < 5u32 {
                    out[id.x] = subgroup_reduce_add(id.x)
                }
            }
        "#,
        )
        .expect_err("local-id guard is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn workgroup_reduce_add_inside_group_id_guard_is_uniform() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {
                if group_id().x < push.n {
                    out[id.x] = workgroup_reduce_add(id.x * 1.0)
                }
            }
        "#,
        )
        .expect("group-id guard is uniform");
    }

    #[test]
    fn workgroup_reduce_add_inside_local_id_guard_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, id: global_id) {
                if local_id().x < 5u32 {
                    out[id.x] = workgroup_reduce_add(id.x * 1.0)
                }
            }
        "#,
        )
        .expect_err("local-id guard is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn collective_inside_uniform_bounds_loop_is_allowed() {
        check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, push: P, id: global_id) {
                if group_id().x < push.n {
                    for j in 0..push.n {
                        subgroup_any(id.x < j)
                    }
                }
            }
        "#,
        )
        .expect("push-constant loop bounds are uniform");
    }

    #[test]
    fn collective_inside_global_id_bounds_loop_is_rejected() {
        let err = check(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<u32>, id: global_id) {
                for j in 0..id.x {
                    subgroup_any(j < 1u32)
                }
            }
        "#,
        )
        .expect_err("global-id loop bound is divergent");
        assert!(err.contains("uniform"));
    }
}
