//! Subgroup-uniformity check: subgroup ops, barriers, workgroup reductions and coop-matrix fragment
//! ops need uniform control flow. Conservative: buffer and shared loads count as divergent. No
//! dataflow or alias analysis.

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
        | IrExpr::SharedLoad { .. }
        | IrExpr::FragLoad { .. } => false,
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
        IrExpr::Sample { .. } | IrExpr::Texel { .. } => false,
        IrExpr::TexSize { .. } => true,
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
                    | BuiltinFn::Mma
                    | BuiltinFn::FragZeroAcc
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
        IrExpr::Texel { args, .. } => args.iter().any(expr_contains_subgroup_collective),
        IrExpr::FragLoad { .. } => true,
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
        | IrExpr::Input(_)
        | IrExpr::TexSize { .. } => false,
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
        return Err(non_uniform_message("subgroup collective"));
    }
    Ok(())
}

fn reject_if_divergent_collective(what: &str, divergent: bool) -> Result<(), String> {
    if divergent {
        return Err(non_uniform_message(what));
    }
    Ok(())
}

fn non_uniform_message(what: &str) -> String {
    format!(
        "ZSL: {what} must occur in subgroup-uniform control flow (not under a local_id or a non-uniform loop bound)"
    )
}

fn check_stmt(
    stmt: &IrStmt,
    locals: &mut HashMap<String, bool>,
    divergent: bool,
) -> Result<(), String> {
    match stmt {
        IrStmt::Let { name, init } => {
            reject_if_divergent(init, divergent)?;
            let uniform = !divergent && expr_is_uniform(init, locals);
            locals.insert(name.clone(), uniform);
            Ok(())
        }
        IrStmt::AssignLocal { name, value } => {
            reject_if_divergent(value, divergent)?;
            let uniform = !divergent && expr_is_uniform(value, locals);
            locals.insert(name.clone(), uniform);
            Ok(())
        }
        IrStmt::FragStore {
            offset, ld, value, ..
        } => {
            reject_if_divergent_collective("frag_store", divergent)?;
            reject_if_divergent(offset, divergent)?;
            reject_if_divergent(ld, divergent)?;
            reject_if_divergent(value, divergent)
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => {
            reject_if_divergent(index, divergent)?;
            reject_if_divergent(value, divergent)
        }
        IrStmt::ImageStore { args, .. } => args
            .iter()
            .try_for_each(|arg| reject_if_divergent(arg, divergent)),
        IrStmt::Barrier => reject_if_divergent_collective("workgroup barrier", divergent),
        IrStmt::Discard => Ok(()),
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
        IrStmt::For { var, lo, hi, body } => {
            reject_if_divergent(lo, divergent)?;
            reject_if_divergent(hi, divergent)?;
            let bounds_uniform = expr_is_uniform(lo, locals) && expr_is_uniform(hi, locals);
            locals.insert(var.clone(), !divergent && bounds_uniform);
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
    fn collectives_are_uniform_under_group_guards_and_divergent_under_local_guards() {
        for call in [
            "subgroup_any(id.x < 10u32)",
            "subgroup_ballot(id.x < 10u32)",
            "subgroup_all(id.x < 10u32)",
            "subgroup_shuffle_xor(id.x * 1.0, 1u32)",
            "subgroup_reduce_add(id.x * 1.0)",
            "workgroup_reduce_add(id.x * 1.0)",
        ] {
            let uniform = format!(
                r#"
            push P {{ n: u32 }}
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {{
                if group_id().x < push.n {{
                    {call}
                }}
            }}
        "#
            );
            check(&uniform).expect("group-id guard is uniform");
            let divergent = format!(
                r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, id: global_id) {{
                if local_id().x < 5u32 {{
                    {call}
                }}
            }}
        "#
            );
            let err = check(&divergent).expect_err("local-id guard is divergent");
            assert!(err.contains("uniform"), "{call}: {err}");
        }
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
    fn barrier_under_local_id_guard_is_rejected_on_every_backend() {
        use crate::backend::{cuda, hip, msl, spirv};
        let accepted = parse_compute(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, id: global_id) {
                if group_id().x < 4u32 {
                    barrier()
                }
                out[id.x] = 1.0
            }
        "#,
        )
        .expect("parse");
        let rejected = parse_compute(
            r#"
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, id: global_id) {
                if local_id().x < 5u32 {
                    barrier()
                }
                out[id.x] = 1.0
            }
        "#,
        )
        .expect("parse");
        assert!(spirv::lower_compute(&accepted).is_ok());
        assert!(hip::lower_compute(&accepted).is_ok());
        assert!(cuda::lower_compute(&accepted).is_ok());
        assert!(msl::lower_compute(&accepted).is_ok());
        for msg in [
            spirv::lower_compute(&rejected)
                .map(|_| ())
                .unwrap_err()
                .message,
            hip::lower_compute(&rejected)
                .map(|_| ())
                .unwrap_err()
                .message,
            cuda::lower_compute(&rejected)
                .map(|_| ())
                .unwrap_err()
                .message,
            msl::lower_compute(&rejected)
                .map(|_| ())
                .unwrap_err()
                .message,
        ] {
            assert!(msg.contains("barrier") && msg.contains("uniform"), "{msg}");
        }
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

    #[test]
    fn barrier_in_loop_with_uniform_bounds_guard_is_accepted() {
        check(
            r#"
            push P { n: u32, m: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {
                for k in 0..push.n {
                    if k < push.m {
                        barrier()
                    }
                }
                out[id.x] = 1.0
            }
        "#,
        )
        .expect("loop variable under uniform bounds is uniform");
    }

    #[test]
    fn barrier_under_loop_variable_guard_with_divergent_bound_is_rejected() {
        let err = check(
            r#"
            push P { m: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {
                for k in 0..id.x {
                    if k < push.m {
                        barrier()
                    }
                }
                out[id.x] = 1.0
            }
        "#,
        )
        .expect_err("global-id loop bound is divergent");
        assert!(err.contains("uniform"));
    }

    #[test]
    fn loop_inside_local_id_guard_stays_divergent() {
        let err = check(
            r#"
            push P { n: u32 }
            @workgroup_size(64)
            kernel k(out: device mut buffer<f32>, push: P, id: global_id) {
                if local_id().x < 5u32 {
                    for k in 0..push.n {
                        barrier()
                    }
                }
                out[id.x] = 1.0
            }
        "#,
        )
        .expect_err("loop inside a divergent if is divergent");
        assert!(err.contains("uniform"));
    }
}
