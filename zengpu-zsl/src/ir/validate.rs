//! Seam between parsed IR and backends.
//!
//! `ValidatedModule` has private fields, so backends only receive the output of [`validate`].
//! Compute checks binary operand types and subgroup collective uniformity. The parser already
//! rejects structural errors.
//!
//! Not yet: graphics checks, and a backend that takes `&ValidatedModule`.

use crate::frontend::diagnostic::{Diagnostic, DiagnosticCode};
use crate::frontend::parser::Shader;

fn compute_type_ctx(entry: &crate::ir::Entry) -> crate::ir::typeck::ComputeTypeCtx {
    let mut ctx = crate::ir::typeck::ComputeTypeCtx {
        locals: entry.locals.iter().cloned().collect(),
        ..Default::default()
    };
    for p in &entry.params {
        match &p.kind {
            crate::ir::ParamKind::Scalar(ty) => {
                ctx.scalars.insert(p.name.clone(), *ty);
            }
            crate::ir::ParamKind::Buffer { elem, .. } => {
                ctx.buffers.insert(p.name.clone(), *elem);
            }
        }
    }
    ctx
}

/// Checks the four operand-type rules that `backend::spirv::compute::lower_binary` enforces. `Err`
/// uses that backend's wording.
fn check_binary_operand_types(
    op: crate::ir::node::IrBinOp,
    lhs_ty: crate::ir::types::TypeId,
    rhs_ty: crate::ir::types::TypeId,
    bool_id: crate::ir::types::TypeId,
) -> Result<(), String> {
    use crate::ir::node::IrBinOp::*;
    match op {
        And | Or => {
            if lhs_ty != bool_id || rhs_ty != bool_id {
                return Err("ZSL: `&&`/`||` require bool operands".into());
            }
        }
        BitAnd | BitOr | BitXor | Shl | Shr => {
            if lhs_ty == bool_id || rhs_ty == bool_id {
                return Err("ZSL: `&`/`|`/`^`/`<<`/`>>` currently require u32 operands".into());
            }
        }
        Div | Lt | Le | Gt | Ge | Eq | Ne | Add | Sub | Mul => {
            if lhs_ty == bool_id || rhs_ty == bool_id {
                return Err(format!("ZSL: `{op:?}` does not support bool operands"));
            }
        }
    }
    Ok(())
}

fn check_expr(
    expr: &crate::ir::node::IrExpr,
    ctx: &crate::ir::typeck::ComputeTypeCtx,
    types: &mut crate::ir::arena::TypeArena,
    bool_id: crate::ir::types::TypeId,
) -> Result<(), String> {
    use crate::ir::node::IrExpr;
    match expr {
        IrExpr::LitU32(_)
        | IrExpr::LitF32(_)
        | IrExpr::Local(_)
        | IrExpr::ScalarParam(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_)
        | IrExpr::Input(_) => Ok(()),
        IrExpr::BufferLoad { index, .. } | IrExpr::SharedLoad { index, .. } => {
            check_expr(index, ctx, types, bool_id)
        }
        IrExpr::FieldAccess { base, .. } => check_expr(base, ctx, types, bool_id),
        IrExpr::Construct { args, .. } | IrExpr::Builtin { args, .. } => {
            for arg in args {
                check_expr(arg, ctx, types, bool_id)?;
            }
            Ok(())
        }
        IrExpr::Neg(inner) => check_expr(inner, ctx, types, bool_id),
        IrExpr::Binary { op, lhs, rhs } => {
            check_expr(lhs, ctx, types, bool_id)?;
            check_expr(rhs, ctx, types, bool_id)?;
            let lhs_ty = crate::ir::typeck::type_of_compute_expr(lhs, ctx, types);
            let rhs_ty = crate::ir::typeck::type_of_compute_expr(rhs, ctx, types);
            check_binary_operand_types(*op, lhs_ty, rhs_ty, bool_id)
        }
    }
}

fn check_stmt(
    stmt: &crate::ir::node::IrStmt,
    ctx: &crate::ir::typeck::ComputeTypeCtx,
    types: &mut crate::ir::arena::TypeArena,
    bool_id: crate::ir::types::TypeId,
) -> Result<(), String> {
    use crate::ir::node::IrStmt;
    match stmt {
        IrStmt::Let { init, .. } | IrStmt::AssignLocal { value: init, .. } | IrStmt::Eval(init) => {
            check_expr(init, ctx, types, bool_id)
        }
        IrStmt::AssignBuffer { index, value, .. }
        | IrStmt::AssignShared { index, value, .. }
        | IrStmt::AtomicAdd { index, value, .. } => {
            check_expr(index, ctx, types, bool_id)?;
            check_expr(value, ctx, types, bool_id)
        }
        IrStmt::Barrier => Ok(()),
        IrStmt::If { cond, then, else_ } => {
            check_expr(cond, ctx, types, bool_id)?;
            for stmt in then {
                check_stmt(stmt, ctx, types, bool_id)?;
            }
            if let Some(else_block) = else_ {
                for stmt in else_block {
                    check_stmt(stmt, ctx, types, bool_id)?;
                }
            }
            Ok(())
        }
        IrStmt::For { lo, hi, body, .. } => {
            check_expr(lo, ctx, types, bool_id)?;
            check_expr(hi, ctx, types, bool_id)?;
            for stmt in body {
                check_stmt(stmt, ctx, types, bool_id)?;
            }
            Ok(())
        }
    }
}

/// A [`Shader`] that passed [`validate`]. Its fields are private, so only `validate` builds one.
pub struct ValidatedModule {
    shader: Shader,
}

impl ValidatedModule {
    pub fn shader(&self) -> &Shader {
        &self.shader
    }

    pub fn into_inner(self) -> Shader {
        self.shader
    }
}

/// Compute checks binary operand types and subgroup collectives in divergent flow. Graphics passes
/// through unchanged.
pub fn validate(shader: Shader) -> Result<ValidatedModule, Diagnostic> {
    if let Shader::Compute(module) = &shader {
        let ctx = compute_type_ctx(&module.entry);
        let mut types = module.types.clone();
        let bool_id = types.bool_id();
        for stmt in &module.entry.body {
            check_stmt(stmt, &ctx, &mut types, bool_id)
                .map_err(|msg| Diagnostic::new(DiagnosticCode::TypeMismatch, msg))?;
        }
        crate::ir::uniformity::check_compute_uniformity(&module.entry)
            .map_err(|msg| Diagnostic::new(DiagnosticCode::NonUniformControlFlow, msg))?;
    }
    Ok(ValidatedModule { shader })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::parse_compute;

    #[test]
    fn validate_wraps_and_unwraps_a_parsed_module() {
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            kernel noop(x: device mut buffer<f32>, id: global_id) {
                let i = id.x
                x[i] = x[i]
            }
        "#,
        )
        .expect("parse");
        let validated = validate(Shader::Compute(module)).expect("validate");
        match validated.shader() {
            Shader::Compute(_) => {}
            Shader::Graphics(_) => panic!("expected compute"),
        }
        match validated.into_inner() {
            Shader::Compute(_) => {}
            Shader::Graphics(_) => panic!("expected compute"),
        }
    }

    #[test]
    fn validate_rejects_bool_plus_u32() {
        let module = parse_compute(
            r#"
            @workgroup_size(1)
            kernel bad(out: device mut buffer<u32>) {
                let value = (1u32 < 2u32) + 1u32
                out[0] = value
            }
        "#,
        )
        .expect("parse");

        let err = match validate(Shader::Compute(module)) {
            Ok(_) => panic!("bool + u32 must be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.code, DiagnosticCode::TypeMismatch);
        assert_eq!(err.message, "ZSL: `Add` does not support bool operands");
    }

    #[test]
    fn validate_rejects_collective_in_divergent_control_flow() {
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            kernel bad(out: device mut buffer<u32>, id: global_id) {
                if local_id().x < 5u32 {
                    subgroup_any(id.x < 10u32)
                }
            }
        "#,
        )
        .expect("parse");

        let err = match validate(Shader::Compute(module)) {
            Ok(_) => panic!("divergent subgroup collective must be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.code, DiagnosticCode::NonUniformControlFlow);
        assert!(err.message.contains("uniform"));
    }
}
