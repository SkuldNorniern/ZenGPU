use super::*;

pub(super) fn bin(op: IrBinOp, lhs: IrExpr, rhs: IrExpr) -> IrExpr {
    IrExpr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// Renames `Local`s per `map`. Names absent from `map` stay as they are.
///
/// Also inlines `fn` bodies at call sites.
pub(super) fn rename_expr(expr: &IrExpr, map: &HashMap<String, String>) -> IrExpr {
    let renamed = |name: &String| map.get(name).cloned().unwrap_or_else(|| name.clone());
    match expr {
        IrExpr::LitU32(v) => IrExpr::LitU32(*v),
        IrExpr::LitF32(v) => IrExpr::LitF32(*v),
        IrExpr::Local(name) => IrExpr::Local(renamed(name)),
        IrExpr::ScalarParam(name) => IrExpr::ScalarParam(name.clone()),
        IrExpr::BufferLoad { buf, index } => IrExpr::BufferLoad {
            buf: buf.clone(),
            index: Box::new(rename_expr(index, map)),
        },
        IrExpr::SharedLoad { name, index } => IrExpr::SharedLoad {
            name: name.clone(),
            index: Box::new(rename_expr(index, map)),
        },
        IrExpr::GlobalId(c) => IrExpr::GlobalId(*c),
        IrExpr::LocalId(c) => IrExpr::LocalId(*c),
        IrExpr::GroupId(c) => IrExpr::GroupId(*c),
        IrExpr::Input(name) => IrExpr::Input(name.clone()),
        IrExpr::FieldAccess { base, component } => IrExpr::FieldAccess {
            base: Box::new(rename_expr(base, map)),
            component: *component,
        },
        IrExpr::Swizzle { base, lanes } => IrExpr::Swizzle {
            base: Box::new(rename_expr(base, map)),
            lanes: lanes.clone(),
        },
        IrExpr::Construct { ty, dim, args } => IrExpr::Construct {
            ty: *ty,
            dim: *dim,
            args: args.iter().map(|a| rename_expr(a, map)).collect(),
        },
        IrExpr::Builtin { func, args } => IrExpr::Builtin {
            func: *func,
            args: args.iter().map(|a| rename_expr(a, map)).collect(),
        },
        IrExpr::Neg(inner) => IrExpr::Neg(Box::new(rename_expr(inner, map))),
        IrExpr::Binary { op, lhs, rhs } => IrExpr::Binary {
            op: *op,
            lhs: Box::new(rename_expr(lhs, map)),
            rhs: Box::new(rename_expr(rhs, map)),
        },
    }
}

pub(super) fn rename_stmt(stmt: &IrStmt, map: &HashMap<String, String>) -> IrStmt {
    let renamed = |name: &String| map.get(name).cloned().unwrap_or_else(|| name.clone());
    match stmt {
        IrStmt::Let { name, init } => IrStmt::Let {
            name: renamed(name),
            init: rename_expr(init, map),
        },
        IrStmt::AssignLocal { name, value } => IrStmt::AssignLocal {
            name: renamed(name),
            value: rename_expr(value, map),
        },
        IrStmt::AssignBuffer { buf, index, value } => IrStmt::AssignBuffer {
            buf: buf.clone(),
            index: rename_expr(index, map),
            value: rename_expr(value, map),
        },
        IrStmt::AtomicAdd { buf, index, value } => IrStmt::AtomicAdd {
            buf: buf.clone(),
            index: rename_expr(index, map),
            value: rename_expr(value, map),
        },
        IrStmt::AssignShared { name, index, value } => IrStmt::AssignShared {
            name: name.clone(),
            index: rename_expr(index, map),
            value: rename_expr(value, map),
        },
        IrStmt::Barrier => IrStmt::Barrier,
        IrStmt::Discard => IrStmt::Discard,
        IrStmt::If { cond, then, else_ } => IrStmt::If {
            cond: rename_expr(cond, map),
            then: then.iter().map(|s| rename_stmt(s, map)).collect(),
            else_: else_
                .as_ref()
                .map(|stmts| stmts.iter().map(|s| rename_stmt(s, map)).collect()),
        },
        IrStmt::For { var, lo, hi, body } => IrStmt::For {
            var: renamed(var),
            lo: rename_expr(lo, map),
            hi: rename_expr(hi, map),
            body: body.iter().map(|s| rename_stmt(s, map)).collect(),
        },
        IrStmt::Eval(e) => IrStmt::Eval(rename_expr(e, map)),
    }
}

/// Fills each `Construct` `ty`. It is `None` after parsing because the parser has no `TypeArena`.
///
/// Runs once at the end of `parse_kernel_rest` and `parse_graphics_rest`.
pub(super) fn resolve_construct_types_expr(expr: &IrExpr, types: &mut TypeArena) -> IrExpr {
    match expr {
        IrExpr::LitU32(v) => IrExpr::LitU32(*v),
        IrExpr::LitF32(v) => IrExpr::LitF32(*v),
        IrExpr::Local(name) => IrExpr::Local(name.clone()),
        IrExpr::ScalarParam(name) => IrExpr::ScalarParam(name.clone()),
        IrExpr::BufferLoad { buf, index } => IrExpr::BufferLoad {
            buf: buf.clone(),
            index: Box::new(resolve_construct_types_expr(index, types)),
        },
        IrExpr::SharedLoad { name, index } => IrExpr::SharedLoad {
            name: name.clone(),
            index: Box::new(resolve_construct_types_expr(index, types)),
        },
        IrExpr::GlobalId(c) => IrExpr::GlobalId(*c),
        IrExpr::LocalId(c) => IrExpr::LocalId(*c),
        IrExpr::GroupId(c) => IrExpr::GroupId(*c),
        IrExpr::Input(name) => IrExpr::Input(name.clone()),
        IrExpr::FieldAccess { base, component } => IrExpr::FieldAccess {
            base: Box::new(resolve_construct_types_expr(base, types)),
            component: *component,
        },
        IrExpr::Swizzle { base, lanes } => IrExpr::Swizzle {
            base: Box::new(resolve_construct_types_expr(base, types)),
            lanes: lanes.clone(),
        },
        IrExpr::Construct { dim, args, .. } => IrExpr::Construct {
            ty: Some(types.vec_id(*dim)),
            dim: *dim,
            args: args
                .iter()
                .map(|a| resolve_construct_types_expr(a, types))
                .collect(),
        },
        IrExpr::Builtin { func, args } => IrExpr::Builtin {
            func: *func,
            args: args
                .iter()
                .map(|a| resolve_construct_types_expr(a, types))
                .collect(),
        },
        IrExpr::Neg(inner) => IrExpr::Neg(Box::new(resolve_construct_types_expr(inner, types))),
        IrExpr::Binary { op, lhs, rhs } => IrExpr::Binary {
            op: *op,
            lhs: Box::new(resolve_construct_types_expr(lhs, types)),
            rhs: Box::new(resolve_construct_types_expr(rhs, types)),
        },
    }
}

pub(super) fn resolve_construct_types_stmt(stmt: &IrStmt, types: &mut TypeArena) -> IrStmt {
    match stmt {
        IrStmt::Let { name, init } => IrStmt::Let {
            name: name.clone(),
            init: resolve_construct_types_expr(init, types),
        },
        IrStmt::AssignLocal { name, value } => IrStmt::AssignLocal {
            name: name.clone(),
            value: resolve_construct_types_expr(value, types),
        },
        IrStmt::AssignBuffer { buf, index, value } => IrStmt::AssignBuffer {
            buf: buf.clone(),
            index: resolve_construct_types_expr(index, types),
            value: resolve_construct_types_expr(value, types),
        },
        IrStmt::AtomicAdd { buf, index, value } => IrStmt::AtomicAdd {
            buf: buf.clone(),
            index: resolve_construct_types_expr(index, types),
            value: resolve_construct_types_expr(value, types),
        },
        IrStmt::AssignShared { name, index, value } => IrStmt::AssignShared {
            name: name.clone(),
            index: resolve_construct_types_expr(index, types),
            value: resolve_construct_types_expr(value, types),
        },
        IrStmt::Barrier => IrStmt::Barrier,
        IrStmt::Discard => IrStmt::Discard,
        IrStmt::If { cond, then, else_ } => IrStmt::If {
            cond: resolve_construct_types_expr(cond, types),
            then: then
                .iter()
                .map(|s| resolve_construct_types_stmt(s, types))
                .collect(),
            else_: else_.as_ref().map(|blk| {
                blk.iter()
                    .map(|s| resolve_construct_types_stmt(s, types))
                    .collect()
            }),
        },
        IrStmt::For { var, lo, hi, body } => IrStmt::For {
            var: var.clone(),
            lo: resolve_construct_types_expr(lo, types),
            hi: resolve_construct_types_expr(hi, types),
            body: body
                .iter()
                .map(|s| resolve_construct_types_stmt(s, types))
                .collect(),
        },
        IrStmt::Eval(e) => IrStmt::Eval(resolve_construct_types_expr(e, types)),
    }
}
