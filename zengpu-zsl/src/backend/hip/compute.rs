use std::collections::HashMap;
use std::fmt::Write;

use crate::backend::hip::{ENTRY, HipShader};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::{BufElem, EntryKind, Module, Mutability, ParamKind, ScalarTy};

pub fn lower_compute(module: &Module) -> HipShader {
    let e = &module.entry;
    let EntryKind::Compute { local_size } = e.kind;

    let locals: HashMap<&str, ScalarTy> = e.locals.iter().map(|(n, t)| (n.as_str(), *t)).collect();

    let buffers: Vec<(&str, Mutability, BufElem)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, mutability } => Some((p.name.as_str(), *mutability, *elem)),
            _ => None,
        })
        .collect();

    let scalars: Vec<(&str, ScalarTy)> = e
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Scalar(t) => Some((p.name.as_str(), *t)),
            _ => None,
        })
        .collect();

    let mut s = String::new();

    let threads = local_size[0] * local_size[1] * local_size[2];
    let _ = writeln!(s, "extern \"C\" __global__ __launch_bounds__({threads})");
    let _ = writeln!(s, "void {ENTRY}(");

    for (i, (name, mutability, elem)) in buffers.iter().enumerate() {
        let comma = if i + 1 < buffers.len() + scalars.len() {
            ","
        } else {
            ""
        };
        match mutability {
            Mutability::Read => {
                let ty = hip_buffer_elem(*elem);
                let _ = writeln!(s, "    const {ty}* __restrict__ {name}{comma}");
            }
            Mutability::ReadWrite => {
                let ty = hip_buffer_elem(*elem);
                let _ = writeln!(s, "    {ty}* __restrict__ {name}{comma}");
            }
        }
    }

    for (i, (name, ty)) in scalars.iter().enumerate() {
        let comma = if i + 1 < scalars.len() { "," } else { "" };
        let cty = hip_scalar(*ty);
        let _ = writeln!(s, "    {cty} {name}{comma}");
    }

    s.push_str(") {\n");

    for shared in &e.shared {
        let _ = writeln!(s, "    __shared__ float {}[{}];", shared.name, shared.len);
    }

    let _ = writeln!(
        s,
        "    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;"
    );
    if local_size[1] > 1 || local_size[2] > 1 {
        let _ = writeln!(
            s,
            "    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;"
        );
        let _ = writeln!(
            s,
            "    unsigned int gz = blockIdx.z * blockDim.z + threadIdx.z;"
        );
    }

    let ctx = Ctx { locals };
    for st in &e.body {
        emit_stmt(&mut s, &ctx, st, 1);
    }

    s.push_str("}\n");

    HipShader {
        source: s,
        entry: ENTRY,
        buffer_count: buffers.len() as u32,
        has_scalars: !scalars.is_empty(),
        local_size,
    }
}

struct Ctx<'a> {
    locals: HashMap<&'a str, ScalarTy>,
}

fn indent(s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push_str("    ");
    }
}

fn hip_scalar(ty: ScalarTy) -> &'static str {
    match ty {
        ScalarTy::U32 => "unsigned int",
        ScalarTy::I32 => "int",
        ScalarTy::F32 => "float",
        ScalarTy::Bool => "bool",
    }
}

fn hip_buffer_elem(elem: BufElem) -> &'static str {
    match elem {
        BufElem::F32 => "float",
        BufElem::U32 => "unsigned int",
        BufElem::I32 => "int",
        _ => unreachable!("unsupported compute buffer element"),
    }
}

fn emit_stmt(s: &mut String, ctx: &Ctx<'_>, stmt: &IrStmt, depth: usize) {
    match stmt {
        IrStmt::Let { name, init } => {
            let ty = ctx
                .locals
                .get(name.as_str())
                .copied()
                .unwrap_or(ScalarTy::U32);
            indent(s, depth);
            let _ = writeln!(s, "{} {} = {};", hip_scalar(ty), name, emit_expr(init));
        }
        IrStmt::AssignLocal { name, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{} = {};", name, emit_expr(value));
        }
        IrStmt::AssignBuffer { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{}[{}] = {};", buf, emit_expr(index), emit_expr(value));
        }
        IrStmt::AtomicAdd { buf, index, value } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "atomicAdd(&{}[(unsigned int)({})], {});",
                buf,
                emit_expr(index),
                emit_expr(value)
            );
        }
        IrStmt::AssignShared { name, index, value } => {
            indent(s, depth);
            let _ = writeln!(s, "{}[{}] = {};", name, emit_expr(index), emit_expr(value));
        }
        IrStmt::Barrier => {
            indent(s, depth);
            s.push_str("__syncthreads();\n");
        }
        IrStmt::If { cond, then, else_ } => {
            indent(s, depth);
            let _ = writeln!(s, "if ({}) {{", emit_expr(cond));
            for st in then {
                emit_stmt(s, ctx, st, depth + 1);
            }
            if let Some(els) = else_ {
                indent(s, depth);
                s.push_str("} else {\n");
                for st in els {
                    emit_stmt(s, ctx, st, depth + 1);
                }
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::For { var, lo, hi, body } => {
            indent(s, depth);
            let _ = writeln!(
                s,
                "for (unsigned int {var} = {}; {var} < {}; {var}++) {{",
                emit_expr(lo),
                emit_expr(hi),
            );
            for st in body {
                emit_stmt(s, ctx, st, depth + 1);
            }
            indent(s, depth);
            s.push_str("}\n");
        }
        IrStmt::Eval(e) => {
            indent(s, depth);
            let _ = writeln!(s, "{};", emit_expr(e));
        }
    }
}

fn emit_expr(e: &IrExpr) -> String {
    match e {
        IrExpr::LitU32(v) => format!("{v}u"),
        IrExpr::LitF32(v) => {
            if v.fract() == 0.0 {
                format!("{v}.0f")
            } else {
                format!("{v}f")
            }
        }
        IrExpr::Local(n) => n.clone(),
        IrExpr::ScalarParam(n) => n.clone(),
        IrExpr::BufferLoad { buf, index } => format!("{}[{}]", buf, emit_expr(index)),
        IrExpr::SharedLoad { name, index } => format!("{}[{}]", name, emit_expr(index)),
        IrExpr::GlobalId(0) => "gx".into(),
        IrExpr::GlobalId(1) => "gy".into(),
        IrExpr::GlobalId(2) => "gz".into(),
        IrExpr::GlobalId(n) => format!("/* bad GlobalId({n}) */ 0u"),
        IrExpr::LocalId(c) => format!("threadIdx.{}", ["x", "y", "z"][*c as usize]),
        IrExpr::GroupId(c) => format!("blockIdx.{}", ["x", "y", "z"][*c as usize]),
        IrExpr::Input(n) => n.clone(),
        IrExpr::FieldAccess { base, component } => {
            let field = ["x", "y", "z", "w"]
                .get(*component as usize)
                .copied()
                .unwrap_or("x");
            format!("{}.{}", emit_expr(base), field)
        }
        IrExpr::VecConstruct { dim, args } => {
            let args_s: Vec<String> = args.iter().map(emit_expr).collect();
            format!("float{}({})", dim, args_s.join(", "))
        }
        IrExpr::Extend { base, scalar } => {
            format!("float4({}, {})", emit_expr(base), emit_expr(scalar))
        }
        IrExpr::Dot { a, b } => format!("dot({}, {})", emit_expr(a), emit_expr(b)),
        IrExpr::Builtin { func, args } => {
            let args_s: Vec<String> = args.iter().map(emit_expr).collect();
            match func {
                BuiltinFn::U32 => format!("((unsigned int)({}))", args_s[0]),
                BuiltinFn::Abs => format!("fabsf({})", args_s[0]),
                BuiltinFn::Sign => format!("(({0} > 0.0f) - ({0} < 0.0f))", args_s[0]),
                BuiltinFn::Exp => format!("expf({})", args_s[0]),
                BuiltinFn::Tanh => format!("tanhf({})", args_s[0]),
                BuiltinFn::Sin => format!("sinf({})", args_s[0]),
                BuiltinFn::Cos => format!("cosf({})", args_s[0]),
                BuiltinFn::Tan => format!("tanf({})", args_s[0]),
                BuiltinFn::IsNan => format!("isnan({})", args_s[0]),
                BuiltinFn::IsInf => format!("isinf({})", args_s[0]),
                BuiltinFn::IsFinite => format!("isfinite({})", args_s[0]),
                BuiltinFn::Log => format!("logf({})", args_s[0]),
                BuiltinFn::Sqrt => format!("sqrtf({})", args_s[0]),
                BuiltinFn::Floor => format!("floorf({})", args_s[0]),
                BuiltinFn::Ceil => format!("ceilf({})", args_s[0]),
                BuiltinFn::Fract => format!("({0} - floorf({0}))", args_s[0]),
                BuiltinFn::Pow => format!("powf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Min => format!("fminf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Max => format!("fmaxf({}, {})", args_s[0], args_s[1]),
                BuiltinFn::Clamp => {
                    format!("fminf(fmaxf({}, {}), {})", args_s[0], args_s[1], args_s[2])
                }
                BuiltinFn::Mix => format!(
                    "({0} + ({2}) * ({1} - {0}))",
                    args_s[0], args_s[1], args_s[2]
                ),
                BuiltinFn::Normalize => format!("__hip_normalize_stub({})", args_s[0]),
                BuiltinFn::Length => format!("sqrtf(dot({0}, {0}))", args_s[0]),
            }
        }
        IrExpr::Neg(inner) => format!("(-{})", emit_expr(inner)),
        IrExpr::Binary { op, lhs, rhs } => {
            let op_s = match op {
                IrBinOp::Add => "+",
                IrBinOp::Sub => "-",
                IrBinOp::Mul => "*",
                IrBinOp::Div => "/",
                IrBinOp::Lt => "<",
                IrBinOp::Le => "<=",
                IrBinOp::Gt => ">",
                IrBinOp::Ge => ">=",
                IrBinOp::Eq => "==",
                IrBinOp::Ne => "!=",
                IrBinOp::And => "&&",
                IrBinOp::Or => "||",
            };
            format!("({} {} {})", emit_expr(lhs), op_s, emit_expr(rhs))
        }
    }
}
