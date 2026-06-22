//! Native ZSL parser (compute) — tokens → [`ir::Module`].
//!
//! Dependency-free (no `syn`/`quote`). Recursive-descent over the [`lex`] token
//! stream. Statements need no terminators: the expression parser is greedy and
//! a statement ends where the next token can't extend it (maximal munch), which
//! matches `zen.md`'s terminator-free syntax without relying on newlines (lost
//! when the proc-macro shell stringifies its input).
//!
//! Grammar (compute slice):
//! ```text
//! module    := ("module" dotted)? push* attr* "kernel" Ident "(" params ")" block
//! push      := "push" Ident "{" (Ident ":" scalar)* "}"
//! attr      := "@" "workgroup_size" "(" Int ("," Int)* ")"
//! param     := Ident ":" ("device" "mut"? "buffer" "<" scalar ">" | "global_id" | Ident)
//! block     := "{" stmt* "}"
//! stmt      := "let" Ident (":" scalar)? "=" expr
//!            | "if" expr block ("else" block)?
//!            | "for" Ident "in" expr ".." expr block
//!            | expr ("=" expr)?
//! ```

use std::collections::HashMap;

use crate::frontend::lex::{Tok, Token, lex};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::{Entry, EntryKind, Module, Mutability, Param, ParamKind, ScalarTy};

/// A parse error: a message and the byte offset where it occurred (if known).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub at: Option<usize>,
}

/// Parse native ZSL compute source into a [`Module`].
pub fn parse_compute(src: &str) -> Result<Module, ParseError> {
    let toks = lex(src).map_err(|e| ParseError {
        msg: e.msg,
        at: Some(e.at),
    })?;
    let mut p = Parser::new(&toks);
    p.parse_module()
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    /// Push-block definitions: struct name → ordered (field, type).
    pushes: HashMap<String, Vec<(String, ScalarTy)>>,
}

/// Resolved symbol tables for the body being parsed.
struct Ctx {
    buffers: HashMap<String, bool>,       // name → writable
    scalars: HashMap<String, ScalarTy>,   // push-field name → type
    push_param: Option<String>,           // the `push:` param name
    id_param: Option<String>,             // the `id: global_id` param name
    locals: HashMap<String, ScalarTy>,    // name → type (accumulated)
    locals_order: Vec<(String, ScalarTy)>,
}

impl<'a> Parser<'a> {
    fn new(toks: &'a [Token]) -> Self {
        Self {
            toks,
            pos: 0,
            pushes: HashMap::new(),
        }
    }

    // ── Token cursor ─────────────────────────────────────────────────────────

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn span(&self) -> Option<usize> {
        self.toks.get(self.pos).map(|t| t.span.start).or_else(|| {
            self.toks.last().map(|t| t.span.end)
        })
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            msg: msg.into(),
            at: self.span(),
        })
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), ParseError> {
        if self.eat(t) {
            Ok(())
        } else {
            self.err(format!("expected {what}"))
        }
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Some(Tok::Ident(s)) => {
                let s = s.clone();
                self.pos += 1;
                Ok(s)
            }
            _ => self.err("expected identifier"),
        }
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ── Module ───────────────────────────────────────────────────────────────

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        // Optional `module dotted.path`.
        if self.eat_kw("module") {
            self.ident()?;
            while self.eat(&Tok::Dot) {
                self.ident()?;
            }
        }

        // Push blocks.
        while self.at_kw("push") {
            self.parse_push_block()?;
        }

        // Attributes (only @workgroup_size today).
        let mut local_size = [1u32, 1, 1];
        while self.eat(&Tok::At) {
            let name = self.ident()?;
            if name != "workgroup_size" {
                return self.err(format!("unknown attribute `@{name}`; expected @workgroup_size"));
            }
            self.expect(&Tok::LParen, "`(` after @workgroup_size")?;
            local_size = self.parse_workgroup_size()?;
            self.expect(&Tok::RParen, "`)`")?;
        }

        if !self.eat_kw("kernel") {
            return self.err("expected `kernel`");
        }
        let _name = self.ident()?;

        let mut ctx = Ctx {
            buffers: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            id_param: None,
            locals: HashMap::new(),
            locals_order: Vec::new(),
        };
        let params = self.parse_params(&mut ctx)?;
        let body = self.parse_block(&mut ctx)?;

        if self.pos != self.toks.len() {
            return self.err("unexpected trailing tokens after kernel");
        }

        Ok(Module {
            entry: Entry {
                kind: EntryKind::Compute { local_size },
                params,
                locals: ctx.locals_order,
                body,
            },
        })
    }

    fn parse_workgroup_size(&mut self) -> Result<[u32; 3], ParseError> {
        let mut dims = [1u32, 1, 1];
        let mut i = 0;
        loop {
            match self.peek() {
                Some(Tok::Int(v)) => {
                    if i < 3 {
                        dims[i] = *v as u32;
                    }
                    i += 1;
                    self.pos += 1;
                }
                _ => return self.err("expected workgroup size integer"),
            }
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(dims)
    }

    fn parse_push_block(&mut self) -> Result<(), ParseError> {
        self.eat_kw("push");
        let name = self.ident()?;
        self.expect(&Tok::LBrace, "`{` after push name")?;
        let mut fields = Vec::new();
        while !self.eat(&Tok::RBrace) {
            let field = self.ident()?;
            self.expect(&Tok::Colon, "`:` after field name")?;
            let ty = self.parse_scalar_type()?;
            fields.push((field, ty));
            self.eat(&Tok::Comma); // optional separator
            if self.peek().is_none() {
                return self.err("unterminated push block");
            }
        }
        self.pushes.insert(name, fields);
        Ok(())
    }

    fn parse_scalar_type(&mut self) -> Result<ScalarTy, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "u32" => Ok(ScalarTy::U32),
            "f32" => Ok(ScalarTy::F32),
            "bool" => Ok(ScalarTy::Bool),
            other => self.err(format!("unsupported scalar type `{other}`; use u32, f32, or bool")),
        }
    }

    // ── Params ───────────────────────────────────────────────────────────────

    fn parse_params(&mut self, ctx: &mut Ctx) -> Result<Vec<Param>, ParseError> {
        self.expect(&Tok::LParen, "`(` after kernel name")?;
        let mut params = Vec::new();
        while !self.eat(&Tok::RParen) {
            let name = self.ident()?;
            self.expect(&Tok::Colon, "`:` after param name")?;
            self.parse_param_type(ctx, &name, &mut params)?;
            self.eat(&Tok::Comma); // optional/trailing
            if self.peek().is_none() {
                return self.err("unterminated parameter list");
            }
        }
        Ok(params)
    }

    fn parse_param_type(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        params: &mut Vec<Param>,
    ) -> Result<(), ParseError> {
        if self.eat_kw("device") {
            let writable = self.eat_kw("mut");
            if !self.eat_kw("buffer") {
                return self.err("expected `buffer` in device buffer type");
            }
            self.expect(&Tok::Lt, "`<` in buffer<…>")?;
            let elem = self.parse_scalar_type()?;
            self.expect(&Tok::Gt, "`>` in buffer<…>")?;
            if elem != ScalarTy::F32 {
                return self.err("only buffer<f32> is supported");
            }
            ctx.buffers.insert(name.to_string(), writable);
            params.push(Param {
                name: name.to_string(),
                kind: ParamKind::Buffer {
                    elem: crate::frontend::types::BufElem::F32,
                    mutability: if writable {
                        Mutability::ReadWrite
                    } else {
                        Mutability::Read
                    },
                },
            });
            Ok(())
        } else if self.eat_kw("uniform") {
            self.err("uniform buffers are not yet supported in compute")
        } else if self.eat_kw("global_id") {
            if ctx.id_param.is_some() {
                return self.err("duplicate global_id parameter");
            }
            ctx.id_param = Some(name.to_string());
            Ok(())
        } else {
            // A push-struct param: `name: StructName`.
            let struct_name = self.ident()?;
            let fields = self
                .pushes
                .get(&struct_name)
                .cloned()
                .ok_or_else(|| ParseError {
                    msg: format!("unknown type `{struct_name}` (no matching push block)"),
                    at: self.span(),
                })?;
            if ctx.push_param.is_some() {
                return self.err("only one push parameter is supported");
            }
            ctx.push_param = Some(name.to_string());
            for (field, ty) in fields {
                ctx.scalars.insert(field.clone(), ty);
                params.push(Param {
                    name: field,
                    kind: ParamKind::Scalar(ty),
                });
            }
            Ok(())
        }
    }

    // ── Statements ─────────────────────────────────────────────────────────────

    fn parse_block(&mut self, ctx: &mut Ctx) -> Result<Vec<IrStmt>, ParseError> {
        self.expect(&Tok::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        while !self.eat(&Tok::RBrace) {
            if self.peek().is_none() {
                return self.err("unterminated block");
            }
            stmts.push(self.parse_stmt(ctx)?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        if self.eat_kw("let") {
            return self.parse_let(ctx);
        }
        if self.eat_kw("if") {
            return self.parse_if(ctx);
        }
        if self.eat_kw("for") {
            return self.parse_for(ctx);
        }
        // Expression statement or assignment.
        let lhs = self.parse_expr(ctx)?;
        if self.eat(&Tok::Eq) {
            let rhs = self.parse_expr(ctx)?;
            match lhs {
                IrExpr::Local(name) => Ok(IrStmt::AssignLocal { name, value: rhs }),
                IrExpr::BufferLoad { buf, index } => {
                    let writable = *ctx.buffers.get(&buf).unwrap_or(&false);
                    if !writable {
                        return self
                            .err(format!("`{buf}` is read-only; declare it `device mut buffer`"));
                    }
                    Ok(IrStmt::AssignBuffer {
                        buf,
                        index: *index,
                        value: rhs,
                    })
                }
                _ => self.err("invalid assignment target; use a local or buf[i]"),
            }
        } else {
            Ok(IrStmt::Eval(lhs))
        }
    }

    fn parse_let(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        let name = self.ident()?;
        let annot = if self.eat(&Tok::Colon) {
            Some(self.parse_scalar_type()?)
        } else {
            None
        };
        self.expect(&Tok::Eq, "`=` in let binding")?;
        let init = self.parse_expr(ctx)?;
        let ty = annot.unwrap_or_else(|| infer_ty(&init, ctx));
        ctx.locals.insert(name.clone(), ty);
        ctx.locals_order.push((name.clone(), ty));
        Ok(IrStmt::Let { name, init })
    }

    fn parse_if(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        let cond = self.parse_expr(ctx)?;
        let then = self.parse_block(ctx)?;
        let else_ = if self.eat_kw("else") {
            Some(self.parse_block(ctx)?)
        } else {
            None
        };
        Ok(IrStmt::If { cond, then, else_ })
    }

    fn parse_for(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        let var = self.ident()?;
        if !self.eat_kw("in") {
            return self.err("expected `in` in for-loop");
        }
        let lo = self.parse_expr(ctx)?;
        self.expect(&Tok::DotDot, "`..` in for-loop range")?;
        let hi = self.parse_expr(ctx)?;
        // The loop variable is a u32 local.
        ctx.locals.insert(var.clone(), ScalarTy::U32);
        ctx.locals_order.push((var.clone(), ScalarTy::U32));
        let body = self.parse_block(ctx)?;
        Ok(IrStmt::For { var, lo, hi, body })
    }

    // ── Expressions (precedence climbing) ──────────────────────────────────────

    fn parse_expr(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        self.parse_or(ctx)
    }

    fn parse_or(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_and(ctx)?;
        while self.eat(&Tok::OrOr) {
            let rhs = self.parse_and(ctx)?;
            lhs = bin(IrBinOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_and(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_cmp(ctx)?;
        while self.eat(&Tok::AndAnd) {
            let rhs = self.parse_cmp(ctx)?;
            lhs = bin(IrBinOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let lhs = self.parse_add(ctx)?;
        let op = match self.peek() {
            Some(Tok::Lt) => IrBinOp::Lt,
            Some(Tok::Le) => IrBinOp::Le,
            Some(Tok::Gt) => IrBinOp::Gt,
            Some(Tok::Ge) => IrBinOp::Ge,
            Some(Tok::EqEq) => IrBinOp::Eq,
            Some(Tok::Ne) => IrBinOp::Ne,
            _ => return Ok(lhs),
        };
        self.pos += 1;
        let rhs = self.parse_add(ctx)?;
        Ok(bin(op, lhs, rhs))
    }

    fn parse_add(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_mul(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => IrBinOp::Add,
                Some(Tok::Minus) => IrBinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_mul(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_unary(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => IrBinOp::Mul,
                Some(Tok::Slash) => IrBinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        if self.eat(&Tok::Minus) {
            let inner = self.parse_unary(ctx)?;
            return Ok(IrExpr::Neg(Box::new(inner)));
        }
        self.parse_primary(ctx)
    }

    fn parse_primary(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        match self.peek() {
            Some(Tok::Int(v)) => {
                let v = *v as u32;
                self.pos += 1;
                Ok(IrExpr::LitU32(v))
            }
            Some(Tok::Float(v)) => {
                let v = *v as f32;
                self.pos += 1;
                Ok(IrExpr::LitF32(v))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let e = self.parse_expr(ctx)?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Some(Tok::Ident(_)) => {
                let name = self.ident()?;
                // Call?
                if self.peek() == Some(&Tok::LParen) {
                    return self.parse_call(ctx, &name);
                }
                // Field access: push.field / id.x
                if self.eat(&Tok::Dot) {
                    let field = self.ident()?;
                    if ctx.push_param.as_deref() == Some(name.as_str()) {
                        if !ctx.scalars.contains_key(&field) {
                            return self.err(format!("push struct has no field `{field}`"));
                        }
                        return Ok(IrExpr::ScalarParam(field));
                    }
                    if ctx.id_param.as_deref() == Some(name.as_str()) {
                        let comp = match field.as_str() {
                            "x" => 0u32,
                            "y" => 1,
                            "z" => 2,
                            other => {
                                return self
                                    .err(format!("`global_id` has no field `.{other}`; use .x/.y/.z"));
                            }
                        };
                        return Ok(IrExpr::GlobalId(comp));
                    }
                    return self.err(format!("unknown field access on `{name}`"));
                }
                // Buffer index: buf[i]
                if self.eat(&Tok::LBracket) {
                    if !ctx.buffers.contains_key(&name) {
                        return self.err(format!("`{name}` is not a buffer; cannot index"));
                    }
                    let index = self.parse_expr(ctx)?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    return Ok(IrExpr::BufferLoad {
                        buf: name,
                        index: Box::new(index),
                    });
                }
                // Bare name → a local.
                if ctx.locals.contains_key(&name) {
                    Ok(IrExpr::Local(name))
                } else {
                    self.err(format!("unknown identifier `{name}`"))
                }
            }
            _ => self.err("expected an expression"),
        }
    }

    fn parse_call(&mut self, ctx: &Ctx, name: &str) -> Result<IrExpr, ParseError> {
        self.expect(&Tok::LParen, "`(`")?;
        let mut args = Vec::new();
        while !self.eat(&Tok::RParen) {
            args.push(self.parse_expr(ctx)?);
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated call arguments");
            }
        }
        let func = builtin_from_name(name)
            .ok_or_else(|| ParseError { msg: format!("unknown function `{name}`"), at: self.span() })?;
        Ok(IrExpr::Builtin { func, args })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bin(op: IrBinOp, lhs: IrExpr, rhs: IrExpr) -> IrExpr {
    IrExpr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "sqrt" => BuiltinFn::Sqrt,
        "floor" => BuiltinFn::Floor,
        "ceil" => BuiltinFn::Ceil,
        "fract" => BuiltinFn::Fract,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "pow" => BuiltinFn::Pow,
        "clamp" => BuiltinFn::Clamp,
        "mix" => BuiltinFn::Mix,
        _ => return None,
    })
}

/// Infer the scalar type of an expression (compute is scalar-only). Mirrors the
/// SPIR-V backend's value-type inference so a `let`'s declared type matches.
fn infer_ty(expr: &IrExpr, ctx: &Ctx) -> ScalarTy {
    match expr {
        IrExpr::LitU32(_) => ScalarTy::U32,
        IrExpr::LitF32(_) => ScalarTy::F32,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(ScalarTy::U32),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(ScalarTy::F32),
        IrExpr::GlobalId(_) => ScalarTy::U32,
        IrExpr::BufferLoad { .. } => ScalarTy::F32,
        IrExpr::Builtin { .. } => ScalarTy::F32,
        IrExpr::Neg(e) => infer_ty(e, ctx),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne
            | IrBinOp::And | IrBinOp::Or => ScalarTy::Bool,
            _ => {
                let l = infer_ty(lhs, ctx);
                let r = infer_ty(rhs, ctx);
                if l == ScalarTy::F32 || r == ScalarTy::F32 {
                    ScalarTy::F32
                } else {
                    ScalarTy::U32
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::spirv::lower_compute;

    fn lower(src: &str) -> Vec<u32> {
        let module = parse_compute(src).expect("parse");
        lower_compute(&module).expect("lower")
    }

    #[test]
    fn saxpy_parses_and_lowers() {
        let src = r#"
            push SaxpyPush { n: u32, alpha: f32 }
            @workgroup_size(256)
            kernel saxpy(
                push: SaxpyPush,
                x: device buffer<f32>,
                y: device mut buffer<f32>,
                id: global_id,
            ) {
                let i = id.x
                if i < push.n {
                    y[i] = push.alpha * x[i] + y[i]
                }
            }
        "#;
        let words = lower(src);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    #[test]
    fn module_and_workgroup_parsed() {
        let src = r#"
            module zengpu.examples.copy
            @workgroup_size(64)
            kernel copy(x: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                let i = id.x
                out[i] = x[i]
            }
        "#;
        let m = parse_compute(src).expect("parse");
        let EntryKind::Compute { local_size } = m.entry.kind;
        assert_eq!(local_size, [64, 1, 1]);
    }

    #[test]
    fn local_type_inference() {
        // `v` from a buffer load is f32; `i` from id.x is u32.
        let src = r#"
            @workgroup_size(1)
            kernel k(src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
                let i = id.x
                let v = src[i]
                dst[i] = v
            }
        "#;
        let m = parse_compute(src).expect("parse");
        let types: std::collections::HashMap<_, _> =
            m.entry.locals.iter().cloned().collect();
        assert_eq!(types["i"], ScalarTy::U32);
        assert_eq!(types["v"], ScalarTy::F32);
    }

    #[test]
    fn for_loop_and_builtins() {
        let src = r#"
            push P { n: u32 }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, p: P, id: global_id) {
                let acc: f32 = 0.0
                for j in 0..p.n {
                    acc = max(acc, sqrt(dst[j]))
                }
                dst[id.x] = acc
            }
        "#;
        let words = lower(src);
        assert!(!words.is_empty());
    }

    #[test]
    fn read_only_buffer_write_is_rejected() {
        let src = r#"
            @workgroup_size(1)
            kernel k(x: device buffer<f32>, id: global_id) {
                x[id.x] = 1.0
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("read-only"), "got: {}", e.msg);
    }

    #[test]
    fn unknown_identifier_is_rejected() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, id: global_id) {
                dst[id.x] = nope
            }
        "#;
        assert!(parse_compute(src).is_err());
    }
}
