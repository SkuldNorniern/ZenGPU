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
//!            | "barrier" "(" ")"
//!            | "atomic_add" "(" Ident "," expr "," expr ")"
//!            | expr ("=" expr)?
//! ```

use std::collections::HashMap;

use crate::frontend::lex::{Tok, Token, lex};
use crate::ir::BufElem;
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::{
    Entry, EntryKind, GfxInput, GfxScalar, GfxTy, GraphicsEntry, GraphicsModule, Module,
    Mutability, Param, ParamKind, ScalarTy, SharedDecl,
};

/// A parse error: a message and the byte offset where it occurred (if known).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub at: Option<usize>,
}

/// A parsed shader: a compute kernel or a graphics (vertex/fragment) entry.
pub enum Shader {
    Compute(Module),
    Graphics(GraphicsModule),
}

/// Parse native ZSL source, dispatching on the entry keyword
/// (`kernel`/`vertex`/`fragment`).
pub fn parse_zsl(src: &str) -> Result<Shader, ParseError> {
    let toks = lex(src).map_err(|e| ParseError {
        msg: e.msg,
        at: Some(e.at),
    })?;
    let mut p = Parser::new(&toks);
    p.parse_shader()
}

/// Parse native ZSL compute source into a [`Module`]. Convenience wrapper over
/// [`parse_zsl`] used by tests; the macro shell dispatches via [`parse_zsl`].
#[allow(dead_code)]
pub fn parse_compute(src: &str) -> Result<Module, ParseError> {
    match parse_zsl(src)? {
        Shader::Compute(m) => Ok(m),
        Shader::Graphics(_) => Err(ParseError {
            msg: "expected a compute `kernel`, found a graphics entry".into(),
            at: None,
        }),
    }
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    /// Push-block definitions: struct name → ordered (field, type). Stored as
    /// the graphics-superset [`GfxTy`]; compute narrows to scalar fields.
    pushes: HashMap<String, Vec<(String, GfxTy)>>,
}

/// Resolved symbol tables for the compute body being parsed.
struct Ctx {
    buffers: HashMap<String, bool>,     // name → writable
    scalars: HashMap<String, ScalarTy>, // push-field name → type
    push_param: Option<String>,         // the `push:` param name
    id_param: Option<String>,           // the `id: global_id` param name
    locals: HashMap<String, ScalarTy>,  // name → type (accumulated)
    locals_order: Vec<(String, ScalarTy)>,
    shared: HashMap<String, SharedDecl>,
    shared_order: Vec<String>,
}

/// Resolved symbol tables for a graphics (vertex/fragment) body.
struct GfxCtx {
    inputs: HashMap<String, GfxTy>,  // @location input name → type
    scalars: HashMap<String, GfxTy>, // push-field name → type
    push_param: Option<String>,      // the push param name
    buffers: HashMap<String, bool>,  // name → writable
    locals: HashMap<String, GfxTy>,
    locals_order: Vec<(String, GfxTy)>,
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
        self.toks
            .get(self.pos)
            .map(|t| t.span.start)
            .or_else(|| self.toks.last().map(|t| t.span.end))
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

    fn parse_shader(&mut self) -> Result<Shader, ParseError> {
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

        // Attributes (only @workgroup_size today, for compute).
        let mut local_size = [1u32, 1, 1];
        while self.eat(&Tok::At) {
            let name = self.ident()?;
            if name != "workgroup_size" {
                return self.err(format!(
                    "unknown attribute `@{name}`; expected @workgroup_size"
                ));
            }
            self.expect(&Tok::LParen, "`(` after @workgroup_size")?;
            local_size = self.parse_workgroup_size()?;
            self.expect(&Tok::RParen, "`)`")?;
        }

        if self.eat_kw("kernel") {
            let _name = self.ident()?;
            Ok(Shader::Compute(self.parse_kernel_rest(local_size)?))
        } else if self.eat_kw("vertex") {
            let _name = self.ident()?;
            Ok(Shader::Graphics(self.parse_graphics_rest(false)?))
        } else if self.eat_kw("fragment") {
            let _name = self.ident()?;
            Ok(Shader::Graphics(self.parse_graphics_rest(true)?))
        } else {
            self.err("expected `kernel`, `vertex`, or `fragment`")
        }
    }

    fn parse_kernel_rest(&mut self, local_size: [u32; 3]) -> Result<Module, ParseError> {
        let mut ctx = Ctx {
            buffers: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            id_param: None,
            locals: HashMap::new(),
            locals_order: Vec::new(),
            shared: HashMap::new(),
            shared_order: Vec::new(),
        };
        let params = self.parse_params(&mut ctx)?;
        let body = self.parse_block(&mut ctx)?;

        let shared = ctx.shared_order.iter().map(|name| SharedDecl {
            name: name.clone(),
            elem: ctx.shared[name].elem,
            len: ctx.shared[name].len,
        }).collect();

        if self.pos != self.toks.len() {
            return self.err("unexpected trailing tokens after kernel");
        }

        Ok(Module {
            entry: Entry {
                kind: EntryKind::Compute { local_size },
                params,
                shared,
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
            let ty = self.parse_any_type()?;
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
            other => self.err(format!(
                "unsupported scalar type `{other}`; use u32, f32, or bool"
            )),
        }
    }

    /// Parse any value type (scalar/vector/matrix), used for push fields and
    /// graphics declarations: `f32`, `u32`, `f32x2/3/4`, `mat4x4<f32>`.
    fn parse_any_type(&mut self) -> Result<GfxTy, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "f32" => Ok(GfxTy::F32),
            "u32" => Ok(GfxTy::U32),
            "f32x2" => Ok(GfxTy::Vec2),
            "f32x3" => Ok(GfxTy::Vec3),
            "f32x4" => Ok(GfxTy::Vec4),
            "mat4x4" => {
                self.expect(&Tok::Lt, "`<` in mat4x4<…>")?;
                let elem = self.ident()?;
                if elem != "f32" {
                    return self.err("only mat4x4<f32> is supported");
                }
                self.expect(&Tok::Gt, "`>` in mat4x4<…>")?;
                Ok(GfxTy::Mat4)
            }
            other => self.err(format!(
                "unsupported type `{other}`; use f32/u32/f32x2/f32x3/f32x4/mat4x4<f32>"
            )),
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
                    elem: BufElem::F32,
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
            for (field, gty) in fields {
                let ty = match gty {
                    GfxTy::U32 => ScalarTy::U32,
                    GfxTy::F32 => ScalarTy::F32,
                    _ => {
                        return self
                            .err(format!("compute push field `{field}` must be u32 or f32"));
                    }
                };
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
            if self.eat_kw("shared") {
                self.parse_shared(ctx)?;
            } else {
                stmts.push(self.parse_stmt(ctx)?);
            }
        }
        Ok(stmts)
    }

    fn parse_shared(&mut self, ctx: &mut Ctx) -> Result<(), ParseError> {
        let name = self.ident()?;
        self.expect(&Tok::Colon, "`:` after shared name")?;
        if !self.eat_kw("array") {
            return self.err("shared declarations require `array<f32, N>`");
        }
        self.expect(&Tok::Lt, "`<` in array<f32, N>")?;
        let elem = self.parse_scalar_type()?;
        if elem != ScalarTy::F32 {
            return self.err("only shared array<f32, N> is supported");
        }
        self.expect(&Tok::Comma, "`,` in array<f32, N>")?;
        let len = match self.peek() {
            Some(Tok::Int(v)) if *v > 0 && *v <= u32::MAX as u64 => {
                let len = *v as u32;
                self.pos += 1;
                len
            }
            _ => return self.err("shared array length must be a positive u32 literal"),
        };
        self.expect(&Tok::Gt, "`>` in array<f32, N>")?;
        if ctx.shared.contains_key(&name) || ctx.buffers.contains_key(&name)
            || ctx.locals.contains_key(&name)
        {
            return self.err(format!("duplicate symbol `{name}`"));
        }
        ctx.shared_order.push(name.clone());
        ctx.shared.insert(name.clone(), SharedDecl { name, elem, len });
        Ok(())
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
        if self.eat_kw("barrier") {
            self.expect(&Tok::LParen, "`(` after barrier")?;
            self.expect(&Tok::RParen, "`)` after barrier(")?;
            return Ok(IrStmt::Barrier);
        }
        if self.eat_kw("atomic_add") {
            self.expect(&Tok::LParen, "`(` after atomic_add")?;
            let buf = self.ident()?;
            self.expect(&Tok::Comma, "`,` after atomic_add buffer")?;
            let index = self.parse_expr(ctx)?;
            self.expect(&Tok::Comma, "`,` after atomic_add index")?;
            let value = self.parse_expr(ctx)?;
            self.expect(&Tok::RParen, "`)` after atomic_add arguments")?;
            if !*ctx.buffers.get(&buf).unwrap_or(&false) {
                return self.err(format!(
                    "`{buf}` is not a mutable device buffer; declare it `device mut buffer`"
                ));
            }
            let index_ty = infer_ty(&index, ctx);
            if index_ty != ScalarTy::U32 && index_ty != ScalarTy::F32 {
                return self.err("atomic_add index must be a u32- or f32-typed expression");
            }
            if infer_ty(&value, ctx) != ScalarTy::F32 {
                return self.err("atomic_add value must be an f32-typed expression");
            }
            return Ok(IrStmt::AtomicAdd { buf, index, value });
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
                        return self.err(format!(
                            "`{buf}` is read-only; declare it `device mut buffer`"
                        ));
                    }
                    Ok(IrStmt::AssignBuffer {
                        buf,
                        index: *index,
                        value: rhs,
                    })
                }
                IrExpr::SharedLoad { name, index } => {
                    if !ctx.shared.contains_key(&name) {
                        return self.err(format!("`{name}` is not a declared shared array"));
                    }
                    Ok(IrStmt::AssignShared { name, index: *index, value: rhs })
                }
                _ => self.err("invalid assignment target; use a local, buffer[i], or shared[i]"),
            }
        } else {
            Ok(IrStmt::Eval(lhs))
        }
    }

    fn parse_let(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        let name = self.ident()?;
        if ctx.shared.contains_key(&name) {
            return self.err(format!("`{name}` is already a shared array"));
        }
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
        if ctx.shared.contains_key(&var) {
            return self.err(format!("`{var}` is already a shared array"));
        }
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
                    if name == "local_id" || name == "group_id" {
                        self.expect(&Tok::LParen, "`(`")?;
                        self.expect(&Tok::RParen, "`)`")?;
                        self.expect(&Tok::Dot, "`.x`, `.y`, or `.z` after index builtin")?;
                        let field = self.ident()?;
                        let comp = match field.as_str() {
                            "x" => 0,
                            "y" => 1,
                            "z" => 2,
                            other => return self.err(format!("index builtin has no field `.{other}`; use .x/.y/.z")),
                        };
                        return Ok(if name == "local_id" { IrExpr::LocalId(comp) } else { IrExpr::GroupId(comp) });
                    }
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
                                return self.err(format!(
                                    "`global_id` has no field `.{other}`; use .x/.y/.z"
                                ));
                            }
                        };
                        return Ok(IrExpr::GlobalId(comp));
                    }
                    return self.err(format!("unknown field access on `{name}`"));
                }
                // Buffer index: buf[i]
                if self.eat(&Tok::LBracket) {
                    let index = self.parse_expr(ctx)?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    if ctx.buffers.contains_key(&name) {
                        return Ok(IrExpr::BufferLoad { buf: name, index: Box::new(index) });
                    }
                    if ctx.shared.contains_key(&name) {
                        if infer_ty(&index, ctx) != ScalarTy::U32 {
                            return self.err("shared array index must be u32");
                        }
                        return Ok(IrExpr::SharedLoad { name, index: Box::new(index) });
                    }
                    return self.err(format!("`{name}` is not a buffer or shared array; cannot index"));
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
        let func = builtin_from_name(name).ok_or_else(|| ParseError {
            msg: format!("unknown function `{name}`"),
            at: self.span(),
        })?;
        if func == BuiltinFn::U32 && args.len() != 1 {
            return self.err("u32() takes 1 arg");
        }
        Ok(IrExpr::Builtin { func, args })
    }

    // ── Graphics (vertex/fragment) ─────────────────────────────────────────────

    fn parse_graphics_rest(&mut self, is_fragment: bool) -> Result<GraphicsModule, ParseError> {
        let mut ctx = GfxCtx {
            inputs: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            buffers: HashMap::new(),
            locals: HashMap::new(),
            locals_order: Vec::new(),
        };
        let (inputs, scalar_params, buf_params) = self.parse_gfx_params(&mut ctx)?;

        self.expect(&Tok::Arrow, "`->` return type")?;
        let ret_types = self.parse_ret_types()?;
        if ret_types[0] != GfxTy::Vec4 {
            return self.err("first return type must be f32x4 (position/color)");
        }
        let varyings: Vec<GfxTy> = if is_fragment {
            Vec::new()
        } else {
            ret_types[1..].to_vec()
        };

        let (body, ret) = self.parse_gfx_body(&mut ctx, &varyings)?;
        if self.pos != self.toks.len() {
            return self.err("unexpected trailing tokens after entry point");
        }
        Ok(GraphicsModule {
            entry: GraphicsEntry {
                is_fragment,
                inputs,
                scalar_params,
                buf_params,
                varyings,
                locals: ctx.locals_order,
                body,
                ret,
            },
        })
    }

    #[allow(clippy::type_complexity)]
    fn parse_gfx_params(
        &mut self,
        ctx: &mut GfxCtx,
    ) -> Result<(Vec<GfxInput>, Vec<GfxScalar>, Vec<String>), ParseError> {
        self.expect(&Tok::LParen, "`(` after entry name")?;
        let mut inputs = Vec::new();
        let mut scalar_params = Vec::new();
        let mut buf_params = Vec::new();
        while !self.eat(&Tok::RParen) {
            // Optional @location(N).
            let mut location = None;
            if self.eat(&Tok::At) {
                let a = self.ident()?;
                if a != "location" {
                    return self.err("only @location is supported on parameters");
                }
                self.expect(&Tok::LParen, "`(`")?;
                location = Some(match self.peek() {
                    Some(Tok::Int(v)) => {
                        let v = *v as u32;
                        self.pos += 1;
                        v
                    }
                    _ => return self.err("expected location index"),
                });
                self.expect(&Tok::RParen, "`)`")?;
            }
            let name = self.ident()?;
            self.expect(&Tok::Colon, "`:` after param name")?;
            if let Some(loc) = location {
                let ty = self.parse_any_type()?;
                if matches!(ty, GfxTy::Mat4) {
                    return self.err("input type must be f32/u32/f32x2/f32x3/f32x4");
                }
                ctx.inputs.insert(name.clone(), ty);
                inputs.push(GfxInput {
                    name,
                    location: loc,
                    ty,
                });
            } else if self.eat_kw("device") {
                let writable = self.eat_kw("mut");
                if !self.eat_kw("buffer") {
                    return self.err("expected `buffer`");
                }
                self.expect(&Tok::Lt, "`<`")?;
                let elem = self.parse_any_type()?;
                self.expect(&Tok::Gt, "`>`")?;
                if elem != GfxTy::F32 {
                    return self.err("only buffer<f32> is supported");
                }
                ctx.buffers.insert(name.clone(), writable);
                buf_params.push(name);
            } else {
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
                ctx.push_param = Some(name);
                for (field, gty) in fields {
                    if !matches!(gty, GfxTy::F32 | GfxTy::U32 | GfxTy::Mat4) {
                        return self.err(format!(
                            "graphics push field `{field}` must be f32/u32/mat4x4"
                        ));
                    }
                    ctx.scalars.insert(field.clone(), gty);
                    scalar_params.push(GfxScalar {
                        name: field,
                        ty: gty,
                    });
                }
            }
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated parameter list");
            }
        }
        Ok((inputs, scalar_params, buf_params))
    }

    /// Parse a return type: `type` or `( type, type, … )`, each optionally
    /// prefixed by an attribute (`@location(0)` / `@builtin(position)`).
    fn parse_ret_types(&mut self) -> Result<Vec<GfxTy>, ParseError> {
        if self.eat(&Tok::LParen) {
            let mut v = vec![self.parse_ret_one()?];
            while self.eat(&Tok::Comma) {
                if self.peek() == Some(&Tok::RParen) {
                    break;
                }
                v.push(self.parse_ret_one()?);
            }
            self.expect(&Tok::RParen, "`)`")?;
            Ok(v)
        } else {
            Ok(vec![self.parse_ret_one()?])
        }
    }

    fn parse_ret_one(&mut self) -> Result<GfxTy, ParseError> {
        if self.eat(&Tok::At) {
            let _attr = self.ident()?;
            self.expect(&Tok::LParen, "`(`")?;
            while !self.eat(&Tok::RParen) {
                if self.peek().is_none() {
                    return self.err("unterminated attribute");
                }
                self.pos += 1;
            }
        }
        self.parse_any_type()
    }

    fn parse_gfx_body(
        &mut self,
        ctx: &mut GfxCtx,
        varyings: &[GfxTy],
    ) -> Result<(Vec<IrStmt>, Vec<IrExpr>), ParseError> {
        self.expect(&Tok::LBrace, "`{`")?;
        let mut body = Vec::new();
        let expected = 1 + varyings.len();
        loop {
            match self.peek() {
                None => return self.err("unterminated block"),
                Some(Tok::RBrace) => {
                    return self.err("vertex/fragment body needs a tail expression");
                }
                _ => {}
            }
            if self.eat_kw("let") {
                body.push(self.parse_gfx_let(ctx)?);
                continue;
            }
            if self.eat_kw("if") {
                body.push(self.parse_gfx_if(ctx)?);
                continue;
            }
            // Tuple tail `(a, b, …)` (or a parenthesized single expression).
            let tail = if self.peek() == Some(&Tok::LParen) {
                self.parse_gfx_tail(ctx)?
            } else {
                let e = self.parse_g_expr(ctx)?;
                if self.eat(&Tok::Eq) {
                    let rhs = self.parse_g_expr(ctx)?;
                    body.push(self.gfx_assign(ctx, e, rhs)?);
                    continue;
                }
                vec![e]
            };
            self.expect(&Tok::RBrace, "`}` after tail expression")?;
            if tail.len() != expected {
                return self.err(format!(
                    "return has {} value(s), expected {} (position + {} varyings)",
                    tail.len(),
                    expected,
                    varyings.len()
                ));
            }
            return Ok((body, tail));
        }
    }

    fn parse_gfx_tail(&mut self, ctx: &GfxCtx) -> Result<Vec<IrExpr>, ParseError> {
        let save = self.pos;
        self.pos += 1; // consume `(`
        let first = self.parse_g_expr(ctx)?;
        if self.peek() == Some(&Tok::Comma) {
            let mut elems = vec![first];
            while self.eat(&Tok::Comma) {
                if self.peek() == Some(&Tok::RParen) {
                    break;
                }
                elems.push(self.parse_g_expr(ctx)?);
            }
            self.expect(&Tok::RParen, "`)`")?;
            Ok(elems)
        } else {
            // Parenthesized single expression — backtrack and parse normally.
            self.pos = save;
            Ok(vec![self.parse_g_expr(ctx)?])
        }
    }

    fn parse_gfx_stmts(&mut self, ctx: &mut GfxCtx) -> Result<Vec<IrStmt>, ParseError> {
        self.expect(&Tok::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        loop {
            if self.eat(&Tok::RBrace) {
                break;
            }
            if self.peek().is_none() {
                return self.err("unterminated block");
            }
            if self.eat_kw("let") {
                stmts.push(self.parse_gfx_let(ctx)?);
                continue;
            }
            if self.eat_kw("if") {
                stmts.push(self.parse_gfx_if(ctx)?);
                continue;
            }
            let e = self.parse_g_expr(ctx)?;
            self.expect(&Tok::Eq, "`=` (block statements must be assignments)")?;
            let rhs = self.parse_g_expr(ctx)?;
            stmts.push(self.gfx_assign(ctx, e, rhs)?);
        }
        Ok(stmts)
    }

    fn parse_gfx_let(&mut self, ctx: &mut GfxCtx) -> Result<IrStmt, ParseError> {
        let name = self.ident()?;
        let annot = if self.eat(&Tok::Colon) {
            Some(self.parse_any_type()?)
        } else {
            None
        };
        self.expect(&Tok::Eq, "`=` in let binding")?;
        let init = self.parse_g_expr(ctx)?;
        let ty = annot.unwrap_or_else(|| infer_gfx_ty(&init, ctx));
        ctx.locals.insert(name.clone(), ty);
        ctx.locals_order.push((name.clone(), ty));
        Ok(IrStmt::Let { name, init })
    }

    fn parse_gfx_if(&mut self, ctx: &mut GfxCtx) -> Result<IrStmt, ParseError> {
        let cond = self.parse_g_expr(ctx)?;
        let then = self.parse_gfx_stmts(ctx)?;
        let else_ = if self.eat_kw("else") {
            Some(self.parse_gfx_stmts(ctx)?)
        } else {
            None
        };
        Ok(IrStmt::If { cond, then, else_ })
    }

    fn gfx_assign(&self, ctx: &GfxCtx, lhs: IrExpr, rhs: IrExpr) -> Result<IrStmt, ParseError> {
        match lhs {
            IrExpr::Local(name) => Ok(IrStmt::AssignLocal { name, value: rhs }),
            IrExpr::BufferLoad { buf, index } => {
                if !*ctx.buffers.get(&buf).unwrap_or(&false) {
                    return self.err(format!(
                        "`{buf}` is read-only; declare it `device mut buffer`"
                    ));
                }
                Ok(IrStmt::AssignBuffer {
                    buf,
                    index: *index,
                    value: rhs,
                })
            }
            _ => self.err("invalid assignment target; use a local or buf[i]"),
        }
    }

    // Graphics expression grammar (same precedence as compute, richer primaries).

    fn parse_g_expr(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        self.parse_g_or(ctx)
    }

    fn parse_g_or(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_and(ctx)?;
        while self.eat(&Tok::OrOr) {
            let rhs = self.parse_g_and(ctx)?;
            lhs = bin(IrBinOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_g_and(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_cmp(ctx)?;
        while self.eat(&Tok::AndAnd) {
            let rhs = self.parse_g_cmp(ctx)?;
            lhs = bin(IrBinOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_g_cmp(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let lhs = self.parse_g_add(ctx)?;
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
        let rhs = self.parse_g_add(ctx)?;
        Ok(bin(op, lhs, rhs))
    }

    fn parse_g_add(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_mul(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => IrBinOp::Add,
                Some(Tok::Minus) => IrBinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_g_mul(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_g_mul(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_unary(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => IrBinOp::Mul,
                Some(Tok::Slash) => IrBinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_g_unary(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_g_unary(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        if self.eat(&Tok::Minus) {
            return Ok(IrExpr::Neg(Box::new(self.parse_g_unary(ctx)?)));
        }
        self.parse_g_primary(ctx)
    }

    fn parse_g_primary(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        match self.peek() {
            Some(Tok::Int(v)) => {
                let v = *v as u32;
                self.pos += 1;
                self.parse_g_postfix(ctx, IrExpr::LitU32(v))
            }
            Some(Tok::Float(v)) => {
                let v = *v as f32;
                self.pos += 1;
                self.parse_g_postfix(ctx, IrExpr::LitF32(v))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let e = self.parse_g_expr(ctx)?;
                self.expect(&Tok::RParen, "`)`")?;
                self.parse_g_postfix(ctx, e)
            }
            Some(Tok::Ident(_)) => {
                let name = self.ident()?;
                if self.peek() == Some(&Tok::LParen) {
                    let call = self.parse_g_call(ctx, &name)?;
                    return self.parse_g_postfix(ctx, call);
                }
                // push.field
                if ctx.push_param.as_deref() == Some(name.as_str()) {
                    self.expect(&Tok::Dot, "`.field` on push parameter")?;
                    let field = self.ident()?;
                    if !ctx.scalars.contains_key(&field) {
                        return self.err(format!("push struct has no field `{field}`"));
                    }
                    return Ok(IrExpr::ScalarParam(field));
                }
                // buf[i]
                if ctx.buffers.contains_key(&name) {
                    self.expect(&Tok::LBracket, "`[` to index a buffer")?;
                    let index = self.parse_g_expr(ctx)?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    return Ok(IrExpr::BufferLoad {
                        buf: name,
                        index: Box::new(index),
                    });
                }
                let base = if ctx.inputs.contains_key(&name) {
                    IrExpr::Input(name)
                } else if ctx.locals.contains_key(&name) {
                    IrExpr::Local(name)
                } else {
                    return self.err(format!("unknown identifier `{name}`"));
                };
                self.parse_g_postfix(ctx, base)
            }
            _ => self.err("expected an expression"),
        }
    }

    fn parse_g_postfix(&mut self, ctx: &GfxCtx, mut base: IrExpr) -> Result<IrExpr, ParseError> {
        while self.eat(&Tok::Dot) {
            let field = self.ident()?;
            if self.peek() == Some(&Tok::LParen) {
                if field != "extend" {
                    return self.err(format!("unknown method `.{field}()`; use `.extend()`"));
                }
                self.expect(&Tok::LParen, "`(`")?;
                let arg = self.parse_g_expr(ctx)?;
                self.expect(&Tok::RParen, "`)`")?;
                base = IrExpr::Extend {
                    base: Box::new(base),
                    scalar: Box::new(arg),
                };
            } else {
                let component = match field.as_str() {
                    "x" => 0u32,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    other => return self.err(format!("unknown field `.{other}`; use .x/.y/.z/.w")),
                };
                base = IrExpr::FieldAccess {
                    base: Box::new(base),
                    component,
                };
            }
        }
        Ok(base)
    }

    fn parse_g_call(&mut self, ctx: &GfxCtx, name: &str) -> Result<IrExpr, ParseError> {
        self.expect(&Tok::LParen, "`(`")?;
        let mut args = Vec::new();
        while !self.eat(&Tok::RParen) {
            args.push(self.parse_g_expr(ctx)?);
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated call arguments");
            }
        }
        match name {
            "dot" => {
                if args.len() != 2 {
                    return self.err("dot(a, b) takes exactly 2 args");
                }
                let mut it = args.into_iter();
                Ok(IrExpr::Dot {
                    a: Box::new(it.next().unwrap()),
                    b: Box::new(it.next().unwrap()),
                })
            }
            "f32x2" | "f32x3" | "f32x4" => {
                let dim = match name {
                    "f32x2" => 2u8,
                    "f32x3" => 3,
                    _ => 4,
                };
                Ok(IrExpr::VecConstruct { dim, args })
            }
            _ => {
                let func = gfx_builtin_from_name(name).ok_or_else(|| ParseError {
                    msg: format!("unknown function `{name}`"),
                    at: self.span(),
                })?;
                Ok(IrExpr::Builtin { func, args })
            }
        }
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
        "u32" => BuiltinFn::U32,
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
        "tanh" => BuiltinFn::Tanh,
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
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => ScalarTy::U32,
        IrExpr::BufferLoad { .. } => ScalarTy::F32,
        IrExpr::SharedLoad { .. } => ScalarTy::F32,
        IrExpr::Builtin { func, .. } => match func {
            BuiltinFn::U32 => ScalarTy::U32,
            _ => ScalarTy::F32,
        },
        IrExpr::Neg(e) => infer_ty(e, ctx),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => ScalarTy::Bool,
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
        // Graphics-only IR nodes are never produced by the compute parser.
        _ => ScalarTy::U32,
    }
}

fn gfx_builtin_from_name(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "abs" => BuiltinFn::Abs,
        "sign" => BuiltinFn::Sign,
        "exp" => BuiltinFn::Exp,
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

fn is_gfx_vec(t: GfxTy) -> bool {
    matches!(t, GfxTy::Vec2 | GfxTy::Vec3 | GfxTy::Vec4)
}

/// Infer a graphics expression's type, mirroring the SPIR-V backend's `GVal`
/// inference so a `let`'s declared type matches what the backend computes.
fn infer_gfx_ty(expr: &IrExpr, ctx: &GfxCtx) -> GfxTy {
    match expr {
        IrExpr::LitU32(_) => GfxTy::U32,
        IrExpr::LitF32(_) => GfxTy::F32,
        IrExpr::Local(n) => ctx.locals.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::ScalarParam(n) => ctx.scalars.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::Input(n) => ctx.inputs.get(n).copied().unwrap_or(GfxTy::F32),
        IrExpr::FieldAccess { .. } => GfxTy::F32,
        IrExpr::VecConstruct { dim, .. } => match dim {
            2 => GfxTy::Vec2,
            3 => GfxTy::Vec3,
            _ => GfxTy::Vec4,
        },
        IrExpr::Extend { base, .. } => match infer_gfx_ty(base, ctx) {
            GfxTy::Vec2 => GfxTy::Vec3,
            _ => GfxTy::Vec4,
        },
        IrExpr::Dot { .. } => GfxTy::F32,
        IrExpr::BufferLoad { .. } => GfxTy::F32,
        IrExpr::GlobalId(_) => GfxTy::U32,
        IrExpr::SharedLoad { .. } => GfxTy::F32,
        IrExpr::LocalId(_) | IrExpr::GroupId(_) => GfxTy::U32,
        IrExpr::Builtin { func, args } => match func {
            BuiltinFn::U32 => GfxTy::U32,
            BuiltinFn::Length => GfxTy::F32,
            _ => args
                .first()
                .map(|a| infer_gfx_ty(a, ctx))
                .unwrap_or(GfxTy::F32),
        },
        IrExpr::Neg(e) => infer_gfx_ty(e, ctx),
        IrExpr::Binary { op, lhs, rhs } => match op {
            IrBinOp::Lt
            | IrBinOp::Le
            | IrBinOp::Gt
            | IrBinOp::Ge
            | IrBinOp::Eq
            | IrBinOp::Ne
            | IrBinOp::And
            | IrBinOp::Or => GfxTy::F32, // invalid in value position; fallback
            _ => {
                let l = infer_gfx_ty(lhs, ctx);
                let r = infer_gfx_ty(rhs, ctx);
                match (l, r) {
                    (GfxTy::Mat4, x) if is_gfx_vec(x) => x,
                    (a, b) if a == b => a,
                    (GfxTy::F32 | GfxTy::U32, x) if is_gfx_vec(x) => x,
                    (x, GfxTy::F32 | GfxTy::U32) if is_gfx_vec(x) => x,
                    _ if l == GfxTy::F32 || r == GfxTy::F32 => GfxTy::F32,
                    _ => GfxTy::U32,
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
        let types: std::collections::HashMap<_, _> = m.entry.locals.iter().cloned().collect();
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
    fn atomic_add_requires_mutable_buffer_and_f32_operands() {
        let read_only = r#"
            @workgroup_size(1)
            kernel k(x: device buffer<f32>) {
                atomic_add(x, 0.0, 1.0)
            }
        "#;
        let e = parse_compute(read_only).unwrap_err();
        assert!(e.msg.contains("mutable device buffer"), "got: {}", e.msg);

        let u32_index = r#"
            @workgroup_size(1)
            kernel k(x: device mut buffer<f32>, id: global_id) {
                atomic_add(x, id.x, 1.0)
            }
        "#;
        parse_compute(u32_index).expect("u32 indices are natural buffer indices");

        let bad_value = r#"
            @workgroup_size(1)
            kernel k(idx: device buffer<f32>, x: device mut buffer<f32>, id: global_id) {
                atomic_add(x, idx[id.x], id.x)
            }
        "#;
        let e = parse_compute(bad_value).unwrap_err();
        assert!(e.msg.contains("value must be an f32"), "got: {}", e.msg);
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
