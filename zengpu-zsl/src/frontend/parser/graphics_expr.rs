use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_g_expr(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        self.parse_g_or(ctx)
    }

    pub(super) fn parse_g_or(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_and(ctx)?;
        while self.eat(&Tok::OrOr) {
            let rhs = self.parse_g_and(ctx)?;
            lhs = bin(IrBinOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_g_and(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_cmp(ctx)?;
        while self.eat(&Tok::AndAnd) {
            let rhs = self.parse_g_cmp(ctx)?;
            lhs = bin(IrBinOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_g_cmp(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
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

    pub(super) fn parse_g_add(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
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

    pub(super) fn parse_g_mul(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
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

    pub(super) fn parse_g_unary(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        if self.eat(&Tok::Minus) {
            return Ok(IrExpr::Neg(Box::new(self.parse_g_unary(ctx)?)));
        }
        self.parse_g_primary(ctx)
    }

    pub(super) fn parse_g_primary(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
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
                if ctx.push_param.as_deref() == Some(name.as_str()) {
                    self.expect(&Tok::Dot, "`.field` on push parameter")?;
                    let field = self.ident()?;
                    if !ctx.scalars.contains_key(&field) {
                        return self.err(format!("push struct has no field `{field}`"));
                    }
                    return Ok(IrExpr::ScalarParam(field));
                }
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

    pub(super) fn parse_g_postfix(
        &mut self,
        ctx: &GfxCtx,
        mut base: IrExpr,
    ) -> Result<IrExpr, ParseError> {
        while self.eat(&Tok::Dot) {
            let field = self.ident()?;
            if self.peek() == Some(&Tok::LParen) {
                if field != "extend" {
                    return self.err(format!("unknown method `.{field}()`; use `.extend()`"));
                }
                self.expect(&Tok::LParen, "`(`")?;
                let arg = self.parse_g_expr(ctx)?;
                self.expect(&Tok::RParen, "`)`")?;
                base = IrExpr::Builtin {
                    func: BuiltinFn::Extend,
                    args: vec![base, arg],
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

    pub(super) fn parse_g_call(&mut self, ctx: &GfxCtx, name: &str) -> Result<IrExpr, ParseError> {
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
                Ok(IrExpr::Builtin {
                    func: BuiltinFn::Dot,
                    args,
                })
            }
            "f32x2" | "f32x3" | "f32x4" => {
                let dim = match name {
                    "f32x2" => 2u8,
                    "f32x3" => 3,
                    _ => 4,
                };
                Ok(IrExpr::Construct {
                    ty: None,
                    dim,
                    args,
                })
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

