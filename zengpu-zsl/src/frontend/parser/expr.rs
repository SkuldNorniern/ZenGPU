use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_expr(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        self.parse_or(ctx)
    }

    pub(super) fn parse_or(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_and(ctx)?;
        while self.eat(&Tok::OrOr) {
            let rhs = self.parse_and(ctx)?;
            lhs = bin(IrBinOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_and(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_cmp(ctx)?;
        while self.eat(&Tok::AndAnd) {
            let rhs = self.parse_cmp(ctx)?;
            lhs = bin(IrBinOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_cmp(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let lhs = self.parse_bitor(ctx)?;
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
        let rhs = self.parse_bitor(ctx)?;
        Ok(bin(op, lhs, rhs))
    }

    pub(super) fn parse_bitor(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_bitxor(ctx)?;
        while self.eat(&Tok::Pipe) {
            let rhs = self.parse_bitxor(ctx)?;
            lhs = bin(IrBinOp::BitOr, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_bitxor(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_bitand(ctx)?;
        while self.eat(&Tok::Caret) {
            let rhs = self.parse_bitand(ctx)?;
            lhs = bin(IrBinOp::BitXor, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_bitand(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_shift(ctx)?;
        while self.eat(&Tok::Amp) {
            let rhs = self.parse_shift(ctx)?;
            lhs = bin(IrBinOp::BitAnd, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_shift(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_add(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Shl) => IrBinOp::Shl,
                Some(Tok::Shr) => IrBinOp::Shr,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_add(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_add(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
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

    pub(super) fn parse_mul(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
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

    pub(super) fn parse_unary(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        if self.eat(&Tok::Minus) {
            let inner = self.parse_unary(ctx)?;
            return Ok(IrExpr::Neg(Box::new(inner)));
        }
        self.parse_primary(ctx)
    }

    pub(super) fn parse_primary(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
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
                            other => {
                                return self.err(format!(
                                    "index builtin has no field `.{other}`; use .x/.y/.z"
                                ));
                            }
                        };
                        return Ok(if name == "local_id" {
                            IrExpr::LocalId(comp)
                        } else {
                            IrExpr::GroupId(comp)
                        });
                    }
                    return self.parse_call(ctx, &name);
                }
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
                if self.eat(&Tok::LBracket) {
                    let index = self.parse_expr(ctx)?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    if ctx.buffers.contains_key(&name) {
                        return Ok(IrExpr::BufferLoad {
                            buf: name,
                            index: Box::new(index),
                        });
                    }
                    if ctx.shared.contains_key(&name) {
                        let idx_ty = crate::ir::typeck::type_of_compute_expr(
                            &index,
                            &ctx.type_ctx(),
                            &mut self.types,
                        );
                        if idx_ty != self.types.u32_id() {
                            return self.err("shared array index must be u32");
                        }
                        return Ok(IrExpr::SharedLoad {
                            name,
                            index: Box::new(index),
                        });
                    }
                    return self.err(format!(
                        "`{name}` is not a buffer or shared array; cannot index"
                    ));
                }
                if ctx.locals.contains_key(&name) {
                    Ok(IrExpr::Local(name))
                } else {
                    self.err(format!("unknown identifier `{name}`"))
                }
            }
            _ => self.err("expected an expression"),
        }
    }

    pub(super) fn parse_call(&mut self, ctx: &Ctx, name: &str) -> Result<IrExpr, ParseError> {
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
        if func == BuiltinFn::MulAddF32 && args.len() != 3 {
            return self.err("mul_add_f32(a, b, acc) takes 3 args");
        }
        Ok(IrExpr::Builtin { func, args })
    }
}
