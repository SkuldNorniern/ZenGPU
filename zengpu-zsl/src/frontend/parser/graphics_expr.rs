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
        let lhs = self.parse_g_bitor(ctx)?;
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
        let rhs = self.parse_g_bitor(ctx)?;
        Ok(bin(op, lhs, rhs))
    }

    pub(super) fn parse_g_bitor(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_bitxor(ctx)?;
        while self.eat(&Tok::Pipe) {
            let rhs = self.parse_g_bitxor(ctx)?;
            lhs = bin(IrBinOp::BitOr, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_g_bitxor(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_bitand(ctx)?;
        while self.eat(&Tok::Caret) {
            let rhs = self.parse_g_bitand(ctx)?;
            lhs = bin(IrBinOp::BitXor, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_g_bitand(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_shift(ctx)?;
        while self.eat(&Tok::Amp) {
            let rhs = self.parse_g_shift(ctx)?;
            lhs = bin(IrBinOp::BitAnd, lhs, rhs);
        }
        Ok(lhs)
    }

    pub(super) fn parse_g_shift(&mut self, ctx: &GfxCtx) -> Result<IrExpr, ParseError> {
        let mut lhs = self.parse_g_add(ctx)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Shl) => IrBinOp::Shl,
                Some(Tok::Shr) => IrBinOp::Shr,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_g_add(ctx)?;
            lhs = bin(op, lhs, rhs);
        }
        Ok(lhs)
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
                // A value name before `(` is a value followed by a tuple
                // tail on the next line, not a call.
                let is_value = ctx.inputs.contains_key(&name) || ctx.locals.contains_key(&name);
                if !is_value && self.peek() == Some(&Tok::LParen) {
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

    pub(super) fn parse_g_postfix(
        &mut self,
        ctx: &GfxCtx,
        mut base: IrExpr,
    ) -> Result<IrExpr, ParseError> {
        while self.eat(&Tok::Dot) {
            let field = self.ident()?;
            if field == "extend" {
                self.expect(&Tok::LParen, "`(`")?;
                let arg = self.parse_g_expr(ctx)?;
                self.expect(&Tok::RParen, "`)`")?;
                base = IrExpr::Builtin {
                    func: BuiltinFn::Extend,
                    args: vec![base, arg],
                };
            } else {
                let Some(lanes) = swizzle_lanes(&field) else {
                    return self.err(format!(
                        "unknown field `.{field}`; use up to four of x/y/z/w or r/g/b/a"
                    ));
                };
                base = match lanes[..] {
                    [component] => IrExpr::FieldAccess {
                        base: Box::new(base),
                        component,
                    },
                    _ => IrExpr::Swizzle {
                        base: Box::new(base),
                        lanes,
                    },
                };
            }
        }
        Ok(base)
    }

    /// For the component-wise builtins, a scalar argument beside a vector
    /// one becomes that vector, `max(v, 0.0)` reading as `max(v, f32x2(0.0))`.
    /// Spelled out here so every backend sees matching argument types.
    fn spread_scalar_args(
        &mut self,
        ctx: &GfxCtx,
        func: BuiltinFn,
        args: Vec<IrExpr>,
    ) -> Vec<IrExpr> {
        if !matches!(
            func,
            BuiltinFn::Min
                | BuiltinFn::Max
                | BuiltinFn::Pow
                | BuiltinFn::Clamp
                | BuiltinFn::Mix
                | BuiltinFn::Step
                | BuiltinFn::Smoothstep
        ) {
            return args;
        }
        let tcx = ctx.type_ctx();
        let vecs = [
            (self.types.vec_id(2), 2u8),
            (self.types.vec_id(3), 3),
            (self.types.vec_id(4), 4),
        ];
        let tys: Vec<TypeId> = args
            .iter()
            .map(|a| type_of_graphics_expr(a, &tcx, &mut self.types))
            .collect();
        let width = |ty: TypeId| vecs.iter().find(|(id, _)| *id == ty).map(|&(_, n)| n);
        let Some(dim) = tys.iter().find_map(|&ty| width(ty)) else {
            return args;
        };
        args.into_iter()
            .zip(tys)
            .map(|(a, ty)| match width(ty) {
                Some(_) => a,
                None => IrExpr::Construct {
                    ty: None,
                    dim,
                    args: vec![a],
                },
            })
            .collect()
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
                let args = self.spread_scalar_args(ctx, func, args);
                Ok(IrExpr::Builtin { func, args })
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// The lanes a `.xy`/`.rgba`-style field picks, or `None` for anything else.
/// One naming set per field, as in WGSL and GLSL.
fn swizzle_lanes(field: &str) -> Option<Vec<u32>> {
    if field.is_empty() || field.len() > 4 {
        return None;
    }
    let lane = |c: char, set: &str| set.find(c).map(|i| i as u32);
    ["xyzw", "rgba"]
        .iter()
        .find_map(|set| field.chars().map(|c| lane(c, set)).collect())
}
