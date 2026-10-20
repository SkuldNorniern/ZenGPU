use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_graphics_rest(
        &mut self,
        is_fragment: bool,
    ) -> Result<GraphicsModule, ParseError> {
        let mut ctx = GfxCtx {
            is_fragment,
            inputs: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            buffers: HashMap::new(),
            textures: None,
            locals: HashMap::new(),
            locals_order: Vec::new(),
            arrays: HashSet::new(),
        };
        let (inputs, scalar_params, buf_params) = self.parse_gfx_params(&mut ctx, is_fragment)?;

        self.expect(&Tok::Arrow, "`->` return type")?;
        let (varyings, outputs) = if is_fragment {
            let slots = self.parse_ret_slots()?;
            (Vec::new(), self.fragment_outputs(&slots)?)
        } else {
            let ret_types = self.parse_ret_types()?;
            if ret_types[0] != self.types.vec_id(4) {
                return self.err("first return type must be f32x4 (position)");
            }
            (ret_types[1..].to_vec(), Vec::new())
        };
        let expected = if is_fragment {
            outputs.len()
        } else {
            1 + varyings.len()
        };

        let (body, ret) = self.parse_gfx_body(&mut ctx, expected)?;
        if self.pos != self.toks.len() {
            return self.err("unexpected trailing tokens after entry point");
        }

        let inputs: Vec<GfxInput> = inputs
            .into_iter()
            .map(|i| GfxInput {
                name: i.name,
                slot: i.slot,
                ty: i.ty,
            })
            .collect();
        let scalar_params: Vec<GfxScalar> = scalar_params
            .into_iter()
            .map(|s| GfxScalar {
                name: s.name,
                ty: s.ty,
            })
            .collect();
        let locals = ctx.locals_order;

        let body: Vec<IrStmt> = body
            .iter()
            .map(|s| resolve_construct_types_stmt(s, &mut self.types))
            .collect();
        let ret: Vec<IrExpr> = ret
            .iter()
            .map(|e| resolve_construct_types_expr(e, &mut self.types))
            .collect();

        Ok(GraphicsModule {
            entry: GraphicsEntry {
                is_fragment,
                inputs,
                scalar_params,
                buf_params,
                textures: ctx.textures.clone(),
                varyings,
                locals,
                body,
                ret,
                outputs,
            },
            types: mem::replace(&mut self.types, TypeArena::new()),
        })
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn parse_gfx_params(
        &mut self,
        ctx: &mut GfxCtx,
        is_fragment: bool,
    ) -> Result<(Vec<PendingGfxInput>, Vec<PendingGfxScalar>, Vec<String>), ParseError> {
        self.expect(&Tok::LParen, "`(` after entry name")?;
        let mut inputs = Vec::new();
        let mut scalar_params = Vec::new();
        let mut buf_params = Vec::new();
        while !self.eat(&Tok::RParen) {
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
                if ty == self.types.mat4_id() {
                    return self.err("input type must be f32/u32/f32x2/f32x3/f32x4");
                }
                ctx.inputs.insert(name.clone(), ty);
                inputs.push(PendingGfxInput {
                    name,
                    slot: GfxSlot::Location(loc),
                    ty,
                });
            } else if let Some(builtin) = self.peek_builtin_input() {
                self.pos += 1;
                if builtin.in_fragment() != is_fragment {
                    let stage = if is_fragment { "fragment" } else { "vertex" };
                    return self.err(format!(
                        "this builtin input is not available in a {stage} shader"
                    ));
                }
                let ty = match builtin {
                    GfxBuiltin::FragCoord => self.types.vec_id(4),
                    GfxBuiltin::VertexIndex | GfxBuiltin::InstanceIndex => self.types.u32_id(),
                };
                ctx.inputs.insert(name.clone(), ty);
                inputs.push(PendingGfxInput {
                    name,
                    slot: GfxSlot::Builtin(builtin),
                    ty,
                });
            } else if self.eat_kw("textures") {
                if ctx.textures.is_some() {
                    return self.err("only one `textures` parameter is supported");
                }
                ctx.textures = Some(name);
            } else if self.eat_kw("device") {
                let writable = self.eat_kw("mut");
                if !self.eat_kw("buffer") {
                    return self.err("expected `buffer`");
                }
                self.expect(&Tok::Lt, "`<`")?;
                let elem = self.parse_any_type()?;
                self.expect(&Tok::Gt, "`>`")?;
                if elem != self.types.f32_id() {
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
                let allowed = [
                    self.types.f32_id(),
                    self.types.u32_id(),
                    self.types.vec_id(2),
                    self.types.vec_id(4),
                    self.types.mat4_id(),
                ];
                for (field, ty) in fields {
                    if !allowed.contains(&ty) {
                        return self.err(format!(
                            "graphics push field `{field}` must be f32/u32/f32x2/f32x4/mat4x4"
                        ));
                    }
                    ctx.scalars.insert(field.clone(), ty);
                    scalar_params.push(PendingGfxScalar { name: field, ty });
                }
            }
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated parameter list");
            }
        }
        Ok((inputs, scalar_params, buf_params))
    }

    /// Location of a fragment output attachment: its `@location`, else its tuple index.
    ///
    /// Dual-source blend uses sources 0 and 1 of location 0.
    fn fragment_outputs(
        &self,
        slots: &[(TypeId, Option<u32>, u32)],
    ) -> Result<Vec<GfxOutput>, ParseError> {
        let outputs: Vec<GfxOutput> = slots
            .iter()
            .enumerate()
            .map(|(i, &(ty, location, blend_src))| GfxOutput {
                ty,
                location: location.unwrap_or(i as u32),
                blend_src,
            })
            .collect();
        if outputs.iter().any(|o| o.blend_src != 0) {
            let pair = match &outputs[..] {
                [a, b] => (a.location, a.blend_src, b.location, b.blend_src) == (0, 0, 0, 1),
                _ => false,
            };
            if !pair {
                return self.err(
                    "dual-source blending needs two outputs: `@location(0) @blend_src(0)` \
                     then `@location(0) @blend_src(1)`",
                );
            }
        }
        for (i, a) in outputs.iter().enumerate() {
            let clash = outputs[..i]
                .iter()
                .any(|b| (b.location, b.blend_src) == (a.location, a.blend_src));
            if clash {
                return self.err(format!("two outputs write location {}", a.location));
            }
        }
        Ok(outputs)
    }

    pub(super) fn parse_gfx_body(
        &mut self,
        ctx: &mut GfxCtx,
        expected: usize,
    ) -> Result<(Vec<IrStmt>, Vec<IrExpr>), ParseError> {
        self.expect(&Tok::LBrace, "`{`")?;
        let mut body = Vec::new();
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
                self.eat(&Tok::Semi);
                continue;
            }
            if self.eat_kw("if") {
                body.push(self.parse_gfx_if(ctx)?);
                continue;
            }
            if self.eat_kw("discard") {
                body.push(self.gfx_discard(ctx)?);
                continue;
            }
            let tail = if self.peek() == Some(&Tok::LParen) {
                self.parse_gfx_tail(ctx)?
            } else {
                let e = self.parse_g_expr(ctx)?;
                if self.eat(&Tok::Eq) {
                    let rhs = self.parse_g_expr(ctx)?;
                    body.push(self.gfx_assign(ctx, e, rhs)?);
                    self.eat(&Tok::Semi);
                    continue;
                }
                vec![e]
            };
            self.expect(&Tok::RBrace, "`}` after tail expression")?;
            if tail.len() != expected {
                return self.err(format!(
                    "return has {} value(s), expected {expected}",
                    tail.len()
                ));
            }
            return Ok((body, tail));
        }
    }

    fn peek_builtin_input(&self) -> Option<GfxBuiltin> {
        match self.peek() {
            Some(Tok::Ident(name)) => GfxBuiltin::from_name(name),
            _ => None,
        }
    }

    pub(super) fn parse_gfx_tail(&mut self, ctx: &GfxCtx) -> Result<Vec<IrExpr>, ParseError> {
        let save = self.pos;
        self.pos += 1;
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
            self.pos = save;
            Ok(vec![self.parse_g_expr(ctx)?])
        }
    }

    pub(super) fn parse_gfx_stmts(&mut self, ctx: &mut GfxCtx) -> Result<Vec<IrStmt>, ParseError> {
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
                self.eat(&Tok::Semi);
                continue;
            }
            if self.eat_kw("if") {
                stmts.push(self.parse_gfx_if(ctx)?);
                continue;
            }
            if self.eat_kw("discard") {
                stmts.push(self.gfx_discard(ctx)?);
                continue;
            }
            let e = self.parse_g_expr(ctx)?;
            self.expect(&Tok::Eq, "`=` (block statements must be assignments)")?;
            let rhs = self.parse_g_expr(ctx)?;
            stmts.push(self.gfx_assign(ctx, e, rhs)?);
            self.eat(&Tok::Semi);
        }
        Ok(stmts)
    }

    pub(super) fn parse_gfx_let(&mut self, ctx: &mut GfxCtx) -> Result<IrStmt, ParseError> {
        let name = self.ident()?;
        let annot = if self.eat(&Tok::Colon) {
            Some(self.parse_any_type()?)
        } else {
            None
        };
        self.expect(&Tok::Eq, "`=` in let binding")?;
        let init = self.parse_g_expr(ctx)?;
        if matches!(init, IrExpr::ArrayLit(_)) {
            if annot.is_some() {
                return self.err("an array's type comes from its elements");
            }
            ctx.arrays.insert(name.clone());
        }
        let ty = match annot {
            Some(ty) => ty,
            None => type_of_graphics_expr(&init, &ctx.type_ctx(), &mut self.types),
        };
        ctx.locals.insert(name.clone(), ty);
        ctx.locals_order.push((name.clone(), ty));
        Ok(IrStmt::Let { name, init })
    }

    fn gfx_discard(&mut self, ctx: &GfxCtx) -> Result<IrStmt, ParseError> {
        if !ctx.is_fragment {
            return self.err("`discard` is only available in fragment shaders");
        }
        self.eat(&Tok::Semi);
        Ok(IrStmt::Discard)
    }

    pub(super) fn parse_gfx_if(&mut self, ctx: &mut GfxCtx) -> Result<IrStmt, ParseError> {
        let cond = self.parse_g_expr(ctx)?;
        let then = self.parse_gfx_stmts(ctx)?;
        let else_ = if !self.eat_kw("else") {
            None
        } else if self.eat_kw("if") {
            Some(vec![self.parse_gfx_if(ctx)?])
        } else {
            Some(self.parse_gfx_stmts(ctx)?)
        };
        Ok(IrStmt::If { cond, then, else_ })
    }

    pub(super) fn gfx_assign(
        &self,
        ctx: &GfxCtx,
        lhs: IrExpr,
        rhs: IrExpr,
    ) -> Result<IrStmt, ParseError> {
        match lhs {
            IrExpr::Local(name) if !ctx.arrays.contains(&name) => {
                Ok(IrStmt::AssignLocal { name, value: rhs })
            }
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
}
