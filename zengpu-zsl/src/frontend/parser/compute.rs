use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_shader(&mut self) -> Result<Shader, ParseError> {
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

        // Helper `fn` definitions (compute only; each may call earlier ones).
        while self.at_kw("fn") {
            self.parse_fn_item()?;
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

    pub(super) fn parse_kernel_rest(&mut self, local_size: [u32; 3]) -> Result<Module, ParseError> {
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

        let params = params
            .into_iter()
            .map(|p| Param {
                name: p.name,
                kind: match p.kind {
                    PendingParamKind::Buffer { elem, mutability } => {
                        ParamKind::Buffer { elem, mutability }
                    }
                    PendingParamKind::Scalar(ty) => ParamKind::Scalar(ty),
                },
            })
            .collect();

        let shared = ctx
            .shared_order
            .iter()
            .map(|name| SharedDecl {
                name: name.clone(),
                elem: ctx.shared[name].elem,
                len: ctx.shared[name].len,
            })
            .collect();

        if self.pos != self.toks.len() {
            return self.err("unexpected trailing tokens after kernel");
        }

        let locals = ctx.locals_order;

        let body: Vec<IrStmt> = body
            .iter()
            .map(|s| resolve_construct_types_stmt(s, &mut self.types))
            .collect();

        Ok(Module {
            entry: Entry {
                kind: EntryKind::Compute { local_size },
                params,
                shared,
                locals,
                body,
            },
            types: std::mem::replace(&mut self.types, crate::ir::arena::TypeArena::new()),
        })
    }

    pub(super) fn parse_workgroup_size(&mut self) -> Result<[u32; 3], ParseError> {
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

    pub(super) fn parse_push_block(&mut self) -> Result<(), ParseError> {
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

    /// `fn name(("out")? ident ":" scalar, ...) block` — scalar-only helper,
    /// no buffers/shared/global_id/push. May call any `fn` declared earlier.
    pub(super) fn parse_fn_item(&mut self) -> Result<(), ParseError> {
        self.eat_kw("fn");
        let name = self.ident()?;
        if self.functions.contains_key(&name) || self.pushes.contains_key(&name) {
            return self.err(format!("duplicate symbol `{name}`"));
        }
        self.expect(&Tok::LParen, "`(` after fn name")?;
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
        let mut params = Vec::new();
        while !self.eat(&Tok::RParen) {
            let is_out = self.eat_kw("out");
            let pname = self.ident()?;
            if ctx.locals.contains_key(&pname) {
                return self.err(format!("duplicate parameter `{pname}`"));
            }
            self.expect(&Tok::Colon, "`:` after fn parameter name")?;
            let ty = self.parse_scalar_type()?;
            ctx.locals.insert(pname.clone(), ty);
            params.push(FnParamDef {
                name: pname,
                is_out,
                ty,
            });
            self.eat(&Tok::Comma); // optional/trailing
            if self.peek().is_none() {
                return self.err("unterminated fn parameter list");
            }
        }
        let body = self.parse_block(&mut ctx)?;
        self.functions.insert(
            name,
            FunctionDef {
                params,
                locals: ctx.locals_order,
                body,
            },
        );
        Ok(())
    }

    pub(super) fn parse_params(&mut self, ctx: &mut Ctx) -> Result<Vec<PendingParam>, ParseError> {
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

    pub(super) fn parse_block(&mut self, ctx: &mut Ctx) -> Result<Vec<IrStmt>, ParseError> {
        self.expect(&Tok::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        while !self.eat(&Tok::RBrace) {
            if self.peek().is_none() {
                return self.err("unterminated block");
            }
            if self.eat_kw("shared") {
                self.parse_shared(ctx)?;
            } else {
                stmts.extend(self.parse_stmt(ctx)?);
            }
        }
        Ok(stmts)
    }

    pub(super) fn parse_shared(&mut self, ctx: &mut Ctx) -> Result<(), ParseError> {
        let name = self.ident()?;
        self.expect(&Tok::Colon, "`:` after shared name")?;
        if !self.eat_kw("array") {
            return self.err("shared declarations require `array<f32, N>`");
        }
        self.expect(&Tok::Lt, "`<` in array<f32, N>")?;
        let elem = self.parse_scalar_type()?;
        if elem != self.types.f32_id() {
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
        if ctx.shared.contains_key(&name)
            || ctx.buffers.contains_key(&name)
            || ctx.locals.contains_key(&name)
        {
            return self.err(format!("duplicate symbol `{name}`"));
        }
        ctx.shared_order.push(name.clone());
        ctx.shared
            .insert(name.clone(), PendingShared { name, elem, len });
        Ok(())
    }

    pub(super) fn parse_stmt(&mut self, ctx: &mut Ctx) -> Result<Vec<IrStmt>, ParseError> {
        if self.eat_kw("let") {
            return Ok(vec![self.parse_let(ctx)?]);
        }
        if self.eat_kw("if") {
            return Ok(vec![self.parse_if(ctx)?]);
        }
        if self.eat_kw("for") {
            return Ok(vec![self.parse_for(ctx)?]);
        }
        if self.eat_kw("barrier") {
            self.expect(&Tok::LParen, "`(` after barrier")?;
            self.expect(&Tok::RParen, "`)` after barrier(")?;
            return Ok(vec![IrStmt::Barrier]);
        }
        if self.eat_kw("atomic_add") {
            self.expect(&Tok::LParen, "`(` after atomic_add")?;
            let buf = self.ident()?;
            self.expect(&Tok::Comma, "`,` after atomic_add buffer")?;
            let index = self.parse_expr(ctx)?;
            self.expect(&Tok::Comma, "`,` after atomic_add index")?;
            let value = self.parse_expr(ctx)?;
            self.expect(&Tok::RParen, "`)` after atomic_add arguments")?;
            let Some(buffer) = ctx.buffers.get(&buf).copied() else {
                return self.err(format!("`{buf}` is not a device buffer"));
            };
            if !buffer.writable {
                return self.err(format!(
                    "`{buf}` is not a mutable device buffer; declare it `device mut buffer`"
                ));
            }
            let f32_id = self.types.f32_id();
            if buffer.elem != f32_id {
                return self.err("atomic_add currently requires buffer<f32>");
            }
            let index_ty =
                crate::ir::typeck::type_of_compute_expr(&index, &ctx.type_ctx(), &mut self.types);
            let u32_id = self.types.u32_id();
            if index_ty != u32_id && index_ty != f32_id {
                return self.err("atomic_add index must be a u32- or f32-typed expression");
            }
            if crate::ir::typeck::type_of_compute_expr(&value, &ctx.type_ctx(), &mut self.types)
                != f32_id
            {
                return self.err("atomic_add value must be an f32-typed expression");
            }
            return Ok(vec![IrStmt::AtomicAdd { buf, index, value }]);
        }
        // A call to a user-defined `fn`, always inlined here (see
        // `parse_fn_call_stmt`); recognized before falling into the general
        // expression parser, since it isn't an expression (it returns via
        // `out` parameters, if any, not a value).
        if let Some(Tok::Ident(name)) = self.peek() {
            let name = name.clone();
            if self.functions.contains_key(&name) {
                return self.parse_fn_call_stmt(&name, ctx);
            }
        }
        // Expression statement or assignment.
        let lhs = self.parse_expr(ctx)?;
        if self.eat(&Tok::Eq) {
            let rhs = self.parse_expr(ctx)?;
            let stmt = match lhs {
                IrExpr::Local(name) => IrStmt::AssignLocal { name, value: rhs },
                IrExpr::BufferLoad { buf, index } => {
                    let Some(buffer) = ctx.buffers.get(&buf).copied() else {
                        return self.err(format!("`{buf}` is not a device buffer"));
                    };
                    if !buffer.writable {
                        return self.err(format!(
                            "`{buf}` is read-only; declare it `device mut buffer`"
                        ));
                    }
                    // `bf16`/`f16` are storage-only: a value assigned into one
                    // of these buffers is f32-typed (there is no bf16/f16-
                    // typed ZSL expression), so the expected type narrows to
                    // f32.
                    let bf16_id = self.types.bf16_id();
                    let f16_id = self.types.f16_id();
                    let f32_id = self.types.f32_id();
                    let expected = if buffer.elem == bf16_id || buffer.elem == f16_id {
                        f32_id
                    } else {
                        buffer.elem
                    };
                    if crate::ir::typeck::type_of_compute_expr(
                        &rhs,
                        &ctx.type_ctx(),
                        &mut self.types,
                    ) != expected
                    {
                        return self.err(format!(
                            "assignment to buffer<{}> has incompatible value type",
                            buffer_elem_name(buffer.elem, &self.types)
                        ));
                    }
                    IrStmt::AssignBuffer {
                        buf,
                        index: *index,
                        value: rhs,
                    }
                }
                IrExpr::SharedLoad { name, index } => {
                    if !ctx.shared.contains_key(&name) {
                        return self.err(format!("`{name}` is not a declared shared array"));
                    }
                    IrStmt::AssignShared {
                        name,
                        index: *index,
                        value: rhs,
                    }
                }
                _ => {
                    return self
                        .err("invalid assignment target; use a local, buffer[i], or shared[i]");
                }
            };
            Ok(vec![stmt])
        } else {
            Ok(vec![IrStmt::Eval(lhs)])
        }
    }

    /// Inline a call to a user-defined `fn` at the current statement position:
    /// bind each `in` argument to a fresh local, alias each `out` argument's
    /// name directly to the caller-supplied local, rename the function's own
    /// locals fresh, and splice the (renamed) body in as the call's expansion.
    pub(super) fn parse_fn_call_stmt(
        &mut self,
        name: &str,
        ctx: &mut Ctx,
    ) -> Result<Vec<IrStmt>, ParseError> {
        self.pos += 1; // the function name identifier, already peeked
        self.expect(&Tok::LParen, "`(` after function name")?;
        let params = self.functions[name].params.clone();
        self.call_counter += 1;
        let call_id = self.call_counter;
        let mut rename: HashMap<String, String> = HashMap::new();
        let mut prelude: Vec<IrStmt> = Vec::new();
        for (index, param) in params.iter().enumerate() {
            if index > 0 {
                self.expect(&Tok::Comma, "`,` between function call arguments")?;
            }
            if param.is_out {
                let arg_name = self.ident()?;
                let Some(&arg_ty) = ctx.locals.get(&arg_name) else {
                    return self.err(format!(
                        "`{arg_name}` is not a declared local; `out` arguments must be a pre-declared local variable"
                    ));
                };
                if arg_ty != param.ty {
                    return self.err(format!(
                        "`out` argument `{arg_name}` does not match parameter `{}`'s type",
                        param.name
                    ));
                }
                rename.insert(param.name.clone(), arg_name);
            } else {
                let arg_expr = self.parse_expr(ctx)?;
                if crate::ir::typeck::type_of_compute_expr(
                    &arg_expr,
                    &ctx.type_ctx(),
                    &mut self.types,
                ) != param.ty
                {
                    return self.err(format!(
                        "argument for parameter `{}` has the wrong type",
                        param.name
                    ));
                }
                let fresh = format!("__call{call_id}_{}", param.name);
                ctx.locals.insert(fresh.clone(), param.ty);
                ctx.locals_order.push((fresh.clone(), param.ty));
                prelude.push(IrStmt::Let {
                    name: fresh.clone(),
                    init: arg_expr,
                });
                rename.insert(param.name.clone(), fresh);
            }
        }
        self.expect(&Tok::RParen, "`)` after function call arguments")?;

        let def_locals = self.functions[name].locals.clone();
        for (local_name, ty) in &def_locals {
            let fresh = format!("__call{call_id}_{local_name}");
            ctx.locals.insert(fresh.clone(), *ty);
            ctx.locals_order.push((fresh.clone(), *ty));
            rename.insert(local_name.clone(), fresh);
        }

        let inlined: Vec<IrStmt> = self.functions[name]
            .body
            .iter()
            .map(|stmt| rename_stmt(stmt, &rename))
            .collect();
        prelude.extend(inlined);
        Ok(prelude)
    }

    pub(super) fn parse_let(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
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
        let ty = match annot {
            Some(ty) => ty,
            None => {
                crate::ir::typeck::type_of_compute_expr(&init, &ctx.type_ctx(), &mut self.types)
            }
        };
        ctx.locals.insert(name.clone(), ty);
        ctx.locals_order.push((name.clone(), ty));
        Ok(IrStmt::Let { name, init })
    }

    pub(super) fn parse_if(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
        let cond = self.parse_expr(ctx)?;
        let then = self.parse_block(ctx)?;
        let else_ = if self.eat_kw("else") {
            Some(self.parse_block(ctx)?)
        } else {
            None
        };
        Ok(IrStmt::If { cond, then, else_ })
    }

    pub(super) fn parse_for(&mut self, ctx: &mut Ctx) -> Result<IrStmt, ParseError> {
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
        let u32_id = self.types.u32_id();
        ctx.locals.insert(var.clone(), u32_id);
        ctx.locals_order.push((var.clone(), u32_id));
        let body = self.parse_block(ctx)?;
        Ok(IrStmt::For { var, lo, hi, body })
    }

    // ── Expressions (precedence climbing) ──────────────────────────────────────
}
