use super::*;
use crate::ir::Occupancy;

impl<'a> Parser<'a> {
    pub(super) fn parse_shader(&mut self) -> Result<Shader, ParseError> {
        if self.eat_kw("module") {
            self.ident()?;
            while self.eat(&Tok::Dot) {
                self.ident()?;
            }
        }

        while self.at_kw("push") {
            self.parse_push_block()?;
        }

        while self.at_kw("fn") {
            self.parse_fn_item()?;
        }

        let mut local_size = [1u32, 1, 1];
        let mut occupancy = Occupancy::default();
        let mut backends: Vec<Backend> = Vec::new();
        while self.eat(&Tok::At) {
            let name = self.ident()?;
            match name.as_str() {
                "workgroup_size" => {
                    self.expect(&Tok::LParen, "`(` after @workgroup_size")?;
                    local_size = self.parse_workgroup_size()?;
                    self.expect(&Tok::RParen, "`)`")?;
                }
                "launch_bounds" => {
                    self.expect(&Tok::LParen, "`(` after @launch_bounds")?;
                    occupancy.max_threads = Some(self.parse_hint_int("@launch_bounds")?);
                    self.expect(&Tok::RParen, "`)` closing @launch_bounds")?;
                }
                "min_blocks" => {
                    self.expect(&Tok::LParen, "`(` after @min_blocks")?;
                    occupancy.min_blocks = Some(self.parse_hint_int("@min_blocks")?);
                    self.expect(&Tok::RParen, "`)` closing @min_blocks")?;
                }
                "waves_per_eu" => {
                    self.expect(&Tok::LParen, "`(` after @waves_per_eu")?;
                    let min = self.parse_hint_int("@waves_per_eu")?;
                    let max = if self.eat(&Tok::Comma) {
                        Some(self.parse_hint_int("@waves_per_eu")?)
                    } else {
                        None
                    };
                    if max.is_some_and(|m| m < min) {
                        return self.err("@waves_per_eu max must be >= min");
                    }
                    occupancy.waves_per_eu = Some((min, max));
                    self.expect(&Tok::RParen, "`)` closing @waves_per_eu")?;
                }
                "backends" => {
                    self.expect(&Tok::LParen, "`(` after @backends")?;
                    loop {
                        let b = self.ident()?;
                        let Some(backend) = Backend::from_name(&b) else {
                            return self.err(format!(
                                "unknown backend `{b}` in @backends; expected spirv, hip, cuda or metal"
                            ));
                        };
                        if !backends.contains(&backend) {
                            backends.push(backend);
                        }
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` closing @backends")?;
                }
                other => {
                    return self.err(format!(
                        "unknown attribute `@{other}`; expected @workgroup_size, @launch_bounds, @waves_per_eu, @min_blocks or @backends"
                    ));
                }
            }
        }

        if let Some(max) = occupancy.max_threads {
            let threads = local_size.iter().product::<u32>();
            if max < threads {
                return self.err(format!(
                    "@launch_bounds({max}) is smaller than the workgroup size ({threads} threads)"
                ));
            }
        }

        let shader = if self.eat_kw("kernel") {
            let _name = self.ident()?;
            Shader::Compute(self.parse_kernel_rest(local_size, occupancy)?)
        } else if self.eat_kw("vertex") {
            let _name = self.ident()?;
            Shader::Graphics(self.parse_graphics_rest(false)?)
        } else if self.eat_kw("fragment") {
            let _name = self.ident()?;
            Shader::Graphics(self.parse_graphics_rest(true)?)
        } else {
            return self.err("expected `kernel`, `vertex`, or `fragment`");
        };
        Ok(match shader {
            Shader::Compute(mut m) => {
                m.backends = backends;
                Shader::Compute(m)
            }
            Shader::Graphics(mut m) => {
                m.backends = backends;
                Shader::Graphics(m)
            }
        })
    }

    pub(super) fn parse_kernel_rest(
        &mut self,
        local_size: [u32; 3],
        occupancy: Occupancy,
    ) -> Result<Module, ParseError> {
        let mut ctx = Ctx {
            buffers: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            id_param: None,
            locals: HashMap::new(),
            locals_order: Vec::new(),
            shared: HashMap::new(),
            shared_order: Vec::new(),
            textures: HashMap::new(),
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
                    PendingParamKind::Texture => ParamKind::Texture,
                    PendingParamKind::Image => ParamKind::Image,
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
                view_of: ctx.shared[name].view_of.clone(),
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
                kind: EntryKind::Compute {
                    local_size,
                    occupancy,
                },
                params,
                shared,
                locals,
                body,
            },
            types: mem::replace(&mut self.types, TypeArena::new()),
            backends: Vec::new(),
        })
    }

    fn parse_hint_int(&mut self, attr: &str) -> Result<u32, ParseError> {
        match self.peek() {
            Some(Tok::Int(v)) if *v >= 1 && *v <= u32::MAX as u64 => {
                let v = *v as u32;
                self.pos += 1;
                Ok(v)
            }
            _ => self.err(format!("{attr} expects a positive integer")),
        }
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
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated push block");
            }
        }
        self.pushes.insert(name, fields);
        Ok(())
    }

    /// Scalar-only helper: no buffers, shared, global_id or push. May call earlier `fn`s.
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
            textures: HashMap::new(),
        };
        let mut params = Vec::new();
        while !self.eat(&Tok::RParen) {
            let is_out = self.eat_kw("out");
            let pname = self.ident()?;
            if ctx.locals.contains_key(&pname) {
                return self.err(format!("duplicate parameter `{pname}`"));
            }
            self.expect(&Tok::Colon, "`:` after fn parameter name")?;
            let ty = if self.at_kw("frag") {
                self.parse_frag_type()?
            } else {
                self.parse_scalar_type()?
            };
            ctx.locals.insert(pname.clone(), ty);
            params.push(FnParamDef {
                name: pname,
                is_out,
                ty,
            });
            self.eat(&Tok::Comma);
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
            self.eat(&Tok::Comma);
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
            } else if self.eat_kw("view") {
                self.parse_view(ctx)?;
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
            return self.err("shared declarations require `array<T, N>`");
        }
        self.expect(&Tok::Lt, "`<` in array<T, N>")?;
        let elem = self.parse_buffer_elem_type()?;
        self.expect(&Tok::Comma, "`,` in array<T, N>")?;
        let len = match self.peek() {
            Some(Tok::Int(v)) if *v > 0 && *v <= u32::MAX as u64 => {
                let len = *v as u32;
                self.pos += 1;
                len
            }
            _ => return self.err("shared array length must be a positive u32 literal"),
        };
        self.expect(&Tok::Gt, "`>` in array<T, N>")?;
        if ctx.shared.contains_key(&name)
            || ctx.buffers.contains_key(&name)
            || ctx.locals.contains_key(&name)
        {
            return self.err(format!("duplicate symbol `{name}`"));
        }
        ctx.shared_order.push(name.clone());
        ctx.shared.insert(
            name.clone(),
            PendingShared {
                elem,
                len,
                view_of: None,
            },
        );
        Ok(())
    }

    /// `view name: array<T> = base` reinterprets `base`'s bytes as `T` (u32, u32x2, u32x4). Length
    /// is base bytes / sizeof(T), which must divide evenly.
    pub(super) fn parse_view(&mut self, ctx: &mut Ctx) -> Result<(), ParseError> {
        let name = self.ident()?;
        self.expect(&Tok::Colon, "`:` after view name")?;
        if !self.eat_kw("array") {
            return self.err("view declarations require `array<T>`");
        }
        self.expect(&Tok::Lt, "`<` in array<T>")?;
        let elem = self.parse_buffer_elem_type()?;
        self.expect(&Tok::Gt, "`>` in array<T>")?;
        self.expect(&Tok::Eq, "`=` in view declaration")?;
        let base = self.ident()?;
        let Some(base_decl) = ctx.shared.get(&base) else {
            return self.err(format!("`{base}` is not a shared array"));
        };
        if base_decl.view_of.is_some() {
            return self.err(format!(
                "`{base}` is a view; views of views are not supported"
            ));
        }
        let (base_elem, base_len) = (base_decl.elem, base_decl.len);
        let Some(base_bytes) = storage_byte_size(&self.types, base_elem) else {
            return self.err(format!(
                "`{base}` has an element type that cannot be viewed"
            ));
        };
        let view_bytes = match u32_vec_lanes(&self.types, elem) {
            Some(lanes) => 4 * u32::from(lanes),
            None if elem == self.types.u32_id() => 4,
            None => return self.err("views are array<u32>, array<u32x2> or array<u32x4>"),
        };
        let total = u64::from(base_len) * u64::from(base_bytes);
        if total % u64::from(view_bytes) != 0 {
            return self.err(format!(
                "view `{name}`: the {total}-byte base `{base}` is not a whole number of {view_bytes}-byte elements"
            ));
        }
        if ctx.shared.contains_key(&name)
            || ctx.buffers.contains_key(&name)
            || ctx.locals.contains_key(&name)
        {
            return self.err(format!("duplicate symbol `{name}`"));
        }
        ctx.shared_order.push(name.clone());
        ctx.shared.insert(
            name,
            PendingShared {
                elem,
                len: (total / u64::from(view_bytes)) as u32,
                view_of: Some(base),
            },
        );
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
        if self.eat_kw("frag_store") {
            return Ok(vec![self.parse_frag_store(ctx)?]);
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
            let index_ty = type_of_compute_expr(&index, &ctx.type_ctx(), &mut self.types);
            let u32_id = self.types.u32_id();
            if index_ty != u32_id && index_ty != f32_id {
                return self.err("atomic_add index must be a u32- or f32-typed expression");
            }
            if type_of_compute_expr(&value, &ctx.type_ctx(), &mut self.types) != f32_id {
                return self.err("atomic_add value must be an f32-typed expression");
            }
            return Ok(vec![IrStmt::AtomicAdd { buf, index, value }]);
        }
        if let Some(Tok::Ident(name)) = self.peek()
            && let Some(&storage) = ctx.textures.get(name)
            && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Dot))
            && matches!(
                self.toks.get(self.pos + 2).map(|t| &t.tok),
                Some(Tok::Ident(m)) if m == "store"
            )
        {
            let image = name.clone();
            if !storage {
                return self.err(format!(
                    "`{image}` is a texture, which only reads; declare it `image` to store"
                ));
            }
            self.pos += 3;
            let (u, f) = (self.types.u32_id(), self.types.f32_id());
            let args = self.parse_texel_args(ctx, "store", &[u, u, f, f, f, f])?;
            return Ok(vec![IrStmt::ImageStore { image, args }]);
        }
        if let Some(Tok::Ident(name)) = self.peek() {
            let name = name.clone();
            if self.functions.contains_key(&name) {
                return self.parse_fn_call_stmt(&name, ctx);
            }
        }
        let lhs = self.parse_expr(ctx)?;
        if self.eat(&Tok::Eq) {
            let rhs = self.parse_expr(ctx)?;
            let stmt = match lhs {
                IrExpr::Local(name) => {
                    self.check_frag_assign(ctx, &name, &rhs)?;
                    if let Some(&local_ty) = ctx.locals.get(&name) {
                        let rhs_ty = type_of_compute_expr(&rhs, &ctx.type_ctx(), &mut self.types);
                        if (u32_vec_lanes(&self.types, local_ty).is_some()
                            || u32_vec_lanes(&self.types, rhs_ty).is_some())
                            && rhs_ty != local_ty
                        {
                            return self.err(format!(
                                "assignment to `{name}` has a different vector type"
                            ));
                        }
                    }
                    IrStmt::AssignLocal { name, value: rhs }
                }
                IrExpr::BufferLoad { buf, index } => {
                    let Some(buffer) = ctx.buffers.get(&buf).copied() else {
                        return self.err(format!("`{buf}` is not a device buffer"));
                    };
                    if !buffer.writable {
                        return self.err(format!(
                            "`{buf}` is read-only; declare it `device mut buffer`"
                        ));
                    }
                    let bf16_id = self.types.bf16_id();
                    let f16_id = self.types.f16_id();
                    let f32_id = self.types.f32_id();
                    let i8_id = self.types.intern(i8_type());
                    let u8_id = self.types.intern(u8_type());
                    let expected = if buffer.elem == bf16_id || buffer.elem == f16_id {
                        f32_id
                    } else if buffer.elem == i8_id {
                        self.types.i32_id()
                    } else if buffer.elem == u8_id {
                        self.types.u32_id()
                    } else {
                        buffer.elem
                    };
                    if type_of_compute_expr(&rhs, &ctx.type_ctx(), &mut self.types) != expected {
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
                    let Some(elem) = ctx.shared.get(&name).map(|d| d.elem) else {
                        return self.err(format!("`{name}` is not a declared shared array"));
                    };
                    let expected = storage_value_type(&mut self.types, elem);
                    if type_of_compute_expr(&rhs, &ctx.type_ctx(), &mut self.types) != expected {
                        return self.err(format!(
                            "assignment to shared array<{}> has incompatible value type",
                            buffer_elem_name(elem, &self.types)
                        ));
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

    /// Inlines a user `fn` call at the current statement: `in` args bind to fresh locals, `out`
    /// args alias the caller's local, and the body's locals are renamed fresh.
    pub(super) fn parse_fn_call_stmt(
        &mut self,
        name: &str,
        ctx: &mut Ctx,
    ) -> Result<Vec<IrStmt>, ParseError> {
        self.pos += 1;
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
                if type_of_compute_expr(&arg_expr, &ctx.type_ctx(), &mut self.types) != param.ty {
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
        let annot = if !self.eat(&Tok::Colon) {
            None
        } else if self.at_kw("frag") {
            Some(self.parse_frag_type()?)
        } else if self.at_kw("u32x2") || self.at_kw("u32x4") {
            Some(self.parse_buffer_elem_type()?)
        } else {
            Some(self.parse_scalar_type()?)
        };
        self.expect(&Tok::Eq, "`=` in let binding")?;
        let init = self.parse_expr(ctx)?;
        let is_zero_acc = matches!(
            init,
            IrExpr::Builtin {
                func: BuiltinFn::FragZeroAcc,
                ..
            }
        );
        let ty = match annot {
            Some(ty) => {
                let init_ty = type_of_compute_expr(&init, &ctx.type_ctx(), &mut self.types);
                let is_frag = frag_parts(&self.types, ty).is_some();
                if is_frag && !is_zero_acc && init_ty != ty {
                    return self.err("let initializer does not match the frag annotation");
                }
                if u32_vec_lanes(&self.types, ty).is_some() && init_ty != ty {
                    return self.err("let initializer does not match the vector annotation");
                }
                ty
            }
            None if is_zero_acc => {
                return self.err("frag_zero_acc() needs a frag<…, acc> annotation on its let");
            }
            None => type_of_compute_expr(&init, &ctx.type_ctx(), &mut self.types),
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
        let u32_id = self.types.u32_id();
        ctx.locals.insert(var.clone(), u32_id);
        ctx.locals_order.push((var.clone(), u32_id));
        let body = self.parse_block(ctx)?;
        Ok(IrStmt::For { var, lo, hi, body })
    }
}
