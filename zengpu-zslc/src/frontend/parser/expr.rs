use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_expr(&mut self, ctx: &Ctx) -> Result<IrExpr, ParseError> {
        self.parse_or(ctx)
    }

    /// Vectors are rejected in arithmetic and comparisons. They are accepted in loads, stores, lets
    /// and fields.
    fn reject_vector_operands(
        &mut self,
        ctx: &Ctx,
        operands: [&IrExpr; 2],
    ) -> Result<(), ParseError> {
        let tc = ctx.type_ctx();
        for operand in operands {
            let ty = type_of_compute_expr(operand, &tc, &mut self.types);
            if u32_vec_lanes(&self.types, ty).is_some() {
                return self.err(
                    "vector values (u32x2, u32x4) only support loads, stores, lets, and field access",
                );
            }
        }
        Ok(())
    }

    fn vector_component(&self, field: &str, lanes: u8) -> Result<u32, ParseError> {
        let comp = match field {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            "w" if lanes == 4 => 3,
            other => {
                return self.err(format!("u32x{lanes} has no field `.{other}`"));
            }
        };
        if comp >= u32::from(lanes) {
            return self.err(format!("u32x{lanes} has no field `.{field}`"));
        }
        Ok(comp)
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
        self.reject_vector_operands(ctx, [&lhs, &rhs])?;
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
            self.reject_vector_operands(ctx, [&lhs, &rhs])?;
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
                Some(Tok::Percent) => IrBinOp::Rem,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary(ctx)?;
            self.reject_vector_operands(ctx, [&lhs, &rhs])?;
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
                    if let Some(load) = self.parse_frag_load(ctx, &name)? {
                        return Ok(load);
                    }
                    return self.parse_call(ctx, &name);
                }
                if self.eat(&Tok::Dot) {
                    let field = self.ident()?;
                    if let Some(&storage) = ctx.textures.get(&name) {
                        return self.parse_texel(ctx, name, storage, &field);
                    }
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
                    if let Some(&ty) = ctx.locals.get(&name)
                        && let Some(lanes) = u32_vec_lanes(&self.types, ty)
                    {
                        let component = self.vector_component(&field, lanes)?;
                        return Ok(IrExpr::FieldAccess {
                            base: Box::new(IrExpr::Local(name)),
                            component,
                        });
                    }
                    return self.err(format!("unknown field access on `{name}`"));
                }
                if self.eat(&Tok::LBracket) {
                    let index = self.parse_expr(ctx)?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    if let Some(buffer) = ctx.buffers.get(&name) {
                        let lanes = u32_vec_lanes(&self.types, buffer.elem);
                        let load = IrExpr::BufferLoad {
                            buf: name,
                            index: Box::new(index),
                        };
                        if let Some(lanes) = lanes
                            && self.eat(&Tok::Dot)
                        {
                            let field = self.ident()?;
                            let component = self.vector_component(&field, lanes)?;
                            return Ok(IrExpr::FieldAccess {
                                base: Box::new(load),
                                component,
                            });
                        }
                        return Ok(load);
                    }
                    if ctx.shared.contains_key(&name) {
                        let idx_ty = type_of_compute_expr(&index, &ctx.type_ctx(), &mut self.types);
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

    pub(super) fn parse_texel_args(
        &mut self,
        ctx: &Ctx,
        method: &str,
        types: &[TypeId],
    ) -> Result<Vec<IrExpr>, ParseError> {
        self.expect(&Tok::LParen, "`(`")?;
        let mut args = Vec::new();
        while !self.eat(&Tok::RParen) {
            args.push(self.parse_expr(ctx)?);
            self.eat(&Tok::Comma);
            if self.peek().is_none() {
                return self.err("unterminated call arguments");
            }
        }
        if args.len() != types.len() {
            return self.err(format!(".{method}() takes {} args", types.len()));
        }
        for (arg, &want) in args.iter().zip(types) {
            if type_of_compute_expr(arg, &ctx.type_ctx(), &mut self.types) != want {
                let what = if want == self.types.u32_id() {
                    "u32 texel coordinates"
                } else {
                    "f32 values"
                };
                return self.err(format!(".{method}() takes {what}"));
            }
        }
        Ok(args)
    }

    /// Parses what follows `tex.method`: a size query or a texel read. `.r`, `.g`, `.b` and `.a`
    /// pick one channel.
    pub(super) fn parse_texel(
        &mut self,
        ctx: &Ctx,
        tex: String,
        storage: bool,
        method: &str,
    ) -> Result<IrExpr, ParseError> {
        let (u, f) = (self.types.u32_id(), self.types.f32_id());
        let (op, args) = match method {
            "width" | "height" => {
                self.parse_texel_args(ctx, method, &[])?;
                let axis = u32::from(method == "height");
                return Ok(IrExpr::TexSize { tex, axis });
            }
            "load" => (TexelOp::Load, self.parse_texel_args(ctx, method, &[u, u])?),
            "sample" | "sample_level" if storage => {
                return self.err(format!(
                    "`{tex}` is an image, which only loads and stores; declare it `texture` to sample"
                ));
            }
            "sample" => (
                TexelOp::Sample,
                self.parse_texel_args(ctx, method, &[f, f])?,
            ),
            "sample_level" => (
                TexelOp::SampleLevel,
                self.parse_texel_args(ctx, method, &[f, f, f])?,
            ),
            "store" => return self.err("`.store(…)` is a statement, not a value"),
            other => {
                return self.err(format!(
                    "unknown texture method `.{other}()`; use .load, .sample, .sample_level, .width or .height"
                ));
            }
        };
        self.expect(&Tok::Dot, "`.r`/`.g`/`.b`/`.a` to pick a channel")?;
        let channel = self.ident()?;
        let component = match channel.as_str() {
            "r" | "x" => 0,
            "g" | "y" => 1,
            "b" | "z" => 2,
            "a" | "w" => 3,
            other => {
                return self.err(format!(
                    "a texel has no channel `.{other}`; use .r/.g/.b/.a"
                ));
            }
        };
        Ok(IrExpr::Texel {
            tex,
            op,
            args,
            component,
        })
    }

    pub(super) fn parse_call(&mut self, ctx: &Ctx, name: &str) -> Result<IrExpr, ParseError> {
        let vec_lanes = match name {
            "u32x2" => Some(2u8),
            "u32x4" => Some(4u8),
            _ => None,
        };
        self.expect(&Tok::LParen, "`(`")?;
        if let Some(lanes) = vec_lanes {
            let mut args = Vec::new();
            while !self.eat(&Tok::RParen) {
                args.push(self.parse_expr(ctx)?);
                self.eat(&Tok::Comma);
                if self.peek().is_none() {
                    return self.err("unterminated call arguments");
                }
            }
            if args.len() != usize::from(lanes) {
                return self.err(format!("u32x{lanes}(…) takes {lanes} u32 lanes"));
            }
            let tc = ctx.type_ctx();
            for arg in &args {
                if type_of_compute_expr(arg, &tc, &mut self.types) != self.types.u32_id() {
                    return self.err(format!("u32x{lanes}(…) lanes must be u32"));
                }
            }
            let ty = self.types.u32_vec_id(lanes);
            return Ok(IrExpr::Construct {
                ty: Some(ty),
                dim: lanes,
                args,
            });
        }
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
        if func == BuiltinFn::F32 && args.len() != 1 {
            return self.err("f32() takes 1 arg");
        }
        if func == BuiltinFn::MulAddF32 && args.len() != 3 {
            return self.err("mul_add_f32(a, b, acc) takes 3 args");
        }
        if matches!(
            func,
            BuiltinFn::Dot4I8 | BuiltinFn::Dot4U8 | BuiltinFn::Dot4IU8
        ) && args.len() != 3
        {
            return self.err(format!("{}(a, b, acc) takes 3 args", func.name()));
        }
        let arity = match func {
            BuiltinFn::PkAddU16
            | BuiltinFn::PkMulU16
            | BuiltinFn::PkAddF16
            | BuiltinFn::PkMulF16 => Some(2),
            BuiltinFn::PkMadU16
            | BuiltinFn::PkFmaF16
            | BuiltinFn::Dot2F16
            | BuiltinFn::FunnelShr
            | BuiltinFn::BytePerm
            | BuiltinFn::Select
            | BuiltinFn::Mma
            | BuiltinFn::FragSet => Some(3),
            BuiltinFn::PackF16x2 | BuiltinFn::Byte | BuiltinFn::SByte => Some(2),
            BuiltinFn::F16Lo
            | BuiltinFn::F16Hi
            | BuiltinFn::BitcastF32
            | BuiltinFn::BitcastU32
            | BuiltinFn::BitcastI32 => Some(1),
            BuiltinFn::FragRow | BuiltinFn::FragCol | BuiltinFn::FragGet => Some(2),
            BuiltinFn::FragLen | BuiltinFn::F16Round => Some(1),
            BuiltinFn::FragZeroAcc => Some(0),
            _ => None,
        };
        if let Some(n) = arity
            && args.len() != n
        {
            return self.err(format!("{}() takes {n} args", func.name()));
        }
        self.check_frag_call(ctx, func, &args)?;
        {
            let tc = ctx.type_ctx();
            for arg in &args {
                let ty = type_of_compute_expr(arg, &tc, &mut self.types);
                if u32_vec_lanes(&self.types, ty).is_some() {
                    return self.err(format!(
                        "{}() does not take vector values (u32x2, u32x4)",
                        func.name()
                    ));
                }
            }
        }
        if func == BuiltinFn::I32 {
            let tc = ctx.type_ctx();
            let arg = type_of_compute_expr(&args[0], &tc, &mut self.types);
            let ok = [
                self.types.f32_id(),
                self.types.u32_id(),
                self.types.i32_id(),
            ];
            if !ok.contains(&arg) {
                return self.err("i32(x) takes an f32, u32 or i32 value");
            }
        }
        if func == BuiltinFn::Select {
            let tc = ctx.type_ctx();
            let cond = type_of_compute_expr(&args[0], &tc, &mut self.types);
            let a = type_of_compute_expr(&args[1], &tc, &mut self.types);
            let b = type_of_compute_expr(&args[2], &tc, &mut self.types);
            if cond != self.types.bool_id() {
                return self.err("select(cond, a, b): `cond` must be a bool");
            }
            if a != b {
                return self.err("select(cond, a, b): both arms must have the same type");
            }
        }
        if matches!(
            func,
            BuiltinFn::SubgroupId | BuiltinFn::SubgroupLocalId | BuiltinFn::SubgroupSize
        ) && !args.is_empty()
        {
            return self.err(format!("{}() takes no args", func.name()));
        }
        if matches!(
            func,
            BuiltinFn::SubgroupAny | BuiltinFn::SubgroupAll | BuiltinFn::SubgroupBallot
        ) && args.len() != 1
        {
            return self.err(format!("{}(predicate) takes 1 arg", func.name()));
        }
        if matches!(
            func,
            BuiltinFn::SubgroupBroadcast
                | BuiltinFn::SubgroupShuffle
                | BuiltinFn::SubgroupShuffleXor
                | BuiltinFn::SubgroupShuffleUp
                | BuiltinFn::SubgroupShuffleDown
        ) && args.len() != 2
        {
            return self.err(format!("{}() takes 2 args", func.name()));
        }
        if matches!(
            func,
            BuiltinFn::SubgroupReduceAdd
                | BuiltinFn::SubgroupReduceMin
                | BuiltinFn::SubgroupReduceMax
                | BuiltinFn::SubgroupReduceAnd
                | BuiltinFn::SubgroupReduceOr
                | BuiltinFn::SubgroupScanInclusiveAdd
                | BuiltinFn::SubgroupScanExclusiveAdd
                | BuiltinFn::WorkgroupReduceAdd
                | BuiltinFn::WorkgroupReduceMin
                | BuiltinFn::WorkgroupReduceMax
        ) && args.len() != 1
        {
            return self.err(format!("{}(value) takes 1 arg", func.name()));
        }
        Ok(IrExpr::Builtin { func, args })
    }
}
