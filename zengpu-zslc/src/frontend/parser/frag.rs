//! Parser for `frag<...>` and `mma` builtins. Backends lower the nodes produced here.

use super::*;
use crate::ir::typeck::{frag_parts, frag16_id, i8_type};
use crate::ir::types::FragUse;

fn frag_use_of_load(name: &str) -> Option<(FragUse, bool)> {
    Some(match name {
        "frag_load_a" => (FragUse::A, false),
        "frag_load_b" => (FragUse::B, false),
        "frag_load_b_t" => (FragUse::B, true),
        "frag_load_acc" => (FragUse::Acc, false),
        _ => return None,
    })
}

impl<'a> Parser<'a> {
    fn i8_id(&mut self) -> TypeId {
        self.types.intern(i8_type())
    }

    /// Lane value type. Storage-only f16 widens to f32, and i8 lanes read as i32.
    pub(super) fn frag_value_type(&mut self, elem: TypeId) -> TypeId {
        if elem == self.types.f16_id() {
            self.types.f32_id()
        } else if elem == self.i8_id() {
            self.types.i32_id()
        } else {
            elem
        }
    }

    /// `frag<T, R, C, use>`: a 16x16 tile. `a`/`b` take `f16`, `f32` or `i8`, `acc` takes `f32` or
    /// `i32`.
    pub(super) fn parse_frag_type(&mut self) -> Result<TypeId, ParseError> {
        if !self.eat_kw("frag") {
            return self.err("expected `frag<…>`");
        }
        self.expect(&Tok::Lt, "`<` in frag<…>")?;
        let elem = self.ident()?;
        self.expect(&Tok::Comma, "`,` in frag<…>")?;
        let rows = self.frag_dim()?;
        self.expect(&Tok::Comma, "`,` in frag<…>")?;
        let cols = self.frag_dim()?;
        self.expect(&Tok::Comma, "`,` in frag<…>")?;
        let usage = match self.ident()?.as_str() {
            "a" => FragUse::A,
            "b" => FragUse::B,
            "acc" => FragUse::Acc,
            other => return self.err(format!("unknown frag use `{other}`; use a, b, or acc")),
        };
        self.expect(&Tok::Gt, "`>` in frag<…>")?;
        if rows != 16 || cols != 16 {
            return self.err("only 16x16 frags are supported");
        }
        let element = match (usage, elem.as_str()) {
            (FragUse::A | FragUse::B, "f16") => self.types.f16_id(),
            (FragUse::A | FragUse::B, "f32") => self.types.f32_id(),
            (FragUse::A | FragUse::B, "i8") => self.i8_id(),
            (FragUse::Acc, "f32") => self.types.f32_id(),
            (FragUse::Acc, "i32") => self.types.i32_id(),
            _ => {
                return self.err(format!(
                    "frag element `{elem}` does not fit its use: a and b take f16, f32 or i8, acc takes f32 or i32"
                ));
            }
        };
        Ok(frag16_id(&mut self.types, element, usage))
    }

    fn frag_dim(&mut self) -> Result<u8, ParseError> {
        match self.peek() {
            Some(Tok::Int(v)) if *v <= 255 => {
                let v = *v as u8;
                self.pos += 1;
                Ok(v)
            }
            _ => self.err("expected a frag dimension (16)"),
        }
    }

    fn frag_src(&mut self, ctx: &Ctx) -> Result<(FragSrc, TypeId), ParseError> {
        let name = self.ident()?;
        self.expect(&Tok::Comma, "`,` after frag source")?;
        if let Some(buffer) = ctx.buffers.get(&name) {
            return Ok((FragSrc::Buffer(name), buffer.elem));
        }
        if let Some(shared) = ctx.shared.get(&name) {
            let elem = shared.elem;
            return Ok((FragSrc::Shared(name), elem));
        }
        self.err(format!("`{name}` is not a buffer or shared array"))
    }

    /// `offset` and `ld` are in source elements (bytes for packed i8 read from u32).
    fn frag_offset_ld(&mut self, ctx: &Ctx) -> Result<(IrExpr, IrExpr), ParseError> {
        let offset = self.parse_expr(ctx)?;
        self.expect(&Tok::Comma, "`,` after frag offset")?;
        let ld = self.parse_expr(ctx)?;
        let u32_id = self.types.u32_id();
        for (what, e) in [("offset", &offset), ("ld", &ld)] {
            if type_of_compute_expr(e, &ctx.type_ctx(), &mut self.types) != u32_id {
                return self.err(format!("frag {what} must be a u32-typed expression"));
            }
        }
        Ok((offset, ld))
    }

    /// Element types. For `a` and `b`: `f16` (rounded from shared `f32`), `f32` from an f32
    /// buffer, or packed `u32`/`i32` i8. A shared f32 load is `f16` unless its `let` asks for
    /// [`Self::frag_load_for`]. For `acc`: `f32` or `i32`.
    pub(super) fn parse_frag_load(
        &mut self,
        ctx: &Ctx,
        name: &str,
    ) -> Result<Option<IrExpr>, ParseError> {
        let Some((usage, trans)) = frag_use_of_load(name) else {
            return Ok(None);
        };
        self.expect(&Tok::LParen, "`(` after frag_load")?;
        let (src, buf_elem) = self.frag_src(ctx)?;
        let (offset, ld) = self.frag_offset_ld(ctx)?;
        self.expect(&Tok::RParen, "`)` after frag_load arguments")?;
        let f16 = self.types.f16_id();
        let f32 = self.types.f32_id();
        let i32 = self.types.i32_id();
        let u32 = self.types.u32_id();
        let shared = matches!(src, FragSrc::Shared(_));
        let i8 = self.i8_id();
        let element = match (usage, buf_elem) {
            (FragUse::A | FragUse::B, e) if e == f16 => f16,
            (FragUse::A | FragUse::B, e) if e == f32 && shared => f16,
            (FragUse::A | FragUse::B, e) if e == f32 => f32,
            (FragUse::A | FragUse::B, e) if e == i8 && shared => i8,
            (FragUse::A | FragUse::B, e) if (e == u32 || e == i32) && !shared => i8,
            (FragUse::Acc, e) if e == f32 => f32,
            (FragUse::Acc, e) if e == i32 => i32,
            _ => {
                return self.err(format!(
                    "{name}: source element does not fit this use (a/b: f16 buffer or shared array, f32 buffer or shared array, i8 shared array, u32/i32 buffer for packed i8, or shared f32; acc: f32 or i32)"
                ));
            }
        };
        let ty = frag16_id(&mut self.types, element, usage);
        Ok(Some(IrExpr::FragLoad {
            ty,
            src,
            offset: Box::new(offset),
            ld: Box::new(ld),
            trans,
        }))
    }

    /// `let x: frag<f32, 16, 16, a|b> = frag_load_a|b(shared, …)`: a shared f32 load takes the
    /// annotated element instead of rounding to f16. Other initializers are returned as they are.
    pub(super) fn frag_load_for(&mut self, ctx: &Ctx, init: IrExpr, want: TypeId) -> IrExpr {
        let f32 = self.types.f32_id();
        let Some((want_elem, want_use)) = frag_parts(&self.types, want) else {
            return init;
        };
        let IrExpr::FragLoad { ty, src, .. } = &init else {
            return init;
        };
        let src_elem = match src {
            FragSrc::Buffer(name) => ctx.buffers.get(name).map(|b| b.elem),
            FragSrc::Shared(name) => ctx.shared.get(name).map(|s| s.elem),
        };
        let load_use = frag_parts(&self.types, *ty).map(|(_, u)| u);
        let f32_ab = want_elem == f32 && want_use != FragUse::Acc && src_elem == Some(f32);
        if !f32_ab || load_use != Some(want_use) {
            return init;
        }
        match init {
            IrExpr::FragLoad {
                src,
                offset,
                ld,
                trans,
                ..
            } => IrExpr::FragLoad {
                ty: want,
                src,
                offset,
                ld,
                trans,
            },
            other => other,
        }
    }

    pub(super) fn parse_frag_store(&mut self, ctx: &Ctx) -> Result<IrStmt, ParseError> {
        self.expect(&Tok::LParen, "`(` after frag_store")?;
        let (src, buf_elem) = self.frag_src(ctx)?;
        if let FragSrc::Buffer(name) = &src
            && !ctx.buffers[name].writable
        {
            return self.err(format!(
                "`{name}` is read-only; declare it `device mut buffer`"
            ));
        }
        let (offset, ld) = self.frag_offset_ld(ctx)?;
        self.expect(&Tok::Comma, "`,` before frag_store value")?;
        let value = self.parse_expr(ctx)?;
        self.expect(&Tok::RParen, "`)` after frag_store arguments")?;
        let ty = type_of_compute_expr(&value, &ctx.type_ctx(), &mut self.types);
        match frag_parts(&self.types, ty) {
            Some((elem, FragUse::Acc)) if elem == buf_elem => {}
            _ => {
                return self.err(
                    "frag_store stores an acc fragment into a buffer or shared array of its element type",
                );
            }
        }
        Ok(IrStmt::FragStore {
            src,
            offset,
            ld,
            value,
        })
    }

    /// Arguments are already counted by `parse_call`.
    pub(super) fn check_frag_call(
        &mut self,
        ctx: &Ctx,
        func: BuiltinFn,
        args: &[IrExpr],
    ) -> Result<(), ParseError> {
        let tc = ctx.type_ctx();
        let u32_id = self.types.u32_id();
        let frag_arg = |this: &mut Self, e: &IrExpr| -> Result<(TypeId, FragUse), ParseError> {
            let ty = type_of_compute_expr(e, &tc, &mut this.types);
            match frag_parts(&this.types, ty) {
                Some((elem, usage)) => Ok((elem, usage)),
                None => this.err(format!("{}() expects a frag argument", func.name())),
            }
        };
        match func {
            BuiltinFn::Mma => {
                let (ea, ua) = frag_arg(self, &args[0])?;
                let (eb, ub) = frag_arg(self, &args[1])?;
                if ua != FragUse::A || ub != FragUse::B {
                    return self
                        .err("mma(a, b, acc) takes an a fragment, a b fragment, then an acc");
                }
                let f16 = self.types.f16_id();
                let f32 = self.types.f32_id();
                let want_acc = if (ea == f16 || ea == f32) && ea == eb {
                    f32
                } else if ea == self.i8_id() && ea == eb {
                    self.types.i32_id()
                } else {
                    return self.err("mma operands must both be f16, both be f32, or both be i8");
                };
                if matches!(
                    args[2],
                    IrExpr::Builtin {
                        func: BuiltinFn::FragZeroAcc,
                        ..
                    }
                ) {
                    return Ok(());
                }
                let (ec, uc) = frag_arg(self, &args[2])?;
                if uc != FragUse::Acc || ec != want_acc {
                    return self.err(
                        "mma accumulator must be acc with f32 (f16 or f32 operands) or i32 (i8 operands)",
                    );
                }
                Ok(())
            }
            BuiltinFn::FragLen => frag_arg(self, &args[0]).map(|_| ()),
            BuiltinFn::FragRow | BuiltinFn::FragCol | BuiltinFn::FragGet | BuiltinFn::FragSet => {
                let (elem, _) = frag_arg(self, &args[0])?;
                if type_of_compute_expr(&args[1], &tc, &mut self.types) != u32_id {
                    return self.err(format!("{}() lane element index must be u32", func.name()));
                }
                if func == BuiltinFn::FragSet {
                    let want = self.frag_value_type(elem);
                    if type_of_compute_expr(&args[2], &tc, &mut self.types) != want {
                        return self.err("frag_set value type does not match the fragment element");
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Assigning to a fragment local: the value must be a fragment of the same type, or
    /// `frag_zero_acc()` into an accumulator.
    pub(super) fn check_frag_assign(
        &mut self,
        ctx: &Ctx,
        name: &str,
        value: &IrExpr,
    ) -> Result<(), ParseError> {
        let Some(&target) = ctx.locals.get(name) else {
            return Ok(());
        };
        if frag_parts(&self.types, target).is_none() {
            return Ok(());
        }
        let zero = matches!(
            value,
            IrExpr::Builtin {
                func: BuiltinFn::FragZeroAcc,
                ..
            }
        );
        let zero_into_acc = frag_parts(&self.types, target).is_some_and(|(_, u)| u == FragUse::Acc);
        if zero && zero_into_acc {
            return Ok(());
        }
        if type_of_compute_expr(value, &ctx.type_ctx(), &mut self.types) != target {
            return self.err(format!(
                "assignment to frag local `{name}` has a different frag type"
            ));
        }
        Ok(())
    }
}
