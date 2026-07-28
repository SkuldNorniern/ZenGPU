use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_scalar_type(&mut self) -> Result<TypeId, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "u32" => Ok(self.types.u32_id()),
            "i32" => Ok(self.types.i32_id()),
            "f32" => Ok(self.types.f32_id()),
            "bool" => Ok(self.types.bool_id()),
            other => self.err(format!(
                "unsupported scalar type `{other}`; use u32, i32, f32, or bool"
            )),
        }
    }

    /// Parse a `device buffer<…>` element type: scalars plus `bf16`. Distinct
    /// from `parse_scalar_type` because `bf16` is storage-only — it is valid
    /// as a buffer element but never as a local/param/push-field type (no
    /// ZSL expression is ever bf16-typed at the value level).
    pub(super) fn parse_buffer_elem_type(&mut self) -> Result<TypeId, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "u32" => Ok(self.types.u32_id()),
            "i32" => Ok(self.types.i32_id()),
            "f32" => Ok(self.types.f32_id()),
            "bf16" => Ok(self.types.bf16_id()),
            other => self.err(format!(
                "unsupported buffer element type `{other}`; use u32, i32, f32, or bf16"
            )),
        }
    }

    /// Parse any value type (scalar/vector/matrix), used for push fields and
    /// graphics declarations: `f32`, `u32`, `f32x2/3/4`, `mat4x4<f32>`.
    pub(super) fn parse_any_type(&mut self) -> Result<TypeId, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "f32" => Ok(self.types.f32_id()),
            "u32" => Ok(self.types.u32_id()),
            "f32x2" => Ok(self.types.vec_id(2)),
            "f32x3" => Ok(self.types.vec_id(3)),
            "f32x4" => Ok(self.types.vec_id(4)),
            "mat4x4" => {
                self.expect(&Tok::Lt, "`<` in mat4x4<…>")?;
                let elem = self.ident()?;
                if elem != "f32" {
                    return self.err("only mat4x4<f32> is supported");
                }
                self.expect(&Tok::Gt, "`>` in mat4x4<…>")?;
                Ok(self.types.mat4_id())
            }
            other => self.err(format!(
                "unsupported type `{other}`; use f32/u32/f32x2/f32x3/f32x4/mat4x4<f32>"
            )),
        }
    }

    // ── Params ───────────────────────────────────────────────────────────────

    pub(super) fn parse_param_type(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        params: &mut Vec<PendingParam>,
    ) -> Result<(), ParseError> {
        if self.eat_kw("device") {
            let writable = self.eat_kw("mut");
            if !self.eat_kw("buffer") {
                return self.err("expected `buffer` in device buffer type");
            }
            self.expect(&Tok::Lt, "`<` in buffer<…>")?;
            let elem = self.parse_buffer_elem_type()?;
            self.expect(&Tok::Gt, "`>` in buffer<…>")?;
            if elem == self.types.bool_id() {
                return self.err("buffer<bool> is not supported");
            }
            ctx.buffers
                .insert(name.to_string(), BufferSymbol { writable, elem });
            params.push(PendingParam {
                name: name.to_string(),
                kind: PendingParamKind::Buffer {
                    elem,
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
            let u32_id = self.types.u32_id();
            let f32_id = self.types.f32_id();
            for (field, ty) in fields {
                if ty != u32_id && ty != f32_id {
                    return self.err(format!("compute push field `{field}` must be u32 or f32"));
                }
                ctx.scalars.insert(field.clone(), ty);
                params.push(PendingParam {
                    name: field,
                    kind: PendingParamKind::Scalar(ty),
                });
            }
            Ok(())
        }
    }

    // ── Statements ─────────────────────────────────────────────────────────────

    pub(super) fn parse_ret_types(&mut self) -> Result<Vec<TypeId>, ParseError> {
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

    pub(super) fn parse_ret_one(&mut self) -> Result<TypeId, ParseError> {
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
}
