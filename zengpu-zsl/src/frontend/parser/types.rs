use super::*;

impl<'a> Parser<'a> {
    pub(super) fn parse_scalar_type(&mut self) -> Result<ScalarTy, ParseError> {
        let name = self.ident()?;
        match name.as_str() {
            "u32" => Ok(ScalarTy::U32),
            "i32" => Ok(ScalarTy::I32),
            "f32" => Ok(ScalarTy::F32),
            "bool" => Ok(ScalarTy::Bool),
            other => self.err(format!(
                "unsupported scalar type `{other}`; use u32, i32, f32, or bool"
            )),
        }
    }

    pub(super) fn parse_any_type(&mut self) -> Result<GfxTy, ParseError> {
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
            let elem = self.parse_scalar_type()?;
            self.expect(&Tok::Gt, "`>` in buffer<…>")?;
            let elem: TypeId = match elem {
                ScalarTy::F32 | ScalarTy::U32 | ScalarTy::I32 => self.types.intern_scalar(elem),
                ScalarTy::Bool => return self.err("buffer<bool> is not supported"),
            };
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
                let ty: TypeId = match gty {
                    GfxTy::U32 => self.types.intern_scalar(ScalarTy::U32),
                    GfxTy::F32 => self.types.intern_scalar(ScalarTy::F32),
                    _ => {
                        return self
                            .err(format!("compute push field `{field}` must be u32 or f32"));
                    }
                };
                ctx.scalars.insert(field.clone(), ty);
                params.push(PendingParam {
                    name: field,
                    kind: PendingParamKind::Scalar(ty),
                });
            }
            Ok(())
        }
    }

    pub(super) fn parse_ret_types(&mut self) -> Result<Vec<GfxTy>, ParseError> {
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

    pub(super) fn parse_ret_one(&mut self) -> Result<GfxTy, ParseError> {
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
