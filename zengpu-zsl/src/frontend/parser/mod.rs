//! Native ZSL parser (compute): tokens to [`crate::ir::Module`].
//!
//! Dependency-free (no `syn` or `quote`). Recursive descent over the [`lex`] token stream.
//! Statements need no terminators. The expression parser is greedy, so a statement ends where the
//! next token cannot extend it (maximal munch). This matches `zen.md`'s terminator-free syntax
//! without relying on newlines, which the proc-macro shell loses when it stringifies its input.
//!
//! Grammar (compute slice):
//! ```text
//! module    := ("module" dotted)? push* attr* "kernel" Ident "(" params ")" block
//! push      := "push" Ident "{" (Ident ":" scalar)* "}"
//! attr      := "@" "workgroup_size" "(" Int ("," Int)* ")"
//! param     := Ident ":" ("device" "mut"? "buffer" "<" scalar ">" | "global_id" | Ident)
//! block     := "{" stmt* "}"
//! stmt      := "let" Ident (":" scalar)? "=" expr
//!            | "if" expr block ("else" block)?
//!            | "for" Ident "in" expr ".." expr block
//!            | "barrier" "(" ")"
//!            | "atomic_add" "(" Ident "," expr "," expr ")"
//!            | expr ("=" expr)?
//! ```

mod compute;
mod expr;
mod graphics;
mod graphics_expr;
mod inference;
mod rewrite;
mod state;
mod types;

#[allow(unused_imports)]
use compute::*;
#[allow(unused_imports)]
use expr::*;
#[allow(unused_imports)]
use graphics::*;
#[allow(unused_imports)]
use graphics_expr::*;
use inference::*;
use rewrite::*;
use state::*;
#[allow(unused_imports)]
use types::*;

use std::collections::HashMap;

use crate::frontend::lex::{Tok, Token, lex};
use crate::ir::node::{BuiltinFn, IrBinOp, IrExpr, IrStmt};
use crate::ir::types::TypeId;
use crate::ir::{
    Entry, EntryKind, GfxInput, GfxScalar, GfxTy, GraphicsEntry, GraphicsModule, Module,
    Mutability, Param, ParamKind, ScalarTy, SharedDecl,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub at: Option<usize>,
}

pub enum Shader {
    Compute(Module),
    Graphics(GraphicsModule),
}

pub fn parse_zsl(src: &str) -> Result<Shader, ParseError> {
    let toks = lex(src).map_err(|e| ParseError {
        msg: e.msg,
        at: Some(e.at),
    })?;
    let mut p = Parser::new(&toks);
    p.parse_shader()
}

/// Test helper. The macro shell dispatches via [`parse_zsl`].
#[allow(dead_code)]
pub fn parse_compute(src: &str) -> Result<Module, ParseError> {
    match parse_zsl(src)? {
        Shader::Compute(m) => Ok(m),
        Shader::Graphics(_) => Err(ParseError {
            msg: "expected a compute `kernel`, found a graphics entry".into(),
            at: None,
        }),
    }
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    /// Push-block fields as the graphics superset [`GfxTy`]. Compute narrows them to scalars.
    pushes: HashMap<String, Vec<(String, GfxTy)>>,
    /// Compute `fn`s, always inlined at their call site. SPIR-V never sees a call.
    functions: HashMap<String, FunctionDef>,
    /// Per-call-site counter that keeps inlined locals from colliding.
    call_counter: u32,
    types: crate::ir::arena::TypeArena,
}

impl<'a> Parser<'a> {
    fn new(toks: &'a [Token]) -> Self {
        Self {
            toks,
            pos: 0,
            pushes: HashMap::new(),
            functions: HashMap::new(),
            call_counter: 0,
            types: crate::ir::arena::TypeArena::new(),
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn span(&self) -> Option<usize> {
        self.toks
            .get(self.pos)
            .map(|t| t.span.start)
            .or_else(|| self.toks.last().map(|t| t.span.end))
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            msg: msg.into(),
            at: self.span(),
        })
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), ParseError> {
        if self.eat(t) {
            Ok(())
        } else {
            self.err(format!("expected {what}"))
        }
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Some(Tok::Ident(s)) => {
                let s = s.clone();
                self.pos += 1;
                Ok(s)
            }
            _ => self.err("expected identifier"),
        }
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::spirv::lower_compute;

    fn lower(src: &str) -> Vec<u32> {
        let module = parse_compute(src).expect("parse");
        lower_compute(&module).expect("lower")
    }

    #[test]
    fn saxpy_parses_and_lowers() {
        let src = r#"
            push SaxpyPush { n: u32, alpha: f32 }
            @workgroup_size(256)
            kernel saxpy(
                push: SaxpyPush,
                x: device buffer<f32>,
                y: device mut buffer<f32>,
                id: global_id,
            ) {
                let i = id.x
                if i < push.n {
                    y[i] = push.alpha * x[i] + y[i]
                }
            }
        "#;
        let words = lower(src);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    #[test]
    fn module_and_workgroup_parsed() {
        let src = r#"
            module zengpu.examples.copy
            @workgroup_size(64)
            kernel copy(x: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                let i = id.x
                out[i] = x[i]
            }
        "#;
        let m = parse_compute(src).expect("parse");
        let EntryKind::Compute { local_size } = m.entry.kind;
        assert_eq!(local_size, [64, 1, 1]);
    }

    #[test]
    fn local_type_inference() {
        let src = r#"
            @workgroup_size(1)
            kernel k(src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
                let i = id.x
                let v = src[i]
                dst[i] = v
            }
        "#;
        let m = parse_compute(src).expect("parse");
        let types: std::collections::HashMap<_, _> = m.entry.locals.iter().cloned().collect();
        assert_eq!(m.types.scalar_ty_of(types["i"]), ScalarTy::U32);
        assert_eq!(m.types.scalar_ty_of(types["v"]), ScalarTy::F32);
    }

    #[test]
    fn for_loop_and_builtins() {
        let src = r#"
            push P { n: u32 }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, p: P, id: global_id) {
                let acc: f32 = 0.0
                for j in 0..p.n {
                    acc = max(acc, sqrt(dst[j]))
                }
                dst[id.x] = acc
            }
        "#;
        let words = lower(src);
        assert!(!words.is_empty());
    }

    #[test]
    fn read_only_buffer_write_is_rejected() {
        let src = r#"
            @workgroup_size(1)
            kernel k(x: device buffer<f32>, id: global_id) {
                x[id.x] = 1.0
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("read-only"), "got: {}", e.msg);
    }

    #[test]
    fn atomic_add_requires_mutable_buffer_and_f32_operands() {
        let read_only = r#"
            @workgroup_size(1)
            kernel k(x: device buffer<f32>) {
                atomic_add(x, 0.0, 1.0)
            }
        "#;
        let e = parse_compute(read_only).unwrap_err();
        assert!(e.msg.contains("mutable device buffer"), "got: {}", e.msg);

        let u32_index = r#"
            @workgroup_size(1)
            kernel k(x: device mut buffer<f32>, id: global_id) {
                atomic_add(x, id.x, 1.0)
            }
        "#;
        parse_compute(u32_index).expect("u32 indices are natural buffer indices");

        let bad_value = r#"
            @workgroup_size(1)
            kernel k(idx: device buffer<f32>, x: device mut buffer<f32>, id: global_id) {
                atomic_add(x, idx[id.x], id.x)
            }
        "#;
        let e = parse_compute(bad_value).unwrap_err();
        assert!(e.msg.contains("value must be an f32"), "got: {}", e.msg);
    }

    #[test]
    fn unknown_identifier_is_rejected() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, id: global_id) {
                dst[id.x] = nope
            }
        "#;
        assert!(parse_compute(src).is_err());
    }

    #[test]
    fn fn_with_in_and_out_params_inlines_and_lowers() {
        let src = r#"
            fn add_one(x: u32, out result: u32) {
                result = x + 1
            }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                let out: u32 = 0
                add_one(id.x, out)
                dst[id.x] = out
            }
        "#;
        let words = lower(src);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    #[test]
    fn fn_can_call_an_earlier_declared_fn() {
        let src = r#"
            fn inc(x: u32, out result: u32) {
                result = x + 1
            }
            fn inc_twice(x: u32, out result: u32) {
                let once: u32 = 0
                inc(x, once)
                inc(once, result)
            }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                let out: u32 = 0
                inc_twice(id.x, out)
                dst[id.x] = out
            }
        "#;
        lower(src);
    }

    #[test]
    fn fn_call_before_declaration_is_rejected() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                let out: u32 = 0
                not_yet_declared(id.x, out)
                dst[id.x] = out
            }
            fn not_yet_declared(x: u32, out result: u32) {
                result = x
            }
        "#;
        assert!(parse_compute(src).is_err());
    }

    #[test]
    fn duplicate_fn_name_is_rejected() {
        let src = r#"
            fn dup(x: u32, out result: u32) { result = x }
            fn dup(x: u32, out result: u32) { result = x }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                dst[id.x] = id.x
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("duplicate symbol"), "got: {}", e.msg);
    }

    #[test]
    fn fn_out_argument_must_be_a_declared_local() {
        let src = r#"
            fn set(out result: u32) { result = 1 }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                set(undeclared)
                dst[id.x] = undeclared
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("not a declared local"), "got: {}", e.msg);
    }

    #[test]
    fn fn_out_argument_type_mismatch_is_rejected() {
        let src = r#"
            fn set(out result: u32) { result = 1 }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, id: global_id) {
                let out: f32 = 0.0
                set(out)
                dst[id.x] = out
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("does not match"), "got: {}", e.msg);
    }

    #[test]
    fn fn_body_cannot_reference_kernel_buffers() {
        let src = r#"
            fn leaks(out result: u32) {
                result = x[0]
            }
            @workgroup_size(1)
            kernel k(x: device buffer<u32>, dst: device mut buffer<u32>, id: global_id) {
                let out: u32 = 0
                leaks(out)
                dst[id.x] = out
            }
        "#;
        assert!(parse_compute(src).is_err());
    }

    #[test]
    fn repeated_calls_to_the_same_fn_do_not_collide() {
        let src = r#"
            fn inc(x: u32, out result: u32) {
                let bumped = x + 1
                result = bumped
            }
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                let a: u32 = 0
                let b: u32 = 0
                inc(id.x, a)
                inc(a, b)
                dst[id.x] = a + b
            }
        "#;
        lower(src);
    }

    #[test]
    fn bitwise_operators_parse_and_lower_to_spirv() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<u32>, id: global_id) {
                let a = id.x & 255
                let b = a | 16
                let c = b ^ a
                let d = c << 2
                let e = d >> 1
                dst[id.x] = a + b + c + d + e
            }
        "#;
        let words = lower(src);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    #[test]
    fn bitwise_operators_reject_non_u32_operands() {
        let src = r#"
            @workgroup_size(1)
            kernel k(dst: device mut buffer<f32>, id: global_id) {
                let x: f32 = 1.0
                dst[id.x] = x & 1
            }
        "#;
        let module = parse_compute(src).expect("parse");
        let err = crate::backend::spirv::lower_compute(&module).unwrap_err();
        assert!(err.message.contains("u32 operands"), "got: {err}");
    }

    #[test]
    fn mix64_style_hash_helper_lowers_to_spirv() {
        let src = r#"
            fn mulhi_u32(u: u32, v: u32, out result: u32) {
                let u0 = u & 65535
                let u1 = u >> 16
                let v0 = v & 65535
                let v1 = v >> 16

                let t = u0 * v0
                let k = t >> 16

                t = u1 * v0 + k
                let w1 = t & 65535
                let w2 = t >> 16

                t = u0 * v1 + w1
                k = t >> 16

                result = u1 * v1 + w2 + k
            }

            fn mix64(hi: u32, lo: u32, out out_hi: u32, out out_lo: u32) {
                // 64-bit shift: lo gets its top bits from the low two bits of hi.
                let shifted_lo: u32 = (hi << 2) | (lo >> 30)
                let shifted_hi: u32 = hi >> 30
                let v_lo = lo ^ shifted_lo
                let v_hi = hi ^ shifted_hi

                let c1_hi: u32 = 3210089581
                let c1_lo: u32 = 483266489
                let low_lo = v_lo * c1_lo
                let cross: u32 = 0
                mulhi_u32(v_lo, c1_lo, cross)
                let hi_mul = v_hi * c1_lo + v_lo * c1_hi + cross

                out_lo = low_lo
                out_hi = hi_mul
            }

            @workgroup_size(64)
            kernel hash(
                in_hi: device buffer<u32>, in_lo: device buffer<u32>,
                out_hi: device mut buffer<u32>, out_lo: device mut buffer<u32>,
                id: global_id,
            ) {
                let h: u32 = 0
                let l: u32 = 0
                mix64(in_hi[id.x], in_lo[id.x], h, l)
                out_hi[id.x] = h
                out_lo[id.x] = l
            }
        "#;
        let words = lower(src);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    #[test]
    fn typeck_infers_local_and_literal_types() {
        let mut arena = crate::ir::arena::TypeArena::new();
        let mut new_ctx = crate::ir::typeck::ComputeTypeCtx::default();
        new_ctx
            .locals
            .insert("x".to_string(), arena.intern_scalar(ScalarTy::F32));
        let local_expr = IrExpr::Local("x".to_string());
        let new_ty = crate::ir::typeck::type_of_compute_expr(&local_expr, &new_ctx, &mut arena);

        assert_eq!(new_ty, arena.intern_scalar(ScalarTy::F32));
    }

    #[test]
    fn typeck_infers_comparison_and_arithmetic_types() {
        let cmp = IrExpr::Binary {
            op: IrBinOp::Lt,
            lhs: Box::new(IrExpr::LitU32(1)),
            rhs: Box::new(IrExpr::LitU32(2)),
        };
        let arith = IrExpr::Binary {
            op: IrBinOp::Add,
            lhs: Box::new(IrExpr::LitU32(1)),
            rhs: Box::new(IrExpr::LitF32(2.0)),
        };

        let mut arena = crate::ir::arena::TypeArena::new();
        let new_ctx = crate::ir::typeck::ComputeTypeCtx::default();

        let new_cmp_ty = crate::ir::typeck::type_of_compute_expr(&cmp, &new_ctx, &mut arena);
        assert_eq!(new_cmp_ty, arena.intern_scalar(ScalarTy::Bool));

        let new_arith_ty = crate::ir::typeck::type_of_compute_expr(&arith, &new_ctx, &mut arena);
        assert_eq!(new_arith_ty, arena.intern_scalar(ScalarTy::F32));
    }

    #[test]
    fn typeck_infers_buffer_load_type() {
        let load = IrExpr::BufferLoad {
            buf: "buf".to_string(),
            index: Box::new(IrExpr::LitU32(0)),
        };

        let mut arena = crate::ir::arena::TypeArena::new();
        let mut new_ctx = crate::ir::typeck::ComputeTypeCtx::default();
        new_ctx
            .buffers
            .insert("buf".to_string(), arena.intern_scalar(ScalarTy::I32));
        let new_ty = crate::ir::typeck::type_of_compute_expr(&load, &new_ctx, &mut arena);

        assert_eq!(new_ty, arena.intern_scalar(ScalarTy::I32));
    }

    #[test]
    fn standalone_checker_matches_infer_gfx_ty_for_mat4_vec4_promotion() {
        let mut ctx = GfxCtx {
            inputs: HashMap::new(),
            scalars: HashMap::new(),
            push_param: None,
            buffers: HashMap::new(),
            locals: HashMap::new(),
            locals_order: Vec::new(),
        };
        ctx.scalars.insert("mvp".to_string(), GfxTy::Mat4);

        let vec4 = IrExpr::Construct {
            ty: None,
            dim: 4,
            args: vec![
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(1.0),
            ],
        };
        let mat_mul = IrExpr::Binary {
            op: IrBinOp::Mul,
            lhs: Box::new(IrExpr::ScalarParam("mvp".to_string())),
            rhs: Box::new(vec4),
        };
        let old_ty = infer_gfx_ty(&mat_mul, &ctx);

        let mut arena = crate::ir::arena::TypeArena::new();
        let mut new_ctx = crate::ir::typeck::GraphicsTypeCtx::default();
        new_ctx.scalars.insert("mvp".to_string(), GfxTy::Mat4);
        let new_vec4 = IrExpr::Construct {
            ty: None,
            dim: 4,
            args: vec![
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(0.0),
                IrExpr::LitF32(1.0),
            ],
        };
        let new_mat_mul = IrExpr::Binary {
            op: IrBinOp::Mul,
            lhs: Box::new(IrExpr::ScalarParam("mvp".to_string())),
            rhs: Box::new(new_vec4),
        };
        let new_ty = crate::ir::typeck::type_of_graphics_expr(&new_mat_mul, &new_ctx, &mut arena);

        assert_eq!(new_ty, arena.intern_gfx_ty(old_ty));
    }
}
