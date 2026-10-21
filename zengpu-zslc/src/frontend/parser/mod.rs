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
mod frag;
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

use std::collections::{HashMap, HashSet};
use std::mem;

use crate::frontend::lex::{Tok, Token, lex};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrBinOp, IrExpr, IrStmt, TexelOp};
use crate::ir::typeck::{
    StorageElem, frag_parts, i8_type, storage_byte_size, storage_elem, storage_value_type,
    type_of_compute_expr, type_of_graphics_expr, u8_type, u32_vec_lanes,
};
use crate::ir::types::TypeId;
use crate::ir::{
    Backend, Entry, EntryKind, GfxBuiltin, GfxInput, GfxOutput, GfxScalar, GfxSlot, GraphicsEntry,
    GraphicsModule, Module, Mutability, Param, ParamKind, SharedDecl,
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
    /// Field types. Graphics allows a superset, and compute narrows them to scalars.
    pushes: HashMap<String, Vec<(String, TypeId)>>,
    /// Compute `fn`s, always inlined at their call site. SPIR-V never sees a call.
    functions: HashMap<String, FunctionDef>,
    /// Per-call-site counter that keeps inlined locals from colliding.
    call_counter: u32,
    types: TypeArena,
}

impl<'a> Parser<'a> {
    fn new(toks: &'a [Token]) -> Self {
        Self {
            toks,
            pos: 0,
            pushes: HashMap::new(),
            functions: HashMap::new(),
            call_counter: 0,
            types: TypeArena::new(),
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::backend::spirv::lower_compute;
    use crate::backend::{cuda, hip, msl};

    #[test]
    fn backends_attribute_is_recorded_and_validated() {
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            @backends(hip, cuda, hip)
            kernel k(x: device mut buffer<f32>, id: global_id) {
                let i = id.x
                x[i] = 1.0
            }
        "#,
        )
        .expect("parse");
        assert_eq!(module.backends, vec![Backend::Hip, Backend::Cuda]);
        assert!(parse_compute("@backends(vulkan)\nkernel k(id: global_id) { }").is_err());
    }

    fn lower(src: &str) -> Vec<u32> {
        let module = parse_compute(src).expect("parse");
        lower_compute(&module).expect("lower")
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
        let EntryKind::Compute { local_size, .. } = m.entry.kind;
        assert_eq!(local_size, [64, 1, 1]);
    }

    #[test]
    fn occupancy_hints_parse_and_lower_per_backend() {
        let src = r#"
            @workgroup_size(256)
            @launch_bounds(512)
            @waves_per_eu(1, 2)
            @min_blocks(4)
            kernel k(x: device mut buffer<f32>, id: global_id) { x[id.x] = 1.0 }
        "#;
        let m = parse_compute(src).expect("parse");
        let EntryKind::Compute {
            local_size,
            occupancy,
        } = m.entry.kind;
        assert_eq!(local_size, [256, 1, 1]);
        assert_eq!(occupancy.max_threads, Some(512));
        assert_eq!(occupancy.waves_per_eu, Some((1, Some(2))));
        assert_eq!(occupancy.min_blocks, Some(4));
        let hip = hip::lower_compute(&m).expect("hip").source;
        assert!(hip.contains("__launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(1, 2)))"));
        let cuda = cuda::lower_compute(&m).expect("cuda").source;
        assert!(cuda.contains("__launch_bounds__(512, 4)"));
        let msl = msl::lower_compute(&m).expect("msl").source;
        assert!(msl.contains("[[max_total_threads_per_threadgroup(512)]] kernel void"));
        assert!(
            parse_compute("@workgroup_size(64)\n@launch_bounds(32)\nkernel k(id: global_id) { }")
                .is_err()
        );
        assert!(parse_compute("@waves_per_eu(3, 2)\nkernel k(id: global_id) { }").is_err());
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
        let locals: HashMap<_, _> = m.entry.locals.iter().cloned().collect();
        let mut arena = m.types.clone();
        assert_eq!(locals["i"], arena.u32_id());
        assert_eq!(locals["v"], arena.f32_id());
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
    fn half_float_buffers_keep_their_element_type() {
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            kernel copy_bf16(x: device buffer<bf16>, out: device mut buffer<bf16>, id: global_id) {
                out[id.x] = x[id.x]
            }
        "#,
        )
        .expect("parse");
        let bf16_id = {
            let mut t = module.types.clone();
            t.bf16_id()
        };
        for p in &module.entry.params {
            if let ParamKind::Buffer { elem, .. } = p.kind {
                assert_eq!(elem, bf16_id, "buffer<bf16> param `{}`", p.name);
            }
        }
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            kernel copy_f16(x: device buffer<f16>, out: device mut buffer<f16>, id: global_id) {
                out[id.x] = x[id.x]
            }
        "#,
        )
        .expect("parse");
        let f16_id = {
            let mut t = module.types.clone();
            t.f16_id()
        };
        for p in &module.entry.params {
            if let ParamKind::Buffer { elem, .. } = p.kind {
                assert_eq!(elem, f16_id, "buffer<f16> param `{}`", p.name);
            }
        }
    }

    #[test]
    fn bf16_buffer_rejects_non_f32_assignment() {
        let src = r#"
            @workgroup_size(1)
            kernel bad(out: device mut buffer<bf16>) {
                out[0] = 1u32
            }
        "#;
        let e = parse_compute(src).unwrap_err();
        assert!(e.msg.contains("incompatible value type"), "got: {}", e.msg);
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
        let err = lower_compute(&module).unwrap_err();
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
}
