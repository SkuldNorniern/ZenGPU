//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod backend;
mod frontend;
mod ir;

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span};
use quote::quote;
use syn::{Attribute, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input};

use frontend::ast::{Stage, ZslEntryPoint};

/// ZSL input: outer attribute(s) + fn item.
struct ZslInput {
    attrs: Vec<Attribute>,
    func: ItemFn,
}

impl Parse for ZslInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mut func: ItemFn = input.parse()?;
        func.attrs.splice(0..0, attrs.clone());
        Ok(ZslInput { attrs, func })
    }
}

fn errors_to_ts(errors: Vec<(Span, String)>) -> proc_macro2::TokenStream {
    errors
        .into_iter()
        .map(|(span, msg)| syn::Error::new(span, msg).to_compile_error())
        .collect()
}

/// Derive `to_scalars()` for a push-constant struct.
///
/// Every field must be `u32`, `i32`, or `f32`. The generated method returns a
/// fixed-size array of `zengpu_hal::Scalar` values in field-declaration order,
/// suitable for passing as `Bindings::scalars` in a dispatch call.
///
/// Import via `use zengpu_spirv::ZslPushConst;`.
#[proc_macro_derive(ZslPushConst)]
pub fn derive_zsl_push_const(input: TokenStream) -> TokenStream {
    match push_const_impl(input) {
        Ok(ts) => ts,
        Err(msg) => compile_error_tokens(&msg),
    }
}

/// Hand-rolled `#[derive(ZslPushConst)]` over the builtin `proc_macro` token
/// trees — no `syn`/`quote`. Generates `to_scalars()` returning a fixed array of
/// `Scalar` in field order; `ZslMat4` expands to 16 column-major `Scalar::F32`.
fn push_const_impl(input: TokenStream) -> Result<TokenStream, String> {
    use proc_macro::{Delimiter, TokenTree};

    // Find `struct <Name> { <fields> }`, skipping attributes/visibility/generics.
    let mut it = input.into_iter();
    let mut name: Option<String> = None;
    let mut fields: Option<proc_macro::Group> = None;
    while let Some(tt) = it.next() {
        if let TokenTree::Ident(id) = &tt {
            if id.to_string() == "struct" {
                match it.next() {
                    Some(TokenTree::Ident(n)) => name = Some(n.to_string()),
                    _ => return Err("ZslPushConst: expected a struct name".into()),
                }
                for tt2 in it.by_ref() {
                    if let TokenTree::Group(g) = tt2 {
                        if g.delimiter() == Delimiter::Brace {
                            fields = Some(g);
                        }
                        break;
                    }
                }
                break;
            }
        }
    }
    let name = name.ok_or("ZslPushConst only supports structs")?;
    let fields = fields.ok_or("ZslPushConst requires a struct with named fields")?;

    // Parse `name : type ,` repeated. Only the type's leading ident is needed.
    let mut exprs: Vec<String> = Vec::new();
    let mut toks = fields.stream().into_iter().peekable();
    loop {
        let fname = match toks.next() {
            Some(TokenTree::Ident(id)) => id.to_string(),
            Some(_) => return Err("ZslPushConst: expected a field name".into()),
            None => break,
        };
        match toks.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => return Err("ZslPushConst: expected `:` after field name".into()),
        }
        let mut type_name = String::new();
        while let Some(tt) = toks.peek() {
            if matches!(tt, TokenTree::Punct(p) if p.as_char() == ',') {
                toks.next();
                break;
            }
            let tt = toks.next().unwrap();
            if type_name.is_empty() {
                if let TokenTree::Ident(id) = &tt {
                    type_name = id.to_string();
                }
            }
        }
        match type_name.as_str() {
            "u32" => exprs.push(format!("::zengpu_spirv::_zsl_priv::Scalar::U32(self.{fname})")),
            "i32" => exprs.push(format!("::zengpu_spirv::_zsl_priv::Scalar::I32(self.{fname})")),
            "f32" => exprs.push(format!("::zengpu_spirv::_zsl_priv::Scalar::F32(self.{fname})")),
            "ZslMat4" => {
                for i in 0..16 {
                    exprs.push(format!(
                        "::zengpu_spirv::_zsl_priv::Scalar::F32(self.{fname}.0[{i}])"
                    ));
                }
            }
            other => {
                return Err(format!(
                    "ZslPushConst fields must be u32, i32, f32, or ZslMat4; got `{other}`"
                ));
            }
        }
    }

    let n = exprs.len();
    let body = exprs.join(", ");
    let code = format!(
        "impl {name} {{ \
            pub fn to_scalars(&self) -> [::zengpu_spirv::_zsl_priv::Scalar; {n}] {{ [{body}] }} \
        }}"
    );
    code.parse()
        .map_err(|_| "ZslPushConst: generated code failed to parse".to_string())
}

/// Internal proc-macro invoked by `zengpu_spirv!` when ZSL input is detected.
#[proc_macro]
pub fn zsl_spirv(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ZslInput);

    let stage = match frontend::parse::stage_from_attrs(&parsed.attrs) {
        Ok(s) => s,
        Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
    };

    let entry = match ZslEntryPoint::from_fn(stage, &parsed.func) {
        Ok(e) => e,
        Err(errs) => return errors_to_ts(errs).into(),
    };

    match stage {
        Stage::Compute => {
            let local_size = match frontend::parse::parse_local_size(&parsed.attrs) {
                Ok(ls) => ls,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let module = match ir::build::build_compute(&entry, &parsed.func.block, local_size) {
                Ok(m) => m,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words = match backend::spirv::lower_compute(&module) {
                Ok(w) => w,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words_lit = words.iter().map(|&w| {
                let lit = Literal::u32_suffixed(w);
                quote!(#lit)
            });
            quote! { &[#(#words_lit),*] }.into()
        }
        Stage::Vertex => {
            let words = match backend::spirv::lower_vertex(&entry, &parsed.func.block) {
                Ok(w) => w,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words_lit = words.iter().map(|&w| {
                let lit = Literal::u32_suffixed(w);
                quote!(#lit)
            });
            quote! { &[#(#words_lit),*] }.into()
        }
        Stage::Fragment => {
            let words = match backend::spirv::lower_fragment(&entry, &parsed.func.block) {
                Ok(w) => w,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words_lit = words.iter().map(|&w| {
                let lit = Literal::u32_suffixed(w);
                quote!(#lit)
            });
            quote! { &[#(#words_lit),*] }.into()
        }
    }
}

// ── Native ZSL macro (no syn/quote) ────────────────────────────────────────────

/// Compile **native ZSL** source to a SPIR-V word slice (`&[u32]`).
///
/// Unlike [`zsl_spirv`] (the Rust-shaped transitional form), this uses ZenGPU's
/// own lexer/parser/lowerer with no `syn`/`quote` — only the built-in
/// `proc_macro` shell. Currently supports compute kernels:
///
/// ```ignore
/// const SPV: &[u32] = zengpu_spirv::zsl!(
///     push Push { n: u32, scale: f32 }
///     @workgroup_size(64)
///     kernel scale(p: Push, src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
///         let i = id.x
///         if i < p.n { dst[i] = src[i] * p.scale }
///     }
/// );
/// ```
#[proc_macro]
pub fn zsl(input: TokenStream) -> TokenStream {
    let src = input.to_string();
    match compile_native_zsl(&src) {
        Ok(words) => words_to_slice_tokens(&words),
        Err(msg) => compile_error_tokens(&msg),
    }
}

fn compile_native_zsl(src: &str) -> Result<Vec<u32>, String> {
    use frontend::parser::Shader;
    match frontend::parser::parse_zsl(src).map_err(|e| format!("ZSL parse error: {}", e.msg))? {
        Shader::Compute(m) => backend::spirv::lower_compute(&m).map_err(|(_, msg)| msg),
        Shader::Graphics(m) => backend::spirv::graphics_ir::lower_graphics(&m),
    }
}

/// Build the `&[u32]` literal token stream without `quote`.
fn words_to_slice_tokens(words: &[u32]) -> TokenStream {
    let mut s = String::with_capacity(words.len() * 12 + 4);
    s.push_str("&[");
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&w.to_string());
        s.push_str("u32");
    }
    s.push(']');
    s.parse().expect("generated &[u32] literal must parse")
}

/// Build a `compile_error!` invocation token stream without `quote`.
fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}
