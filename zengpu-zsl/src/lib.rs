//! Proc-macro internals. Use `zengpu_spirv` or `zengpu`, not this crate directly.

extern crate proc_macro;

mod backend;
mod frontend;
mod ir;

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span};
use quote::quote;
use syn::{
    Attribute, DeriveInput, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input,
};

use frontend::ast::{Stage, ZslEntryPoint};

/// ZSL input: outer attributes and a fn item.
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

/// Derives `to_scalars()` for a push struct: fields `u32`, `i32` or `f32` only, in declaration
/// order. Result goes to `Bindings::scalars`. Import via `use zengpu_spirv::ZslPushConst;`.
#[proc_macro_derive(ZslPushConst)]
pub fn derive_zsl_push_const(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    push_const_impl(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn push_const_impl(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(ZslPushConst)] only supports structs",
        ));
    };

    let syn::Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(ZslPushConst)] requires named fields",
        ));
    };

    let mut exprs: Vec<proc_macro2::TokenStream> = Vec::new();
    for field in &fields.named {
        let fname = field.ident.as_ref().unwrap();
        let mut field_exprs = scalar_field_exprs(&field.ty, fname)?;
        exprs.append(&mut field_exprs);
    }

    let n = exprs.len();
    Ok(quote! {
        impl #name {
            pub fn to_scalars(&self) -> [::zengpu_spirv::_zsl_priv::Scalar; #n] {
                [#(#exprs),*]
            }
        }
    })
}

/// Builds the `Scalar` values for one field. A `ZslMat4` gives 16 `Scalar::F32` entries in
/// column-major order.
fn scalar_field_exprs(
    ty: &syn::Type,
    fname: &syn::Ident,
) -> syn::Result<Vec<proc_macro2::TokenStream>> {
    let syn::Type::Path(tp) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "ZslPushConst fields must be u32, i32, f32, or ZslMat4",
        ));
    };
    if tp.path.is_ident("u32") {
        Ok(vec![
            quote!(::zengpu_spirv::_zsl_priv::Scalar::U32(self.#fname)),
        ])
    } else if tp.path.is_ident("i32") {
        Ok(vec![
            quote!(::zengpu_spirv::_zsl_priv::Scalar::I32(self.#fname)),
        ])
    } else if tp.path.is_ident("f32") {
        Ok(vec![
            quote!(::zengpu_spirv::_zsl_priv::Scalar::F32(self.#fname)),
        ])
    } else if tp.path.is_ident("ZslMat4") {
        let entries: Vec<proc_macro2::TokenStream> = (0usize..16)
            .map(|i| quote!(::zengpu_spirv::_zsl_priv::Scalar::F32(self.#fname.0[#i])))
            .collect();
        Ok(entries)
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "ZslPushConst fields must be u32, i32, f32, or ZslMat4",
        ))
    }
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

/// Native ZSL to a SPIR-V word slice (`&[u32]`). Unlike [`zsl_spirv`] (Rust-shaped syntax), this
/// uses ZenGPU's own front end without `syn` or `quote`. Compute kernels only:
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
    let module = frontend::parser::parse_compute(src)
        .map_err(|e| format!("ZSL parse error: {}", e.msg))?;
    backend::spirv::lower_compute(&module).map_err(|(_, msg)| msg)
}

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

fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}
