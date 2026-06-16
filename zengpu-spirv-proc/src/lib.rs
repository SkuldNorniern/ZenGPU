//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod ast;
mod lower;
mod spirv;
mod types;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Attribute, ItemFn, Meta, parse::Parse, parse::ParseStream, parse_macro_input, spanned::Spanned,
};

use ast::{Stage, ZslEntryPoint};

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

fn stage_from_attrs(attrs: &[Attribute]) -> Result<Stage, (Span, String)> {
    for attr in attrs {
        if attr.path().is_ident("vertex") {
            return Ok(Stage::Vertex);
        }
        if attr.path().is_ident("fragment") {
            return Ok(Stage::Fragment);
        }
        if attr.path().is_ident("compute") {
            return Ok(Stage::Compute);
        }
    }
    Err((
        Span::call_site(),
        "zengpu_spirv!: ZSL entry point must have a \
         #[vertex], #[fragment], or #[compute] attribute"
            .to_string(),
    ))
}

/// Parse `#[compute]` or `#[compute(local_size_x = N)]`.
/// Returns `(x, y, z)` defaulting to 1 for unspecified axes.
fn parse_local_size(attrs: &[Attribute]) -> Result<(u32, u32, u32), (Span, String)> {
    for attr in attrs {
        if !attr.path().is_ident("compute") {
            continue;
        }
        // bare `#[compute]` → (1, 1, 1)
        let Meta::List(list) = &attr.meta else {
            return Ok((1, 1, 1));
        };
        // Parse `local_size_x = N, local_size_y = N, local_size_z = N`
        let mut x = 1u32;
        let mut y = 1u32;
        let mut z = 1u32;
        let nested: syn::punctuated::Punctuated<syn::MetaNameValue, syn::Token![,]> = match list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::MetaNameValue, syn::Token![,]>::parse_terminated,
            ) {
            Ok(p) => p,
            Err(e) => {
                return Err((
                    attr.span(),
                    format!("#[compute(...)]: expected `local_size_x = N` pairs: {e}"),
                ));
            }
        };
        for nv in &nested {
            let key = nv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            let val: u32 = match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(li),
                    ..
                }) => li.base10_parse().map_err(|_| {
                    (
                        nv.value.span(),
                        "local_size value must be a u32 literal".to_string(),
                    )
                })?,
                _ => {
                    return Err((
                        nv.value.span(),
                        "local_size value must be an integer literal".to_string(),
                    ));
                }
            };
            match key.as_str() {
                "local_size_x" => x = val,
                "local_size_y" => y = val,
                "local_size_z" => z = val,
                other => {
                    return Err((
                        nv.path.span(),
                        format!("unknown compute attribute `{other}`; expected local_size_x/y/z"),
                    ));
                }
            }
        }
        return Ok((x, y, z));
    }
    Ok((1, 1, 1))
}

fn errors_to_ts(errors: Vec<(Span, String)>) -> proc_macro2::TokenStream {
    errors
        .into_iter()
        .map(|(span, msg)| syn::Error::new(span, msg).to_compile_error())
        .collect()
}

/// Internal proc-macro invoked by `zengpu_spirv!` when ZSL input is detected.
#[proc_macro]
pub fn zsl_spirv(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ZslInput);

    let stage = match stage_from_attrs(&parsed.attrs) {
        Ok(s) => s,
        Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
    };

    let entry = match ZslEntryPoint::from_fn(stage, &parsed.func) {
        Ok(e) => e,
        Err(errs) => return errors_to_ts(errs).into(),
    };

    match stage {
        Stage::Compute => {
            let local_size = match parse_local_size(&parsed.attrs) {
                Ok(ls) => ls,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words = match lower::lower_compute(&entry, &parsed.func.block, local_size) {
                Ok(w) => w,
                Err((span, msg)) => return syn::Error::new(span, msg).to_compile_error().into(),
            };
            let words_lit = words.iter().map(|&w| quote!(#w u32));
            quote! { &[#(#words_lit),*] }.into()
        }
        Stage::Vertex | Stage::Fragment => {
            let name = &entry.ident;
            syn::Error::new(
                parsed.func.sig.fn_token.span(),
                format!(
                    "zengpu_spirv!: ZSL {stage} shader lowering not yet implemented (step 5). \
                     Parsed `{name}`.",
                    stage = stage.name(),
                ),
            )
            .to_compile_error()
            .into()
        }
    }
}
