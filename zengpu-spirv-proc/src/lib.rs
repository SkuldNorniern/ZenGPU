//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod ast;
mod lower;
mod lower_graphics;
mod spirv;
mod types;

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span};
use quote::quote;
use syn::{
    Attribute, DeriveInput, ItemFn, Meta, parse::Parse, parse::ParseStream, parse_macro_input,
    spanned::Spanned,
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

/// Derive `to_scalars()` for a push-constant struct.
///
/// Every field must be `u32`, `i32`, or `f32`. The generated method returns a
/// fixed-size array of [`zengpu_hal::Scalar`] in field-declaration order,
/// suitable for passing as `Bindings::scalars` in a dispatch call.
///
/// Import via `use zengpu_spirv::ZslPushConst;`.
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
        let expr = scalar_field_expr(&field.ty, fname)?;
        exprs.push(expr);
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

fn scalar_field_expr(ty: &syn::Type, fname: &syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    let syn::Type::Path(tp) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "ZslPushConst fields must be u32, i32, or f32",
        ));
    };
    if tp.path.is_ident("u32") {
        Ok(quote!(::zengpu_spirv::_zsl_priv::Scalar::U32(self.#fname)))
    } else if tp.path.is_ident("i32") {
        Ok(quote!(::zengpu_spirv::_zsl_priv::Scalar::I32(self.#fname)))
    } else if tp.path.is_ident("f32") {
        Ok(quote!(::zengpu_spirv::_zsl_priv::Scalar::F32(self.#fname)))
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "ZslPushConst fields must be u32, i32, or f32",
        ))
    }
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
            let words_lit = words.iter().map(|&w| {
                let lit = Literal::u32_suffixed(w);
                quote!(#lit)
            });
            quote! { &[#(#words_lit),*] }.into()
        }
        Stage::Vertex => {
            let words = match lower_graphics::lower_vertex(&entry, &parsed.func.block) {
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
            let words = match lower_graphics::lower_fragment(&entry, &parsed.func.block) {
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
