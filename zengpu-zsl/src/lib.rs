//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod frontend;
mod lower;
mod lower_graphics;
mod spirv;

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span};
use quote::quote;
use syn::{
    Attribute, DeriveInput, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input,
};

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

/// Returns one or more `Scalar` constructor expressions for a single field.
/// `ZslMat4` expands to 16 `Scalar::F32` entries (column-major).
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
        // Column-major flat [f32; 16] stored inside ZslMat4(pub [f32; 16])
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
