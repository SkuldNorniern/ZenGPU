//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{Attribute, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input};

/// Input parsed by [`zsl_spirv`]: one or more outer attributes followed by a fn.
struct ZslInput {
    attrs: Vec<Attribute>,
    func: ItemFn,
}

impl Parse for ZslInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mut func: ItemFn = input.parse()?;
        // Move attributes from the parse input onto the fn (they arrived before it).
        func.attrs.splice(0..0, attrs.clone());
        Ok(ZslInput { attrs, func })
    }
}

fn stage_from_attrs(attrs: &[Attribute]) -> Option<&'static str> {
    for attr in attrs {
        if attr.path().is_ident("vertex") {
            return Some("vertex");
        }
        if attr.path().is_ident("fragment") {
            return Some("fragment");
        }
        if attr.path().is_ident("compute") {
            return Some("compute");
        }
    }
    None
}

/// Internal proc-macro invoked by the `zengpu_spirv!` routing macro when ZSL
/// input (a function annotated with `#[vertex]`, `#[fragment]`, or `#[compute]`)
/// is detected.
///
/// **Step 2 stub** — validates the stage attribute and entry-point signature.
/// Actual SPIR-V lowering is implemented in step 3.
#[proc_macro]
pub fn zsl_spirv(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ZslInput);

    let stage = match stage_from_attrs(&parsed.attrs) {
        Some(s) => s,
        None => {
            return syn::Error::new(
                Span::call_site(),
                "zengpu_spirv!: ZSL entry point must have a #[vertex], #[fragment], \
                 or #[compute] attribute",
            )
            .to_compile_error()
            .into();
        }
    };

    let fn_name = &parsed.func.sig.ident;
    let msg = format!(
        "zengpu_spirv!: ZSL codegen not yet implemented (step 3). \
         Entry point `{fn_name}` with stage `{stage}` was parsed successfully. \
         Use GLSL string input until ZSL lowering lands."
    );

    syn::Error::new(Span::call_site(), msg)
        .to_compile_error()
        .into()
}
