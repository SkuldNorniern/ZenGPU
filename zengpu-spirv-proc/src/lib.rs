//! Proc-macro internals. Use `zengpu_spirv` or `zengpu`, not this crate directly.

extern crate proc_macro;

mod ast;
mod types;

use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{
    Attribute, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input, spanned::Spanned,
};

use ast::{Stage, ZslEntryPoint};

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

fn errors_to_ts(errors: Vec<(Span, String)>) -> proc_macro2::TokenStream {
    errors
        .into_iter()
        .map(|(span, msg)| syn::Error::new(span, msg).to_compile_error())
        .collect()
}

/// Internal proc-macro for ZSL input. Checks the entry point's param and return types (Vec2/3/4,
/// Mat4, Buf<T>, BufMut<T>, f32).
#[proc_macro]
pub fn zsl_spirv(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ZslInput);

    let stage = match stage_from_attrs(&parsed.attrs) {
        Ok(s) => s,
        Err((span, msg)) => {
            return syn::Error::new(span, msg).to_compile_error().into();
        }
    };

    let entry = match ZslEntryPoint::from_fn(stage, &parsed.func) {
        Ok(e) => e,
        Err(errs) => return errors_to_ts(errs).into(),
    };

    let fn_name = &entry.ident;
    let stage_name = entry.stage.name();
    let param_summary: Vec<String> = entry
        .params
        .iter()
        .map(|p| format!("{}: {}", p.ident, p.ty.display()))
        .collect();

    let msg = format!(
        "zengpu_spirv!: ZSL SPIR-V lowering not yet implemented (step 4). \
         Parsed `{fn_name}` ({stage_name}): params: [{}] → {}",
        param_summary.join(", "),
        entry.ret.display(),
    );

    syn::Error::new(parsed.func.sig.fn_token.span(), msg)
        .to_compile_error()
        .into()
}
