//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod ast;
mod types;

use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{
    Attribute, ItemFn, parse::Parse, parse::ParseStream, parse_macro_input, spanned::Spanned,
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

fn errors_to_ts(errors: Vec<(Span, String)>) -> proc_macro2::TokenStream {
    errors
        .into_iter()
        .map(|(span, msg)| syn::Error::new(span, msg).to_compile_error())
        .collect()
}

/// Internal proc-macro invoked by `zengpu_spirv!` when ZSL input is detected.
///
/// **Step 3**: parses the entry-point signature and validates all parameter and
/// return types against the ZSL type system (Vec2/3/4, Mat4, Buf<T>, BufMut<T>,
/// f32, …). SPIR-V lowering is step 4.
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

    // Step 3 complete: entry point parsed and type-checked.
    // Step 4 will lower `entry` to SPIR-V words and emit `&[u32]`.
    let fn_name = &entry.ident;
    let stage_name = entry.stage.name();
    let param_summary: Vec<String> = entry
        .params
        .iter()
        .map(|p| format!("{}: {}", p.ident, p.ty.display()))
        .collect();

    let msg = format!(
        "zengpu_spirv!: ZSL SPIR-V lowering not yet implemented (step 4). \
         Parsed `{fn_name}` ({stage_name}) — params: [{}] → {}",
        param_summary.join(", "),
        entry.ret.display(),
    );

    syn::Error::new(parsed.func.sig.fn_token.span(), msg)
        .to_compile_error()
        .into()
}
