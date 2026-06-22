//! ZSL front-end parsing: `syn` items/attributes → [`ZslEntryPoint`] AST.
//!
//! Keeps the syn-shaped parsing separate from the AST data structures
//! (`ast.rs`) and from token emission (`lib.rs`). Backend-neutral.

use proc_macro2::Span;
use syn::{Attribute, FnArg, ItemFn, Meta, ReturnType, spanned::Spanned};

use crate::frontend::ast::{Stage, ZslEntryPoint, ZslParam};
use crate::frontend::types::ZslType;

/// Determine the shader stage from the entry point's outer attributes.
pub fn stage_from_attrs(attrs: &[Attribute]) -> Result<Stage, (Span, String)> {
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
pub fn parse_local_size(attrs: &[Attribute]) -> Result<(u32, u32, u32), (Span, String)> {
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

impl ZslEntryPoint {
    /// Parse a [`ItemFn`] that has already been annotated with a stage
    /// attribute. Returns errors with spans attached.
    pub fn from_fn(stage: Stage, func: &ItemFn) -> Result<Self, Vec<(Span, String)>> {
        let mut errors: Vec<(Span, String)> = Vec::new();
        let mut params: Vec<ZslParam> = Vec::new();

        for arg in &func.sig.inputs {
            match arg {
                FnArg::Receiver(r) => {
                    errors.push((r.span(), "ZSL entry points cannot take `self`".to_string()));
                }
                FnArg::Typed(pat_ty) => {
                    let ident = match &*pat_ty.pat {
                        syn::Pat::Ident(p) => p.ident.clone(),
                        other => {
                            errors.push((
                                other.span(),
                                "ZSL parameter must be a simple identifier".to_string(),
                            ));
                            continue;
                        }
                    };

                    let ty = match ZslType::from_syn(&pat_ty.ty) {
                        Ok(t) => t,
                        Err((span, msg)) => {
                            errors.push((span, msg));
                            continue;
                        }
                    };

                    let (location, builtin) = parse_param_attrs(&pat_ty.attrs, &mut errors);

                    // Buffer element type validity is enforced by BufElem at parse time.

                    params.push(ZslParam {
                        ident,
                        ty,
                        location,
                        builtin,
                    });
                }
            }
        }

        let ret = match &func.sig.output {
            ReturnType::Default => ZslType::Void,
            ReturnType::Type(_, ty) => match ZslType::from_syn(ty) {
                Ok(t) => t,
                Err((span, msg)) => {
                    errors.push((span, msg));
                    ZslType::Void
                }
            },
        };

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(ZslEntryPoint { stage, params, ret })
    }
}

/// Extract `#[location(N)]` and `#[builtin(name)]` from parameter attributes.
fn parse_param_attrs(
    attrs: &[syn::Attribute],
    errors: &mut Vec<(Span, String)>,
) -> (Option<u32>, Option<String>) {
    let mut location = None;
    let mut builtin = None;

    for attr in attrs {
        if attr.path().is_ident("location") {
            match attr.parse_args::<syn::LitInt>() {
                Ok(lit) => match lit.base10_parse::<u32>() {
                    Ok(n) => location = Some(n),
                    Err(_) => errors.push((lit.span(), "location index must be a u32".to_string())),
                },
                Err(_) => errors.push((
                    attr.span(),
                    "#[location(N)] expects a u32 literal".to_string(),
                )),
            }
        } else if attr.path().is_ident("builtin") {
            match &attr.meta {
                Meta::List(list) => {
                    let tokens = list.tokens.to_string();
                    let name = tokens.trim().to_string();
                    if name.is_empty() {
                        errors.push((
                            attr.span(),
                            "#[builtin(name)] expects a built-in name".to_string(),
                        ));
                    } else {
                        builtin = Some(name);
                    }
                }
                _ => errors.push((
                    attr.span(),
                    "#[builtin(name)] expects a built-in name".to_string(),
                )),
            }
        }
    }

    (location, builtin)
}
