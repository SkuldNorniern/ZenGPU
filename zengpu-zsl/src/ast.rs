//! //! Entry-point AST for ZSL. Holds data only; parsing lives elsewhere.

use proc_macro2::Span;
use syn::{FnArg, Ident, ItemFn, Meta, ReturnType, spanned::Spanned};

use crate::types::ZslType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
            Stage::Compute => "compute",
        }
    }
}

/// Entry-point param: resolved ZSL type, plus optional location/builtin binding.
#[derive(Debug)]
pub struct ZslParam {
    pub ident: Ident,
    pub ty: ZslType,
    #[allow(dead_code)]
    pub location: Option<u32>,
    #[allow(dead_code)]
    pub builtin: Option<String>,
}

#[derive(Debug)]
pub struct ZslEntryPoint {
    #[allow(dead_code)]
    pub stage: Stage,
    pub ident: Ident,
    pub params: Vec<ZslParam>,
    #[allow(dead_code)]
    pub ret: ZslType,
}

impl ZslEntryPoint {
    /// Parse an [`ItemFn`] that already has a stage attribute. Errors carry spans.
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

                    if matches!(ty, ZslType::Buf(_) | ZslType::BufMut(_)) && stage != Stage::Compute
                    {
                        errors.push((
                            pat_ty.ty.span(),
                            format!(
                                "`{name}` is only valid in `#[compute]` entry points",
                                name = ty.display()
                            ),
                        ));
                    }

                    if let ZslType::Buf(inner) | ZslType::BufMut(inner) = &ty {
                        if !inner.is_buffer_elem() {
                            errors.push((
                                pat_ty.ty.span(),
                                format!(
                                    "unsupported buffer element type `{}`. \
                                     Buffer elements must be scalar, vector, or matrix types.",
                                    inner.display()
                                ),
                            ));
                        }
                    }

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

        Ok(ZslEntryPoint {
            stage,
            ident: func.sig.ident.clone(),
            params,
            ret,
        })
    }
}

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
