//! ZSL built-in type system — maps Rust-style type paths to SPIR-V types.

use proc_macro2::Span;
use syn::{GenericArgument, PathArguments, Type, TypePath};

/// Scalar and vector types allowed as `Buf<T>` / `BufMut<T>` element types.
///
/// Using a separate flat enum avoids `Box<ZslType>` in the recursive `ZslType`
/// while statically encoding which types are valid buffer elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufElem {
    F32,
    I32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    IVec2,
    IVec3,
    IVec4,
    UVec2,
    UVec3,
    UVec4,
    Mat2,
    Mat3,
    Mat4,
}

impl BufElem {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(BufElem::F32),
            "i32" => Some(BufElem::I32),
            "u32" => Some(BufElem::U32),
            "Vec2" => Some(BufElem::Vec2),
            "Vec3" => Some(BufElem::Vec3),
            "Vec4" => Some(BufElem::Vec4),
            "IVec2" => Some(BufElem::IVec2),
            "IVec3" => Some(BufElem::IVec3),
            "IVec4" => Some(BufElem::IVec4),
            "UVec2" => Some(BufElem::UVec2),
            "UVec3" => Some(BufElem::UVec3),
            "UVec4" => Some(BufElem::UVec4),
            "Mat2" => Some(BufElem::Mat2),
            "Mat3" => Some(BufElem::Mat3),
            "Mat4" => Some(BufElem::Mat4),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn display(self) -> &'static str {
        match self {
            BufElem::F32 => "f32",
            BufElem::I32 => "i32",
            BufElem::U32 => "u32",
            BufElem::Vec2 => "Vec2",
            BufElem::Vec3 => "Vec3",
            BufElem::Vec4 => "Vec4",
            BufElem::IVec2 => "IVec2",
            BufElem::IVec3 => "IVec3",
            BufElem::IVec4 => "IVec4",
            BufElem::UVec2 => "UVec2",
            BufElem::UVec3 => "UVec3",
            BufElem::UVec4 => "UVec4",
            BufElem::Mat2 => "Mat2",
            BufElem::Mat3 => "Mat3",
            BufElem::Mat4 => "Mat4",
        }
    }
}

/// Every type a ZSL shader function can use as a parameter or return type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZslType {
    // Scalars
    F32,
    I32,
    U32,
    Bool,
    // Float vectors
    Vec2,
    Vec3,
    Vec4,
    // Integer vectors
    IVec2,
    IVec3,
    IVec4,
    UVec2,
    UVec3,
    UVec4,
    BVec2,
    BVec3,
    BVec4,
    // Float matrices (column-major)
    Mat2,
    Mat3,
    Mat4,
    // Compute storage buffers — element type is a flat BufElem (no Box needed)
    Buf(BufElem),
    BufMut(BufElem),
    // Void (unit return type)
    Void,
    // Tuple return: (Vec4, T1, T2, ...) — vertex position + varyings
    Tuple(Vec<ZslType>),
}

impl ZslType {
    /// Parse a [`syn::Type`] into a [`ZslType`]. Returns an error span+message
    /// if the type is not a known ZSL type.
    pub fn from_syn(ty: &Type) -> Result<Self, (Span, String)> {
        match ty {
            Type::Path(TypePath { qself: None, path }) => {
                let seg = path
                    .segments
                    .last()
                    .ok_or_else(|| (Span::call_site(), "empty type path".to_string()))?;
                let name = seg.ident.to_string();
                match name.as_str() {
                    "f32" => return Ok(ZslType::F32),
                    "i32" => return Ok(ZslType::I32),
                    "u32" => return Ok(ZslType::U32),
                    "bool" => return Ok(ZslType::Bool),
                    "Vec2" => return Ok(ZslType::Vec2),
                    "Vec3" => return Ok(ZslType::Vec3),
                    "Vec4" => return Ok(ZslType::Vec4),
                    "IVec2" => return Ok(ZslType::IVec2),
                    "IVec3" => return Ok(ZslType::IVec3),
                    "IVec4" => return Ok(ZslType::IVec4),
                    "UVec2" => return Ok(ZslType::UVec2),
                    "UVec3" => return Ok(ZslType::UVec3),
                    "UVec4" => return Ok(ZslType::UVec4),
                    "BVec2" => return Ok(ZslType::BVec2),
                    "BVec3" => return Ok(ZslType::BVec3),
                    "BVec4" => return Ok(ZslType::BVec4),
                    "Mat2" => return Ok(ZslType::Mat2),
                    "Mat3" => return Ok(ZslType::Mat3),
                    "Mat4" => return Ok(ZslType::Mat4),
                    "Buf" | "BufMut" => {
                        let inner =
                            extract_single_generic(&seg.arguments, &name, seg.ident.span())?;
                        let elem = parse_buf_elem(inner)?;
                        return Ok(if name == "Buf" {
                            ZslType::Buf(elem)
                        } else {
                            ZslType::BufMut(elem)
                        });
                    }
                    _ => {}
                }
                Err((
                    seg.ident.span(),
                    format!(
                        "unknown ZSL type `{name}`. \
                         Built-in types: f32, i32, u32, bool, \
                         Vec2/3/4, IVec2/3/4, UVec2/3/4, BVec2/3/4, \
                         Mat2/3/4, Buf<T>, BufMut<T>"
                    ),
                ))
            }
            Type::Tuple(t) if t.elems.is_empty() => Ok(ZslType::Void),
            Type::Tuple(t) => {
                let elems: Result<Vec<ZslType>, _> =
                    t.elems.iter().map(ZslType::from_syn).collect();
                Ok(ZslType::Tuple(elems?))
            }
            _ => Err((
                Span::call_site(),
                format!(
                    "unsupported ZSL type form `{}`. \
                     ZSL types must be simple paths (Vec4, Buf<f32>, …) or `()`.",
                    quote::quote!(#ty)
                ),
            )),
        }
    }

    /// Human-readable name, matching the ZSL source spelling.
    pub fn display(&self) -> &'static str {
        match self {
            ZslType::F32 => "f32",
            ZslType::I32 => "i32",
            ZslType::U32 => "u32",
            ZslType::Bool => "bool",
            ZslType::Vec2 => "Vec2",
            ZslType::Vec3 => "Vec3",
            ZslType::Vec4 => "Vec4",
            ZslType::IVec2 => "IVec2",
            ZslType::IVec3 => "IVec3",
            ZslType::IVec4 => "IVec4",
            ZslType::UVec2 => "UVec2",
            ZslType::UVec3 => "UVec3",
            ZslType::UVec4 => "UVec4",
            ZslType::BVec2 => "BVec2",
            ZslType::BVec3 => "BVec3",
            ZslType::BVec4 => "BVec4",
            ZslType::Mat2 => "Mat2",
            ZslType::Mat3 => "Mat3",
            ZslType::Mat4 => "Mat4",
            ZslType::Buf(_) => "Buf<T>",
            ZslType::BufMut(_) => "BufMut<T>",
            ZslType::Void => "()",
            ZslType::Tuple(_) => "(..)",
        }
    }
}

/// Parse a `syn::Type` as a buffer element type. Returns an error if the type
/// is not one of the allowed scalar / vector / matrix variants.
fn parse_buf_elem(ty: &Type) -> Result<BufElem, (Span, String)> {
    let Type::Path(TypePath { qself: None, path }) = ty else {
        return Err((
            Span::call_site(),
            "buffer element type must be a simple type (f32, Vec4, …)".into(),
        ));
    };
    let seg = path
        .segments
        .last()
        .ok_or_else(|| (Span::call_site(), "empty type path".to_string()))?;
    let name = seg.ident.to_string();
    BufElem::from_name(&name).ok_or_else(|| {
        (
            seg.ident.span(),
            format!(
                "unsupported buffer element type `{name}`. \
                 Buffer elements must be scalar, vector, or matrix types."
            ),
        )
    })
}

fn extract_single_generic<'a>(
    args: &'a PathArguments,
    name: &str,
    span: Span,
) -> Result<&'a Type, (Span, String)> {
    let PathArguments::AngleBracketed(ab) = args else {
        return Err((
            span,
            format!("`{name}` requires exactly one type argument, e.g. `{name}<f32>`"),
        ));
    };
    let mut types = ab.args.iter().filter_map(|a| {
        if let GenericArgument::Type(t) = a {
            Some(t)
        } else {
            None
        }
    });
    let first = types.next().ok_or_else(|| {
        (
            span,
            format!("`{name}` requires a type argument, e.g. `{name}<f32>`"),
        )
    })?;
    if types.next().is_some() {
        return Err((span, format!("`{name}` takes exactly one type argument")));
    }
    Ok(first)
}
