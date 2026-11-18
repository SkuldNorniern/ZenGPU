//! User names a C-like backend (MSL, CUDA, HIP) cannot emit as they are: keywords, the type and
//! library names its output uses, and its own names. Such a name gets a free spelling in a copy
//! of the module that backend lowers, so reflection and diagnostics keep the names as written.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::ir::node::{FragSrc, IrExpr, IrStmt};
use crate::ir::{GraphicsModule, Module};

const KEYWORDS: &[&str] = &[
    // C++
    "alignas",
    "alignof",
    "and",
    "asm",
    "auto",
    "bitand",
    "bitor",
    "break",
    "case",
    "catch",
    "class",
    "compl",
    "const",
    "const_cast",
    "constexpr",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "nullptr",
    "operator",
    "or",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "return",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "volatile",
    "while",
    "xor",
    // Metal and CUDA qualifiers and namespaces
    "constant",
    "device",
    "thread",
    "threadgroup",
    "threadgroup_imageblock",
    "ray_data",
    "object_data",
    "kernel",
    "vertex",
    "fragment",
    "visible",
    "stitchable",
    "metal",
    "std",
    "access",
    "sampler",
    "texture",
    "array",
    "atomic",
    "vec",
    "matrix",
    "packed",
    "simd",
    "mem_flags",
    "address",
    "filter",
    "coord",
    "uint3",
    "dim3",
    "blockIdx",
    "threadIdx",
    "blockDim",
    "gridDim",
    "warpSize",
    // names the backends emit themselves
    "pc",
    "in",
    "out",
    "Push",
    "StageIn",
    // library functions the backends call
    "abs",
    "fabs",
    "fabsf",
    "acos",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "ceilf",
    "clamp",
    "cos",
    "cosf",
    "cosh",
    "cross",
    "distance",
    "dot",
    "exp",
    "exp2",
    "expf",
    "floor",
    "floorf",
    "fma",
    "fmaf",
    "fmax",
    "fmaxf",
    "fmin",
    "fminf",
    "fmod",
    "fmodf",
    "fract",
    "isfinite",
    "isinf",
    "isnan",
    "length",
    "log",
    "log2",
    "logf",
    "max",
    "min",
    "mix",
    "normalize",
    "pow",
    "powf",
    "rint",
    "rintf",
    "round",
    "rsqrt",
    "rsqrtf",
    "saturate",
    "select",
    "sign",
    "sin",
    "sinf",
    "sinh",
    "smoothstep",
    "sqrt",
    "sqrtf",
    "step",
    "tan",
    "tanf",
    "tanh",
    "tanhf",
    "trunc",
    "as_type",
    "popcount",
    "clz",
    "ctz",
    "any",
    "all",
    "read",
    "write",
    "sample",
    "sample_compare",
    "discard_fragment",
    "make_float4",
    "make_float2",
];

/// Base type names; with a lane count (`float4`) or a matrix shape (`float3x3`) too.
const TYPES: &[&str] = &[
    "void",
    "bool",
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "half",
    "float",
    "double",
    "bfloat",
    "size_t",
    "ptrdiff_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
];

/// Names starting so belong to the backend or the compiler.
fn prefixed(name: &str) -> bool {
    ["zsl_", "Zsl", "ZSL_", "__", "simd_", "atomic_"]
        .iter()
        .any(|p| name.starts_with(p))
        || name.starts_with('_') && name[1..].starts_with(|c: char| c.is_ascii_uppercase())
}

fn type_name(name: &str) -> bool {
    let name = name.strip_prefix("packed_").unwrap_or(name);
    TYPES.iter().any(|base| {
        let Some(rest) = name.strip_prefix(base) else {
            return false;
        };
        let lanes = |s: &str| matches!(s, "2" | "3" | "4");
        rest.is_empty()
            || lanes(rest)
            || rest
                .split_once('x')
                .is_some_and(|(a, b)| lanes(a) && lanes(b))
    })
}

pub fn taken(name: &str) -> bool {
    prefixed(name) || type_name(name) || KEYWORDS.contains(&name)
}

/// A free spelling for every taken name in `names`.
fn spellings<'a>(names: impl Iterator<Item = &'a str>) -> HashMap<String, String> {
    let names: HashSet<&str> = names.collect();
    let mut map = HashMap::new();
    for &name in &names {
        if !taken(name) {
            continue;
        }
        let mut new = if prefixed(name) {
            format!("u{name}")
        } else {
            format!("{name}_")
        };
        while taken(&new) || names.contains(new.as_str()) || map.values().any(|v| v == &new) {
            new.push('_');
        }
        map.insert(name.to_string(), new);
    }
    map
}

fn rename(name: &mut String, map: &HashMap<String, String>) {
    if let Some(new) = map.get(name.as_str()) {
        name.clone_from(new);
    }
}

fn stmt_names<'a>(stmts: &'a [IrStmt], out: &mut Vec<&'a str>) {
    for s in stmts {
        match s {
            IrStmt::For { var, body, .. } => {
                out.push(var);
                stmt_names(body, out);
            }
            IrStmt::If { then, else_, .. } => {
                stmt_names(then, out);
                if let Some(e) = else_ {
                    stmt_names(e, out);
                }
            }
            IrStmt::Let { name, .. } => out.push(name),
            IrStmt::Atomic { old: Some(old), .. } => out.push(old),
            _ => {}
        }
    }
}

fn frag_src(src: &mut FragSrc, map: &HashMap<String, String>) {
    match src {
        FragSrc::Buffer(n) | FragSrc::Shared(n) => rename(n, map),
    }
}

fn expr(e: &mut IrExpr, map: &HashMap<String, String>) {
    match e {
        IrExpr::LitU32(_)
        | IrExpr::LitU64(_)
        | IrExpr::LitF32(_)
        | IrExpr::GlobalId(_)
        | IrExpr::LocalId(_)
        | IrExpr::GroupId(_) => {}
        IrExpr::Local(n) | IrExpr::ScalarParam(n) | IrExpr::Input(n) => rename(n, map),
        IrExpr::BufferLoad { buf: n, index } | IrExpr::SharedLoad { name: n, index } => {
            rename(n, map);
            expr(index, map);
        }
        IrExpr::Index { array, index } => {
            rename(array, map);
            expr(index, map);
        }
        IrExpr::Texel { tex, args, .. } => {
            rename(tex, map);
            args.iter_mut().for_each(|a| expr(a, map));
        }
        IrExpr::TexSize { tex, .. } => rename(tex, map),
        IrExpr::FieldAccess { base, .. } | IrExpr::Swizzle { base, .. } | IrExpr::Neg(base) => {
            expr(base, map);
        }
        IrExpr::ArrayLit(args) | IrExpr::Construct { args, .. } | IrExpr::Builtin { args, .. } => {
            args.iter_mut().for_each(|a| expr(a, map));
        }
        IrExpr::Sample { index, uv, lod } => {
            expr(index, map);
            expr(uv, map);
            if let Some(l) = lod {
                expr(l, map);
            }
        }
        IrExpr::SampleCompare { index, uv, depth } => {
            expr(index, map);
            expr(uv, map);
            expr(depth, map);
        }
        IrExpr::FragLoad {
            src, offset, ld, ..
        } => {
            frag_src(src, map);
            expr(offset, map);
            expr(ld, map);
        }
        IrExpr::Binary { lhs, rhs, .. } => {
            expr(lhs, map);
            expr(rhs, map);
        }
    }
}

fn stmts(list: &mut [IrStmt], map: &HashMap<String, String>) {
    for s in list {
        match s {
            IrStmt::Let { name, init: value } | IrStmt::AssignLocal { name, value } => {
                rename(name, map);
                expr(value, map);
            }
            IrStmt::AssignBuffer {
                buf: name,
                index,
                value,
            }
            | IrStmt::AssignShared { name, index, value } => {
                rename(name, map);
                expr(index, map);
                expr(value, map);
            }
            IrStmt::Atomic {
                buf,
                index,
                value,
                compare,
                old,
                ..
            } => {
                rename(buf, map);
                expr(index, map);
                expr(value, map);
                if let Some(c) = compare {
                    expr(c, map);
                }
                if let Some(o) = old {
                    rename(o, map);
                }
            }
            IrStmt::ImageStore { image, args } => {
                rename(image, map);
                args.iter_mut().for_each(|a| expr(a, map));
            }
            IrStmt::If { cond, then, else_ } => {
                expr(cond, map);
                stmts(then, map);
                if let Some(e) = else_ {
                    stmts(e, map);
                }
            }
            IrStmt::For { var, lo, hi, body } => {
                rename(var, map);
                expr(lo, map);
                expr(hi, map);
                stmts(body, map);
            }
            IrStmt::FragStore {
                src,
                offset,
                ld,
                value,
            } => {
                frag_src(src, map);
                expr(offset, map);
                expr(ld, map);
                expr(value, map);
            }
            IrStmt::Eval(e) => expr(e, map),
            IrStmt::Barrier | IrStmt::SharedBarrier | IrStmt::Discard => {}
        }
    }
}

/// `module` with its taken names respelled, or itself when none is taken.
pub fn compute(module: &Module) -> Cow<'_, Module> {
    let e = &module.entry;
    let mut names: Vec<&str> = e.params.iter().map(|p| p.name.as_str()).collect();
    names.extend(e.shared.iter().map(|s| s.name.as_str()));
    names.extend(e.locals.iter().map(|(n, _)| n.as_str()));
    stmt_names(&e.body, &mut names);
    let map = spellings(names.into_iter());
    if map.is_empty() {
        return Cow::Borrowed(module);
    }
    let mut module = module.clone();
    let e = &mut module.entry;
    e.params.iter_mut().for_each(|p| rename(&mut p.name, &map));
    for s in &mut e.shared {
        rename(&mut s.name, &map);
        if let Some(base) = &mut s.view_of {
            rename(base, &map);
        }
    }
    e.locals.iter_mut().for_each(|(n, _)| rename(n, &map));
    stmts(&mut e.body, &map);
    Cow::Owned(module)
}

/// `module` with its taken names respelled, or itself when none is taken.
pub fn graphics(module: &GraphicsModule) -> Cow<'_, GraphicsModule> {
    let e = &module.entry;
    let mut names: Vec<&str> = e.inputs.iter().map(|i| i.name.as_str()).collect();
    names.extend(e.scalar_params.iter().map(|s| s.name.as_str()));
    names.extend(e.buf_params.iter().map(String::as_str));
    names.extend(e.textures.as_deref());
    names.extend(e.locals.iter().map(|(n, _)| n.as_str()));
    stmt_names(&e.body, &mut names);
    let map = spellings(names.into_iter());
    if map.is_empty() {
        return Cow::Borrowed(module);
    }
    let mut module = module.clone();
    let e = &mut module.entry;
    e.inputs.iter_mut().for_each(|i| rename(&mut i.name, &map));
    e.scalar_params
        .iter_mut()
        .for_each(|s| rename(&mut s.name, &map));
    e.buf_params.iter_mut().for_each(|b| rename(b, &map));
    if let Some(t) = &mut e.textures {
        rename(t, &map);
    }
    e.locals.iter_mut().for_each(|(n, _)| rename(n, &map));
    stmts(&mut e.body, &map);
    e.ret.iter_mut().for_each(|r| expr(r, &map));
    Cow::Owned(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_types_and_backend_names_are_taken() {
        for name in [
            "half",
            "float4",
            "half3x3",
            "packed_float3",
            "device",
            "pc",
            "floor",
        ] {
            assert!(taken(name), "{name}");
        }
        for name in ["zsl_x", "__y", "_Z"] {
            assert!(taken(name), "{name}");
        }
        for name in ["span", "halfway", "float5", "x", "uv", "colour", "p"] {
            assert!(!taken(name), "{name}");
        }
    }

    #[test]
    fn respellings_are_free_and_distinct() {
        let map = spellings(["half", "half_", "zsl_a", "x"].into_iter());
        assert_eq!(map.len(), 2);
        assert_eq!(map["half"], "half__", "half_ is a user name already");
        assert_eq!(map["zsl_a"], "uzsl_a");
    }
}
