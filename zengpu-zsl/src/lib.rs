//! ZSL proc-macro glue over `zengpu-zslc`. Use `zengpu_spirv!` or `zengpu`.
//!
//! Legacy forms (`push P { ... }`, `device mut buffer<T>`, `shared name: array<T, N>`, `barrier()`)
//! are deprecated.

extern crate proc_macro;

use proc_macro::TokenStream;

use zengpu_zslc::frontend::parser::Shader;
use zengpu_zslc::ir::Backend;
use zengpu_zslc::ir::arena::TypeArena;
use zengpu_zslc::ir::layout::LayoutKind;
use zengpu_zslc::ir::reflection::{ModuleReflection, reflect_compute, reflect_graphics};
use zengpu_zslc::ir::types::TypeId;
use zengpu_zslc::{backend, frontend};

/// Derives `to_scalars()` for a push struct: fields `u32`, `i32` or `f32` only, in declaration
/// order. Result goes to `Bindings::scalars`. Import via `use zengpu_spirv::ZslPushConst;`.
#[proc_macro_derive(ZslPushConst)]
pub fn derive_zsl_push_const(input: TokenStream) -> TokenStream {
    match push_const_impl(input) {
        Ok(ts) => ts,
        Err(msg) => compile_error_tokens(&msg),
    }
}

/// Hand-rolled derive, no `syn`/`quote`. `ZslMat4` expands to 16 column-major `Scalar::F32`.
#[allow(clippy::unwrap_used)]
fn push_const_impl(input: TokenStream) -> Result<TokenStream, String> {
    use proc_macro::{Delimiter, TokenTree};

    let mut it = input.into_iter();
    let mut name: Option<String> = None;
    let mut fields: Option<proc_macro::Group> = None;
    while let Some(tt) = it.next() {
        if let TokenTree::Ident(id) = &tt
            && id.to_string() == "struct"
        {
            match it.next() {
                Some(TokenTree::Ident(n)) => name = Some(n.to_string()),
                _ => return Err("ZslPushConst: expected a struct name".into()),
            }
            for tt2 in it.by_ref() {
                if let TokenTree::Group(g) = tt2 {
                    if g.delimiter() == Delimiter::Brace {
                        fields = Some(g);
                    }
                    break;
                }
            }
            break;
        }
    }
    let name = name.ok_or("ZslPushConst only supports structs")?;
    let fields = fields.ok_or("ZslPushConst requires a struct with named fields")?;

    let mut exprs: Vec<String> = Vec::new();
    let mut toks = fields.stream().into_iter().peekable();
    loop {
        let fname = match toks.next() {
            Some(TokenTree::Ident(id)) => id.to_string(),
            Some(_) => return Err("ZslPushConst: expected a field name".into()),
            None => break,
        };
        match toks.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => return Err("ZslPushConst: expected `:` after field name".into()),
        }
        let mut type_name = String::new();
        while let Some(tt) = toks.peek() {
            if matches!(tt, TokenTree::Punct(p) if p.as_char() == ',') {
                toks.next();
                break;
            }
            let tt = toks.next().unwrap();
            if type_name.is_empty()
                && let TokenTree::Ident(id) = &tt
            {
                type_name = id.to_string();
            }
        }
        match type_name.as_str() {
            "u32" => exprs.push(format!(
                "::zengpu_spirv::_zsl_priv::Scalar::U32(self.{fname})"
            )),
            "i32" => exprs.push(format!(
                "::zengpu_spirv::_zsl_priv::Scalar::I32(self.{fname})"
            )),
            "f32" => exprs.push(format!(
                "::zengpu_spirv::_zsl_priv::Scalar::F32(self.{fname})"
            )),
            "ZslMat4" => {
                for i in 0..16 {
                    exprs.push(format!(
                        "::zengpu_spirv::_zsl_priv::Scalar::F32(self.{fname}.0[{i}])"
                    ));
                }
            }
            other => {
                return Err(format!(
                    "ZslPushConst fields must be u32, i32, f32, or ZslMat4; got `{other}`"
                ));
            }
        }
    }

    let n = exprs.len();
    let body = exprs.join(", ");
    let code = format!(
        "impl {name} {{ \
            pub fn to_scalars(&self) -> [::zengpu_spirv::_zsl_priv::Scalar; {n}] {{ [{body}] }} \
        }}"
    );
    code.parse()
        .map_err(|_| "ZslPushConst: generated code failed to parse".to_string())
}

/// Compiles native ZSL source to a `ZslShader` that holds every backend form, built at
/// compile time.
///
/// The returned `zengpu_spirv::ZslShader` carries SPIR-V (Vulkan), HIP C++ (ROCm), MSL
/// (Metal) and CUDA C++ (NVIDIA), all compiled from the same ZSL source. Pick the form at
/// runtime with `ZslShader::for_backend`.
///
/// ```ignore
/// const SHADER: ZslShader = zengpu_spirv::zsl!(
///     push Push { n: u32, scale: f32 }
///     @workgroup_size(64)
///     kernel scale(p: Push, src: device buffer<f32>, dst: device mut buffer<f32>, id: global_id) {
///         let i = id.x
///         if i < p.n { dst[i] = src[i] * p.scale }
///     }
/// );
/// ```
#[proc_macro]
pub fn zsl(input: TokenStream) -> TokenStream {
    let src = input.to_string();
    match compile_zsl_all(&src) {
        Ok(ts) => ts,
        Err(msg) => compile_error_tokens(&msg),
    }
}

type Outcome<T> = Result<T, String>;

struct Lowered {
    spv: Outcome<Vec<u32>>,
    hip: Outcome<String>,
    msl: Outcome<String>,
    cuda: Outcome<String>,
}

impl Lowered {
    fn outcomes(&self) -> [(Backend, Result<(), String>); 4] {
        [
            (
                Backend::Spirv,
                self.spv.as_ref().map(|_| ()).map_err(Clone::clone),
            ),
            (
                Backend::Hip,
                self.hip.as_ref().map(|_| ()).map_err(Clone::clone),
            ),
            (
                Backend::Metal,
                self.msl.as_ref().map(|_| ()).map_err(Clone::clone),
            ),
            (
                Backend::Cuda,
                self.cuda.as_ref().map(|_| ()).map_err(Clone::clone),
            ),
        ]
    }
}

/// Returns an error if a backend named in `@backends` fails to lower, or if all of them do.
fn gate_outcomes(
    required: &[Backend],
    outcomes: &[(Backend, Result<(), String>)],
) -> Result<(), String> {
    for (backend, outcome) in outcomes {
        if let Err(msg) = outcome
            && required.contains(backend)
        {
            return Err(format!(
                "ZSL: @backends requires `{}`, which cannot lower this shader: {msg}",
                backend.name()
            ));
        }
    }
    if outcomes.iter().all(|(_, o)| o.is_err()) {
        let reasons: Vec<String> = outcomes
            .iter()
            .filter_map(|(b, o)| o.as_ref().err().map(|m| format!("{}: {m}", b.name())))
            .collect();
        return Err(format!(
            "ZSL: no enabled backend can lower this shader ({})",
            reasons.join("; ")
        ));
    }
    Ok(())
}

fn not_compiled_in(backend: Backend) -> String {
    format!(
        "the `{0}` backend is not compiled in; enable zengpu-spirv's `{0}` feature",
        backend.name()
    )
}

fn lower_all(shader: &Shader) -> Lowered {
    let spv = if cfg!(feature = "spirv") {
        match shader {
            Shader::Compute(m) => backend::spirv::lower_compute(m),
            Shader::Graphics(m) => backend::spirv::lower_graphics(m),
        }
        .map_err(|d| d.message)
    } else {
        Err(not_compiled_in(Backend::Spirv))
    };

    let hip = if cfg!(feature = "hip") {
        match shader {
            Shader::Compute(m) => backend::hip::lower_compute(m)
                .map(|s| s.source)
                .map_err(|d| d.message),
            Shader::Graphics(_) => Err("the hip backend has no graphics lowering".into()),
        }
    } else {
        Err(not_compiled_in(Backend::Hip))
    };

    let msl = if cfg!(feature = "metal") {
        match shader {
            Shader::Compute(m) => backend::msl::lower_compute(m)
                .map(|s| s.source)
                .map_err(|d| d.message),
            Shader::Graphics(m) => backend::msl::lower_graphics(m)
                .map(|s| s.source)
                .map_err(|d| d.message),
        }
    } else {
        Err(not_compiled_in(Backend::Metal))
    };

    let cuda = if cfg!(feature = "cuda") {
        match shader {
            Shader::Compute(m) => backend::cuda::lower_compute(m)
                .map(|s| s.source)
                .map_err(|d| d.message),
            Shader::Graphics(_) => Err("the cuda backend has no graphics lowering".into()),
        }
    } else {
        Err(not_compiled_in(Backend::Cuda))
    };

    Lowered {
        spv,
        hip,
        msl,
        cuda,
    }
}

fn compile_zsl_all(src: &str) -> Result<TokenStream, String> {
    use frontend::parser::parse_zsl;
    let shader = parse_zsl(src).map_err(|e| format!("ZSL parse error: {}", e.msg))?;

    let reflection: ModuleReflection = match &shader {
        Shader::Compute(m) => reflect_compute(m),
        Shader::Graphics(m) => reflect_graphics(m),
    };
    let types: &TypeArena = match &shader {
        Shader::Compute(m) => &m.types,
        Shader::Graphics(m) => &m.types,
    };
    let required: &[Backend] = match &shader {
        Shader::Compute(m) => &m.backends,
        Shader::Graphics(m) => &m.backends,
    };

    let lowered = lower_all(&shader);
    gate_outcomes(required, &lowered.outcomes())?;

    let mut s = String::new();
    s.push_str("::zengpu_spirv::ZslShader { spv: ");
    match &lowered.spv {
        Ok(words) => {
            s.push_str("Ok(");
            s.push_str(&words_to_slice_str(words));
            s.push(')');
        }
        Err(msg) => push_err(&mut s, msg),
    }
    for (field, text) in [
        ("hip", &lowered.hip),
        ("msl", &lowered.msl),
        ("cuda", &lowered.cuda),
    ] {
        s.push_str(", ");
        s.push_str(field);
        s.push_str(": ");
        match text {
            Ok(src) => {
                s.push_str("Ok(");
                push_str_lit(&mut s, src);
                s.push(')');
            }
            Err(msg) => push_err(&mut s, msg),
        }
    }
    s.push_str(", reflection: ");
    s.push_str(&reflection_to_slice_str(&reflection, types));
    s.push_str(" }");

    s.parse()
        .map_err(|_| "ZSL: generated ZslShader literal failed to parse".to_string())
}

fn push_err(s: &mut String, msg: &str) {
    s.push_str("Err(");
    push_str_lit(s, msg);
    s.push(')');
}

fn words_to_slice_str(words: &[u32]) -> String {
    let mut s = String::with_capacity(words.len() * 12 + 4);
    s.push_str("&[");
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&w.to_string());
        s.push_str("u32");
    }
    s.push(']');
    s
}

/// Variant name for `ty`, printed into generated source. See `reflection_to_slice_str`.
fn scalar_kind_name(ty: TypeId, types: &TypeArena) -> &'static str {
    use zengpu_zslc::ir::types::{FloatFormat, IntWidth, Signedness, Type};
    match types.get(ty) {
        Type::Float {
            format: FloatFormat::F32,
        } => "F32",
        Type::Float {
            format: FloatFormat::F16,
        } => "F16",
        Type::Float {
            format: FloatFormat::Bf16,
        } => "Bf16",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Unsigned,
        } => "U32",
        Type::Int {
            width: IntWidth::W32,
            signedness: Signedness::Signed,
        } => "I32",
        Type::Matrix {
            rows: 4,
            columns: 4,
            ..
        } => "Mat4",
        Type::Vector { lanes: 2, .. } => "Vec2",
        Type::Vector { lanes: 4, .. } => "Vec4",
        other => panic!("ZSL: reflection encountered an unsupported push-constant type {other:?}"),
    }
}

fn reflection_to_slice_str(reflection: &ModuleReflection, types: &TypeArena) -> String {
    let mut s = String::from("&[");
    for (i, p) in reflection.params.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("::zengpu_spirv::ParamLayout { name: ");
        push_str_lit(&mut s, &p.name);
        s.push_str(", offset: ");
        s.push_str(&p.layout.offset.to_string());
        s.push_str(", size: ");
        s.push_str(&p.layout.size.to_string());
        s.push_str(", kind: ");
        match (&p.layout.kind, p.binding) {
            (LayoutKind::BufferIndex { elem }, Some(binding)) => {
                s.push_str(&format!(
                    "::zengpu_spirv::ParamKind::BufferIndex {{ elem: ::zengpu_spirv::ScalarKind::{}, binding: {} }}",
                    scalar_kind_name(*elem, types),
                    binding.binding,
                ));
            }
            (LayoutKind::TextureIndex, Some(binding)) => {
                s.push_str(&format!(
                    "::zengpu_spirv::ParamKind::TextureIndex {{ binding: {} }}",
                    binding.binding,
                ));
            }
            (LayoutKind::ImageIndex, Some(binding)) => {
                s.push_str(&format!(
                    "::zengpu_spirv::ParamKind::ImageIndex {{ binding: {} }}",
                    binding.binding,
                ));
            }
            (LayoutKind::Scalar { ty }, None) => {
                s.push_str(&format!(
                    "::zengpu_spirv::ParamKind::Scalar {{ kind: ::zengpu_spirv::ScalarKind::{} }}",
                    scalar_kind_name(*ty, types),
                ));
            }
            _ => unreachable!(
                "ZSL: reflection invariant broken: a BufferIndex layout must have a binding, a Scalar layout must not"
            ),
        }
        s.push_str(" }");
    }
    s.push(']');
    s
}

fn push_str_lit(s: &mut String, v: &str) {
    s.push('"');
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c => s.push(c),
        }
    }
    s.push('"');
}

#[allow(clippy::expect_used)]
fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    fn ok() -> Result<(), String> {
        Ok(())
    }

    fn err(msg: &str) -> Result<(), String> {
        Err(msg.to_string())
    }

    #[test]
    fn one_lowering_backend_is_enough() {
        let outcomes = [
            (Backend::Spirv, err("frag_get")),
            (Backend::Hip, ok()),
            (Backend::Metal, err("frag_get")),
            (Backend::Cuda, err("frag_get")),
        ];
        assert!(gate_outcomes(&[], &outcomes).is_ok());
    }

    #[test]
    fn required_backend_failure_is_an_error() {
        let outcomes = [
            (Backend::Spirv, err("frag_get")),
            (Backend::Hip, ok()),
            (Backend::Metal, ok()),
            (Backend::Cuda, ok()),
        ];
        let msg = gate_outcomes(&[Backend::Spirv], &outcomes).expect_err("required spirv");
        assert!(msg.contains("`spirv`") && msg.contains("frag_get"), "{msg}");
    }

    #[test]
    fn no_lowering_backend_is_an_error() {
        let outcomes = [
            (Backend::Spirv, err("a")),
            (Backend::Hip, err("b")),
            (Backend::Metal, err("c")),
            (Backend::Cuda, err("d")),
        ];
        let msg = gate_outcomes(&[], &outcomes).expect_err("nothing lowers");
        assert!(msg.contains("hip: b") && msg.contains("cuda: d"), "{msg}");
    }
}
