//! Proc-macro internals. Use `zengpu_spirv` or `zengpu`, not this crate directly.

extern crate proc_macro;

mod backend;
mod frontend;
mod ir;

use proc_macro::TokenStream;

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
fn push_const_impl(input: TokenStream) -> Result<TokenStream, String> {
    use proc_macro::{Delimiter, TokenTree};

    let mut it = input.into_iter();
    let mut name: Option<String> = None;
    let mut fields: Option<proc_macro::Group> = None;
    while let Some(tt) = it.next() {
        if let TokenTree::Ident(id) = &tt {
            if id.to_string() == "struct" {
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
            if type_name.is_empty() {
                if let TokenTree::Ident(id) = &tt {
                    type_name = id.to_string();
                }
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

/// Compiles native ZSL source to a [`ZslShader`] holding every backend's compiled form.
///
/// Carries SPIR-V (Vulkan), HIP C++ (ROCm), MSL (Metal) and CUDA C++ (NVIDIA), all from the same
/// ZSL source. At runtime pick one with
/// [`ZslShader::for_backend`](::zengpu_spirv::ZslShader::for_backend).
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

fn compile_zsl_all(src: &str) -> Result<TokenStream, String> {
    use frontend::parser::{Shader, parse_zsl};
    let shader = parse_zsl(src).map_err(|e| format!("ZSL parse error: {}", e.msg))?;

    let spv_words = match &shader {
        Shader::Compute(m) => backend::spirv::lower_compute(m)?,
        Shader::Graphics(m) => backend::spirv::lower_graphics(m)?,
    };
    let hip_src = match &shader {
        Shader::Compute(m) => backend::hip::lower_compute(m).source,
        Shader::Graphics(_) => String::new(),
    };
    let msl_src = match &shader {
        Shader::Compute(m) => backend::msl::lower_compute(m).source,
        Shader::Graphics(m) => backend::msl::lower_graphics(m).source,
    };
    let cuda_src = match &shader {
        Shader::Compute(m) => backend::cuda::lower_compute(m).source,
        Shader::Graphics(_) => String::new(),
    };

    let mut s = String::new();
    s.push_str("::zengpu_spirv::ZslShader { spv: ");
    s.push_str(&words_to_slice_str(&spv_words));
    s.push_str(", hip: ");
    push_str_lit(&mut s, &hip_src);
    s.push_str(", msl: ");
    push_str_lit(&mut s, &msl_src);
    s.push_str(", cuda: ");
    push_str_lit(&mut s, &cuda_src);
    s.push_str(" }");

    s.parse()
        .map_err(|_| "ZSL: generated ZslShader literal failed to parse".to_string())
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

fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}
