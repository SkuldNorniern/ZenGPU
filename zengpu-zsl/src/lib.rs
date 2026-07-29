//! ZenGPU shader proc-macro internals. Import `zengpu_spirv` or `zengpu` and use the
//! `zengpu_spirv!` macro; this crate is not for direct use.
//!
//! Legacy syntax (`push P { ... }`, `device mut buffer<T>`, `shared name: array<T, N>`,
//! `barrier()`) is accepted. A later release warns on it, then removes it.

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

    let reflection: crate::ir::reflection::ModuleReflection = match &shader {
        Shader::Compute(m) => crate::ir::reflection::reflect_compute(m),
        Shader::Graphics(m) => crate::ir::reflection::reflect_graphics(m),
    };
    let types: &crate::ir::arena::TypeArena = match &shader {
        Shader::Compute(m) => &m.types,
        Shader::Graphics(m) => &m.types,
    };

    let spv_words = match &shader {
        Shader::Compute(m) => backend::spirv::lower_compute(m)?,
        Shader::Graphics(m) => backend::spirv::lower_graphics(m)?,
    };

    #[cfg(feature = "hip")]
    let hip_src = match &shader {
        Shader::Compute(m) => backend::hip::lower_compute(m)
            .map(|shader| shader.source)
            .unwrap_or_default(),
        Shader::Graphics(_) => String::new(),
    };
    #[cfg(not(feature = "hip"))]
    let hip_src = String::new();

    #[cfg(feature = "metal")]
    let msl_src = match &shader {
        Shader::Compute(m) => backend::msl::lower_compute(m)
            .map(|shader| shader.source)
            .unwrap_or_default(),
        Shader::Graphics(m) => backend::msl::lower_graphics(m)
            .map(|shader| shader.source)
            .unwrap_or_default(),
    };
    #[cfg(not(feature = "metal"))]
    let msl_src = String::new();

    #[cfg(feature = "cuda")]
    let cuda_src = match &shader {
        Shader::Compute(m) => backend::cuda::lower_compute(m)
            .map(|shader| shader.source)
            .unwrap_or_default(),
        Shader::Graphics(_) => String::new(),
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_src = String::new();

    let mut s = String::new();
    s.push_str("::zengpu_spirv::ZslShader { spv: ");
    s.push_str(&words_to_slice_str(&spv_words));
    s.push_str(", hip: ");
    push_str_lit(&mut s, &hip_src);
    s.push_str(", msl: ");
    push_str_lit(&mut s, &msl_src);
    s.push_str(", cuda: ");
    push_str_lit(&mut s, &cuda_src);
    s.push_str(", reflection: ");
    s.push_str(&reflection_to_slice_str(&reflection, types));
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

/// Variant name for `ty`, printed into generated source. See `reflection_to_slice_str`.
fn scalar_kind_name(
    ty: crate::ir::types::TypeId,
    types: &crate::ir::arena::TypeArena,
) -> &'static str {
    use crate::ir::types::{FloatFormat, IntWidth, Signedness, Type};
    match types.get(ty) {
        Type::Float {
            format: FloatFormat::F32,
        } => "F32",
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
        other => panic!("ZSL: reflection encountered an unsupported push-constant type {other:?}"),
    }
}

fn reflection_to_slice_str(
    reflection: &crate::ir::reflection::ModuleReflection,
    types: &crate::ir::arena::TypeArena,
) -> String {
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
            (crate::ir::layout::LayoutKind::BufferIndex { elem }, Some(binding)) => {
                s.push_str(&format!(
                    "::zengpu_spirv::ParamKind::BufferIndex {{ elem: ::zengpu_spirv::ScalarKind::{}, binding: {} }}",
                    scalar_kind_name(*elem, types),
                    binding.binding,
                ));
            }
            (crate::ir::layout::LayoutKind::Scalar { ty }, None) => {
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

fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}

#[cfg(test)]
mod phase_z_tests {
    #![cfg(feature = "test-all-backends")]
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::backend::{cuda, hip, msl, spirv};
    use crate::frontend::parser::parse_compute;

    const TILED_GEMM: &str = r#"
        push Shape { m: u32, n: u32, k: u32 }
        @workgroup_size(16, 16)
        kernel tiled_gemm(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: Shape) {
            shared as_tile: array<f32, 256>
            shared bs_tile: array<f32, 256>
            let lx = local_id().x
            let ly = local_id().y
            let row = group_id().y * 16 + ly
            let col = group_id().x * 16 + lx
            let sum: f32 = 0.0
            let tile_count = (p.k + 15) / 16
            for tile in 0..tile_count {
                let ak = tile * 16 + lx
                let bk = tile * 16 + ly
                if row < p.m && ak < p.k {
                    as_tile[ly * 16 + lx] = a[row * p.k + ak]
                } else {
                    as_tile[ly * 16 + lx] = 0.0
                }
                if bk < p.k && col < p.n {
                    bs_tile[ly * 16 + lx] = b[bk * p.n + col]
                } else {
                    bs_tile[ly * 16 + lx] = 0.0
                }
                barrier()
                for q in 0..16 {
                    sum = sum + as_tile[ly * 16 + q] * bs_tile[q * 16 + lx]
                }
                barrier()
            }
            if row < p.m && col < p.n {
                c[row * p.n + col] = sum
            }
        }
    "#;

    const SCATTER_ADD: &str = r#"
        push Scatter { n: u32 }
        @workgroup_size(256)
        kernel scatter_add(idx: device buffer<f32>, val: device buffer<f32>, out: device mut buffer<f32>, p: Scatter, id: global_id) {
            let i = id.x
            if i < p.n {
                atomic_add(out, idx[i], val[i])
            }
        }
    "#;

    const TRIG: &str = r#"
        @workgroup_size(64)
        kernel trig(src: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
            let x = src[id.x]
            out[id.x] = sin(x) + cos(x) + tan(x)
        }
    "#;

    const TYPED_BUFFERS: &str = r#"
        @workgroup_size(64)
        kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, id: global_id) {
            out_u[id.x] = src_u[id.x]
            out_i[id.x] = src_i[id.x]
        }
    "#;

    const COPY_BF16: &str = r#"
        @workgroup_size(64)
        kernel copy_bf16(src: device buffer<bf16>, out: device mut buffer<bf16>, id: global_id) {
            out[id.x] = src[id.x]
        }
    "#;

    const COPY_F16: &str = r#"
        @workgroup_size(64)
        kernel copy_f16(src: device buffer<f16>, out: device mut buffer<f16>, id: global_id) {
            out[id.x] = src[id.x]
        }
    "#;

    const FINITE_CLASSIFY: &str = r#"
        @workgroup_size(64)
        kernel classify(src: device buffer<f32>, out: device mut buffer<u32>, id: global_id) {
            let x = src[id.x]
            if isfinite(x) {
                out[id.x] = 1
            } else {
                if isnan(x) {
                    out[id.x] = 2
                } else {
                    if isinf(x) { out[id.x] = 3 } else { out[id.x] = 4 }
                }
            }
        }
    "#;

    #[test]
    fn finite_number_builtins_lower_on_all_compute_backends() {
        let module = parse_compute(FINITE_CLASSIFY).expect("parse finite classification");
        let hip_source = hip::lower_compute(&module).expect("lower finite classification to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower finite classification to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower finite classification to MSL").source;
        for source in [&hip_source, &cuda_source, &msl_source] {
            assert!(source.contains("isfinite(x)"));
            assert!(source.contains("isnan(x)"));
            assert!(source.contains("isinf(x)"));
        }
        spirv::lower_compute(&module).expect("lower finite classification to SPIR-V");
    }

    #[test]
    fn integer_storage_buffers_lower_on_all_compute_backends() {
        let module = parse_compute(TYPED_BUFFERS).expect("parse typed buffers");
        let hip_source = hip::lower_compute(&module).expect("lower typed buffers to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower typed buffers to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower typed buffers to MSL").source;
        assert!(hip_source.contains("const unsigned int* __restrict__ src_u"));
        assert!(hip_source.contains("const int* __restrict__ src_i"));
        assert!(hip_source.contains("unsigned int* __restrict__ out_u"));
        assert!(hip_source.contains("int* __restrict__ out_i"));
        assert!(cuda_source.contains("const unsigned int* __restrict__ src_u"));
        assert!(cuda_source.contains("const int* __restrict__ src_i"));
        assert!(cuda_source.contains("unsigned int* __restrict__ out_u"));
        assert!(cuda_source.contains("int* __restrict__ out_i"));
        assert!(msl_source.contains("device uint* src_u"));
        assert!(msl_source.contains("device int* src_i"));
        assert!(msl_source.contains("device uint* out_u"));
        assert!(msl_source.contains("device int* out_i"));
        spirv::lower_compute(&module).expect("lower typed buffers to SPIR-V");
    }

    #[test]
    fn bf16_buffer_lowers_to_hip_with_conversion_helpers() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let hip_source = hip::lower_compute(&module).expect("lower copy_bf16 to HIP").source;
        assert!(hip_source.contains("const unsigned short* __restrict__ src"));
        assert!(hip_source.contains("unsigned short* __restrict__ out"));
        assert!(hip_source.contains("f32_to_bf16(bf16_to_f32(src[gx]))"));
        assert!(hip_source.contains("__device__ __forceinline__ unsigned short f32_to_bf16"));
        assert!(hip_source.contains("__device__ __forceinline__ float bf16_to_f32"));
    }

    #[test]
    fn bf16_buffer_lowers_to_cuda_with_conversion_helpers() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let cuda_source = cuda::lower_compute(&module).expect("lower copy_bf16 to CUDA").source;
        assert!(cuda_source.contains("const unsigned short* __restrict__ src"));
        assert!(cuda_source.contains("unsigned short* __restrict__ out"));
        assert!(cuda_source.contains("f32_to_bf16(bf16_to_f32(src[gx]))"));
        assert!(cuda_source.contains("__device__ __forceinline__ unsigned short f32_to_bf16"));
        assert!(cuda_source.contains("__device__ __forceinline__ float bf16_to_f32"));
    }

    #[test]
    fn bf16_buffer_lowers_to_msl_with_conversion_helpers() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let msl_source = msl::lower_compute(&module).expect("lower copy_bf16 to MSL").source;
        assert!(msl_source.contains("device ushort* src"));
        assert!(msl_source.contains("device ushort* out"));
        assert!(msl_source.contains("f32_to_bf16(bf16_to_f32(src[gid.x]))"));
        assert!(msl_source.contains("inline ushort f32_to_bf16"));
        assert!(msl_source.contains("inline float bf16_to_f32"));
    }

    #[test]
    fn bf16_buffer_rejects_cleanly_on_spirv_backend() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let err = spirv::lower_compute(&module).expect_err("bf16 must be rejected on SPIR-V");
        assert!(err.message.contains("does not yet support"), "got: {}", err.message);
    }

    #[test]
    fn bf16_hip_round_trip_exact_match_when_gpu_present() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let shader = hip::lower_compute(&module).expect("lower copy_bf16 to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP bf16 proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_bf16_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_bf16_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
{}
static unsigned short f32_to_bf16_host(float x) {{
    unsigned int bits;
    std::memcpy(&bits, &x, sizeof(bits));
    unsigned int bias = 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)((bits + bias) >> 16);
}}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned N = 65536;
    std::vector<unsigned short> src(N), out(N, 0);
    for (unsigned i = 0; i < N; ++i) {{
        float x = (float)i * 0.01f - 300.0f;
        src[i] = f32_to_bf16_host(x);
    }}
    unsigned short *dsrc = nullptr, *dout = nullptr;
    if (hipMalloc(&dsrc, src.size()*sizeof(unsigned short)) != hipSuccess ||
        hipMalloc(&dout, out.size()*sizeof(unsigned short)) != hipSuccess) return 2;
    hipMemcpy(dsrc, src.data(), src.size()*sizeof(unsigned short), hipMemcpyHostToDevice);
    zsl_kernel<<<dim3((N+63)/64), dim3(64)>>>(dsrc, dout);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(out.data(), dout, out.size()*sizeof(unsigned short), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) mismatches += out[i] != src[i];
    std::printf("BF16_EXACT mismatches=%u\n", mismatches);
    hipFree(dsrc); hipFree(dout);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP bf16 proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP bf16 compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path).output().expect("run HIP copy_bf16");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP copy_bf16 failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP copy_bf16: {}", stdout.trim());
    }

    #[test]
    fn f16_buffer_lowers_on_all_compute_backends() {
        let module = parse_compute(COPY_F16).expect("parse copy_f16");
        let hip_source = hip::lower_compute(&module).expect("lower copy_f16 to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower copy_f16 to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower copy_f16 to MSL").source;
        assert!(hip_source.contains("#include <hip/hip_fp16.h>"));
        assert!(hip_source.contains("const __half* __restrict__ src"));
        assert!(hip_source.contains("__half* __restrict__ out"));
        assert!(hip_source.contains("__float2half(__half2float(src[gx]))"));
        assert!(cuda_source.contains("#include <cuda_fp16.h>"));
        assert!(cuda_source.contains("const __half* __restrict__ src"));
        assert!(cuda_source.contains("__float2half(__half2float(src[gx]))"));
        assert!(msl_source.contains("device half* src"));
        assert!(msl_source.contains("half(float(src[gid.x]))"));
    }

    #[test]
    fn f16_buffer_rejects_cleanly_on_spirv_backend() {
        let module = parse_compute(COPY_F16).expect("parse copy_f16");
        let err = spirv::lower_compute(&module).expect_err("f16 must be rejected on SPIR-V");
        assert!(err.message.contains("does not yet support"), "got: {}", err.message);
    }

    #[test]
    fn f16_hip_round_trip_exact_match_when_gpu_present() {
        let module = parse_compute(COPY_F16).expect("parse copy_f16");
        let shader = hip::lower_compute(&module).expect("lower copy_f16 to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP f16 proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_f16_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_f16_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned N = 65536;
    std::vector<unsigned short> src(N), out(N, 0);
    for (unsigned i = 0; i < N; ++i) {{
        float x = (float)i * 0.01f - 300.0f;
        __half h = __float2half(x);
        std::memcpy(&src[i], &h, sizeof(unsigned short));
    }}
    unsigned short *dsrc = nullptr, *dout = nullptr;
    if (hipMalloc(&dsrc, src.size()*sizeof(unsigned short)) != hipSuccess ||
        hipMalloc(&dout, out.size()*sizeof(unsigned short)) != hipSuccess) return 2;
    hipMemcpy(dsrc, src.data(), src.size()*sizeof(unsigned short), hipMemcpyHostToDevice);
    zsl_kernel<<<dim3((N+63)/64), dim3(64)>>>((const __half*)dsrc, (__half*)dout);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(out.data(), dout, out.size()*sizeof(unsigned short), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) mismatches += out[i] != src[i];
    std::printf("F16_EXACT mismatches=%u\n", mismatches);
    hipFree(dsrc); hipFree(dout);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP f16 proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP f16 compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path).output().expect("run HIP copy_f16");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP copy_f16 failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP copy_f16: {}", stdout.trim());
    }

    const MUL_ADD_F32: &str = r#"
        @workgroup_size(64)
        kernel fma3(a: device buffer<f32>, b: device buffer<f32>, c: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
            out[id.x] = mul_add_f32(a[id.x], b[id.x], c[id.x])
        }
    "#;

    const SUBGROUP_QUERY: &str = r#"
        @workgroup_size(256)
        kernel query(
            out_size: device mut buffer<u32>,
            out_id: device mut buffer<u32>,
            out_lane: device mut buffer<u32>,
            id: global_id,
        ) {
            out_size[id.x] = subgroup_size()
            out_id[id.x] = subgroup_id()
            out_lane[id.x] = subgroup_local_id()
        }
    "#;

    const SUBGROUP_VOTE: &str = r#"
        push P { n: u32 }
        @workgroup_size(256)
        kernel vote(
            out_any: device mut buffer<u32>,
            out_all: device mut buffer<u32>,
            p: P,
            id: global_id,
        ) {
            if group_id().x < p.n {
                out_any[id.x] = u32(subgroup_any(subgroup_local_id() < 1u32))
                out_all[id.x] = u32(subgroup_all(subgroup_local_id() < 1u32))
            }
        }
    "#;

    const SUBGROUP_BALLOT: &str = r#"
        @workgroup_size(256)
        kernel ballot(out: device mut buffer<u32>, id: global_id) {
            subgroup_ballot(id.x < 10u32)
        }
    "#;

    const SUBGROUP_EXCHANGE: &str = r#"
        push P { n: u32 }
        @workgroup_size(256)
        kernel exchange(
            out_broadcast: device mut buffer<f32>,
            out_shuffle: device mut buffer<f32>,
            out_xor: device mut buffer<f32>,
            out_up: device mut buffer<f32>,
            out_down: device mut buffer<f32>,
            p: P,
            id: global_id,
        ) {
            if group_id().x < p.n {
                let value = id.x * 1.0
                out_broadcast[id.x] = subgroup_broadcast(value, 0u32)
                out_shuffle[id.x] = subgroup_shuffle(value, 1u32)
                out_xor[id.x] = subgroup_shuffle_xor(value, 1u32)
                out_up[id.x] = subgroup_shuffle_up(value, 1u32)
                out_down[id.x] = subgroup_shuffle_down(value, 1u32)
            }
        }
    "#;

    const SUBGROUP_SHUFFLE_XOR: &str = r#"
        @workgroup_size(256)
        kernel shuffle_xor(out: device mut buffer<f32>, id: global_id) {
            out[id.x] = subgroup_shuffle_xor(id.x * 1.0, 1u32)
        }
    "#;

    const SUBGROUP_ARITHMETIC: &str = r#"
        push P { n: u32 }
        @workgroup_size(256)
        kernel subgroup_arithmetic(
            src_f: device buffer<f32>,
            src_u: device buffer<u32>,
            out_f: device mut buffer<f32>,
            out_u: device mut buffer<u32>,
            p: P,
            id: global_id,
        ) {
            if group_id().x < p.n {
                let f = src_f[id.x]
                let u = src_u[id.x]
                out_f[id.x] = subgroup_reduce_add(f)
                out_f[id.x] = subgroup_reduce_min(f)
                out_f[id.x] = subgroup_reduce_max(f)
                out_f[id.x] = subgroup_scan_inclusive_add(f)
                out_f[id.x] = subgroup_scan_exclusive_add(f)
                out_u[id.x] = subgroup_reduce_add(u)
                out_u[id.x] = subgroup_reduce_min(u)
                out_u[id.x] = subgroup_reduce_max(u)
                out_u[id.x] = subgroup_reduce_and(u)
                out_u[id.x] = subgroup_reduce_or(u)
                out_u[id.x] = subgroup_scan_inclusive_add(u)
                out_u[id.x] = subgroup_scan_exclusive_add(u)
            }
        }
    "#;

    const SUBGROUP_ARITHMETIC_PROOF: &str = r#"
        @workgroup_size(256)
        kernel subgroup_arithmetic_proof(
            out_reduce_add: device mut buffer<u32>,
            out_scan_inclusive: device mut buffer<u32>,
            out_scan_exclusive: device mut buffer<u32>,
            out_reduce_min: device mut buffer<u32>,
            out_reduce_max: device mut buffer<u32>,
            id: global_id,
        ) {
            out_reduce_add[id.x] = subgroup_reduce_add(1u32)
            out_scan_inclusive[id.x] = subgroup_scan_inclusive_add(1u32)
            out_scan_exclusive[id.x] = subgroup_scan_exclusive_add(1u32)
            out_reduce_min[id.x] = subgroup_reduce_min(subgroup_local_id())
            out_reduce_max[id.x] = subgroup_reduce_max(subgroup_local_id())
        }
    "#;

    const WORKGROUP_REDUCE: &str = r#"
        push P { n: u32 }
        @workgroup_size(256)
        kernel workgroup_reduce(
            src: device buffer<f32>,
            out_add: device mut buffer<f32>,
            out_min: device mut buffer<f32>,
            out_max: device mut buffer<f32>,
            p: P,
            id: global_id,
        ) {
            if group_id().x < p.n {
                let value = src[id.x]
                out_add[id.x] = workgroup_reduce_add(value)
                out_min[id.x] = workgroup_reduce_min(value)
                out_max[id.x] = workgroup_reduce_max(value)
            }
        }
    "#;

    const WORKGROUP_REDUCE_PROOF: &str = r#"
        @workgroup_size(256)
        kernel workgroup_reduce_proof(
            out_add: device mut buffer<f32>,
            out_min: device mut buffer<f32>,
            out_max: device mut buffer<f32>,
            id: global_id,
        ) {
            out_add[id.x] = workgroup_reduce_add(1.0)
            let lane = local_id().x * 1.0
            out_min[id.x] = workgroup_reduce_min(lane)
            out_max[id.x] = workgroup_reduce_max(lane)
        }
    "#;

    #[test]
    fn subgroup_queries_lower_on_all_compute_backends() {
        let module = parse_compute(SUBGROUP_QUERY).expect("parse query");
        let hip_source = hip::lower_compute(&module).expect("lower query to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower query to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower query to MSL").source;
        assert!(hip_source.contains("((unsigned int)warpSize)"));
        assert!(hip_source.contains("(threadIdx.x / warpSize)"));
        assert!(hip_source.contains("__lane_id()"));
        assert!(cuda_source.contains("((unsigned int)warpSize)"));
        assert!(cuda_source.contains("(threadIdx.x / warpSize)"));
        assert!(cuda_source.contains("(threadIdx.x % warpSize)"));
        assert!(msl_source.contains("uint simd_size [[threads_per_simdgroup]]"));
        assert!(msl_source.contains("uint simd_group [[simdgroup_index_in_threadgroup]]"));
        assert!(msl_source.contains("uint simd_lane [[thread_index_in_simdgroup]]"));
        assert!(msl_source.contains("out_size[gid.x] = simd_size"));
        assert!(msl_source.contains("out_id[gid.x] = simd_group"));
        assert!(msl_source.contains("out_lane[gid.x] = simd_lane"));
        spirv::lower_compute(&module).expect("lower query to SPIR-V");
    }

    #[test]
    fn subgroup_any_and_all_lower_on_all_compute_backends() {
        let module = parse_compute(SUBGROUP_VOTE).expect("parse vote");
        let hip_source = hip::lower_compute(&module).expect("lower vote to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower vote to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower vote to MSL").source;
        assert!(hip_source.contains("__any("));
        assert!(hip_source.contains("__all("));
        assert!(cuda_source.contains("__any_sync("));
        assert!(cuda_source.contains("__all_sync("));
        assert!(msl_source.contains("simd_any("));
        assert!(msl_source.contains("simd_all("));
        spirv::lower_compute(&module).expect("lower vote to SPIR-V");
    }

    #[test]
    fn subgroup_ballot_is_spirv_only_for_now() {
        let module = parse_compute(SUBGROUP_BALLOT).expect("parse ballot");
        assert!(hip::lower_compute(&module).is_err(), "HIP ballot must reject");
        assert!(
            cuda::lower_compute(&module).is_err(),
            "CUDA ballot must reject"
        );
        assert!(msl::lower_compute(&module).is_err(), "MSL ballot must reject");
        spirv::lower_compute(&module).expect("lower ballot to SPIR-V");
    }

    #[test]
    fn subgroup_data_exchange_lowers_on_all_compute_backends() {
        let module = parse_compute(SUBGROUP_EXCHANGE).expect("parse subgroup exchange");
        let hip_source = hip::lower_compute(&module).expect("lower subgroup exchange to HIP").source;
        let cuda_source =
            cuda::lower_compute(&module).expect("lower subgroup exchange to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower subgroup exchange to MSL").source;
        assert!(hip_source.contains("__shfl("));
        assert!(hip_source.contains("__shfl_xor("));
        assert!(hip_source.contains("__shfl_up("));
        assert!(hip_source.contains("__shfl_down("));
        assert!(cuda_source.contains("__shfl_sync("));
        assert!(cuda_source.contains("__shfl_xor_sync("));
        assert!(cuda_source.contains("__shfl_up_sync("));
        assert!(cuda_source.contains("__shfl_down_sync("));
        assert!(msl_source.contains("simd_broadcast("));
        assert!(msl_source.contains("simd_shuffle("));
        assert!(msl_source.contains("simd_shuffle_xor("));
        assert!(msl_source.contains("simd_shuffle_up("));
        assert!(msl_source.contains("simd_shuffle_down("));
        spirv::lower_compute(&module).expect("lower subgroup exchange to SPIR-V");
    }

    #[test]
    fn subgroup_arithmetic_lowers_on_all_compute_backends() {
        let module = parse_compute(SUBGROUP_ARITHMETIC).expect("parse subgroup arithmetic");
        let hip_source =
            hip::lower_compute(&module).expect("lower subgroup arithmetic to HIP").source;
        let cuda_source =
            cuda::lower_compute(&module).expect("lower subgroup arithmetic to CUDA").source;
        let msl_source =
            msl::lower_compute(&module).expect("lower subgroup arithmetic to MSL").source;
        assert!(hip_source.contains("auto o = __shfl_xor(v, off)"));
        assert!(hip_source.contains("auto o = __shfl_up(v, off)"));
        assert!(hip_source.contains("fminf(v, o)"));
        assert!(hip_source.contains("(v & o)"));
        assert!(hip_source.contains("excl = 0.0f"));
        assert!(hip_source.contains("excl = 0u"));
        assert!(cuda_source.contains("__shfl_xor_sync(0xffffffffu, v, off)"));
        assert!(cuda_source.contains("__shfl_up_sync(0xffffffffu, v, off)"));
        assert!(cuda_source.contains("(threadIdx.x % warpSize) >= off"));
        assert!(msl_source.contains("simd_sum("));
        assert!(msl_source.contains("simd_min("));
        assert!(msl_source.contains("simd_max("));
        assert!(msl_source.contains("simd_and("));
        assert!(msl_source.contains("simd_or("));
        assert!(msl_source.contains("simd_prefix_inclusive_sum("));
        assert!(msl_source.contains("simd_prefix_exclusive_sum("));
        spirv::lower_compute(&module).expect("lower subgroup arithmetic to SPIR-V");
    }

    #[test]
    fn workgroup_reduce_lowers_on_all_compute_backends() {
        let module = parse_compute(WORKGROUP_REDUCE).expect("parse workgroup reduce");
        let hip_source = hip::lower_compute(&module)
            .expect("lower workgroup reduce to HIP")
            .source;
        let cuda_source = cuda::lower_compute(&module)
            .expect("lower workgroup reduce to CUDA")
            .source;
        let msl_source = msl::lower_compute(&module)
            .expect("lower workgroup reduce to MSL")
            .source;
        assert!(hip_source.contains("__shared__ float __zsl_wg_reduce_f32[256]"));
        assert!(hip_source.contains("fminf(v, o)"));
        assert!(hip_source.contains("fmaxf(v, o)"));
        assert!(hip_source.contains("__zsl_wg_reduce_f32[0]"));
        assert!(cuda_source.contains("__shared__ float __zsl_wg_reduce_f32[256]"));
        assert!(cuda_source.contains("__syncthreads()"));
        assert!(msl_source.contains("threadgroup float __zsl_wg_reduce_f32[256]"));
        assert!(msl_source.contains("min(v, o)"));
        assert!(msl_source.contains("max(v, o)"));
        assert!(msl_source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        spirv::lower_compute(&module).expect("lower workgroup reduce to SPIR-V");
    }

    #[test]
    fn workgroup_reduce_rejects_non_power_of_two_workgroups() {
        let source = r#"
            @workgroup_size(48)
            kernel bad(out: device mut buffer<f32>, id: global_id) {
                out[id.x] = workgroup_reduce_add(1.0)
            }
        "#;
        let module = parse_compute(source).expect("parse non-power-of-two workgroup reduce");
        let expected =
            "workgroup_reduce_* requires a power-of-two workgroup thread count, got 48";
        assert!(
            hip::lower_compute(&module)
                .err()
                .expect("HIP must reject")
                .message
                .contains(expected)
        );
        assert!(
            cuda::lower_compute(&module)
                .err()
                .expect("CUDA must reject")
                .message
                .contains(expected)
        );
        assert!(
            msl::lower_compute(&module)
                .err()
                .expect("MSL must reject")
                .message
                .contains(expected)
        );
        assert!(
            spirv::lower_compute(&module)
                .expect_err("SPIR-V must reject")
                .message
                .contains(expected)
        );
    }

    #[test]
    fn subgroup_query_kernel_without_use_has_no_msl_simd_params() {
        let module = parse_compute(COPY_BF16).expect("parse copy_bf16");
        let msl_source = msl::lower_compute(&module).expect("lower copy_bf16 to MSL").source;
        assert!(!msl_source.contains("simdgroup_index_in_threadgroup"));
    }

    #[test]
    fn subgroup_query_hip_matches_real_wave_layout_when_gpu_present() {
        let module = parse_compute(SUBGROUP_QUERY).expect("parse query");
        let shader = hip::lower_compute(&module).expect("lower query to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP subgroup query proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_subgroup_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_subgroup_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned WG = 256, N = WG * 256;
    std::vector<unsigned> size(N, 0), id(N, 0), lane(N, 0);
    unsigned *dsize = nullptr, *did = nullptr, *dlane = nullptr;
    if (hipMalloc(&dsize, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&did, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&dlane, N*sizeof(unsigned)) != hipSuccess) return 2;
    zsl_kernel<<<dim3(N/WG), dim3(WG)>>>(dsize, did, dlane);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(size.data(), dsize, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(id.data(), did, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(lane.data(), dlane, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) {{
        unsigned local = i % WG;
        bool ok = size[i] == 32 && lane[i] < size[i] && (id[i] * size[i] + lane[i]) == local;
        mismatches += !ok;
    }}
    std::printf("SUBGROUP_QUERY_EXACT mismatches=%u\n", mismatches);
    hipFree(dsize); hipFree(did); hipFree(dlane);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP subgroup query proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP subgroup query compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path).output().expect("run HIP query");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP query failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP query: {}", stdout.trim());
    }

    #[test]
    fn subgroup_any_and_all_hip_match_real_wave_votes_when_gpu_present() {
        let module = parse_compute(SUBGROUP_VOTE).expect("parse vote");
        let shader = hip::lower_compute(&module).expect("lower vote to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP subgroup vote proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_subgroup_vote_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_subgroup_vote_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned WG = 256, GROUPS = 256, N = WG * GROUPS;
    std::vector<unsigned> any(N, 0), all(N, 1);
    unsigned *dany = nullptr, *dall = nullptr;
    if (hipMalloc(&dany, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&dall, N*sizeof(unsigned)) != hipSuccess) return 2;
    zsl_kernel<<<dim3(GROUPS), dim3(WG)>>>(dany, dall, GROUPS);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(any.data(), dany, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(all.data(), dall, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) {{
        mismatches += any[i] != 1u;
        mismatches += all[i] != 0u;
    }}
    std::printf("SUBGROUP_VOTE_EXACT mismatches=%u\n", mismatches);
    hipFree(dany); hipFree(dall);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP subgroup vote proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP subgroup vote compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path).output().expect("run HIP vote");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            !stdout.contains("SKIP_NO_HIP_GPU"),
            "HIP subgroup vote proof unexpectedly skipped"
        );
        assert!(
            run.status.success(),
            "HIP vote failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP subgroup vote: {}", stdout.trim());
    }

    #[test]
    fn subgroup_shuffle_xor_hip_swaps_adjacent_lanes_when_gpu_present() {
        let module = parse_compute(SUBGROUP_SHUFFLE_XOR).expect("parse subgroup shuffle xor");
        let shader = hip::lower_compute(&module).expect("lower subgroup shuffle xor to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!(
                "skipping HIP subgroup shuffle-xor proof: {} not found",
                hipcc.display()
            );
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!(
            "zsl_subgroup_shuffle_xor_{}_{}.hip",
            std::process::id(),
            tag
        ));
        let exe_path = dir.join(format!(
            "zsl_subgroup_shuffle_xor_{}_{}",
            std::process::id(),
            tag
        ));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned WG = 256, GROUPS = 256, N = WG * GROUPS;
    std::vector<float> out(N, -1.0f);
    float *dout = nullptr;
    if (hipMalloc(&dout, N*sizeof(float)) != hipSuccess) return 2;
    zsl_kernel<<<dim3(GROUPS), dim3(WG)>>>(dout);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(out.data(), dout, N*sizeof(float), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) {{
        const unsigned local = i % WG;
        const unsigned subgroup_local = local % 32;
        const unsigned expected_id = i - subgroup_local + (subgroup_local ^ 1u);
        mismatches += out[i] != (float)expected_id;
    }}
    std::printf("SUBGROUP_SHUFFLE_XOR_EXACT mismatches=%u\n", mismatches);
    hipFree(dout);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP subgroup shuffle-xor proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP subgroup shuffle-xor compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path)
            .output()
            .expect("run HIP subgroup shuffle xor");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            !stdout.contains("SKIP_NO_HIP_GPU"),
            "HIP subgroup shuffle-xor proof unexpectedly skipped"
        );
        assert!(
            run.status.success(),
            "HIP subgroup shuffle-xor failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP subgroup shuffle xor: {}", stdout.trim());
    }

    #[test]
    fn subgroup_arithmetic_hip_exact_invariants_when_gpu_present() {
        let module =
            parse_compute(SUBGROUP_ARITHMETIC_PROOF).expect("parse subgroup arithmetic proof");
        let shader =
            hip::lower_compute(&module).expect("lower subgroup arithmetic proof to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!(
                "skipping HIP subgroup arithmetic proof: {} not found",
                hipcc.display()
            );
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!(
            "zsl_subgroup_arithmetic_{}_{}.hip",
            std::process::id(),
            tag
        ));
        let exe_path = dir.join(format!(
            "zsl_subgroup_arithmetic_{}_{}",
            std::process::id(),
            tag
        ));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned WG = 256, GROUPS = 256, N = WG * GROUPS;
    std::vector<unsigned> reduce_add(N), scan_inclusive(N), scan_exclusive(N), reduce_min(N), reduce_max(N);
    unsigned *d_reduce_add = nullptr, *d_scan_inclusive = nullptr, *d_scan_exclusive = nullptr;
    unsigned *d_reduce_min = nullptr, *d_reduce_max = nullptr;
    if (hipMalloc(&d_reduce_add, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&d_scan_inclusive, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&d_scan_exclusive, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&d_reduce_min, N*sizeof(unsigned)) != hipSuccess ||
        hipMalloc(&d_reduce_max, N*sizeof(unsigned)) != hipSuccess) return 2;
    zsl_kernel<<<dim3(GROUPS), dim3(WG)>>>(
        d_reduce_add, d_scan_inclusive, d_scan_exclusive, d_reduce_min, d_reduce_max);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(reduce_add.data(), d_reduce_add, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(scan_inclusive.data(), d_scan_inclusive, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(scan_exclusive.data(), d_scan_exclusive, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(reduce_min.data(), d_reduce_min, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    hipMemcpy(reduce_max.data(), d_reduce_max, N*sizeof(unsigned), hipMemcpyDeviceToHost);
    unsigned add_mismatches = 0, inclusive_mismatches = 0, exclusive_mismatches = 0;
    unsigned min_mismatches = 0, max_mismatches = 0;
    for (unsigned i = 0; i < N; ++i) {{
        const unsigned lane = i % 32u;
        add_mismatches += reduce_add[i] != 32u;
        inclusive_mismatches += scan_inclusive[i] != lane + 1u;
        exclusive_mismatches += scan_exclusive[i] != lane;
        min_mismatches += reduce_min[i] != 0u;
        max_mismatches += reduce_max[i] != 31u;
    }}
    std::printf(
        "SUBGROUP_ARITHMETIC_EXACT reduce_add=%u scan_inclusive=%u scan_exclusive=%u reduce_min=%u reduce_max=%u\n",
        add_mismatches, inclusive_mismatches, exclusive_mismatches, min_mismatches, max_mismatches);
    hipFree(d_reduce_add);
    hipFree(d_scan_inclusive);
    hipFree(d_scan_exclusive);
    hipFree(d_reduce_min);
    hipFree(d_reduce_max);
    return add_mismatches + inclusive_mismatches + exclusive_mismatches +
        min_mismatches + max_mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP subgroup arithmetic proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP subgroup arithmetic compilation failed:\n{}\nsource: {}",
                String::from_utf8_lossy(&compile.stderr),
                src_path.display()
            );
        }
        let run = Command::new(&exe_path)
            .output()
            .expect("run HIP subgroup arithmetic");
        let stdout = String::from_utf8_lossy(&run.stdout);
        if run.status.success() {
            let _ = std::fs::remove_file(&src_path);
            let _ = std::fs::remove_file(&exe_path);
        }
        assert!(
            !stdout.contains("SKIP_NO_HIP_GPU"),
            "HIP subgroup arithmetic proof unexpectedly skipped"
        );
        assert!(
            run.status.success(),
            "HIP subgroup arithmetic failed: {stdout}\n{}\nsource: {}",
            String::from_utf8_lossy(&run.stderr),
            src_path.display()
        );
        eprintln!("HIP subgroup arithmetic: {}", stdout.trim());
    }

    #[test]
    fn workgroup_reduce_hip_exact_invariants_when_gpu_present() {
        let module = parse_compute(WORKGROUP_REDUCE_PROOF).expect("parse workgroup reduce proof");
        let shader = hip::lower_compute(&module).expect("lower workgroup reduce proof to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!(
                "skipping HIP workgroup reduce proof: {} not found",
                hipcc.display()
            );
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!(
            "zsl_workgroup_reduce_{}_{}.hip",
            std::process::id(),
            tag
        ));
        let exe_path = dir.join(format!(
            "zsl_workgroup_reduce_{}_{}",
            std::process::id(),
            tag
        ));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned WG = 256, GROUPS = 256, N = WG * GROUPS;
    std::vector<float> add(N), min(N), max(N);
    float *d_add = nullptr, *d_min = nullptr, *d_max = nullptr;
    if (hipMalloc(&d_add, N*sizeof(float)) != hipSuccess ||
        hipMalloc(&d_min, N*sizeof(float)) != hipSuccess ||
        hipMalloc(&d_max, N*sizeof(float)) != hipSuccess) return 2;
    zsl_kernel<<<dim3(GROUPS), dim3(WG)>>>(d_add, d_min, d_max);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(add.data(), d_add, N*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(min.data(), d_min, N*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(max.data(), d_max, N*sizeof(float), hipMemcpyDeviceToHost);
    unsigned add_mismatches = 0, min_mismatches = 0, max_mismatches = 0;
    for (unsigned i = 0; i < N; ++i) {{
        add_mismatches += add[i] != 256.0f;
        min_mismatches += min[i] != 0.0f;
        max_mismatches += max[i] != 255.0f;
    }}
    std::printf(
        "WORKGROUP_REDUCE_EXACT add=%u min=%u max=%u\n",
        add_mismatches, min_mismatches, max_mismatches);
    hipFree(d_add);
    hipFree(d_min);
    hipFree(d_max);
    return add_mismatches + min_mismatches + max_mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP workgroup reduce proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP workgroup reduce compilation failed:\n{}\nsource: {}",
                String::from_utf8_lossy(&compile.stderr),
                src_path.display()
            );
        }
        let run = Command::new(&exe_path)
            .output()
            .expect("run HIP workgroup reduce");
        let stdout = String::from_utf8_lossy(&run.stdout);
        if run.status.success() {
            let _ = std::fs::remove_file(&src_path);
            let _ = std::fs::remove_file(&exe_path);
        }
        assert!(
            !stdout.contains("SKIP_NO_HIP_GPU"),
            "HIP workgroup reduce proof unexpectedly skipped"
        );
        assert!(
            run.status.success(),
            "HIP workgroup reduce failed: {stdout}\n{}\nsource: {}",
            String::from_utf8_lossy(&run.stderr),
            src_path.display()
        );
        eprintln!("HIP workgroup reduce: {}", stdout.trim());
    }

    #[test]
    fn mul_add_f32_lowers_on_all_compute_backends() {
        let module = parse_compute(MUL_ADD_F32).expect("parse fma3");
        let hip_source = hip::lower_compute(&module).expect("lower fma3 to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower fma3 to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower fma3 to MSL").source;
        assert!(hip_source.contains("fmaf(a[gx], b[gx], c[gx])"));
        assert!(cuda_source.contains("fmaf(a[gx], b[gx], c[gx])"));
        assert!(msl_source.contains("fma(a[gid.x], b[gid.x], c[gid.x])"));
        spirv::lower_compute(&module).expect("lower fma3 to SPIR-V");
    }

    #[test]
    fn mul_add_f32_hip_matches_rust_fma_when_gpu_present() {
        let module = parse_compute(MUL_ADD_F32).expect("parse fma3");
        let shader = hip::lower_compute(&module).expect("lower fma3 to HIP");

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP mul_add_f32 proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_fma_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_fma_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned N = 65536;
    std::vector<float> a(N), b(N), c(N), out(N, 0.0f), ref(N, 0.0f);
    for (unsigned i = 0; i < N; ++i) {{
        a[i] = 1.0000001f + (float)(i % 997) * 0.000013f;
        b[i] = 1.0000001f - (float)(i % 883) * 0.000017f;
        c[i] = (float)((int)(i % 2011) - 1005) * 0.0009f;
        ref[i] = fmaf(a[i], b[i], c[i]);
    }}
    float *da = nullptr, *db = nullptr, *dc = nullptr, *dout = nullptr;
    if (hipMalloc(&da, a.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&db, b.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&dc, c.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&dout, out.size()*sizeof(float)) != hipSuccess) return 2;
    hipMemcpy(da, a.data(), a.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(db, b.data(), b.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dc, c.data(), c.size()*sizeof(float), hipMemcpyHostToDevice);
    zsl_kernel<<<dim3((N+63)/64), dim3(64)>>>(da, db, dc, dout);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(out.data(), dout, out.size()*sizeof(float), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < N; ++i) mismatches += out[i] != ref[i];
    std::printf("FMA_EXACT mismatches=%u\n", mismatches);
    hipFree(da); hipFree(db); hipFree(dc); hipFree(dout);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP fma proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP fma compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path).output().expect("run HIP fma3");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP fma3 failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP fma3: {}", stdout.trim());
    }

    #[test]
    fn trigonometry_lowers_on_all_compute_backends() {
        let module = parse_compute(TRIG).expect("parse trigonometry");
        let hip_source = hip::lower_compute(&module).expect("lower trigonometry to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower trigonometry to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower trigonometry to MSL").source;
        assert!(hip_source.contains("sinf(x)"));
        assert!(hip_source.contains("cosf(x)"));
        assert!(hip_source.contains("tanf(x)"));
        assert!(cuda_source.contains("sinf(x)"));
        assert!(cuda_source.contains("cosf(x)"));
        assert!(cuda_source.contains("tanf(x)"));
        assert!(msl_source.contains("sin(x)"));
        assert!(msl_source.contains("cos(x)"));
        assert!(msl_source.contains("tan(x)"));
        spirv::lower_compute(&module).expect("lower trigonometry to SPIR-V");
    }

    #[test]
    fn spirv_bindless_descriptor_array_declares_extension() {
        let module = parse_compute(TRIG).expect("parse bindless-buffer shader");
        let words = spirv::lower_compute(&module).expect("lower bindless-buffer shader to SPIR-V");
        let mut at = 5usize;
        let mut runtime_descriptor_array = false;
        let mut descriptor_indexing_extension = false;
        while at < words.len() {
            let wc = (words[at] >> 16) as usize;
            let opcode = words[at] & 0xffff;
            assert!(
                wc > 0 && at + wc <= words.len(),
                "malformed SPIR-V instruction at word {at}"
            );
            if opcode == 17 && wc == 2 && words[at + 1] == 5302 {
                runtime_descriptor_array = true;
            }
            if opcode == 10 {
                let bytes: Vec<u8> = words[at + 1..at + wc]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect();
                descriptor_indexing_extension |= bytes == b"SPV_EXT_descriptor_indexing";
            }
            at += wc;
        }
        assert!(
            runtime_descriptor_array,
            "missing RuntimeDescriptorArray capability"
        );
        assert!(
            descriptor_indexing_extension,
            "missing SPV_EXT_descriptor_indexing extension"
        );
    }

    #[test]
    fn u32_cast_lowers_on_all_compute_backends() {
        let source = r#"
            @workgroup_size(1)
            kernel cast_index(src: device buffer<f32>, out: device mut buffer<f32>, id: global_id) {
                let index = u32(src[id.x])
                out[id.x] = src[index]
            }
        "#;
        let module = parse_compute(source).expect("parse u32 cast");
        assert!(
            hip::lower_compute(&module)
                .expect("lower u32 cast to HIP")
                .source
                .contains("(unsigned int)")
        );
        assert!(
            cuda::lower_compute(&module)
                .expect("lower u32 cast to CUDA")
                .source
                .contains("(unsigned int)")
        );
        assert!(msl::lower_compute(&module).expect("lower u32 cast to MSL").source.contains("uint("));
        spirv::lower_compute(&module).expect("lower u32 cast to SPIR-V");
    }

    #[test]
    fn scatter_add_lowers_on_all_backends_and_has_spirv_atomic_extension() {
        let module = parse_compute(SCATTER_ADD).expect("parse scatter_add");
        let hip_source = hip::lower_compute(&module).expect("lower scatter_add to HIP").source;
        let cuda_source = cuda::lower_compute(&module).expect("lower scatter_add to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower scatter_add to MSL").source;
        assert!(hip_source.contains("atomicAdd(&out["));
        assert!(cuda_source.contains("atomicAdd(&out["));
        assert!(msl_source.contains("device atomic_float* out"));
        assert!(msl_source.contains("atomic_fetch_add_explicit(&out["));
        assert!(msl_source.contains("memory_order_relaxed"));

        let words = spirv::lower_compute(&module).expect("lower scatter_add to SPIR-V");
        let mut at = 5usize;
        let mut atomic_fadds = 0;
        let mut atomic_float_caps = 0;
        let mut atomic_extension = false;
        while at < words.len() {
            let wc = (words[at] >> 16) as usize;
            let opcode = words[at] & 0xffff;
            assert!(
                wc > 0 && at + wc <= words.len(),
                "malformed SPIR-V instruction at word {at}"
            );
            if opcode == 6035 {
                assert_eq!(wc, 7, "malformed OpAtomicFAddEXT");
                atomic_fadds += 1;
            }
            if opcode == 17 && wc == 2 && words[at + 1] == 6033 {
                atomic_float_caps += 1;
            }
            if opcode == 10 {
                let bytes: Vec<u8> = words[at + 1..at + wc]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect();
                atomic_extension |= bytes == b"SPV_EXT_shader_atomic_float";
            }
            at += wc;
        }
        assert_eq!(atomic_fadds, 1, "missing OpAtomicFAddEXT");
        assert_eq!(
            atomic_float_caps, 1,
            "missing AtomicFloat32AddEXT capability"
        );
        assert!(
            atomic_extension,
            "missing SPV_EXT_shader_atomic_float extension"
        );
    }

    #[test]
    fn scatter_add_hip_exact_match_when_gpu_present() {
        let module = parse_compute(SCATTER_ADD).expect("parse scatter_add");
        let shader = hip::lower_compute(&module).expect("lower scatter_add to HIP");
        assert!(shader.source.contains("atomicAdd(&out["));

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!("skipping HIP scatter proof: {} not found", hipcc.display());
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!(
            "zsl_scatter_add_{}_{}.hip",
            std::process::id(),
            tag
        ));
        let exe_path = dir.join(format!("zsl_scatter_add_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned N = 65536, BINS = 37;
    std::vector<float> idx(N), val(N), out(BINS, 0.0f), ref(BINS, 0.0f);
    for (unsigned i = 0; i < N; ++i) {{
        unsigned bin = (i * 17 + i / 11) % BINS;
        idx[i] = float(bin);
        val[i] = float((i % 4) + 1);
        ref[bin] += val[i];
    }}
    float *didx = nullptr, *dval = nullptr, *dout = nullptr;
    if (hipMalloc(&didx, idx.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&dval, val.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&dout, out.size()*sizeof(float)) != hipSuccess) return 2;
    hipMemcpy(didx, idx.data(), idx.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dval, val.data(), val.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(dout, 0, out.size()*sizeof(float));
    zsl_kernel<<<dim3((N+255)/256), dim3(256)>>>(didx, dval, dout, N);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(out.data(), dout, out.size()*sizeof(float), hipMemcpyDeviceToHost);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < BINS; ++i) mismatches += out[i] != ref[i];
    std::printf("SCATTER_EXACT mismatches=%u\n", mismatches);
    hipFree(didx); hipFree(dval); hipFree(dout);
    return mismatches == 0 ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP scatter proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP scatter compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path)
            .output()
            .expect("run HIP scatter_add");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP scatter_add failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP scatter_add: {}", stdout.trim());
    }

    #[test]
    fn tiled_gemm_spirv_has_workgroup_memory_and_barrier() {
        let module = parse_compute(TILED_GEMM).expect("parse tiled GEMM");
        assert_eq!(module.entry.shared.len(), 2);
        let cuda_source = cuda::lower_compute(&module).expect("lower tiled GEMM to CUDA").source;
        let msl_source = msl::lower_compute(&module).expect("lower tiled GEMM to MSL").source;
        assert!(cuda_source.contains("__shared__ float as_tile[256]"));
        assert!(cuda_source.contains("threadIdx.x") && cuda_source.contains("blockIdx.y"));
        assert!(cuda_source.contains("__syncthreads()"));
        assert!(msl_source.contains("threadgroup float as_tile[256]"));
        assert!(msl_source.contains("thread_position_in_threadgroup"));
        assert!(msl_source.contains("threadgroup_position_in_grid"));
        assert!(msl_source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        let words = spirv::lower_compute(&module).expect("lower tiled GEMM to SPIR-V");
        let mut at = 5usize;
        let mut workgroup_vars = 0;
        let mut barriers = 0;
        while at < words.len() {
            let wc = (words[at] >> 16) as usize;
            let opcode = words[at] & 0xffff;
            assert!(
                wc > 0 && at + wc <= words.len(),
                "malformed SPIR-V instruction at word {at}"
            );
            if opcode == 59 && wc >= 4 && words[at + 3] == 4 {
                workgroup_vars += 1;
            }
            if opcode == 224 {
                assert_eq!(wc, 4, "malformed OpControlBarrier");
                barriers += 1;
            }
            at += wc;
        }
        assert_eq!(
            at,
            words.len(),
            "SPIR-V instruction stream ended mid-instruction"
        );
        assert_eq!(workgroup_vars, 2, "missing Workgroup variables");
        assert_eq!(barriers, 2, "missing control barriers");
    }

    #[test]
    fn tiled_gemm_hip_compiles_and_matches_reference_when_gpu_present() {
        let module = parse_compute(TILED_GEMM).expect("parse tiled GEMM");
        let shader = hip::lower_compute(&module).expect("lower tiled GEMM to HIP");
        assert!(shader.source.contains("__shared__ float as_tile[256]"));
        assert!(shader.source.contains("__syncthreads()"));

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let hipcc = std::path::Path::new(&rocm).join("bin/hipcc");
        if !hipcc.exists() {
            eprintln!(
                "skipping HIP compile/runtime proof: {} not found",
                hipcc.display()
            );
            return;
        }

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("zsl_tiled_gemm_{}_{}.hip", std::process::id(), tag));
        let exe_path = dir.join(format!("zsl_tiled_gemm_{}_{}", std::process::id(), tag));
        let harness = format!(
            r#"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
{}
int main() {{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {{ std::puts("SKIP_NO_HIP_GPU"); return 0; }}
    const unsigned M = 31, N = 29, K = 23;
    std::vector<float> a(M*K), b(K*N), c(M*N, 0.0f);
    for (unsigned i = 0; i < M*K; ++i) a[i] = float(int(i % 17) - 8) * 0.03125f;
    for (unsigned i = 0; i < K*N; ++i) b[i] = float(int(i % 13) - 6) * 0.025f;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    if (hipMalloc(&da, a.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&db, b.size()*sizeof(float)) != hipSuccess ||
        hipMalloc(&dc, c.size()*sizeof(float)) != hipSuccess) return 2;
    hipMemcpy(da, a.data(), a.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(db, b.data(), b.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(dc, 0, c.size()*sizeof(float));
    zsl_kernel<<<dim3((N+15)/16, (M+15)/16), dim3(16,16)>>>(da, db, dc, M, N, K);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(c.data(), dc, c.size()*sizeof(float), hipMemcpyDeviceToHost);
    float max_err = 0.0f;
    for (unsigned row = 0; row < M; ++row) for (unsigned col = 0; col < N; ++col) {{
        float ref = 0.0f;
        for (unsigned q = 0; q < K; ++q) ref += a[row*K+q] * b[q*N+col];
        max_err = std::fmax(max_err, std::fabs(c[row*N+col] - ref));
    }}
    std::printf("MAX_ERR %.9g\n", max_err);
    hipFree(da); hipFree(db); hipFree(dc);
    return max_err < 1.0e-3f ? 0 : 4;
}}
"#,
            shader.source
        );
        std::fs::write(&src_path, harness).expect("write HIP proof source");
        let compile = Command::new(&hipcc)
            .arg("-O2")
            .arg(&src_path)
            .arg("-o")
            .arg(&exe_path)
            .output()
            .expect("run hipcc");
        if !compile.status.success() {
            panic!(
                "HIP compilation failed:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        let run = Command::new(&exe_path)
            .output()
            .expect("run HIP tiled GEMM");
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&exe_path);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "HIP tiled GEMM failed: {stdout}\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        eprintln!("HIP tiled GEMM: {}", stdout.trim());
        if !stdout.contains("SKIP_NO_HIP_GPU") {
            let max_err: f32 = stdout.split_whitespace().last().unwrap().parse().unwrap();
            assert!(max_err < 1.0e-3, "HIP max error {max_err}");
        }
    }
}
