//! ZenGPU shader proc-macro internals.
//!
//! Do not use this crate directly — import `zengpu_spirv` or `zengpu` and use
//! the `zengpu_spirv!` macro.

extern crate proc_macro;

mod backend;
mod frontend;
mod ir;

use proc_macro::TokenStream;

/// Derive `to_scalars()` for a push-constant struct.
///
/// Every field must be `u32`, `i32`, or `f32`. The generated method returns a
/// fixed-size array of `zengpu_hal::Scalar` values in field-declaration order,
/// suitable for passing as `Bindings::scalars` in a dispatch call.
///
/// Import via `use zengpu_spirv::ZslPushConst;`.
#[proc_macro_derive(ZslPushConst)]
pub fn derive_zsl_push_const(input: TokenStream) -> TokenStream {
    match push_const_impl(input) {
        Ok(ts) => ts,
        Err(msg) => compile_error_tokens(&msg),
    }
}

/// Hand-rolled `#[derive(ZslPushConst)]` over the builtin `proc_macro` token
/// trees — no `syn`/`quote`. Generates `to_scalars()` returning a fixed array of
/// `Scalar` in field order; `ZslMat4` expands to 16 column-major `Scalar::F32`.
fn push_const_impl(input: TokenStream) -> Result<TokenStream, String> {
    use proc_macro::{Delimiter, TokenTree};

    // Find `struct <Name> { <fields> }`, skipping attributes/visibility/generics.
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

    // Parse `name : type ,` repeated. Only the type's leading ident is needed.
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

// ── Native ZSL macro (no syn/quote) ────────────────────────────────────────────

/// Compile **native ZSL** source to a [`ZslShader`] containing all backend
/// compiled forms built at compile time.
///
/// The returned [`::zengpu_spirv::ZslShader`] carries SPIR-V (Vulkan), HIP C++
/// (ROCm), MSL (Metal), and CUDA C++ (NVIDIA) — all compiled from the same ZSL
/// source. At runtime, pick the right form with
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

/// Build a `compile_error!` invocation token stream without `quote`.
fn compile_error_tokens(msg: &str) -> TokenStream {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    format!("::core::compile_error!{{\"{escaped}\"}}")
        .parse()
        .expect("compile_error invocation must parse")
}

#[cfg(test)]
mod phase_z_tests {
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
                .source
                .contains("(unsigned int)")
        );
        assert!(
            cuda::lower_compute(&module)
                .source
                .contains("(unsigned int)")
        );
        assert!(msl::lower_compute(&module).source.contains("uint("));
        spirv::lower_compute(&module).expect("lower u32 cast to SPIR-V");
    }

    #[test]
    fn scatter_add_lowers_on_all_backends_and_has_spirv_atomic_extension() {
        let module = parse_compute(SCATTER_ADD).expect("parse scatter_add");
        let hip_source = hip::lower_compute(&module).source;
        let cuda_source = cuda::lower_compute(&module).source;
        let msl_source = msl::lower_compute(&module).source;
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
        let shader = hip::lower_compute(&module);
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
        let cuda_source = cuda::lower_compute(&module).source;
        let msl_source = msl::lower_compute(&module).source;
        assert!(cuda_source.contains("__shared__ float as_tile[256]"));
        assert!(cuda_source.contains("threadIdx.x") && cuda_source.contains("blockIdx.y"));
        assert!(cuda_source.contains("__syncthreads()"));
        assert!(msl_source.contains("threadgroup float as_tile[256]"));
        assert!(msl_source.contains("thread_position_in_threadgroup"));
        assert!(msl_source.contains("threadgroup_position_in_grid"));
        assert!(msl_source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        let words = spirv::lower_compute(&module).expect("lower tiled GEMM to SPIR-V");
        // zengpu-spv's current opcode table predates OpTypeArray and
        // OpControlBarrier, so inspect those standard opcodes directly.
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
        let shader = hip::lower_compute(&module);
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
