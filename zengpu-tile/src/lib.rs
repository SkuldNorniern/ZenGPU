//! ZenGPU tile layer: kernels over tile blocks, lowered to ZSL and then to every backend.
//!
//! i8 inputs are `u32` buffers, four values per word, little-endian. `cols` must be a multiple
//! of 4.
//!
//! ```ignore
//! let mut b = TileKernel::builder("gemm", [32, 32]);
//! let a = b.input("a", DType::F32, [m, k]);
//! let w = b.input("b", DType::F32, [n, k]);
//! let c = b.output("c", DType::F32, [m, n]);
//! let mut acc = b.zeros(DType::F32, [32, 32]);
//! b.loop_k(k, 32, |kb| {
//!     let at = kb.load(Block::new(a, Affine::grid_m(), Affine::k(), [32, 32]));
//!     let bt = kb.load(Block::new(w, Affine::grid_n(), Affine::k(), [32, 32]));
//!     acc = kb.dot(at, bt, acc);
//! });
//! b.store(Block::new(c, Affine::grid_m(), Affine::grid_n(), [32, 32]), acc);
//! let kernel = b.build()?;
//! let compiled = compile(&kernel)?;
//! ```

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

use zengpu_zslc::backend::{cuda, hip, msl, spirv};
use zengpu_zslc::frontend::diagnostic::Diagnostic;
use zengpu_zslc::frontend::parser::parse_compute;

mod compiled;
mod ir;
mod lower;

pub use compiled::{BackendSource, CompiledTile, TileBinding, TileProgram};
pub use ir::{Affine, Block, DType, KernelBuilder, Op, Stmt, Tensor, Tile, TileKernel};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileError {
    Build(String),
    Lower(String),
    Backend {
        backend: &'static str,
        message: String,
    },
    Unsupported(String),
    Gpu(String),
}

impl Display for TileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TileError::Build(m) => write!(f, "tile build: {m}"),
            TileError::Lower(m) => write!(f, "tile lowering: {m}"),
            TileError::Backend { backend, message } => {
                write!(f, "tile {backend} backend: {message}")
            }
            TileError::Unsupported(m) => write!(f, "tile unsupported: {m}"),
            TileError::Gpu(m) => write!(f, "tile gpu: {m}"),
        }
    }
}

impl Error for TileError {}

/// backend failures reported on use, not at build, so one missing backend does not block the rest.
pub fn compile(kernel: &TileKernel) -> Result<CompiledTile, TileError> {
    let (zsl, workgroup) = lower::zsl_source(kernel)?;
    let module = parse_compute(&zsl).map_err(|e| {
        TileError::Lower(format!(
            "generated ZSL does not parse: {} (at {:?})",
            e.msg, e.at
        ))
    })?;
    let [grid_m, grid_n] = kernel.grid();
    let grid = [grid_n as u32, grid_m as u32, 1];
    let bindings = kernel
        .tensors
        .iter()
        .map(|t| TileBinding {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape,
            is_output: t.is_output,
        })
        .collect();
    Ok(CompiledTile {
        name: kernel.name.clone(),
        zsl,
        grid,
        workgroup: [workgroup, 1, 1],
        bindings,
        spirv: backend("spirv", spirv::lower_compute(&module).map(|w| w.to_vec())),
        hip: backend(
            "hip",
            hip::lower_compute(&module).map(|s| BackendSource {
                text: s.source,
                entry: s.entry,
            }),
        ),
        cuda: backend(
            "cuda",
            cuda::lower_compute(&module).map(|s| BackendSource {
                text: s.source,
                entry: s.entry,
            }),
        ),
        msl: backend(
            "msl",
            msl::lower_compute(&module).map(|s| BackendSource {
                text: s.source,
                entry: s.entry,
            }),
        ),
    })
}

fn backend<T>(name: &'static str, r: Result<T, Diagnostic>) -> Result<T, TileError> {
    r.map_err(|d| TileError::Backend {
        backend: name,
        message: d.message,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use zengpu_hal::{
        AdapterRequest, BufferDesc, BufferHandle, BufferUsage, DeviceRequest, GpuDevice,
        GpuInstance, MemoryUsage,
    };
    use zengpu_hip::HipInstance;
    use zengpu_vulkan::VulkanInstance;

    use super::*;

    /// Computes `C[M,N] = A[M,K] * B[N,K]^T + bias[N]`, cast to the output dtype.
    fn gemm(
        input: DType,
        out: DType,
        m: usize,
        n: usize,
        k: usize,
        tile: [usize; 2],
        bk: usize,
    ) -> Result<TileKernel, TileError> {
        let acc_dt = if input == DType::I8 {
            DType::I32
        } else {
            DType::F32
        };
        let mut b = TileKernel::builder("gemm", tile);
        let a = b.input("a", input, [m, k]);
        let w = b.input("b", input, [n, k]);
        let bias = b.input("bias", acc_dt, [1, n]);
        let c = b.output("c", out, [m, n]);
        let mut acc = b.zeros(acc_dt, tile);
        b.loop_k(k, bk, |kb| {
            let at = kb.load(Block::new(a, Affine::grid_m(), Affine::k(), [tile[0], bk]));
            let bt = kb.load(Block::new(w, Affine::grid_n(), Affine::k(), [tile[1], bk]));
            acc = kb.dot(at, bt, acc);
        });
        let bi = b.load(Block::new(
            bias,
            Affine::constant(0),
            Affine::grid_n(),
            [1, tile[1]],
        ));
        let sum = b.add(acc, bi);
        let cast = b.cast(sum, out);
        b.store(
            Block::new(c, Affine::grid_m(), Affine::grid_n(), tile),
            cast,
        );
        b.build()
    }

    #[test]
    fn builder_reports_invalid_kernels() {
        let origin = Affine::constant(0);
        let dot_dims = {
            let mut b = TileKernel::builder("bad", [16, 16]);
            let a = b.input("a", DType::F32, [32, 8]);
            let w = b.input("b", DType::F32, [16, 4]);
            let acc = b.zeros(DType::F32, [16, 16]);
            let at = b.load(Block::new(a, origin, origin, [16, 8]));
            let bt = b.load(Block::new(w, origin, origin, [16, 4]));
            let _ = b.dot(at, bt, acc);
            b.build()
        };
        assert!(matches!(dot_dims, Err(TileError::Build(m)) if m.contains("dot shapes")));

        let i8_cols = {
            let mut b = TileKernel::builder("bad", [16, 16]);
            b.input("a", DType::I8, [16, 6]);
            b.output("c", DType::I32, [16, 16]);
            b.build()
        };
        assert!(matches!(i8_cols, Err(TileError::Build(m)) if m.contains("multiple of 4")));

        let reuse = {
            let mut b = TileKernel::builder("bad", [16, 16]);
            let a = b.input("a", DType::F32, [16, 16]);
            let w = b.input("b", DType::F32, [16, 16]);
            b.output("c", DType::F32, [16, 16]);
            let acc = b.zeros(DType::F32, [16, 16]);
            let at = b.load(Block::new(a, origin, origin, [16, 16]));
            let bt = b.load(Block::new(w, origin, origin, [16, 16]));
            let _ = b.dot(at, bt, acc);
            let _ = b.add(acc, acc);
            b.build()
        };
        assert!(matches!(reuse, Err(TileError::Build(m)) if m.contains("used after dot consumed")));
    }

    #[test]
    fn gemm_display_snapshot() {
        let kernel = gemm(DType::F16, DType::F16, 100, 70, 37, [32, 32], 16).unwrap();
        let text = kernel.to_string();
        assert_eq!(text, EXPECTED_GEMM_TEXT);
    }

    #[test]
    fn gemm_lowers_to_every_backend() {
        let kernel = gemm(DType::F32, DType::F32, 64, 64, 64, [32, 32], 32).unwrap();
        let compiled = compile(&kernel).unwrap();
        assert_eq!(compiled.grid(), [2, 2, 1]);
        assert_eq!(compiled.workgroup(), [256, 1, 1]);
        assert!(compiled.zsl().contains("kernel gemm("));
        assert!(compiled.spirv().unwrap().len() > 5);
        assert!(compiled.hip().unwrap().text.contains("zsl_kernel"));
        assert!(compiled.cuda().is_ok());
        assert!(compiled.msl().is_ok());
    }

    fn hip_device() -> Option<Box<dyn GpuDevice>> {
        let inst = HipInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    fn vulkan_device() -> Option<Box<dyn GpuDevice>> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    #[derive(Clone, Copy)]
    struct Case {
        input: DType,
        out: DType,
        m: usize,
        n: usize,
        k: usize,
        tile: [usize; 2],
        bk: usize,
    }

    const CASES: [Case; 3] = [
        Case {
            input: DType::F32,
            out: DType::F32,
            m: 64,
            n: 64,
            k: 64,
            tile: [32, 32],
            bk: 32,
        },
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 100,
            n: 70,
            k: 37,
            tile: [32, 32],
            bk: 16,
        },
        Case {
            input: DType::I8,
            out: DType::I32,
            m: 48,
            n: 32,
            k: 64,
            tile: [16, 16],
            bk: 16,
        },
    ];

    #[test]
    fn gemm_on_hip() {
        let Some(dev) = hip_device() else { return };
        for case in CASES {
            run_case(&*dev, case).unwrap();
        }
    }

    #[test]
    fn gemm_on_vulkan() {
        let Some(dev) = vulkan_device() else { return };
        for case in CASES {
            run_case(&*dev, case).unwrap();
        }
    }

    /// Binary16 bits for multiples of 1/4 with magnitude 1/4-8, or zero.
    fn f16_bits(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as u16 - 112;
        sign | (exp << 10) | ((b & 0x007f_ffff) >> 13) as u16
    }

    fn f16_value(h: u16) -> f32 {
        let sign = h & 0x8000 != 0;
        let exp = i32::from((h >> 10) & 0x1f);
        let man = f32::from(h & 0x3ff);
        let mag = match exp {
            0 => man * 2f32.powi(-24),
            _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
        };
        if sign { -mag } else { mag }
    }

    fn run_case(dev: &dyn GpuDevice, case: Case) -> Result<(), String> {
        let Case {
            input,
            out,
            m,
            n,
            k,
            tile,
            bk,
        } = case;
        let kernel = gemm(input, out, m, n, k, tile, bk).map_err(|e| e.to_string())?;
        let compiled = compile(&kernel).map_err(|e| e.to_string())?;
        let program = compiled.load(dev).map_err(|e| e.to_string())?;

        let int8 = input == DType::I8;
        let (span, lo) = if int8 { (17, 8) } else { (9, 4) };
        let a_ref: Vec<i32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % span) as i32 - lo)
            .collect();
        let b_ref: Vec<i32> = (0..n * k)
            .map(|i| ((i * 5 + 1) % span) as i32 - lo)
            .collect();
        let bias_ref: Vec<i32> = (0..n).map(|j| ((j * 3) % 5) as i32 - 2).collect();

        let mut uploads = Vec::new();
        for binding in compiled.bindings() {
            let bytes = match binding.name.as_str() {
                "a" | "b" if int8 => {
                    let src = if binding.name == "a" { &a_ref } else { &b_ref };
                    let cols = binding.shape[1];
                    let mut words = vec![0u32; binding.shape[0] * cols / 4];
                    for (i, v) in src.iter().enumerate() {
                        let (r, c) = (i / cols, i % cols);
                        words[r * cols / 4 + c / 4] |= ((*v as i8) as u8 as u32) << (8 * (c % 4));
                    }
                    words.iter().flat_map(|w| w.to_le_bytes()).collect()
                }
                "a" | "b" => {
                    let src = if binding.name == "a" { &a_ref } else { &b_ref };
                    src.iter().flat_map(|&v| half_or_float(input, v)).collect()
                }
                "bias" if int8 => bias_ref.iter().flat_map(|v| v.to_le_bytes()).collect(),
                "bias" => bias_ref
                    .iter()
                    .flat_map(|&v| (v as f32 / 4.0).to_le_bytes())
                    .collect(),
                _ => vec![0u8; binding.shape[0] * binding.shape[1] * dtype_bytes(binding.dtype)],
            };
            uploads.push(bytes);
        }

        let mut handles: Vec<BufferHandle> = Vec::new();
        for bytes in &uploads {
            let h = dev
                .create_buffer(BufferDesc {
                    size: bytes.len() as u64,
                    usage: BufferUsage::STORAGE
                        | BufferUsage::READBACK
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                    memory: MemoryUsage::GpuOnly,
                })
                .map_err(|e| e.to_string())?;
            dev.write_buffer(h, 0, bytes).map_err(|e| e.to_string())?;
            handles.push(h);
        }
        program.dispatch(dev, &handles).map_err(|e| e.to_string())?;
        let out_index = compiled
            .bindings()
            .iter()
            .position(|b| b.is_output)
            .unwrap();
        let got = dev
            .read_buffer(handles[out_index], 0, uploads[out_index].len() as u64)
            .map_err(|e| e.to_string())?;
        for h in handles {
            dev.destroy_buffer(h);
        }
        program.destroy(dev);

        let mut bad = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let dot: i32 = (0..k).map(|kk| a_ref[r * k + kk] * b_ref[c * k + kk]).sum();
                let at = r * n + c;
                if int8 {
                    let want = dot + bias_ref[c];
                    let g = i32::from_le_bytes(got[at * 4..at * 4 + 4].try_into().unwrap());
                    if g != want {
                        bad.push(format!("({r},{c}) {g} != {want}"));
                    }
                    continue;
                }
                let want = dot as f32 / 16.0 + bias_ref[c] as f32 / 4.0;
                let (g, tol) = if out == DType::F16 {
                    let bits = u16::from_le_bytes(got[at * 2..at * 2 + 2].try_into().unwrap());
                    (f16_value(bits), want.abs() / 1024.0 + 1e-6)
                } else {
                    let g = f32::from_le_bytes(got[at * 4..at * 4 + 4].try_into().unwrap());
                    (g, 1e-4 * want.abs().max(1.0))
                };
                if (g - want).abs() > tol {
                    bad.push(format!("({r},{c}) {g} != {want}"));
                }
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{:?}->{:?} {m}x{n}x{k}: {} mismatches, first {}",
                input,
                out,
                bad.len(),
                bad[0]
            ))
        }
    }

    fn half_or_float(dtype: DType, v: i32) -> Vec<u8> {
        match dtype {
            DType::F16 => f16_bits(v as f32 / 4.0).to_le_bytes().to_vec(),
            _ => (v as f32 / 4.0).to_le_bytes().to_vec(),
        }
    }

    fn dtype_bytes(dtype: DType) -> usize {
        match dtype {
            DType::F16 => 2,
            DType::I8 => 1,
            _ => 4,
        }
    }

    const EXPECTED_GEMM_TEXT: &str = "tile v0 kernel gemm tile [32, 32]
input a f16 [100, 37]
input b f16 [70, 37]
input bias f32 [1, 70]
output c f16 [100, 70]
%0 = zeros f32 [32, 32]
loop k 0..37 step 16 {
  %1 = load a [gm, k] shape=[32, 16]
  %2 = load b [gn, k] shape=[32, 16]
  %3 = dot %1 %2 %0
}
%4 = load bias [0, gn] shape=[1, 32]
%5 = add %3 %4
%6 = cast %5 f16
store c [gm, gn] shape=[32, 32] %6
";
}
