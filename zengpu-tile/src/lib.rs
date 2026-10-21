//! Lowers tile kernels to ZSL, then to every backend that `zengpu-zslc` emits.
//! Schedules pick the form. Unsupported requests fall back, and `CompiledTile::notes` says why.
//! i8 tensors: one byte per element, read as i32 in the kernel.
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
mod schedule;

pub use compiled::{BackendSource, CompiledTile, TileBinding, TileProgram};
pub use ir::{Affine, Block, DType, KernelBuilder, Op, Stmt, Tensor, Tile, TileKernel};
pub use schedule::{Matrix, Schedule, Target};

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

/// O0 reference schedule. Backend errors are kept and reported on use, so one missing backend does
/// not stop the rest.
pub fn compile(kernel: &TileKernel) -> Result<CompiledTile, TileError> {
    compile_with(
        kernel,
        &Schedule::o0(),
        &Target::scalar_only(zengpu_hal::BackendPreference::Auto),
    )
}

/// Never fails on the schedule: unsupported requests fall back, with reasons in
/// `CompiledTile::notes`.
pub fn compile_with(
    kernel: &TileKernel,
    schedule: &Schedule,
    target: &Target,
) -> Result<CompiledTile, TileError> {
    let (plan, notes) = schedule::resolve(kernel, schedule, target);
    let zsl = lower::zsl_source(kernel, &plan)?;
    let aliases = lower::alias_tensors(kernel, &plan);
    let [grid_m, grid_n] = kernel.grid();
    let split = plan.split_k.max(1);
    let grid = [grid_n as u32, grid_m as u32, split as u32];
    let workgroup = plan.workgroup as u32;
    let bindings: Vec<TileBinding> = kernel
        .tensors
        .iter()
        .map(|t| TileBinding {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape,
            is_output: t.is_output,
        })
        .collect();
    let out = kernel
        .tensors
        .iter()
        .find(|t| t.is_output)
        .map_or([0, 0], |t| t.shape);
    let workspace = (split * out[0] * out[1] * 4) as u64 * u64::from(split > 1);
    let main = lower_backends(&zsl)?;
    let reduce = match lower::reduce_zsl(kernel, &plan)? {
        Some(rzsl) => {
            let r = lower_backends(&rzsl)?;
            Some(Box::new(CompiledTile {
                name: format!("{}_reduce", kernel.name),
                zsl: rzsl,
                grid: [grid_n as u32, grid_m as u32, 1],
                workgroup: [workgroup, 1, 1],
                bindings: bindings.clone(),
                mode: "reduce",
                subgroup: None,
                notes: Vec::new(),
                aliases: Vec::new(),
                workspace,
                reduce: None,
                spirv: r.spirv,
                hip: r.hip,
                cuda: r.cuda,
                msl: r.msl,
            }))
        }
        None => None,
    };
    Ok(CompiledTile {
        name: kernel.name.clone(),
        zsl,
        grid,
        workgroup: [workgroup, 1, 1],
        mode: match plan.mode {
            lower::Mode::Naive => "naive",
            lower::Mode::Staged => "staged",
            lower::Mode::Mma => "mma",
        },
        subgroup: (plan.mode == lower::Mode::Mma).then_some(plan.wave_size as u32),
        notes,
        bindings,
        aliases,
        workspace,
        reduce,
        spirv: main.spirv,
        hip: main.hip,
        cuda: main.cuda,
        msl: main.msl,
    })
}

struct Lowered {
    spirv: Result<Vec<u32>, TileError>,
    hip: Result<BackendSource, TileError>,
    cuda: Result<BackendSource, TileError>,
    msl: Result<BackendSource, TileError>,
}

/// A parse failure is an error. Backend failures are kept per field.
fn lower_backends(zsl: &str) -> Result<Lowered, TileError> {
    let module = parse_compute(zsl).map_err(|e| {
        TileError::Lower(format!(
            "generated ZSL does not parse: {} (at {:?})",
            e.msg, e.at
        ))
    })?;
    Ok(Lowered {
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

    use std::env;
    use std::process::Command;
    use std::time::{Duration, Instant};

    use zengpu_cuda::CudaInstance;
    use zengpu_hal::{
        AdapterRequest, BackendPreference, BufferDesc, BufferHandle, BufferUsage, DeviceRequest,
        GpuDevice, GpuInstance, MemoryUsage,
    };
    use zengpu_hip::HipInstance;
    use zengpu_metal::MetalInstance;
    use zengpu_vulkan::VulkanInstance;

    use super::*;

    /// Computes `C[M,N] = A[M,K] * B[N,K]^T + bias[N]`, cast to the output dtype. `rounded`: rounds
    /// the f16 accumulator to binary16 before the bias is added.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        input: DType,
        out: DType,
        m: usize,
        n: usize,
        k: usize,
        tile: [usize; 2],
        bk: usize,
        rounded: bool,
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
        if rounded {
            let r = b.cast(acc, DType::F16);
            acc = b.cast(r, DType::F32);
        }
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

        let i8_odd = {
            let mut b = TileKernel::builder("ok", [16, 16]);
            b.input("a", DType::I8, [16, 6]);
            b.output("c", DType::I32, [16, 16]);
            b.build()
        };
        assert!(i8_odd.is_ok());

        let k_width = {
            let mut b = TileKernel::builder("bad", [16, 16]);
            let a = b.input("a", DType::F32, [16, 64]);
            let w = b.input("b", DType::F32, [16, 64]);
            b.output("c", DType::F32, [16, 16]);
            let mut acc = b.zeros(DType::F32, [16, 16]);
            b.loop_k(64, 16, |kb| {
                let at = kb.load(Block::new(a, Affine::grid_m(), Affine::k(), [16, 32]));
                let bt = kb.load(Block::new(w, Affine::grid_n(), Affine::k(), [16, 32]));
                acc = kb.dot(at, bt, acc);
            });
            b.build()
        };
        assert!(matches!(k_width, Err(TileError::Build(m)) if m.contains("wide along k")));

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
    fn gemm_lowers_to_every_backend() {
        let kernel = gemm(DType::F32, DType::F32, 64, 64, 64, [32, 32], 32, false).unwrap();
        let compiled = compile(&kernel).unwrap();
        assert_eq!(compiled.grid(), [2, 2, 1]);
        assert_eq!(compiled.workgroup(), [256, 1, 1]);
        assert_eq!(compiled.schedule_name(), "naive");
        assert!(compiled.notes().is_empty());
        assert!(compiled.zsl().contains("kernel gemm("));
        assert!(compiled.spirv().unwrap().len() > 5);
        assert!(compiled.hip().unwrap().text.contains("zsl_kernel"));
        assert!(compiled.cuda().is_ok());
        assert!(compiled.msl().is_ok());
    }

    #[test]
    fn scheduled_forms_lower_to_every_backend() {
        let kernel = gemm(DType::F16, DType::F16, 100, 70, 37, [64, 64], 16, true).unwrap();
        let target = Target::hip_gfx12();
        let mma = compile_with(
            &kernel,
            &Schedule {
                matrix: Matrix::Mma,
                ..Schedule::default()
            },
            &target,
        )
        .unwrap();
        assert_eq!(mma.schedule_name(), "mma");
        assert!(mma.notes().is_empty());
        assert_eq!(mma.workgroup(), [128, 1, 1]);
        assert!(mma.zsl().contains("frag_load_a(sa"));
        assert!(mma.zsl().contains("frag_store(sc,"));
        assert!(mma.spirv().is_ok());
        assert!(mma.hip().is_ok());
        assert!(mma.cuda().is_ok());
        assert!(mma.msl().is_ok());

        let staged = compile_with(
            &kernel,
            &Schedule {
                matrix: Matrix::Scalar,
                ..Schedule::default()
            },
            &target,
        )
        .unwrap();
        assert_eq!(staged.schedule_name(), "staged");
        assert!(staged.spirv().is_ok() && staged.msl().is_ok());
    }

    #[test]
    fn unsupported_schedules_fall_back_with_notes() {
        let target = Target::hip_gfx12();
        let f32_kernel = gemm(DType::F32, DType::F32, 64, 64, 64, [32, 32], 32, false).unwrap();
        let mma = Schedule {
            matrix: Matrix::Mma,
            ..Schedule::default()
        };
        let got = compile_with(&f32_kernel, &mma, &target).unwrap();
        assert_eq!(got.schedule_name(), "staged");
        assert!(got.notes()[0].starts_with("mma: dot is f32 x f32"));

        let i8_kernel = gemm(DType::I8, DType::I32, 48, 32, 64, [16, 16], 8, false).unwrap();
        let got = compile_with(&i8_kernel, &mma, &target).unwrap();
        assert_eq!(got.schedule_name(), "naive");
        assert_eq!(got.notes().len(), 2, "{:?}", got.notes());

        let auto = compile_with(
            &kernel_f16(64, 64, 64),
            &Schedule::default(),
            &Target::scalar_only(BackendPreference::Auto),
        )
        .unwrap();
        assert_eq!(auto.schedule_name(), "staged");
        assert!(auto.notes()[0].starts_with("mma: target has no f16 matrix unit"));

        let odd = gemm(DType::F16, DType::F16, 100, 70, 37, [24, 32], 16, false).unwrap();
        let got = compile_with(&odd, &mma, &target).unwrap();
        assert_eq!(got.schedule_name(), "staged");
        assert!(got.notes()[0].contains("BM 24"));
    }

    fn kernel_f16(m: usize, n: usize, k: usize) -> TileKernel {
        gemm(DType::F16, DType::F16, m, n, k, [64, 64], 16, true).unwrap()
    }

    fn cuda_device() -> Option<Box<dyn GpuDevice>> {
        let inst = CudaInstance::new();
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
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

    fn metal_device() -> Option<Box<dyn GpuDevice>> {
        let inst = MetalInstance::new();
        let adapter = inst.request_adapter(AdapterRequest::default())?;
        adapter.open(DeviceRequest::default()).ok()
    }

    fn caps_target(dev: &dyn GpuDevice, backend: BackendPreference) -> Target {
        Target::from_caps(backend, &dev.compute_caps())
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

    /// Naive (O0) cases. f16 rounds the accumulator mid-kernel.
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

    /// Scheduled cases: edges in M, N, K. 64-wide tiles run four waves. i8 uses the matrix unit
    /// where the target has it.
    const SCHEDULED: [Case; 9] = [
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 64,
            n: 64,
            k: 64,
            tile: [64, 64],
            bk: 16,
        },
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 260,
            n: 130,
            k: 72,
            tile: [64, 64],
            bk: 16,
        },
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 100,
            n: 70,
            k: 37,
            tile: [64, 64],
            bk: 16,
        },
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 256,
            n: 128,
            k: 96,
            tile: [64, 64],
            bk: 32,
        },
        Case {
            input: DType::F32,
            out: DType::F32,
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
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 256,
            n: 256,
            k: 256,
            tile: [128, 128],
            bk: 32,
        },
        Case {
            input: DType::I8,
            out: DType::I32,
            m: 256,
            n: 256,
            k: 256,
            tile: [128, 128],
            bk: 32,
        },
        Case {
            input: DType::F16,
            out: DType::F16,
            m: 200,
            n: 136,
            k: 72,
            tile: [128, 128],
            bk: 32,
        },
    ];

    #[test]
    fn gemm_on_hip() {
        let Some(dev) = hip_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Hip);
        for case in CASES {
            run_case(&*dev, &target, case, &Schedule::o0()).unwrap();
        }
    }

    #[test]
    fn gemm_on_cuda() {
        let Some(dev) = cuda_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Cuda);
        for case in CASES {
            run_case(&*dev, &target, case, &Schedule::o0()).unwrap();
        }
    }

    #[test]
    fn gemm_on_vulkan() {
        let Some(dev) = vulkan_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Vulkan);
        for case in CASES {
            run_case(&*dev, &target, case, &Schedule::o0()).unwrap();
        }
    }

    #[test]
    fn gemm_on_metal() {
        let Some(dev) = metal_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Metal);
        for case in CASES {
            run_case(&*dev, &target, case, &Schedule::o0()).unwrap();
        }
    }

    #[test]
    fn scheduled_gemm_on_hip() {
        let Some(dev) = hip_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Hip);
        check_scheduled(&*dev, &target);
    }

    #[test]
    fn scheduled_gemm_on_cuda() {
        let Some(dev) = cuda_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Cuda);
        check_scheduled(&*dev, &target);
    }

    #[test]
    fn scheduled_gemm_on_vulkan() {
        let Some(dev) = vulkan_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Vulkan);
        check_scheduled(&*dev, &target);
    }

    #[test]
    fn scheduled_gemm_on_metal() {
        let Some(dev) = metal_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Metal);
        check_scheduled(&*dev, &target);
    }

    #[test]
    fn split_k_on_hip() {
        let Some(dev) = hip_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Hip);
        check_split(&*dev, &target);
    }

    #[test]
    fn split_k_on_cuda() {
        let Some(dev) = cuda_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Cuda);
        check_split(&*dev, &target);
    }

    #[test]
    fn split_k_on_vulkan() {
        let Some(dev) = vulkan_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Vulkan);
        check_split(&*dev, &target);
    }

    #[test]
    fn split_k_on_metal() {
        let Some(dev) = metal_device() else { return };
        let target = caps_target(&*dev, BackendPreference::Metal);
        check_split(&*dev, &target);
    }

    /// Split-K vs reference: uneven splits, K not divisible by split * bk, empty
    /// groups, edge tiles.
    fn check_split(dev: &dyn GpuDevice, target: &Target) {
        let cases = [
            (
                Case {
                    input: DType::F16,
                    out: DType::F16,
                    m: 256,
                    n: 256,
                    k: 256,
                    tile: [128, 128],
                    bk: 32,
                },
                4,
            ),
            (
                Case {
                    input: DType::F16,
                    out: DType::F16,
                    m: 260,
                    n: 130,
                    k: 72,
                    tile: [64, 64],
                    bk: 16,
                },
                3,
            ),
            (
                Case {
                    input: DType::F16,
                    out: DType::F16,
                    m: 100,
                    n: 70,
                    k: 37,
                    tile: [64, 64],
                    bk: 16,
                },
                2,
            ),
            (
                Case {
                    input: DType::F16,
                    out: DType::F16,
                    m: 256,
                    n: 128,
                    k: 96,
                    tile: [64, 64],
                    bk: 32,
                },
                8,
            ),
        ];
        for (case, split) in cases {
            let sched = Schedule {
                matrix: Matrix::Mma,
                split_k: split,
                ..Schedule::default()
            };
            let (shape, notes) = run_case(dev, target, case, &sched).unwrap();
            if target.has_mma_f16 {
                assert_eq!(shape, "mma", "{notes:?}");
                assert!(notes.is_empty(), "{notes:?}");
            }
        }
    }

    fn check_scheduled(dev: &dyn GpuDevice, target: &Target) {
        let mma = Schedule {
            matrix: Matrix::Mma,
            ..Schedule::default()
        };
        let staged = Schedule {
            matrix: Matrix::Scalar,
            ..Schedule::default()
        };
        let scalar_loads = Schedule {
            vector_loads: false,
            ..mma
        };
        let double = Schedule { stages: 2, ..mma };
        for case in SCHEDULED {
            for sched in [Schedule::o0(), staged, mma, scalar_loads, double] {
                let (shape, notes) = run_case(dev, target, case, &sched).unwrap();
                if sched == mma && case.input == DType::F16 && target.has_mma_f16 {
                    assert_eq!(shape, "mma", "{}x{}x{}", case.m, case.n, case.k);
                    assert!(notes.is_empty(), "{notes:?}");
                }
                if sched == mma && case.input == DType::I8 {
                    let want = if target.has_mma_i8 { "mma" } else { "naive" };
                    assert_eq!(shape, want, "{notes:?}");
                }
            }
        }
    }

    /// Main pass plus reduce pass, same bindings. Other element types run unsplit.
    /// 16-byte copies for interior blocks of aligned rows. Other shapes keep the scalar copy.
    /// The occupancy hint reaches the ZSL header and HIP launch bounds. Vulkan stages f16 natively
    /// and uses 16-byte loads like HIP.
    #[test]
    fn occupancy_hint_and_native_vulkan_staging() {
        let kernel = gemm(DType::F16, DType::F16, 256, 256, 256, [128, 128], 32, true).unwrap();
        let sched = Schedule {
            matrix: Matrix::Mma,
            waves_per_eu: Some((1, 2)),
            ..Schedule::default()
        };
        let hip = compile_with(&kernel, &sched, &Target::hip_gfx12()).unwrap();
        assert!(hip.zsl().contains("@waves_per_eu(1, 2)"), "{}", hip.zsl());
        assert!(
            hip.hip()
                .unwrap()
                .text
                .contains("amdgpu_waves_per_eu(1, 2)")
        );

        let vulkan = Target {
            backend: BackendPreference::Vulkan,
            ..Target::hip_gfx12()
        };
        let plain = Schedule {
            matrix: Matrix::Mma,
            ..Schedule::default()
        };
        let vk = compile_with(&kernel, &plain, &vulkan).unwrap();
        assert_eq!(vk.schedule_name(), "mma", "{:?}", vk.notes());
        assert!(vk.zsl().contains("shared sa: array<f16, "), "{}", vk.zsl());
        assert!(
            vk.zsl().contains("b_a_v: device buffer<u32x4>"),
            "{}",
            vk.zsl()
        );
        assert!(!vk.zsl().contains("@waves_per_eu"));
    }

    /// Metal: full k blocks run without a masked path, rows are padded at bk 32, and the
    /// epilogue reuses A's staging memory instead of its own shared array.
    #[test]
    fn metal_mma_plan_has_no_branch_or_extra_shared() {
        let kernel = gemm(DType::F16, DType::F16, 200, 136, 72, [64, 64], 32, true).unwrap();
        let metal = Target {
            backend: BackendPreference::Metal,
            lds_bytes: 32768,
            ..Target::hip_gfx12()
        };
        let sched = Schedule {
            matrix: Matrix::Mma,
            ..Schedule::default()
        };
        let c = compile_with(&kernel, &sched, &metal).unwrap();
        assert_eq!(c.schedule_name(), "mma", "{:?}", c.notes());
        let zsl = c.zsl();
        assert!(zsl.contains("view sc: array<f32> = sa"), "{zsl}");
        assert!(!zsl.contains("shared sc"), "{zsl}");
        assert!(zsl.contains("shared sa: array<f16, 2560>"), "{zsl}");
        assert!(zsl.contains("min("), "{zsl}");
        // 72 = 2 full blocks of 32 plus a masked tail.
        assert!(zsl.contains("for kt in 0..2 {"), "{zsl}");
        assert!(c.msl().is_ok());
    }

    /// The k loop runs past the operands' columns: those reads are zero on every path.
    #[test]
    fn k_loop_past_the_tensor_on_metal() {
        let Some(dev) = metal_device() else { return };
        short_operands(&*dev, (70, 70), 50, 40, 64);
    }

    /// The output has more rows than A: A's missing rows read as zero.
    #[test]
    fn output_past_the_operand_rows_on_metal() {
        let Some(dev) = metal_device() else { return };
        short_operands(&*dev, (64, 128), 50, 64, 64);
    }

    /// `C[out_m, n] = A[m, kc] * B[n, kc]^T` over a k loop of `extent`, checked on the host.
    fn short_operands(
        dev: &dyn GpuDevice,
        (m, out_m): (usize, usize),
        n: usize,
        kc: usize,
        extent: usize,
    ) {
        let target = caps_target(dev, dev.backend());
        let mut b = TileKernel::builder("short", [64, 64]);
        let a = b.input("a", DType::F16, [m, kc]);
        let w = b.input("b", DType::F16, [n, kc]);
        let c = b.output("c", DType::F32, [out_m, n]);
        let mut acc = b.zeros(DType::F32, [64, 64]);
        b.loop_k(extent, 16, |kb| {
            let at = kb.load(Block::new(a, Affine::grid_m(), Affine::k(), [64, 16]));
            let bt = kb.load(Block::new(w, Affine::grid_n(), Affine::k(), [64, 16]));
            acc = kb.dot(at, bt, acc);
        });
        b.store(
            Block::new(c, Affine::grid_m(), Affine::grid_n(), [64, 64]),
            acc,
        );
        let kernel = b.build().unwrap();

        let av: Vec<f32> = (0..m * kc).map(|i| ((i * 7) % 9) as f32 - 4.0).collect();
        let bv: Vec<f32> = (0..n * kc).map(|i| ((i * 5) % 7) as f32 - 3.0).collect();
        let f16_buf = |v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect();
            let h = dev
                .create_buffer(BufferDesc {
                    size: bytes.len() as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                    memory: MemoryUsage::GpuOnly,
                })
                .unwrap();
            dev.write_buffer(h, 0, &bytes).unwrap();
            h
        };
        let (ha, hb) = (f16_buf(&av), f16_buf(&bv));
        let hc = dev
            .create_buffer(BufferDesc {
                size: (out_m * n * 4) as u64,
                usage: BufferUsage::STORAGE | BufferUsage::READBACK,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        let sched = Schedule {
            matrix: Matrix::Mma,
            ..Schedule::default()
        };
        let compiled = compile_with(&kernel, &sched, &target).unwrap();
        if target.has_mma_f16 {
            assert_eq!(compiled.schedule_name(), "mma", "{:?}", compiled.notes());
        }
        let program = compiled.load(dev).unwrap();
        program.dispatch(dev, &[ha, hb, hc]).unwrap();
        program.destroy(dev);
        let got: Vec<f32> = dev
            .read_buffer(hc, 0, (out_m * n * 4) as u64)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        for i in 0..out_m {
            for j in 0..n {
                let want: f32 = if i < m {
                    (0..kc).map(|k| av[i * kc + k] * bv[j * kc + k]).sum()
                } else {
                    0.0
                };
                assert_eq!(got[i * n + j], want, "({i}, {j})");
            }
        }
        for h in [ha, hb, hc] {
            dev.destroy_buffer(h);
        }
    }

    /// Exact binary16 bits for small integers, quarters and zero.
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

    /// Rounds to the nearest binary16, ties to even. Normal values only.
    fn round_f16(x: f32) -> f32 {
        if x == 0.0 {
            return 0.0;
        }
        let exp = ((x.to_bits() >> 23) & 0xff) as i32 - 127;
        let ulp = 2f32.powi(exp - 10);
        (x / ulp).round_ties_even() * ulp
    }

    fn run_case(
        dev: &dyn GpuDevice,
        target: &Target,
        case: Case,
        schedule: &Schedule,
    ) -> Result<(String, Vec<String>), String> {
        let Case {
            input,
            out,
            m,
            n,
            k,
            tile,
            bk,
        } = case;
        let kernel =
            gemm(input, out, m, n, k, tile, bk, input == DType::F16).map_err(|e| e.to_string())?;
        let compiled = compile_with(&kernel, schedule, target).map_err(|e| e.to_string())?;
        let shape = (
            compiled.schedule_name().to_string(),
            compiled.notes().to_vec(),
        );
        let program = compiled.load(dev).map_err(|e| e.to_string())?;

        let int8 = input == DType::I8;
        let (span, lo) = match input {
            DType::I8 => (17, 8),
            DType::F16 => (33, 16),
            _ => (9, 4),
        };
        let a_ref: Vec<i32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % span) as i32 - lo)
            .collect();
        let b_ref: Vec<i32> = (0..n * k)
            .map(|i| ((i * 5 + 1) % span) as i32 - lo)
            .collect();
        let bias_ref: Vec<i32> = (0..n).map(|j| ((j * 3) % 5) as i32 - 2).collect();

        let mut uploads = Vec::new();
        for binding in compiled.bindings() {
            let bytes: Vec<u8> = match binding.name.as_str() {
                "a" | "b" => {
                    let src = if binding.name == "a" { &a_ref } else { &b_ref };
                    src.iter()
                        .flat_map(|&v| match input {
                            DType::I8 => vec![(v as i8) as u8],
                            DType::F16 => f16_bits(v as f32).to_le_bytes().to_vec(),
                            _ => (v as f32 / 4.0).to_le_bytes().to_vec(),
                        })
                        .collect()
                }
                "bias" if int8 => bias_ref.iter().flat_map(|v| v.to_le_bytes()).collect(),
                "bias" => bias_ref
                    .iter()
                    .flat_map(|&v| {
                        let scale = if input == DType::F16 { 1.0 } else { 0.25 };
                        (v as f32 * scale).to_le_bytes()
                    })
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
        if compiled.workspace_bytes() > 0 {
            let h = dev
                .create_buffer(BufferDesc {
                    size: compiled.workspace_bytes(),
                    usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                    memory: MemoryUsage::GpuOnly,
                })
                .map_err(|e| e.to_string())?;
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
                let dot: i64 = (0..k)
                    .map(|kk| i64::from(a_ref[r * k + kk]) * i64::from(b_ref[c * k + kk]))
                    .sum();
                let at = r * n + c;
                match input {
                    DType::I8 => {
                        let want = dot as i32 + bias_ref[c];
                        let g = i32::from_le_bytes(got[at * 4..at * 4 + 4].try_into().unwrap());
                        if g != want {
                            bad.push(format!("({r},{c}) {g} != {want}"));
                        }
                    }
                    DType::F16 => {
                        let mid = round_f16(dot as f32);
                        let want = round_f16(mid + bias_ref[c] as f32);
                        let bits = u16::from_le_bytes(got[at * 2..at * 2 + 2].try_into().unwrap());
                        let g = f16_value(bits);
                        if g != want {
                            bad.push(format!("({r},{c}) {g} != {want}"));
                        }
                    }
                    _ => {
                        let want = dot as f32 / 16.0 + bias_ref[c] as f32 / 4.0;
                        let g = f32::from_le_bytes(got[at * 4..at * 4 + 4].try_into().unwrap());
                        if (g - want).abs() > 1e-4 * want.abs().max(1.0) {
                            bad.push(format!("({r},{c}) {g} != {want}"));
                        }
                    }
                }
            }
        }
        if bad.is_empty() {
            Ok(shape)
        } else {
            Err(format!(
                "{:?}->{:?} {m}x{n}x{k} {}: {} mismatches, first {}",
                input,
                out,
                schedule_label(schedule),
                bad.len(),
                bad[0]
            ))
        }
    }

    fn schedule_label(s: &Schedule) -> String {
        format!("{:?} stage={}", s.matrix, s.stage_lds)
    }

    fn dtype_bytes(dtype: DType) -> usize {
        match dtype {
            DType::F16 => 2,
            DType::I8 => 1,
            _ => 4,
        }
    }

    /// Warm-up before timing: GPU clocks ramp up from idle over a few hundred ms.
    const WARMUP: Duration = Duration::from_millis(300);
    /// Each rep is one batch of dispatches, with one sync.
    const REPS: usize = 10;
    const BATCH: Duration = Duration::from_millis(20);

    /// Milliseconds per dispatch.
    #[derive(Clone, Copy)]
    struct Timing {
        median: f64,
        min: f64,
        max: f64,
    }

    fn timing(mut ms: Vec<f64>) -> Timing {
        ms.sort_by(f64::total_cmp);
        let n = ms.len();
        let median = if n % 2 == 1 {
            ms[n / 2]
        } else {
            (ms[n / 2 - 1] + ms[n / 2]) / 2.0
        };
        Timing {
            median,
            min: ms[0],
            max: ms[n - 1],
        }
    }

    /// Two single dispatches size the batch. Then `WARMUP`, then `REPS` reps of `k` dispatches, one
    /// sync each.
    fn time_program(
        dev: &dyn GpuDevice,
        program: &TileProgram,
        handles: &[BufferHandle],
    ) -> Timing {
        program.dispatch_many(dev, handles, 1).unwrap();
        let t0 = Instant::now();
        program.dispatch_many(dev, handles, 1).unwrap();
        let one = t0.elapsed().as_secs_f64().max(1e-6);
        let k = ((BATCH.as_secs_f64() / one).round() as usize).max(1);
        let warm = Instant::now();
        while warm.elapsed() < WARMUP {
            program.dispatch_many(dev, handles, k).unwrap();
        }
        let ms = (0..REPS)
            .map(|_| {
                let t = Instant::now();
                program.dispatch_many(dev, handles, k).unwrap();
                t.elapsed().as_secs_f64() * 1e3 / k as f64
            })
            .collect();
        timing(ms)
    }

    /// Naive timing (`ZENGPU_TILE_BENCH_NAIVE=1`): one warm-up, three synced dispatches, wall
    /// clock mean.
    fn time_naive(dev: &dyn GpuDevice, program: &TileProgram, handles: &[BufferHandle]) -> f64 {
        program.dispatch_many(dev, handles, 1).unwrap();
        let t0 = Instant::now();
        for _ in 0..3 {
            program.dispatch_many(dev, handles, 1).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e3 / 3.0
    }

    /// Printed once per run. A process on the GPU (e.g. `zetna serve`) shows in the PID list.
    fn print_environment() {
        println!("environment (rocm-smi):");
        let Ok(out) = Command::new("rocm-smi")
            .args(["--showclocks", "--showpower", "--showtemp", "--showpids"])
            .output()
        else {
            println!("  rocm-smi unavailable");
            return;
        };
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let keep = line.contains("sclk")
                || line.contains("junction")
                || line.contains("Power")
                || line.trim_start().starts_with(|c: char| c.is_ascii_digit());
            if keep {
                println!("  {}", line.trim());
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MmaCfg {
        tile: [usize; 2],
        bk: usize,
        waves: u32,
        rows: u32,
        split: u32,
        /// Stages per operand; 0 is the schedule default.
        stages: u32,
        /// `@waves_per_eu(1, 2)`: trade occupancy for fewer spills.
        wpe: bool,
    }

    impl MmaCfg {
        fn label(&self) -> String {
            let mut s = format!(
                "mma {}x{} bk{} w{} r{}",
                self.tile[0], self.tile[1], self.bk, self.waves, self.rows
            );
            if self.split > 1 {
                s += &format!(" k{}", self.split);
            }
            if self.stages > 1 {
                s += &format!(" s{}", self.stages);
            }
            if self.wpe {
                s += " wpe";
            }
            s
        }

        fn schedule(&self) -> Schedule {
            Schedule {
                matrix: Matrix::Mma,
                waves: self.waves,
                wave_rows: self.rows,
                stages: self.stages,
                split_k: self.split,
                waves_per_eu: self.wpe.then_some((1, 2)),
                ..Schedule::default()
            }
        }

        fn kernel(&self, (m, n, k): (usize, usize, usize)) -> TileKernel {
            gemm(DType::F16, DType::F16, m, n, k, self.tile, self.bk, true).unwrap()
        }
    }

    /// One timed config. Nonzero scratch bytes means the kernel spills.
    struct Measured {
        label: String,
        ms: f64,
        scratch: u32,
        cfg: Option<MmaCfg>,
        compiled: CompiledTile,
    }

    /// Splits of waves into rows, as `(waves, wave rows)`, tried. 0 rows is auto. Eight waves as 4
    /// rows of 2x4 fragments beat auto on 128x128.
    const WAVE_SPLITS: [(u32, u32); 3] = [(4, 0), (8, 0), (8, 4)];

    fn hip_grid() -> Vec<MmaCfg> {
        let mut grid = Vec::new();
        for tile in [[64, 64], [128, 64], [128, 128]] {
            for bk in [16, 32] {
                for (waves, rows) in WAVE_SPLITS {
                    grid.push(MmaCfg {
                        tile,
                        bk,
                        waves,
                        rows,
                        split: 1,
                        stages: 0,
                        wpe: false,
                    });
                }
            }
        }
        grid
    }

    /// Vulkan grid: same tiles at k step 16, two wave splits. Small to keep the bench short.
    fn vulkan_grid() -> Vec<MmaCfg> {
        let mut grid = Vec::new();
        for tile in [[64, 64], [128, 64], [128, 128]] {
            for (waves, rows) in [(4, 0), (8, 4)] {
                grid.push(MmaCfg {
                    tile,
                    bk: 16,
                    waves,
                    rows,
                    split: 1,
                    stages: 0,
                    wpe: false,
                });
            }
        }
        grid
    }

    /// Metal grid: simdgroup matrices, k steps 16 and 32, the same wave splits as HIP.
    fn metal_grid() -> Vec<MmaCfg> {
        let mut grid = Vec::new();
        for tile in [[32, 32], [64, 32], [64, 64], [128, 64], [128, 128]] {
            for bk in [16, 32] {
                for (waves, rows) in [(1, 0), (2, 0), (4, 0), (4, 2), (8, 0)] {
                    for stages in [1, 2] {
                        grid.push(MmaCfg {
                            tile,
                            bk,
                            waves,
                            rows,
                            split: 1,
                            stages,
                            wpe: false,
                        });
                    }
                }
            }
        }
        grid
    }

    fn shape_buffers(dev: &dyn GpuDevice, (m, n, k): (usize, usize, usize)) -> Vec<BufferHandle> {
        let f16_bytes = |len: usize| -> Vec<u8> {
            (0..len)
                .flat_map(|i| f16_bits(((i * 7 + 3) % 9) as f32 - 4.0).to_le_bytes())
                .collect()
        };
        let inputs = [
            f16_bytes(m * k),
            f16_bytes(n * k),
            vec![0u8; n * 4],
            vec![0u8; m * n * 2],
        ];
        inputs
            .iter()
            .map(|bytes| {
                let h = dev
                    .create_buffer(BufferDesc {
                        size: bytes.len() as u64,
                        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                        memory: MemoryUsage::GpuOnly,
                    })
                    .unwrap();
                dev.write_buffer(h, 0, bytes).unwrap();
                h
            })
            .collect()
    }

    /// `None` when the schedule does not resolve to `want` (the row says why).
    #[allow(clippy::too_many_arguments)]
    fn measure(
        dev: &dyn GpuDevice,
        target: &Target,
        backend: &str,
        shape: (usize, usize, usize),
        label: &str,
        kernel: &TileKernel,
        sched: &Schedule,
        handles: &[BufferHandle],
        naive: bool,
    ) -> Option<Measured> {
        let (m, n, k) = shape;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let compiled = compile_with(kernel, sched, target).unwrap();
        if compiled.schedule_name() != "mma" && sched.matrix == Matrix::Mma {
            println!(
                "| {m} | {n} | {k} | {backend} {label} | skipped: {:?} |",
                compiled.notes()
            );
            return None;
        }
        let program = compiled.load(dev).unwrap();
        let stats = program.stats();
        let (vgprs, scratch) = stats.map_or((0, 0), |s| (s.vgprs, s.scratch_bytes));
        let (run_handles, ws) = with_workspace(dev, &compiled, handles);
        let (ms, min, max) = if naive {
            let ms = time_naive(dev, &program, &run_handles);
            (ms, ms, ms)
        } else {
            let t = time_program(dev, &program, &run_handles);
            (t.median, t.min, t.max)
        };
        if let Some(ws) = ws {
            dev.destroy_buffer(ws);
        }
        let spill = if scratch > 0 { " SPILL" } else { "" };
        println!(
            "| {m} | {n} | {k} | {backend} {label} ({}) | {ms:.3} | {min:.3} | {max:.3} | {:.2} | {vgprs} / {scratch}{spill} |",
            compiled.schedule_name(),
            flops / (ms * 1e-3) / 1e12
        );
        program.destroy(dev);
        Some(Measured {
            label: label.to_string(),
            ms,
            scratch,
            cfg: None,
            compiled,
        })
    }

    /// Fastest measured config. Ties go to less scratch. A faster spilling config is kept.
    fn pick_best(rows: &[Measured]) -> Option<&Measured> {
        rows.iter()
            .min_by(|a, b| a.ms.total_cmp(&b.ms).then(a.scratch.cmp(&b.scratch)))
    }

    /// Times `grid`, then tries split-K variants of the four fastest unsplit configs whose tiles
    /// leave the GPU short.
    #[allow(clippy::too_many_arguments)]
    fn search(
        dev: &dyn GpuDevice,
        target: &Target,
        backend: &str,
        shape: (usize, usize, usize),
        grid: &[MmaCfg],
        splits: &[u32],
        handles: &[BufferHandle],
        naive: bool,
    ) -> Vec<Measured> {
        let (m, n, _) = shape;
        let mut rows = Vec::new();
        for cfg in grid {
            let label = cfg.label();
            let kernel = cfg.kernel(shape);
            if let Some(mut row) = measure(
                dev,
                target,
                backend,
                shape,
                &label,
                &kernel,
                &cfg.schedule(),
                handles,
                naive,
            ) {
                row.cfg = Some(*cfg);
                rows.push(row);
            }
        }
        let spilled: Vec<MmaCfg> = rows
            .iter()
            .filter(|r| r.scratch > 0)
            .filter_map(|r| r.cfg)
            .filter(|c| !c.wpe && c.tile[0] * c.tile[1] >= 128 * 64)
            .collect();
        for base in spilled {
            let cfg = MmaCfg { wpe: true, ..base };
            let kernel = cfg.kernel(shape);
            if let Some(mut row) = measure(
                dev,
                target,
                backend,
                shape,
                &cfg.label(),
                &kernel,
                &cfg.schedule(),
                handles,
                naive,
            ) {
                row.cfg = Some(cfg);
                rows.push(row);
            }
        }
        if splits.is_empty() {
            return rows;
        }
        let mut short: Vec<(MmaCfg, f64)> = rows
            .iter()
            .filter_map(|r| r.cfg.map(|c| (c, r.ms)))
            .filter(|(c, _)| m.div_ceil(c.tile[0]) * n.div_ceil(c.tile[1]) < 128)
            .collect();
        short.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (base, _) in short.into_iter().take(4) {
            for &sk in splits {
                let cfg = MmaCfg { split: sk, ..base };
                let kernel = cfg.kernel(shape);
                if let Some(mut row) = measure(
                    dev,
                    target,
                    backend,
                    shape,
                    &cfg.label(),
                    &kernel,
                    &cfg.schedule(),
                    handles,
                    naive,
                ) {
                    row.cfg = Some(cfg);
                    rows.push(row);
                }
            }
        }
        rows
    }

    /// Benchmarks f16 GEMM schedules per shape and prints the results.
    ///
    /// Run with `cargo test --release -p zengpu-tile bench_f16_gemm -- --ignored --nocapture`. Each
    /// shape runs HIP O0 (to 2048), scalar-staged and an mma search. Vulkan gets a smaller mma
    /// search. `SPILL` marks spilling configs. The fastest measured config wins. The best HIP
    /// config is re-run 3x for spread.
    #[test]
    #[ignore]
    fn bench_f16_gemm() {
        let hip = hip_device();
        let vulkan = vulkan_device();
        let metal = metal_device();
        if hip.is_none() && vulkan.is_none() && metal.is_none() {
            return;
        }
        let naive = env::var_os("ZENGPU_TILE_BENCH_NAIVE").is_some();
        print_environment();
        println!(
            "method: {}",
            if naive {
                "naive (step-3b: 1 warm-up, 3 single dispatches, wall clock)".to_string()
            } else {
                format!(
                    "warm-up {} ms, {REPS} reps of ~{} ms batches, median/min/max per dispatch",
                    WARMUP.as_millis(),
                    BATCH.as_millis()
                )
            }
        );
        let shapes = [
            (128, 5120, 5120),
            (128, 17408, 5120),
            (512, 5120, 5120),
            (512, 17408, 5120),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];
        let only: Option<(usize, usize, usize)> =
            env::var("ZENGPU_TILE_BENCH_ONLY").ok().map(|v| {
                let d: Vec<usize> = v.split('x').map(|x| x.parse().unwrap()).collect();
                (d[0], d[1], d[2])
            });
        let mut summary: Vec<String> = Vec::new();
        for shape in shapes {
            if only.is_some_and(|o| o != shape) {
                continue;
            }
            let (m, n, k) = shape;
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            println!(
                "\n| M | N | K | schedule | ms median | min | max | TFLOPS | VGPR / scratch |"
            );
            println!("|---|---|---|---|---|---|---|---|---|");

            if let Some(dev) = &hip {
                let dev: &dyn GpuDevice = &**dev;
                let target = caps_target(dev, BackendPreference::Hip);
                let handles = shape_buffers(dev, shape);
                let base = gemm(DType::F16, DType::F16, m, n, k, [64, 64], 16, true).unwrap();
                if m.max(n).max(k) <= 2048 {
                    measure(
                        dev,
                        &target,
                        "hip",
                        shape,
                        "O0",
                        &base,
                        &Schedule::o0(),
                        &handles,
                        naive,
                    );
                }
                let scalar = Schedule {
                    matrix: Matrix::Scalar,
                    ..Schedule::default()
                };
                measure(
                    dev,
                    &target,
                    "hip",
                    shape,
                    "scalar-tiled 64x64x16",
                    &base,
                    &scalar,
                    &handles,
                    naive,
                );
                let rows = search(
                    dev,
                    &target,
                    "hip",
                    shape,
                    &hip_grid(),
                    &[2, 4, 8],
                    &handles,
                    naive,
                );
                if let Some(best) = pick_best(&rows) {
                    let cfg = best.cfg.unwrap();
                    println!("best search: {} at {:.3} ms", best.label, best.ms);
                    let kernel = cfg.kernel(shape);
                    let scalar_loads = Schedule {
                        vector_loads: false,
                        ..cfg.schedule()
                    };
                    measure(
                        dev,
                        &target,
                        "hip",
                        shape,
                        &format!("{} scalar loads", best.label),
                        &kernel,
                        &scalar_loads,
                        &handles,
                        naive,
                    );
                    if env::var_os("ZENGPU_TILE_BENCH_ZSL").is_some() {
                        println!("{}", best.compiled.zsl());
                    }
                    let compiled = compile_with(&kernel, &cfg.schedule(), &target).unwrap();
                    let program = compiled.load(dev).unwrap();
                    let (run_handles, ws) = with_workspace(dev, &compiled, &handles);
                    let mut medians = Vec::new();
                    for rep in 0..3 {
                        let t = time_program(dev, &program, &run_handles);
                        println!(
                            "rerun {rep}: {} median {:.3} ms min {:.3} max {:.3} TFLOPS {:.2}",
                            best.label,
                            t.median,
                            t.min,
                            t.max,
                            flops / (t.median * 1e-3) / 1e12
                        );
                        medians.push(t.median);
                    }
                    if let Some(ws) = ws {
                        dev.destroy_buffer(ws);
                    }
                    program.destroy(dev);
                    let med = timing(medians);
                    summary.push(format!(
                        "| {m} x {n} x {k} | hip {} | {:.3} | {:.3}..{:.3} | {:.2} |",
                        best.label,
                        med.median,
                        med.min,
                        med.max,
                        flops / (med.median * 1e-3) / 1e12
                    ));
                }
                for h in handles {
                    dev.destroy_buffer(h);
                }
            }

            if let Some(dev) = &vulkan {
                let dev: &dyn GpuDevice = &**dev;
                let target = caps_target(dev, BackendPreference::Vulkan);
                let handles = shape_buffers(dev, shape);
                let rows = search(
                    dev,
                    &target,
                    "vulkan",
                    shape,
                    &vulkan_grid(),
                    &[],
                    &handles,
                    naive,
                );
                if let Some(best) = pick_best(&rows) {
                    println!("best vulkan search: {} at {:.3} ms", best.label, best.ms);
                    summary.push(format!(
                        "| {m} x {n} x {k} | vulkan {} | {:.3} | single run | {:.2} |",
                        best.label,
                        best.ms,
                        flops / (best.ms * 1e-3) / 1e12
                    ));
                }
                for h in handles {
                    dev.destroy_buffer(h);
                }
            }
            if let Some(dev) = &metal {
                let dev: &dyn GpuDevice = &**dev;
                let target = caps_target(dev, BackendPreference::Metal);
                let handles = shape_buffers(dev, shape);
                let base = gemm(DType::F16, DType::F16, m, n, k, [64, 64], 16, true).unwrap();
                let scalar = Schedule {
                    matrix: Matrix::Scalar,
                    ..Schedule::default()
                };
                measure(
                    dev,
                    &target,
                    "metal",
                    shape,
                    "scalar-tiled 64x64x16",
                    &base,
                    &scalar,
                    &handles,
                    naive,
                );
                let rows = search(
                    dev,
                    &target,
                    "metal",
                    shape,
                    &metal_grid(),
                    &[2, 4],
                    &handles,
                    naive,
                );
                if let Some(best) = pick_best(&rows) {
                    println!("best metal search: {} at {:.3} ms", best.label, best.ms);
                    if env::var_os("ZENGPU_TILE_BENCH_ZSL").is_some() {
                        println!("{}", best.compiled.zsl());
                    }
                    summary.push(format!(
                        "| {m} x {n} x {k} | metal {} | {:.3} | single run | {:.2} |",
                        best.label,
                        best.ms,
                        flops / (best.ms * 1e-3) / 1e12
                    ));
                }
                for h in handles {
                    dev.destroy_buffer(h);
                }
            }
        }
        println!("\n| M x N x K | best config | median ms | rerun range ms / note | TFLOPS |");
        println!("|---|---|---|---|---|");
        for line in summary {
            println!("{line}");
        }
    }

    /// Handles to run with: bindings, then the workspace when split-K. The caller destroys
    /// the workspace.
    fn with_workspace(
        dev: &dyn GpuDevice,
        compiled: &CompiledTile,
        handles: &[BufferHandle],
    ) -> (Vec<BufferHandle>, Option<BufferHandle>) {
        let mut all = handles.to_vec();
        if compiled.workspace_bytes() == 0 {
            return (all, None);
        }
        let ws = dev
            .create_buffer(BufferDesc {
                size: compiled.workspace_bytes(),
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::GpuOnly,
            })
            .unwrap();
        all.push(ws);
        (all, Some(ws))
    }
}
