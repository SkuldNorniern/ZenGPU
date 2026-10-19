//! BLAS Level 3: matrix-matrix operations.

use std::array::from_fn;

use zengpu_compute::{BufferPool, DeviceArray, OpList};
use zengpu_hal::{
    BackendPreference, ComputePipelineDesc, ComputePipelineHints, DType, GpuDevice, GpuError,
    PipelineHandle, Result, Scalar, ShaderDesc, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::gemm_cuda;
use crate::gemm_hip::{self, B_TYPES, Plan, SKINNY_ROWS};
use crate::gemm_skinny::{
    self, SKINNY_1_ZSL, SKINNY_4_ZSL, SKINNY_8_ZSL, SKINNY_16_ZSL, SPLIT_REDUCE_ZSL,
};
use crate::gemm_tiled::{GEMM_TILED_ZSL, TILE as TILED_GEMM_TILE};
use crate::gemm_wmma;

/// Computes `C[m,n] = alpha * sum_k A[m,k] * B[k,n]`.
///
/// Naive (untiled) f32 GEMM with a 16×16 workgroup.
const GEMM_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, alpha: f32 }
    @workgroup_size(16, 16)
    kernel cs_gemm(a: device buffer<f32>, b: device buffer<f32>, c: device mut buffer<f32>, p: P, id: global_id) {
        let row = id.y
        let col = id.x
        if row < p.m && col < p.n {
            let sum: f32 = 0.0
            for i in 0..p.k {
                sum = sum + a[row * p.k + i] * b[i * p.n + col]
            }
            c[row * p.n + col] = p.alpha * sum
        }
    }
);

/// Compiled GEMM pipeline. Create once per device, reuse across dispatches.
pub struct GemmKernel {
    shader: ShaderHandle,
    /// The general-purpose pipeline (on HIP, the 128×128-tile kernel).
    pub pipeline: PipelineHandle,
    /// Output tile per workgroup: 16 for the naive kernel, 128 for the register-blocked ones.
    tile: u32,
    /// The shape-specific kernels, on HIP.
    hip: Option<HipGemm>,
    /// The few-row kernels, on Vulkan.
    skinny: Option<ZslSkinny>,
    /// The tensor-core kernels for half-precision inputs, on CUDA.
    cuda: Option<CudaGemm>,
}

struct CudaGemm {
    shader: ShaderHandle,
    /// Per entry of `gemm_cuda::VARIANTS`, f16 then bf16.
    pipelines: [[PipelineHandle; 2]; gemm_cuda::VARIANTS.len()],
    /// The f32 SIMT kernel (`sgemm_simt128`).
    simt: PipelineHandle,
    reduce: PipelineHandle,
}

struct HipGemm {
    /// Per B element type, in `gemm_hip::B_TYPES` order (f32 first; the f32 128 tile is
    /// also `GemmKernel::pipeline`).
    t128: [PipelineHandle; B_TYPES.len()],
    t64: [PipelineHandle; B_TYPES.len()],
    skinny: [[PipelineHandle; SKINNY_ROWS.len()]; B_TYPES.len()],
    reduce: PipelineHandle,
    /// Half-to-f32 conversion of A, f16 then bf16.
    widen: [PipelineHandle; 2],
    /// Matrix-core kernels, when the device has them.
    wmma: Option<HipWmma>,
    compute_units: u32,
}

/// The RDNA4 WMMA kernels ([`gemm_wmma`]): per entry of [`gemm_wmma::VARIANTS`], f16 then bf16.
struct HipWmma {
    shader: ShaderHandle,
    pipelines: [[PipelineHandle; 2]; gemm_wmma::VARIANTS.len()],
}

struct ZslSkinny {
    skinny: [PipelineHandle; 4],
    shaders: [ShaderHandle; 4],
    reduce: PipelineHandle,
    reduce_shader: ShaderHandle,
}

impl GemmKernel {
    /// Compile the GEMM pipeline. HIP: hand-written kernel. Vulkan, Metal, CUDA: register-blocked
    /// ZSL. Others: naive ZSL. Vulkan `sgemm` uses the skinny kernels for `m <= 16`. CUDA: half
    /// inputs use tensor cores, f32 inputs with more than 16 rows the SIMT kernel.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        if matches!(
            device.backend(),
            BackendPreference::Vulkan | BackendPreference::Metal | BackendPreference::Cuda
        ) {
            let (shader, entry) = GEMM_TILED_ZSL.create(device)?;
            let pipeline = device.create_compute_pipeline_with(
                ComputePipelineDesc {
                    shader,
                    entry,
                    block: [16, 16, 1],
                },
                ComputePipelineHints {
                    subgroup_size: Some(64),
                },
            );
            let pipeline = match pipeline {
                Ok(pipeline) => pipeline,
                Err(e) => {
                    device.destroy_shader(shader);
                    return Err(e);
                }
            };
            let skinny = if matches!(
                device.backend(),
                BackendPreference::Vulkan | BackendPreference::Cuda
            ) {
                match Self::new_zsl_skinny(device) {
                    Ok(skinny) => Some(skinny),
                    Err(e) => {
                        device.destroy_pipeline(pipeline);
                        device.destroy_shader(shader);
                        return Err(e);
                    }
                }
            } else {
                None
            };
            let cuda = if device.backend() == BackendPreference::Cuda {
                Self::new_cuda(device)
            } else {
                None
            };
            return Ok(Self {
                shader,
                pipeline,
                tile: TILED_GEMM_TILE,
                hip: None,
                skinny,
                cuda,
            });
        }
        if device.backend() == BackendPreference::Hip {
            return Self::new_hip(device);
        }
        let (shader, entry) = GEMM_ZSL.create(device)?;
        let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
            shader,
            entry,
            block: [16, 16, 1],
        })?;
        Ok(Self {
            shader,
            pipeline,
            tile: 16,
            hip: None,
            skinny: None,
            cuda: None,
        })
    }

    /// CUDA tensor-core kernels, or `None` below sm_80 (the module compiles empty).
    fn new_cuda(device: &dyn GpuDevice) -> Option<CudaGemm> {
        let shader = match device.create_shader(ShaderDesc::cuda_src(gemm_cuda::SOURCE)) {
            Ok(shader) => shader,
            Err(e) => {
                log::warn!("zengpu-blas: CUDA tensor-core GEMM failed to build: {e}");
                return None;
            }
        };
        let mut made = Vec::new();
        let mut built = || -> Result<CudaGemm> {
            for variant in &gemm_cuda::VARIANTS {
                for suffix in ["_f16", "_bf16"] {
                    made.push(device.create_compute_pipeline(ComputePipelineDesc {
                        shader,
                        entry: &format!("{}{suffix}", variant.name),
                        block: [variant.threads, 1, 1],
                    })?);
                }
            }
            let simt = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm_simt128",
                block: [256, 1, 1],
            })?;
            let reduce = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry: "sgemm_tc_reduce",
                block: [256, 1, 1],
            })?;
            Ok(CudaGemm {
                shader,
                pipelines: from_fn(|v| [made[2 * v], made[2 * v + 1]]),
                simt,
                reduce,
            })
        };
        match built() {
            Ok(cuda) => Some(cuda),
            Err(e) => {
                log::warn!("zengpu-blas: CUDA tensor-core GEMM failed to build: {e}");
                for p in made {
                    device.destroy_pipeline(p);
                }
                device.destroy_shader(shader);
                None
            }
        }
    }

    /// Builds the Vulkan skinny kernels and the split-K reduction.
    ///
    /// # Errors
    ///
    /// Everything created here is destroyed before the error is returned.
    fn new_zsl_skinny(device: &dyn GpuDevice) -> Result<ZslSkinny> {
        let mut shaders: Vec<ShaderHandle> = Vec::new();
        let mut pipelines: Vec<PipelineHandle> = Vec::new();
        let built = (|| {
            let mut build = |source: &ZslShader| -> Result<(ShaderHandle, PipelineHandle)> {
                let (shader, entry) = source.create(device)?;
                shaders.push(shader);
                let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
                    shader,
                    entry,
                    block: [256, 1, 1],
                })?;
                pipelines.push(pipeline);
                Ok((shader, pipeline))
            };
            let (s1, p1) = build(&SKINNY_1_ZSL)?;
            let (s4, p4) = build(&SKINNY_4_ZSL)?;
            let (s8, p8) = build(&SKINNY_8_ZSL)?;
            let (s16, p16) = build(&SKINNY_16_ZSL)?;
            let (reduce_shader, reduce) = build(&SPLIT_REDUCE_ZSL)?;
            Ok(ZslSkinny {
                skinny: [p1, p4, p8, p16],
                shaders: [s1, s4, s8, s16],
                reduce,
                reduce_shader,
            })
        })();
        match built {
            Ok(skinny) => Ok(skinny),
            Err(e) => {
                for p in pipelines {
                    device.destroy_pipeline(p);
                }
                for s in shaders {
                    device.destroy_shader(s);
                }
                Err(e)
            }
        }
    }

    fn new_hip(device: &dyn GpuDevice) -> Result<Self> {
        let shader = device.create_shader(ShaderDesc::hip(gemm_hip::SOURCE))?;
        let mut made: Vec<PipelineHandle> = Vec::new();
        let mut pipeline = |entry: &str, block: u32| -> Result<PipelineHandle> {
            let p = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [block, 1, 1],
            })?;
            made.push(p);
            Ok(p)
        };
        let built = (|| {
            let reduce = pipeline("sgemm_reduce", 256)?;
            let mut t128 = [reduce; B_TYPES.len()];
            let mut t64 = [reduce; B_TYPES.len()];
            let mut skinny = [[reduce; SKINNY_ROWS.len()]; B_TYPES.len()];
            for (i, (_, suffix)) in B_TYPES.iter().enumerate() {
                t128[i] = pipeline(&format!("sgemm_t128{suffix}"), 256)?;
                t64[i] = pipeline(&format!("sgemm_t64{suffix}"), 256)?;
                for (slot, rows) in skinny[i].iter_mut().zip(SKINNY_ROWS) {
                    *slot = pipeline(
                        &format!("sgemm_skinny_{rows}{suffix}"),
                        gemm_hip::SKINNY_THREADS,
                    )?;
                }
            }
            let widen = [pipeline("widen_f16", 256)?, pipeline("widen_bf16", 256)?];
            Ok((t128, t64, reduce, skinny, widen))
        })();
        match built {
            Ok((t128, t64, reduce, skinny, widen)) => Ok(Self {
                shader,
                pipeline: t128[0],
                tile: 128,
                hip: Some(HipGemm {
                    t128,
                    t64,
                    skinny,
                    reduce,
                    widen,
                    wmma: Self::new_wmma(device),
                    compute_units: device.limits().compute_units,
                }),
                skinny: None,
                cuda: None,
            }),
            Err(e) => {
                for p in made {
                    device.destroy_pipeline(p);
                }
                device.destroy_shader(shader);
                Err(e)
            }
        }
    }

    /// WMMA kernels, or `None` off gfx12 (the module compiles empty there).
    fn new_wmma(device: &dyn GpuDevice) -> Option<HipWmma> {
        let shader = match device.create_shader(ShaderDesc::hip(gemm_wmma::SOURCE)) {
            Ok(shader) => shader,
            Err(e) => {
                log::warn!("zengpu-blas: WMMA GEMM failed to build: {e}");
                return None;
            }
        };
        let mut made = Vec::new();
        let mut built = || -> Result<HipWmma> {
            for variant in &gemm_wmma::VARIANTS {
                for suffix in ["_f16", "_bf16"] {
                    made.push(device.create_compute_pipeline(ComputePipelineDesc {
                        shader,
                        entry: &format!("{}{suffix}", variant.name),
                        block: [variant.threads, 1, 1],
                    })?);
                }
            }
            Ok(HipWmma {
                shader,
                pipelines: from_fn(|v| [made[2 * v], made[2 * v + 1]]),
            })
        };
        match built() {
            Ok(wmma) => Some(wmma),
            Err(_) => {
                for p in made {
                    device.destroy_pipeline(p);
                }
                device.destroy_shader(shader);
                None
            }
        }
    }

    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        if let Some(hip) = self.hip {
            for p in hip.t128.into_iter().skip(1).chain(hip.t64) {
                device.destroy_pipeline(p);
            }
            device.destroy_pipeline(hip.reduce);
            for p in hip.skinny.into_iter().flatten().chain(hip.widen) {
                device.destroy_pipeline(p);
            }
            if let Some(wmma) = hip.wmma {
                for p in wmma.pipelines.into_iter().flatten() {
                    device.destroy_pipeline(p);
                }
                device.destroy_shader(wmma.shader);
            }
        }
        if let Some(zsl) = self.skinny {
            for p in zsl.skinny {
                device.destroy_pipeline(p);
            }
            device.destroy_pipeline(zsl.reduce);
            for s in zsl.shaders {
                device.destroy_shader(s);
            }
            device.destroy_shader(zsl.reduce_shader);
        }
        if let Some(cuda) = self.cuda {
            for p in cuda.pipelines.into_iter().flatten().chain([cuda.simt]) {
                device.destroy_pipeline(p);
            }
            device.destroy_pipeline(cuda.reduce);
            device.destroy_shader(cuda.shader);
        }
        device.destroy_shader(self.shader);
    }

    /// Computes `C[m,n] = alpha * A[m,k] @ B[k,n]` in row-major f32.
    ///
    /// `C` is allocated from `pool`. This waits for the GPU. Use [`Self::record_sgemm`] to chain
    /// ops in one submission.
    pub fn sgemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        let mut ops = OpList::new(pool);
        let c = self.record_sgemm(&mut ops, alpha, a, b)?;
        match ops.run(device) {
            Ok(()) => Ok(c),
            Err(e) => {
                pool.free(c);
                Err(e)
            }
        }
    }

    /// Records [`Self::sgemm`] into `ops`.
    ///
    /// `C` is allocated when recorded and written when the list runs. The list keeps any
    /// split-K workspace.
    pub fn record_sgemm(
        &self,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        ops.transaction(|ops| self.record_sgemm_inner(ops, alpha, a, b))
    }

    fn record_sgemm_inner(
        &self,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        let b_type = B_TYPES.iter().position(|&(dtype, _)| dtype == b.dtype);
        let a_type = B_TYPES.iter().position(|&(dtype, _)| dtype == a.dtype);
        let types_ok = match (a_type, b_type) {
            (Some(0), Some(0)) => true,
            (Some(_), Some(_)) => self.hip.is_some() || (self.cuda.is_some() && a_type == b_type),
            _ => false,
        };
        if !types_ok {
            return Err(GpuError::Dispatch(format!(
                "sgemm: A and B must be f32 (or both f16 or both bf16 on HIP and CUDA), got {:?} and {:?}",
                a.dtype, b.dtype
            )));
        }
        let a_type = a_type.unwrap_or(0);
        let b_type = b_type.unwrap_or(0);
        let (&[m, k], &[k2, n]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(GpuError::Dispatch(format!(
                "sgemm: expected 2D arrays, got shapes {:?} and {:?}",
                a.shape, b.shape
            )));
        };
        if k != k2 {
            return Err(GpuError::Dispatch(format!(
                "sgemm: inner dimensions mismatch ({k} vs {k2})"
            )));
        }
        let mn = m.checked_mul(n).ok_or_else(|| {
            GpuError::Dispatch(format!("sgemm: {m}×{n} output exceeds 2^32 elements"))
        })?;
        if self.hip.is_none() && a_type != 0 && !gemm_cuda::fits(n, k) {
            return Err(GpuError::Dispatch(format!(
                "sgemm: f16/bf16 on CUDA needs N and K to be multiples of 8, got n={n}, k={k}"
            )));
        }

        ops.validate_array(a)?;
        ops.validate_array(b)?;
        let c = ops.pool().alloc(vec![m, n], DType::F32)?;
        ops.require_buffer(c.buffer, c.size_bytes());
        let recorded = if let Some(hip) = &self.hip {
            Self::record_hip(hip, a_type, b_type, ops, alpha, a, b, &c, [m, n, k, mn])
        } else if let Some(cuda) = self.cuda.as_ref().filter(|_| a_type != 0) {
            Self::record_cuda(cuda, a_type, ops, alpha, a, b, &c, [m, n, k, mn])
        } else if let Some(cuda) = self
            .cuda
            .as_ref()
            .filter(|_| m > gemm_cuda::SKINNY_ROWS && gemm_cuda::simt_fits(n, k))
        {
            Self::record_cuda_simt(cuda, ops, alpha, a, b, &c, [m, n, k])
        } else if let Some(zsl) = self.skinny.as_ref().filter(|_| m <= 16) {
            record_skinny(zsl, ops, alpha, a, b, &c, [m, n, k, mn])
        } else {
            ops.dispatch(
                self.pipeline,
                &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                &[
                    Scalar::U32(m),
                    Scalar::U32(n),
                    Scalar::U32(k),
                    Scalar::F32(alpha),
                ],
                [n.div_ceil(self.tile), m.div_ceil(self.tile), 1],
            );
            Ok(())
        };
        match recorded {
            Ok(()) => Ok(c),
            Err(e) => {
                ops.pool().free(c);
                Err(e)
            }
        }
    }

    /// Records the CUDA tensor-core kernel for half A and B (`half`: 1 for f16, 2 for bf16), then
    /// the split-K reduction if K was split.
    #[allow(clippy::too_many_arguments)]
    fn record_cuda(
        cuda: &CudaGemm,
        half: usize,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k, mn]: [u32; 4],
    ) -> Result<()> {
        let variant = gemm_cuda::pick(m, n);
        let (splits, kchunk) = if variant == gemm_cuda::SKINNY {
            gemm_cuda::split(n, k)
        } else {
            (1, k)
        };
        let target = if splits > 1 {
            ops.scratch(vec![splits, m, n], DType::F32)?
        } else {
            c.buffer.index()
        };
        ops.dispatch(
            cuda.pipelines[variant][half - 1],
            &[a.buffer.index(), b.buffer.index(), target],
            &[
                Scalar::U32(m),
                Scalar::U32(n),
                Scalar::U32(k),
                Scalar::U32(kchunk),
                Scalar::F32(alpha),
            ],
            gemm_cuda::grid(variant, m, n, splits),
        );
        if splits > 1 {
            ops.dispatch(
                cuda.reduce,
                &[target, c.buffer.index()],
                &[Scalar::U32(mn), Scalar::U32(splits)],
                [mn.div_ceil(256), 1, 1],
            );
        }
        Ok(())
    }

    /// Records the f32 SIMT kernel: 128×128 tiles, no split-K.
    fn record_cuda_simt(
        cuda: &CudaGemm,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k]: [u32; 3],
    ) -> Result<()> {
        ops.dispatch(
            cuda.simt,
            &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
            &[
                Scalar::U32(m),
                Scalar::U32(n),
                Scalar::U32(k),
                Scalar::F32(alpha),
            ],
            [n.div_ceil(128), m.div_ceil(128), 1],
        );
        Ok(())
    }

    /// Records the HIP kernel for this shape. WMMA runs for matching half A and B on matrix-core
    /// devices. Otherwise the [`gemm_hip::plan`] kernel runs, with A widened to f32 if it is half.
    /// The split-K reduction follows if K was split.
    #[allow(clippy::too_many_arguments)]
    fn record_hip(
        hip: &HipGemm,
        a_type: usize,
        b_type: usize,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k, mn]: [u32; 4],
    ) -> Result<()> {
        if let Some(wmma) = &hip.wmma
            && a_type != 0
            && a_type == b_type
            && gemm_wmma::fits(n, k)
        {
            let variant = gemm_wmma::pick(m);
            ops.dispatch(
                wmma.pipelines[variant][a_type - 1],
                &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                &[
                    Scalar::U32(m),
                    Scalar::U32(n),
                    Scalar::U32(k),
                    Scalar::U32(gemm_wmma::GROUP),
                    Scalar::F32(alpha),
                ],
                gemm_wmma::grid(&gemm_wmma::VARIANTS[variant], m, n),
            );
            return Ok(());
        }
        let a_slot = if a_type == 0 {
            a.buffer.index()
        } else {
            let mk = m * k;
            let wide = ops.scratch(vec![m, k], DType::F32)?;
            ops.dispatch(
                hip.widen[a_type - 1],
                &[a.buffer.index(), wide],
                &[Scalar::U32(mk)],
                [mk.div_ceil(256), 1, 1],
            );
            wide
        };
        let plan = gemm_hip::plan(m, n, k, hip.compute_units);
        let split = plan.split();
        let target = if split > 1 {
            ops.scratch(vec![split, m, n], DType::F32)?
        } else {
            c.buffer.index()
        };
        let (pipeline, kparam) = match plan {
            Plan::Skinny { rows, kc, .. } => {
                let slot = SKINNY_ROWS.iter().position(|&r| r == rows).unwrap_or(0);
                (hip.skinny[b_type][slot], kc)
            }
            Plan::Tile { tile, kchunk, .. } => (
                if tile == 128 {
                    hip.t128[b_type]
                } else {
                    hip.t64[b_type]
                },
                kchunk,
            ),
        };
        ops.dispatch(
            pipeline,
            &[a_slot, b.buffer.index(), target],
            &[
                Scalar::U32(m),
                Scalar::U32(n),
                Scalar::U32(k),
                Scalar::U32(kparam),
                Scalar::F32(alpha),
            ],
            plan.grid(m, n),
        );
        if split > 1 {
            ops.dispatch(
                hip.reduce,
                &[target, c.buffer.index()],
                &[Scalar::U32(mn), Scalar::U32(split)],
                [mn.div_ceil(256), 1, 1],
            );
        }
        Ok(())
    }

    /// Computes `C = A @ B`. This is [`Self::sgemm`] with `alpha = 1.0`.
    pub fn gemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        self.sgemm(device, pool, 1.0, a, b)
    }
}

/// Records the Vulkan skinny kernel for the smallest row class that holds `m` rows, then the
/// split-K reduction.
///
/// The skinny kernel always writes partials, so the reduction always runs.
#[allow(clippy::too_many_arguments)]
fn record_skinny(
    zsl: &ZslSkinny,
    ops: &mut OpList<'_>,
    alpha: f32,
    a: &DeviceArray,
    b: &DeviceArray,
    c: &DeviceArray,
    [m, n, k, mn]: [u32; 4],
) -> Result<()> {
    let split = k.max(1).div_ceil(gemm_skinny::KC);
    let slot = gemm_skinny::ROWS
        .iter()
        .position(|&r| r >= m)
        .unwrap_or(gemm_skinny::ROWS.len() - 1);
    let workspace = ops.scratch(vec![split, m, n], DType::F32)?;
    ops.dispatch(
        zsl.skinny[slot],
        &[a.buffer.index(), b.buffer.index(), workspace],
        &[
            Scalar::U32(m),
            Scalar::U32(n),
            Scalar::U32(k),
            Scalar::U32(gemm_skinny::KC),
            Scalar::F32(alpha),
        ],
        [n.div_ceil(256), split, 1],
    );
    ops.dispatch(
        zsl.reduce,
        &[workspace, c.buffer.index()],
        &[Scalar::U32(mn), Scalar::U32(split)],
        [mn.div_ceil(256), 1, 1],
    );
    Ok(())
}
