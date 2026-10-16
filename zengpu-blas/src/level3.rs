//! BLAS Level 3 — matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray, OpList};
use zengpu_hal::{
    BackendPreference, ComputePipelineDesc, ComputePipelineHints, DType, GpuDevice, GpuError,
    PipelineHandle, Result, Scalar, ShaderDesc, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::gemm_hip::{self, Plan, SKINNY_ROWS};
use crate::gemm_skinny::{
    self, SKINNY_1_ZSL, SKINNY_4_ZSL, SKINNY_8_ZSL, SKINNY_16_ZSL, SPLIT_REDUCE_ZSL,
};
use crate::gemm_tiled::{GEMM_TILED_ZSL, TILE as TILED_GEMM_TILE};

/// `C[m,n] = alpha * sum_k A[m,k] * B[k,n]`
///
/// Naive (untiled) f32 GEMM. 16×16 local workgroup.
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
///
/// This implements the Level-3 BLAS `sgemm` operation.
pub struct GemmKernel {
    shader: ShaderHandle,
    /// The general-purpose pipeline (on HIP, the 128×128-tile kernel).
    pub pipeline: PipelineHandle,
    /// Output tile per workgroup: 16 for the naive kernel, 128 for the
    /// register-blocked ones.
    tile: u32,
    /// The shape-specific kernels, on HIP.
    hip: Option<HipGemm>,
    /// The few-row kernels, on Vulkan.
    skinny: Option<ZslSkinny>,
}

/// Every HIP GEMM entry point, from one module.
struct HipGemm {
    t64: PipelineHandle,
    skinny: [PipelineHandle; SKINNY_ROWS.len()],
    reduce: PipelineHandle,
    compute_units: u32,
}

/// The Vulkan few-row GEMM kernels and their split-K reduction.
struct ZslSkinny {
    /// One per entry of `gemm_skinny::ROWS`.
    skinny: [PipelineHandle; 4],
    shaders: [ShaderHandle; 4],
    reduce: PipelineHandle,
    reduce_shader: ShaderHandle,
}

impl GemmKernel {
    /// Compile the GEMM pipeline on `device`: the hand-written kernel on
    /// HIP, the register-blocked ZSL kernel on Vulkan and Metal, and the
    /// naive ZSL kernel on backends the blocked one has not been measured on.
    /// On Vulkan, `sgemm` also uses the skinny ZSL kernels when `m <= 16`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        if matches!(
            device.backend(),
            BackendPreference::Vulkan | BackendPreference::Metal
        ) {
            let (shader, entry) = GEMM_TILED_ZSL.create(device)?;
            // 64-lane subgroups where the device allows choosing: RADV runs
            // this kernel at wave32 by default, 7.3 -> 9.2 TFLOP/s at 4096³
            // on an RX 9060 XT with wave64.
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
            let skinny = if device.backend() == BackendPreference::Vulkan {
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
            return Ok(Self {
                shader,
                pipeline,
                tile: TILED_GEMM_TILE,
                hip: None,
                skinny,
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
        })
    }

    /// Build the Vulkan skinny kernels and the split-K reduction. On error,
    /// everything created here is destroyed before returning.
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
            let t128 = pipeline("sgemm_t128", 256)?;
            let t64 = pipeline("sgemm_t64", 256)?;
            let reduce = pipeline("sgemm_reduce", 256)?;
            let mut skinny = [t128; SKINNY_ROWS.len()];
            for (slot, rows) in skinny.iter_mut().zip(SKINNY_ROWS) {
                *slot = pipeline(&format!("sgemm_skinny_{rows}"), gemm_hip::SKINNY_THREADS)?;
            }
            Ok((t128, t64, reduce, skinny))
        })();
        match built {
            Ok((t128, t64, reduce, skinny)) => Ok(Self {
                shader,
                pipeline: t128,
                tile: 128,
                hip: Some(HipGemm {
                    t64,
                    skinny,
                    reduce,
                    compute_units: device.limits().compute_units,
                }),
                skinny: None,
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

    /// Destroy the pipelines and shader created by [`Self::new`].
    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        if let Some(hip) = self.hip {
            device.destroy_pipeline(hip.t64);
            device.destroy_pipeline(hip.reduce);
            for p in hip.skinny {
                device.destroy_pipeline(p);
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
        device.destroy_shader(self.shader);
    }

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`  (row-major, f32).
    ///
    /// `a.shape = [m, k]`, `b.shape = [k, n]`. `C` is allocated from `pool`
    /// and returned as a new `[m, n]` `DeviceArray`. `alpha = 1.0` gives
    /// the simple `C = A @ B` form. Waits for the GPU; use
    /// [`Self::record_sgemm`] to chain it with other ops in one submission.
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

    /// Record [`Self::sgemm`] into `ops`. `C` is allocated now and written
    /// when the list runs; any split-K workspace is kept by the list.
    pub fn record_sgemm(
        &self,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(GpuError::Dispatch(format!(
                "sgemm: only f32 is supported, got {:?} and {:?}",
                a.dtype, b.dtype
            )));
        }
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

        let c = ops.pool().alloc(vec![m, n], DType::F32)?;
        let recorded = if let Some(hip) = &self.hip {
            self.record_hip(hip, ops, alpha, a, b, &c, [m, n, k, mn])
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

    /// Record the HIP kernel [`gemm_hip::plan`] picks for this shape, then
    /// the split-K reduction when K was split.
    #[allow(clippy::too_many_arguments)]
    fn record_hip(
        &self,
        hip: &HipGemm,
        ops: &mut OpList<'_>,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k, mn]: [u32; 4],
    ) -> Result<()> {
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
                (hip.skinny[slot], kc)
            }
            Plan::Tile { tile, kchunk, .. } => {
                (if tile == 128 { self.pipeline } else { hip.t64 }, kchunk)
            }
        };
        ops.dispatch(
            pipeline,
            &[a.buffer.index(), b.buffer.index(), target],
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

    /// `C = A @ B` — shorthand for [`Self::sgemm`] with `alpha = 1.0`.
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

/// Record the Vulkan skinny kernel for the smallest row class holding `m`
/// rows, then the split-K reduction. The skinny kernel always writes partial
/// sums to the workspace, so the reduction always runs.
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
