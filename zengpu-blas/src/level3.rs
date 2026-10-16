//! BLAS Level 3: matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    BackendPreference, Bindings, ComputePipelineDesc, ComputePipelineHints, DType, DispatchOp,
    GpuDevice, GpuError, PipelineHandle, Result, Scalar, ShaderDesc, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::gemm_hip::{self, Plan, SKINNY_ROWS};
use crate::gemm_skinny::{
    self, SKINNY_1_ZSL, SKINNY_4_ZSL, SKINNY_8_ZSL, SKINNY_16_ZSL, SPLIT_REDUCE_ZSL,
};
use crate::gemm_tiled::{GEMM_TILED_ZSL, TILE as TILED_GEMM_TILE};

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
}

struct HipGemm {
    t64: PipelineHandle,
    skinny: [PipelineHandle; SKINNY_ROWS.len()],
    reduce: PipelineHandle,
    compute_units: u32,
}

struct ZslSkinny {
    skinny: [PipelineHandle; 4],
    shaders: [ShaderHandle; 4],
    reduce: PipelineHandle,
    reduce_shader: ShaderHandle,
}

impl GemmKernel {
    /// Compile the GEMM pipeline. HIP: hand-written. Vulkan, Metal: register-blocked ZSL. Others:
    /// naive ZSL. Vulkan `sgemm` uses the skinny kernels for `m <= 16`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        if matches!(
            device.backend(),
            BackendPreference::Vulkan | BackendPreference::Metal
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

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`, row-major f32. `C` is allocated from `pool`.
    pub fn sgemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
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

        let c = pool.alloc(vec![m, n], DType::F32)?;
        if let Some(hip) = &self.hip {
            return match self.sgemm_hip(hip, device, pool, alpha, a, b, &c, [m, n, k]) {
                Ok(()) => Ok(c),
                Err(e) => {
                    pool.free(c);
                    Err(e)
                }
            };
        }
        if let Some(zsl) = self.skinny.as_ref().filter(|_| m <= 16) {
            return match self.sgemm_skinny(zsl, device, pool, alpha, a, b, &c, [m, n, k]) {
                Ok(()) => Ok(c),
                Err(e) => {
                    pool.free(c);
                    Err(e)
                }
            };
        }
        device.dispatch(
            self.pipeline,
            Bindings {
                buffers: &[a.buffer.index(), b.buffer.index(), c.buffer.index()],
                scalars: &[
                    Scalar::U32(m),
                    Scalar::U32(n),
                    Scalar::U32(k),
                    Scalar::F32(alpha),
                ],
                textures: &[],
                images: &[],
            },
            [n.div_ceil(self.tile), m.div_ceil(self.tile), 1],
        )?;
        Ok(c)
    }

    /// Runs the [`gemm_hip::plan`] kernel for this shape, then the split-K reduction if K was
    /// split, as one batch.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_hip(
        &self,
        hip: &HipGemm,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k]: [u32; 3],
    ) -> Result<()> {
        let mn = m.checked_mul(n).ok_or_else(|| {
            GpuError::Dispatch(format!("sgemm: {m}×{n} output exceeds 2^32 elements"))
        })?;
        let plan = gemm_hip::plan(m, n, k, hip.compute_units);
        let split = plan.split();
        let workspace = if split > 1 {
            Some(pool.alloc(vec![split, m, n], DType::F32)?)
        } else {
            None
        };
        let target = workspace.as_ref().unwrap_or(c);
        let (pipeline, kparam) = match plan {
            Plan::Skinny { rows, kc, .. } => {
                let slot = SKINNY_ROWS.iter().position(|&r| r == rows).unwrap_or(0);
                (hip.skinny[slot], kc)
            }
            Plan::Tile { tile, kchunk, .. } => {
                (if tile == 128 { self.pipeline } else { hip.t64 }, kchunk)
            }
        };
        let gemm_buffers = [a.buffer.index(), b.buffer.index(), target.buffer.index()];
        let gemm_scalars = [
            Scalar::U32(m),
            Scalar::U32(n),
            Scalar::U32(k),
            Scalar::U32(kparam),
            Scalar::F32(alpha),
        ];
        let reduce_buffers = [target.buffer.index(), c.buffer.index()];
        let reduce_scalars = [Scalar::U32(mn), Scalar::U32(split)];
        let mut ops = vec![DispatchOp {
            pipeline,
            bindings: Bindings {
                buffers: &gemm_buffers,
                textures: &[],
                images: &[],
                scalars: &gemm_scalars,
            },
            grid: plan.grid(m, n),
        }];
        if workspace.is_some() {
            ops.push(DispatchOp {
                pipeline: hip.reduce,
                bindings: Bindings {
                    buffers: &reduce_buffers,
                    textures: &[],
                    images: &[],
                    scalars: &reduce_scalars,
                },
                grid: [mn.div_ceil(256), 1, 1],
            });
        }
        let result = device.dispatch_batch(&ops);
        if let Some(workspace) = workspace {
            pool.free(workspace);
        }
        result
    }

    /// Runs the Vulkan skinny kernel for the smallest row class holding `m` rows, then the split-K
    /// reduction, as one batch. The skinny kernel always writes partials, so the reduction runs.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_skinny(
        &self,
        zsl: &ZslSkinny,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
        c: &DeviceArray,
        [m, n, k]: [u32; 3],
    ) -> Result<()> {
        let mn = m.checked_mul(n).ok_or_else(|| {
            GpuError::Dispatch(format!("sgemm: {m}×{n} output exceeds 2^32 elements"))
        })?;
        let split = k.max(1).div_ceil(gemm_skinny::KC);
        let slot = gemm_skinny::ROWS
            .iter()
            .position(|&r| r >= m)
            .unwrap_or(gemm_skinny::ROWS.len() - 1);
        let workspace = pool.alloc(vec![split, m, n], DType::F32)?;
        let gemm_buffers = [a.buffer.index(), b.buffer.index(), workspace.buffer.index()];
        let gemm_scalars = [
            Scalar::U32(m),
            Scalar::U32(n),
            Scalar::U32(k),
            Scalar::U32(gemm_skinny::KC),
            Scalar::F32(alpha),
        ];
        let reduce_buffers = [workspace.buffer.index(), c.buffer.index()];
        let reduce_scalars = [Scalar::U32(mn), Scalar::U32(split)];
        let ops = [
            DispatchOp {
                pipeline: zsl.skinny[slot],
                bindings: Bindings {
                    buffers: &gemm_buffers,
                    textures: &[],
                    images: &[],
                    scalars: &gemm_scalars,
                },
                grid: [n.div_ceil(256), split, 1],
            },
            DispatchOp {
                pipeline: zsl.reduce,
                bindings: Bindings {
                    buffers: &reduce_buffers,
                    textures: &[],
                    images: &[],
                    scalars: &reduce_scalars,
                },
                grid: [mn.div_ceil(256), 1, 1],
            },
        ];
        let result = device.dispatch_batch(&ops);
        pool.free(workspace);
        result
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
