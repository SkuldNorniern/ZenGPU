//! BLAS Level 3 — matrix-matrix operations.

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{
    BackendPreference, Bindings, ComputePipelineDesc, DType, DispatchOp, GpuDevice, GpuError,
    PipelineHandle, Result, Scalar, ShaderDesc, ShaderHandle,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::gemm_hip::{self, Plan, SKINNY_ROWS};
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
}

/// Every HIP GEMM entry point, from one module.
struct HipGemm {
    t64: PipelineHandle,
    skinny: [PipelineHandle; SKINNY_ROWS.len()],
    reduce: PipelineHandle,
    compute_units: u32,
}

impl GemmKernel {
    /// Compile the GEMM pipeline on `device`: the hand-written kernel on
    /// HIP, the register-blocked ZSL kernel on Vulkan and Metal, and the
    /// naive ZSL kernel on backends the blocked one has not been measured on.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        if matches!(
            device.backend(),
            BackendPreference::Vulkan | BackendPreference::Metal
        ) {
            let (shader, entry) = GEMM_TILED_ZSL.create(device)?;
            let pipeline = device.create_compute_pipeline(ComputePipelineDesc {
                shader,
                entry,
                block: [16, 16, 1],
            });
            return match pipeline {
                Ok(pipeline) => Ok(Self {
                    shader,
                    pipeline,
                    tile: TILED_GEMM_TILE,
                    hip: None,
                }),
                Err(e) => {
                    device.destroy_shader(shader);
                    Err(e)
                }
            };
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
        })
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
        device.destroy_shader(self.shader);
    }

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`  (row-major, f32).
    ///
    /// `a.shape = [m, k]`, `b.shape = [k, n]`. `C` is allocated from `pool`
    /// and returned as a new `[m, n]` `DeviceArray`. `alpha = 1.0` gives
    /// the simple `C = A @ B` form.
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

    /// Run the HIP kernel [`gemm_hip::plan`] picks for this shape, then the
    /// split-K reduction when K was split, as one batch.
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
