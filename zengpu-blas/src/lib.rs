//! ZenGPU BLAS — a GPU-accelerated BLAS extension for ZenGPU.
//!
//! A drop-in BLAS library that dispatches through any ZenGPU-compatible
//! backend (Vulkan, CUDA, Metal, CPU). Long-term: routes to vendor
//! libraries (cuBLAS, rocBLAS, MPS) when the matching backend is detected,
//! falling back to portable compute shaders otherwise.
//!
//! # Usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use zengpu_blas::Blas;
//! use zengpu_compute::BufferPool;
//! use zengpu_hal::{DType, GpuDevice};
//!
//! # fn run(device: Arc<dyn GpuDevice>) -> zengpu_hal::Result<()> {
//! let blas = Blas::new(&*device)?;
//! let pool = BufferPool::new(device.clone());
//!
//! let a = pool.alloc(vec![4, 4], DType::F32)?;
//! let b = pool.alloc(vec![4, 4], DType::F32)?;
//!
//! // C = 2.0 * A @ B
//! let c = blas.sgemm(&*device, &pool, 2.0, &a, &b)?;
//! # Ok(()) }
//! ```
//!
//! # Crate features
//!
//! This crate is an optional extension — add `zengpu-blas` directly to your
//! `Cargo.toml`. It depends only on `zengpu-hal` and `zengpu-compute`, not
//! on the root `zengpu` crate, so it works with any HAL-compatible backend.

pub mod level1;
pub mod level3;

pub use level1::Level1Kernels;
pub use level3::GemmKernel;

use zengpu_compute::{BufferPool, DeviceArray};
use zengpu_hal::{GpuDevice, Result};

// ── Blas aggregate ────────────────────────────────────────────────────────────

/// All BLAS pipelines compiled for a single device.
///
/// Create once per device with [`Blas::new`] and reuse across calls. This
/// compiles the Level-1 (saxpy, sscal) and Level-3 (sgemm) pipelines in one
/// shot. For fine-grained control, build [`Level1Kernels`] and [`GemmKernel`]
/// separately.
pub struct Blas {
    pub level1: Level1Kernels,
    pub gemm:   GemmKernel,
}

impl Blas {
    /// Compile all BLAS pipelines on `device`.
    pub fn new(device: &dyn GpuDevice) -> Result<Self> {
        Ok(Self {
            level1: Level1Kernels::new(device)?,
            gemm:   GemmKernel::new(device)?,
        })
    }

    /// Destroy all pipelines and shaders created by [`Self::new`].
    pub fn destroy(self, device: &dyn GpuDevice) {
        self.level1.destroy(device);
        self.gemm.destroy(device);
    }

    // ── Level 3 ───────────────────────────────────────────────────────────────

    /// `C[m,n] = alpha * A[m,k] @ B[k,n]`  (row-major f32 GEMM).
    pub fn sgemm(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        a: &DeviceArray,
        b: &DeviceArray,
    ) -> Result<DeviceArray> {
        self.gemm.sgemm(device, pool, alpha, a, b)
    }

    // ── Level 1 ───────────────────────────────────────────────────────────────

    /// `y[i] += alpha * x[i]`  (SAXPY — in-place, no allocation).
    pub fn saxpy(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        x: &DeviceArray,
        y: &DeviceArray,
    ) -> Result<()> {
        self.level1.saxpy(device, pool, alpha, x, y)
    }

    /// `x[i] *= alpha`  (SSCAL — in-place, no allocation).
    pub fn sscal(
        &self,
        device: &dyn GpuDevice,
        pool: &BufferPool,
        alpha: f32,
        x: &DeviceArray,
    ) -> Result<()> {
        self.level1.sscal(device, pool, alpha, x)
    }
}
