use crate::api::CUresult;
use zengpu_hal::{GpuError, MemoryUsage, Result};

/// Map a `CUresult` to `GpuError`. Call this after every Driver API operation.
#[allow(dead_code)]
pub(crate) fn cu_check(result: CUresult, op: &'static str) -> Result<()> {
    if result == 0 {
        return Ok(());
    }
    Err(match result {
        2 => GpuError::OutOfMemory(MemoryUsage::GpuOnly),
        // CUDA_ERROR_NOT_INITIALIZED / CUDA_ERROR_DEINITIALIZED / CUDA_ERROR_NO_DEVICE
        3 | 4 | 100 => GpuError::Backend(format!("cuda: {op}: driver not ready (CUresult={result})")),
        // CUDA_ERROR_INVALID_DEVICE
        101 => GpuError::Backend(format!("cuda: {op}: invalid device ordinal")),
        // CUDA_ERROR_INVALID_CONTEXT / device lost classes
        201 | 202 | 710 | 719 => GpuError::DeviceLost,
        _ => GpuError::Backend(format!("cuda: {op} failed (CUresult={result})")),
    })
}
