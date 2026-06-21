use crate::api::CUresult;
use zengpu_hal::{GpuError, MemoryUsage, Result};

/// Checks the status code returned by a Driver API call.
#[allow(dead_code)]
pub(crate) fn cu_check(result: CUresult, op: &'static str) -> Result<()> {
    if result == 0 {
        return Ok(());
    }
    Err(match result {
        2 => GpuError::OutOfMemory(MemoryUsage::GpuOnly),
        3 | 4 | 100 => GpuError::Backend(format!("cuda: {op}: driver not ready (CUresult={result})")),
        101 => GpuError::Backend(format!("cuda: {op}: invalid device ordinal")),
        201 | 202 | 710 | 719 => GpuError::DeviceLost,
        _ => GpuError::Backend(format!("cuda: {op} failed (CUresult={result})")),
    })
}
