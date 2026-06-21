use cuda_oxide::error::ErrorCode;
use zengpu_hal::{GpuError, MemoryUsage, Result as HalResult};

/// Convert a `cuda-oxide` result into a ZenGPU HAL result.
#[allow(dead_code)]
pub(crate) fn from_cuda<T>(r: Result<T, ErrorCode>) -> HalResult<T> {
    r.map_err(|code| match code {
        ErrorCode::OutOfMemory => GpuError::OutOfMemory(MemoryUsage::GpuOnly),
        ErrorCode::ContextIsDestroyed | ErrorCode::Deinitialized => GpuError::DeviceLost,
        ErrorCode::NoDevice => GpuError::Backend("cuda: no device found".into()),
        ErrorCode::InvalidDevice => GpuError::Backend("cuda: invalid device ordinal".into()),
        ErrorCode::LaunchFailed
        | ErrorCode::LaunchOutOfResources
        | ErrorCode::LaunchTimeout
        | ErrorCode::CooperativeLaunchTooLarge => {
            GpuError::Dispatch(format!("cuda: kernel launch error ({code:?})"))
        }
        ErrorCode::InvalidPtx
        | ErrorCode::NoBinaryForGpu
        | ErrorCode::UnsupportedPtxVersion
        | ErrorCode::InvalidSource => {
            GpuError::ShaderCompile(format!("cuda: PTX/module error ({code:?})"))
        }
        other => GpuError::Backend(format!("cuda: {other:?}")),
    })
}
