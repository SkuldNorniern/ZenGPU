//! Structured, enum-based errors with no external error crates; backend
//! detail is preserved but raw backend types never leak into the public API.

use crate::types::{Features, MemoryUsage};

/// The result type used throughout ZenGPU.
pub type Result<T> = core::result::Result<T, GpuError>;

/// Top-level error for every ZenGPU operation.
#[derive(Debug)]
pub enum GpuError {
    /// The device was lost (driver reset, removal, hang).
    DeviceLost,
    /// An allocation of the given residency class failed.
    OutOfMemory(MemoryUsage),
    /// One or more required features are unavailable on this adapter.
    UnsupportedFeatures(Features),
    /// A resource was used in a way its creation did not permit (see
    /// [`UsageError`]). This is the class the validation layer raises.
    InvalidUsage(UsageError),
    /// A shader module failed to compile/load.
    ShaderCompile(String),
    /// A pipeline failed to create.
    PipelineCreation(String),
    /// A dispatch or draw was invalid.
    Dispatch(String),
    /// A surface/swapchain error (see [`SurfaceError`]).
    Surface(SurfaceError),
    /// An opaque, backend-tagged error. Carries a message, never a raw handle.
    Backend(String),
}

/// Invalid-usage detail. Most of these are caught by the validation layer
/// before they reach the driver.
#[derive(Debug)]
pub enum UsageError {
    /// A handle whose generation no longer matches its slot (use-after-free).
    StaleHandle {
        index: u32,
        expected_gen: u32,
        actual_gen: u32,
    },
    /// A resource bound without the usage flag the operation requires.
    MissingUsage {
        resource: &'static str,
        needed: &'static str,
    },
    /// A binding that does not match the pipeline's expectation.
    BindingMismatch(String),
}

/// Surface acquisition / presentation errors.
#[derive(Debug)]
pub enum SurfaceError {
    /// The surface is no longer usable; recreate it.
    Lost,
    /// The swapchain is out of date (e.g. resized); reconfigure.
    Outdated,
    /// Acquiring the next image timed out.
    Timeout,
    /// Out of memory while acquiring/presenting.
    OutOfMemory,
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::DeviceLost => write!(f, "device lost"),
            GpuError::OutOfMemory(usage) => write!(f, "out of memory ({usage:?})"),
            GpuError::UnsupportedFeatures(features) => {
                write!(f, "unsupported features: {features:?}")
            }
            GpuError::InvalidUsage(e) => write!(f, "invalid usage: {e}"),
            GpuError::ShaderCompile(msg) => write!(f, "shader compile error: {msg}"),
            GpuError::PipelineCreation(msg) => write!(f, "pipeline creation error: {msg}"),
            GpuError::Dispatch(msg) => write!(f, "dispatch error: {msg}"),
            GpuError::Surface(e) => write!(f, "surface error: {e}"),
            GpuError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl core::fmt::Display for UsageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UsageError::StaleHandle {
                index,
                expected_gen,
                actual_gen,
            } => write!(
                f,
                "stale handle (index {index}, gen {expected_gen} != slot gen {actual_gen}) \
                 — use after free"
            ),
            UsageError::MissingUsage { resource, needed } => {
                write!(f, "{resource} was created without {needed} usage")
            }
            UsageError::BindingMismatch(msg) => write!(f, "binding mismatch: {msg}"),
        }
    }
}

impl core::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            SurfaceError::Lost => "surface lost",
            SurfaceError::Outdated => "surface out of date",
            SurfaceError::Timeout => "surface acquire timed out",
            SurfaceError::OutOfMemory => "surface out of memory",
        };
        f.write_str(s)
    }
}

impl std::error::Error for GpuError {}

impl From<SurfaceError> for GpuError {
    fn from(e: SurfaceError) -> Self {
        GpuError::Surface(e)
    }
}
impl From<UsageError> for GpuError {
    fn from(e: UsageError) -> Self {
        GpuError::InvalidUsage(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_handle_message_is_actionable() {
        let e = GpuError::from(UsageError::StaleHandle {
            index: 12,
            expected_gen: 1,
            actual_gen: 2,
        });
        let msg = e.to_string();
        assert!(msg.contains("use after free"), "got: {msg}");
        assert!(msg.contains("12"));
    }

    #[test]
    fn surface_error_converts() {
        let e: GpuError = SurfaceError::Outdated.into();
        assert!(matches!(e, GpuError::Surface(SurfaceError::Outdated)));
    }
}
