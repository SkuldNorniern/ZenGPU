//! Enum-based errors with no external crates. Raw backend types never leak into the public API.

use core::fmt::{Display, Formatter, Result as FmtResult};
use core::result::Result as CoreResult;
use std::error::Error;

use crate::types::{Features, MemoryUsage};

pub type Result<T> = CoreResult<T, GpuError>;

#[derive(Debug)]
pub enum GpuError {
    DeviceLost,
    OutOfMemory(MemoryUsage),
    UnsupportedFeatures(Features),
    Unsupported(String),
    /// Used without the creation usage it needs (see `UsageError`). The validation layer
    /// raises this.
    InvalidUsage(UsageError),
    ShaderCompile(String),
    PipelineCreation(String),
    Dispatch(String),
    Surface(SurfaceError),
    /// An opaque, backend-tagged error. Carries a message, never a raw handle.
    Backend(String),
}

/// Most of these are caught by the validation layer before they reach the driver.
#[derive(Debug)]
pub enum UsageError {
    /// A handle whose generation no longer matches its slot (use-after-free).
    StaleHandle {
        index: u32,
        expected_gen: u32,
        actual_gen: u32,
    },
    MissingUsage {
        resource: &'static str,
        needed: &'static str,
    },
    BindingMismatch(String),
}

#[derive(Debug)]
pub enum SurfaceError {
    /// The surface is no longer usable and must be recreated.
    Lost,
    /// The swapchain is out of date, e.g. after a resize, and must be reconfigured.
    Outdated,
    Timeout,
    OutOfMemory,
}

impl Display for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            GpuError::DeviceLost => write!(f, "device lost"),
            GpuError::OutOfMemory(usage) => write!(f, "out of memory ({usage:?})"),
            GpuError::UnsupportedFeatures(features) => {
                write!(f, "unsupported features: {features:?}")
            }
            GpuError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            GpuError::InvalidUsage(e) => write!(f, "invalid usage: {e}"),
            GpuError::ShaderCompile(msg) => write!(f, "shader compile error: {msg}"),
            GpuError::PipelineCreation(msg) => write!(f, "pipeline creation error: {msg}"),
            GpuError::Dispatch(msg) => write!(f, "dispatch error: {msg}"),
            GpuError::Surface(e) => write!(f, "surface error: {e}"),
            GpuError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl Display for UsageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            UsageError::StaleHandle {
                index,
                expected_gen,
                actual_gen,
            } => write!(
                f,
                "stale handle (index {index}, gen {expected_gen} != slot gen {actual_gen}) \
                : use after free"
            ),
            UsageError::MissingUsage { resource, needed } => {
                write!(f, "{resource} was created without {needed} usage")
            }
            UsageError::BindingMismatch(msg) => write!(f, "binding mismatch: {msg}"),
        }
    }
}

impl Display for SurfaceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let s = match self {
            SurfaceError::Lost => "surface lost",
            SurfaceError::Outdated => "surface out of date",
            SurfaceError::Timeout => "surface acquire timed out",
            SurfaceError::OutOfMemory => "surface out of memory",
        };
        f.write_str(s)
    }
}

impl Error for GpuError {}

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
