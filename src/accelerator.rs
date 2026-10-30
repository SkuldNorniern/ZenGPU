//! Reserved model-execution targets for future NPU/TPU work.
//!
//! These placeholders do not register graphics adapters, load vendor libraries,
//! or substitute a GPU/CPU runner. Tensor layouts, quantization, model loading,
//! graph compilation, asynchronous completion, and memory ownership need a
//! separate execution contract before these targets can run work.

use zengpu_hal::{GpuError, Result};

/// A planned accelerator target, not a detected or usable device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorTarget {
    /// Qualcomm NPU. Future integration: QAIRT / QNN HTP.
    /// See <https://docs.qualcomm.com/bundle/publicresource/topics/80-70020-15B/qairt.html>.
    QualcommNpu,
    /// Google Cloud TPU. Kept separate from local Edge TPU runtimes.
    /// See <https://docs.cloud.google.com/tpu/docs/intro-to-tpu>.
    GoogleCloudTpu,
    /// Local Google Edge TPU. Future runtime and model support are unimplemented.
    /// See <https://www.coral.withgoogle.com/docs/edgetpu/faq/>.
    GoogleEdgeTpu,
}

/// Targets reserved for later work. This is a roadmap, not an availability probe.
pub const PLANNED_TARGETS: [AcceleratorTarget; 3] = [
    AcceleratorTarget::QualcommNpu,
    AcceleratorTarget::GoogleCloudTpu,
    AcceleratorTarget::GoogleEdgeTpu,
];

impl AcceleratorTarget {
    /// Implementation status only. Does not inspect the host or load a driver.
    pub const fn is_implemented(self) -> bool {
        false
    }

    /// Refuses all planned targets until a real execution backend exists.
    pub fn require_available(self) -> Result<()> {
        Err(GpuError::Unsupported(format!(
            "{self:?}: accelerator execution is planned but not implemented"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_targets_never_claim_execution_or_silently_fall_back() {
        for target in PLANNED_TARGETS {
            assert!(!target.is_implemented());
            assert!(matches!(
                target.require_available(),
                Err(GpuError::Unsupported(_))
            ));
        }
    }
}
