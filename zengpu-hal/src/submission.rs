//! Backend-neutral GPU submission completion contract.

use std::time::Duration;

use crate::Result;

/// Non-blocking completion state for a submitted batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionStatus {
    Pending,
    Complete,
}

/// A submitted unit of GPU work.
///
/// The cycle identifier is supplied by the caller and is never interpreted by
/// the backend. Real-time consumers use it to reject a completion belonging to
/// an older control cycle. Backends must keep referenced native resources and
/// descriptor slots alive until completion. A destroy request made while work
/// is pending invalidates the public handle immediately but may defer native
/// destruction and slot reuse.
pub trait GpuSubmission: Send + Sync {
    /// Caller-defined control-cycle identifier associated with this work.
    fn cycle_id(&self) -> u64;

    /// Query completion without blocking.
    fn poll(&self) -> Result<SubmissionStatus>;

    /// Wait for completion for no longer than `timeout`.
    ///
    /// Returns [`GpuError::Timeout`] if the deadline expires. A timeout does
    /// not cancel the GPU work; the handle remains valid and may be polled or
    /// waited again.
    fn wait(&self, timeout: Duration) -> Result<()>;
}

/// Boxed, object-safe submission handle returned by [`crate::GpuDevice`].
pub type Submission = Box<dyn GpuSubmission>;

/// Completed handle used by synchronous and reference backends.
pub struct CompletedSubmission {
    cycle_id: u64,
}

impl CompletedSubmission {
    pub fn new(cycle_id: u64) -> Self {
        Self { cycle_id }
    }
}

impl GpuSubmission for CompletedSubmission {
    fn cycle_id(&self) -> u64 {
        self.cycle_id
    }

    fn poll(&self) -> Result<SubmissionStatus> {
        Ok(SubmissionStatus::Complete)
    }

    fn wait(&self, _timeout: Duration) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_submission_preserves_cycle_id() {
        let submission = CompletedSubmission::new(42);
        assert_eq!(submission.cycle_id(), 42);
        assert_eq!(submission.poll().unwrap(), SubmissionStatus::Complete);
        submission.wait(Duration::ZERO).unwrap();
    }
}
