
use std::time::Duration;

use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionStatus {
    Pending,
    Complete,
}

/// The cycle id is caller-defined and never read by the backend. Real-time consumers use it to
/// reject completions from an older cycle.
///
/// Backends keep native resources and descriptor slots alive until completion. Destroying a
/// handle while work is pending invalidates it immediately, but native destruction and slot
/// reuse may be deferred.
pub trait GpuSubmission: Send + Sync {
    fn cycle_id(&self) -> u64;

    fn poll(&self) -> Result<SubmissionStatus>;

    /// Waits for the work to finish, or returns early on timeout.
    ///
    /// A timeout does not cancel the GPU work. The handle stays valid and can be polled or waited
    /// on again.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Timeout`](crate::GpuError::Timeout) on expiry.
    fn wait(&self, timeout: Duration) -> Result<()>;
}

pub type Submission = Box<dyn GpuSubmission>;

/// Synchronous and reference backends return a handle that is already complete.
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
