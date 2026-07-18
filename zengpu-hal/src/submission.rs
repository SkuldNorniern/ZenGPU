
use std::time::Duration;

use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionStatus {
    Pending,
    Complete,
}

/// A submitted unit of GPU work. `cycle` is caller-supplied and never read by the backend;
/// real-time code uses it to drop completions from older cycles. Referenced resources stay alive
/// until [`SubmissionStatus::Complete`].
pub trait GpuSubmission: Send + Sync {
    fn cycle_id(&self) -> u64;

    fn poll(&self) -> Result<SubmissionStatus>;

    /// Wait at most `timeout`. Returns [`GpuError::Timeout`] on expiry. The GPU work keeps running,
    /// and the handle stays valid for another poll or wait.
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
