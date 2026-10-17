
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

/// Bytes are taken once the read has completed.
pub trait GpuReadback: GpuSubmission {
    /// Waits for completion if needed.
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>>;

    /// Waits for completion if needed.
    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()>;
}

pub type ReadSubmission = Box<dyn GpuReadback>;

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

/// Already complete. This is the case for synchronous and reference backends.
pub struct CompletedReadback {
    cycle_id: u64,
    bytes: Vec<u8>,
}

impl CompletedReadback {
    pub fn new(cycle_id: u64, bytes: Vec<u8>) -> Self {
        Self { cycle_id, bytes }
    }
}

impl GpuSubmission for CompletedReadback {
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

impl GpuReadback for CompletedReadback {
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>> {
        Ok(self.bytes)
    }

    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(&self.bytes);
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

    #[test]
    fn completed_readback_returns_bytes_in_both_forms() {
        let readback: ReadSubmission = Box::new(CompletedReadback::new(7, vec![1, 2, 3]));
        assert_eq!(readback.cycle_id(), 7);
        assert_eq!(readback.poll().unwrap(), SubmissionStatus::Complete);
        assert_eq!(readback.into_bytes().unwrap(), vec![1, 2, 3]);

        let readback: ReadSubmission = Box::new(CompletedReadback::new(8, vec![4, 5]));
        let mut dst = [0u8; 2];
        readback.read_into(&mut dst).unwrap();
        assert_eq!(dst, [4, 5]);
    }
}
