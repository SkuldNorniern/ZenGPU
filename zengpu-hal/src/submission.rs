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
    /// Returns [`GpuError::Timeout`](crate::GpuError::Timeout) if the deadline expires. A timeout does
    /// not cancel the GPU work; the handle remains valid and may be polled or
    /// waited again.
    fn wait(&self, timeout: Duration) -> Result<()>;
}

/// Boxed, object-safe submission handle returned by [`crate::GpuDevice`].
pub type Submission = Box<dyn GpuSubmission>;

/// A submitted buffer read whose bytes are taken once it has completed.
pub trait GpuReadback: GpuSubmission {
    /// Take the bytes read back, waiting for completion if needed.
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>>;

    /// Copy the bytes read back into `dst`, waiting for completion if needed.
    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()>;
}

/// Boxed, object-safe readback handle returned by [`crate::GpuDevice::submit_read_buffer`].
pub type ReadSubmission = Box<dyn GpuReadback>;

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

/// Completed readback used by synchronous and reference backends.
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
