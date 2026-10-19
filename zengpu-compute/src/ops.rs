//! Recorded kernel launches that run together.
//!
//! One-at-a-time launches wait for the GPU after each op (~29 µs on HIP, ~50 µs on Vulkan, even for
//! tiny ops). An [`OpList`] records launches and submits them as one batch: each op sees the
//! previous one's writes, and the list waits once.
//!
//! A list can run repeatedly with the same arrays (e.g. once per token). Scratch arrays (split-K
//! workspaces etc.) are held by the list and returned to the pool on clear or drop, never while
//! work may be pending.

use std::mem::take;
use std::ptr::addr_eq;

use zengpu_hal::{
    Bindings, BufferHandle, DType, DispatchOp, GpuDevice, PipelineHandle, Result, Scalar,
    Submission, UsageError,
};

use crate::{BufferPool, DeviceArray};

/// One recorded launch, owning the args [`DispatchOp`] borrows.
#[derive(Debug, Clone)]
struct Recorded {
    pipeline: PipelineHandle,
    buffers: Vec<u32>,
    scalars: Vec<Scalar>,
    grid: [u32; 3],
}

/// Launches recorded for one submission, with their scratch arrays.
pub struct OpList<'p> {
    pool: &'p BufferPool,
    ops: Vec<Recorded>,
    temporaries: Vec<DeviceArray>,
    required_buffers: Vec<(BufferHandle, u64)>,
}

impl<'p> OpList<'p> {
    /// Empty list. Scratch arrays come from and return to `pool`.
    pub fn new(pool: &'p BufferPool) -> Self {
        Self {
            pool,
            ops: Vec::new(),
            temporaries: Vec::new(),
            required_buffers: Vec::new(),
        }
    }

    pub fn pool(&self) -> &'p BufferPool {
        self.pool
    }

    /// Record a launch of `pipeline` over `grid` workgroups. `buffers` are bindless
    /// slots ([`zengpu_hal::Handle::index`]).
    pub fn dispatch(
        &mut self,
        pipeline: PipelineHandle,
        buffers: &[u32],
        scalars: &[Scalar],
        grid: [u32; 3],
    ) {
        self.ops.push(Recorded {
            pipeline,
            buffers: buffers.to_vec(),
            scalars: scalars.to_vec(),
            grid,
        });
    }

    /// Keep this handle and byte capacity valid until the list runs. Raw bindless slots carry
    /// no lifetime.
    pub fn require_buffer(&mut self, buffer: BufferHandle, bytes: u64) {
        self.required_buffers.push((buffer, bytes));
    }

    /// Validates `array` immediately and keeps its full handle, so `submit` can re-check it before
    /// slot indices reach the GPU.
    pub fn validate_array(&mut self, array: &DeviceArray) -> Result<u32> {
        let len = array.validate_contiguous(self.pool.device())?;
        self.require_buffer(
            array.buffer,
            u64::from(len) * array.dtype.size_bytes() as u64,
        );
        Ok(len)
    }

    /// Keeps `array` alive until the list is cleared or dropped. Use this for scratch that the ops
    /// need only while they run.
    pub fn keep(&mut self, array: DeviceArray) {
        self.require_buffer(array.buffer, array.allocation_size_bytes());
        self.temporaries.push(array);
    }

    pub fn scratch(&mut self, shape: Vec<u32>, dtype: DType) -> Result<u32> {
        let array = self.pool.alloc(shape, dtype)?;
        let slot = array.buffer.index();
        self.keep(array);
        Ok(slot)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Appends a recording atomically. On error, discards this attempt's launches and requirements,
    /// and returns its scratch arrays to the pool.
    ///
    /// The closure must only record. It must not clear or run the list.
    pub fn transaction<T>(&mut self, record: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let checkpoint = (
            self.ops.len(),
            self.temporaries.len(),
            self.required_buffers.len(),
        );
        let result = record(self);
        if result.is_err() {
            self.ops.truncate(checkpoint.0);
            self.required_buffers.truncate(checkpoint.2);
            for array in self.temporaries.drain(checkpoint.1..) {
                self.pool.free(array);
            }
        }
        result
    }

    fn with_ops<T>(&self, run: impl FnOnce(&[DispatchOp<'_>]) -> T) -> T {
        let ops: Vec<DispatchOp<'_>> = self
            .ops
            .iter()
            .map(|op| DispatchOp {
                pipeline: op.pipeline,
                bindings: Bindings {
                    buffers: &op.buffers,
                    textures: &[],
                    images: &[],
                    scalars: &op.scalars,
                },
                grid: op.grid,
            })
            .collect();
        run(&ops)
    }

    /// Run all launches as one batch and wait. The list keeps its launches and scratch, so it can
    /// run again.
    pub fn run(&self, device: &dyn GpuDevice) -> Result<()> {
        self.check_device(device)?;
        if self.ops.is_empty() {
            return Ok(());
        }
        self.with_ops(|ops| device.dispatch_batch(ops))
    }

    /// Submit as one batch without waiting. Scratch stays with the list: keep it alive and don't
    /// clear until the submission completes.
    pub fn submit(&self, device: &dyn GpuDevice, cycle_id: u64) -> Result<Submission> {
        self.check_device(device)?;
        self.with_ops(|ops| device.submit_batch(cycle_id, ops))
    }

    fn check_device(&self, device: &dyn GpuDevice) -> Result<()> {
        if !addr_eq(device.as_any(), self.pool.device().as_any()) {
            return Err(UsageError::BindingMismatch(
                "operation list must run on its pool device".into(),
            )
            .into());
        }
        for &(buffer, bytes) in &self.required_buffers {
            if device.buffer_size(buffer)? < bytes {
                return Err(
                    UsageError::BindingMismatch("recorded buffer is too small".into()).into(),
                );
            }
        }
        Ok(())
    }

    /// Forget the launches and return scratch arrays to the pool. Only call when no
    /// submission is pending.
    pub fn clear(&mut self) {
        self.ops.clear();
        self.required_buffers.clear();
        for array in take(&mut self.temporaries) {
            self.pool.free(array);
        }
    }
}

impl Drop for OpList<'_> {
    fn drop(&mut self) {
        self.clear();
    }
}
