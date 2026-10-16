//! Recorded kernel launches that run together.
//!
//! Calling a kernel through [`crate::ElementwiseKernels`] or BLAS one op at a
//! time waits for the GPU after every op, ~29 µs each on HIP and ~50 µs on
//! Vulkan even for tiny ones. A transformer layer is dozens of ops, so that
//! wait would dominate. An [`OpList`] records the launches instead; running
//! it submits them as one batch (each op still sees the previous one's
//! writes) and waits once.
//!
//! A list can be run again and again with the same arrays, e.g. once per
//! generated token. Scratch arrays an op needs only while it runs (split-K
//! workspaces and the like) are kept by the list and given back to the pool
//! when the list is cleared or dropped, never while its work may be pending.

use std::mem::take;

use zengpu_hal::{
    Bindings, DType, DispatchOp, GpuDevice, PipelineHandle, Result, Scalar, Submission,
};

use crate::{BufferPool, DeviceArray};

/// One recorded launch, owning the arguments a [`DispatchOp`] borrows.
#[derive(Debug, Clone)]
struct Recorded {
    pipeline: PipelineHandle,
    buffers: Vec<u32>,
    scalars: Vec<Scalar>,
    grid: [u32; 3],
}

/// Kernel launches recorded for one submission, plus the scratch arrays they
/// use. See the module docs.
pub struct OpList<'p> {
    pool: &'p BufferPool,
    ops: Vec<Recorded>,
    temporaries: Vec<DeviceArray>,
}

impl<'p> OpList<'p> {
    /// An empty list whose scratch arrays come from, and go back to, `pool`.
    pub fn new(pool: &'p BufferPool) -> Self {
        Self {
            pool,
            ops: Vec::new(),
            temporaries: Vec::new(),
        }
    }

    /// The pool this list allocates scratch arrays from.
    pub fn pool(&self) -> &'p BufferPool {
        self.pool
    }

    /// Record a launch of `pipeline` over `grid` workgroups. `buffers` are
    /// bindless slot indices ([`zengpu_hal::Handle::index`]).
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

    /// Keep `array` until the list is cleared or dropped, for scratch memory
    /// the recorded ops use only while they run.
    pub fn keep(&mut self, array: DeviceArray) {
        self.temporaries.push(array);
    }

    /// Allocate a scratch array the list keeps (see [`Self::keep`]).
    pub fn scratch(&mut self, shape: Vec<u32>, dtype: DType) -> Result<u32> {
        let array = self.pool.alloc(shape, dtype)?;
        let slot = array.buffer.index();
        self.temporaries.push(array);
        Ok(slot)
    }

    /// Number of recorded launches.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
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

    /// Run every recorded launch as one batch and wait for it. The list
    /// keeps its launches and scratch arrays, so it can run again.
    pub fn run(&self, device: &dyn GpuDevice) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }
        self.with_ops(|ops| device.dispatch_batch(ops))
    }

    /// Submit every recorded launch as one batch without waiting. The
    /// scratch arrays stay with the list: keep it alive (and do not clear
    /// it) until the submission completes.
    pub fn submit(&self, device: &dyn GpuDevice, cycle_id: u64) -> Result<Submission> {
        self.with_ops(|ops| device.submit_batch(cycle_id, ops))
    }

    /// Forget the recorded launches and hand the scratch arrays back to the
    /// pool. Only call once no submission of this list is pending.
    pub fn clear(&mut self) {
        self.ops.clear();
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
