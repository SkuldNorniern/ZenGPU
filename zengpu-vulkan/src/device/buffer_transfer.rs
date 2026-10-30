//! Submission-owned staging for asynchronous buffer readback.

use super::*;
use zengpu_hal::{CompletedReadback, GpuReadback, ReadSubmission};

/// Removed from the resource table before submission; its raw allocation is
/// retained by the fence state and, for reads, by the returned readback job.
pub(super) struct AsyncBuffer {
    inner: Arc<VulkanDeviceInner>,
    buffer: VulkanBuffer,
}

impl Drop for AsyncBuffer {
    fn drop(&mut self) {
        // The final owner is released only after completion, or before a failed
        // submission reaches the queue. Readback memory is always coherent.
        unsafe {
            if !self.buffer.mapped.is_null() {
                self.inner.device.unmap_memory(self.buffer.memory);
            }
            self.inner.device.destroy_buffer(self.buffer.buffer, None);
            self.inner.device.free_memory(self.buffer.memory, None);
        }
    }
}

struct VulkanReadback {
    work: Submission,
    storage: Arc<AsyncBuffer>,
    skip: usize,
    len: usize,
}

impl GpuSubmission for VulkanReadback {
    fn cycle_id(&self) -> u64 {
        self.work.cycle_id()
    }
    fn poll(&self) -> Result<SubmissionStatus> {
        self.work.poll()
    }
    fn wait(&self, timeout: Duration) -> Result<()> {
        self.work.wait(timeout)
    }
}

impl GpuReadback for VulkanReadback {
    fn into_bytes(self: Box<Self>) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.try_reserve_exact(self.len)
            .map_err(|_| GpuError::OutOfMemory(MemoryUsage::Readback))?;
        out.resize(self.len, 0);
        self.read_into(&mut out)?;
        Ok(out)
    }

    fn read_into(self: Box<Self>, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.len {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!(
                    "readback has {} bytes, destination has {}",
                    self.len,
                    dst.len()
                ),
            )));
        }
        self.work.wait(Duration::MAX)?;
        // SAFETY: the completed transfer wrote this mapped, host-coherent range.
        // The job owns its allocation and the caller uniquely borrows dst.
        unsafe {
            copy_nonoverlapping(
                self.storage.buffer.mapped.add(self.skip),
                dst.as_mut_ptr(),
                self.len,
            );
        }
        Ok(())
    }
}

fn copy_span(offset: u64, len: u64, size: u64) -> Result<(u64, u64, usize)> {
    let end = offset
        .checked_add(len)
        .filter(|&end| end <= size)
        .ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "readback range {offset} + {len} exceeds buffer size {size}"
            )))
        })?;
    let start = offset & !3;
    let aligned_end = end.checked_add(3).map(|end| end & !3).ok_or_else(|| {
        GpuError::InvalidUsage(UsageError::BindingMismatch(
            "readback alignment overflow".into(),
        ))
    })?;
    let span = aligned_end - start;
    if span > isize::MAX as u64 {
        return Err(GpuError::OutOfMemory(MemoryUsage::Readback));
    }
    Ok((start, span, (offset - start) as usize))
}

impl VulkanDevice {
    pub(super) fn submit_read_buffer_impl(
        &self,
        cycle: u64,
        source: BufferHandle,
        offset: u64,
        len: u64,
    ) -> Result<ReadSubmission> {
        self.reap_orphans();
        // Validate before allocating. Recheck the full handle after allocation,
        // and keep the table locked through queue submission and lifetime tracking.
        let validate =
            |buffers: &SlotMap<marker::Buffer, VulkanBuffer>| -> Result<(u64, u64, usize)> {
                let buffer = buffers.get(source).ok_or_else(|| stale(source, buffers))?;
                if !buffer.usage.contains(BufferUsage::READBACK) {
                    return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                        resource: "buffer",
                        needed: "READBACK",
                    }));
                }
                copy_span(offset, len, buffer.size)
            };
        let (start, span, skip) = validate(&self.buffers.lock().unwrap())?;
        if len == 0 {
            return Ok(Box::new(CompletedReadback::new(cycle, Vec::new())));
        }
        let staging = self.create_buffer(BufferDesc {
            size: span,
            usage: BufferUsage::TRANSFER_DST,
            memory: MemoryUsage::Readback,
        })?;
        let allocation = self
            .buffers
            .lock()
            .unwrap()
            .remove(staging)
            .ok_or_else(|| GpuError::Backend("readback staging handle disappeared".into()))?;
        let storage = Arc::new(AsyncBuffer {
            inner: Arc::clone(&self.inner),
            buffer: allocation,
        });
        if storage.buffer.mapped.is_null() {
            return Err(GpuError::Backend(
                "readback memory is not host visible".into(),
            ));
        }
        let buffers = self.buffers.lock().unwrap();
        validate(&buffers)?;
        let input = buffers.get(source).expect("source was validated").buffer;
        let output = storage.buffer.buffer;
        let (work, _) = self.one_shot_submit_tracked_inner(
            cycle,
            &[source.index()],
            Some(Arc::clone(&storage)),
            |device, cmd| {
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier {
                            src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                            ..Default::default()
                        }],
                        &[],
                        &[],
                    );
                    device.cmd_copy_buffer(
                        cmd,
                        input,
                        output,
                        &[vk::BufferCopy {
                            src_offset: start,
                            dst_offset: 0,
                            size: span,
                        }],
                    );
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier {
                            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                            dst_access_mask: vk::AccessFlags::HOST_READ,
                            ..Default::default()
                        }],
                        &[],
                        &[],
                    );
                }
                Ok(())
            },
        )?;
        Ok(Box::new(VulkanReadback {
            work,
            storage,
            skip,
            len: len as usize,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::VulkanInstance;
    use zengpu_hal::{AdapterRequest, BufferCopyOp, DeviceRequest, GpuInstance};

    fn device() -> Option<Box<dyn GpuDevice>> {
        let instance = VulkanInstance::new().ok()?;
        instance
            .request_adapter(AdapterRequest::default())?
            .open(DeviceRequest::default())
            .ok()
    }

    fn buffer(
        device: &dyn GpuDevice,
        size: u64,
        usage: BufferUsage,
        memory: MemoryUsage,
    ) -> BufferHandle {
        device
            .create_buffer(BufferDesc {
                size,
                usage,
                memory,
            })
            .unwrap()
    }

    #[test]
    fn readback_spans_align_copies_without_exposing_padding() {
        assert_eq!(copy_span(3, 5, 9).unwrap(), (0, 8, 3));
        assert_eq!(copy_span(8, 1, 9).unwrap(), (8, 4, 0));
        assert!(copy_span(8, 2, 9).is_err());
        assert!(copy_span(u64::MAX, 1, u64::MAX).is_err());
    }

    #[test]
    fn async_reads_preserve_unaligned_ranges_and_validate_requests() {
        let _guard = crate::test_gpu_lock();
        let Some(device) = device() else { return };
        let source = buffer(&*device, 17, BufferUsage::READBACK, MemoryUsage::Upload);
        let data: Vec<u8> = (0..17).collect();
        device.write_buffer(source, 0, &data).unwrap();
        for (offset, len) in [(0u64, 17u64), (3, 5), (16, 1), (17, 0)] {
            let read = device.submit_read_buffer(7, source, offset, len).unwrap();
            assert_eq!(read.cycle_id(), 7);
            assert_eq!(
                read.into_bytes().unwrap(),
                data[offset as usize..(offset + len) as usize]
            );
        }
        let read = device.submit_read_buffer(8, source, 1, 3).unwrap();
        let mut bytes = [0; 3];
        read.read_into(&mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3]);
        assert!(device.submit_read_buffer(0, source, 16, 2).is_err());
        assert!(device.submit_read_buffer(0, source, u64::MAX, 1).is_err());
        let read = device.submit_read_buffer(0, source, 0, 4).unwrap();
        assert!(read.read_into(&mut [0; 3]).is_err());
        let forbidden = buffer(&*device, 4, BufferUsage::STORAGE, MemoryUsage::Upload);
        assert!(device.submit_read_buffer(0, forbidden, 0, 4).is_err());
        device.destroy_buffer(forbidden);
        device.destroy_buffer(source);
        assert!(device.submit_read_buffer(0, source, 0, 0).is_err());
    }

    #[test]
    fn async_read_observes_a_queued_producer_and_survives_source_destruction() {
        let _guard = crate::test_gpu_lock();
        let Some(device) = device() else { return };
        let input = buffer(&*device, 16, BufferUsage::TRANSFER_SRC, MemoryUsage::Upload);
        let output = buffer(
            &*device,
            16,
            BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
            MemoryUsage::GpuOnly,
        );
        let data: Vec<u8> = (40..56).collect();
        device.write_buffer(input, 0, &data).unwrap();
        let producer = device
            .submit_compute_ops(
                11,
                &[ComputeOp::CopyBuffer(BufferCopyOp {
                    src: input,
                    dst: output,
                    src_offset: 0,
                    dst_offset: 0,
                    len: 16,
                })],
            )
            .unwrap();
        // Do not poll/wait for the producer. A synchronous read currently rejects
        // this busy handle; the async path must enqueue after its GPU write.
        let read = device.submit_read_buffer(12, output, 3, 9).unwrap();
        device.destroy_buffer(input);
        device.destroy_buffer(output);
        assert_eq!(read.cycle_id(), 12);
        assert_eq!(read.into_bytes().unwrap(), data[3..12]);
        drop(producer);
    }

    #[test]
    fn dropping_async_reads_keeps_staging_alive_until_completion() {
        let _guard = crate::test_gpu_lock();
        let Some(device) = device() else { return };
        let source = buffer(
            &*device,
            4 << 20,
            BufferUsage::READBACK,
            MemoryUsage::GpuOnly,
        );
        for _ in 0..16 {
            drop(
                device
                    .submit_read_buffer(13, source, 1, (4 << 20) - 2)
                    .unwrap(),
            );
        }
        device.destroy_buffer(source);
        let native = device.as_any().downcast_ref::<VulkanDevice>().unwrap();
        native.wait_idle().unwrap();
        assert!(native.orphans.lock().unwrap().is_empty());
    }
}
