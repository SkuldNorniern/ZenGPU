use super::*;

/// Largest device-local host-visible heap treated as the legacy BAR aperture, not Resizable BAR
/// (which spans VRAM).
const SMALL_BAR_HEAP: u64 = 512 << 20;

impl VulkanDevice {
    pub fn context(&self) -> DeviceContext {
        DeviceContext::from_inner(Arc::clone(&self.inner))
    }

    pub fn wait_idle(&self) -> Result<()> {
        {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.inner
                    .device
                    .device_wait_idle()
                    .map_err(|e| GpuError::Backend(format!("device_wait_idle: {e}")))?;
            }
            self.inner.serials.complete(self.inner.serials.submitted());
        }
        self.collect_garbage();
        Ok(())
    }

    /// Blocks until the GPU finishes it.
    pub fn submit_and_wait(&self, list: VulkanCommandList) -> Result<()> {
        let cmd = list.cmd;
        let pool = Arc::clone(&list.pool);
        unsafe {
            self.inner
                .device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end_command_buffer: {e}")))?;
        }
        let fence = self.fence_pool.acquire()?;
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        let mut submitted = false;
        let _queue = self.inner.queue_lock.lock().unwrap();
        let submit_result = unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &[submit_info], fence)
                .map_err(|e| map_vk_err("vkQueueSubmit", e))
        };
        let mut serial = None;
        if submit_result.is_ok() {
            submitted = true;
            serial = Some(self.inner.serials.next());
        }
        let wait_result = submit_result.and_then(|()| unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| map_vk_err("vkWaitForFences", e))
        });
        if let (Some(serial), Ok(())) = (serial, &wait_result) {
            self.inner.serials.complete(serial);
        }
        if !submitted || wait_result.is_ok() {
            self.fence_pool.release(fence);
            pool.release(cmd);
        } else {
            unsafe {
                self.inner.device.destroy_fence(fence, None);
            }
        }
        wait_result
    }

    /// Records a one-shot command buffer with `f` and waits for it to complete.
    ///
    /// For staging uploads and layout transitions.
    pub(crate) fn one_shot_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let pending = self.submit_raw(record)?;
        self.finish_raw(pending)
    }

    /// Submits a one-shot command buffer without waiting.
    ///
    /// Completion never destroys deferred resources, so it is safe while resource
    /// tables are locked.
    pub(crate) fn submit_raw<F>(&self, record: F) -> Result<RawSubmit>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let cmd = self.cmd_list_pool.acquire()?;
        let record_result = record(&self.inner.device, cmd);
        let end_result = unsafe {
            self.inner
                .device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("vkEndCommandBuffer: {e}")))
        };
        if let Err(e) = record_result.and(end_result) {
            self.cmd_list_pool.release(cmd);
            return Err(e);
        }
        let fence = match self.fence_pool.acquire() {
            Ok(fence) => fence,
            Err(e) => {
                self.cmd_list_pool.release(cmd);
                return Err(e);
            }
        };
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        let submitted = {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.inner
                    .device
                    .queue_submit(self.inner.queue, &[submit_info], fence)
                    .map_err(|e| map_vk_err("vkQueueSubmit", e))
            }
            .map(|()| self.inner.serials.next())
        };
        match submitted {
            Ok(serial) => Ok(RawSubmit { cmd, fence, serial }),
            Err(e) => {
                self.fence_pool.release(fence);
                self.cmd_list_pool.release(cmd);
                Err(e)
            }
        }
    }

    pub(crate) fn finish_raw(&self, pending: RawSubmit) -> Result<()> {
        let RawSubmit { cmd, fence, serial } = pending;
        let waited = unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| map_vk_err("vkWaitForFences", e))
        };
        match waited {
            Ok(()) => {
                self.inner.serials.complete(serial);
                self.fence_pool.release(fence);
                self.cmd_list_pool.release(cmd);
                Ok(())
            }
            Err(e) => {
                unsafe { self.inner.device.destroy_fence(fence, None) };
                Err(e)
            }
        }
    }

    /// The submission owns the fence and command buffer until it completes.
    fn one_shot_submit_async<F>(&self, cycle_id: u64, uses: &[u32], record: F) -> Result<Submission>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        // Compute callers already reaped before taking their resource-table locks.
        self.one_shot_submit_tracked_inner(cycle_id, uses, record)
            .map(|(submission, _)| submission)
    }

    pub(crate) fn one_shot_submit_tracked<F>(
        &self,
        cycle_id: u64,
        uses: &[u32],
        record: F,
    ) -> Result<(Submission, u64)>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        self.reap_orphans();
        self.one_shot_submit_tracked_inner(cycle_id, uses, record)
    }

    fn one_shot_submit_tracked_inner<F>(
        &self,
        cycle_id: u64,
        uses: &[u32],
        record: F,
    ) -> Result<(Submission, u64)>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let cmd = self.cmd_list_pool.acquire()?;
        let record_result = record(&self.inner.device, cmd);
        let end_result = unsafe {
            self.inner
                .device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("vkEndCommandBuffer: {e}")))
        };
        if let Err(e) = record_result.and(end_result) {
            self.cmd_list_pool.release(cmd);
            return Err(e);
        }

        let fence = self.fence_pool.acquire()?;
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        let submit_result = {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.inner
                    .device
                    .queue_submit(self.inner.queue, &[submit_info], fence)
                    .map_err(|e| map_vk_err("vkQueueSubmit", e))
            }
            .map(|()| self.inner.serials.next())
        };
        let serial = match submit_result {
            Ok(serial) => serial,
            Err(e) => {
                self.fence_pool.release(fence);
                self.cmd_list_pool.release(cmd);
                return Err(e);
            }
        };
        self.lifetime.lock().unwrap().mark_used(uses, serial);

        let submission: Submission = Box::new(VulkanSubmission {
            cycle_id,
            serial,
            inner: Arc::clone(&self.inner),
            fence_pool: Arc::clone(&self.fence_pool),
            cmd_pool: Arc::clone(&self.cmd_list_pool),
            buffers: Arc::clone(&self.buffers),
            textures: Arc::clone(&self.textures),
            samplers: Arc::clone(&self.samplers),
            pipelines: Arc::clone(&self.pipelines),
            level_slots: Arc::clone(&self.bindless.level_slots),
            lifetime: Arc::clone(&self.lifetime),
            orphans: Arc::clone(&self.orphans),
            state: Mutex::new(VulkanSubmissionState {
                fence: Some(fence),
                cmd: Some(cmd),
            }),
        });
        Ok((submission, serial))
    }

    pub(super) fn find_memory_type(
        &self,
        type_bits: u32,
        props: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_props = &self.inner.memory_properties;
        (0..mem_props.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
        })
    }

    /// Writes to buffers the host cannot map (device-local VRAM) through a temporary
    /// host-visible buffer.
    pub(super) fn staged_write(&self, dst: vk::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        if data.len() as u64 <= INLINE_UPDATE_MAX
            && offset.is_multiple_of(4)
            && data.len().is_multiple_of(4)
        {
            return self.one_shot_submit(|dev, cmd| {
                unsafe {
                    dev.cmd_update_buffer(cmd, dst, offset, data);
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier {
                            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                            dst_access_mask: vk::AccessFlags::MEMORY_READ
                                | vk::AccessFlags::MEMORY_WRITE,
                            ..Default::default()
                        }],
                        &[],
                        &[],
                    );
                }
                Ok(())
            });
        }
        let mut slots = self.transfer_slots.lock().unwrap();
        let slots = transfer_slots(self, &mut slots)?;
        let mut in_flight: [Option<RawSubmit>; 2] = [None, None];
        let result = (|| {
            for (i, chunk) in data.chunks(STAGING_CHUNK as usize).enumerate() {
                let slot = &slots.upload[i % 2];
                if let Some(pending) = in_flight[i % 2].take() {
                    self.finish_raw(pending)?;
                }
                unsafe { copy_nonoverlapping(chunk.as_ptr(), slot.mapped, chunk.len()) };
                let dst_offset = offset + (i as u64) * STAGING_CHUNK;
                let len = chunk.len() as u64;
                in_flight[i % 2] = Some(self.submit_raw(|dev, cmd| {
                    unsafe {
                        dev.cmd_copy_buffer(
                            cmd,
                            slot.buffer,
                            dst,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset,
                                size: len,
                            }],
                        );
                        dev.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::DependencyFlags::empty(),
                            &[vk::MemoryBarrier {
                                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                                dst_access_mask: vk::AccessFlags::MEMORY_READ
                                    | vk::AccessFlags::MEMORY_WRITE,
                                ..Default::default()
                            }],
                            &[],
                            &[],
                        );
                    }
                    Ok(())
                })?);
            }
            Ok(())
        })();
        let drained = in_flight
            .into_iter()
            .flatten()
            .map(|pending| self.finish_raw(pending))
            .fold(Ok(()), Result::and);
        result.and(drained)
    }

    pub(super) fn staged_read(&self, src: vk::Buffer, offset: u64, out: &mut [u8]) -> Result<()> {
        let mut slots = self.transfer_slots.lock().unwrap();
        let slots = transfer_slots(self, &mut slots)?;
        let total = out.len() as u64;
        let chunks = total.div_ceil(STAGING_CHUNK) as usize;
        let enqueue = |i: usize| -> Result<RawSubmit> {
            let slot = &slots.readback[i % 2];
            let src_offset = offset + i as u64 * STAGING_CHUNK;
            let len = STAGING_CHUNK.min(total - i as u64 * STAGING_CHUNK);
            self.submit_raw(|dev, cmd| {
                unsafe {
                    dev.cmd_pipeline_barrier(
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
                    dev.cmd_copy_buffer(
                        cmd,
                        src,
                        slot.buffer,
                        &[vk::BufferCopy {
                            src_offset,
                            dst_offset: 0,
                            size: len,
                        }],
                    );
                    dev.cmd_pipeline_barrier(
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
            })
        };
        let mut in_flight: [Option<RawSubmit>; 2] = [None, None];
        let result = (|| {
            in_flight[0] = Some(enqueue(0)?);
            for (i, chunk) in out.chunks_mut(STAGING_CHUNK as usize).enumerate() {
                if i + 1 < chunks {
                    in_flight[(i + 1) % 2] = Some(enqueue(i + 1)?);
                }
                if let Some(pending) = in_flight[i % 2].take() {
                    self.finish_raw(pending)?;
                }
                let slot = &slots.readback[i % 2];
                unsafe { copy_nonoverlapping(slot.mapped, chunk.as_mut_ptr(), chunk.len()) };
            }
            Ok(())
        })();
        let drained = in_flight
            .into_iter()
            .flatten()
            .map(|pending| self.finish_raw(pending))
            .fold(Ok(()), Result::and);
        result.and(drained)
    }

    /// Memory types for a buffer of `size` bytes, best first.
    ///
    /// Without ReBAR the only device-local host-visible heap is the 256 MiB legacy aperture, often
    /// with a few MiB free (4.6 MiB on an RX 9060 XT under RADV). Large `CpuToGpu` buffers skip it.
    /// With ReBAR the heap spans VRAM.
    pub(super) fn memory_type_candidates(
        &self,
        type_bits: u32,
        usage: MemoryUsage,
        size: u64,
    ) -> Vec<u32> {
        let mem_props = &self.inner.memory_properties;
        let mut candidates = Vec::with_capacity(2);
        if let Some(preferred) = self.find_memory_type(type_bits, memory_usage_to_vk(usage)) {
            let heap = mem_props.memory_heaps
                [mem_props.memory_types[preferred as usize].heap_index as usize];
            let flags = mem_props.memory_types[preferred as usize].property_flags;
            let small_bar = flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) && heap.size <= SMALL_BAR_HEAP;
            if !(small_bar && size > heap.size / 16) {
                candidates.push(preferred);
            }
        }
        if let Some(fallback) =
            memory_usage_fallback(usage).and_then(|props| self.find_memory_type(type_bits, props))
            && !candidates.contains(&fallback)
        {
            candidates.push(fallback);
        }
        candidates
    }

    pub(super) fn memory_type_flags(&self, type_index: u32) -> vk::MemoryPropertyFlags {
        self.inner.memory_properties.memory_types[type_index as usize].property_flags
    }

    pub(super) fn submit_mixed_compute_async(
        &self,
        cycle_id: u64,
        ops: &[ComputeOp<'_>],
    ) -> Result<Submission> {
        self.reap_orphans();
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        enum ResolvedOp {
            Copy {
                src: vk::Buffer,
                src_offset: u64,
                dst: vk::Buffer,
                dst_offset: u64,
                len: u64,
            },
            Dispatch {
                pipeline: vk::Pipeline,
                layout: vk::PipelineLayout,
                pc: [u8; COMPUTE_PUSH_CONSTANT_BYTES],
                pc_len: usize,
                grid: [u32; 3],
            },
        }

        let buffers = self.buffers.lock().unwrap();
        let textures = self.textures.lock().unwrap();
        let bound_textures = self.bindless.bound_textures.lock().unwrap();
        let pipelines = self.pipelines.lock().unwrap();
        let samplers = self.samplers.lock().unwrap();
        let mut resolved = Vec::with_capacity(ops.len());

        for op in ops {
            match op {
                ComputeOp::CopyBuffer(copy) => {
                    let src = buffers
                        .get(copy.src)
                        .ok_or_else(|| stale(copy.src, &buffers))?;
                    let dst = buffers
                        .get(copy.dst)
                        .ok_or_else(|| stale(copy.dst, &buffers))?;
                    if copy.src == copy.dst {
                        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                            "copy_buffer requires distinct source and destination buffers".into(),
                        )));
                    }
                    if !src.usage.contains(BufferUsage::TRANSFER_SRC) {
                        return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                            resource: "source buffer",
                            needed: "TRANSFER_SRC",
                        }));
                    }
                    if !dst.usage.contains(BufferUsage::TRANSFER_DST) {
                        return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                            resource: "destination buffer",
                            needed: "TRANSFER_DST",
                        }));
                    }
                    let src_end = copy.src_offset.checked_add(copy.len).ok_or_else(|| {
                        GpuError::InvalidUsage(UsageError::BindingMismatch(
                            "source copy range overflows u64".into(),
                        ))
                    })?;
                    let dst_end = copy.dst_offset.checked_add(copy.len).ok_or_else(|| {
                        GpuError::InvalidUsage(UsageError::BindingMismatch(
                            "destination copy range overflows u64".into(),
                        ))
                    })?;
                    if src_end > src.size || dst_end > dst.size {
                        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                            format!(
                                "copy ranges src {}..{src_end}/{} dst {}..{dst_end}/{}",
                                copy.src_offset, src.size, copy.dst_offset, dst.size
                            ),
                        )));
                    }
                    resolved.push(ResolvedOp::Copy {
                        src: src.buffer,
                        src_offset: copy.src_offset,
                        dst: dst.buffer,
                        dst_offset: copy.dst_offset,
                        len: copy.len,
                    });
                }
                ComputeOp::Dispatch(op) => {
                    if op.grid.contains(&0)
                        || op
                            .grid
                            .iter()
                            .zip(self.inner.limits.max_dispatch_size)
                            .any(|(requested, limit)| *requested > limit)
                    {
                        return Err(GpuError::Dispatch(format!(
                            "invalid dispatch grid {:?}; device limit {:?}",
                            op.grid, self.inner.limits.max_dispatch_size
                        )));
                    }
                    for &idx in op.bindings.buffers {
                        let buffer = buffers.get_by_slot_index(idx).ok_or_else(|| {
                            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                                "buffer binding index {idx} is not live"
                            )))
                        })?;
                        if idx >= self.bindless.buffer_capacity {
                            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                                format!("buffer binding index {idx} exceeds bindless capacity"),
                            )));
                        }
                        if !buffer.usage.contains(BufferUsage::STORAGE) {
                            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                                resource: "bound buffer",
                                needed: "STORAGE",
                            }));
                        }
                    }
                    self.check_texel_bindings(&op.bindings, &textures, &bound_textures)?;
                    let (pipeline, layout) = match pipelines
                        .get(op.pipeline)
                        .ok_or_else(|| GpuError::Dispatch("stale pipeline handle".to_string()))?
                    {
                        VulkanPipeline::Compute { layout, pipeline } => (*pipeline, *layout),
                        VulkanPipeline::Graphics { .. } => {
                            return Err(GpuError::Dispatch(
                                "dispatch called with a graphics pipeline handle".into(),
                            ));
                        }
                    };
                    let (pc, pc_len) = push_constants(&op.bindings)?;
                    resolved.push(ResolvedOp::Dispatch {
                        pipeline,
                        layout,
                        pc,
                        pc_len,
                        grid: op.grid,
                    });
                }
            }
        }

        let uses = mixed_buffer_uses(ops);
        let bindless_set = self.bindless.set;
        let submission = self.one_shot_submit_async(cycle_id, &uses, move |dev, cmd| {
            let barrier = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            unsafe {
                for (index, op) in resolved.iter().enumerate() {
                    if index != 0 {
                        dev.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER
                                | vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER
                                | vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[barrier],
                            &[],
                            &[],
                        );
                    }
                    match op {
                        ResolvedOp::Copy {
                            src,
                            src_offset,
                            dst,
                            dst_offset,
                            len,
                        } => {
                            if *len != 0 {
                                dev.cmd_copy_buffer(
                                    cmd,
                                    *src,
                                    *dst,
                                    &[vk::BufferCopy {
                                        src_offset: *src_offset,
                                        dst_offset: *dst_offset,
                                        size: *len,
                                    }],
                                );
                            }
                        }
                        ResolvedOp::Dispatch {
                            pipeline,
                            layout,
                            pc,
                            pc_len,
                            grid,
                        } => {
                            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
                            dev.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                *layout,
                                0,
                                &[bindless_set],
                                &[],
                            );
                            if *pc_len != 0 {
                                dev.cmd_push_constants(
                                    cmd,
                                    *layout,
                                    vk::ShaderStageFlags::COMPUTE,
                                    0,
                                    &pc[..*pc_len],
                                );
                            }
                            dev.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
                        }
                    }
                }
                hand_off(dev, cmd);
            }
            Ok(())
        });
        drop(samplers);
        submission
    }

    pub(super) fn submit_compute_batch_async(
        &self,
        cycle_id: u64,
        ops: &[DispatchOp<'_>],
    ) -> Result<Submission> {
        self.reap_orphans();
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        let buffers = self.buffers.lock().unwrap();
        let textures = self.textures.lock().unwrap();
        let bound_textures = self.bindless.bound_textures.lock().unwrap();
        let pipelines = self.pipelines.lock().unwrap();
        let samplers = self.samplers.lock().unwrap();

        #[allow(clippy::type_complexity)]
        let resolved: Vec<(
            vk::Pipeline,
            vk::PipelineLayout,
            [u8; COMPUTE_PUSH_CONSTANT_BYTES],
            usize,
            [u32; 3],
        )> = {
            let mut out = Vec::with_capacity(ops.len());
            for op in ops {
                if op.grid.contains(&0) {
                    return Err(GpuError::Dispatch(format!(
                        "dispatch grid dimensions must be non-zero, got {:?}",
                        op.grid
                    )));
                }
                if op
                    .grid
                    .iter()
                    .zip(self.inner.limits.max_dispatch_size)
                    .any(|(requested, limit)| *requested > limit)
                {
                    return Err(GpuError::Dispatch(format!(
                        "dispatch grid {:?} exceeds device limit {:?}",
                        op.grid, self.inner.limits.max_dispatch_size
                    )));
                }
                for &idx in op.bindings.buffers {
                    if idx >= self.bindless.buffer_capacity {
                        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                            format!(
                                "buffer binding index {idx} exceeds bindless capacity {}",
                                self.bindless.buffer_capacity
                            ),
                        )));
                    }
                    let buffer = buffers.get_by_slot_index(idx).ok_or_else(|| {
                        GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                            "buffer binding index {idx} is not live"
                        )))
                    })?;
                    if !buffer.usage.contains(BufferUsage::STORAGE) {
                        return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                            resource: "bound buffer",
                            needed: "STORAGE",
                        }));
                    }
                }
                self.check_texel_bindings(&op.bindings, &textures, &bound_textures)?;
                let p = pipelines
                    .get(op.pipeline)
                    .ok_or_else(|| GpuError::Dispatch("stale pipeline handle".to_string()))?;
                let (vk_pipeline, vk_layout) = match p {
                    VulkanPipeline::Compute { layout, pipeline } => (*pipeline, *layout),
                    VulkanPipeline::Graphics { .. } => {
                        return Err(GpuError::Dispatch(
                            "dispatch called with a graphics pipeline handle".to_string(),
                        ));
                    }
                };

                let (pc, pc_len) = push_constants(&op.bindings)?;
                out.push((vk_pipeline, vk_layout, pc, pc_len, op.grid));
            }
            out
        };

        let uses: Vec<u32> = ops
            .iter()
            .flat_map(|op| op.bindings.buffers.iter().copied())
            .collect();
        let bindless_set = self.bindless.set;
        let last = resolved.len() - 1;
        let submission = self.one_shot_submit_async(cycle_id, &uses, move |dev, cmd| {
            let barrier = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };
            unsafe {
                for (i, (vk_pipeline, vk_layout, pc, pc_len, grid)) in resolved.iter().enumerate() {
                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *vk_pipeline);
                    dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        *vk_layout,
                        0,
                        &[bindless_set],
                        &[],
                    );
                    if *pc_len != 0 {
                        dev.cmd_push_constants(
                            cmd,
                            *vk_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &pc[..*pc_len],
                        );
                    }
                    dev.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
                    if i != last {
                        dev.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[barrier],
                            &[],
                            &[],
                        );
                    }
                }
                hand_off(dev, cmd);
            }
            Ok(())
        });
        drop(samplers);
        submission
    }

    /// Pooled [`VulkanCommandList`], already recording.
    pub(crate) fn create_command_list_impl(&self) -> Result<VulkanCommandList> {
        let cmd = self.cmd_list_pool.acquire()?;
        Ok(VulkanCommandList::new(
            Arc::clone(&self.inner),
            Arc::clone(&self.cmd_list_pool),
            cmd,
            Arc::clone(&self.pipelines),
            Arc::clone(&self.render_targets),
            Arc::clone(&self.textures),
            Arc::clone(&self.buffers),
            Arc::clone(&self.query_pools),
            self.bindless.set,
        ))
    }
}

impl VulkanDevice {
    pub(super) fn check_texel_bindings(
        &self,
        bindings: &Bindings<'_>,
        textures: &SlotMap<marker::Texture, VulkanTexture>,
        bound_textures: &[bool],
    ) -> Result<()> {
        for &idx in bindings.textures {
            if idx >= self.bindless.texture_capacity
                || textures.get_by_slot_index(idx).is_none()
                || !bound_textures[idx as usize]
            {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    format!("texture binding index {idx} is not live and bound"),
                )));
            }
        }
        for &idx in bindings.images {
            // Level-view slots map to their parent texture's slot.
            let owner = if idx < self.bindless.storage_capacity {
                Some(idx)
            } else {
                self.bindless.level_slots.lock().unwrap().owner(idx)
            };
            let texture = owner
                .and_then(|owner| textures.get_by_slot_index(owner))
                .ok_or_else(|| {
                    GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                        "image binding index {idx} is not a live texture"
                    )))
                })?;
            if !texture.usage.contains(vk::ImageUsageFlags::STORAGE) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "bound image",
                    needed: "STORAGE",
                }));
            }
        }
        Ok(())
    }
}

/// Push-constant layout that ZSL kernels expect: buffers, textures, images, then scalars, 4
/// bytes each.
pub(super) fn push_constants(
    bindings: &Bindings<'_>,
) -> Result<([u8; COMPUTE_PUSH_CONSTANT_BYTES], usize)> {
    let mut pc = [0u8; COMPUTE_PUSH_CONSTANT_BYTES];
    let mut pc_len = 0usize;
    let slots = bindings
        .buffers
        .iter()
        .chain(bindings.textures)
        .chain(bindings.images)
        .map(|v| v.to_ne_bytes());
    let scalars = bindings.scalars.iter().map(|scalar| match scalar {
        Scalar::U32(v) => v.to_ne_bytes(),
        Scalar::I32(v) => v.to_ne_bytes(),
        Scalar::F32(v) => v.to_bits().to_ne_bytes(),
    });
    for bytes in slots.chain(scalars) {
        if pc_len + 4 > pc.len() {
            return Err(GpuError::Dispatch(format!(
                "push constants exceed {} bytes",
                pc.len()
            )));
        }
        pc[pc_len..pc_len + 4].copy_from_slice(&bytes);
        pc_len += 4;
    }
    Ok((pc, pc_len))
}

/// Buffers a mixed batch reads or writes, by bindless index: dispatch bindings and both ends of
/// each copy.
fn mixed_buffer_uses(ops: &[ComputeOp<'_>]) -> Vec<u32> {
    let mut uses = Vec::new();
    for op in ops {
        match op {
            ComputeOp::Dispatch(dispatch) => uses.extend_from_slice(dispatch.bindings.buffers),
            ComputeOp::CopyBuffer(copy) => uses.extend([copy.src.index(), copy.dst.index()]),
        }
    }
    uses
}

/// Makes compute and copy writes visible to the next reader: a later dispatch, a sampling render
/// pass, a copy, or the host.
///
/// # Safety
/// `cmd` must be recording on `dev`.
pub(super) unsafe fn hand_off(dev: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier {
        src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::MEMORY_READ
            | vk::AccessFlags::MEMORY_WRITE
            | vk::AccessFlags::HOST_READ,
        ..Default::default()
    };
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

pub(crate) struct RawSubmit {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    serial: u64,
}

/// Largest write sent inline with vkCmdUpdateBuffer (the Vulkan limit).
const INLINE_UPDATE_MAX: u64 = 65536;

const STAGING_CHUNK: u64 = 8 << 20;

/// A persistently mapped host-visible transfer buffer.
pub(crate) struct HostSlot {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

/// Staging for buffers the host cannot map: two write-combined upload slots, two host-cached
/// readback slots. Kept after first use: allocating and zeroing 8 MiB per transfer costs more than
/// the transfer.
pub(crate) struct TransferSlots {
    upload: [HostSlot; 2],
    readback: [HostSlot; 2],
}

// SAFETY: raw handles and mapped pointers, used only under the device's
unsafe impl Send for TransferSlots {}

impl TransferSlots {
    fn new(device: &VulkanDevice) -> Result<Self> {
        let mut made: Vec<HostSlot> = Vec::with_capacity(4);
        for readback in [false, false, true, true] {
            match HostSlot::new(device, readback) {
                Ok(slot) => made.push(slot),
                Err(e) => {
                    for slot in made {
                        slot.destroy(&device.inner.device);
                    }
                    return Err(e);
                }
            }
        }
        let mut made = made.into_iter();
        let mut next = || {
            made.next()
                .ok_or(GpuError::OutOfMemory(MemoryUsage::Upload))
        };
        Ok(Self {
            upload: [next()?, next()?],
            readback: [next()?, next()?],
        })
    }

    /// Free every slot; the device must be idle.
    pub(crate) fn destroy(self, dev: &Device) {
        for slot in self.upload.into_iter().chain(self.readback) {
            slot.destroy(dev);
        }
    }
}

fn transfer_slots<'a>(
    device: &VulkanDevice,
    slots: &'a mut Option<TransferSlots>,
) -> Result<&'a TransferSlots> {
    if slots.is_none() {
        *slots = Some(TransferSlots::new(device)?);
    }
    slots
        .as_ref()
        .ok_or(GpuError::OutOfMemory(MemoryUsage::Upload))
}

impl HostSlot {
    /// `readback` prefers host-cached memory, which the CPU reads far faster.
    fn new(device: &VulkanDevice, readback: bool) -> Result<Self> {
        let dev = &device.inner.device;
        let buffer = unsafe {
            dev.create_buffer(
                &vk::BufferCreateInfo {
                    size: STAGING_CHUNK,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| GpuError::Backend(format!("vkCreateBuffer (staging): {e}")))?;
        let reqs = unsafe { dev.get_buffer_memory_requirements(buffer) };
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = readback
            .then(|| {
                device.find_memory_type(
                    reqs.memory_type_bits,
                    coherent | vk::MemoryPropertyFlags::HOST_CACHED,
                )
            })
            .flatten()
            .or_else(|| device.find_memory_type(reqs.memory_type_bits, coherent));
        let memory = type_index.and_then(|memory_type_index| {
            unsafe {
                dev.allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: reqs.size,
                        memory_type_index,
                        ..Default::default()
                    },
                    None,
                )
            }
            .ok()
        });
        let Some(memory) = memory else {
            unsafe { dev.destroy_buffer(buffer, None) };
            return Err(GpuError::OutOfMemory(MemoryUsage::Upload));
        };
        let mapped = unsafe {
            dev.bind_buffer_memory(buffer, memory, 0).and_then(|()| {
                dev.map_memory(memory, 0, STAGING_CHUNK, vk::MemoryMapFlags::empty())
            })
        };
        match mapped {
            Ok(mapped) => Ok(Self {
                buffer,
                memory,
                mapped: mapped as *mut u8,
            }),
            Err(e) => {
                unsafe {
                    dev.destroy_buffer(buffer, None);
                    dev.free_memory(memory, None);
                }
                Err(GpuError::Backend(format!("staging buffer: {e}")))
            }
        }
    }

    fn destroy(self, dev: &Device) {
        unsafe {
            dev.unmap_memory(self.memory);
            dev.destroy_buffer(self.buffer, None);
            dev.free_memory(self.memory, None);
        }
    }
}
