use super::*;

impl VulkanDevice {
    /// Cloneable access to the raw Vulkan device context used by render targets,
    /// frame graphs, and engine-side graphics resources.
    pub fn context(&self) -> DeviceContext {
        DeviceContext::from_inner(Arc::clone(&self.inner))
    }

    /// Wait until all work submitted to this logical device has completed.
    pub fn wait_idle(&self) -> Result<()> {
        let _queue = self.inner.queue_lock.lock().unwrap();
        unsafe {
            self.inner
                .device
                .device_wait_idle()
                .map_err(|e| GpuError::Backend(format!("device_wait_idle: {e}")))
        }
    }

    /// End and submit a recorded [`VulkanCommandList`], then block until the
    /// GPU has finished executing it.
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
        // The synchronous API owns the queue until its fence signals. Besides
        // satisfying VkQueue external synchronization, this avoids concurrent
        // driver waits racing teardown/reuse of pooled submission resources.
        let _queue = self.inner.queue_lock.lock().unwrap();
        let submit_result = unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &[submit_info], fence)
                .map_err(|e| map_vk_err("vkQueueSubmit", e))
        };
        if submit_result.is_ok() {
            submitted = true;
        }
        let wait_result = submit_result.and_then(|()| unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| map_vk_err("vkWaitForFences", e))
        });
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

    /// Submit a one-shot command buffer that records work via `f`, then waits
    /// for completion. Used for staging uploads and layout transitions.
    pub(crate) fn one_shot_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Device, vk::CommandBuffer) -> Result<()>,
    {
        let cmd = self.cmd_list_pool.acquire()?;

        let record_result = record(&self.inner.device, cmd);

        unsafe {
            let _ = self.inner.device.end_command_buffer(cmd);
        }

        if let Err(e) = record_result {
            self.cmd_list_pool.release(cmd);
            return Err(e);
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
        if submit_result.is_ok() {
            submitted = true;
        }
        let wait_result = submit_result.and_then(|()| unsafe {
            self.inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| map_vk_err("vkWaitForFences", e))
        });
        if !submitted || wait_result.is_ok() {
            self.fence_pool.release(fence);
            self.cmd_list_pool.release(cmd);
        } else {
            unsafe {
                self.inner.device.destroy_fence(fence, None);
            }
        }

        wait_result
    }

    /// Record and enqueue a one-shot command buffer without waiting. The
    /// returned submission owns the fence and command buffer until completion.
    fn one_shot_submit_async<F>(&self, cycle_id: u64, record: F) -> Result<Submission>
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
        {
            let mut lifetime = self.lifetime.lock().unwrap();
            lifetime.in_flight += 1;
        }
        let submit_result = {
            let _queue = self.inner.queue_lock.lock().unwrap();
            unsafe {
                self.inner
                    .device
                    .queue_submit(self.inner.queue, &[submit_info], fence)
                    .map_err(|e| map_vk_err("vkQueueSubmit", e))
            }
        };
        if let Err(e) = submit_result {
            self.lifetime.lock().unwrap().in_flight -= 1;
            self.fence_pool.release(fence);
            self.cmd_list_pool.release(cmd);
            return Err(e);
        }

        Ok(Box::new(VulkanSubmission {
            cycle_id,
            inner: Arc::clone(&self.inner),
            fence_pool: Arc::clone(&self.fence_pool),
            cmd_pool: Arc::clone(&self.cmd_list_pool),
            buffers: Arc::clone(&self.buffers),
            textures: Arc::clone(&self.textures),
            samplers: Arc::clone(&self.samplers),
            pipelines: Arc::clone(&self.pipelines),
            lifetime: Arc::clone(&self.lifetime),
            state: Mutex::new(VulkanSubmissionState {
                fence: Some(fence),
                cmd: Some(cmd),
            }),
        }))
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

    pub(super) fn memory_type_flags(&self, type_index: u32) -> vk::MemoryPropertyFlags {
        self.inner.memory_properties.memory_types[type_index as usize].property_flags
    }

    pub(super) fn submit_mixed_compute_async(
        &self,
        cycle_id: u64,
        ops: &[ComputeOp<'_>],
    ) -> Result<Submission> {
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
                    for &idx in op.bindings.textures {
                        if idx >= self.bindless.texture_capacity
                            || textures.get_by_slot_index(idx).is_none()
                            || !bound_textures[idx as usize]
                        {
                            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                                format!("texture binding index {idx} is not live and bound"),
                            )));
                        }
                    }
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
                    let mut pc = [0u8; COMPUTE_PUSH_CONSTANT_BYTES];
                    let mut pc_len = 0usize;
                    for bytes in op.bindings.buffers.iter().map(|v| v.to_ne_bytes()).chain(
                        op.bindings.scalars.iter().map(|scalar| match scalar {
                            Scalar::U32(v) => v.to_ne_bytes(),
                            Scalar::I32(v) => v.to_ne_bytes(),
                            Scalar::F32(v) => v.to_bits().to_ne_bytes(),
                        }),
                    ) {
                        if pc_len + 4 > pc.len() {
                            return Err(GpuError::Dispatch(format!(
                                "push constants exceed {} bytes",
                                pc.len()
                            )));
                        }
                        pc[pc_len..pc_len + 4].copy_from_slice(&bytes);
                        pc_len += 4;
                    }
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

        let bindless_set = self.bindless.set;
        let submission = self.one_shot_submit_async(cycle_id, move |dev, cmd| {
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
        if ops.is_empty() {
            return Ok(Box::new(zengpu_hal::CompletedSubmission::new(cycle_id)));
        }

        // Raw bindless indices are validated while the tables are locked.
        // The asynchronous submission contract requires callers to keep all
        // referenced resources alive until the returned token completes.
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
                for &idx in op.bindings.textures {
                    if idx >= self.bindless.texture_capacity {
                        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                            format!(
                                "texture binding index {idx} exceeds bindless capacity {}",
                                self.bindless.texture_capacity
                            ),
                        )));
                    }
                    if textures.get_by_slot_index(idx).is_none() || !bound_textures[idx as usize] {
                        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                            format!("texture binding index {idx} is not live and bound"),
                        )));
                    }
                }
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

                let mut pc = [0u8; COMPUTE_PUSH_CONSTANT_BYTES];
                let mut pc_len = 0usize;
                let mut push_pc = |bytes: [u8; 4]| -> Result<()> {
                    if pc_len + 4 > pc.len() {
                        return Err(GpuError::Dispatch(format!(
                            "push constants exceed {} bytes",
                            pc.len()
                        )));
                    }
                    pc[pc_len..pc_len + 4].copy_from_slice(&bytes);
                    pc_len += 4;
                    Ok(())
                };
                for &idx in op.bindings.buffers {
                    push_pc(idx.to_ne_bytes())?;
                }
                for scalar in op.bindings.scalars {
                    push_pc(match scalar {
                        Scalar::U32(v) => v.to_ne_bytes(),
                        Scalar::I32(v) => v.to_ne_bytes(),
                        Scalar::F32(v) => v.to_bits().to_ne_bytes(),
                    })?;
                }
                out.push((vk_pipeline, vk_layout, pc, pc_len, op.grid));
            }
            out
        };

        let bindless_set = self.bindless.set;
        let last = resolved.len() - 1;
        let submission = self.one_shot_submit_async(cycle_id, move |dev, cmd| {
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
            }
            Ok(())
        });
        drop(samplers);
        submission
    }

    /// Acquire a pooled, reset-reusable [`VulkanCommandList`] and begin
    /// recording. Part of the unified graphics API (D17/GU); see
    /// [`zengpu_hal::GraphicsDevice::create_command_list`].
    pub(crate) fn create_command_list_impl(&self) -> Result<VulkanCommandList> {
        let cmd = self.cmd_list_pool.acquire()?;
        Ok(VulkanCommandList::new(
            Arc::clone(&self.inner),
            Arc::clone(&self.cmd_list_pool),
            cmd,
            Arc::clone(&self.pipelines),
            Arc::clone(&self.render_targets),
            Arc::clone(&self.buffers),
            Arc::clone(&self.query_pools),
            self.bindless.set,
        ))
    }
}
