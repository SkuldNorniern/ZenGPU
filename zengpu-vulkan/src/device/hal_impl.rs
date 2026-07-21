use super::*;

impl GpuDevice for VulkanDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> HalCapabilities {
        HalCapabilities {
            graphics: self.inner.graphics,
            compute: true,
            features: self.inner.features,
        }
    }

    fn limits(&self) -> DeviceLimits {
        self.inner.limits
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle> {
        if desc.usage.contains(BufferUsage::STORAGE)
            && desc.size > self.inner.limits.max_storage_buffer_range
        {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!(
                    "storage buffer size {} exceeds device limit {}",
                    desc.size, self.inner.limits.max_storage_buffer_range
                ),
            )));
        }
        let vk_usage = buffer_usage_to_vk(desc.usage);
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size,
            usage: vk_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            self.inner
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateBuffer: {e}")))?
        };

        let mem_reqs = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };

        let preferred_flags = memory_usage_to_vk(desc.memory);
        let type_index = self
            .find_memory_type(mem_reqs.memory_type_bits, preferred_flags)
            .or_else(|| {
                memory_usage_fallback(desc.memory)
                    .and_then(|props| self.find_memory_type(mem_reqs.memory_type_bits, props))
            });

        let type_index = match type_index {
            Some(i) => i,
            None => {
                unsafe { self.inner.device.destroy_buffer(buffer, None) };
                return Err(GpuError::OutOfMemory(desc.memory));
            }
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index: type_index,
            ..Default::default()
        };

        let memory = unsafe {
            match self.inner.device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(_) => {
                    self.inner.device.destroy_buffer(buffer, None);
                    return Err(GpuError::OutOfMemory(desc.memory));
                }
            }
        };

        if let Err(e) = unsafe { self.inner.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.inner.device.destroy_buffer(buffer, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::Backend(format!("vkBindBufferMemory: {e}")));
        }

        let actual_flags = self.memory_type_flags(type_index);
        let is_host_visible = actual_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

        let mapped = if is_host_visible {
            match unsafe {
                self.inner
                    .device
                    .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => ptr as *mut u8,
                Err(e) => {
                    unsafe {
                        self.inner.device.destroy_buffer(buffer, None);
                        self.inner.device.free_memory(memory, None);
                    }
                    return Err(GpuError::Backend(format!("vkMapMemory: {e}")));
                }
            }
        } else {
            null_mut()
        };

        let vk_buf = buffer; // Copy before moving into struct
        let mut buffers = self.buffers.lock().unwrap();
        if desc.usage.contains(BufferUsage::STORAGE)
            && buffers.next_index() >= self.bindless.buffer_capacity
        {
            unsafe {
                if !mapped.is_null() {
                    self.inner.device.unmap_memory(memory);
                }
                self.inner.device.destroy_buffer(buffer, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!(
                    "storage buffer slot exceeds bindless capacity {}",
                    self.bindless.buffer_capacity
                ),
            )));
        }
        let handle = buffers.insert(VulkanBuffer {
            buffer: vk_buf,
            memory,
            size: desc.size,
            usage: desc.usage,
            mapped,
        });
        drop(buffers);
        if desc.usage.contains(BufferUsage::STORAGE) {
            self.bind_buffer_to_bindless(handle.index(), vk_buf, desc.size);
        }
        Ok(handle)
    }

    fn write_buffer(&self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if self.lifetime.lock().unwrap().in_flight != 0 {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "write_buffer cannot mutate buffers while an asynchronous submission is pending"
                    .into(),
            )));
        }

        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "write_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(data.len()).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("range {start}..{end} exceeds buffer size {}", buf.size),
            )));
        }
        unsafe {
            copy_nonoverlapping(data.as_ptr(), buf.mapped.add(start), data.len());
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: BufferHandle, offset: u64, len: u64) -> Result<Vec<u8>> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if self.lifetime.lock().unwrap().in_flight != 0 {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "read_buffer cannot observe buffers while an asynchronous submission is pending"
                    .into(),
            )));
        }

        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "read_buffer on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(len as usize).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("range {start}..{end} exceeds buffer size {}", buf.size),
            )));
        }
        let mut out = vec![0u8; len as usize];
        unsafe {
            copy_nonoverlapping(buf.mapped.add(start), out.as_mut_ptr(), len as usize);
        }
        Ok(out)
    }

    fn read_buffer_into(&self, buffer: BufferHandle, offset: u64, dst: &mut [u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(buffer).ok_or_else(|| stale(buffer, &buffers))?;
        if self.lifetime.lock().unwrap().in_flight != 0 {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "read_buffer_into cannot observe buffers while an asynchronous submission is pending"
                    .into(),
            )));
        }
        if !buf.usage.contains(BufferUsage::READBACK) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: "READBACK",
            }));
        }
        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "read_buffer_into on non-host-visible buffer".to_string(),
            ));
        }
        let start = offset as usize;
        let end = start.checked_add(dst.len()).ok_or_else(|| {
            GpuError::InvalidUsage(UsageError::BindingMismatch(format!(
                "range {start}..overflow exceeds buffer size {}",
                buf.size
            )))
        })?;
        if end > buf.size as usize {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("range {start}..{end} exceeds buffer size {}", buf.size),
            )));
        }
        unsafe {
            copy_nonoverlapping(buf.mapped.add(start), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        if self.lifetime.lock().unwrap().in_flight != 0 {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "copy_buffer cannot run while an asynchronous submission is pending".into(),
            )));
        }
        let (src_buffer, dst_buffer) = {
            let src_buf = buffers.get(src).ok_or_else(|| stale(src, &buffers))?;
            let dst_buf = buffers.get(dst).ok_or_else(|| stale(dst, &buffers))?;
            if !src_buf.usage.contains(BufferUsage::TRANSFER_SRC) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "source buffer",
                    needed: "TRANSFER_SRC",
                }));
            }
            if !dst_buf.usage.contains(BufferUsage::TRANSFER_DST) {
                return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                    resource: "destination buffer",
                    needed: "TRANSFER_DST",
                }));
            }
            if src == dst {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "copy_buffer requires distinct source and destination buffers".into(),
                )));
            }
            let src_end = src_offset.checked_add(len).ok_or_else(|| {
                GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "source copy range overflows u64".into(),
                ))
            })?;
            let dst_end = dst_offset.checked_add(len).ok_or_else(|| {
                GpuError::InvalidUsage(UsageError::BindingMismatch(
                    "destination copy range overflows u64".into(),
                ))
            })?;
            if src_end > src_buf.size || dst_end > dst_buf.size {
                return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                    format!(
                        "copy ranges src {src_offset}..{src_end}/{} dst {dst_offset}..{dst_end}/{}",
                        src_buf.size, dst_buf.size
                    ),
                )));
            }
            (src_buf.buffer, dst_buf.buffer)
        };
        if len == 0 {
            return Ok(());
        }
        let result = self.one_shot_submit(|dev, cmd| {
            unsafe {
                dev.cmd_copy_buffer(
                    cmd,
                    src_buffer,
                    dst_buffer,
                    &[vk::BufferCopy {
                        src_offset,
                        dst_offset,
                        size: len,
                    }],
                );
            }
            Ok(())
        });
        drop(buffers);
        result
    }

    fn destroy_buffer(&self, buffer: BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        let mut lifetime = self.lifetime.lock().unwrap();
        if lifetime.in_flight != 0 {
            if let Some(buf) = buffers.retire(buffer) {
                lifetime
                    .deferred
                    .push(DeferredVulkanResource::Buffer(buffer.index(), buf));
            }
        } else if let Some(buf) = buffers.remove(buffer) {
            unsafe {
                if !buf.mapped.is_null() {
                    self.inner.device.unmap_memory(buf.memory);
                }
                self.inner.device.destroy_buffer(buf.buffer, None);
                self.inner.device.free_memory(buf.memory, None);
            }
        }
    }

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
        let mip_levels = desc.mip_levels.max(1);
        let array_layers = desc.array_layers.max(1);
        let depth = if desc.dimension == TexDim::D3 {
            desc.depth.max(1)
        } else {
            1
        };
        if desc.dimension == TexDim::Cube && array_layers != 6 {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("TexDim::Cube requires array_layers == 6, got {array_layers}"),
            )));
        }

        let format = hal_format_to_vk(desc.format);
        let mut usage = vk::ImageUsageFlags::empty();
        if desc.usage.contains(TextureUsage::SAMPLED) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.contains(TextureUsage::RENDER_TARGET) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if desc.usage.contains(TextureUsage::DEPTH_STENCIL) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if desc.usage.contains(TextureUsage::TRANSFER_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(TextureUsage::TRANSFER_DST) {
            usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        let image_type = match desc.dimension {
            TexDim::D2 | TexDim::Cube => vk::ImageType::TYPE_2D,
            TexDim::D3 => vk::ImageType::TYPE_3D,
        };
        let flags = if desc.dimension == TexDim::Cube {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };

        let image_info = vk::ImageCreateInfo {
            flags,
            image_type,
            format,
            extent: vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth,
            },
            mip_levels,
            array_layers,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let image = unsafe {
            self.inner
                .device
                .create_image(&image_info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateImage: {e}")))?
        };
        let mem_reqs = unsafe { self.inner.device.get_image_memory_requirements(image) };
        let type_index = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let type_index = match type_index {
            Some(i) => i,
            None => {
                unsafe { self.inner.device.destroy_image(image, None) };
                return Err(GpuError::OutOfMemory(MemoryUsage::GpuOnly));
            }
        };
        let memory = unsafe {
            match self.inner.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: mem_reqs.size,
                    memory_type_index: type_index,
                    ..Default::default()
                },
                None,
            ) {
                Ok(m) => m,
                Err(_) => {
                    self.inner.device.destroy_image(image, None);
                    return Err(GpuError::OutOfMemory(MemoryUsage::GpuOnly));
                }
            }
        };
        if let Err(e) = unsafe { self.inner.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                self.inner.device.destroy_image(image, None);
                self.inner.device.free_memory(memory, None);
            }
            return Err(GpuError::Backend(format!("vkBindImageMemory: {e}")));
        }
        let view_type = match desc.dimension {
            TexDim::D2 if array_layers > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
            TexDim::D2 => vk::ImageViewType::TYPE_2D,
            TexDim::D3 => vk::ImageViewType::TYPE_3D,
            TexDim::Cube => vk::ImageViewType::CUBE,
        };
        let view = unsafe {
            match self.inner.device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: mip_levels,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                    ..Default::default()
                },
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.inner.device.destroy_image(image, None);
                    self.inner.device.free_memory(memory, None);
                    return Err(GpuError::Backend(format!("vkCreateImageView: {e}")));
                }
            }
        };
        Ok(self.textures.lock().unwrap().insert(VulkanTexture {
            image,
            view,
            memory,
            format,
            extent: vk::Extent2D {
                width: desc.width,
                height: desc.height,
            },
            depth,
            mip_levels,
            array_layers,
            usage,
        }))
    }

    fn upload_texture_data(&self, texture: TextureHandle, data: &[u8]) -> Result<()> {
        self.upload_texture_data_impl(texture, 0, 0, data)
    }

    fn copy_texture_to_buffer(&self, texture: TextureHandle, buffer: BufferHandle) -> Result<()> {
        self.copy_texture_to_buffer_impl(texture, buffer)
    }

    fn upload_texture_data_region(
        &self,
        texture: TextureHandle,
        mip_level: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        self.upload_texture_data_impl(texture, mip_level, layer, data)
    }

    fn generate_mipmaps(&self, texture: TextureHandle) -> Result<()> {
        let usage = {
            let textures = self.textures.lock().unwrap();
            let tex = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("generate_mipmaps: stale texture handle".to_string())
            })?;
            tex.usage
        };
        let needed = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !usage.contains(needed) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "texture",
                needed: "TRANSFER_SRC | TRANSFER_DST",
            }));
        }
        self.generate_mipmaps_impl(texture)
    }

    fn destroy_texture(&self, texture: TextureHandle) {
        let mut textures = self.textures.lock().unwrap();
        let mut bound_textures = self.bindless.bound_textures.lock().unwrap();
        let mut lifetime = self.lifetime.lock().unwrap();
        let (removed, tex) = if lifetime.in_flight != 0 {
            if let Some(tex) = textures.retire(texture) {
                lifetime
                    .deferred
                    .push(DeferredVulkanResource::Texture(texture.index(), tex));
                (true, None)
            } else {
                (false, None)
            }
        } else {
            let tex = textures.remove(texture);
            (tex.is_some(), tex)
        };
        if removed && texture.index() < self.bindless.texture_capacity {
            bound_textures[texture.index() as usize] = false;
        }
        if let Some(tex) = tex {
            unsafe {
                self.inner.device.destroy_image_view(tex.view, None);
                self.inner.device.destroy_image(tex.image, None);
                self.inner.device.free_memory(tex.memory, None);
            }
        }
    }

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerHandle> {
        let min = filter_to_vk(desc.min_filter);
        let mag = filter_to_vk(desc.mag_filter);
        let addr = address_to_vk(desc.address);
        let mipmap_mode = match desc.mip_filter {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let anisotropy_enable = self.inner.sampler_anisotropy && desc.anisotropy > 1;
        let max_anisotropy = (desc.anisotropy as f32).min(self.inner.max_sampler_anisotropy);
        let info = vk::SamplerCreateInfo {
            mag_filter: mag,
            min_filter: min,
            mipmap_mode,
            address_mode_u: addr,
            address_mode_v: addr,
            address_mode_w: addr,
            min_lod: desc.lod_min,
            max_lod: desc.lod_max,
            anisotropy_enable: if anisotropy_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            max_anisotropy,
            compare_enable: if desc.compare.is_some() {
                vk::TRUE
            } else {
                vk::FALSE
            },
            compare_op: compare_fn_to_vk(desc.compare.unwrap_or(CompareFn::Always)),
            border_color: border_color_to_vk(desc.border),
            ..Default::default()
        };
        let sampler = unsafe {
            self.inner
                .device
                .create_sampler(&info, None)
                .map_err(|e| GpuError::Backend(format!("vkCreateSampler: {e}")))?
        };
        Ok(self.samplers.lock().unwrap().insert(sampler))
    }

    fn destroy_sampler(&self, sampler: SamplerHandle) {
        let mut samplers = self.samplers.lock().unwrap();
        let mut lifetime = self.lifetime.lock().unwrap();
        if lifetime.in_flight != 0 {
            if let Some(s) = samplers.retire(sampler) {
                lifetime
                    .deferred
                    .push(DeferredVulkanResource::Sampler(sampler.index(), s));
            }
        } else if let Some(s) = samplers.remove(sampler) {
            unsafe { self.inner.device.destroy_sampler(s, None) };
        }
    }

    fn supports_anisotropic_filtering(&self) -> bool {
        self.inner.sampler_anisotropy
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    fn create_shader(&self, desc: ShaderDesc<'_>) -> Result<ShaderHandle> {
        let ShaderSource::Spirv(spirv) = desc.source else {
            return Err(GpuError::ShaderCompile(
                "vulkan backend only accepts SPIR-V shaders".to_string(),
            ));
        };
        if spirv.len() % 4 != 0 {
            return Err(GpuError::ShaderCompile(
                "SPIR-V byte length must be a multiple of 4".to_string(),
            ));
        }
        let (prefix, aligned_words, suffix) = unsafe { spirv.align_to::<u32>() };
        let copied_words;
        let words = if prefix.is_empty() && suffix.is_empty() {
            aligned_words
        } else {
            copied_words = spirv
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>();
            &copied_words
        };

        // Structurally pre-check the SPIR-V before handing it to the driver.
        // Malformed SPIR-V otherwise faults inside the driver with no diagnostic
        // (and validation layers may be absent). This check intentionally only
        // *warns*: it is a heuristic over the opcode subset ZenGPU models, so it
        // can flag valid external SPIR-V it does not understand — it must never
        // block a shader the driver would have accepted.
        if let Err(e) = zengpu_spv::validate(words) {
            log::warn!("[zengpu-vulkan] SPIR-V structural check reported issues:\n{e}");
            log::warn!(
                "[zengpu-vulkan] disassembly:\n{}",
                zengpu_spv::disassemble(words)
            );
        }
        log::trace!(
            "[zengpu-vulkan] shader SPIR-V ({} words):\n{}",
            words.len(),
            zengpu_spv::disassemble(words)
        );

        let module = unsafe {
            self.inner
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: spirv.len(),
                        p_code: words.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| GpuError::ShaderCompile(format!("vkCreateShaderModule: {e}")))?
        };
        Ok(self.shaders.lock().unwrap().insert(module))
    }

    fn destroy_shader(&self, shader: ShaderHandle) {
        let mut shaders = self.shaders.lock().unwrap();
        if let Some(m) = shaders.remove(shader) {
            unsafe {
                self.inner.device.destroy_shader_module(m, None);
            }
        }
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc<'_>) -> Result<PipelineHandle> {
        let shader_module = {
            let shaders = self.shaders.lock().unwrap();
            *shaders
                .get(desc.shader)
                .ok_or_else(|| GpuError::PipelineCreation("stale shader handle".to_string()))?
        };

        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: COMPUTE_PUSH_CONSTANT_BYTES as u32,
        };
        let layout = unsafe {
            self.inner
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        set_layout_count: 1,
                        p_set_layouts: &self.bindless.layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &pc_range,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| GpuError::PipelineCreation(format!("vkCreatePipelineLayout: {e}")))?
        };

        let entry = CString::new(desc.entry)
            .map_err(|e| GpuError::PipelineCreation(format!("entry name nul: {e}")))?;
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        };
        let result = unsafe {
            self.inner.device.create_compute_pipelines(
                self.pipeline_cache,
                &[vk::ComputePipelineCreateInfo {
                    stage,
                    layout,
                    ..Default::default()
                }],
                None,
            )
        };
        match result {
            Ok(pipelines) => Ok(self
                .pipelines
                .lock()
                .unwrap()
                .insert(VulkanPipeline::Compute {
                    layout,
                    pipeline: pipelines[0],
                })),
            Err((_, e)) => {
                unsafe {
                    self.inner.device.destroy_pipeline_layout(layout, None);
                }
                Err(GpuError::PipelineCreation(format!(
                    "vkCreateComputePipelines: {e}"
                )))
            }
        }
    }

    fn destroy_pipeline(&self, pipeline: PipelineHandle) {
        let mut pipelines = self.pipelines.lock().unwrap();
        let mut lifetime = self.lifetime.lock().unwrap();
        if lifetime.in_flight != 0 {
            if let Some(p) = pipelines.retire(pipeline) {
                lifetime
                    .deferred
                    .push(DeferredVulkanResource::Pipeline(pipeline.index(), p));
            }
        } else if let Some(p) = pipelines.remove(pipeline) {
            let (pipeline, layout) = p.handles();
            unsafe {
                self.inner.device.destroy_pipeline(pipeline, None);
                self.inner.device.destroy_pipeline_layout(layout, None);
            }
        }
    }

    fn dispatch(
        &self,
        pipeline: PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        self.dispatch_batch(&[DispatchOp {
            pipeline,
            bindings,
            grid,
        }])
    }

    fn dispatch_batch(&self, ops: &[DispatchOp<'_>]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        // Keep every resolved resource locked through fence completion. This
        // both validates raw bindless indices and prevents concurrent destroy
        // from invalidating descriptors or pipeline handles in this synchronous
        // submission path.
        let buffers = self.buffers.lock().unwrap();
        let textures = self.textures.lock().unwrap();
        let bound_textures = self.bindless.bound_textures.lock().unwrap();
        let pipelines = self.pipelines.lock().unwrap();
        let samplers = self.samplers.lock().unwrap();

        // Resolve pipeline handles and pack push constants for every op up
        // front.
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

                // Pack push constants: [buffer_indices, scalars], each as 4 bytes.
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
        let result = self.one_shot_submit(move |dev, cmd| {
            // A later op in the batch may read a buffer an earlier op wrote
            // (e.g. `relu(add(a, b))`); a barrier between dispatches makes
            // those writes visible instead of relying on submission order
            // alone, which Vulkan does not guarantee within one command buffer.
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
        drop(pipelines);
        drop(bound_textures);
        drop(textures);
        drop(buffers);
        result
    }

    fn submit_batch(&self, cycle_id: u64, ops: &[DispatchOp<'_>]) -> Result<Submission> {
        self.submit_compute_batch_async(cycle_id, ops)
    }

    fn submit_compute_ops(&self, cycle_id: u64, ops: &[ComputeOp<'_>]) -> Result<Submission> {
        self.submit_mixed_compute_async(cycle_id, ops)
    }
}
