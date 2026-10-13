use super::*;

impl VulkanDevice {
    /// Registers `texture` in the combined-image-sampler table at its own slot,
    /// for `Bindings::textures`.
    ///
    /// The image must be in the layout the descriptor declares (`SHADER_READ_ONLY_OPTIMAL`,
    /// or `GENERAL` for `STORAGE`) when sampled: after `upload_texture_data`, or a render
    /// pass with `sample_after`.
    ///
    /// Readiness covers same-device, same-queue ordering only. There is no cross-submission
    /// semaphore: external queues and other devices need caller sync.
    pub fn bind_texture(&self, texture: TextureHandle, sampler: SamplerHandle) -> Option<u32> {
        if texture.index() >= self.bindless.texture_capacity {
            return None;
        }
        let textures = self.textures.lock().unwrap();
        let (view, layout) = textures.get(texture).map(|t| (t.view, t.resting))?;
        let samplers = self.samplers.lock().unwrap();
        let vk_sampler = *samplers.get(sampler)?;
        let info = vk::DescriptorImageInfo {
            sampler: vk_sampler,
            image_view: view,
            image_layout: layout,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.bindless.set,
            dst_binding: SSBO_BINDING_TEXTURES,
            dst_array_element: texture.index(),
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &info,
            ..Default::default()
        };
        unsafe {
            self.inner.device.update_descriptor_sets(&[write], &[]);
        }
        self.bindless.bound_textures.lock().unwrap()[texture.index() as usize] = true;
        Some(texture.index())
    }

    /// Shader stages an upload hands its texels to: compute, plus fragment when the queue draws.
    fn shader_stages(&self) -> vk::PipelineStageFlags {
        if self.inner.graphics {
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags::COMPUTE_SHADER
        }
    }

    /// Moves a new `STORAGE` texture to `GENERAL` (where it stays) and registers its mip 0 in the
    /// storage-image table. Contents start undefined.
    pub(super) fn init_storage_texture(&self, texture: TextureHandle) -> Result<()> {
        if texture.index() >= self.bindless.storage_capacity {
            return Err(GpuError::Unsupported(format!(
                "the storage-image table holds {} textures",
                self.bindless.storage_capacity
            )));
        }
        let (image, view, levels) = {
            let textures = self.textures.lock().unwrap();
            let tex = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("init_storage_texture: stale texture handle".to_string())
            })?;
            let view = if tex.storage_view == vk::ImageView::null() {
                tex.view
            } else {
                tex.storage_view
            };
            (tex.image, view, tex.mip_levels)
        };
        self.one_shot_submit(|dev, cmd| {
            let to_general = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: levels,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
            Ok(())
        })?;
        let info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.bindless.set,
            dst_binding: STORAGE_IMAGE_BINDING,
            dst_array_element: texture.index(),
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &info,
            ..Default::default()
        };
        unsafe {
            self.inner.device.update_descriptor_sets(&[write], &[]);
        }
        if let Some(tex) = self.textures.lock().unwrap().get_mut(texture) {
            for mip in 0..levels {
                tex.mark_ready(mip, 0);
            }
        }
        Ok(())
    }

    /// Register a `STORAGE` buffer in every SSBO table, at its slot index. Called
    /// by `create_buffer`.
    ///
    /// Buffers are untyped at the HAL, so each kernel reads through the table matching its ZSL
    /// element type: no bitcast.
    pub(super) fn bind_buffer_to_bindless(&self, slot: u32, buffer: vk::Buffer, size: u64) {
        debug_assert!(slot < self.bindless.buffer_capacity);
        let info = vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: if size == 0 { vk::WHOLE_SIZE } else { size },
        };
        let write_for = |dst_binding: u32| vk::WriteDescriptorSet {
            dst_set: self.bindless.set,
            dst_binding,
            dst_array_element: slot,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &info,
            ..Default::default()
        };
        let writes = [
            write_for(SSBO_BINDING_F32),
            write_for(SSBO_BINDING_U32),
            write_for(SSBO_BINDING_I32),
        ];
        unsafe {
            self.inner.device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Copies the offscreen image into a host-visible buffer. Call after `submit_and_wait`. The
    /// image must be in `COLOR_ATTACHMENT_OPTIMAL`, and is returned there after the copy.
    pub fn copy_offscreen_to_buffer(
        &self,
        offscreen: &OffscreenTarget,
        buffer: BufferHandle,
    ) -> Result<()> {
        GpuDevice::copy_texture_to_buffer(self, offscreen.texture_handle(), buffer)
    }

    pub(super) fn copy_texture_to_buffer_impl(
        &self,
        texture: TextureHandle,
        buffer: BufferHandle,
    ) -> Result<()> {
        let (image, extent) = {
            let textures = self.textures.lock().unwrap();
            let texture = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("copy_texture_to_buffer: stale texture handle".to_string())
            })?;
            (texture.image, texture.extent)
        };
        let vk_buffer = {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(buffer).map(|b| b.buffer).ok_or_else(|| {
                GpuError::Backend("copy_texture_to_buffer: stale buffer handle".to_string())
            })?
        };
        self.one_shot_submit(|dev, cmd| {
            let to_transfer = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                image,
                subresource_range: COLOR_SUBRESOURCE,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                dev.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk_buffer,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    }],
                );
            }
            let to_color = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_READ,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                image,
                subresource_range: COLOR_SUBRESOURCE,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_color],
                );
            }
            Ok(())
        })
    }

    /// Shared path for `upload_texture_data` (mip 0, layer 0) and `upload_texture_data_region`.
    pub(super) fn upload_texture_data_impl(
        &self,
        texture: TextureHandle,
        mip_level: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        let (image, extent, depth) = {
            let textures = self.textures.lock().unwrap();
            let tex = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("upload_texture_data: stale texture handle".to_string())
            })?;
            ((tex.image, tex.resting), tex.extent, tex.depth)
        };
        let mip_width = (extent.width >> mip_level).max(1);
        let mip_height = (extent.height >> mip_level).max(1);
        let mip_depth = (depth >> mip_level).max(1);

        let staging = self.create_buffer(zengpu_hal::BufferDesc {
            size: data.len() as u64,
            usage: zengpu_hal::BufferUsage::TRANSFER_SRC,
            memory: MemoryUsage::Upload,
        })?;
        let uploaded = self.upload_through(
            staging,
            image,
            mip_level,
            layer,
            data,
            [mip_width, mip_height, mip_depth],
        );
        self.destroy_buffer(staging);
        uploaded?;
        if let Some(tex) = self.textures.lock().unwrap().get_mut(texture) {
            tex.mark_ready(mip_level, layer);
        }
        Ok(())
    }

    /// Copies `data` into one mip and layer of `image`. The image is left ready for sampling.
    fn upload_through(
        &self,
        staging: BufferHandle,
        (image, resting): (vk::Image, vk::ImageLayout),
        mip_level: u32,
        layer: u32,
        data: &[u8],
        [mip_width, mip_height, mip_depth]: [u32; 3],
    ) -> Result<()> {
        self.write_buffer(staging, 0, data)?;

        let staging_vk = {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(staging).map(|b| b.buffer).ok_or_else(|| {
                GpuError::Backend("upload_texture_data: staging buffer vanished".to_string())
            })?
        };

        let shader_stages = self.shader_stages();
        self.one_shot_submit(|dev, cmd| {
            let to_transfer = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_level,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                dev.cmd_copy_buffer_to_image(
                    cmd,
                    staging_vk,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level,
                            base_array_layer: layer,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D {
                            width: mip_width,
                            height: mip_height,
                            depth: mip_depth,
                        },
                    }],
                );
            }
            let to_shader = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: resting,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_level,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    shader_stages,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_shader],
                );
            }
            Ok(())
        })
    }

    /// Mip chain from the base level by sequential blits, each level half the last.
    pub(crate) fn generate_mipmaps_impl(&self, texture: TextureHandle) -> Result<()> {
        let (image, extent, depth, mip_levels, array_layers, resting) = {
            let textures = self.textures.lock().unwrap();
            let tex = textures.get(texture).ok_or_else(|| {
                GpuError::Backend("generate_mipmaps: stale texture handle".to_string())
            })?;
            (
                tex.image,
                tex.extent,
                tex.depth,
                tex.mip_levels,
                tex.array_layers,
                tex.resting,
            )
        };
        if mip_levels < 2 {
            return Ok(());
        }

        let shader_stages = self.shader_stages();
        self.one_shot_submit(|dev, cmd| {
            let mut mip_width = extent.width as i32;
            let mut mip_height = extent.height as i32;
            let mut mip_depth = depth as i32;

            for level in 0..mip_levels - 1 {
                let next_width = (mip_width / 2).max(1);
                let next_height = (mip_height / 2).max(1);
                let next_depth = (mip_depth / 2).max(1);

                let to_src = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    ..Default::default()
                };
                let dst_to_transfer = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level + 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };
                unsafe {
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_src, dst_to_transfer],
                    );
                    dev.cmd_blit_image(
                        cmd,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageBlit {
                            src_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: level,
                                base_array_layer: 0,
                                layer_count: array_layers,
                            },
                            src_offsets: [
                                vk::Offset3D::default(),
                                vk::Offset3D {
                                    x: mip_width,
                                    y: mip_height,
                                    z: mip_depth,
                                },
                            ],
                            dst_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: level + 1,
                                base_array_layer: 0,
                                layer_count: array_layers,
                            },
                            dst_offsets: [
                                vk::Offset3D::default(),
                                vk::Offset3D {
                                    x: next_width,
                                    y: next_height,
                                    z: next_depth,
                                },
                            ],
                        }],
                        vk::Filter::LINEAR,
                    );
                }
                let src_to_shader = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: resting,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                    src_access_mask: vk::AccessFlags::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                };
                unsafe {
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        shader_stages,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[src_to_shader],
                    );
                }

                mip_width = next_width;
                mip_height = next_height;
                mip_depth = next_depth;
            }

            let last_to_shader = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: resting,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: array_layers,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    shader_stages,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[last_to_shader],
                );
            }
            Ok(())
        })?;
        if let Some(tex) = self.textures.lock().unwrap().get_mut(texture) {
            for layer in 0..array_layers {
                for mip in 0..mip_levels {
                    tex.mark_ready(mip, layer);
                }
            }
        }
        Ok(())
    }
}
