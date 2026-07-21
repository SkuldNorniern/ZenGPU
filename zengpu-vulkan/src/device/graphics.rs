use super::*;

impl VulkanDevice {
    /// Create a graphics pipeline via `VK_KHR_dynamic_rendering` — no
    /// `vk::RenderPass`/`vk::Framebuffer` objects. Part of the unified
    /// graphics API (D17/GU); see [`zengpu_hal::GraphicsDevice::create_graphics_pipeline`].
    pub(crate) fn create_graphics_pipeline_impl(
        &self,
        desc: GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle> {
        if desc.raster.polygon != PolygonMode::Fill && !self.inner.fill_mode_non_solid {
            return Err(GpuError::PipelineCreation(
                "PolygonMode::Line/Point requires fillModeNonSolid, which this device does not support".to_string(),
            ));
        }
        log::debug!("[zengpu-vulkan] create_graphics_pipeline: resolve shaders");
        let (vert_module, frag_module) = {
            let shaders = self.shaders.lock().unwrap();
            let vert = *shaders.get(desc.vertex_shader).ok_or_else(|| {
                GpuError::PipelineCreation("stale vertex shader handle".to_string())
            })?;
            let frag = *shaders.get(desc.fragment_shader).ok_or_else(|| {
                GpuError::PipelineCreation("stale fragment shader handle".to_string())
            })?;
            (vert, frag)
        };
        log::debug!("[zengpu-vulkan] create_graphics_pipeline: build stages");
        let entry = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
        ];

        // One binding per vertex layout; binding index = slice position, which
        // is the `slot` passed to set_vertex_buffer. Attributes carry the binding
        // of the layout they belong to so multiple streams (e.g. per-vertex quad
        // + per-instance data) coexist.
        let bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_layouts
            .iter()
            .enumerate()
            .map(|(i, layout)| vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: layout.stride,
                input_rate: step_mode_to_vk(layout.step_mode),
            })
            .collect();
        let attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_layouts
            .iter()
            .enumerate()
            .flat_map(|(i, layout)| {
                layout
                    .attributes
                    .iter()
                    .map(move |a| vk::VertexInputAttributeDescription {
                        location: a.location,
                        binding: i as u32,
                        format: vertex_format_to_vk(a.format),
                        offset: a.offset,
                    })
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: bindings.len() as u32,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: attributes.len() as u32,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: topology_to_vk(desc.topology),
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterization = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: polygon_mode_to_vk(desc.raster.polygon),
            cull_mode: cull_mode_to_vk(desc.raster.cull),
            front_face: front_face_to_vk(desc.raster.front_face),
            line_width: 1.0,
            ..Default::default()
        };

        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: sample_count_to_vk(desc.samples),
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: if desc.depth.test { vk::TRUE } else { vk::FALSE },
            depth_write_enable: if desc.depth.write {
                vk::TRUE
            } else {
                vk::FALSE
            },
            depth_compare_op: compare_fn_to_vk(desc.depth.compare),
            ..Default::default()
        };

        let blend_att = blend_mode_to_vk(desc.blend);
        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &blend_att,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let color_format = hal_format_to_vk(desc.color_format);
        let depth_format = desc
            .depth_format
            .map(hal_format_to_vk)
            .unwrap_or(vk::Format::UNDEFINED);
        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 1,
            p_color_attachment_formats: &color_format,
            depth_attachment_format: depth_format,
            ..Default::default()
        };

        log::debug!("[zengpu-vulkan] create_graphics_pipeline: create_pipeline_layout");
        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 256, // 64 u32 slots: scalars + buffer/texture indices
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

        let create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &rendering_info as *const _ as *const c_void,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout,
            ..Default::default()
        };

        log::debug!("[zengpu-vulkan] create_graphics_pipeline: vkCreateGraphicsPipelines");
        let result = unsafe {
            self.inner
                .device
                .create_graphics_pipelines(self.pipeline_cache, &[create_info], None)
        };
        log::debug!("[zengpu-vulkan] create_graphics_pipeline: vkCreateGraphicsPipelines returned");
        match result {
            Ok(pipelines) => Ok(self
                .pipelines
                .lock()
                .unwrap()
                .insert(VulkanPipeline::Graphics {
                    layout,
                    pipeline: pipelines[0],
                })),
            Err((_, e)) => {
                unsafe {
                    self.inner.device.destroy_pipeline_layout(layout, None);
                }
                Err(GpuError::PipelineCreation(format!(
                    "vkCreateGraphicsPipelines: {e}"
                )))
            }
        }
    }

    /// Register `depth`'s image/view as a render target for use as
    /// [`zengpu_hal::DepthAttachment::target`]. Call again with the recreated
    /// [`crate::DepthTarget`] after a resize and use [`unregister_render_target`](Self::unregister_render_target)
    /// to drop the stale handle.
    pub fn register_depth_target(&self, depth: &DepthTarget) -> TargetHandle {
        let (width, height) = depth.extent();
        self.render_targets
            .lock()
            .unwrap()
            .insert(VulkanRenderTarget {
                image: depth.image(),
                view: depth.view(),
                format: DEPTH_FORMAT,
                extent: vk::Extent2D { width, height },
                layout: vk::ImageLayout::UNDEFINED,
            })
    }

    /// Register `texture`'s image/view as a render target for use as
    /// [`zengpu_hal::ColorAttachment::target`]. `texture` must have been
    /// created with [`TextureUsage::RENDER_TARGET`]. Use
    /// [`unregister_render_target`](Self::unregister_render_target) to drop
    /// the handle when the texture is destroyed. Returns `None` for a stale
    /// `texture` handle.
    pub fn register_color_target(&self, texture: TextureHandle) -> Option<TargetHandle> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(texture)?;
        Some(
            self.render_targets
                .lock()
                .unwrap()
                .insert(VulkanRenderTarget {
                    image: tex.image,
                    view: tex.view,
                    format: tex.format,
                    extent: tex.extent,
                    layout: vk::ImageLayout::UNDEFINED,
                }),
        )
    }

    /// Drop a render-target registration created by [`register_depth_target`](Self::register_depth_target)
    /// or [`register_color_target`](Self::register_color_target).
    pub fn unregister_render_target(&self, handle: TargetHandle) {
        self.render_targets.lock().unwrap().remove(handle);
    }
}

impl GraphicsDevice for VulkanDevice {
    type Surface = VulkanSurface;
    type CommandList = VulkanCommandList;

    fn create_surface(
        &self,
        window: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self::Surface> {
        VulkanSurface::new(self, window, config)
    }

    fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc<'_>) -> Result<PipelineHandle> {
        self.create_graphics_pipeline_impl(desc)
    }

    fn create_command_list(&self) -> Result<Self::CommandList> {
        self.create_command_list_impl()
    }

    fn supports_dual_source_blending(&self) -> bool {
        self.inner.dual_src_blend
    }

    fn supports_non_solid_fill(&self) -> bool {
        self.inner.fill_mode_non_solid
    }
}
