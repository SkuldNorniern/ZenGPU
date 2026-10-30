use std::sync::{Arc, Mutex};

use ash::vk;
use zengpu_hal::{
    Format, GpuDevice, GpuError, Result, SlotMap, TargetHandle, TextureDesc, TextureHandle,
    TextureUsage, marker,
};

use crate::VulkanDevice;
use crate::device::{VulkanDeviceInner, VulkanRenderTarget, VulkanTexture};

/// Fixed-size device-local color image, usable as a render target and a sampled texture. No raw
/// Vulkan types are public.
///
/// Render into `target_handle()` with `sample_after: true`, then bind `texture_handle()`. Drop it
/// before its `VulkanDevice`.
pub struct OffscreenTarget {
    inner: Arc<VulkanDeviceInner>,
    textures: Arc<Mutex<SlotMap<marker::Texture, VulkanTexture>>>,
    render_targets: Arc<Mutex<SlotMap<marker::RenderTarget, VulkanRenderTarget>>>,
    texture: TextureHandle,
    target: TargetHandle,
    format: Format,
    extent: vk::Extent2D,
}

unsafe impl Send for OffscreenTarget {}
unsafe impl Sync for OffscreenTarget {}

impl OffscreenTarget {
    /// Device-local render target. Starts in `UNDEFINED`: the first writing pass transitions
    /// it to `COLOR_ATTACHMENT_OPTIMAL`.
    pub fn new(device: &VulkanDevice, format: Format, width: u32, height: u32) -> Result<Self> {
        let texture = device.create_texture(TextureDesc {
            width,
            height,
            depth: 1,
            format,
            usage: TextureUsage::SAMPLED | TextureUsage::RENDER_TARGET | TextureUsage::TRANSFER_SRC,
            samples: 1,
            dimension: zengpu_hal::TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        })?;
        let target = device.register_color_target(texture).ok_or_else(|| {
            device.destroy_texture(texture);
            GpuError::Backend("register_color_target: stale texture handle".to_string())
        })?;
        Ok(Self {
            inner: Arc::clone(&device.inner),
            textures: Arc::clone(&device.textures),
            render_targets: Arc::clone(&device.render_targets),
            texture,
            target,
            format,
            extent: vk::Extent2D { width, height },
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    pub fn target_handle(&self) -> TargetHandle {
        self.target
    }

    /// Sampling handle: pass to `bind_texture` for a bindless slot.
    pub fn texture_handle(&self) -> TextureHandle {
        self.texture
    }

    /// Recreate at a new size. Handles from before this call (`texture_handle`, `target_handle`)
    /// are stale.
    pub fn resize(&mut self, device: &VulkanDevice, width: u32, height: u32) -> Result<()> {
        *self = Self::new(device, self.format, width, height)?;
        Ok(())
    }

    /// Crate-internal: frame-graph barrier tracking.
    pub(crate) fn image(&self) -> vk::Image {
        self.textures
            .lock()
            .unwrap()
            .get(self.texture)
            .map(|t| t.image)
            .unwrap_or(vk::Image::null())
    }

    /// Crate-internal: frame-graph barrier tracking.
    pub(crate) fn view(&self) -> vk::ImageView {
        self.textures
            .lock()
            .unwrap()
            .get(self.texture)
            .map(|t| t.view)
            .unwrap_or(vk::ImageView::null())
    }

    pub(crate) fn raw_format(&self) -> vk::Format {
        self.textures
            .lock()
            .unwrap()
            .get(self.texture)
            .map(|t| t.format)
            .unwrap_or(vk::Format::UNDEFINED)
    }

    pub(crate) fn raw_extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for OffscreenTarget {
    fn drop(&mut self) {
        self.render_targets.lock().unwrap().remove(self.target);
        if let Some(tex) = self.textures.lock().unwrap().remove(self.texture) {
            unsafe { tex.destroy(&self.inner.device) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VulkanInstance;
    use zengpu_hal::DeviceRequest;

    #[test]
    fn offscreen_target_resize() {
        let _guard = crate::test_gpu_lock();
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = inst.request_vulkan_adapter() else {
            return;
        };
        let Ok(device) = adapter.open_headless(DeviceRequest::default()) else {
            return;
        };

        let mut target = OffscreenTarget::new(&device, Format::Rgba8Unorm, 64, 32).unwrap();
        target.resize(&device, 128, 96).unwrap();
        assert_eq!(target.extent(), (128, 96));
        assert_eq!(target.format(), Format::Rgba8Unorm);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod zsl_parity_tests {
    //! Runs the same ZSL shaders and checks the same expected pixels as the `zengpu-metal`
    //! graphics tests.

    use crate::{OffscreenTarget, VulkanDevice, VulkanInstance};
    use zengpu_hal::{
        Bindings, BufferDesc, BufferUsage, ColorAttachment, ColorTargetState, DepthState,
        DeviceRequest, Format, GpuDevice, GraphicsDevice, GraphicsPipelineDesc, LoadOp,
        MemoryUsage, PipelineHandle, PrimitiveTopology, RasterState, RenderCommands,
        RenderPassDesc, SamplerDesc, Scalar, TexDim, TextureDesc, TextureHandle, TextureUsage,
        Viewport, ViewportScissor,
    };
    use zengpu_spirv::{ZslShader, zsl};

    fn open() -> Option<VulkanDevice> {
        VulkanInstance::new_with_surface()
            .ok()?
            .request_graphics_adapter()?
            .open_graphics(DeviceRequest::default())
            .ok()
    }

    fn pipeline(dev: &VulkanDevice, vs: &ZslShader, fs: &ZslShader) -> PipelineHandle {
        let v = dev
            .create_shader(vs.spirv_desc().unwrap())
            .expect("vertex shader");
        let f = dev
            .create_shader(fs.spirv_desc().unwrap())
            .expect("fragment shader");
        dev.create_graphics_pipeline(GraphicsPipelineDesc {
            vertex_shader: v,
            fragment_shader: f,
            vertex_layouts: &[],
            topology: PrimitiveTopology::TriangleList,
            color_targets: &[ColorTargetState {
                format: Format::Rgba8Unorm,
                blend: None,
            }],
            depth_format: None,
            depth: DepthState::default(),
            stencil: None,
            raster: RasterState::default(),
            samples: 1,
        })
        .expect("pipeline")
    }

    fn draw(
        dev: &VulkanDevice,
        target: &OffscreenTarget,
        p: PipelineHandle,
        bindings: Bindings<'_>,
    ) -> Vec<[u8; 4]> {
        let (w, h) = target.extent();
        let mut cmd = dev.create_command_list().expect("list");
        cmd.begin_render_pass(&RenderPassDesc {
            color: &[ColorAttachment {
                target: target.target_handle(),
                load: LoadOp::clear_rgb(0.0, 0.0, 0.0),
                store: true,
                sample_after: false,
            }],
            depth: None,
        });
        cmd.set_pipeline(p);
        cmd.set_viewport_scissor(ViewportScissor {
            viewport: Viewport {
                x: 0.0,
                y: 0.0,
                width: w as f32,
                height: h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: None,
        });
        cmd.bind(bindings);
        cmd.draw(0..6, 0..1);
        cmd.end_render_pass();
        dev.submit_and_wait(cmd).expect("submit");
        let len = u64::from(w * h * 4);
        let buf = dev
            .create_buffer(BufferDesc {
                size: len,
                usage: BufferUsage::READBACK | BufferUsage::TRANSFER_DST,
                memory: MemoryUsage::Readback,
            })
            .expect("readback buffer");
        dev.copy_offscreen_to_buffer(target, buf).expect("copy");
        let raw = dev.read_buffer(buf, 0, len).expect("read");
        raw.as_chunks::<4>().0.to_vec()
    }

    const RED_FS: ZslShader = zsl!(
        fragment fs() -> f32x4 { f32x4(1.0, 0.0, 0.0, 1.0) }
    );

    #[test]
    fn clip_space_y_points_down() {
        let _guard = crate::test_gpu_lock();
        const TOP_HALF: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> f32x4 {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                f32x4(corner.x * 2.0 - 1.0, corner.y - 1.0, 0.0, 1.0)
            }
        );
        let Some(dev) = open() else { return };
        let target = OffscreenTarget::new(&dev, Format::Rgba8Unorm, 4, 4).expect("target");
        let p = pipeline(&dev, &TOP_HALF, &RED_FS);
        let px = draw(&dev, &target, p, Bindings::default());
        for (i, p) in px.iter().enumerate() {
            let want = if i / 4 < 2 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            };
            assert_eq!(*p, want, "pixel {i} (row {})", i / 4);
        }
    }

    #[test]
    fn texture_indices_follow_scalars_in_the_push_block() {
        let _guard = crate::test_gpu_lock();
        const QUAD_VS: ZslShader = zsl!(
            vertex vs(vi: vertex_index) -> (f32x4, f32x2) {
                let corner = f32x2(f32((26 >> vi) & 1), f32((52 >> vi) & 1))
                (f32x4(corner * 2.0 - 1.0, 0.0, 1.0), corner)
            }
        );
        const TINTED_FS: ZslShader = zsl!(
            push P { tint: f32x2, slot: u32 }
            fragment fs(@location(0) uv: f32x2, tex: textures, p: P) -> f32x4 {
                tex[p.slot].sample(uv) * f32x4(p.tint.x, p.tint.y, 1.0, 1.0)
            }
        );
        let Some(dev) = open() else { return };
        let src: TextureHandle = dev
            .create_texture(TextureDesc {
                width: 1,
                height: 1,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .expect("texture");
        dev.upload_texture_data(src, &[255, 255, 255, 255])
            .expect("upload");
        let sampler = dev.create_sampler(SamplerDesc::default()).expect("sampler");
        let slot = dev.bind_texture(src, sampler).expect("slot");
        let target = OffscreenTarget::new(&dev, Format::Rgba8Unorm, 2, 2).expect("target");
        let p = pipeline(&dev, &QUAD_VS, &TINTED_FS);
        let scalars = [Scalar::F32(1.0), Scalar::F32(0.0)];
        let px = draw(
            &dev,
            &target,
            p,
            Bindings {
                buffers: &[],
                textures: &[slot],
                images: &[],
                scalars: &scalars,
            },
        );
        assert!(px.iter().all(|p| *p == [255, 0, 255, 255]), "{px:?}");
    }
}
