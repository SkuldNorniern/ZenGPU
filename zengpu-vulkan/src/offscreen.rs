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
            unsafe {
                self.inner.device.destroy_image_view(tex.view, None);
                self.inner.device.destroy_image(tex.image, None);
                self.inner.device.free_memory(tex.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VulkanInstance;
    use zengpu_hal::DeviceRequest;

    #[test]
    fn offscreen_target_handles_are_valid() {
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

        let target = OffscreenTarget::new(&device, Format::Rgba8Unorm, 64, 32).unwrap();
        assert_eq!(target.format(), Format::Rgba8Unorm);
        assert_eq!(target.extent(), (64, 32));
        let _ = target.texture_handle();
        let _ = target.target_handle();
    }

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
