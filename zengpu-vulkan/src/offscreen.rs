use std::sync::{Arc, Mutex};

use ash::vk;
use zengpu_hal::{
    Format, GpuDevice, GpuError, Result, SlotMap, TargetHandle, TextureDesc, TextureHandle,
    TextureUsage, marker,
};

use crate::VulkanDevice;
use crate::device::{VulkanDeviceInner, VulkanRenderTarget, VulkanTexture};

/// Fixed-size, device-local color image usable as a render target and as a
/// sampled texture. Built on top of the HAL [`TextureHandle`]/[`TargetHandle`]
/// pair — no raw Vulkan types are exposed publicly.
///
/// Typical use:
/// 1. Render into this target: `ColorAttachment { target: offscreen.target_handle(), ... }`
/// 2. Set `sample_after: true` on the attachment to transition to shader-readable.
/// 3. Sample it: `device.bind_texture(offscreen.texture_handle(), sampler) -> slot`
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
    /// Allocate a device-local render target of `format` × `width` × `height`.
    /// The image starts in `UNDEFINED` layout; the first render pass that writes
    /// it will transition it to `COLOR_ATTACHMENT_OPTIMAL`.
    pub fn new(device: &VulkanDevice, format: Format, width: u32, height: u32) -> Result<Self> {
        let texture = device.create_texture(TextureDesc {
            width,
            height,
            format,
            usage: TextureUsage::SAMPLED | TextureUsage::RENDER_TARGET | TextureUsage::TRANSFER_SRC,
            samples: 1,
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

    /// HAL handle for use as a [`zengpu_hal::ColorAttachment::target`].
    pub fn target_handle(&self) -> TargetHandle {
        self.target
    }

    /// HAL handle for sampling: pass to [`VulkanDevice::bind_texture`] to get
    /// a bindless slot, then use that slot in [`zengpu_hal::Bindings::textures`].
    pub fn texture_handle(&self) -> TextureHandle {
        self.texture
    }

    /// Recreate the offscreen target at a new size, replacing the existing image.
    ///
    /// All handles returned before this call (`texture_handle`, `target_handle`) become stale
    /// and must not be used after the call returns.
    pub fn resize(&mut self, device: &VulkanDevice, width: u32, height: u32) -> Result<()> {
        *self = Self::new(device, self.format, width, height)?;
        Ok(())
    }

    /// Raw Vulkan image — crate-internal access for frame-graph barrier tracking.
    pub(crate) fn image(&self) -> vk::Image {
        self.textures
            .lock()
            .unwrap()
            .get(self.texture)
            .map(|t| t.image)
            .unwrap_or(vk::Image::null())
    }

    /// Raw Vulkan image view — crate-internal access for frame-graph barrier tracking.
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
    use crate::{VulkanDevice, VulkanInstance};
    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};

    #[test]
    fn offscreen_target_handles_are_valid() {
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let Ok(dev) = adapter.open(DeviceRequest::default()) else {
            return;
        };
        let Some(device) = dev.as_any().downcast_ref::<VulkanDevice>() else {
            return;
        };

        let target = OffscreenTarget::new(device, Format::Rgba8Unorm, 64, 32).unwrap();
        assert_eq!(target.format(), Format::Rgba8Unorm);
        assert_eq!(target.extent(), (64, 32));
        let _ = target.texture_handle();
        let _ = target.target_handle();
    }

    #[test]
    fn offscreen_target_resize() {
        let Ok(inst) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = inst.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let Ok(dev) = adapter.open(DeviceRequest::default()) else {
            return;
        };
        let Some(device) = dev.as_any().downcast_ref::<VulkanDevice>() else {
            return;
        };

        let mut target = OffscreenTarget::new(device, Format::Rgba8Unorm, 64, 32).unwrap();
        target.resize(device, 128, 96).unwrap();
        assert_eq!(target.extent(), (128, 96));
        assert_eq!(target.format(), Format::Rgba8Unorm);
    }
}
