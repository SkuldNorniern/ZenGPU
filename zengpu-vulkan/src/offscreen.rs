use std::sync::Arc;

use ash::vk;
use zengpu_hal::{Format, GpuError, Result};

use crate::device::VulkanDeviceInner;
use crate::swapchain::DeviceContext;
use crate::{from_vk_format, to_vk_format};

/// Fixed-size, device-local color image, usable as render target and sampled texture.
///
/// The consuming pass should set `finalLayout = SHADER_READ_ONLY_OPTIMAL`. Use an explicit barrier
/// between the two passes:
///
/// ```text
/// srcStage = COLOR_ATTACHMENT_OUTPUT
/// dstStage = FRAGMENT_SHADER
/// srcAccess = COLOR_ATTACHMENT_WRITE
/// dstAccess = SHADER_READ
/// old/new layout = SHADER_READ_ONLY_OPTIMAL
/// ```
///
/// The image starts in `UNDEFINED`. The offscreen pass uses `initialLayout = UNDEFINED`, `loadOp =
/// CLEAR`, and `finalLayout = SHADER_READ_ONLY_OPTIMAL`.
pub struct OffscreenTarget {
    inner: Arc<VulkanDeviceInner>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
}

/// Sampled-image view of a render target. [`Self::raw`] for descriptor binding,
/// [`Self::belongs_to`] for same-device checks.
#[derive(Clone, Copy)]
pub struct SampledImageView<'a> {
    pub(crate) inner: &'a Arc<VulkanDeviceInner>,
    pub(crate) view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl SampledImageView<'_> {
    pub fn format(&self) -> Format {
        from_vk_format(self.format).expect("SampledImageView has unsupported vk::Format")
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    pub fn raw(&self) -> vk::ImageView {
        self.view
    }

    pub fn belongs_to(&self, context: &DeviceContext) -> bool {
        Arc::ptr_eq(self.inner, &context.inner)
    }
}

unsafe impl Send for OffscreenTarget {}
unsafe impl Sync for OffscreenTarget {}

impl OffscreenTarget {
    /// Image usage is `COLOR_ATTACHMENT | SAMPLED`. It starts in `UNDEFINED`, and the first render
    /// pass transitions it.
    pub fn new(ctx: &DeviceContext, format: Format, width: u32, height: u32) -> Result<Self> {
        let vk_fmt = to_vk_format(format);
        let inner = ctx.inner_arc();
        let dev = &inner.device;
        let extent = vk::Extent2D { width, height };

        let image = unsafe {
            dev.create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk_fmt,
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| GpuError::Backend(format!("create_image: {e}")))?;

        let reqs = unsafe { dev.get_image_memory_requirements(image) };
        let mem_type = find_memory_type(
            &ctx.memory_properties(),
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let memory = unsafe {
            dev.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| GpuError::Backend(format!("allocate_memory: {e}")))?;

        unsafe {
            dev.bind_image_memory(image, memory, 0)
                .map_err(|e| GpuError::Backend(format!("bind_image_memory: {e}")))?;
        }

        let view = unsafe {
            dev.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk_fmt,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| GpuError::Backend(format!("create_image_view: {e}")))?;

        Ok(Self {
            inner,
            image,
            memory,
            view,
            format: vk_fmt,
            extent,
        })
    }

    pub fn format(&self) -> Format {
        from_vk_format(self.format).expect("OffscreenTarget has unsupported vk::Format")
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn raw_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn raw_format(&self) -> vk::Format {
        self.format
    }

    /// The target must belong to the same logical device. It must stay alive and in
    /// `SHADER_READ_ONLY_OPTIMAL` while the descriptor is used.
    pub fn sampled_view(&self) -> SampledImageView<'_> {
        SampledImageView {
            inner: &self.inner,
            view: self.view,
            format: self.format,
            extent: self.extent,
        }
    }
}

impl Drop for OffscreenTarget {
    fn drop(&mut self) {
        let dev = &self.inner.device;
        unsafe {
            dev.destroy_image_view(self.view, None);
            dev.destroy_image(self.image, None);
            dev.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    (0..props.memory_type_count)
        .find(|&i| {
            (type_filter & (1 << i)) != 0
                && props.memory_types[i as usize]
                    .property_flags
                    .contains(required)
        })
        .ok_or_else(|| GpuError::Backend("no device-local memory type found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VulkanDevice, VulkanInstance};
    use zengpu_hal::{AdapterRequest, DeviceRequest, GpuInstance};

    #[test]
    fn offscreen_target_exposes_sampled_view_metadata() {
        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Some(adapter) = instance.request_adapter(AdapterRequest::default()) else {
            return;
        };
        let Ok(device) = adapter.open(DeviceRequest::default()) else {
            return;
        };
        let device = device
            .as_any()
            .downcast_ref::<VulkanDevice>()
            .expect("Vulkan adapter returned a non-Vulkan device");

        let target =
            OffscreenTarget::new(&device.context(), zengpu_hal::Format::Rgba8Unorm, 64, 32)
                .unwrap();
        let sampled = target.sampled_view();

        assert_eq!(sampled.format(), zengpu_hal::Format::Rgba8Unorm);
        assert_eq!(sampled.extent(), (64, 32));
    }
}
