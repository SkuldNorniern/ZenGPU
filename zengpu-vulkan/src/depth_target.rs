use std::sync::Arc;

use ash::{Device, vk};
use zengpu_hal::{GpuError, Result};

use crate::device::VulkanDeviceInner;
use crate::swapchain::DeviceContext;

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Device-local depth image. One per swapchain size, rebuilt on resize. Register it
/// with `FrameGraph::add_depth_resource`.
///
/// Declare it before the `Swapchain` field so it is freed before the device.
pub struct DepthTarget {
    inner: Arc<VulkanDeviceInner>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    extent: vk::Extent2D,
}

unsafe impl Send for DepthTarget {}
unsafe impl Sync for DepthTarget {}

impl DepthTarget {
    /// Allocate a `D32_SFLOAT` depth image of `width` × `height`.
    pub fn new(ctx: &DeviceContext, width: u32, height: u32) -> Result<Self> {
        let inner = ctx.inner_arc();
        let dev = &inner.device;
        let extent = vk::Extent2D { width, height };

        let image = unsafe {
            dev.create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format: DEPTH_FORMAT,
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| GpuError::Backend(format!("create depth image: {e}")))?;

        let (memory, view) = match back_image(&inner, image, ctx) {
            Ok(done) => done,
            Err(e) => {
                unsafe { dev.destroy_image(image, None) };
                return Err(e);
            }
        };

        Ok(Self {
            inner,
            image,
            memory,
            view,
            extent,
        })
    }

    /// Recreate at a new size. Views from `view()` taken before this call are invalid after it.
    pub fn resize(&mut self, ctx: &DeviceContext, width: u32, height: u32) -> Result<()> {
        *self = Self::new(ctx, width, height)?;
        Ok(())
    }

    pub fn format(&self) -> zengpu_hal::Format {
        zengpu_hal::Format::Depth32Float
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// For FrameGraph barrier tracking.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn raw_extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for DepthTarget {
    fn drop(&mut self) {
        let dev = &self.inner.device;
        unsafe {
            dev.destroy_image_view(self.view, None);
            dev.destroy_image(self.image, None);
            dev.free_memory(self.memory, None);
        }
    }
}

fn back_image(
    inner: &VulkanDeviceInner,
    image: vk::Image,
    ctx: &DeviceContext,
) -> Result<(vk::DeviceMemory, vk::ImageView)> {
    let dev = &inner.device;
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
    .map_err(|e| GpuError::Backend(format!("allocate depth memory: {e}")))?;

    match view_on(dev, image, memory) {
        Ok(view) => Ok((memory, view)),
        Err(e) => {
            unsafe { dev.free_memory(memory, None) };
            Err(e)
        }
    }
}

fn view_on(dev: &Device, image: vk::Image, memory: vk::DeviceMemory) -> Result<vk::ImageView> {
    unsafe {
        dev.bind_image_memory(image, memory, 0)
            .map_err(|e| GpuError::Backend(format!("bind depth memory: {e}")))?;
    }

    unsafe {
        dev.create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: DEPTH_FORMAT,
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
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
    .map_err(|e| GpuError::Backend(format!("create depth view: {e}")))
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
        .ok_or_else(|| GpuError::Backend("no device-local memory type for depth".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VulkanInstance, swapchain::DeviceContext};
    use zengpu_hal::DeviceRequest;

    fn make_device_and_ctx() -> Option<(crate::VulkanDevice, DeviceContext)> {
        let inst = VulkanInstance::new().ok()?;
        let adapter = inst.request_vulkan_adapter()?;
        let dev = adapter.open_headless(DeviceRequest::default()).ok()?;
        let ctx = dev.context();
        Some((dev, ctx))
    }

    #[test]
    fn depth_target_resize() {
        let _guard = crate::test_gpu_lock();
        let Some((_dev, ctx)) = make_device_and_ctx() else {
            return;
        };
        let mut depth = DepthTarget::new(&ctx, 64, 32).unwrap();
        assert_eq!(depth.extent(), (64, 32));
        depth.resize(&ctx, 128, 96).unwrap();
        assert_eq!(depth.extent(), (128, 96));
    }
}
