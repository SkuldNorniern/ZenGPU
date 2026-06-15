use std::sync::Arc;

use ash::vk;
use zengpu_hal::{GpuError, Result};

use crate::device::VulkanDeviceInner;
use crate::swapchain::DeviceContext;

/// The depth format used by [`DepthTarget`].
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Device-local depth image for use as a depth-stencil attachment.
///
/// Create one per swapchain size; rebuild on resize. Register with
/// [`FrameGraph::add_depth_resource`](crate::frame_graph::FrameGraph::add_depth_resource)
/// and declare as [`AttachmentUsage::DepthWrite`](crate::frame_graph::AttachmentUsage::DepthWrite)
/// in the pass that writes depth.
///
/// Drop ordering: place before the [`Swapchain`](crate::swapchain::Swapchain) field in the
/// consumer struct so depth resources are freed before the device goes away.
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
                    extent: vk::Extent3D { width, height, depth: 1 },
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

        unsafe {
            dev.bind_image_memory(image, memory, 0)
                .map_err(|e| GpuError::Backend(format!("bind depth memory: {e}")))?;
        }

        let view = unsafe {
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
        .map_err(|e| GpuError::Backend(format!("create depth view: {e}")))?;

        Ok(Self { inner, image, memory, view, extent })
    }

    pub fn format(&self) -> zengpu_hal::Format {
        zengpu_hal::Format::Depth32Float
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// Raw Vulkan image view for render pass and FrameGraph setup.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Raw Vulkan image for FrameGraph barrier tracking.
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

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    (0..props.memory_type_count)
        .find(|&i| {
            (type_filter & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(required)
        })
        .ok_or_else(|| GpuError::Backend("no device-local memory type for depth".to_string()))
}
