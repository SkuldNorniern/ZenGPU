//! Barrier builders. Sites give the stages they order and the resource when they know it.

use ash::{Device, vk};

/// Stages that may use a buffer after a write, with exact access. Graphics stages only when
/// `graphics`: a compute-only queue family rejects them (VUID-vkCmdPipelineBarrier-dstStageMask-06462).
pub(crate) fn buffer_users(graphics: bool) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let mut stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    let mut access = vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::UNIFORM_READ
        | vk::AccessFlags::TRANSFER_READ
        | vk::AccessFlags::TRANSFER_WRITE;
    if graphics {
        stages |= vk::PipelineStageFlags::DRAW_INDIRECT
            | vk::PipelineStageFlags::VERTEX_INPUT
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER;
        access |= vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::INDEX_READ
            | vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    (stages, access)
}

/// Stages that may use an image after a write. Access stays broad because layouts vary.
pub(crate) fn image_users(graphics: bool) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let mut stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    if graphics {
        stages |= vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    }
    (
        stages,
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
    )
}

/// Buffer barrier for `offset..offset + size` of `buffer`.
pub(crate) fn buffer(
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::BufferMemoryBarrier<'static> {
    vk::BufferMemoryBarrier {
        src_access_mask: src,
        dst_access_mask: dst,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer,
        offset,
        size,
        ..Default::default()
    }
}

/// Image barrier for one subresource, with a layout change.
pub(crate) fn image(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: range,
        src_access_mask: src,
        dst_access_mask: dst,
        ..Default::default()
    }
}

/// Records one barrier call with the given stages and barriers.
///
/// # Safety
/// `cmd` must be recording on `dev`.
pub(crate) unsafe fn submit(
    dev: &Device,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    memory: &[vk::MemoryBarrier<'_>],
    buffers: &[vk::BufferMemoryBarrier<'_>],
    images: &[vk::ImageMemoryBarrier<'_>],
) {
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            memory,
            buffers,
            images,
        );
    }
}
