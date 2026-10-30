//! Moves texels between host, buffers and texture parts without waiting for the GPU.
//!
//! Host data goes through one reused staging ring. A span returns once the submission reading it
//! finishes; the ring grows when a write does not fit. Busy-ring overflow uses a bounded buffer
//! pool. Each copy ends with a barrier for whatever reads next.

use std::collections::VecDeque;
use std::ptr::copy_nonoverlapping;

use ash::{Device, vk};
use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, GpuDevice, GpuError, MemoryUsage, Result, Submission,
    TextureHandle, TextureRegion, TextureWrite, UsageError,
};

use super::VulkanDevice;

/// Ring starts at this size and doubles up to `RING_MAX`.
const RING_START: u64 = 4 << 20;
/// Writes bigger than the ring get their own buffer.
const RING_MAX: u64 = 64 << 20;
/// Ring offsets are kept below this, which every copy accepts.
const RING_ALIGN: u64 = 16;
/// Bound retained overflow memory even after a burst of asynchronous uploads.
const SPARE_MAX_BYTES: u64 = 16 << 20;
const SPARE_MAX_BUFFERS: usize = 32;

#[derive(Clone, Copy, Debug)]
struct Span {
    start: u64,
    serial: u64,
}

#[derive(Default)]
pub(crate) struct StagingRing {
    buffer: Option<(BufferHandle, u64)>,
    head: u64,
    in_use: VecDeque<Span>,
    spares: Vec<Spare>,
}

struct Spare {
    buffer: BufferHandle,
    size: u64,
    serial: u64,
}

impl StagingRing {
    fn take_spare(&mut self, len: u64, completed: u64) -> Option<BufferHandle> {
        let spare = self
            .spares
            .iter_mut()
            .filter(|s| s.serial != PENDING && s.serial <= completed && s.size >= len)
            .min_by_key(|s| s.size)?;
        spare.serial = PENDING;
        Some(spare.buffer)
    }

    fn can_keep_spare(&self, size: u64) -> bool {
        self.spares.len() < SPARE_MAX_BUFFERS
            && size <= SPARE_MAX_BYTES
            && self.spares.iter().map(|s| s.size).sum::<u64>() <= SPARE_MAX_BYTES - size
    }

    fn settle_spare(&mut self, buffer: BufferHandle, serial: u64) -> bool {
        let Some(spare) = self.spares.iter_mut().find(|s| s.buffer == buffer) else {
            return false;
        };
        spare.serial = serial;
        true
    }

    fn reclaim(&mut self, completed: u64) {
        while self
            .in_use
            .front()
            .is_some_and(|span| span.serial <= completed)
        {
            self.in_use.pop_front();
        }
        if self.in_use.is_empty() {
            self.head = 0;
        }
    }

    fn place(&self, len: u64, size: u64) -> Option<u64> {
        let start = self.head.next_multiple_of(RING_ALIGN);
        let Some(oldest) = self.in_use.front() else {
            return (len <= size).then_some(0);
        };
        if oldest.start < self.head {
            if start + len <= size {
                return Some(start);
            }
            return (len <= oldest.start).then_some(0);
        }
        (start + len <= oldest.start).then_some(start)
    }

    fn take(&mut self, start: u64, len: u64, serial: u64) {
        self.head = start + len;
        self.in_use.push_back(Span { start, serial });
    }

    /// Ties the span at `start` to the submission that reads it. Serial `0` returns it at once, for
    /// a write never submitted.
    fn settle(&mut self, start: u64, serial: u64) {
        if let Some(span) = self
            .in_use
            .iter_mut()
            .find(|span| span.start == start && span.serial == PENDING)
        {
            span.serial = serial;
        }
    }
}

/// Serial of a span reserved for a submission not yet made: never completes until settled.
const PENDING: u64 = u64::MAX;

struct Placed {
    image: vk::Image,
    resting: vk::ImageLayout,
    region: TextureRegion,
    was_ready: bool,
    offset: u64,
    row_texels: u32,
}

impl VulkanDevice {
    /// Releases idle upload staging buffers without waiting for GPU work. Returns
    /// their buffer capacities in bytes (the driver allocation can be larger).
    /// Busy spans and reservations currently being filled remain owned until a
    /// later call. The next upload lazily allocates staging again.
    pub fn trim_staging_memory(&self) -> u64 {
        self.reap_orphans();
        let completed = self.inner.serials.completed();
        let mut ring = self.staging.lock().unwrap();
        ring.reclaim(completed);
        let mut released = Vec::new();
        if ring.in_use.is_empty()
            && let Some(buffer) = ring.buffer.take()
        {
            released.push(buffer);
        }
        ring.spares.retain(|spare| {
            if spare.serial != PENDING && spare.serial <= completed {
                released.push((spare.buffer, spare.size));
                false
            } else {
                true
            }
        });
        drop(ring);
        let bytes = released.iter().map(|(_, size)| size).sum();
        for (buffer, _) in released {
            self.destroy_buffer(buffer);
        }
        bytes
    }

    pub(crate) fn write_textures_impl(&self, writes: &[TextureWrite<'_>]) -> Result<Submission> {
        if writes.is_empty() {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                "write_textures needs at least one write".into(),
            )));
        }
        let mut sizes = Vec::with_capacity(writes.len());
        for write in writes {
            sizes.push(self.check_write(write)?);
        }
        let total: u64 = sizes.iter().map(|s| s.next_multiple_of(RING_ALIGN)).sum();
        let (staging, base, own_buffer) = self.stage(total)?;
        let submitted = self.fill_and_submit(writes, &sizes, (staging, base));
        let serial = submitted.as_ref().map_or(0, |(_, serial)| *serial);
        if own_buffer {
            let kept = self.staging.lock().unwrap().settle_spare(staging, serial);
            if !kept {
                self.destroy_buffer(staging);
            }
        } else {
            self.staging.lock().unwrap().settle(base, serial);
        }
        let (submission, _) = submitted?;
        for write in writes {
            self.mark_ready(write.texture, write.region);
        }
        Ok(submission)
    }

    fn fill_and_submit(
        &self,
        writes: &[TextureWrite<'_>],
        sizes: &[u64],
        (staging, base): (BufferHandle, u64),
    ) -> Result<(Submission, u64)> {
        let mut placed = Vec::with_capacity(writes.len());
        let mut at = base;
        for (write, size) in writes.iter().zip(sizes) {
            self.copy_rows_into(staging, at, write)?;
            let (image, resting) = self.image_of(write.texture)?;
            placed.push(Placed {
                image,
                resting,
                region: write.region,
                was_ready: self.is_ready(write.texture, write.region),
                offset: at,
                row_texels: write.region.width,
            });
            at += size.next_multiple_of(RING_ALIGN);
        }
        let staging_vk = self.vk_buffer(staging)?;
        self.one_shot_submit_tracked(0, &[staging.index()], |dev, cmd| {
            record_buffer_to_images(dev, cmd, staging_vk, &placed);
            Ok(())
        })
    }

    pub(crate) fn copy_buffer_to_texture_impl(
        &self,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
        texture: TextureHandle,
        region: TextureRegion,
    ) -> Result<Submission> {
        let texel = self.texel_bytes(texture)?;
        self.check_region(texture, region, vk::ImageUsageFlags::TRANSFER_DST)?;
        let row_texels = rows_in_texels(bytes_per_row, region.width, texel)?;
        let span = span_bytes(region, bytes_per_row, texel);
        self.check_buffer(buffer, offset, span, BufferUsage::TRANSFER_SRC)?;
        let (image, resting) = self.image_of(texture)?;
        let placed = [Placed {
            image,
            resting,
            region,
            was_ready: self.is_ready(texture, region),
            offset,
            row_texels,
        }];
        let vk_buffer = self.vk_buffer(buffer)?;
        let (submission, _) = self.one_shot_submit_tracked(0, &[buffer.index()], |dev, cmd| {
            memory_barrier(
                dev,
                cmd,
                (
                    vk::AccessFlags::MEMORY_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                ),
            );
            record_buffer_to_images(dev, cmd, vk_buffer, &placed);
            Ok(())
        })?;
        self.mark_ready(texture, region);
        Ok(submission)
    }

    pub(crate) fn copy_texture_region_to_buffer_impl(
        &self,
        texture: TextureHandle,
        region: TextureRegion,
        buffer: BufferHandle,
        offset: u64,
        bytes_per_row: u32,
    ) -> Result<Submission> {
        let texel = self.texel_bytes(texture)?;
        self.check_region(texture, region, vk::ImageUsageFlags::TRANSFER_SRC)?;
        let row_texels = rows_in_texels(bytes_per_row, region.width, texel)?;
        let span = span_bytes(region, bytes_per_row, texel);
        self.check_buffer(buffer, offset, span, BufferUsage::TRANSFER_DST)?;
        let (image, resting) = self.image_of(texture)?;
        let was_ready = self.is_ready(texture, region);
        let vk_buffer = self.vk_buffer(buffer)?;
        let (submission, _) = self.one_shot_submit_tracked(0, &[buffer.index()], |dev, cmd| {
            let from = if was_ready {
                resting
            } else {
                vk::ImageLayout::UNDEFINED
            };
            image_barrier(
                dev,
                cmd,
                image,
                region,
                (from, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            );
            unsafe {
                dev.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk_buffer,
                    &[buffer_image_copy(region, offset, row_texels)],
                );
            }
            let back = (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, resting);
            image_barrier(dev, cmd, image, region, back);
            memory_barrier(
                dev,
                cmd,
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ,
                ),
            );
            Ok(())
        })?;
        self.mark_ready(texture, region);
        Ok(submission)
    }

    /// Bytes a write takes in staging, rows packed.
    fn check_write(&self, write: &TextureWrite<'_>) -> Result<u64> {
        let texel = self.texel_bytes(write.texture)?;
        self.check_region(
            write.texture,
            write.region,
            vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let row = u64::from(write.region.width) * u64::from(texel);
        let need = span_bytes(write.region, write.bytes_per_row, texel);
        if u64::from(write.bytes_per_row) < row || (write.data.len() as u64) < need {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!(
                    "texture write of {}x{} texels needs rows of {row} bytes and {need} bytes of data, got rows of {} and {}",
                    write.region.width,
                    write.region.height,
                    write.bytes_per_row,
                    write.data.len()
                ),
            )));
        }
        Ok(row * u64::from(write.region.height))
    }

    fn check_region(
        &self,
        texture: TextureHandle,
        region: TextureRegion,
        needed: vk::ImageUsageFlags,
    ) -> Result<()> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(texture).ok_or_else(stale_texture)?;
        if !tex.usage.contains(needed) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "texture",
                needed: if needed == vk::ImageUsageFlags::TRANSFER_DST {
                    "TRANSFER_DST"
                } else {
                    "TRANSFER_SRC"
                },
            }));
        }
        let mip_w = tex
            .extent
            .width
            .checked_shr(region.mip_level)
            .unwrap_or(0)
            .max(1);
        let mip_h = tex
            .extent
            .height
            .checked_shr(region.mip_level)
            .unwrap_or(0)
            .max(1);
        let inside = region.mip_level < tex.mip_levels
            && region.layer < tex.array_layers
            && region.width > 0
            && region.height > 0
            && region
                .x
                .checked_add(region.width)
                .is_some_and(|r| r <= mip_w)
            && region
                .y
                .checked_add(region.height)
                .is_some_and(|b| b <= mip_h);
        if inside {
            Ok(())
        } else {
            Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!(
                    "region {region:?} is outside the {mip_w}x{mip_h} mip {} of a texture with {} mips and {} layers",
                    region.mip_level, tex.mip_levels, tex.array_layers
                ),
            )))
        }
    }

    fn check_buffer(
        &self,
        buffer: BufferHandle,
        offset: u64,
        len: u64,
        usage: BufferUsage,
    ) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(buffer)
            .ok_or_else(|| GpuError::Backend("stale buffer handle".into()))?;
        if !buf.usage.contains(usage) {
            return Err(GpuError::InvalidUsage(UsageError::MissingUsage {
                resource: "buffer",
                needed: if usage == BufferUsage::TRANSFER_SRC {
                    "TRANSFER_SRC"
                } else {
                    "TRANSFER_DST"
                },
            }));
        }
        if offset.checked_add(len).is_none_or(|end| end > buf.size) {
            return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
                format!("{len} bytes at {offset} run past a buffer of {}", buf.size),
            )));
        }
        Ok(())
    }

    fn texel_bytes(&self, texture: TextureHandle) -> Result<u32> {
        let textures = self.textures.lock().unwrap();
        textures
            .get(texture)
            .map(|tex| tex.texel_bytes)
            .ok_or_else(stale_texture)
    }

    fn image_of(&self, texture: TextureHandle) -> Result<(vk::Image, vk::ImageLayout)> {
        let textures = self.textures.lock().unwrap();
        textures
            .get(texture)
            .map(|tex| (tex.image, tex.resting))
            .ok_or_else(stale_texture)
    }

    /// Whether the region holds texels: a partial write must keep the rest.
    fn is_ready(&self, texture: TextureHandle, region: TextureRegion) -> bool {
        let textures = self.textures.lock().unwrap();
        textures
            .get(texture)
            .is_some_and(|tex| tex.is_ready(region.mip_level, region.layer))
    }

    fn mark_ready(&self, texture: TextureHandle, region: TextureRegion) {
        let mut textures = self.textures.lock().unwrap();
        if let Some(tex) = textures.get_mut(texture) {
            tex.mark_ready(region.mip_level, region.layer);
        }
    }

    /// Room for `len` bytes: in the ring, an idle overflow buffer, or a new allocation.
    /// The boolean marks overflow buffers, which are settled separately from ring spans.
    fn stage(&self, len: u64) -> Result<(BufferHandle, u64, bool)> {
        self.reap_orphans();
        let completed = self.inner.serials.completed();
        let mut ring = self.staging.lock().unwrap();
        ring.reclaim(completed);
        if let Some((buffer, size)) = ring.buffer
            && let Some(at) = ring.place(len, size)
        {
            ring.take(at, len, PENDING);
            return Ok((buffer, at, false));
        }
        let wanted = len.next_power_of_two().max(RING_START);
        if ring.in_use.is_empty() && wanted <= RING_MAX {
            let grown = wanted
                .max(ring.buffer.map_or(0, |(_, size)| size * 2))
                .min(RING_MAX);
            let fresh = self.create_staging(grown)?;
            if let Some((old, _)) = ring.buffer.replace((fresh, grown)) {
                self.destroy_buffer(old);
            }
            ring.head = 0;
            ring.take(0, len, PENDING);
            return Ok((fresh, 0, false));
        }
        if let Some(buffer) = ring.take_spare(len, completed) {
            return Ok((buffer, 0, true));
        }
        let buffer = self.create_staging(len)?;
        if ring.can_keep_spare(len) {
            ring.spares.push(Spare {
                buffer,
                size: len,
                serial: PENDING,
            });
        }
        Ok((buffer, 0, true))
    }

    fn create_staging(&self, size: u64) -> Result<BufferHandle> {
        self.create_buffer(BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_SRC,
            memory: MemoryUsage::Upload,
        })
    }

    fn copy_rows_into(
        &self,
        staging: BufferHandle,
        at: u64,
        write: &TextureWrite<'_>,
    ) -> Result<()> {
        let texel = self.texel_bytes(write.texture)?;
        let row = (write.region.width * texel) as usize;
        let pitch = write.bytes_per_row as usize;
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(staging)
            .ok_or_else(|| GpuError::Backend("staging buffer went away".into()))?;
        if buf.mapped.is_null() {
            return Err(GpuError::Backend(
                "staging buffer is not host visible".into(),
            ));
        }
        for y in 0..write.region.height as usize {
            let src = &write.data[y * pitch..y * pitch + row];
            // SAFETY: the span [at, at + rows * row) was reserved for this write, inside the buffer
            unsafe {
                copy_nonoverlapping(src.as_ptr(), buf.mapped.add(at as usize + y * row), row);
            }
        }
        Ok(())
    }

    fn vk_buffer(&self, buffer: BufferHandle) -> Result<vk::Buffer> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(buffer)
            .map(|b| b.buffer)
            .ok_or_else(|| GpuError::Backend("stale buffer handle".into()))
    }
}

fn stale_texture() -> GpuError {
    GpuError::Backend("stale texture handle".into())
}

/// Row pitch in texels, for a buffer-image copy.
fn rows_in_texels(bytes_per_row: u32, width: u32, texel: u32) -> Result<u32> {
    if !bytes_per_row.is_multiple_of(texel) || bytes_per_row / texel < width {
        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
            format!("rows of {bytes_per_row} bytes do not hold {width} texels of {texel} bytes"),
        )));
    }
    Ok(bytes_per_row / texel)
}

/// Bytes from the region's first texel to its last, rows `bytes_per_row` apart.
fn span_bytes(region: TextureRegion, bytes_per_row: u32, texel: u32) -> u64 {
    u64::from(region.height.saturating_sub(1)) * u64::from(bytes_per_row)
        + u64::from(region.width) * u64::from(texel)
}

fn subresource(region: TextureRegion) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: region.mip_level,
        level_count: 1,
        base_array_layer: region.layer,
        layer_count: 1,
    }
}

fn buffer_image_copy(region: TextureRegion, offset: u64, row_texels: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: row_texels,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: region.mip_level,
            base_array_layer: region.layer,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: i32::try_from(region.x).unwrap_or(i32::MAX),
            y: i32::try_from(region.y).unwrap_or(i32::MAX),
            z: 0,
        },
        image_extent: vk::Extent3D {
            width: region.width,
            height: region.height,
            depth: 1,
        },
    }
}

fn image_barrier(
    dev: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    region: TextureRegion,
    (from, to): (vk::ImageLayout, vk::ImageLayout),
) {
    let barrier = vk::ImageMemoryBarrier {
        old_layout: from,
        new_layout: to,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: subresource(region),
        src_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        dst_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ..Default::default()
    };
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn memory_barrier(
    dev: &Device,
    cmd: vk::CommandBuffer,
    (src, dst): (vk::AccessFlags, vk::AccessFlags),
) {
    let barrier = vk::MemoryBarrier {
        src_access_mask: src,
        dst_access_mask: dst,
        ..Default::default()
    };
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Copies each span of `buffer` into its image region. Keeps texels around it if the subresource
/// had some.
fn record_buffer_to_images(
    dev: &Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    placed: &[Placed],
) {
    for p in placed {
        let from = if p.was_ready {
            p.resting
        } else {
            vk::ImageLayout::UNDEFINED
        };
        image_barrier(
            dev,
            cmd,
            p.image,
            p.region,
            (from, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        );
        unsafe {
            dev.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                p.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[buffer_image_copy(p.region, p.offset, p.row_texels)],
            );
        }
        let back = (vk::ImageLayout::TRANSFER_DST_OPTIMAL, p.resting);
        image_barrier(dev, cmd, p.image, p.region, back);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_buffers_wait_for_completion_and_choose_the_smallest_fit() {
        let mut ring = StagingRing::default();
        let a = BufferHandle::from_parts(1, 0);
        let b = BufferHandle::from_parts(2, 0);
        ring.spares = vec![
            Spare {
                buffer: a,
                size: 1024,
                serial: 3,
            },
            Spare {
                buffer: b,
                size: 512,
                serial: 4,
            },
        ];
        assert_eq!(ring.take_spare(256, 2), None);
        assert_eq!(ring.take_spare(256, 4), Some(b));
        assert_eq!(ring.take_spare(256, 4), Some(a));
        assert_eq!(
            ring.take_spare(256, 4),
            None,
            "reserved buffers cannot be reused"
        );
        assert!(ring.settle_spare(b, 5));
        assert_eq!(ring.take_spare(256, 4), None, "GPU still reads the buffer");
        assert_eq!(ring.take_spare(256, 5), Some(b));
        assert!(ring.settle_spare(b, 0));
        assert_eq!(
            ring.take_spare(256, 5),
            Some(b),
            "failed writes return their buffer"
        );
        assert!(!ring.settle_spare(BufferHandle::from_parts(9, 0), 0));
    }

    #[test]
    fn overflow_retention_is_bounded_by_bytes_and_buffer_count() {
        let mut ring = StagingRing::default();
        assert!(ring.can_keep_spare(SPARE_MAX_BYTES));
        assert!(!ring.can_keep_spare(SPARE_MAX_BYTES + 1));
        ring.spares.push(Spare {
            buffer: BufferHandle::from_parts(1, 0),
            size: SPARE_MAX_BYTES,
            serial: 0,
        });
        assert!(!ring.can_keep_spare(1));
        ring.spares = (0..SPARE_MAX_BUFFERS)
            .map(|i| Spare {
                buffer: BufferHandle::from_parts(i as u32, 0),
                size: 16,
                serial: 0,
            })
            .collect();
        assert!(!ring.can_keep_spare(16));
    }

    #[test]
    fn busy_ring_uploads_reuse_one_overflow_buffer_and_keep_their_texels() {
        use crate::instance::VulkanInstance;
        use std::time::Duration;
        use zengpu_hal::{
            AdapterRequest, DeviceRequest, Format, GpuInstance, TexDim, TextureDesc, TextureUsage,
        };

        let _guard = crate::test_gpu_lock();
        let instance = VulkanInstance::new().expect("Vulkan instance");
        let Some(adapter) = instance.request_adapter(AdapterRequest::default()) else {
            eprintln!("no Vulkan adapter; skipping overflow upload test");
            return;
        };
        let device = adapter.open(DeviceRequest::default()).expect("device");
        let native = device
            .as_any()
            .downcast_ref::<VulkanDevice>()
            .expect("Vulkan");
        let (_, base, overflow) = native.stage(RING_START).expect("reserve whole ring");
        assert!(!overflow);
        assert_eq!(
            native.trim_staging_memory(),
            0,
            "a caller still fills the ring"
        );
        let texture = device
            .create_texture(TextureDesc {
                width: 2,
                height: 2,
                depth: 1,
                format: Format::Rgba8Unorm,
                usage: TextureUsage::SAMPLED
                    | TextureUsage::TRANSFER_DST
                    | TextureUsage::TRANSFER_SRC,
                samples: 1,
                dimension: TexDim::D2,
                mip_levels: 1,
                array_layers: 1,
            })
            .expect("texture");
        let readback = device
            .create_buffer(BufferDesc {
                size: 16,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::READBACK,
                memory: MemoryUsage::Readback,
            })
            .expect("readback");
        let mut first = None;
        for value in 1..=16 {
            let data = [value; 16];
            device
                .write_textures(&[TextureWrite {
                    texture,
                    region: TextureRegion::rect(0, 0, 2, 2),
                    data: &data,
                    bytes_per_row: 8,
                }])
                .expect("upload")
                .wait(Duration::from_secs(5))
                .expect("upload done");
            {
                let ring = native.staging.lock().unwrap();
                assert_eq!(
                    ring.spares.len(),
                    1,
                    "one allocation across repeated uploads"
                );
                assert_eq!(
                    *first.get_or_insert(ring.spares[0].buffer),
                    ring.spares[0].buffer
                );
            }
            device
                .copy_texture_region_to_buffer(
                    texture,
                    TextureRegion::rect(0, 0, 2, 2),
                    readback,
                    0,
                    8,
                )
                .expect("copy")
                .wait(Duration::from_secs(5))
                .expect("readback done");
            assert_eq!(device.read_buffer(readback, 0, 16).expect("pixels"), data);
        }
        native.staging.lock().unwrap().settle(base, 0);
        assert_eq!(native.trim_staging_memory(), RING_START + 16);
        {
            let ring = native.staging.lock().unwrap();
            assert!(ring.buffer.is_none() && ring.spares.is_empty());
        }
        device
            .write_textures(&[TextureWrite {
                texture,
                region: TextureRegion::rect(0, 0, 2, 2),
                data: &[42; 16],
                bytes_per_row: 8,
            }])
            .expect("upload after trimming")
            .wait(Duration::from_secs(5))
            .expect("done");
        device.destroy_buffer(readback);
        device.destroy_texture(texture);
    }

    fn ring(size: u64) -> (StagingRing, u64) {
        let ring = StagingRing {
            buffer: None,
            head: 0,
            in_use: VecDeque::new(),
            spares: Vec::new(),
        };
        (ring, size)
    }

    #[test]
    fn spans_come_back_once_their_work_is_done() {
        let (mut ring, size) = ring(100);
        let a = ring.place(40, size).expect("room");
        ring.take(a, 40, 1);
        let b = ring.place(40, size).expect("room");
        ring.take(b, 40, 2);
        assert_eq!((a, b), (0, 48), "aligned after the first");
        assert_eq!(ring.place(40, size), None, "no room while both are read");
        ring.reclaim(1);
        assert_eq!(ring.place(40, size), Some(0), "the first span came back");
        ring.reclaim(2);
        assert_eq!(ring.head, 0, "an idle ring starts over");
    }

    #[test]
    fn a_wrapped_ring_stops_at_its_oldest_span() {
        let (mut ring, size) = ring(100);
        ring.take(0, 40, 1);
        ring.take(48, 40, 2);
        ring.reclaim(1);
        let c = ring.place(30, size).expect("room at the start");
        assert_eq!(c, 0);
        ring.take(c, 30, 3);
        assert_eq!(
            ring.place(20, size),
            None,
            "only 16 bytes to the oldest span"
        );
        assert!(ring.place(10, size).is_some());
    }

    #[test]
    fn a_reserved_span_waits_for_its_submission() {
        let (mut ring, size) = ring(100);
        ring.take(0, 40, PENDING);
        ring.reclaim(5);
        assert_eq!(ring.in_use.len(), 1, "not reclaimed before it is settled");
        ring.settle(0, 3);
        ring.reclaim(5);
        assert!(ring.in_use.is_empty());
        assert_eq!(ring.place(90, size), Some(0));
    }

    #[test]
    fn spans_and_pitches_are_measured() {
        let region = TextureRegion::rect(0, 0, 3, 2);
        assert_eq!(span_bytes(region, 64, 4), 64 + 12);
        assert_eq!(rows_in_texels(64, 3, 4).expect("fits"), 16);
        assert!(rows_in_texels(10, 3, 4).is_err());
        assert!(rows_in_texels(66, 3, 4).is_err());
    }
}
