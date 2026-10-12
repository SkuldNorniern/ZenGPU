//! Moving texels between the host, buffers and parts of textures without
//! waiting for the GPU.
//!
//! Host data goes through one staging ring that is reused: a write takes a
//! span of it, the span comes back once the submission reading it has
//! finished, and the ring grows when a write does not fit. Every operation
//! is one submission that returns at once; later work on the device sees
//! its result because the queue runs in order and each copy ends with a
//! barrier for whatever reads next.

use std::collections::VecDeque;
use std::ptr::copy_nonoverlapping;

use ash::{Device, vk};
use zengpu_hal::{
    BufferDesc, BufferHandle, BufferUsage, GpuDevice, GpuError, MemoryUsage, Result, Submission,
    TextureHandle, TextureRegion, TextureWrite, UsageError,
};

use super::VulkanDevice;

/// The ring starts this big and doubles up to [`RING_MAX`].
const RING_START: u64 = 4 << 20;
/// A write bigger than the ring at this size gets a buffer of its own.
const RING_MAX: u64 = 64 << 20;
/// Offsets into the ring are kept to this, which every copy accepts.
const RING_ALIGN: u64 = 16;

/// A span of the ring a submission is still reading.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u64,
    serial: u64,
}

/// Host-visible staging memory, reused across writes.
#[derive(Default)]
pub(crate) struct StagingRing {
    buffer: Option<(BufferHandle, u64)>,
    head: u64,
    in_use: VecDeque<Span>,
}

impl StagingRing {
    /// Gives back the spans whose submissions have finished.
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

    /// Where `len` bytes fit in a ring of `size`, if they do.
    fn place(&self, len: u64, size: u64) -> Option<u64> {
        let start = self.head.next_multiple_of(RING_ALIGN);
        let Some(oldest) = self.in_use.front() else {
            return (len <= size).then_some(0);
        };
        if oldest.start < self.head {
            // Not wrapped: room after the head, or from the start up to the
            // oldest span.
            if start + len <= size {
                return Some(start);
            }
            return (len <= oldest.start).then_some(0);
        }
        // Wrapped: room between the head and the oldest span.
        (start + len <= oldest.start).then_some(start)
    }

    fn take(&mut self, start: u64, len: u64, serial: u64) {
        self.head = start + len;
        self.in_use.push_back(Span { start, serial });
    }

    /// Gives the span reserved at `start` the submission that reads it;
    /// `0` hands it straight back, for a write that was never submitted.
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

/// The serial of a span reserved for a submission not made yet; it never
/// completes until settled.
const PENDING: u64 = u64::MAX;

/// One copy into a texture, checked and placed in staging memory.
struct Placed {
    image: vk::Image,
    region: TextureRegion,
    was_ready: bool,
    offset: u64,
    row_texels: u32,
}

impl VulkanDevice {
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
        if own_buffer {
            // Freed once the submission has finished with it, or now.
            self.destroy_buffer(staging);
        } else {
            let serial = submitted.as_ref().map_or(0, |(_, serial)| *serial);
            self.staging.lock().unwrap().settle(base, serial);
        }
        let (submission, _) = submitted?;
        for write in writes {
            self.mark_ready(write.texture, write.region);
        }
        Ok(submission)
    }

    /// Copies the writes into staging at `base` and submits the copies.
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
            placed.push(Placed {
                image: self.image_of(write.texture)?,
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
        let placed = [Placed {
            image: self.image_of(texture)?,
            region,
            was_ready: self.is_ready(texture, region),
            offset,
            row_texels,
        }];
        let vk_buffer = self.vk_buffer(buffer)?;
        let (submission, _) = self.one_shot_submit_tracked(0, &[buffer.index()], |dev, cmd| {
            // What earlier work wrote into the buffer has to be visible to
            // the copy.
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
        let image = self.image_of(texture)?;
        let was_ready = self.is_ready(texture, region);
        let vk_buffer = self.vk_buffer(buffer)?;
        let (submission, _) = self.one_shot_submit_tracked(0, &[buffer.index()], |dev, cmd| {
            let from = if was_ready {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
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
            let back = (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(dev, cmd, image, region, back);
            // The bytes are for the host, or for whatever reads next.
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

    // ── checks ───────────────────────────────────────────────────────────

    /// Checks a write and returns the bytes it takes in staging, rows packed.
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
        let mip_w = (tex.extent.width >> region.mip_level).max(1);
        let mip_h = (tex.extent.height >> region.mip_level).max(1);
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

    // ── texture state ────────────────────────────────────────────────────

    fn texel_bytes(&self, texture: TextureHandle) -> Result<u32> {
        let textures = self.textures.lock().unwrap();
        textures
            .get(texture)
            .map(|tex| tex.texel_bytes)
            .ok_or_else(stale_texture)
    }

    fn image_of(&self, texture: TextureHandle) -> Result<vk::Image> {
        let textures = self.textures.lock().unwrap();
        textures
            .get(texture)
            .map(|tex| tex.image)
            .ok_or_else(stale_texture)
    }

    /// Whether the region's mip and layer hold texels already, so a write to
    /// part of it must keep the rest.
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

    // ── staging ──────────────────────────────────────────────────────────

    /// Room for `len` bytes of staging: in the ring when it fits, growing
    /// it when it is idle, else a buffer of its own. Returns the buffer,
    /// the offset, and whether the buffer is a one-off to free after.
    fn stage(&self, len: u64) -> Result<(BufferHandle, u64, bool)> {
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
        drop(ring);
        Ok((self.create_staging(len)?, 0, true))
    }

    fn create_staging(&self, size: u64) -> Result<BufferHandle> {
        self.create_buffer(BufferDesc {
            size,
            usage: BufferUsage::TRANSFER_SRC,
            memory: MemoryUsage::Upload,
        })
    }

    /// Copies a write's rows, packed, into staging at `at`.
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
            // SAFETY: the span [at, at + rows * row) was reserved for this
            // write inside the mapped buffer, which `stage` sized for it.
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

/// The row pitch in texels, which a buffer-image copy takes.
fn rows_in_texels(bytes_per_row: u32, width: u32, texel: u32) -> Result<u32> {
    if bytes_per_row % texel != 0 || bytes_per_row / texel < width {
        return Err(GpuError::InvalidUsage(UsageError::BindingMismatch(
            format!("rows of {bytes_per_row} bytes do not hold {width} texels of {texel} bytes"),
        )));
    }
    Ok(bytes_per_row / texel)
}

/// Bytes from the first texel of the region's first row to the last of its
/// last row, rows `bytes_per_row` apart.
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

/// Moves one subresource between layouts, after whatever earlier work did
/// with it and before whatever comes next.
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

/// Copies each placed span of `buffer` into its image region, keeping the
/// texels around it when the subresource already held some.
fn record_buffer_to_images(
    dev: &Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    placed: &[Placed],
) {
    for p in placed {
        let from = if p.was_ready {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
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
        let back = (
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        image_barrier(dev, cmd, p.image, p.region, back);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(size: u64) -> (StagingRing, u64) {
        let ring = StagingRing {
            buffer: None,
            head: 0,
            in_use: VecDeque::new(),
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
