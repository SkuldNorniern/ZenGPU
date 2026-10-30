//! Suballocations for optimal-tiling images only. Buffers and linear images never
//! enter these blocks, so no buffer/image granularity padding is needed.

use std::sync::{Arc, Mutex, Weak};

use ash::{Device, vk};
use zengpu_hal::{GpuError, MemoryUsage, Result};

const BLOCK_BYTES: u64 = 16 << 20;
const MIN_BLOCK_BYTES: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    offset: u64,
    size: u64,
}

struct FreeRanges(Vec<Span>);

impl FreeRanges {
    fn new(size: u64) -> Self {
        Self(vec![Span { offset: 0, size }])
    }

    fn take(&mut self, size: u64, alignment: u64) -> Option<Span> {
        let (index, offset) = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, span)| {
                let offset = span.offset.checked_add(alignment - 1)? / alignment * alignment;
                let end = offset.checked_add(size)?;
                (end <= span.offset + span.size).then_some((i, offset))
            })
            .min_by_key(|(i, _)| self.0[*i].size)?;
        let old = self.0.remove(index);
        if offset > old.offset {
            self.0.insert(
                index,
                Span {
                    offset: old.offset,
                    size: offset - old.offset,
                },
            );
        }
        let end = offset + size;
        if end < old.offset + old.size {
            let at = self.0.partition_point(|s| s.offset < end);
            self.0.insert(
                at,
                Span {
                    offset: end,
                    size: old.offset + old.size - end,
                },
            );
        }
        Some(Span { offset, size })
    }

    fn release(&mut self, span: Span) {
        let mut at = self.0.partition_point(|s| s.offset < span.offset);
        self.0.insert(at, span);
        if at > 0 && self.0[at - 1].offset + self.0[at - 1].size == self.0[at].offset {
            let right = self.0.remove(at);
            at -= 1;
            self.0[at].size += right.size;
        }
        if at + 1 < self.0.len() && self.0[at].offset + self.0[at].size == self.0[at + 1].offset {
            let right = self.0.remove(at + 1);
            self.0[at].size += right.size;
        }
    }
}

struct Block {
    // Does not own the logical device. Textures and their deferred destruction are
    // drained before VulkanDeviceInner destroys it, as with the previous raw memory.
    device: Device,
    memory: vk::DeviceMemory,
    memory_type: u32,
    size: u64,
    free: Mutex<FreeRanges>,
}

impl Drop for Block {
    fn drop(&mut self) {
        unsafe { self.device.free_memory(self.memory, None) };
    }
}

pub(crate) struct ImageAllocation {
    block: Arc<Block>,
    span: Span,
}

impl ImageAllocation {
    pub(crate) fn memory(&self) -> vk::DeviceMemory {
        self.block.memory
    }

    pub(crate) fn offset(&self) -> u64 {
        self.span.offset
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.block.size
    }

    #[cfg(test)]
    pub(crate) fn used_bytes(&self) -> u64 {
        self.span.size
    }
}

impl Drop for ImageAllocation {
    fn drop(&mut self) {
        self.block.free.lock().unwrap().release(self.span);
    }
}

#[derive(Default)]
pub(crate) struct ImageMemory {
    // Weak references retain neither an idle block nor the device. The last live
    // allocation frees the block immediately, including deferred GPU allocations.
    blocks: Vec<Weak<Block>>,
    #[cfg(test)]
    pub(crate) force_dedicated: bool,
}

impl ImageMemory {
    #[cfg(test)]
    pub(crate) fn blocks_empty(&self) -> bool {
        self.blocks.iter().all(|block| block.strong_count() == 0)
    }

    pub(crate) fn allocate(
        &mut self,
        device: &Device,
        image: vk::Image,
        requirements: vk::MemoryRequirements,
        memory_type: u32,
        dedicated: bool,
    ) -> Result<ImageAllocation> {
        let pooled = !dedicated && requirements.size <= BLOCK_BYTES / 2;
        #[cfg(test)]
        let pooled = pooled && !self.force_dedicated;
        let mut block_bytes = if pooled {
            (requirements.size * 4)
                .next_power_of_two()
                .clamp(MIN_BLOCK_BYTES, BLOCK_BYTES)
        } else {
            requirements.size
        };
        if pooled {
            self.blocks.retain(|block| block.strong_count() > 0);
            for weak in &self.blocks {
                if let Some(block) = weak.upgrade()
                    && block.memory_type == memory_type
                {
                    block_bytes = block_bytes.max((block.size * 2).min(BLOCK_BYTES));
                    let span = block
                        .free
                        .lock()
                        .unwrap()
                        .take(requirements.size, requirements.alignment);
                    if let Some(span) = span {
                        return Ok(ImageAllocation { block, span });
                    }
                }
            }
        }
        // A large block can fail under pressure while an exact-sized allocation
        // still fits. In that case fall back without retaining unused block space.
        let mut size = if pooled {
            block_bytes
        } else {
            requirements.size
        };
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut make = |size, dedicated| {
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(memory_type);
            if dedicated {
                info = info.push_next(&mut dedicated_info);
            }
            unsafe { device.allocate_memory(&info, None) }
        };
        let memory = match make(size, !pooled) {
            Ok(memory) => memory,
            Err(_) if pooled => {
                size = requirements.size;
                make(size, true).map_err(|_| GpuError::OutOfMemory(MemoryUsage::GpuOnly))?
            }
            Err(_) => return Err(GpuError::OutOfMemory(MemoryUsage::GpuOnly)),
        };
        let block = Arc::new(Block {
            device: device.clone(),
            memory,
            memory_type,
            size,
            free: Mutex::new(FreeRanges::new(size)),
        });
        let span = block
            .free
            .lock()
            .unwrap()
            .take(requirements.size, requirements.alignment)
            .ok_or(GpuError::OutOfMemory(MemoryUsage::GpuOnly))?;
        if pooled && size == block_bytes {
            self.blocks.push(Arc::downgrade(&block));
        }
        Ok(ImageAllocation { block, span })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_padding_is_reused_and_both_neighbors_coalesce() {
        let mut ranges = FreeRanges::new(1024);
        let a = ranges.take(17, 16).unwrap();
        let b = ranges.take(100, 256).unwrap();
        assert_eq!(b.offset, 256);
        let c = ranges.take(32, 32).unwrap();
        assert_eq!(c.offset, 32, "alignment padding remains available");
        assert!(ranges.take(1024, 1).is_none());
        for span in [a, b, c] {
            ranges.release(span);
        }
        assert_eq!(
            ranges.0,
            [Span {
                offset: 0,
                size: 1024
            }]
        );
        assert_eq!(ranges.take(1024, 1).unwrap().offset, 0);
    }

    #[test]
    fn fragmented_allocations_never_overlap_and_all_space_returns() {
        let mut ranges = FreeRanges::new(65536);
        let mut live = Vec::<Span>::new();
        let mut seed = 42u64;
        for _ in 0..10000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if seed & 3 == 0 && !live.is_empty() {
                let index = seed as usize % live.len();
                ranges.release(live.swap_remove(index));
            } else {
                let alignment = 1 << ((seed >> 16) % 10);
                let size = (seed >> 32) % 1000 + 1;
                if let Some(span) = ranges.take(size, alignment) {
                    assert_eq!(span.offset % alignment, 0);
                    assert!(span.offset + span.size <= 65536);
                    assert!(
                        live.iter().all(|s| span.offset >= s.offset + s.size
                            || span.offset + span.size <= s.offset)
                    );
                    live.push(span);
                }
            }
        }
        for span in live {
            ranges.release(span);
        }
        assert_eq!(
            ranges.0,
            [Span {
                offset: 0,
                size: 65536
            }]
        );
    }
}
