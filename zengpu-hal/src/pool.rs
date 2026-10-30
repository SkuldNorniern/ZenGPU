//! Reuses released textures. Textures are kept by exact description and handed out again. Free
//! textures past a byte budget are destroyed, oldest first.
//!
//! A handed-out texture may still be read by work submitted before its release. On one queue that
//! is safe for later work, since the queue runs in order. Contents are stale, so write all of a
//! texture before reading it.

use std::collections::{HashMap, VecDeque};

use crate::desc::TextureDesc;
use crate::device::GpuDevice;
use crate::error::Result;
use crate::handle::TextureHandle;

/// Textures are matched by creation description.
#[derive(Debug)]
pub struct TexturePool {
    free: HashMap<TextureDesc, VecDeque<(TextureHandle, u64)>>,
    live: HashMap<TextureHandle, TextureDesc>,
    free_bytes: u64,
    budget: u64,
    clock: u64,
}

impl TexturePool {
    pub fn new(budget: u64) -> Self {
        Self {
            free: HashMap::new(),
            live: HashMap::new(),
            free_bytes: 0,
            budget,
            clock: 0,
        }
    }

    pub fn acquire(&mut self, device: &dyn GpuDevice, desc: TextureDesc) -> Result<TextureHandle> {
        let reused = self.free.get_mut(&desc).and_then(VecDeque::pop_back);
        let texture = match reused {
            Some((texture, _)) => {
                self.free_bytes = self.free_bytes.saturating_sub(bytes_of(&desc));
                texture
            }
            None => device.create_texture(desc)?,
        };
        self.live.insert(texture, desc);
        Ok(texture)
    }

    /// Textures the pool did not hand out are destroyed, not pooled. Duplicate
    /// returns of an already cached handle are ignored. Return a handle only
    /// once per checkout; old aliases cannot identify a later checkout.
    pub fn release(&mut self, device: &dyn GpuDevice, texture: TextureHandle) {
        let Some(desc) = self.live.remove(&texture) else {
            if self
                .free
                .values()
                .any(|list| list.iter().any(|(cached, _)| *cached == texture))
            {
                return;
            }
            device.destroy_texture(texture);
            return;
        };
        self.clock = self.clock.saturating_add(1);
        self.free
            .entry(desc)
            .or_default()
            .push_back((texture, self.clock));
        self.free_bytes = self.free_bytes.saturating_add(bytes_of(&desc));
        self.evict(device, self.budget);
    }

    /// Destroys free textures, oldest first, until `keep` bytes remain.
    ///
    /// Under memory pressure, pass less than the budget.
    pub fn trim(&mut self, device: &dyn GpuDevice, keep: u64) {
        self.free.retain(|_, list| !list.is_empty());
        self.evict(device, keep);
    }

    fn evict(&mut self, device: &dyn GpuDevice, keep: u64) {
        if self.free_bytes > keep {
            self.free.retain(|_, list| !list.is_empty());
        }
        while self.free_bytes > keep {
            let Some(desc) = self.oldest_free() else {
                break;
            };
            let Some(list) = self.free.get_mut(&desc) else {
                break;
            };
            let Some((texture, _)) = list.pop_front() else {
                break;
            };
            if list.is_empty() {
                self.free.remove(&desc);
            }
            self.free_bytes = self.free_bytes.saturating_sub(bytes_of(&desc));
            device.destroy_texture(texture);
        }
    }

    /// Textures that are handed out are not touched.
    pub fn clear(&mut self, device: &dyn GpuDevice) {
        self.trim(device, 0);
    }

    pub fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub fn live(&self) -> usize {
        self.live.len()
    }

    fn oldest_free(&self) -> Option<TextureDesc> {
        self.free
            .iter()
            .filter_map(|(desc, list)| list.front().map(|(_, at)| (*at, *desc)))
            .min_by_key(|(at, _)| *at)
            .map(|(_, desc)| desc)
    }
}

/// Approximate memory a texture takes, counting every mip and every layer.
fn bytes_of(desc: &TextureDesc) -> u64 {
    let texel = u64::from(desc.format.bytes_per_texel()) * u64::from(desc.samples.max(1));
    let mut total = 0u64;
    let mips = desc.mip_levels.max(1);
    for mip in 0..mips.min(u32::BITS) {
        let w = u64::from((desc.width >> mip).max(1));
        let h = u64::from((desc.height >> mip).max(1));
        let d = u64::from((desc.depth >> mip).max(1));
        total = total.saturating_add(w.saturating_mul(h).saturating_mul(d).saturating_mul(texel));
    }
    // Overspecified descriptors must not overflow a shift or require billions
    // of iterations in accounting. Dimensions beyond bit 31 all clamp to one.
    total = total.saturating_add(u64::from(mips.saturating_sub(u32::BITS)).saturating_mul(texel));
    total.saturating_mul(u64::from(desc.array_layers.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::TexDim;
    use crate::types::{Format, TextureUsage};
    use crate::{
        BufferDesc, BufferHandle, GpuError, HalCapabilities, SamplerDesc, SamplerHandle, SlotMap,
        marker,
    };
    use core::any::Any;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Textures {
        slots: Mutex<SlotMap<marker::Texture, TextureDesc>>,
        destroyed: Mutex<Vec<TextureHandle>>,
    }

    fn unsupported<T>() -> Result<T> {
        Err(GpuError::Unsupported("test texture device".into()))
    }

    impl GpuDevice for Textures {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn capabilities(&self) -> HalCapabilities {
            HalCapabilities::default()
        }
        fn create_buffer(&self, _: BufferDesc) -> Result<BufferHandle> {
            unsupported()
        }
        fn write_buffer(&self, _: BufferHandle, _: u64, _: &[u8]) -> Result<()> {
            unsupported()
        }
        fn read_buffer(&self, _: BufferHandle, _: u64, _: u64) -> Result<Vec<u8>> {
            unsupported()
        }
        fn destroy_buffer(&self, _: BufferHandle) {}
        fn create_texture(&self, desc: TextureDesc) -> Result<TextureHandle> {
            Ok(self.slots.lock().unwrap().insert(desc))
        }
        fn upload_texture_data(&self, texture: TextureHandle, _: &[u8]) -> Result<()> {
            if self.slots.lock().unwrap().get(texture).is_some() {
                Ok(())
            } else {
                Err(GpuError::Backend("stale texture".into()))
            }
        }
        fn destroy_texture(&self, texture: TextureHandle) {
            if self.slots.lock().unwrap().remove(texture).is_some() {
                self.destroyed.lock().unwrap().push(texture);
            }
        }
        fn create_sampler(&self, _: SamplerDesc) -> Result<SamplerHandle> {
            unsupported()
        }
        fn destroy_sampler(&self, _: SamplerHandle) {}
    }

    fn tile(size: u32) -> TextureDesc {
        TextureDesc {
            width: size,
            height: size,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        }
    }

    #[test]
    fn sizes_count_every_mip_and_layer() {
        assert_eq!(bytes_of(&tile(256)), 256 * 256 * 4);
        let mipped = TextureDesc {
            mip_levels: 2,
            array_layers: 3,
            ..tile(4)
        };
        assert_eq!(bytes_of(&mipped), (16 + 4) * 4 * 3);
        assert_eq!(
            bytes_of(&TextureDesc {
                mip_levels: 33,
                ..tile(1)
            }),
            33 * 4
        );
        assert_eq!(
            bytes_of(&TextureDesc {
                mip_levels: u32::MAX,
                ..tile(1)
            }),
            u64::from(u32::MAX) * 4
        );
        assert_eq!(
            bytes_of(&TextureDesc {
                width: u32::MAX,
                height: u32::MAX,
                depth: u32::MAX,
                ..tile(1)
            }),
            u64::MAX
        );
    }

    #[test]
    fn duplicate_returns_do_not_destroy_cached_textures_or_hand_them_out_twice() {
        let device = Textures::default();
        let mut pool = TexturePool::new(1024);
        let texture = pool.acquire(&device, tile(4)).unwrap();
        pool.release(&device, texture);
        pool.release(&device, texture);
        assert_eq!(pool.free_bytes(), 64);
        assert!(device.destroyed.lock().unwrap().is_empty());
        let reused = pool.acquire(&device, tile(4)).unwrap();
        assert_eq!(reused, texture);
        device.upload_texture_data(reused, &[255; 64]).unwrap();
        let other = pool.acquire(&device, tile(4)).unwrap();
        assert_ne!(reused, other);
        pool.release(&device, reused);
        pool.release(&device, other);
        pool.clear(&device);
        assert_eq!(device.destroyed.lock().unwrap().len(), 2);
    }

    #[test]
    fn trimming_preserves_oldest_first_order_across_texture_descriptions() {
        let device = Textures::default();
        let mut pool = TexturePool::new(1024);
        let first = pool.acquire(&device, tile(2)).unwrap();
        let middle = pool.acquire(&device, tile(4)).unwrap();
        let last = pool.acquire(&device, tile(2)).unwrap();
        for texture in [first, middle, last] {
            pool.release(&device, texture);
        }
        pool.trim(&device, 80);
        assert_eq!(&*device.destroyed.lock().unwrap(), &[first]);
        assert_eq!(pool.free_bytes(), 80);
        assert_eq!(pool.acquire(&device, tile(2)).unwrap(), last);
        assert_eq!(pool.free_bytes(), 64);
        pool.clear(&device);
        assert_eq!(&*device.destroyed.lock().unwrap(), &[first, middle]);
        assert_eq!(
            pool.live(),
            1,
            "trimming never touches a checked-out texture"
        );
    }
}
