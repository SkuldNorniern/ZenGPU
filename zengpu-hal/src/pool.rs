//! Reuses released textures. Textures are kept by exact description and handed out again. Free
//! textures past a byte budget are destroyed, oldest first.
//!
//! A handed-out texture may still be read by work submitted before its release. On one queue that
//! is safe for later work, since the queue runs in order. Contents are stale, so write all of a
//! texture before reading it.

use std::collections::HashMap;

use crate::desc::TextureDesc;
use crate::device::GpuDevice;
use crate::error::Result;
use crate::handle::TextureHandle;

/// Textures are matched by creation description.
#[derive(Debug)]
pub struct TexturePool {
    free: HashMap<TextureDesc, Vec<(TextureHandle, u64)>>,
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
        let reused = self.free.get_mut(&desc).and_then(Vec::pop);
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

    /// Textures the pool did not hand out are destroyed, not pooled.
    pub fn release(&mut self, device: &dyn GpuDevice, texture: TextureHandle) {
        let Some(desc) = self.live.remove(&texture) else {
            device.destroy_texture(texture);
            return;
        };
        self.clock += 1;
        self.free
            .entry(desc)
            .or_default()
            .push((texture, self.clock));
        self.free_bytes += bytes_of(&desc);
        self.trim(device, self.budget);
    }

    /// Destroys free textures, oldest first, until `keep` bytes remain.
    ///
    /// Under memory pressure, pass less than the budget.
    pub fn trim(&mut self, device: &dyn GpuDevice, keep: u64) {
        while self.free_bytes > keep {
            let Some((desc, at)) = self.oldest_free() else {
                break;
            };
            let Some(list) = self.free.get_mut(&desc) else {
                break;
            };
            let (texture, _) = list.remove(at);
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

    fn oldest_free(&self) -> Option<(TextureDesc, usize)> {
        self.free
            .iter()
            .flat_map(|(desc, list)| {
                list.iter()
                    .enumerate()
                    .map(move |(i, (_, at))| (*at, *desc, i))
            })
            .min_by_key(|(at, _, _)| *at)
            .map(|(_, desc, i)| (desc, i))
    }
}

/// Approximate memory a texture takes, counting every mip and every layer.
fn bytes_of(desc: &TextureDesc) -> u64 {
    let texel = u64::from(desc.format.bytes_per_texel()) * u64::from(desc.samples.max(1));
    let mut total = 0u64;
    for mip in 0..desc.mip_levels.max(1) {
        let w = u64::from((desc.width >> mip).max(1));
        let h = u64::from((desc.height >> mip).max(1));
        let d = u64::from((desc.depth >> mip).max(1));
        total += w * h * d * texel;
    }
    total * u64::from(desc.array_layers.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::TexDim;
    use crate::types::{Format, TextureUsage};

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
    }
}
