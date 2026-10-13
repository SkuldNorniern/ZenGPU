//! A sampled render target for offscreen passes: draw into it, then sample it like any texture.

use std::sync::Arc;

use zengpu_hal::{
    Format, GpuDevice, GpuError, Result, TargetHandle, TexDim, TextureDesc, TextureHandle,
    TextureUsage,
};

use super::device::{Inner, MetalDevice};

/// Render target that is also sampled. Matches `zengpu_vulkan::OffscreenTarget`. Both drop with it.
pub struct OffscreenTarget {
    inner: Arc<Inner>,
    texture: TextureHandle,
    target: TargetHandle,
    format: Format,
    extent: (u32, u32),
}

impl OffscreenTarget {
    pub fn new(device: &MetalDevice, format: Format, width: u32, height: u32) -> Result<Self> {
        let texture = device.create_texture(TextureDesc {
            width,
            height,
            depth: 1,
            format,
            usage: TextureUsage::SAMPLED | TextureUsage::RENDER_TARGET | TextureUsage::TRANSFER_SRC,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 1,
            array_layers: 1,
        })?;
        let target = device.register_color_target(texture).ok_or_else(|| {
            device.destroy_texture(texture);
            GpuError::Backend("register_color_target: stale texture handle".to_string())
        })?;
        Ok(Self {
            inner: Arc::clone(&device.inner),
            texture,
            target,
            format,
            extent: (width, height),
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }

    pub fn target_handle(&self) -> TargetHandle {
        self.target
    }

    pub fn texture_handle(&self) -> TextureHandle {
        self.texture
    }

    /// Recreates the target at a new size. The old texture and target are dropped.
    pub fn resize(&mut self, device: &MetalDevice, width: u32, height: u32) -> Result<()> {
        *self = Self::new(device, self.format, width, height)?;
        Ok(())
    }
}

impl Drop for OffscreenTarget {
    fn drop(&mut self) {
        if let Ok(mut targets) = self.inner.targets.lock() {
            targets.remove(self.target);
        }
        let removed = self
            .inner
            .textures
            .lock()
            .ok()
            .and_then(|mut t| t.remove(self.texture));
        if let (Some(t), Ok(mut table)) = (removed, self.inner.table.lock()) {
            table.unbind(self.texture.index(), &t.tex);
        }
    }
}
