//! Checked transfer geometry, shared by every Metal storage mode.

use zengpu_hal::{
    BufferUsage, Format, GpuError, Result, TexDim, TextureDesc, TextureRegion, TextureUsage,
    UsageError,
};

pub(crate) struct Layout {
    pub row_bytes: u32,
    pub image_bytes: u64,
    pub span_bytes: u64,
    pub texel_bytes: u32,
}

fn invalid(reason: &str) -> GpuError {
    UsageError::BindingMismatch(format!("metal: {reason}")).into()
}

pub(crate) fn texture_layout(
    desc: &TextureDesc,
    region: TextureRegion,
    pitch: u32,
    needed: TextureUsage,
    allow_packed: bool,
) -> Result<Layout> {
    if desc.format == Format::Depth24PlusStencil8 {
        return Err(GpuError::Unsupported(
            "metal: packed depth/stencil transfers".into(),
        ));
    }
    if !desc.usage.contains(needed) {
        return Err(UsageError::MissingUsage {
            resource: "texture",
            needed: if needed == TextureUsage::TRANSFER_SRC {
                "TRANSFER_SRC"
            } else {
                "TRANSFER_DST"
            },
        }
        .into());
    }
    let layers = if desc.dimension == TexDim::Cube {
        6
    } else {
        desc.array_layers.max(1)
    };
    if region.mip_level >= desc.mip_levels.max(1)
        || region.mip_level >= u32::BITS
        || region.layer >= layers
        || region.width == 0
        || region.height == 0
        || desc.samples > 1
    {
        return Err(invalid("invalid texture subresource"));
    }
    let width = (desc.width >> region.mip_level).max(1);
    let height = (desc.height >> region.mip_level).max(1);
    if region
        .x
        .checked_add(region.width)
        .is_none_or(|end| end > width)
        || region
            .y
            .checked_add(region.height)
            .is_none_or(|end| end > height)
    {
        return Err(invalid("texture region out of bounds"));
    }
    let texel_bytes = desc.format.bytes_per_texel();
    let packed = region
        .width
        .checked_mul(texel_bytes)
        .ok_or_else(|| invalid("texture row size overflow"))?;
    let row_bytes = if pitch == 0 && allow_packed {
        packed
    } else {
        pitch
    };
    if row_bytes < packed {
        return Err(invalid("invalid texture row pitch"));
    }
    let image_bytes = u64::from(row_bytes) * u64::from(region.height);
    let span_bytes = u64::from(row_bytes) * u64::from(region.height - 1) + u64::from(packed);
    Ok(Layout {
        row_bytes,
        image_bytes,
        span_bytes,
        texel_bytes,
    })
}

pub(crate) fn data_len(layout: &Layout, len: u64) -> Result<()> {
    if len < layout.span_bytes {
        Err(invalid("texture data shorter than its region"))
    } else {
        Ok(())
    }
}

pub(crate) fn buffer_range(
    layout: &Layout,
    size: u64,
    usage: BufferUsage,
    offset: u64,
    needed: BufferUsage,
) -> Result<()> {
    if !usage.contains(needed) {
        return Err(UsageError::MissingUsage {
            resource: "buffer",
            needed: if needed == BufferUsage::TRANSFER_SRC {
                "TRANSFER_SRC"
            } else {
                "TRANSFER_DST"
            },
        }
        .into());
    }
    if !offset.is_multiple_of(u64::from(layout.texel_bytes))
        || !layout.row_bytes.is_multiple_of(layout.texel_bytes)
        || offset
            .checked_add(layout.span_bytes)
            .is_none_or(|end| end > size)
    {
        return Err(invalid("buffer transfer range or alignment is invalid"));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use zengpu_hal::Format;

    fn desc() -> TextureDesc {
        TextureDesc {
            width: 8,
            height: 4,
            depth: 1,
            format: Format::Rgba8Unorm,
            usage: TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST,
            samples: 1,
            dimension: TexDim::D2,
            mip_levels: 4,
            array_layers: 2,
        }
    }

    #[test]
    fn rejects_bad_subresources_extents_and_pitch() {
        let d = desc();
        let valid = TextureRegion::rect(0, 0, 3, 2);
        for region in [
            TextureRegion {
                mip_level: 32,
                ..valid
            },
            TextureRegion {
                mip_level: 4,
                ..valid
            },
            TextureRegion { layer: 2, ..valid },
            TextureRegion {
                x: u32::MAX,
                ..valid
            },
            TextureRegion {
                y: u32::MAX,
                ..valid
            },
            TextureRegion { width: 0, ..valid },
            TextureRegion { width: 9, ..valid },
        ] {
            assert!(texture_layout(&d, region, 16, TextureUsage::TRANSFER_DST, false).is_err());
        }
        for pitch in [0, 8] {
            assert!(texture_layout(&d, valid, pitch, TextureUsage::TRANSFER_DST, false).is_err());
        }
        let layout = texture_layout(&d, valid, 16, TextureUsage::TRANSFER_DST, false).unwrap();
        assert_eq!((layout.span_bytes, layout.image_bytes), (28, 32));
        assert!(data_len(&layout, 27).is_err());
        assert!(data_len(&layout, 28).is_ok());
        let padded = texture_layout(&d, valid, 13, TextureUsage::TRANSFER_DST, false).unwrap();
        assert!(data_len(&padded, 25).is_ok());
        assert!(
            buffer_range(
                &padded,
                32,
                BufferUsage::TRANSFER_DST,
                0,
                BufferUsage::TRANSFER_DST
            )
            .is_err()
        );
        let mut depth_stencil = d;
        depth_stencil.format = Format::Depth24PlusStencil8;
        assert!(matches!(
            texture_layout(&depth_stencil, valid, 16, TextureUsage::TRANSFER_DST, false),
            Err(GpuError::Unsupported(_))
        ));
    }

    #[test]
    fn rejects_wrong_usage_overflow_and_short_buffers() {
        let mut d = desc();
        let r = TextureRegion::rect(0, 0, 3, 2);
        let layout = texture_layout(&d, r, 0, TextureUsage::TRANSFER_SRC, true).unwrap();
        assert_eq!(layout.row_bytes, 12);
        assert!(
            buffer_range(
                &layout,
                28,
                BufferUsage::TRANSFER_DST,
                4,
                BufferUsage::TRANSFER_DST
            )
            .is_ok()
        );
        assert!(
            buffer_range(
                &layout,
                27,
                BufferUsage::TRANSFER_DST,
                4,
                BufferUsage::TRANSFER_DST
            )
            .is_err()
        );
        assert!(
            buffer_range(
                &layout,
                u64::MAX,
                BufferUsage::TRANSFER_DST,
                u64::MAX - 3,
                BufferUsage::TRANSFER_DST
            )
            .is_err()
        );
        assert!(
            buffer_range(
                &layout,
                28,
                BufferUsage::TRANSFER_DST,
                1,
                BufferUsage::TRANSFER_DST
            )
            .is_err()
        );
        assert!(
            buffer_range(
                &layout,
                28,
                BufferUsage::TRANSFER_SRC,
                0,
                BufferUsage::TRANSFER_DST
            )
            .is_err()
        );
        d.usage = TextureUsage::SAMPLED;
        assert!(texture_layout(&d, r, 12, TextureUsage::TRANSFER_DST, false).is_err());
        d.usage = TextureUsage::TRANSFER_DST;
        d.width = u32::MAX;
        assert!(
            texture_layout(
                &d,
                TextureRegion::rect(0, 0, u32::MAX, 1),
                u32::MAX,
                TextureUsage::TRANSFER_DST,
                false
            )
            .is_err()
        );
    }
}
