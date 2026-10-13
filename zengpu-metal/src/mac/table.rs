//! The bindless texture table: one argument buffer of `{ texture, sampler }`
//! pairs, the layout ZSL's MSL graphics shaders read as
//! `const device ZslTexture* zsl_textures [[buffer(17)]]`.

use metal::foreign_types::ForeignType;
use metal::{
    ArgumentEncoder, Buffer, CompileOptions, DeviceRef, MTLResourceOptions, SamplerState, Texture,
};
use zengpu_hal::{GpuError, Result};

/// Where graphics shaders find their push constants (ZSL's
/// `PUSH_BUFFER_INDEX`), above any vertex buffer slot.
pub(crate) const PUSH_BUFFER_INDEX: u64 = 16;
/// Where graphics shaders find the texture table (ZSL's
/// `TEXTURE_BUFFER_INDEX`).
pub(crate) const TEXTURE_BUFFER_INDEX: u64 = 17;
/// Vertex buffer slots below the push constants.
pub(crate) const MAX_VERTEX_BUFFERS: u32 = PUSH_BUFFER_INDEX as u32;

/// Slots in the table. A texture sits at its own handle's slot index, as on
/// Vulkan, so this bounds how many textures can be bound at once.
pub(crate) const TABLE_CAPACITY: u32 = 4096;

/// A shader with the table's exact layout, compiled only so Metal will hand
/// out an encoder for it: the encoder knows the layout, the shader is never
/// run.
const PROBE_MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct ZslTexture {
    texture2d<float> tex;
    sampler smp;
};
fragment float4 zengpu_table_probe(const device ZslTexture* zsl_textures [[buffer(17)]]) {
    return zsl_textures[0].tex.sample(zsl_textures[0].smp, float2(0.0));
}
";

pub(crate) struct TextureTable {
    encoder: ArgumentEncoder,
    buffer: Buffer,
    stride: u64,
    /// What each slot holds, kept so passes can make them resident and
    /// compute can bind them directly.
    bound: Vec<Option<(Texture, SamplerState)>>,
}

impl TextureTable {
    pub(crate) fn new(device: &DeviceRef) -> Result<Self> {
        let library = device
            .new_library_with_source(PROBE_MSL, &CompileOptions::new())
            .map_err(|e| GpuError::Backend(format!("metal: texture table probe: {e}")))?;
        let function = library
            .get_function("zengpu_table_probe", None)
            .map_err(|e| GpuError::Backend(format!("metal: texture table probe: {e}")))?;
        let encoder = function.new_argument_encoder(TEXTURE_BUFFER_INDEX);
        let align = encoder.alignment().max(1);
        let stride = encoder.encoded_length().div_ceil(align) * align;
        let buffer = device.new_buffer(
            stride * u64::from(TABLE_CAPACITY),
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            encoder,
            buffer,
            stride,
            bound: vec![None; TABLE_CAPACITY as usize],
        })
    }

    /// Write `texture` and `sampler` into `slot`. The slot is assumed not to
    /// be read by work still in flight, as on Vulkan's update-after-bind set.
    pub(crate) fn bind(&mut self, slot: u32, texture: &Texture, sampler: &SamplerState) -> bool {
        let Some(entry) = self.bound.get_mut(slot as usize) else {
            return false;
        };
        self.encoder
            .set_argument_buffer(&self.buffer, u64::from(slot) * self.stride);
        self.encoder.set_texture(0, texture);
        self.encoder.set_sampler_state(1, sampler);
        *entry = Some((texture.clone(), sampler.clone()));
        true
    }

    /// Forget `slot` if it holds `texture`.
    pub(crate) fn unbind(&mut self, slot: u32, texture: &Texture) {
        if let Some(entry) = self.bound.get_mut(slot as usize)
            && entry
                .as_ref()
                .is_some_and(|(t, _)| t.as_ptr() == texture.as_ptr())
        {
            *entry = None;
        }
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Every bound texture, which a pass must declare resident before its
    /// shaders can reach them through the table.
    pub(crate) fn resident(&self) -> Vec<Texture> {
        self.bound
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect()
    }

    /// The texture and sampler in `slot`, if it holds one.
    pub(crate) fn get(&self, slot: u32) -> Option<(Texture, SamplerState)> {
        self.bound.get(slot as usize)?.clone()
    }

    pub(crate) fn used(&self) -> u32 {
        u32::try_from(self.bound.iter().flatten().count()).unwrap_or(u32::MAX)
    }
}
