//! Bindless table: an argument buffer of `{ texture, sampler }` pairs, read by ZSL's MSL as
//! `zsl_textures [[buffer(17)]]`.

use metal::foreign_types::ForeignType;
use metal::{
    ArgumentEncoder, Buffer, CompileOptions, DeviceRef, MTLResourceOptions, SamplerState, Texture,
};
use zengpu_hal::{GpuError, Result};

/// Push constants slot (ZSL `PUSH_BUFFER_INDEX`), above the vertex buffers.
pub(crate) const PUSH_BUFFER_INDEX: u64 = 16;
/// Texture table slot (ZSL `TEXTURE_BUFFER_INDEX`).
pub(crate) const TEXTURE_BUFFER_INDEX: u64 = 17;
/// Vertex buffer slots below the push constants.
pub(crate) const MAX_VERTEX_BUFFERS: u32 = PUSH_BUFFER_INDEX as u32;

/// Table size. A texture sits at its handle's slot index, as on Vulkan. This caps bound textures.
pub(crate) const TABLE_CAPACITY: u32 = 4096;
pub(crate) const TIER1_CAPACITY: usize = 16;
pub(crate) const TEXTURE_IDS_BUFFER_INDEX: u64 = 18;

/// Dummy shader with the table's layout, compiled so Metal hands out an encoder. Never run.
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
    pub(crate) tier1: bool,
    encoder: ArgumentEncoder,
    buffer: Buffer,
    stride: u64,
    /// Each slot's contents: passes make them resident, and compute binds them.
    bound: Vec<Option<(Texture, SamplerState)>>,
}

impl TextureTable {
    pub(crate) fn new(device: &DeviceRef) -> Result<Self> {
        Self::with_tier(
            device,
            device.argument_buffers_support() == metal::MTLArgumentBuffersTier::Tier1,
        )
    }

    pub(crate) fn with_tier(device: &DeviceRef, tier1: bool) -> Result<Self> {
        let tier1_probe = "#include <metal_stdlib>\nusing namespace metal;\nstruct Table { array<texture2d<float>,16> tex [[id(0)]]; array<sampler,16> smp [[id(16)]]; };\nfragment float4 zengpu_table_probe(constant Table& t [[buffer(17)]]) { return t.tex[0].sample(t.smp[0], float2(0.0)); }";
        let library = device
            .new_library_with_source(
                if tier1 { tier1_probe } else { PROBE_MSL },
                &CompileOptions::new(),
            )
            .map_err(|e| GpuError::Backend(format!("metal: texture table probe: {e}")))?;
        let function = library
            .get_function("zengpu_table_probe", None)
            .map_err(|e| GpuError::Backend(format!("metal: texture table probe: {e}")))?;
        let encoder = function.new_argument_encoder(TEXTURE_BUFFER_INDEX);
        let align = encoder.alignment().max(1);
        let stride = encoder.encoded_length().div_ceil(align) * align;
        let buffer = device.new_buffer(
            if tier1 {
                stride
            } else {
                stride * u64::from(TABLE_CAPACITY)
            },
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            tier1,
            encoder,
            buffer,
            stride,
            bound: vec![None; TABLE_CAPACITY as usize],
        })
    }

    /// The slot must not be read by in-flight work, as with Vulkan's update-after-bind.
    pub(crate) fn bind(&mut self, slot: u32, texture: &Texture, sampler: &SamplerState) -> bool {
        let Some(entry) = self.bound.get_mut(slot as usize) else {
            return false;
        };
        if !self.tier1 {
            self.encoder
                .set_argument_buffer(&self.buffer, u64::from(slot) * self.stride);
            self.encoder.set_texture(0, texture);
            self.encoder.set_sampler_state(1, sampler);
        }
        *entry = Some((texture.clone(), sampler.clone()));
        true
    }

    /// A fresh buffer per draw: later bindings must not overwrite in-flight arguments.
    pub(crate) fn snapshot(
        &self,
        device: &DeviceRef,
        slots: &[u32],
    ) -> Result<(Buffer, [u32; TIER1_CAPACITY], Vec<Texture>)> {
        let slots: Vec<u32> = if slots.is_empty() {
            // Older callers put a table ID in scalars. Keep working for small tables.
            if self.used() as usize > TIER1_CAPACITY {
                return Err(GpuError::Unsupported("metal tier 1: list the draw's texture IDs in Bindings::textures when more than 16 textures are bound".into()));
            }
            self.bound
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.as_ref().map(|_| i as u32))
                .take(TIER1_CAPACITY)
                .collect()
        } else {
            slots.to_vec()
        };
        if slots.len() > TIER1_CAPACITY {
            return Err(GpuError::Unsupported(
                "metal tier 1: at most 16 textures per draw; split the draw".into(),
            ));
        }
        let buffer = device.new_buffer(
            self.encoder.encoded_length(),
            MTLResourceOptions::StorageModeShared,
        );
        self.encoder.set_argument_buffer(&buffer, 0);
        let mut ids = [u32::MAX; TIER1_CAPACITY];
        let mut resident = Vec::new();
        for (i, slot) in slots.into_iter().enumerate() {
            let (texture, sampler) = self.get(slot).ok_or_else(|| {
                GpuError::Backend(format!("metal tier 1: texture slot {slot} is not bound"))
            })?;
            self.encoder.set_texture(i as u64, &texture);
            self.encoder
                .set_sampler_state((TIER1_CAPACITY + i) as u64, &sampler);
            ids[i] = slot;
            resident.push(texture);
        }
        Ok((buffer, ids, resident))
    }

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

    /// A pass must make bound textures resident before shaders reach them.
    pub(crate) fn resident(&self) -> Vec<Texture> {
        self.bound
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect()
    }

    pub(crate) fn get(&self, slot: u32) -> Option<(Texture, SamplerState)> {
        self.bound.get(slot as usize)?.clone()
    }

    pub(crate) fn used(&self) -> u32 {
        u32::try_from(self.bound.iter().flatten().count()).unwrap_or(u32::MAX)
    }
}
