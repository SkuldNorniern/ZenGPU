//! Optional GPU still-image codec boundary. No backend implements it yet.
//!
//! Capability queries describe actual backend implementations, not planned SDK
//! support. Unsupported calls must not silently encode or decode on the CPU.

use crate::TextureHandle;

/// Encoded still-image format. This does not imply hardware codec support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageCodec {
    Png,
    Jpeg,
    WebP,
    Avif,
    Bmp,
}

/// Support for one format on the current device. Defaults claim nothing.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageCodecCaps {
    pub decode: bool,
    pub encode: bool,
}

/// Decode into a caller-created RGBA8Unorm 2D texture, mip 0, layer 0.
///
/// An implementing backend must validate the header, destination extent/usage,
/// and decoded byte limit before submitting work. Encoded data must be copied or
/// consumed before returning, so the caller can reuse its input immediately.
/// Colour-profile conversion, EXIF orientation and animation are not implicit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct ImageDecodeRequest<'a> {
    pub codec: ImageCodec,
    pub encoded: &'a [u8],
    pub destination: TextureHandle,
    /// Maximum decoded RGBA bytes, not compressed input bytes. Must be nonzero.
    pub max_decoded_bytes: u64,
}

impl<'a> ImageDecodeRequest<'a> {
    pub fn new(
        codec: ImageCodec,
        encoded: &'a [u8],
        destination: TextureHandle,
        max_decoded_bytes: u64,
    ) -> Self {
        Self {
            codec,
            encoded,
            destination,
            max_decoded_bytes,
        }
    }
}

/// Encode mip 0, layer 0 of an RGBA8Unorm 2D texture. An implementing backend
/// owns its temporary output and returns encoded bytes through `ReadSubmission`.
/// Backend resources remain alive until completion, even if the caller drops
/// the job. The limit applies to encoded output; overflow must fail, not truncate
/// or return a successful partial image. Codec-specific settings are reserved
/// for a later implementation rather than silently ignored.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct ImageEncodeRequest {
    pub codec: ImageCodec,
    pub source: TextureHandle,
    /// Maximum encoded bytes. Must be nonzero.
    pub max_encoded_bytes: u64,
}

impl ImageEncodeRequest {
    pub fn new(codec: ImageCodec, source: TextureHandle, max_encoded_bytes: u64) -> Self {
        Self {
            codec,
            source,
            max_encoded_bytes,
        }
    }
}
