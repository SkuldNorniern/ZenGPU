//! Placeholder for a future OpenGL backend for older hardware.
//!
//! No context, adapter, device, shader compiler, or presentation path is
//! implemented. Enabling this crate never loads a driver or claims GPU support.
//! Context ownership and thread affinity, shader lowering, resource binding,
//! capability limits, and window presentation need implementations before this
//! backend can enumerate a usable adapter. No minimum GL version is promised yet.

use zengpu_hal::{AdapterRequest, GpuAdapter, GpuInstance};

/// Whether this backend can provide usable adapters. Always false for this stub.
pub const IMPLEMENTED: bool = false;

/// Reserved instance entry point. Enumerates no adapters on every platform.
#[derive(Debug, Default)]
pub struct OpenGlInstance;

impl OpenGlInstance {
    pub fn new() -> Self {
        Self
    }
}

impl GpuInstance for OpenGlInstance {
    fn enumerate_adapters(&self) -> Vec<Box<dyn GpuAdapter>> {
        Vec::new()
    }

    fn request_adapter(&self, _request: AdapterRequest) -> Option<Box<dyn GpuAdapter>> {
        None
    }
}
