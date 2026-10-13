//! The `CAMetalLayer` MoltenVK presents to, found or attached on an AppKit
//! view.

use objc::runtime::{Class, NO, Object, YES};
use objc::{msg_send, sel, sel_impl};
use zengpu_hal::{GpuError, Result};

/// `kCALayerWidthSizable | kCALayerHeightSizable`.
const LAYER_FILLS_PARENT: u64 = 2 | 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct CgRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

// SAFETY: matches CoreGraphics' CGRect, four doubles, as objc encodes it.
unsafe impl objc::Encode for CgRect {
    fn encode() -> objc::Encoding {
        // SAFETY: the string is CGRect's type encoding.
        unsafe { objc::Encoding::from_str("{CGRect={CGPoint=dd}{CGSize=dd}}") }
    }
}

/// The view's own `CAMetalLayer`, or a new one laid over it that tracks its
/// size and scale. The view's layer tree keeps the returned layer alive.
///
/// # Safety
/// `view` must be a live `NSView`, used on the main thread.
pub(crate) unsafe fn metal_layer_for(view: *mut Object) -> Result<*mut Object> {
    let metal_class = Class::get("CAMetalLayer")
        .ok_or_else(|| GpuError::Backend("CAMetalLayer unavailable".into()))?;
    unsafe {
        let existing: *mut Object = msg_send![view, layer];
        if !existing.is_null() {
            let is_metal: bool = msg_send![existing, isKindOfClass: metal_class];
            if is_metal {
                return Ok(existing);
            }
        }
        let _: () = msg_send![view, setWantsLayer: YES];
        let parent: *mut Object = msg_send![view, layer];
        if parent.is_null() {
            return Err(GpuError::Backend("the view has no layer".into()));
        }
        let layer: *mut Object = msg_send![metal_class, layer];
        let bounds: CgRect = msg_send![parent, bounds];
        let window: *mut Object = msg_send![view, window];
        let scale: f64 = if window.is_null() {
            1.0
        } else {
            msg_send![window, backingScaleFactor]
        };
        let _: () = msg_send![layer, setFrame: bounds];
        let _: () = msg_send![layer, setAutoresizingMask: LAYER_FILLS_PARENT];
        let _: () = msg_send![layer, setNeedsDisplayOnBoundsChange: NO];
        let _: () = msg_send![layer, setContentsScale: scale];
        let _: () = msg_send![parent, addSublayer: layer];
        Ok(layer)
    }
}
