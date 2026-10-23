//! Presenting to an AppKit view through a `CAMetalLayer`.

use std::sync::{Arc, Mutex};

use core_graphics_types::geometry::{CGRect, CGSize};
use metal::foreign_types::ForeignTypeRef;
use metal::{MetalDrawable, MetalLayer, MetalLayerRef};
use objc::rc::autoreleasepool;
use objc::runtime::{Class, NO, Object, YES};
use objc::{msg_send, sel, sel_impl};
use zen_window_handle::WindowHandle;
use zengpu_hal::{
    Acquire, Frame, GpuError, GraphicsDevice, GraphicsPipelineDesc, PipelineHandle, PresentMode,
    QueryPoolHandle, Result, Surface, SurfaceConfig, TargetHandle, WindowHandles,
};

use super::command::MetalCommandList;
use super::convert;
use super::device::{Inner, MetalDevice, MetalTarget, lock};

const LAYER_FILLS_PARENT: u64 = 2 | 16;

/// Returns the view's `CAMetalLayer`, or creates one that tracks its size. An existing one is used
/// as is.
///
/// # Safety
/// `view` must be a live `NSView`, used on the main thread.
unsafe fn metal_layer_for(view: *mut Object) -> Result<MetalLayer> {
    let metal_class = Class::get("CAMetalLayer")
        .ok_or_else(|| GpuError::Backend("metal: CAMetalLayer unavailable".into()))?;
    unsafe {
        let existing: *mut Object = msg_send![view, layer];
        if !existing.is_null() {
            let is_metal: bool = msg_send![existing, isKindOfClass: metal_class];
            if is_metal {
                return Ok(MetalLayerRef::from_ptr(existing.cast()).to_owned());
            }
        }
        let _: () = msg_send![view, setWantsLayer: YES];
        let parent: *mut Object = msg_send![view, layer];
        if parent.is_null() {
            return Err(GpuError::Backend("metal: the view has no layer".into()));
        }
        let layer = MetalLayer::new();
        let bounds: CGRect = msg_send![parent, bounds];
        let scale: f64 = {
            let window: *mut Object = msg_send![view, window];
            if window.is_null() {
                1.0
            } else {
                msg_send![window, backingScaleFactor]
            }
        };
        let raw: *mut Object = layer.as_ref() as *const MetalLayerRef as *mut Object;
        let _: () = msg_send![raw, setFrame: bounds];
        let _: () = msg_send![raw, setAutoresizingMask: LAYER_FILLS_PARENT];
        let _: () = msg_send![raw, setNeedsDisplayOnBoundsChange: NO];
        layer.set_contents_scale(scale);
        let _: () = msg_send![parent, addSublayer: raw];
        Ok(layer)
    }
}

/// Swapchain backed by a `CAMetalLayer`. The layer owns the drawables and throttles to frames
/// in flight.
pub struct MetalSurface {
    inner: Arc<Inner>,
    layer: MetalLayer,
    config: Mutex<SurfaceConfig>,
}

// SAFETY: `CAMetalLayer` needs one thread at a time. The render thread satisfies that.
unsafe impl Send for MetalSurface {}
unsafe impl Sync for MetalSurface {}

impl MetalSurface {
    fn apply(&self, config: SurfaceConfig) {
        self.layer
            .set_pixel_format(convert::pixel_format(config.format));
        self.layer.set_drawable_size(CGSize::new(
            f64::from(config.width.max(1)),
            f64::from(config.height.max(1)),
        ));
        // iOS always syncs to the display, and has no such setting.
        if cfg!(target_os = "macos") {
            self.layer
                .set_display_sync_enabled(config.present_mode != PresentMode::Immediate);
        }
        self.layer.set_maximum_drawable_count(3);
    }
}

/// Drawable for one frame, used as a render target until presented or dropped.
pub struct MetalFrame {
    inner: Arc<Inner>,
    drawable: MetalDrawable,
    target: TargetHandle,
}

impl Frame for MetalFrame {
    fn target(&self) -> TargetHandle {
        self.target
    }
}

impl Drop for MetalFrame {
    fn drop(&mut self) {
        if let Ok(mut targets) = self.inner.targets.lock() {
            targets.remove(self.target);
        }
    }
}

impl Surface for MetalSurface {
    type Frame = MetalFrame;
    type CommandList = MetalCommandList;

    fn configure(&self, config: SurfaceConfig) -> Result<()> {
        self.apply(config);
        *lock(&self.config)? = config;
        Ok(())
    }

    fn resize(&self, width: u32, height: u32) -> Result<()> {
        let config = SurfaceConfig {
            width,
            height,
            ..*lock(&self.config)?
        };
        self.configure(config)
    }

    fn size(&self) -> (u32, u32) {
        self.config
            .lock()
            .map(|c| (c.width, c.height))
            .unwrap_or((0, 0))
    }

    fn acquire(&self) -> Result<Acquire<Self::Frame>> {
        let (w, h) = self.size();
        if w == 0 || h == 0 {
            return Ok(Acquire::Skip);
        }
        let Some(drawable) = autoreleasepool(|| self.layer.next_drawable().map(|d| d.to_owned()))
        else {
            return Ok(Acquire::Skip);
        };
        let tex = drawable.texture().to_owned();
        let format = lock(&self.config)?.format;
        let target = lock(&self.inner.targets)?.insert(MetalTarget { tex, format });
        Ok(Acquire::Frame(MetalFrame {
            inner: Arc::clone(&self.inner),
            drawable,
            target,
        }))
    }

    fn present(&self, frame: Self::Frame, list: Self::CommandList) -> Result<()> {
        let cmd = list.finish();
        cmd.present_drawable(&frame.drawable);
        self.inner.commit(&cmd);
        drop(frame);
        Ok(())
    }
}

impl GraphicsDevice for MetalDevice {
    type Surface = MetalSurface;
    type CommandList = MetalCommandList;

    fn create_surface(
        &self,
        window: &WindowHandles,
        config: SurfaceConfig,
    ) -> Result<Self::Surface> {
        let WindowHandle::AppKit(appkit) = window.window else {
            return Err(GpuError::Backend(
                "metal: expected an AppKit window handle".into(),
            ));
        };
        // SAFETY: `view` is a live `NSView`. Make surfaces on the main thread.
        let layer = unsafe { metal_layer_for(appkit.ns_view.as_ptr().cast())? };
        layer.set_device(&self.inner.device);
        layer.set_framebuffer_only(true);
        layer.set_presents_with_transaction(false);
        let surface = MetalSurface {
            inner: Arc::clone(&self.inner),
            layer,
            config: Mutex::new(config),
        };
        surface.apply(config);
        Ok(surface)
    }

    fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc<'_>) -> Result<PipelineHandle> {
        self.create_render_pipeline(desc)
    }

    fn create_command_list(&self) -> Result<Self::CommandList> {
        Ok(MetalCommandList::new(Arc::clone(&self.inner)))
    }

    fn create_query_pool(&self, _count: u32) -> Result<QueryPoolHandle> {
        Err(GpuError::Unsupported("metal: timestamp query pools".into()))
    }

    fn supports_dual_source_blending(&self) -> bool {
        true
    }

    fn supports_non_solid_fill(&self) -> bool {
        true
    }
}
