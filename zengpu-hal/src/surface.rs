//! Surface/swapchain HAL types (plan G2).
//!
//! [`WindowHandles`] bridges any `raw-window-handle`-capable window (aurea,
//! winit, etc.) to ZenGPU without pulling consumer types into the public API
//! (plan D10).  [`GpuSurface`] is the backend-independent presentable surface.

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::desc::SurfaceConfig;
use crate::error::Result;

/// Platform window and display handles extracted from any window that
/// implements [`raw_window_handle::HasWindowHandle`] +
/// [`raw_window_handle::HasDisplayHandle`].
///
/// Holds raw (lifetime-erased) handles; the caller must guarantee the
/// underlying window outlives any surface created from these handles.
#[derive(Clone, Copy)]
pub struct WindowHandles {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
}

// raw-window-handle 0.6 marks RawWindowHandle and RawDisplayHandle as
// Send + Sync, so WindowHandles inherits those impls automatically.

impl WindowHandles {
    /// Extract raw handles from any window.
    ///
    /// # Errors
    /// Returns the `HandleError` from `raw-window-handle` if the window is
    /// not backed by a real OS handle (e.g. a headless mock).
    pub fn from_window<W>(window: &W) -> core::result::Result<Self, raw_window_handle::HandleError>
    where
        W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    {
        Ok(Self {
            window: window.window_handle()?.as_raw(),
            display: window.display_handle()?.as_raw(),
        })
    }

    /// Construct directly from raw handles.  The caller must ensure the
    /// underlying window outlives any surface created from these handles.
    pub fn from_raw(window: RawWindowHandle, display: RawDisplayHandle) -> Self {
        Self { window, display }
    }
}

/// Token returned by [`GpuSurface::acquire_frame`] and consumed by
/// [`GpuSurface::present_frame`].  Carries the swapchain image index
/// (also the bindless slot index for the render target, plan D4).
#[must_use = "SurfaceFrame must be passed to present_frame or explicitly dropped"]
pub struct SurfaceFrame {
    pub index: u32,
}

/// A presentable surface: wraps the OS window, the swapchain, and the
/// acquire/present handshake.  `Send + Sync` — backends protect mutable
/// swapchain state with a `Mutex`.
pub trait GpuSurface: Send + Sync {
    /// Configure or reconfigure the swapchain (call once after creation and
    /// again on resize / surface-lost events).
    fn configure(&self, config: SurfaceConfig) -> Result<()>;

    /// Acquire the next swapchain image.  Blocks until an image is available
    /// (up to the driver timeout).
    fn acquire_frame(&self) -> Result<SurfaceFrame>;

    /// Submit and present the frame acquired by [`Self::acquire_frame`].
    fn present_frame(&self, frame: SurfaceFrame) -> Result<()>;

    /// Current surface extent in pixels `(width, height)`.
    fn size(&self) -> (u32, u32);

    /// Number of swapchain images.
    fn image_count(&self) -> u32;
}
