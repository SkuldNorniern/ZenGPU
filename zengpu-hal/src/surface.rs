//! Surface and swapchain HAL types.
//!
//! [`WindowHandles`] takes any `raw-window-handle` window (aurea, winit, etc.) without exposing
//! consumer types. [`GpuSurface`] is the backend-independent presentable surface.

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::desc::SurfaceConfig;
use crate::error::Result;

/// Window and display handles from any window implementing [`raw_window_handle::HasWindowHandle`]
/// and [`raw_window_handle::HasDisplayHandle`].
///
/// Handles are lifetime-erased. The window must outlive every surface made from them.
#[derive(Clone, Copy)]
pub struct WindowHandles {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
}

impl WindowHandles {
    /// Extracts the raw window and display handles.
    ///
    /// # Errors
    /// Returns `HandleError` if the window has no OS handle, such as a headless mock.
    pub fn from_window<W>(window: &W) -> core::result::Result<Self, raw_window_handle::HandleError>
    where
        W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    {
        Ok(Self {
            window: window.window_handle()?.as_raw(),
            display: window.display_handle()?.as_raw(),
        })
    }

    /// Wraps raw handles. The window must outlive any surface made from them.
    pub fn from_raw(window: RawWindowHandle, display: RawDisplayHandle) -> Self {
        Self { window, display }
    }
}

/// Token from [`GpuSurface::acquire_frame`], consumed by [`GpuSurface::present_frame`]. Holds the
/// swapchain image index, which is also the bindless slot of the render target.
#[must_use = "SurfaceFrame must be passed to present_frame or explicitly dropped"]
pub struct SurfaceFrame {
    pub index: u32,
}

/// Presentable surface: wraps the OS window, the swapchain, and the acquire/present handshake.
/// `Send + Sync`: backends guard swapchain state with a `Mutex`.
pub trait GpuSurface: Send + Sync {
    /// Configure or reconfigure the swapchain. Call after creation and on resize / surface loss.
    fn configure(&self, config: SurfaceConfig) -> Result<()>;

    /// Acquire the next swapchain image. Blocks up to the driver timeout.
    fn acquire_frame(&self) -> Result<SurfaceFrame>;

    fn present_frame(&self, frame: SurfaceFrame) -> Result<()>;

    /// Extent in pixels, `(width, height)`.
    fn size(&self) -> (u32, u32);

    fn image_count(&self) -> u32;
}
