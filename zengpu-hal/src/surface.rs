//! Window-handle bridge for presentable surfaces.
//!
//! [`WindowHandles`] bridges any `raw-window-handle`-capable window (aurea,
//! winit, etc.) to ZenGPU without pulling consumer types into the public API
//! Surfaces themselves are concrete per-feature backend types
//! (e.g. `Vulkan2dSurface`, `Vulkan3dSurface`) built on top of this — a
//! generic `GpuSurface`/`Surface` HAL trait is deferred until a second graphics
//! backend exists to shape it.

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

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
