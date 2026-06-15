//! Window-handle bridge for presentable surfaces.
//!
//! [`WindowHandles`] wraps any `raw-window-handle` window (such as winit) without exposing its
//! types. There is no generic surface trait: a second graphics backend is needed to shape one.

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

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
