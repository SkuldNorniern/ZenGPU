//! Window-handle bridge for presentable surfaces.
//!
//! [`WindowHandles`] carries the platform-native window + display handles a GPU
//! backend needs to create a presentable surface, using the dependency-free
//! [`zen_window_handle`] types (no `raw-window-handle` in the library). A
//! windowing library (winit today, an in-house one later) produces these; the
//! caller must guarantee the underlying window outlives any surface built from
//! them.

pub use zen_window_handle::{DisplayHandle, WindowHandle};

/// Platform window and display handles for presentable-surface creation.
#[derive(Clone, Copy)]
pub struct WindowHandles {
    pub window: WindowHandle,
    pub display: DisplayHandle,
}

impl WindowHandles {
    /// Construct directly from platform handles. The caller must ensure the
    /// underlying window outlives any surface created from these handles.
    pub fn from_raw(window: WindowHandle, display: DisplayHandle) -> Self {
        Self { window, display }
    }
}
