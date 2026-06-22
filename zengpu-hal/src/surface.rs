//! Window handles for presentable surfaces, from the dependency-free [`zen_window_handle`] crate.
//! The caller must keep the window alive for any surface built from them.

pub use zen_window_handle::{DisplayHandle, WindowHandle};

#[derive(Clone, Copy)]
pub struct WindowHandles {
    pub window: WindowHandle,
    pub display: DisplayHandle,
}

impl WindowHandles {
    /// The window must outlive any surface created from these handles.
    pub fn from_raw(window: WindowHandle, display: DisplayHandle) -> Self {
        Self { window, display }
    }
}
