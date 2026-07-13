//! Dependency-free window and display handles for the Zen ecosystem.
//!
//! Mirrors the subset of `raw-window-handle` 0.6 that GPU surfaces need, so it can sit at the root
//! of the ecosystem without a cycle. Field shapes match 0.6. Handles are raw: the producer keeps
//! the window alive while the handle is in use.

#![no_std]

use core::ffi::c_void;
use core::num::{NonZeroIsize, NonZeroU32};
use core::ptr::NonNull;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WindowHandle {
    /// macOS / AppKit: `NSView`.
    AppKit(AppKitWindowHandle),
    /// Windows: `HWND` (+ owning `HINSTANCE`).
    Win32(Win32WindowHandle),
    /// Linux X11 window via XCB.
    Xcb(XcbWindowHandle),
    /// Linux Wayland surface.
    Wayland(WaylandWindowHandle),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DisplayHandle {
    /// macOS / AppKit: no display object.
    AppKit,
    /// Windows: no display object.
    Windows,
    /// Linux X11 connection via XCB.
    Xcb(XcbDisplayHandle),
    /// Linux Wayland display.
    Wayland(WaylandDisplayHandle),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AppKitWindowHandle {
    pub ns_view: NonNull<c_void>,
}

/// Windows `HWND` plus its owning `HINSTANCE` (if known).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win32WindowHandle {
    pub hwnd: NonZeroIsize,
    pub hinstance: Option<NonZeroIsize>,
}

/// X11 window id (XCB).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XcbWindowHandle {
    pub window: NonZeroU32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaylandWindowHandle {
    pub surface: NonNull<c_void>,
}

/// X11 connection pointer (XCB); `None` lets the consumer open the default one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XcbDisplayHandle {
    pub connection: Option<NonNull<c_void>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaylandDisplayHandle {
    pub display: NonNull<c_void>,
}

impl AppKitWindowHandle {
    pub fn new(ns_view: NonNull<c_void>) -> Self {
        Self { ns_view }
    }
}

impl Win32WindowHandle {
    pub fn new(hwnd: NonZeroIsize) -> Self {
        Self {
            hwnd,
            hinstance: None,
        }
    }
}

impl XcbWindowHandle {
    pub fn new(window: NonZeroU32) -> Self {
        Self { window }
    }
}

impl WaylandWindowHandle {
    pub fn new(surface: NonNull<c_void>) -> Self {
        Self { surface }
    }
}

// SAFETY: inert raw pointers/ids here. The producer guarantees validity and lifetime.
unsafe impl Send for WindowHandle {}
unsafe impl Sync for WindowHandle {}
unsafe impl Send for DisplayHandle {}
unsafe impl Sync for DisplayHandle {}
