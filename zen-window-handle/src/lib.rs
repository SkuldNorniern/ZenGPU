//! Minimal, dependency-free window and display handle types for the Zen
//! ecosystem (ZenGPU, aurea, and a future in-house winit).
//!
//! This mirrors the small subset of `raw-window-handle` that GPU surface
//! creation actually needs — the platform-native window/display pointers — with
//! **no external dependencies**, so it can sit at the root of the ecosystem
//! without forming a dependency cycle (e.g. ZenGPU ← this → aurea).
//!
//! A windowing library produces these; a GPU backend consumes them to build a
//! presentable surface. The handles are raw pointers/ids: the producer must
//! guarantee the underlying window outlives any use of the handle.
//!
//! Field shapes match `raw-window-handle` 0.6 so converting to/from it is
//! mechanical (see the crate README / consumer-side `From` impls).

#![no_std]

use core::ffi::c_void;
use core::num::{NonZeroIsize, NonZeroU32};
use core::ptr::NonNull;

/// A platform-native window handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WindowHandle {
    /// macOS / AppKit — an `NSView`.
    AppKit(AppKitWindowHandle),
    /// Windows — an `HWND` (+ owning `HINSTANCE`).
    Win32(Win32WindowHandle),
    /// Linux — an X11 window via XCB.
    Xcb(XcbWindowHandle),
    /// Linux — a Wayland surface.
    Wayland(WaylandWindowHandle),
}

/// A platform-native display/connection handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DisplayHandle {
    /// macOS / AppKit — no display object is needed.
    AppKit,
    /// Windows — no display object is needed.
    Windows,
    /// Linux — an X11 connection via XCB.
    Xcb(XcbDisplayHandle),
    /// Linux — a Wayland display.
    Wayland(WaylandDisplayHandle),
}

/// macOS `NSView` pointer.
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

/// Wayland surface pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaylandWindowHandle {
    pub surface: NonNull<c_void>,
}

/// X11 connection pointer (XCB); `None` lets the consumer open the default one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XcbDisplayHandle {
    pub connection: Option<NonNull<c_void>>,
}

/// Wayland display pointer.
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

// SAFETY: these carry raw pointers/ids that are inert data here; the producer
// guarantees validity and lifetime. Mirrors `raw-window-handle`, which also
// marks its handle types `Send + Sync`.
unsafe impl Send for WindowHandle {}
unsafe impl Sync for WindowHandle {}
unsafe impl Send for DisplayHandle {}
unsafe impl Sync for DisplayHandle {}
