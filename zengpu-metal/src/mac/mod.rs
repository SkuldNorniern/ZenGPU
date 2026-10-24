//! The Metal backend proper, on macOS and iOS.

mod command;
mod convert;
mod device;
mod instance;
mod offscreen;
mod pipeline;
mod surface;
mod table;

pub use command::MetalCommandList;
pub use device::{DepthTarget, MetalDevice};
pub use instance::{MetalAdapter, MetalInstance};
pub use offscreen::OffscreenTarget;
pub use surface::{MetalFrame, MetalSurface};
