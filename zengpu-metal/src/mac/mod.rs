//! The Metal backend proper, on macOS.

mod command;
mod convert;
mod device;
mod instance;
mod pipeline;
mod surface;
mod table;

pub use command::MetalCommandList;
pub use device::MetalDevice;
pub use instance::{MetalAdapter, MetalInstance};
pub use surface::{MetalFrame, MetalSurface};
