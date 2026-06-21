use zengpu_hal::{AdapterRequest, GpuInstance};

use crate::adapter::Adapter;

/// Holds a backend instance for every enabled feature and collects their adapters. Same role
/// as `wgpu::Instance`.
///
/// Open a [`crate::Device`] from an adapter via [`enumerate_adapters`] or [`request_adapter`].
///
/// [`enumerate_adapters`]: Instance::enumerate_adapters [`request_adapter`]:
/// Instance::request_adapter
pub struct Instance {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl Instance {
    /// Vulkan: loads surface extensions. Falls back to headless (no display server) so compute
    /// still works.
    pub fn new() -> Self {
        let mut backends: Vec<Box<dyn GpuInstance>> = Vec::new();

        #[cfg(feature = "vulkan")]
        {
            let vk = zengpu_vulkan::VulkanInstance::new_with_surface()
                .or_else(|_| zengpu_vulkan::VulkanInstance::new());
            match vk {
                Ok(inst) => backends.push(Box::new(inst)),
                Err(e) => log::warn!("zengpu: Vulkan instance unavailable: {e}"),
            }
        }

        #[cfg(feature = "cuda")]
        {
            backends.push(Box::new(zengpu_cuda::CudaInstance::new()));
        }

        #[cfg(feature = "cpu")]
        backends.push(Box::new(zengpu_cpu::CpuInstance));

        Self { backends }
    }

    /// Like [`new`], but skips surface extension loading. For headless or CI use.
    ///
    /// [`new`]: Instance::new
    pub fn headless() -> Self {
        let mut backends: Vec<Box<dyn GpuInstance>> = Vec::new();

        #[cfg(feature = "vulkan")]
        match zengpu_vulkan::VulkanInstance::new() {
            Ok(inst) => backends.push(Box::new(inst)),
            Err(e) => log::warn!("zengpu: Vulkan instance unavailable: {e}"),
        }

        #[cfg(feature = "cuda")]
        backends.push(Box::new(zengpu_cuda::CudaInstance::new()));

        #[cfg(feature = "cpu")]
        backends.push(Box::new(zengpu_cpu::CpuInstance));

        Self { backends }
    }

    /// Adapters from all enabled backends, in priority order: Vulkan, CUDA, CPU.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.backends
            .iter()
            .flat_map(|b| b.enumerate_adapters())
            .map(|inner| Adapter { inner })
            .collect()
    }

    pub fn request_adapter(&self, req: AdapterRequest) -> Option<Adapter> {
        for backend in &self.backends {
            if let Some(inner) = backend.request_adapter(req) {
                return Some(Adapter { inner });
            }
        }
        None
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}
