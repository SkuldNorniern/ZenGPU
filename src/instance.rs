use zengpu_hal::{AdapterRequest, GpuInstance};

use crate::adapter::Adapter;

/// The ZenGPU entry-point. Holds a backend instance for every enabled feature
/// and collects their adapters under one roof — the same concept as `wgpu::Instance`.
///
/// Construct with [`Instance::new`]; then call [`enumerate_adapters`] or
/// [`request_adapter`] to get an [`Adapter`] and open a [`crate::Device`].
///
/// [`enumerate_adapters`]: Instance::enumerate_adapters
/// [`request_adapter`]: Instance::request_adapter
pub struct Instance {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl Instance {
    /// Create an instance for all compiled-in backends.
    ///
    /// Vulkan: tries to load surface extensions first; if the environment is
    /// headless (no display server / surface loader), falls back to a headless
    /// Vulkan instance so compute still works.
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
            // CudaInstance::new() never fails; it yields empty adapters when
            // CUDA is absent rather than returning an error.
            backends.push(Box::new(zengpu_cuda::CudaInstance::new()));
        }

        #[cfg(feature = "cpu")]
        backends.push(Box::new(zengpu_cpu::CpuInstance));

        Self { backends }
    }

    /// Like [`new`], but skips surface extension loading — for headless /
    /// server / CI environments where no display is available.
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

    /// All adapters from all enabled backends, in priority order:
    /// Vulkan → CUDA → CPU.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.backends
            .iter()
            .flat_map(|b| b.enumerate_adapters())
            .map(|inner| Adapter { inner })
            .collect()
    }

    /// Select the best single adapter matching `req`. Returns `None` if no
    /// backend can satisfy the request.
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
