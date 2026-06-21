use zengpu_hal::{AdapterRequest, GpuInstance};

use crate::adapter::Adapter;

/// The ZenGPU entry-point. Holds the set of backend instances the caller
/// explicitly registered via [`InstanceBuilder`]. Obtain one with
/// [`Instance::builder()`].
pub struct Instance {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl Instance {
    /// Start composing an [`Instance`]. Add backends with the `with_*` methods,
    /// then call [`InstanceBuilder::build`].
    ///
    /// ```no_run
    /// # use zengpu::Instance;
    /// let instance = Instance::builder()
    ///     .vulkan_with_surface()?   // Err if Vulkan loader absent
    ///     .cuda()                   // no-op if CUDA absent at runtime
    ///     .build();
    /// # Ok::<(), zengpu::GpuError>(())
    /// ```
    pub fn builder() -> InstanceBuilder {
        InstanceBuilder {
            backends: Vec::new(),
        }
    }

    /// All adapters from every registered backend, in the order the backends
    /// were added.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.backends
            .iter()
            .flat_map(|b| b.enumerate_adapters())
            .map(|inner| Adapter { inner })
            .collect()
    }

    /// Select the best single adapter matching `req` across all registered
    /// backends. Returns `None` if no backend satisfies the request.
    pub fn request_adapter(&self, req: AdapterRequest) -> Option<Adapter> {
        for backend in &self.backends {
            if let Some(inner) = backend.request_adapter(req) {
                return Some(Adapter { inner });
            }
        }
        None
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Constructs an [`Instance`] by explicitly adding backends.
///
/// Fallible methods (Vulkan) return `Result<Self>` so the caller decides
/// whether a missing backend is an error or a graceful skip:
///
/// ```no_run
/// # use zengpu::Instance;
/// // Error on Vulkan absence:
/// let inst = Instance::builder().vulkan_with_surface()?.build();
///
/// // Graceful skip:
/// let mut b = Instance::builder();
/// if let Ok(b2) = b.try_vulkan_with_surface() { b = b2; }
/// let inst = b.cuda().build();
/// # Ok::<(), zengpu::GpuError>(())
/// ```
pub struct InstanceBuilder {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl InstanceBuilder {
    // ── Vulkan ────────────────────────────────────────────────────────────────

    /// Add the Vulkan backend with surface/swapchain extensions loaded.
    /// Returns `Err` if the Vulkan loader or a required extension is absent.
    ///
    /// Use this for windowed applications; use [`vulkan`] for headless compute.
    ///
    /// [`vulkan`]: InstanceBuilder::vulkan
    #[cfg(feature = "vulkan")]
    pub fn vulkan_with_surface(mut self) -> zengpu_hal::Result<Self> {
        let inst = zengpu_vulkan::VulkanInstance::new_with_surface()?;
        self.backends.push(Box::new(inst));
        Ok(self)
    }

    /// Add the Vulkan backend without surface extensions (headless / compute).
    /// Returns `Err` if the Vulkan loader is absent.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(mut self) -> zengpu_hal::Result<Self> {
        let inst = zengpu_vulkan::VulkanInstance::new()?;
        self.backends.push(Box::new(inst));
        Ok(self)
    }

    /// Try to add Vulkan with surface extensions; silently returns `self`
    /// unchanged if Vulkan is unavailable. Useful when Vulkan is optional.
    #[cfg(feature = "vulkan")]
    pub fn try_vulkan_with_surface(self) -> Result<Self, Self> {
        match zengpu_vulkan::VulkanInstance::new_with_surface() {
            Ok(inst) => {
                let mut s = self;
                s.backends.push(Box::new(inst));
                Ok(s)
            }
            Err(e) => {
                log::debug!("zengpu: Vulkan (with surface) unavailable: {e}");
                Err(self)
            }
        }
    }

    /// Try to add headless Vulkan; silently returns `self` unchanged if Vulkan
    /// is unavailable.
    #[cfg(feature = "vulkan")]
    pub fn try_vulkan(self) -> Result<Self, Self> {
        match zengpu_vulkan::VulkanInstance::new() {
            Ok(inst) => {
                let mut s = self;
                s.backends.push(Box::new(inst));
                Ok(s)
            }
            Err(e) => {
                log::debug!("zengpu: Vulkan unavailable: {e}");
                Err(self)
            }
        }
    }

    // ── CUDA ──────────────────────────────────────────────────────────────────

    /// Add the CUDA backend. Never fails at construction — if CUDA is absent at
    /// runtime, the backend simply yields no adapters from
    /// [`Instance::enumerate_adapters`].
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self) -> Self {
        self.backends.push(Box::new(zengpu_cuda::CudaInstance::new()));
        self
    }

    // ── CPU ───────────────────────────────────────────────────────────────────

    /// Add the CPU reference backend (always available; never fails).
    #[cfg(feature = "cpu")]
    pub fn cpu(mut self) -> Self {
        self.backends.push(Box::new(zengpu_cpu::CpuInstance));
        self
    }

    // ── Finalise ──────────────────────────────────────────────────────────────

    /// Consume the builder and produce an [`Instance`] with the registered
    /// backends.
    pub fn build(self) -> Instance {
        Instance {
            backends: self.backends,
        }
    }
}
