use zengpu_hal::{AdapterRequest, BackendPreference, GpuInstance};

use crate::adapter::Adapter;

/// Entry point. Holds the backends registered via [`InstanceBuilder`]. Start with
/// [`Instance::builder()`].
pub struct Instance {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl Instance {
    /// Starts composing an [`Instance`]. Add backends with the builder methods, then call
    /// [`InstanceBuilder::build`].
    ///
    /// ```no_run
    /// # use zengpu::Instance;
    /// # #[cfg(feature = "vulkan")] {
    /// let instance = Instance::builder()
    ///     .vulkan_with_surface()?   // Err if Vulkan loader absent
    ///     .build();
    /// # }
    /// # Ok::<(), zengpu::GpuError>(())
    /// ```
    pub fn builder() -> InstanceBuilder {
        InstanceBuilder {
            backends: Vec::new(),
        }
    }

    /// Adapters from all backends, in registration order.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.backends
            .iter()
            .flat_map(|b| b.enumerate_adapters())
            .map(|inner| Adapter { inner })
            .collect()
    }

    /// Best adapter matching `req` across all backends, or `None`. A backend other than
    /// [`BackendPreference::Auto`] only matches adapters of that backend.
    pub fn request_adapter(&self, req: AdapterRequest) -> Option<Adapter> {
        self.backends
            .iter()
            .filter_map(|b| b.request_adapter(req))
            .find(|a| req.backend == BackendPreference::Auto || a.info().backend == req.backend)
            .map(|inner| Adapter { inner })
    }
}

/// Builds an [`Instance`] by explicitly adding backends.
///
/// Fallible methods (Vulkan, HIP) return `Result<Self>`, so the caller decides whether a missing
/// backend is an error or a graceful skip:
///
/// ```no_run
/// # use zengpu::Instance;
/// # #[cfg(feature = "vulkan")] {
/// // Error on Vulkan absence:
/// let inst = Instance::builder().vulkan_with_surface()?.build();
///
/// // Graceful skip (try_vulkan_with_surface consumes self either way):
/// let b = match Instance::builder().try_vulkan_with_surface() {
///     Ok(b) | Err(b) => b,
/// };
/// let inst = b.build();
/// # }
/// # Ok::<(), zengpu::GpuError>(())
/// ```
pub struct InstanceBuilder {
    backends: Vec<Box<dyn GpuInstance>>,
}

impl InstanceBuilder {
    /// Adds Vulkan with surface and swapchain extensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader or an extension is missing.
    ///
    /// For windowed apps. Headless compute uses [`vulkan`].
    ///
    /// [`vulkan`]: InstanceBuilder::vulkan
    #[cfg(feature = "vulkan")]
    pub fn vulkan_with_surface(mut self) -> zengpu_hal::Result<Self> {
        let inst = zengpu_vulkan::VulkanInstance::new_with_surface()?;
        self.backends.push(Box::new(inst));
        Ok(self)
    }

    /// Adds headless Vulkan (no surface extensions).
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader is absent.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(mut self) -> zengpu_hal::Result<Self> {
        let inst = zengpu_vulkan::VulkanInstance::new()?;
        self.backends.push(Box::new(inst));
        Ok(self)
    }

    /// Vulkan with surface extensions if available, otherwise `self` unchanged.
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

    /// Headless Vulkan if available, otherwise `self` unchanged.
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

    /// Add CUDA. Never fails here. Without CUDA at runtime it yields no adapters.
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self) -> Self {
        self.backends
            .push(Box::new(zengpu_cuda::CudaInstance::new()));
        self
    }

    /// Add Metal. Never fails here. Yields no adapters off Apple platforms.
    #[cfg(feature = "metal")]
    pub fn metal(mut self) -> Self {
        self.backends
            .push(Box::new(zengpu_metal::MetalInstance::new()));
        self
    }

    /// Adds HIP (ROCm).
    ///
    /// # Errors
    ///
    /// Returns an error if the HIP runtime is absent or fails to initialize.
    #[cfg(feature = "hip")]
    pub fn hip(mut self) -> zengpu_hal::Result<Self> {
        let inst = zengpu_hip::HipInstance::new()?;
        self.backends.push(Box::new(inst));
        Ok(self)
    }

    /// Add HIP if available, otherwise `self` unchanged.
    #[cfg(feature = "hip")]
    pub fn try_hip(self) -> Result<Self, Self> {
        match zengpu_hip::HipInstance::new() {
            Ok(inst) => {
                let mut s = self;
                s.backends.push(Box::new(inst));
                Ok(s)
            }
            Err(e) => {
                log::debug!("zengpu: HIP unavailable: {e}");
                Err(self)
            }
        }
    }

    /// Adds DX12. Never fails here. Yields no adapters off Windows. DXGI enumeration is not
    /// yet implemented.
    #[cfg(feature = "dx12")]
    pub fn dx12(mut self) -> Self {
        self.backends
            .push(Box::new(zengpu_dx12::Dx12Instance::new()));
        self
    }

    /// Add the CPU reference backend. Never fails.
    #[cfg(feature = "cpu")]
    pub fn cpu(mut self) -> Self {
        self.backends.push(Box::new(zengpu_cpu::CpuInstance));
        self
    }

    pub fn build(self) -> Instance {
        Instance {
            backends: self.backends,
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use zengpu_hal::{AdapterRequest, BackendPreference};

    use super::Instance;

    /// A named backend never falls back to another registered backend.
    #[test]
    fn named_backend_only_matches_its_own_adapters() {
        let inst = Instance::builder().metal().build();
        let want = |backend| AdapterRequest {
            backend,
            ..AdapterRequest::default()
        };
        assert!(
            inst.request_adapter(want(BackendPreference::Vulkan))
                .is_none()
        );
        assert!(
            inst.request_adapter(want(BackendPreference::Cuda))
                .is_none()
        );
        if let Some(a) = inst.request_adapter(want(BackendPreference::Auto)) {
            assert_eq!(a.info().backend, BackendPreference::Metal);
            let named = inst.request_adapter(want(BackendPreference::Metal));
            assert!(named.is_some());
        }
    }
}
