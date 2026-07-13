use zengpu_hal::{AdapterInfo, GpuInstance};

/// A backend that was found to have at least one usable device at runtime.
#[derive(Debug, Clone)]
pub struct BackendAvailability {
    /// Short identifier matching the Cargo feature name: `"vulkan"`, `"cuda"`,
    /// `"metal"`, `"hip"`, `"dx12"`, `"cpu"`.
    pub name: &'static str,
    /// All adapters the backend can see (at least one — empty backends are
    /// omitted from the result).
    pub adapters: Vec<AdapterInfo>,
}

/// Probe every backend enabled at compile time and return those that have at
/// least one device available at runtime.
///
/// Only features compiled in are checked; a build with only `--features cuda`
/// will never probe Vulkan. The returned list is in probe order:
/// Vulkan → CUDA → Metal → HIP → DX12 → CPU.
///
/// # Example
///
/// ```no_run
/// for b in zengpu::detect_backends() {
///     println!("{}: {} device(s)", b.name, b.adapters.len());
///     for a in &b.adapters {
///         println!("  - {}", a.name);
///     }
/// }
/// ```
pub fn detect_backends() -> Vec<BackendAvailability> {
    let mut out = Vec::new();
    probe_all(&mut out);
    out
}

fn push_if_nonempty(
    out: &mut Vec<BackendAvailability>,
    name: &'static str,
    inst: &dyn GpuInstance,
) {
    let adapters: Vec<AdapterInfo> = inst
        .enumerate_adapters()
        .into_iter()
        .map(|a| a.info().clone())
        .collect();
    if !adapters.is_empty() {
        out.push(BackendAvailability { name, adapters });
    }
}

fn probe_all(out: &mut Vec<BackendAvailability>) {
    #[cfg(feature = "vulkan")]
    {
        if let Ok(inst) = zengpu_vulkan::VulkanInstance::new() {
            push_if_nonempty(out, "vulkan", &inst);
        }
    }

    #[cfg(feature = "cuda")]
    {
        let inst = zengpu_cuda::CudaInstance::new();
        push_if_nonempty(out, "cuda", &inst);
    }

    #[cfg(feature = "metal")]
    {
        let inst = zengpu_metal::MetalInstance::new();
        push_if_nonempty(out, "metal", &inst);
    }

    #[cfg(feature = "hip")]
    {
        if let Ok(inst) = zengpu_hip::HipInstance::new() {
            push_if_nonempty(out, "hip", &inst);
        }
    }

    #[cfg(feature = "dx12")]
    {
        let inst = zengpu_dx12::Dx12Instance::new();
        push_if_nonempty(out, "dx12", &inst);
    }

    #[cfg(feature = "cpu")]
    {
        let inst = zengpu_cpu::CpuInstance;
        push_if_nonempty(out, "cpu", &inst);
    }
}
