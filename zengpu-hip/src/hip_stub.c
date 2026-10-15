/* Link-only stub for libamdhip64. Compiled in when no ROCm installation is
 * found, so the crate still links on machines without HIP installed. Every
 * function returns hipErrorNotSupported (801); nothing here is meant to be
 * called at runtime: HipInstance::new() probes hipInit()'s result and
 * reports unavailability before any other stubbed symbol would be reached.
 * Keep this list in sync with the extern block in lib.rs. */
#include <stddef.h>
#define HIP_STUB(name) int name(void) { return 801; }

HIP_STUB(hipInit)
HIP_STUB(hipGetDeviceCount)
HIP_STUB(hipDeviceGetName)
HIP_STUB(hipDeviceTotalMem)
HIP_STUB(hipGetDeviceProperties)
HIP_STUB(hipGetDevicePropertiesR0600)
HIP_STUB(hipDeviceGetAttribute)
HIP_STUB(hipDeviceComputeCapability)
HIP_STUB(hipSetDevice)
HIP_STUB(hipDeviceCanAccessPeer)
HIP_STUB(hipDeviceEnablePeerAccess)
HIP_STUB(hipMemGetInfo)
HIP_STUB(hipMalloc)
HIP_STUB(hipFree)
HIP_STUB(hipHostMalloc)
HIP_STUB(hipHostFree)
HIP_STUB(hipHostGetDevicePointer)
HIP_STUB(hipMemcpy)
HIP_STUB(hipMemcpyAsync)
HIP_STUB(hipMemcpyPeer)
HIP_STUB(hipStreamCreateWithFlags)
HIP_STUB(hipStreamDestroy)
HIP_STUB(hipStreamSynchronize)
HIP_STUB(hipEventCreateWithFlags)
HIP_STUB(hipEventRecord)
HIP_STUB(hipEventQuery)
HIP_STUB(hipEventSynchronize)
HIP_STUB(hipEventDestroy)
HIP_STUB(hipModuleLoadData)
HIP_STUB(hipModuleGetFunction)
HIP_STUB(hipModuleUnload)
HIP_STUB(hipModuleLaunchKernel)
HIP_STUB(hipRuntimeGetVersion)

const char *hipGetErrorName(int code) { (void)code; return NULL; }
