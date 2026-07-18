#define NVRTC_STUB(name) int name(void) { return 1; }

NVRTC_STUB(nvrtcCompileProgram)
NVRTC_STUB(nvrtcCreateProgram)
NVRTC_STUB(nvrtcDestroyProgram)
NVRTC_STUB(nvrtcGetPTX)
NVRTC_STUB(nvrtcGetPTXSize)
NVRTC_STUB(nvrtcGetProgramLog)
NVRTC_STUB(nvrtcGetProgramLogSize)
