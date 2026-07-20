/* Link-only stub for libhiprtc. See hip_stub.c for why this exists. */
#define HIPRTC_STUB(name) int name(void) { return 1; }

HIPRTC_STUB(hiprtcCreateProgram)
HIPRTC_STUB(hiprtcCompileProgram)
HIPRTC_STUB(hiprtcGetCodeSize)
HIPRTC_STUB(hiprtcGetCode)
HIPRTC_STUB(hiprtcDestroyProgram)
HIPRTC_STUB(hiprtcGetProgramLogSize)
HIPRTC_STUB(hiprtcGetProgramLog)
