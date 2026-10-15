//! HIP runtime and hipRTC entry points, loaded at run time with `dlopen`/`LoadLibrary`.
//!
//! A `hip` build starts without ROCm (no adapters are reported). One binary runs on ROCm 5, 6, and
//! 7, and no ROCm is needed at build time. Every symbol is in the `hip_4.2` ABI except
//! `hipGetDevicePropertiesR0600`, which is looked up optionally.
//!
//! hipRTC moved from `libamdhip64` to `libhiprtc` in ROCm 5.x; both are searched.

#![allow(non_snake_case, clippy::too_many_arguments)]

use std::env;
use std::ffi::{CString, c_char, c_void};
use std::mem::transmute;
use std::ptr::null;
use std::sync::OnceLock;

pub type HipError = i32;
pub type HipModuleRaw = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HiprtcProg = *mut c_void;
pub type HipMemPool = *mut c_void;

/// Returned by every entry point when the runtime did not load.
pub const HIP_ERROR_NOT_SUPPORTED: HipError = 801;
const HIPRTC_ERROR_NOT_LOADED: i32 = 1;

struct Library(*mut c_void);

// SAFETY: a library handle is process-global and never closed.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

#[cfg(unix)]
mod os {
    use std::ffi::{c_char, c_int, c_void};

    pub const RTLD_NOW: c_int = 2;
    pub const RTLD_GLOBAL: c_int = 0x100;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::{c_char, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }
}

impl Library {
    fn open(name: &str) -> Option<Self> {
        let c_name = CString::new(name).ok()?;
        // SAFETY: plain library load; the handle is kept for the process.
        #[cfg(unix)]
        let handle = unsafe { os::dlopen(c_name.as_ptr(), os::RTLD_NOW | os::RTLD_GLOBAL) };
        #[cfg(windows)]
        let handle = unsafe { os::LoadLibraryA(c_name.as_ptr()) };
        #[cfg(not(any(unix, windows)))]
        let handle: *mut c_void = {
            let _ = c_name;
            null::<c_void>() as *mut c_void
        };
        (!handle.is_null()).then_some(Self(handle))
    }

    /// First library in `names` that loads, also trying `$ROCM_PATH/lib`.
    fn open_any(names: &[&str]) -> Option<Self> {
        let roots: Vec<String> = env::var("ROCM_PATH")
            .into_iter()
            .chain(env::var("HIP_PATH"))
            .flat_map(|root| [format!("{root}/lib"), format!("{root}/bin")])
            .chain(["/opt/rocm/lib".to_string()])
            .collect();
        names.iter().find_map(|name| {
            Self::open(name).or_else(|| {
                roots
                    .iter()
                    .find_map(|dir| Self::open(&format!("{dir}/{name}")))
            })
        })
    }

    fn symbol(&self, name: &str) -> Option<*mut c_void> {
        let c_name = CString::new(name).ok()?;
        #[cfg(unix)]
        let sym = unsafe { os::dlsym(self.0, c_name.as_ptr()) };
        #[cfg(windows)]
        let sym = unsafe { os::GetProcAddress(self.0, c_name.as_ptr()) };
        #[cfg(not(any(unix, windows)))]
        let sym: *mut c_void = {
            let _ = c_name;
            null::<c_void>() as *mut c_void
        };
        (!sym.is_null()).then_some(sym)
    }
}

#[cfg(windows)]
const HIP_NAMES: &[&str] = &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
#[cfg(not(windows))]
const HIP_NAMES: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
];

#[cfg(windows)]
const HIPRTC_NAMES: &[&str] = &[
    "hiprtc0702.dll",
    "hiprtc0701.dll",
    "hiprtc0700.dll",
    "hiprtc0604.dll",
    "hiprtc0603.dll",
    "hiprtc0602.dll",
    "hiprtc0601.dll",
    "hiprtc0600.dll",
    "hiprtc0507.dll",
    "hiprtc.dll",
];
#[cfg(not(windows))]
const HIPRTC_NAMES: &[&str] = &[
    "libhiprtc.so",
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
];

macro_rules! entry_points {
    (
        $table:ident, $lib:ident;
        $( fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty = $fallback:expr; )*
    ) => {
        struct $table {
            $( $name: unsafe extern "C" fn($($ty),*) -> $ret, )*
        }

        impl $table {
            fn load(libs: &[&Library]) -> Result<Self, String> {
                Ok(Self {
                    $( $name: {
                        let sym = libs
                            .iter()
                            .find_map(|lib| lib.symbol(stringify!($name)))
                            .ok_or_else(|| format!("missing symbol {}", stringify!($name)))?;
                        // SAFETY: the symbol has the C signature declared here
                        unsafe { transmute::<*mut c_void, unsafe extern "C" fn($($ty),*) -> $ret>(sym) }
                    }, )*
                })
            }
        }

        $(
            /// Returns the "not supported" value if the runtime is missing.
            pub unsafe fn $name($($arg: $ty),*) -> $ret {
                match $lib() {
                    // SAFETY: caller upholds the HIP API contract.
                    Some(api) => unsafe { (api.$name)($($arg),*) },
                    None => $fallback,
                }
            }
        )*
    };
}

entry_points! {
    HipTable, hip_table;
    fn hipInit(flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipRuntimeGetVersion(ver: *mut i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipGetDeviceCount(count: *mut i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceGetName(name: *mut c_char, len: i32, device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceTotalMem(bytes: *mut usize, device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceComputeCapability(major: *mut i32, minor: *mut i32, device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipGetErrorName(code: HipError) -> *const c_char = null();
    fn hipSetDevice(device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipSetDeviceFlags(flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceCanAccessPeer(can_access: *mut i32, device: i32, peer: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipDeviceEnablePeerAccess(peer: i32, flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipFree(ptr: *mut c_void) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipHostFree(ptr: *mut c_void) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMemcpyPeer(dst: *mut c_void, dst_device: i32, src: *const c_void, src_device: i32, bytes: usize) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipStreamCreateWithFlags(stream: *mut HipStream, flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipStreamDestroy(stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipStreamSynchronize(stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipEventCreateWithFlags(event: *mut HipEvent, flags: u32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipEventRecord(event: HipEvent, stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipEventQuery(event: HipEvent) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipEventSynchronize(event: HipEvent) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipEventDestroy(event: HipEvent) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipModuleLoadData(module: *mut HipModuleRaw, image: *const c_void) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipModuleGetFunction(func: *mut HipFunction, module: HipModuleRaw, name: *const c_char) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipModuleUnload(module: HipModuleRaw) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipFuncGetAttribute(value: *mut i32, attr: i32, func: HipFunction) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipModuleLaunchKernel(
        f: HipFunction, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
        shared_bytes: u32, stream: HipStream, params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> HipError = HIP_ERROR_NOT_SUPPORTED;
}

entry_points! {
    RtcTable, rtc_table;
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProg, src: *const c_char, name: *const c_char,
        num_headers: i32, headers: *const *const c_char, include_names: *const *const c_char,
    ) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcCompileProgram(prog: HiprtcProg, num_opts: i32, opts: *const *const c_char) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcGetCodeSize(prog: HiprtcProg, size: *mut usize) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcGetCode(prog: HiprtcProg, code: *mut c_char) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProg) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcGetProgramLogSize(prog: HiprtcProg, size: *mut usize) -> i32 = HIPRTC_ERROR_NOT_LOADED;
    fn hiprtcGetProgramLog(prog: HiprtcProg, log: *mut c_char) -> i32 = HIPRTC_ERROR_NOT_LOADED;
}

entry_points! {
    PoolTable, pool_table;
    fn hipDeviceGetDefaultMemPool(pool: *mut HipMemPool, device: i32) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMemPoolSetAttribute(pool: HipMemPool, attr: i32, value: *mut c_void) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
    fn hipFreeAsync(ptr: *mut c_void, stream: HipStream) -> HipError = HIP_ERROR_NOT_SUPPORTED;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropsAbi {
    /// `hipGetDevicePropertiesR0600` for the ROCm 6+ struct layout.
    R0600,
    /// `hipGetDeviceProperties` for the ROCm 5 struct layout.
    R0000,
}

struct Runtime {
    hip: HipTable,
    rtc: Option<RtcTable>,
    /// Stream-ordered allocator (ROCm 5.2+), absent on older runtimes.
    pool: Option<PoolTable>,
    props: unsafe extern "C" fn(*mut u8, i32) -> HipError,
    props_abi: PropsAbi,
}

fn runtime() -> Option<&'static Runtime> {
    static RUNTIME: OnceLock<Option<Runtime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| match load() {
            Ok(rt) => Some(rt),
            Err(e) => {
                log::debug!("hip: runtime unavailable: {e}");
                None
            }
        })
        .as_ref()
}

fn load() -> Result<Runtime, String> {
    let hip = Library::open_any(HIP_NAMES).ok_or("libamdhip64 not found")?;
    let hip = Box::leak(Box::new(hip));
    let table = HipTable::load(&[hip])?;

    let (props, props_abi) = match hip.symbol("hipGetDevicePropertiesR0600") {
        Some(sym) => (sym, PropsAbi::R0600),
        None => (
            hip.symbol("hipGetDeviceProperties")
                .ok_or("missing symbol hipGetDeviceProperties")?,
            PropsAbi::R0000,
        ),
    };

    let rtc_lib = Library::open_any(HIPRTC_NAMES).map(|lib| &*Box::leak(Box::new(lib)));
    let rtc = match rtc_lib {
        Some(lib) => RtcTable::load(&[lib, hip]),
        None => RtcTable::load(&[hip]),
    };
    let rtc = match rtc {
        Ok(rtc) => Some(rtc),
        Err(e) => {
            log::warn!("hip: hipRTC unavailable ({e}); kernels cannot be compiled");
            None
        }
    };

    let pool = PoolTable::load(&[hip]).ok();

    Ok(Runtime {
        hip: table,
        rtc,
        pool,
        // SAFETY: both symbols take (hipDeviceProp_t*, int) -> hipError_t.
        props: unsafe {
            transmute::<*mut c_void, unsafe extern "C" fn(*mut u8, i32) -> HipError>(props)
        },
        props_abi,
    })
}

fn hip_table() -> Option<&'static HipTable> {
    runtime().map(|rt| &rt.hip)
}

fn rtc_table() -> Option<&'static RtcTable> {
    runtime().and_then(|rt| rt.rtc.as_ref())
}

fn pool_table() -> Option<&'static PoolTable> {
    runtime().and_then(|rt| rt.pool.as_ref())
}

pub fn pools_loaded() -> bool {
    pool_table().is_some()
}

pub fn runtime_loaded() -> bool {
    runtime().is_some()
}

pub fn hiprtc_loaded() -> bool {
    rtc_table().is_some()
}

pub fn props_abi() -> Option<PropsAbi> {
    runtime().map(|rt| rt.props_abi)
}

/// `props` must hold at least the runtime's `hipDeviceProp_t`.
pub unsafe fn hipGetDeviceProperties(props: *mut u8, device: i32) -> HipError {
    match runtime() {
        // SAFETY: caller passes a buffer large enough for either layout.
        Some(rt) => unsafe { (rt.props)(props, device) },
        None => HIP_ERROR_NOT_SUPPORTED,
    }
}
