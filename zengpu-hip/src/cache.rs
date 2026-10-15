//! Persistent code-object cache for hipRTC output.
//!
//! A cold hipRTC compile costs ~250 ms per kernel, so an application that
//! builds a dozen kernels at start-up waits seconds every launch. When the
//! device was opened with [`zengpu_hal::DeviceRequest::pipeline_cache_path`],
//! code objects are kept in the directory `<path>.hip/`, one file per kernel.
//!
//! Each file stores its complete key (source, compile options, target, ROCm
//! runtime version) ahead of the code, and a load succeeds only on an exact
//! key match, so a hash collision or a ROCm upgrade can never hand back the
//! wrong code. Writes go to a temporary file renamed into place, so readers
//! never see a partial entry.
//!
//! The code also carries a checksum, verified before it reaches the runtime:
//! `hipModuleLoadData` on a damaged code object throws a C++ exception
//! through the C API, which aborts the whole process instead of returning
//! an error.

use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"ZGPUHIP2";

/// Everything that decides the bytes hipRTC produces.
pub(crate) struct CacheKey {
    bytes: Vec<u8>,
}

impl CacheKey {
    pub(crate) fn new(src: &[u8], opts: &[String], target: &str, runtime_version: i32) -> Self {
        let mut bytes = Vec::with_capacity(src.len() + 128);
        let mut field = |data: &[u8]| {
            bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(data);
        };
        field(&runtime_version.to_le_bytes());
        field(target.as_bytes());
        for opt in opts {
            field(opt.as_bytes());
        }
        field(src);
        Self { bytes }
    }

    fn file_name(&self) -> String {
        format!("{:016x}.co", fnv1a(&self.bytes))
    }
}

/// Stable 64-bit FNV-1a: names files and checks code integrity.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Cache directory for a device's `pipeline_cache_path`.
pub(crate) fn dir_for(pipeline_cache_path: &Path) -> PathBuf {
    let mut name = OsString::from(pipeline_cache_path.as_os_str());
    name.push(".hip");
    PathBuf::from(name)
}

pub(crate) fn load(dir: &Path, key: &CacheKey) -> Option<Vec<u8>> {
    let data = fs::read(dir.join(key.file_name())).ok()?;
    let rest = data.strip_prefix(MAGIC.as_slice())?;
    let key_len = u64::from_le_bytes(rest.get(..8)?.try_into().ok()?) as usize;
    let stored_key = rest.get(8..8usize.checked_add(key_len)?)?;
    if stored_key != key.bytes.as_slice() {
        return None;
    }
    let rest = &rest[8 + key_len..];
    let checksum = u64::from_le_bytes(rest.get(..8)?.try_into().ok()?);
    let code = &rest[8..];
    (!code.is_empty() && fnv1a(code) == checksum).then(|| code.to_vec())
}

pub(crate) fn store(dir: &Path, key: &CacheKey, code: &[u8]) {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Err(e) = fs::create_dir_all(dir) {
        log::debug!("hip: cannot create cache dir {}: {e}", dir.display());
        return;
    }
    let mut data = Vec::with_capacity(MAGIC.len() + 8 + key.bytes.len() + code.len());
    data.extend_from_slice(MAGIC);
    data.extend_from_slice(&(key.bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(&key.bytes);
    data.extend_from_slice(&fnv1a(code).to_le_bytes());
    data.extend_from_slice(code);

    let final_path = dir.join(key.file_name());
    let tmp = dir.join(format!(
        ".{}.{}.{}.tmp",
        key.file_name(),
        process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let result = fs::write(&tmp, &data).and_then(|()| fs::rename(&tmp, &final_path));
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        log::debug!(
            "hip: cannot write cache entry {}: {e}",
            final_path.display()
        );
    }
}

/// Drop an entry whose code the runtime refused to load.
pub(crate) fn evict(dir: &Path, key: &CacheKey) {
    match fs::remove_file(dir.join(key.file_name())) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => log::debug!("hip: cannot evict cache entry: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = temp_dir().join(format!("zengpu-hip-cache-{tag}-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn round_trip_and_exact_key_match() {
        let dir = scratch_dir("rt");
        let opts = vec!["--gpu-architecture=gfx1010".to_string()];
        let key = CacheKey::new(b"kernel", &opts, "gfx1010", 70_201_000);
        assert!(load(&dir, &key).is_none());
        store(&dir, &key, b"code-object");
        assert_eq!(load(&dir, &key).as_deref(), Some(b"code-object".as_slice()));

        // Any change of input misses.
        for other in [
            CacheKey::new(b"kernel2", &opts, "gfx1010", 70_201_000),
            CacheKey::new(b"kernel", &[], "gfx1010", 70_201_000),
            CacheKey::new(b"kernel", &opts, "gfx1030", 70_201_000),
            CacheKey::new(b"kernel", &opts, "gfx1010", 60_400_000),
        ] {
            assert!(load(&dir, &other).is_none());
        }

        evict(&dir, &key);
        assert!(load(&dir, &key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn colliding_file_with_another_key_is_ignored() {
        let dir = scratch_dir("collide");
        let key = CacheKey::new(b"a", &[], "gfx1200", 1);
        let other = CacheKey::new(b"b", &[], "gfx1200", 1);
        store(&dir, &other, b"wrong");
        fs::rename(dir.join(other.file_name()), dir.join(key.file_name())).unwrap();
        assert!(load(&dir, &key).is_none());
        fs::write(dir.join(key.file_name()), b"garbage").unwrap();
        assert!(load(&dir, &key).is_none());

        // Damaged code under an intact key is refused, never loaded.
        store(&dir, &key, b"code-object-bytes");
        let path = dir.join(key.file_name());
        let mut data = fs::read(&path).unwrap();
        let last = data.len() - 1;
        data[last] ^= 0xff;
        fs::write(&path, &data).unwrap();
        assert!(load(&dir, &key).is_none());
        let mut truncated = fs::read(&path).unwrap();
        truncated.truncate(truncated.len() - 4);
        fs::write(&path, &truncated).unwrap();
        assert!(load(&dir, &key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_is_a_sibling_of_the_vulkan_cache_file() {
        assert_eq!(
            dir_for(Path::new("/tmp/app/pipelines.bin")),
            PathBuf::from("/tmp/app/pipelines.bin.hip")
        );
    }
}
