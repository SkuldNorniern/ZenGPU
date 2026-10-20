//! Reads the per-kernel `.sgpr_count` from the AMDGPU metadata note, since HIP's function
//! attributes do not carry it.

use std::collections::HashMap;

const SHT_NOTE: u32 = 7;
/// Reads the `NT_AMDGPU_METADATA` note, whose descriptor is MessagePack.
const NT_AMDGPU_METADATA: u32 = 32;

/// Maps each kernel name to its `.sgpr_count`. Empty if the file is not an AMDGPU ELF or has no
/// metadata note.
pub fn sgpr_counts(image: &[u8]) -> HashMap<String, u32> {
    let mut counts = HashMap::new();
    for note in metadata_notes(image) {
        let Some(Value::Map(root)) = Reader::new(note).value() else {
            continue;
        };
        let Some(Value::Array(kernels)) = map_get(&root, "amdhsa.kernels") else {
            continue;
        };
        for kernel in kernels {
            let Value::Map(fields) = kernel else { continue };
            let (Some(Value::Str(name)), Some(Value::Int(sgprs))) =
                (map_get(fields, ".name"), map_get(fields, ".sgpr_count"))
            else {
                continue;
            };
            if let Ok(sgprs) = u32::try_from(*sgprs) {
                counts.insert(name.clone(), sgprs);
            }
        }
    }
    counts
}

fn le16(b: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(u16::from_le_bytes(
        b.get(at..at + 2)?.try_into().ok()?,
    )))
}

fn le32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn le64(b: &[u8], at: usize) -> Option<usize> {
    usize::try_from(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?)).ok()
}

fn metadata_notes(image: &[u8]) -> Vec<&[u8]> {
    let mut notes = Vec::new();
    if image.get(..4) != Some(b"\x7fELF") || image.get(4) != Some(&2) || image.get(5) != Some(&1) {
        return notes;
    }
    let (Some(shoff), Some(shentsize), Some(shnum)) =
        (le64(image, 0x28), le16(image, 0x3a), le16(image, 0x3c))
    else {
        return notes;
    };
    for i in 0..shnum {
        let header = shoff + i * shentsize;
        if le32(image, header + 4) != Some(SHT_NOTE) {
            continue;
        }
        let (Some(off), Some(size)) = (le64(image, header + 0x18), le64(image, header + 0x20))
        else {
            continue;
        };
        let Some(section) = off.checked_add(size).and_then(|end| image.get(off..end)) else {
            continue;
        };
        let mut p = 0;
        while let (Some(namesz), Some(descsz), Some(kind)) =
            (le32(section, p), le32(section, p + 4), le32(section, p + 8))
        {
            let (namesz, descsz) = (namesz as usize, descsz as usize);
            let name = p + 12;
            let desc = name + namesz.next_multiple_of(4);
            let next = desc + descsz.next_multiple_of(4);
            if next > section.len() {
                break;
            }
            if kind == NT_AMDGPU_METADATA && section[name..].starts_with(b"AMDGPU") {
                notes.push(&section[desc..desc + descsz]);
            }
            p = next;
        }
    }
    notes
}

/// Decodes the subset of MessagePack that the metadata uses. Anything else ends the parse.
enum Value {
    Int(i64),
    Str(String),
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Other,
}

fn map_get<'v>(map: &'v [(Value, Value)], key: &str) -> Option<&'v Value> {
    map.iter().find_map(|(k, v)| match k {
        Value::Str(s) if s == key => Some(v),
        _ => None,
    })
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b, p: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }

    fn uint(&mut self, n: usize) -> Option<u64> {
        Some(
            self.take(n)?
                .iter()
                .fold(0u64, |acc, &x| (acc << 8) | u64::from(x)),
        )
    }

    fn count(&mut self, n: usize) -> Option<usize> {
        usize::try_from(self.uint(n)?).ok()
    }

    fn string(&mut self, len: usize) -> Option<Value> {
        let bytes = self.take(len)?;
        Some(Value::Str(String::from_utf8_lossy(bytes).into_owned()))
    }

    fn value(&mut self) -> Option<Value> {
        let t = self.take(1)?[0];
        Some(match t {
            0x00..=0x7f => Value::Int(i64::from(t)),
            0xe0..=0xff => Value::Int(i64::from(t as i8)),
            0x80..=0x8f => self.map(usize::from(t & 0x0f))?,
            0x90..=0x9f => self.array(usize::from(t & 0x0f))?,
            0xa0..=0xbf => self.string(usize::from(t & 0x1f))?,
            0xc0 | 0xc2 | 0xc3 => Value::Other,
            0xc4..=0xc6 => {
                let n = [1, 2, 4][usize::from(t - 0xc4)];
                let len = self.count(n)?;
                self.take(len)?;
                Value::Other
            }
            0xca => {
                self.take(4)?;
                Value::Other
            }
            0xcb => {
                self.take(8)?;
                Value::Other
            }
            0xcc => Value::Int(self.uint(1)? as i64),
            0xcd => Value::Int(self.uint(2)? as i64),
            0xce => Value::Int(self.uint(4)? as i64),
            0xcf => Value::Int(self.uint(8)? as i64),
            0xd0 => Value::Int(i64::from(self.uint(1)? as u8 as i8)),
            0xd1 => Value::Int(i64::from(self.uint(2)? as u16 as i16)),
            0xd2 => Value::Int(i64::from(self.uint(4)? as u32 as i32)),
            0xd3 => Value::Int(self.uint(8)? as i64),
            0xd9..=0xdb => {
                let n = [1, 2, 4][usize::from(t - 0xd9)];
                let len = self.count(n)?;
                self.string(len)?
            }
            0xdc | 0xdd => {
                let n = if t == 0xdc { 2 } else { 4 };
                let len = self.count(n)?;
                self.array(len)?
            }
            0xde | 0xdf => {
                let n = if t == 0xde { 2 } else { 4 };
                let len = self.count(n)?;
                self.map(len)?
            }
            _ => return None,
        })
    }

    fn array(&mut self, len: usize) -> Option<Value> {
        let items = (0..len).map(|_| self.value()).collect::<Option<Vec<_>>>()?;
        Some(Value::Array(items))
    }

    fn map(&mut self, len: usize) -> Option<Value> {
        let mut pairs = Vec::with_capacity(len);
        for _ in 0..len {
            pairs.push((self.value()?, self.value()?));
        }
        Some(Value::Map(pairs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msgpack_subset_decodes_kernel_fields() {
        let doc = [
            0x81, 0xae, b'a', b'm', b'd', b'h', b's', b'a', b'.', b'k', b'e', b'r', b'n', b'e',
            b'l', b's', 0x91, 0x82, 0xa5, b'.', b'n', b'a', b'm', b'e', 0xa1, b'k', 0xab, b'.',
            b's', b'g', b'p', b'r', b'_', b'c', b'o', b'u', b'n', b't', 0x06,
        ];
        let Some(Value::Map(root)) = Reader::new(&doc).value() else {
            panic!("root map");
        };
        let Some(Value::Array(kernels)) = map_get(&root, "amdhsa.kernels") else {
            panic!("kernels");
        };
        let Value::Map(fields) = &kernels[0] else {
            panic!("kernel map");
        };
        assert!(matches!(
            map_get(fields, ".sgpr_count"),
            Some(Value::Int(6))
        ));
    }

    #[test]
    fn non_elf_bytes_have_no_counts() {
        assert!(sgpr_counts(b"not an elf image at all, just text").is_empty());
    }
}
