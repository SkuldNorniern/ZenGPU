//! A small, dependency-free SPIR-V disassembler and structural validator for ZenGPU.
//!
//! Malformed SPIR-V often faults inside `vkCreateShaderModule` without a diagnostic, and validation
//! layers are not always installed. These two catch the likely defects first:
//!
//! - [`disassemble`]: readable assembly from a word stream, for logs.
//! - [`validate()`]: dangling ids, id `0`, duplicate definitions, ids past the bound, missing entry
//!   point.
//!
//! Not a full validator (no type system). Use the Khronos layers or `spirv-val` for that.
//!
//! ```
//! # const SPV: &[u32] = &[]; // placeholder
//! if let Err(e) = zengpu_spv::validate(SPV) {
//!     eprintln!("bad SPIR-V:\n{}", zengpu_spv::disassemble(SPV));
//!     eprintln!("{e}");
//! }
//! ```

pub mod decode;
pub mod disasm;
pub mod opcodes;
pub mod validate;

pub use decode::{DecodeError, Header, Instruction, Module, decode};
pub use disasm::{disassemble, disassemble_module};
pub use validate::{Issue, ValidateError, validate, validate_module};

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid module: void `main` with one return, declared as the entry point.
    fn minimal_module() -> Vec<u32> {
        fn inst(op: u16, words: &[u32]) -> Vec<u32> {
            let wc = (1 + words.len()) as u32;
            let mut v = vec![(wc << 16) | op as u32];
            v.extend_from_slice(words);
            v
        }
        let mut m = vec![decode::MAGIC, 0x0001_0300, 0, 5, 0];
        m.extend(inst(17, &[1]));
        let mut ep = vec![5u32, 3];
        ep.extend_from_slice(&str_words("main"));
        m.extend(inst(15, &ep));
        m.extend(inst(19, &[1]));
        m.extend(inst(33, &[2, 1]));
        m.extend(inst(54, &[1, 3, 0, 2]));
        m.extend(inst(248, &[4]));
        m.extend(inst(253, &[]));
        m.extend(inst(56, &[]));
        m
    }

    fn str_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        let words = (bytes.len() + 1).div_ceil(4);
        let mut out = vec![0u32; words];
        for (i, &b) in bytes.iter().enumerate() {
            out[i / 4] |= (b as u32) << (8 * (i % 4));
        }
        out
    }

    #[test]
    fn decodes_and_validates_minimal_module() {
        let words = minimal_module();
        let module = decode(&words).expect("decode");
        assert_eq!(module.header.version_pair(), (1, 3));
        assert_eq!(module.header.bound, 5);
        validate(&words).expect("should be structurally valid");
    }

    #[test]
    fn rejects_bad_magic() {
        let words = [0xDEAD_BEEF, 0, 0, 1, 0];
        assert!(matches!(
            validate(&words),
            Err(ValidateError::Decode(DecodeError::BadMagic(_)))
        ));
    }

    #[test]
    fn catches_dangling_and_zero_references() {
        for (bad_id, needle) in [(99, "undefined id %99"), (0, "id 0")] {
            let mut words = vec![decode::MAGIC, 0x0001_0300, 0, 3, 0];
            words.extend([(2u32 << 16) | 19, 1]);
            words.extend([(3u32 << 16) | 33, 2, bad_id]);
            match validate(&words) {
                Err(ValidateError::Structural(issues)) => {
                    assert!(issues.iter().any(|i| i.message.contains(needle)));
                }
                other => panic!("expected structural issue, got {other:?}"),
            }
        }
    }

    #[test]
    fn disassembles_minimal_module() {
        let text = disassemble(&minimal_module());
        assert!(text.contains("OpEntryPoint"));
        assert!(text.contains("\"main\""));
        assert!(text.contains("OpFunction"));
        assert!(text.contains("OpReturn"));
    }
}
