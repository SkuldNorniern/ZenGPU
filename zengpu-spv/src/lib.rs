//! ZenGPU SPIR-V tooling — a small, dependency-free disassembler and
//! structural validator for the SPIR-V that ZenGPU generates and consumes.
//!
//! ZenGPU emits SPIR-V from ZSL (and accepts external SPIR-V). When that
//! SPIR-V is malformed, drivers tend to fault *inside* `vkCreateShaderModule`
//! or `vkCreateGraphicsPipelines` with no usable diagnostic — and Vulkan
//! validation layers are not always installed on user machines. This crate
//! fills that gap with two always-available tools:
//!
//! - [`disassemble`] — render a word stream as readable assembly text, so a
//!   generated shader can be inspected from a log without external tools.
//! - [`validate()`] — catch the structural defects a code generator is most
//!   likely to emit (dangling id references, use of id `0`, duplicate
//!   definitions, ids past the bound, missing entry point) *before* the bytes
//!   reach the driver.
//!
//! It is intentionally not a full validator: there is no type system here. Use
//! it as a fast, embeddable first line of defence; use the Khronos validation
//! layers or `spirv-val` for exhaustive checking.
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

    /// Build a minimal-but-well-formed SPIR-V module by hand:
    /// a void function `main` with a single return, declared as the entry point.
    fn minimal_module() -> Vec<u32> {
        fn inst(op: u16, words: &[u32]) -> Vec<u32> {
            let wc = (1 + words.len()) as u32;
            let mut v = vec![(wc << 16) | op as u32];
            v.extend_from_slice(words);
            v
        }
        // ids: 1 = void type, 2 = fn type, 3 = function, 4 = label
        let mut m = vec![decode::MAGIC, 0x0001_0300, 0, 5, 0];
        m.extend(inst(17, &[1])); // OpCapability Shader
        // OpEntryPoint GLCompute %3 "main"
        let mut ep = vec![5u32, 3];
        ep.extend_from_slice(&str_words("main"));
        m.extend(inst(15, &ep));
        m.extend(inst(19, &[1])); // %1 = OpTypeVoid
        m.extend(inst(33, &[2, 1])); // %2 = OpTypeFunction %1
        m.extend(inst(54, &[1, 3, 0, 2])); // %3 = OpFunction %1 None %2
        m.extend(inst(248, &[4])); // %4 = OpLabel
        m.extend(inst(253, &[])); // OpReturn
        m.extend(inst(56, &[])); // OpFunctionEnd
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
    fn catches_dangling_reference() {
        // %2 = OpTypeFunction %99  (99 never defined)
        let mut words = vec![decode::MAGIC, 0x0001_0300, 0, 3, 0];
        words.extend([(2u32 << 16) | 19, 1]); // %1 = OpTypeVoid
        words.extend([(3u32 << 16) | 33, 2, 99]); // %2 = OpTypeFunction %99
        match validate(&words) {
            Err(ValidateError::Structural(issues)) => {
                assert!(
                    issues
                        .iter()
                        .any(|i| i.message.contains("undefined id %99"))
                );
            }
            other => panic!("expected structural issue, got {other:?}"),
        }
    }

    #[test]
    fn catches_id_zero_reference() {
        // %2 = OpTypeFunction %0  (id 0 invalid)
        let mut words = vec![decode::MAGIC, 0x0001_0300, 0, 3, 0];
        words.extend([(2u32 << 16) | 19, 1]);
        words.extend([(3u32 << 16) | 33, 2, 0]);
        match validate(&words) {
            Err(ValidateError::Structural(issues)) => {
                assert!(issues.iter().any(|i| i.message.contains("id 0")));
            }
            other => panic!("expected structural issue, got {other:?}"),
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
