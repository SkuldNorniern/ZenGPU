//! Decode a SPIR-V word stream into a header + instruction list.

use crate::opcodes::{self, OpInfo};

/// SPIR-V magic number (`0x07230203`).
pub const MAGIC: u32 = 0x0723_0203;

/// A decoded module header.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub version: u32,
    pub generator: u32,
    pub bound: u32,
    pub schema: u32,
}

impl Header {
    /// Version as `(major, minor)`.
    pub fn version_pair(&self) -> (u8, u8) {
        let major = ((self.version >> 16) & 0xff) as u8;
        let minor = ((self.version >> 8) & 0xff) as u8;
        (major, minor)
    }
}

/// One decoded instruction.
#[derive(Clone, Debug)]
pub struct Instruction {
    pub opcode: u16,
    pub info: OpInfo,
    pub result_type: Option<u32>,
    pub result_id: Option<u32>,
    /// Operands after the optional result-type and result-id words.
    pub trailing: Vec<u32>,
    /// Word offset of this instruction in the original stream (for diagnostics).
    pub word_offset: usize,
}

impl Instruction {
    /// Ids referenced by this instruction's trailing operands.
    pub fn id_operands(&self) -> Vec<u32> {
        opcodes::id_operands(self.info.trailing, &self.trailing)
    }
}

/// A decoded SPIR-V module.
#[derive(Clone, Debug)]
pub struct Module {
    pub header: Header,
    pub instructions: Vec<Instruction>,
}

/// Failure to decode a SPIR-V word stream.
#[derive(Clone, Debug)]
pub enum DecodeError {
    TooShort,
    BadMagic(u32),
    /// An instruction's declared word count ran past the end of the stream.
    TruncatedInstruction {
        word_offset: usize,
    },
    /// An instruction declared a word count of zero (would loop forever).
    ZeroWordCount {
        word_offset: usize,
    },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "SPIR-V stream shorter than 5-word header"),
            Self::BadMagic(m) => {
                write!(f, "bad SPIR-V magic 0x{m:08X} (expected 0x{MAGIC:08X})")
            }
            Self::TruncatedInstruction { word_offset } => {
                write!(
                    f,
                    "instruction at word {word_offset} runs past end of stream"
                )
            }
            Self::ZeroWordCount { word_offset } => {
                write!(f, "instruction at word {word_offset} has zero word count")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decode a SPIR-V word stream.
pub fn decode(words: &[u32]) -> Result<Module, DecodeError> {
    if words.len() < 5 {
        return Err(DecodeError::TooShort);
    }
    if words[0] != MAGIC {
        return Err(DecodeError::BadMagic(words[0]));
    }
    let header = Header {
        version: words[1],
        generator: words[2],
        bound: words[3],
        schema: words[4],
    };

    let mut instructions = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let first = words[i];
        let word_count = (first >> 16) as usize;
        let opcode = (first & 0xffff) as u16;
        if word_count == 0 {
            return Err(DecodeError::ZeroWordCount { word_offset: i });
        }
        if i + word_count > words.len() {
            return Err(DecodeError::TruncatedInstruction { word_offset: i });
        }
        let body = &words[i + 1..i + word_count];
        let info = opcodes::lookup(opcode);

        let mut idx = 0;
        let result_type = if info.has_result_type {
            let v = body.get(idx).copied();
            idx += 1;
            v
        } else {
            None
        };
        let result_id = if info.has_result_id {
            let v = body.get(idx).copied();
            idx += 1;
            v
        } else {
            None
        };
        let trailing = body.get(idx..).unwrap_or(&[]).to_vec();

        instructions.push(Instruction {
            opcode,
            info,
            result_type,
            result_id,
            trailing,
            word_offset: i,
        });
        i += word_count;
    }

    Ok(Module {
        header,
        instructions,
    })
}
