//! Opcode metadata for the subset ZenGPU emits, plus a generic fallback. Enough to disassemble and
//! check id refs, but no full grammar.

/// How trailing operands split into id refs and literals. Shared by the disassembler and validator.
#[derive(Clone, Copy, Debug)]
pub enum Trailing {
    /// No trailing operands carry ids (all literals or none).
    Literals,
    /// Every trailing operand is an id-ref.
    AllIds,
    /// Trailing operands are all ids starting at `from` (earlier ones literal).
    IdsFrom { from: usize },
    /// A fixed set of trailing operand positions are ids; the rest are literals.
    IdsAt { idx: &'static [usize] },
    /// `OpEntryPoint`: [exec-model lit, fn id, name string…, interface ids…].
    EntryPoint,
    ExtInstImport,
    /// `OpExtInst`: [set id, instruction lit, operand ids…].
    ExtInst,
}

#[derive(Clone, Copy, Debug)]
pub struct OpInfo {
    pub name: &'static str,
    pub has_result_type: bool,
    pub has_result_id: bool,
    pub trailing: Trailing,
}

impl OpInfo {
    const fn new(
        name: &'static str,
        has_result_type: bool,
        has_result_id: bool,
        trailing: Trailing,
    ) -> Self {
        Self {
            name,
            has_result_type,
            has_result_id,
            trailing,
        }
    }
}

/// Unknown opcodes get a conservative entry (no result, no id operands), so they render as-is
/// without false positives.
pub fn lookup(opcode: u16) -> OpInfo {
    use Trailing::*;
    match opcode {
        11 => OpInfo::new("OpExtInstImport", false, true, ExtInstImport),
        12 => OpInfo::new("OpExtInst", true, true, ExtInst),
        14 => OpInfo::new("OpMemoryModel", false, false, Literals),
        15 => OpInfo::new("OpEntryPoint", false, false, EntryPoint),
        16 => OpInfo::new("OpExecutionMode", false, false, IdsAt { idx: &[0] }),
        17 => OpInfo::new("OpCapability", false, false, Literals),
        5 => OpInfo::new("OpName", false, false, IdsAt { idx: &[0] }),
        71 => OpInfo::new("OpDecorate", false, false, IdsAt { idx: &[0] }),
        72 => OpInfo::new("OpMemberDecorate", false, false, IdsAt { idx: &[0] }),
        19 => OpInfo::new("OpTypeVoid", false, true, Literals),
        20 => OpInfo::new("OpTypeBool", false, true, Literals),
        21 => OpInfo::new("OpTypeInt", false, true, Literals),
        22 => OpInfo::new("OpTypeFloat", false, true, Literals),
        23 => OpInfo::new("OpTypeVector", false, true, IdsAt { idx: &[0] }),
        24 => OpInfo::new("OpTypeMatrix", false, true, IdsAt { idx: &[0] }),
        25 => OpInfo::new("OpTypeImage", false, true, IdsAt { idx: &[0] }),
        26 => OpInfo::new("OpTypeSampler", false, true, Literals),
        27 => OpInfo::new("OpTypeSampledImage", false, true, IdsAt { idx: &[0] }),
        28 => OpInfo::new("OpTypeArray", false, true, IdsAt { idx: &[0, 1] }),
        29 => OpInfo::new("OpTypeRuntimeArray", false, true, IdsAt { idx: &[0] }),
        30 => OpInfo::new("OpTypeStruct", false, true, AllIds),
        32 => OpInfo::new("OpTypePointer", false, true, IdsAt { idx: &[1] }),
        33 => OpInfo::new("OpTypeFunction", false, true, AllIds),
        43 => OpInfo::new("OpConstant", true, true, Literals),
        59 => OpInfo::new("OpVariable", true, true, IdsFrom { from: 1 }),
        54 => OpInfo::new("OpFunction", true, true, IdsAt { idx: &[1] }),
        55 => OpInfo::new("OpFunctionParameter", true, true, Literals),
        56 => OpInfo::new("OpFunctionEnd", false, false, Literals),
        248 => OpInfo::new("OpLabel", false, true, Literals),
        249 => OpInfo::new("OpBranch", false, false, IdsAt { idx: &[0] }),
        250 => OpInfo::new(
            "OpBranchConditional",
            false,
            false,
            IdsAt { idx: &[0, 1, 2] },
        ),
        247 => OpInfo::new("OpSelectionMerge", false, false, IdsAt { idx: &[0] }),
        246 => OpInfo::new("OpLoopMerge", false, false, IdsAt { idx: &[0, 1] }),
        253 => OpInfo::new("OpReturn", false, false, Literals),
        61 => OpInfo::new("OpLoad", true, true, IdsAt { idx: &[0] }),
        62 => OpInfo::new("OpStore", false, false, IdsAt { idx: &[0, 1] }),
        65 => OpInfo::new("OpAccessChain", true, true, AllIds),
        80 => OpInfo::new("OpCompositeConstruct", true, true, AllIds),
        81 => OpInfo::new("OpCompositeExtract", true, true, IdsAt { idx: &[0] }),
        79 => OpInfo::new("OpVectorShuffle", true, true, IdsAt { idx: &[0, 1] }),
        87 => OpInfo::new("OpImageSampleImplicitLod", true, true, IdsAt { idx: &[0, 1] }),
        88 => OpInfo::new("OpImageSampleExplicitLod", true, true, IdsAt { idx: &[0, 1, 3] }),
        109 => OpInfo::new("OpConvertFToU", true, true, AllIds),
        110 => OpInfo::new("OpConvertFToS", true, true, AllIds),
        111 => OpInfo::new("OpConvertSToF", true, true, AllIds),
        112 => OpInfo::new("OpConvertUToF", true, true, AllIds),
        124 => OpInfo::new("OpBitcast", true, true, AllIds),
        126 => OpInfo::new("OpSNegate", true, true, AllIds),
        127 => OpInfo::new("OpFNegate", true, true, AllIds),
        128 => OpInfo::new("OpIAdd", true, true, AllIds),
        129 => OpInfo::new("OpFAdd", true, true, AllIds),
        130 => OpInfo::new("OpISub", true, true, AllIds),
        131 => OpInfo::new("OpFSub", true, true, AllIds),
        132 => OpInfo::new("OpIMul", true, true, AllIds),
        133 => OpInfo::new("OpFMul", true, true, AllIds),
        134 => OpInfo::new("OpUDiv", true, true, AllIds),
        135 => OpInfo::new("OpSDiv", true, true, AllIds),
        136 => OpInfo::new("OpFDiv", true, true, AllIds),
        142 => OpInfo::new("OpVectorTimesScalar", true, true, AllIds),
        143 => OpInfo::new("OpMatrixTimesScalar", true, true, AllIds),
        144 => OpInfo::new("OpVectorTimesMatrix", true, true, AllIds),
        145 => OpInfo::new("OpMatrixTimesVector", true, true, AllIds),
        146 => OpInfo::new("OpMatrixTimesMatrix", true, true, AllIds),
        148 => OpInfo::new("OpDot", true, true, AllIds),
        156 => OpInfo::new("OpIsNan", true, true, AllIds),
        157 => OpInfo::new("OpIsInf", true, true, AllIds),
        166 => OpInfo::new("OpLogicalOr", true, true, AllIds),
        167 => OpInfo::new("OpLogicalAnd", true, true, AllIds),
        170 => OpInfo::new("OpIEqual", true, true, AllIds),
        171 => OpInfo::new("OpINotEqual", true, true, AllIds),
        172 => OpInfo::new("OpUGreaterThan", true, true, AllIds),
        174 => OpInfo::new("OpUGreaterThanEqual", true, true, AllIds),
        176 => OpInfo::new("OpULessThan", true, true, AllIds),
        177 => OpInfo::new("OpSLessThan", true, true, AllIds),
        178 => OpInfo::new("OpULessThanEqual", true, true, AllIds),
        180 => OpInfo::new("OpFOrdEqual", true, true, AllIds),
        182 => OpInfo::new("OpFOrdNotEqual", true, true, AllIds),
        184 => OpInfo::new("OpFOrdLessThan", true, true, AllIds),
        185 => OpInfo::new("OpFOrdGreaterThan", true, true, AllIds),
        186 => OpInfo::new("OpFUnordGreaterThan", true, true, AllIds),
        188 => OpInfo::new("OpFOrdLessThanEqual", true, true, AllIds),
        190 => OpInfo::new("OpFOrdGreaterThanEqual", true, true, AllIds),
        169 => OpInfo::new("OpSelect", true, true, AllIds),
        194 => OpInfo::new("OpShiftRightLogical", true, true, AllIds),
        196 => OpInfo::new("OpShiftLeftLogical", true, true, AllIds),
        197 => OpInfo::new("OpBitwiseOr", true, true, AllIds),
        198 => OpInfo::new("OpBitwiseXor", true, true, AllIds),
        199 => OpInfo::new("OpBitwiseAnd", true, true, AllIds),
        252 => OpInfo::new("OpKill", false, false, Literals),
        _ => OpInfo::new("OpUnknown", false, false, Literals),
    }
}

/// Id-ref operands of `trailing`, as raw ids.
pub fn id_operands(trailing_kind: Trailing, trailing: &[u32]) -> Vec<u32> {
    match trailing_kind {
        Trailing::Literals => Vec::new(),
        Trailing::AllIds => trailing.to_vec(),
        Trailing::IdsFrom { from } => trailing.iter().skip(from).copied().collect(),
        Trailing::IdsAt { idx } => idx
            .iter()
            .filter_map(|&i| trailing.get(i).copied())
            .collect(),
        Trailing::EntryPoint => {
            let mut ids = Vec::new();
            if let Some(&fn_id) = trailing.get(1) {
                ids.push(fn_id);
            }
            if let Some(skip) = entry_point_name_words(trailing) {
                let iface_start = 2 + skip;
                ids.extend(trailing.iter().skip(iface_start).copied());
            }
            ids
        }
        Trailing::ExtInstImport => Vec::new(),
        Trailing::ExtInst => {
            let mut ids = Vec::new();
            if let Some(&set) = trailing.first() {
                ids.push(set);
            }
            ids.extend(trailing.iter().skip(2).copied());
            ids
        }
    }
}

/// Word count of the inline name of `OpEntryPoint`. `None` if unterminated.
pub fn entry_point_name_words(trailing: &[u32]) -> Option<usize> {
    string_words(trailing.get(2..)?)
}

/// Word count of a null-terminated literal string. The word holding a zero byte ends it.
pub fn string_words(words: &[u32]) -> Option<usize> {
    for (i, w) in words.iter().enumerate() {
        if w.to_le_bytes().contains(&0) {
            return Some(i + 1);
        }
    }
    None
}

pub fn decode_string(words: &[u32]) -> String {
    let mut bytes = Vec::new();
    for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                return String::from_utf8_lossy(&bytes).into_owned();
            }
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
