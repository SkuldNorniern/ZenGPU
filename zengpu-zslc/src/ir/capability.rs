//! Semantic capability vocabulary: what a program needs, not a Cargo feature or vendor.
//!
//! Not yet: `requires {}` syntax and a per-backend target table.

/// A capability a program needs, independent of backend or Cargo feature.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Capability {
    Int8,
    Int16,
    Int64,
    Float16Storage,
    Float16Arithmetic,
    BFloat16Storage,
    BFloat16Arithmetic,
    Float64,
    Float8Storage,
    Int64Atomics,
    Float32Atomics,
    Float64Atomics,
    SubgroupBasic,
    SubgroupVote,
    SubgroupBallot,
    SubgroupShuffle,
    SubgroupArithmetic,
    VariableSubgroupSize,
    TensorMma,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportLevel {
    Native,
    /// Has no native form, but is emulated exactly (same bits, rounding and sync) via `strategy`,
    /// for example 'packed u16 + F32 convert'.
    Emulated {
        strategy: &'static str,
    },
    /// Reason shown in diagnostics.
    Unsupported {
        reason: String,
    },
}
