//! Semantic capability vocabulary (design doc §2.2), inert-first.
//!
//! [`Capability`] names a semantic operation or storage-format requirement —
//! not a Cargo feature, not a vendor name. [`SupportLevel`] describes how
//! well a given backend can satisfy one.
//!
//! **Scoped smaller than the design doc's full model on purpose.** The
//! design doc also specifies `RequiredCapability` (a `requires {}`-block
//! entry tied to a source span) and `TargetCapabilities` (a per-backend
//! device-profile support map, keyed by a `BackendId` and carrying live
//! device limits). Neither is added here: `zengpu-zsl` has no internal
//! `BackendId` concept today (backends are dispatched via `#[cfg(feature =
//! ...)]` in `compile_zsl_all`, not a shared enum), and `zengpu-hal::Features`
//! already carries the device-side capability bits separately (`FLOAT16`/
//! `BFLOAT16`/`TENSOR_CORES`) — wiring a compile-time capability check
//! against a live device profile, plus `requires {}` parser syntax, is real,
//! separate, later work that nothing needs before Step 3's next commit (HIP
//! bf16 storage/load). This module exists so that commit — and every later
//! capability-gated backend arm — has a shared vocabulary to describe
//! requirements against, instead of inventing ad-hoc checks per backend.

/// A semantic operation or storage-format requirement. Describes what a ZSL
/// program needs, independent of which backend or Cargo feature is active.
#[allow(dead_code)] // Deliberately inert infrastructure described in the module documentation.
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

/// How well a given backend can satisfy a [`Capability`].
#[allow(dead_code)] // Deliberately inert infrastructure described in the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportLevel {
    /// The backend has a native instruction/type for this capability.
    Native,
    /// The backend has no native form but can emulate it exactly (same bit
    /// pattern, same rounding/synchronization behavior) via `strategy` (a
    /// short, human-readable description — e.g. "packed u16 + F32 convert").
    Emulated { strategy: &'static str },
    /// The backend cannot represent or emulate this capability; `reason`
    /// explains why (surfaced in diagnostics).
    Unsupported { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_is_orderable_for_use_as_a_map_key() {
        // TargetCapabilities-style tables (a later step) key off Capability
        // in a BTreeMap; confirm the derive actually gives a total order.
        let mut caps = vec![Capability::TensorMma, Capability::Int8, Capability::Float64];
        caps.sort();
        assert_eq!(
            caps,
            vec![Capability::Int8, Capability::Float64, Capability::TensorMma]
        );
    }

    #[test]
    fn support_level_variants_construct_and_compare() {
        assert_eq!(SupportLevel::Native, SupportLevel::Native);
        assert_ne!(
            SupportLevel::Emulated {
                strategy: "packed u16 + F32 convert"
            },
            SupportLevel::Unsupported {
                reason: "no bf16 type on this target".to_string()
            }
        );
    }
}
