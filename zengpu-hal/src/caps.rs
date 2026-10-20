//! Compute capabilities beyond the scalar limits in `DeviceLimits`: subgroups, shared memory, and
//! matrix shapes.

use crate::DType;

pub type ElemType = DType;

/// Matrix multiply-accumulate `C[m x n] += A[m x k] * B[k x n]`, run by one subgroup of
/// `subgroup` lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: ElemType,
    pub b: ElemType,
    pub acc: ElemType,
    pub subgroup: u32,
}

/// Compute work a device can run. Unreported caps stay at `scalar_only`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeCaps {
    pub subgroup_sizes: (u32, u32),
    /// The subgroup size used when a pipeline does not ask for one.
    pub default_subgroup: u32,
    /// Workgroup-shared memory per workgroup, in bytes.
    pub shared_bytes: u32,
    pub max_workgroup: u32,
    pub mma: Vec<MmaShape>,
}

impl ComputeCaps {
    /// The default is one lane, no shared memory, and no matrix units. Backends claim only what
    /// they check.
    pub fn scalar_only() -> Self {
        Self {
            subgroup_sizes: (1, 1),
            default_subgroup: 1,
            shared_bytes: 0,
            max_workgroup: 1,
            mma: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_only_claims_no_matrix_units() {
        let caps = ComputeCaps::scalar_only();
        assert_eq!(caps.subgroup_sizes, (1, 1));
        assert_eq!(caps.default_subgroup, 1);
        assert_eq!(caps.shared_bytes, 0);
        assert!(caps.mma.is_empty());
    }
}
