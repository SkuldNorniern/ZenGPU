//! Few-row f32 GEMM for ZSL backends (LLM decode). One output column per lane in 256-lane
//! workgroups; each sums a 128-deep K chunk with that chunk of A in shared memory. Partials go to
//! `part[split][m][n]`, summed by `SPLIT_REDUCE_ZSL`.

use zengpu_spirv::{ZslShader, zsl};

/// K depth per workgroup. Must match the shared tile stride in the kernels.
pub(crate) const KC: u32 = 128;

pub(crate) const ROWS: [u32; 4] = [1, 4, 8, 16];

pub(crate) const SKINNY_1_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, kc: u32, alpha: f32 }
    @workgroup_size(256)
    kernel cs_skinny_1(a: device buffer<f32>, b: device buffer<f32>, part: device mut buffer<f32>, p: P) {
        shared as_tile: array<f32, 128>
        let lx = local_id().x
        let col = group_id().x * 256 + lx
        let gy = group_id().y
        let k0 = gy * p.kc
        for s in 0..1 {
            let idx = lx + s * 256
            if idx < 128 {
                let r = idx / 128
                let kk = idx - r * 128
                if r < p.m && kk < p.kc && k0 + kk < p.k {
                    as_tile[idx] = a[r * p.k + k0 + kk]
                } else {
                    as_tile[idx] = 0.0
                }
            }
        }
        barrier()
        let c0: f32 = 0.0
        if col < p.n {
            for kk in 0..p.kc {
                let kg = k0 + kk
                if kg < p.k {
                    let bv = b[kg * p.n + col]
                c0 = c0 + as_tile[0 + kk] * bv
                }
            }
            if 0 < p.m {
                part[(gy * p.m + 0) * p.n + col] = p.alpha * c0
            }
        }
    }
);

pub(crate) const SKINNY_4_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, kc: u32, alpha: f32 }
    @workgroup_size(256)
    kernel cs_skinny_4(a: device buffer<f32>, b: device buffer<f32>, part: device mut buffer<f32>, p: P) {
        shared as_tile: array<f32, 512>
        let lx = local_id().x
        let col = group_id().x * 256 + lx
        let gy = group_id().y
        let k0 = gy * p.kc
        for s in 0..2 {
            let idx = lx + s * 256
            if idx < 512 {
                let r = idx / 128
                let kk = idx - r * 128
                if r < p.m && kk < p.kc && k0 + kk < p.k {
                    as_tile[idx] = a[r * p.k + k0 + kk]
                } else {
                    as_tile[idx] = 0.0
                }
            }
        }
        barrier()
        let c0: f32 = 0.0
        let c1: f32 = 0.0
        let c2: f32 = 0.0
        let c3: f32 = 0.0
        if col < p.n {
            for kk in 0..p.kc {
                let kg = k0 + kk
                if kg < p.k {
                    let bv = b[kg * p.n + col]
                c0 = c0 + as_tile[0 + kk] * bv
                c1 = c1 + as_tile[128 + kk] * bv
                c2 = c2 + as_tile[256 + kk] * bv
                c3 = c3 + as_tile[384 + kk] * bv
                }
            }
            if 0 < p.m {
                part[(gy * p.m + 0) * p.n + col] = p.alpha * c0
            }
            if 1 < p.m {
                part[(gy * p.m + 1) * p.n + col] = p.alpha * c1
            }
            if 2 < p.m {
                part[(gy * p.m + 2) * p.n + col] = p.alpha * c2
            }
            if 3 < p.m {
                part[(gy * p.m + 3) * p.n + col] = p.alpha * c3
            }
        }
    }
);

pub(crate) const SKINNY_8_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, kc: u32, alpha: f32 }
    @workgroup_size(256)
    kernel cs_skinny_8(a: device buffer<f32>, b: device buffer<f32>, part: device mut buffer<f32>, p: P) {
        shared as_tile: array<f32, 1024>
        let lx = local_id().x
        let col = group_id().x * 256 + lx
        let gy = group_id().y
        let k0 = gy * p.kc
        for s in 0..4 {
            let idx = lx + s * 256
            if idx < 1024 {
                let r = idx / 128
                let kk = idx - r * 128
                if r < p.m && kk < p.kc && k0 + kk < p.k {
                    as_tile[idx] = a[r * p.k + k0 + kk]
                } else {
                    as_tile[idx] = 0.0
                }
            }
        }
        barrier()
        let c0: f32 = 0.0
        let c1: f32 = 0.0
        let c2: f32 = 0.0
        let c3: f32 = 0.0
        let c4: f32 = 0.0
        let c5: f32 = 0.0
        let c6: f32 = 0.0
        let c7: f32 = 0.0
        if col < p.n {
            for kk in 0..p.kc {
                let kg = k0 + kk
                if kg < p.k {
                    let bv = b[kg * p.n + col]
                c0 = c0 + as_tile[0 + kk] * bv
                c1 = c1 + as_tile[128 + kk] * bv
                c2 = c2 + as_tile[256 + kk] * bv
                c3 = c3 + as_tile[384 + kk] * bv
                c4 = c4 + as_tile[512 + kk] * bv
                c5 = c5 + as_tile[640 + kk] * bv
                c6 = c6 + as_tile[768 + kk] * bv
                c7 = c7 + as_tile[896 + kk] * bv
                }
            }
            if 0 < p.m {
                part[(gy * p.m + 0) * p.n + col] = p.alpha * c0
            }
            if 1 < p.m {
                part[(gy * p.m + 1) * p.n + col] = p.alpha * c1
            }
            if 2 < p.m {
                part[(gy * p.m + 2) * p.n + col] = p.alpha * c2
            }
            if 3 < p.m {
                part[(gy * p.m + 3) * p.n + col] = p.alpha * c3
            }
            if 4 < p.m {
                part[(gy * p.m + 4) * p.n + col] = p.alpha * c4
            }
            if 5 < p.m {
                part[(gy * p.m + 5) * p.n + col] = p.alpha * c5
            }
            if 6 < p.m {
                part[(gy * p.m + 6) * p.n + col] = p.alpha * c6
            }
            if 7 < p.m {
                part[(gy * p.m + 7) * p.n + col] = p.alpha * c7
            }
        }
    }
);

pub(crate) const SKINNY_16_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32, kc: u32, alpha: f32 }
    @workgroup_size(256)
    kernel cs_skinny_16(a: device buffer<f32>, b: device buffer<f32>, part: device mut buffer<f32>, p: P) {
        shared as_tile: array<f32, 2048>
        let lx = local_id().x
        let col = group_id().x * 256 + lx
        let gy = group_id().y
        let k0 = gy * p.kc
        for s in 0..8 {
            let idx = lx + s * 256
            if idx < 2048 {
                let r = idx / 128
                let kk = idx - r * 128
                if r < p.m && kk < p.kc && k0 + kk < p.k {
                    as_tile[idx] = a[r * p.k + k0 + kk]
                } else {
                    as_tile[idx] = 0.0
                }
            }
        }
        barrier()
        let c0: f32 = 0.0
        let c1: f32 = 0.0
        let c2: f32 = 0.0
        let c3: f32 = 0.0
        let c4: f32 = 0.0
        let c5: f32 = 0.0
        let c6: f32 = 0.0
        let c7: f32 = 0.0
        let c8: f32 = 0.0
        let c9: f32 = 0.0
        let c10: f32 = 0.0
        let c11: f32 = 0.0
        let c12: f32 = 0.0
        let c13: f32 = 0.0
        let c14: f32 = 0.0
        let c15: f32 = 0.0
        if col < p.n {
            for kk in 0..p.kc {
                let kg = k0 + kk
                if kg < p.k {
                    let bv = b[kg * p.n + col]
                c0 = c0 + as_tile[0 + kk] * bv
                c1 = c1 + as_tile[128 + kk] * bv
                c2 = c2 + as_tile[256 + kk] * bv
                c3 = c3 + as_tile[384 + kk] * bv
                c4 = c4 + as_tile[512 + kk] * bv
                c5 = c5 + as_tile[640 + kk] * bv
                c6 = c6 + as_tile[768 + kk] * bv
                c7 = c7 + as_tile[896 + kk] * bv
                c8 = c8 + as_tile[1024 + kk] * bv
                c9 = c9 + as_tile[1152 + kk] * bv
                c10 = c10 + as_tile[1280 + kk] * bv
                c11 = c11 + as_tile[1408 + kk] * bv
                c12 = c12 + as_tile[1536 + kk] * bv
                c13 = c13 + as_tile[1664 + kk] * bv
                c14 = c14 + as_tile[1792 + kk] * bv
                c15 = c15 + as_tile[1920 + kk] * bv
                }
            }
            if 0 < p.m {
                part[(gy * p.m + 0) * p.n + col] = p.alpha * c0
            }
            if 1 < p.m {
                part[(gy * p.m + 1) * p.n + col] = p.alpha * c1
            }
            if 2 < p.m {
                part[(gy * p.m + 2) * p.n + col] = p.alpha * c2
            }
            if 3 < p.m {
                part[(gy * p.m + 3) * p.n + col] = p.alpha * c3
            }
            if 4 < p.m {
                part[(gy * p.m + 4) * p.n + col] = p.alpha * c4
            }
            if 5 < p.m {
                part[(gy * p.m + 5) * p.n + col] = p.alpha * c5
            }
            if 6 < p.m {
                part[(gy * p.m + 6) * p.n + col] = p.alpha * c6
            }
            if 7 < p.m {
                part[(gy * p.m + 7) * p.n + col] = p.alpha * c7
            }
            if 8 < p.m {
                part[(gy * p.m + 8) * p.n + col] = p.alpha * c8
            }
            if 9 < p.m {
                part[(gy * p.m + 9) * p.n + col] = p.alpha * c9
            }
            if 10 < p.m {
                part[(gy * p.m + 10) * p.n + col] = p.alpha * c10
            }
            if 11 < p.m {
                part[(gy * p.m + 11) * p.n + col] = p.alpha * c11
            }
            if 12 < p.m {
                part[(gy * p.m + 12) * p.n + col] = p.alpha * c12
            }
            if 13 < p.m {
                part[(gy * p.m + 13) * p.n + col] = p.alpha * c13
            }
            if 14 < p.m {
                part[(gy * p.m + 14) * p.n + col] = p.alpha * c14
            }
            if 15 < p.m {
                part[(gy * p.m + 15) * p.n + col] = p.alpha * c15
            }
        }
    }
);

pub(crate) const SPLIT_REDUCE_ZSL: ZslShader = zsl!(
    push P { mn: u32, split: u32 }
    @workgroup_size(256)
    kernel cs_split_reduce(part: device buffer<f32>, c: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.mn {
            let sum: f32 = 0.0
            for s in 0..p.split {
                sum = sum + part[s * p.mn + i]
            }
            c[i] = sum
        }
    }
);
