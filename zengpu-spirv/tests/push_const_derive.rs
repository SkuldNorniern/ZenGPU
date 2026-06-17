use zengpu_spirv::{_zsl_priv::Scalar, ZslMat4, ZslPushConst};

#[derive(ZslPushConst)]
struct ScalePush {
    len: u32,
    offset: i32,
    scale: f32,
}

#[derive(ZslPushConst)]
struct AllU32 {
    a: u32,
    b: u32,
    c: u32,
}

#[derive(ZslPushConst)]
struct Empty {}

#[test]
fn to_scalars_types_and_order() {
    let p = ScalePush {
        len: 1024,
        offset: -7,
        scale: 2.5,
    };
    let s = p.to_scalars();
    assert_eq!(s[0], Scalar::U32(1024));
    assert_eq!(s[1], Scalar::I32(-7));
    assert_eq!(s[2], Scalar::F32(2.5));
}

#[test]
fn to_scalars_all_u32() {
    let p = AllU32 { a: 3, b: 5, c: 7 };
    let [a, b, c] = p.to_scalars();
    assert_eq!(a, Scalar::U32(3));
    assert_eq!(b, Scalar::U32(5));
    assert_eq!(c, Scalar::U32(7));
}

#[test]
fn to_scalars_empty() {
    let p = Empty {};
    let s: [Scalar; 0] = p.to_scalars();
    assert_eq!(s.len(), 0);
}

#[test]
fn to_scalars_coerces_to_slice() {
    let p = AllU32 { a: 1, b: 2, c: 3 };
    let scalars = p.to_scalars();
    let slice: &[Scalar] = &scalars;
    assert_eq!(slice.len(), 3);
}

// ── ZslMat4 ──────────────────────────────────────────────────────────────────

#[derive(ZslPushConst)]
struct MvpPush {
    mvp: ZslMat4,
}

#[derive(ZslPushConst)]
struct MixedPush {
    scale: f32,
    mvp: ZslMat4,
    tint: u32,
}

#[test]
fn mat4_expands_to_16_scalars() {
    let identity = [
        1.0f32, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ];
    let p = MvpPush {
        mvp: ZslMat4(identity),
    };
    let s = p.to_scalars();
    assert_eq!(s.len(), 16);
    assert_eq!(s[0], Scalar::F32(1.0));
    assert_eq!(s[5], Scalar::F32(1.0));
    assert_eq!(s[10], Scalar::F32(1.0));
    assert_eq!(s[15], Scalar::F32(1.0));
    // off-diagonal must be 0
    assert_eq!(s[1], Scalar::F32(0.0));
    assert_eq!(s[4], Scalar::F32(0.0));
}

#[test]
fn mixed_push_mat4_ordering() {
    let identity = [1.0f32; 16];
    let p = MixedPush {
        scale: 2.0,
        mvp: ZslMat4(identity),
        tint: 255,
    };
    let s = p.to_scalars();
    // 1 (scale) + 16 (mvp) + 1 (tint) = 18
    assert_eq!(s.len(), 18);
    assert_eq!(s[0], Scalar::F32(2.0));
    assert_eq!(s[1], Scalar::F32(1.0)); // mvp[0]
    assert_eq!(s[16], Scalar::F32(1.0)); // mvp[15]
    assert_eq!(s[17], Scalar::U32(255));
}
