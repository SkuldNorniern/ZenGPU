use zengpu_spirv::{_zsl_priv::Scalar, ZslPushConst};

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
