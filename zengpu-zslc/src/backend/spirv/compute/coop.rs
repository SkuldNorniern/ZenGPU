//! `frag<…>` onto `VK_KHR_cooperative_matrix`: subgroup scope, 16x16x16, MatrixA /
//! MatrixB / MatrixAccumulator.
//!
//! Element-to-lane mapping is opaque, so only load / mma / store lower. Loads read a 16- or
//! 8-bit view of the `buffer<u32>` binding. Packed i8 fragments address bytes, f16 fragments
//! address halves.

use std::collections::{HashMap, HashSet};

use super::{LowerCtx, SharedVar, Val, coerce};
use crate::backend::spirv::builder::{Id, SpvBuilder, cap, deco, sc};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrExpr};
use crate::ir::typeck::{StorageElem, frag_parts, frag16_id, storage_elem};
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Type, TypeId};

const ROW_MAJOR: u32 = 0;
/// ColumnMajorKHR for `frag_load_b_t`: N x K row-major B is column-major over K x N
const COLUMN_MAJOR: u32 = 1;
const SCOPE_SUBGROUP: u32 = 3;
/// `CooperativeMatrixOperands` for signed A, B, C and result.
const SIGNED_OPERANDS: u32 = 0x1 | 0x2 | 0x4 | 0x8;
/// Binding of the u32 SSBO table shared by `buffer<f16>` and `buffer<u32>`.
const U32_TABLE_BINDING: u32 = 2;

/// The matrix and table declarations fragments need, created on first use.
#[derive(Default)]
pub(super) struct Coop {
    matrices: HashMap<TypeId, Id>,
    caps: HashSet<u32>,
    exts: HashSet<&'static str>,
    t_f16: Option<Id>,
    t_i8: Option<Id>,
    /// Aliased views of the u32 table, keyed by element width in bits.
    views: HashMap<u32, (Id, Id)>,
    vec_views: HashMap<u8, (Id, Id)>,
    /// Workgroup f16 staging array for `frag_load_a|b`: the variable and its pointer type.
    stage: Option<(Id, Id)>,
}

/// Subgroup size pinned to match `zengpu-vulkan`, with one 16x16 tile per subgroup.
const SUBGROUP_LANES: u32 = 32;
const TILE_ELEMS: u32 = 16 * 16;
const SEMANTICS_WG_ACQ_REL: u32 = 0x108;

impl Coop {
    fn require(&mut self, spv: &mut SpvBuilder, capability: u32, ext: Option<&'static str>) {
        if self.caps.insert(capability) {
            spv.capability(capability);
        }
        if let Some(name) = ext
            && self.exts.insert(name)
        {
            spv.extension(name);
        }
    }

    fn t_f16(&mut self, spv: &mut SpvBuilder) -> Id {
        *self.t_f16.get_or_insert_with(|| spv.type_float(16))
    }

    fn t_i8(&mut self, spv: &mut SpvBuilder) -> Id {
        *self.t_i8.get_or_insert_with(|| spv.type_int(8, true))
    }

    pub(super) fn type_of(
        &mut self,
        spv: &mut SpvBuilder,
        types: &TypeArena,
        ty: TypeId,
        t_u32: Id,
        t_f32: Id,
        t_i32: Id,
    ) -> Result<Id, String> {
        if let Some(&id) = self.matrices.get(&ty) {
            return Ok(id);
        }
        let (elem, usage) =
            frag_parts(types, ty).ok_or_else(|| "ZSL: expected a frag type".to_string())?;
        let component = match types.get(elem) {
            Type::Float {
                format: FloatFormat::F16,
            } => {
                self.require(spv, cap::FLOAT16, None);
                self.t_f16(spv)
            }
            Type::Int {
                width: IntWidth::W8,
                ..
            } => {
                self.require(spv, cap::INT8, None);
                self.t_i8(spv)
            }
            Type::Float {
                format: FloatFormat::F32,
            } => t_f32,
            Type::Int {
                width: IntWidth::W32,
                ..
            } => t_i32,
            other => {
                return Err(format!("ZSL: no SPIR-V cooperative matrix for {other:?}"));
            }
        };
        self.require(
            spv,
            cap::COOPERATIVE_MATRIX_KHR,
            Some("SPV_KHR_cooperative_matrix"),
        );
        let use_code = match usage {
            FragUse::A => 0,
            FragUse::B => 1,
            FragUse::Acc => 2,
        };
        let scope = spv.type_constant_u32(t_u32, SCOPE_SUBGROUP);
        let rows = spv.type_constant_u32(t_u32, 16);
        let cols = spv.type_constant_u32(t_u32, 16);
        let usage_id = spv.type_constant_u32(t_u32, use_code);
        let id = spv.type_cooperative_matrix(component, scope, rows, cols, usage_id);
        self.matrices.insert(ty, id);
        Ok(id)
    }

    fn stage(&mut self, spv: &mut SpvBuilder, t_u32: Id, subgroups: u32) -> (Id, Id) {
        if let Some(stage) = self.stage {
            return stage;
        }
        let t_f16 = self.t_f16(spv);
        let len = spv.constant_u32(t_u32, subgroups * TILE_ELEMS);
        let array_ty = spv.type_array(t_f16, len);
        let ptr_ty = spv.type_pointer_global(sc::WORKGROUP, array_ty);
        let var = spv.global_variable(ptr_ty, sc::WORKGROUP);
        let t_ptr_elem = spv.type_pointer(sc::WORKGROUP, t_f16);
        self.stage = Some((var, t_ptr_elem));
        (var, t_ptr_elem)
    }

    fn view(&mut self, spv: &mut SpvBuilder, bits: u32) -> (Id, Id) {
        if let Some(&v) = self.views.get(&bits) {
            return v;
        }
        let (t_elem, stride) = if bits == 16 {
            (self.t_f16(spv), 2)
        } else {
            (self.t_i8(spv), 1)
        };
        if bits == 16 {
            self.require(
                spv,
                cap::STORAGE_BUFFER_16BIT_ACCESS,
                Some("SPV_KHR_16bit_storage"),
            );
        } else {
            self.require(
                spv,
                cap::STORAGE_BUFFER_8BIT_ACCESS,
                Some("SPV_KHR_8bit_storage"),
            );
        }
        let view = alias_u32_table(spv, t_elem, stride);
        self.views.insert(bits, view);
        view
    }

    pub(super) fn vec_view(&mut self, spv: &mut SpvBuilder, t_u32: Id, lanes: u8) -> (Id, Id) {
        if let Some(&v) = self.vec_views.get(&lanes) {
            return v;
        }
        let t_elem = spv.type_vector(t_u32, u32::from(lanes));
        let view = alias_u32_table(spv, t_elem, 4 * u32::from(lanes));
        self.vec_views.insert(lanes, view);
        view
    }
}

/// Aliased SSBO at the u32 binding. `stride` is the element size in bytes.
fn alias_u32_table(spv: &mut SpvBuilder, t_elem: Id, stride: u32) -> (Id, Id) {
    let t_ra = spv.type_runtime_array(t_elem);
    spv.decorate(t_ra, deco::ARRAY_STRIDE, &[stride]);
    let t_struct = spv.type_struct(&[t_ra]);
    spv.decorate(t_struct, deco::BLOCK, &[]);
    spv.member_decorate(t_struct, 0, deco::OFFSET, &[0]);
    let t_ra_struct = spv.type_runtime_array(t_struct);
    let t_ptr_ra = spv.type_pointer(sc::STORAGE_BUFFER, t_ra_struct);
    let t_ptr_elem = spv.type_pointer(sc::STORAGE_BUFFER, t_elem);
    let var = spv.global_variable(t_ptr_ra, sc::STORAGE_BUFFER);
    spv.decorate(var, deco::DESCRIPTOR_SET, &[0]);
    spv.decorate(var, deco::BINDING, &[U32_TABLE_BINDING]);
    spv.decorate(var, deco::ALIASED, &[]);
    (var, t_ptr_elem)
}

impl LowerCtx<'_> {
    pub(super) fn frag_spv_ty(&mut self, ty: TypeId) -> Result<Id, String> {
        self.coop.type_of(
            self.spv,
            &self.types,
            ty,
            self.t_u32,
            self.t_f32,
            self.t_i32,
        )
    }

    /// Offset in bytes for packed i8 operands, in elements otherwise.
    fn frag_elem_ptr(&mut self, src: &FragSrc, frag_ty: TypeId, offset: Id) -> Result<Id, String> {
        let (_, usage) = frag_parts(&self.types, frag_ty)
            .ok_or_else(|| "ZSL: expected a frag type".to_string())?;
        match src {
            FragSrc::Shared(name) => {
                let sv = *self
                    .shared
                    .get(name)
                    .ok_or_else(|| format!("ZSL: `{name}` is not a shared array"))?;
                Ok(self.spv.op_access_chain(sv.elem_ptr, sv.var, &[offset]))
            }
            FragSrc::Buffer(name) => {
                let info = *self
                    .buf_params
                    .get(name)
                    .ok_or_else(|| format!("ZSL: `{name}` is not a buffer param"))?;
                let elem_ty = self.types.get(info.elem).clone();
                let (var, t_ptr_elem) = match (usage, elem_ty) {
                    (FragUse::Acc, _) => {
                        let table = *self
                            .ssbo_tables
                            .get(&info.table)
                            .ok_or_else(|| "ZSL: buffer has no SSBO table".to_string())?;
                        (table.var, table.t_ptr_elem)
                    }
                    (_, Type::Float { .. }) => self.coop.view(self.spv, 16),
                    (_, Type::Int { .. }) => self.coop.view(self.spv, 8),
                    _ => return Err("ZSL: frag source element does not fit this use".into()),
                };
                let pc_var = self
                    .pc_var
                    .ok_or_else(|| "ZSL: no push constant block".to_string())?;
                let pc_field = self.spv.constant_u32(self.t_u32, info.pc_index);
                let chain = self
                    .spv
                    .op_access_chain(self.t_ptr_pc_u32, pc_var, &[pc_field]);
                let buf_idx = self.spv.op_load(self.t_u32, chain);
                let c0 = self.const_0_u32;
                Ok(self
                    .spv
                    .op_access_chain(t_ptr_elem, var, &[buf_idx, c0, offset]))
            }
        }
    }

    fn frag_offset_and_stride(&mut self, offset: &IrExpr, ld: &IrExpr) -> Result<(Id, Id), String> {
        let u32_id = self.u32_id;
        let off = super::lower_expr(self, offset)?;
        let off = coerce(self, off, u32_id);
        let stride = super::lower_expr(self, ld)?;
        let stride = coerce(self, stride, u32_id);
        Ok((off, stride))
    }

    /// Cooperative loads cannot convert f32 to f16, so f32 shared is staged first. f16 and i8
    /// load directly.
    fn is_f32_shared(&self, sv: &SharedVar) -> bool {
        storage_elem(&self.types, sv.elem) == Some(StorageElem::F32)
    }

    pub(super) fn lower_frag_load(
        &mut self,
        ty: TypeId,
        src: &FragSrc,
        offset: &IrExpr,
        ld: &IrExpr,
        trans: bool,
    ) -> Result<Val, String> {
        let (_, usage) =
            frag_parts(&self.types, ty).ok_or_else(|| "ZSL: expected a frag type".to_string())?;
        if trans
            && let FragSrc::Shared(name) = src
            && self
                .shared
                .get(name)
                .is_some_and(|sv| self.is_f32_shared(sv))
        {
            return Err(format!(
                "ZSL: SPIR-V frag_load_b_t from f32 shared `{name}` is not lowered (stage it as f16 first)"
            ));
        }
        if let FragSrc::Shared(name) = src
            && usage != FragUse::Acc
            && self
                .shared
                .get(name)
                .is_some_and(|sv| self.is_f32_shared(sv))
        {
            return self.lower_shared_frag_ab(name, ty, offset, ld);
        }
        let (off, stride) = self.frag_offset_and_stride(offset, ld)?;
        let ptr = self.frag_elem_ptr(src, ty, off)?;
        let coop_ty = self.frag_spv_ty(ty)?;
        let layout = self
            .spv
            .constant_u32(self.t_u32, if trans { COLUMN_MAJOR } else { ROW_MAJOR });
        let id = self
            .spv
            .op_cooperative_matrix_load(coop_ty, ptr, layout, stride);
        Ok(Val { id, ty })
    }

    /// Loads from f32 shared via `frag_load_a|b(shared, offset, ld)`.
    ///
    /// Cooperative loads cannot convert, so each lane writes its share as f16 into its subgroup's
    /// slice of a workgroup f16 array, then the subgroup loads it. Element `e = lane + 32 * i` is
    /// `(2 * i + lane / 16, lane % 16)`: lane `l` reads
    /// `off + (l / 16) * ld + l % 16 + 2 * i * ld`.
    fn lower_shared_frag_ab(
        &mut self,
        name: &str,
        ty: TypeId,
        offset: &IrExpr,
        ld: &IrExpr,
    ) -> Result<Val, String> {
        let sv = *self
            .shared
            .get(name)
            .ok_or_else(|| format!("ZSL: `{name}` is not a shared array"))?;
        if !self.is_f32_shared(&sv) {
            return Err(format!(
                "ZSL: SPIR-V frag loads and stores from shared `{name}` need an f32 array"
            ));
        }
        let tile = sv.var;
        let (off, stride) = self.frag_offset_and_stride(offset, ld)?;
        let coop_ty = self.frag_spv_ty(ty)?;
        let [sx, sy, sz] = self.local_size;
        let subgroups = (sx * sy * sz).div_ceil(SUBGROUP_LANES);
        let (stage, t_ptr_f16) = self.coop.stage(self.spv, self.t_u32, subgroups);
        let t_f16 = self.coop.t_f16(self.spv);
        let t_u32 = self.t_u32;
        let t_f32 = self.t_f32;
        let scope = self.spv.constant_u32(t_u32, SCOPE_SUBGROUP);
        let semantics = self.spv.constant_u32(t_u32, SEMANTICS_WG_ACQ_REL);

        let lid3 = self.spv.op_load(self.t_uvec3, self.lid_var);
        let x = self.spv.op_composite_extract(t_u32, lid3, &[0]);
        let y = self.spv.op_composite_extract(t_u32, lid3, &[1]);
        let z = self.spv.op_composite_extract(t_u32, lid3, &[2]);
        let sy_c = self.spv.constant_u32(t_u32, sy);
        let sx_c = self.spv.constant_u32(t_u32, sx);
        let zs = self.spv.op_imul(t_u32, z, sy_c);
        let ys = self.spv.op_iadd(t_u32, y, zs);
        let ys_sx = self.spv.op_imul(t_u32, ys, sx_c);
        let lin = self.spv.op_iadd(t_u32, x, ys_sx);
        let c31 = self.spv.constant_u32(t_u32, SUBGROUP_LANES - 1);
        let lane = self.spv.op_bitwise_and(t_u32, lin, c31);
        let c5 = self.spv.constant_u32(t_u32, 5);
        let sub = self.spv.op_shift_right_logical(t_u32, lin, c5);
        let c8 = self.spv.constant_u32(t_u32, 8);
        let slice = self.spv.op_shift_left_logical(t_u32, sub, c8);

        let c4 = self.spv.constant_u32(t_u32, 4);
        let row_hi = self.spv.op_shift_right_logical(t_u32, lane, c4);
        let c15 = self.spv.constant_u32(t_u32, 15);
        let col = self.spv.op_bitwise_and(t_u32, lane, c15);
        let row_off = self.spv.op_imul(t_u32, row_hi, stride);
        let base = self.spv.op_iadd(t_u32, off, row_off);
        let mut src_idx = self.spv.op_iadd(t_u32, base, col);
        let c1 = self.spv.constant_u32(t_u32, 1);
        let step = self.spv.op_shift_left_logical(t_u32, stride, c1);
        let stage_lane = self.spv.op_iadd(t_u32, slice, lane);

        self.spv.op_control_barrier(scope, scope, semantics);
        for i in 0..8u32 {
            let src_ptr = self
                .spv
                .op_access_chain(self.t_ptr_wg_f32, tile, &[src_idx]);
            let value = self.spv.op_load(t_f32, src_ptr);
            let half = self.spv.op_convert_f_to_f(t_f16, value);
            let dst_idx = if i == 0 {
                stage_lane
            } else {
                let k = self.spv.constant_u32(t_u32, SUBGROUP_LANES * i);
                self.spv.op_iadd(t_u32, stage_lane, k)
            };
            let dst = self.spv.op_access_chain(t_ptr_f16, stage, &[dst_idx]);
            self.spv.op_store(dst, half);
            if i < 7 {
                src_idx = self.spv.op_iadd(t_u32, src_idx, step);
            }
        }
        self.spv.op_control_barrier(scope, scope, semantics);

        let tile_ptr = self.spv.op_access_chain(t_ptr_f16, stage, &[slice]);
        let layout = self.spv.constant_u32(t_u32, ROW_MAJOR);
        let row_stride = self.spv.constant_u32(t_u32, 16);
        let id = self
            .spv
            .op_cooperative_matrix_load(coop_ty, tile_ptr, layout, row_stride);
        self.spv.op_control_barrier(scope, scope, semantics);
        Ok(Val { id, ty })
    }

    pub(super) fn lower_frag_store(
        &mut self,
        src: &FragSrc,
        offset: &IrExpr,
        ld: &IrExpr,
        value: &IrExpr,
    ) -> Result<(), String> {
        let val = super::lower_expr(self, value)?;
        let (off, stride) = self.frag_offset_and_stride(offset, ld)?;
        let ptr = self.frag_elem_ptr(src, val.ty, off)?;
        let layout = self.spv.constant_u32(self.t_u32, ROW_MAJOR);
        self.spv
            .op_cooperative_matrix_store(ptr, val.id, layout, stride);
        Ok(())
    }

    pub(super) fn lower_mma(&mut self, args: &[IrExpr]) -> Result<Val, String> {
        let a = super::lower_expr(self, &args[0])?;
        let b = super::lower_expr(self, &args[1])?;
        let (elem, _) = frag_parts(&self.types, a.ty)
            .ok_or_else(|| "ZSL: mma operand is not a frag".to_string())?;
        let int_operands = matches!(
            self.types.get(elem),
            Type::Int {
                width: IntWidth::W8,
                ..
            }
        );
        let acc_elem = if int_operands {
            self.i32_id
        } else {
            self.f32_id
        };
        let acc_ty = frag16_id(&mut self.types, acc_elem, FragUse::Acc);
        let coop_acc = self.frag_spv_ty(acc_ty)?;
        let c = self.frag_value_or_zero(&args[2], acc_ty)?;
        let flags = if int_operands { SIGNED_OPERANDS } else { 0 };
        let id = self
            .spv
            .op_cooperative_matrix_mul_add(coop_acc, a.id, b.id, c, flags);
        Ok(Val { id, ty: acc_ty })
    }

    /// Null matrix of the wanted type, as produced by `frag_zero_acc()`.
    pub(super) fn frag_value_or_zero(&mut self, expr: &IrExpr, want: TypeId) -> Result<Id, String> {
        if let IrExpr::Builtin {
            func: BuiltinFn::FragZeroAcc,
            ..
        } = expr
        {
            let coop = self.frag_spv_ty(want)?;
            return Ok(self.spv.constant_null(coop));
        }
        Ok(super::lower_expr(self, expr)?.id)
    }
}
