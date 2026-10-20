//! `frag<…>` onto `VK_KHR_cooperative_matrix`: subgroup scope, 16x16x16, MatrixA /
//! MatrixB / MatrixAccumulator.
//!
//! Element-to-lane mapping is opaque, so only load / mma / store lower. Loads read a 16- or
//! 8-bit view of the `buffer<u32>` binding. Packed i8 fragments address bytes, f16 fragments
//! address halves.

use std::collections::{HashMap, HashSet};

use super::{LowerCtx, Val, coerce};
use crate::backend::spirv::builder::{Id, SpvBuilder, cap, deco, sc};
use crate::ir::arena::TypeArena;
use crate::ir::node::{BuiltinFn, FragSrc, IrExpr};
use crate::ir::typeck::{frag_parts, frag16_id};
use crate::ir::types::{FloatFormat, FragUse, IntWidth, Type, TypeId};

const ROW_MAJOR: u32 = 0;
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
}

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
        self.views.insert(bits, (var, t_ptr_elem));
        (var, t_ptr_elem)
    }
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
                let var = *self
                    .shared
                    .get(name)
                    .ok_or_else(|| format!("ZSL: `{name}` is not a shared array"))?;
                Ok(self.spv.op_access_chain(self.t_ptr_wg_f32, var, &[offset]))
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

    pub(super) fn lower_frag_load(
        &mut self,
        ty: TypeId,
        src: &FragSrc,
        offset: &IrExpr,
        ld: &IrExpr,
    ) -> Result<Val, String> {
        let (_, usage) =
            frag_parts(&self.types, ty).ok_or_else(|| "ZSL: expected a frag type".to_string())?;
        if usage != FragUse::Acc && matches!(src, FragSrc::Shared(_)) {
            return Err(
                "SPIR-V backend: frag_load_a/frag_load_b from a shared array is not lowered yet"
                    .into(),
            );
        }
        let (off, stride) = self.frag_offset_and_stride(offset, ld)?;
        let ptr = self.frag_elem_ptr(src, ty, off)?;
        let coop_ty = self.frag_spv_ty(ty)?;
        let layout = self.spv.constant_u32(self.t_u32, ROW_MAJOR);
        let id = self
            .spv
            .op_cooperative_matrix_load(coop_ty, ptr, layout, stride);
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
