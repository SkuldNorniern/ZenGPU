//! Minimal SPIR-V binary builder.
//!
//! Emits word-encoded SPIR-V for the subset used by ZSL shaders: SSBOs,
//! push constants, scalars, vectors, matrices, and arithmetic/logic ops.

use std::collections::HashMap;

/// A SPIR-V result ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub u32);

/// Opcodes we actually emit.
#[allow(dead_code)]
mod op {
    pub const EXTENSION: u32 = 10;
    pub const CAPABILITY: u32 = 17;
    pub const EXT_INST_IMPORT: u32 = 11;
    pub const MEMORY_MODEL: u32 = 14;
    pub const ENTRY_POINT: u32 = 15;
    pub const EXECUTION_MODE: u32 = 16;
    pub const DECORATE: u32 = 71;
    pub const MEMBER_DECORATE: u32 = 72;
    pub const TYPE_VOID: u32 = 19;
    pub const TYPE_BOOL: u32 = 20;
    pub const TYPE_INT: u32 = 21;
    pub const TYPE_FLOAT: u32 = 22;
    pub const TYPE_VECTOR: u32 = 23;
    pub const TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const TYPE_ARRAY: u32 = 28;
    pub const TYPE_STRUCT: u32 = 30;
    pub const TYPE_POINTER: u32 = 32;
    pub const TYPE_FUNCTION: u32 = 33;
    pub const CONSTANT: u32 = 43;
    pub const VARIABLE: u32 = 59;
    pub const LOAD: u32 = 61;
    pub const STORE: u32 = 62;
    pub const ACCESS_CHAIN: u32 = 65;
    pub const FUNCTION: u32 = 54;
    pub const FUNCTION_PARAMETER: u32 = 55;
    pub const FUNCTION_END: u32 = 56;
    pub const LABEL: u32 = 248;
    pub const BRANCH: u32 = 249;
    pub const BRANCH_CONDITIONAL: u32 = 250;
    pub const SELECTION_MERGE: u32 = 247;
    pub const LOOP_MERGE: u32 = 246;
    pub const RETURN: u32 = 253;
    pub const CONTROL_BARRIER: u32 = 224;
    pub const ATOMIC_FADD_EXT: u32 = 6035;
    pub const IADD: u32 = 128;
    pub const FADD: u32 = 129;
    pub const ISUB: u32 = 130;
    pub const FSUB: u32 = 131;
    pub const IMUL: u32 = 132;
    pub const FMUL: u32 = 133;
    pub const UDIV: u32 = 134;
    pub const FDIV: u32 = 136;
    pub const ULESS_THAN: u32 = 176;
    pub const ULESS_THAN_EQ: u32 = 178;
    pub const UGREATER_THAN: u32 = 172;
    pub const UGREATER_THAN_EQ: u32 = 174;
    pub const SLESS_THAN: u32 = 177;
    pub const FORD_LESS_THAN: u32 = 184;
    pub const FORD_GREATER_THAN: u32 = 186;
    pub const FORD_LESS_THAN_EQ: u32 = 188;
    pub const FORD_GREATER_THAN_EQ: u32 = 190;
    pub const FLESS_THAN: u32 = 185;
    pub const COMPOSITE_CONSTRUCT: u32 = 80;
    pub const COMPOSITE_EXTRACT: u32 = 81;
    pub const CONVERT_F_TO_U: u32 = 109;
    pub const CONVERT_U_TO_F: u32 = 112;
    pub const BITCAST: u32 = 124;
    pub const SNEGATE: u32 = 126;
    pub const FNEGATE: u32 = 127;
    pub const VECTOR_TIMES_SCALAR: u32 = 142;
    pub const TYPE_MATRIX: u32 = 24;
    // SPIR-V: 144 = OpVectorTimesMatrix, 145 = OpMatrixTimesVector. We need the
    // latter (matrix on the left, vector on the right); 144 swaps the operand
    // roles and makes drivers fault during pipeline compilation.
    pub const MATRIX_TIMES_VECTOR: u32 = 145;
    pub const DOT: u32 = 148;
    pub const LOGICAL_OR: u32 = 166;
    pub const LOGICAL_AND: u32 = 167;
    pub const IEQUAL: u32 = 170;
    pub const INOT_EQUAL: u32 = 171;
    pub const FORD_EQUAL: u32 = 180;
    pub const FORD_NOT_EQUAL: u32 = 182;
    pub const EXT_INST: u32 = 12;
}

/// SPIR-V decoration constants.
#[allow(dead_code)]
pub mod deco {
    pub const BLOCK: u32 = 2;
    pub const BUFFER_BLOCK: u32 = 3;
    pub const COL_MAJOR: u32 = 5;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const BUILT_IN: u32 = 11;
    pub const NON_WRITABLE: u32 = 24;
    pub const LOCATION: u32 = 30;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

/// SPIR-V storage class constants.
#[allow(dead_code)]
pub mod sc {
    pub const INPUT: u32 = 1;
    pub const OUTPUT: u32 = 3;
    pub const UNIFORM: u32 = 2;
    pub const STORAGE_BUFFER: u32 = 12;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const FUNCTION: u32 = 7;
    pub const WORKGROUP: u32 = 4;
}

/// SPIR-V capability constants.
#[allow(dead_code)]
mod cap {
    pub const SHADER: u32 = 1;
    pub const RUNTIME_DESCRIPTOR_ARRAY: u32 = 4437;
    pub const ATOMIC_FLOAT32_ADD_EXT: u32 = 6033;
}

/// SPIR-V execution model constants.
mod exec_model {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
}

/// SPIR-V built-in constants.
pub mod builtin {
    pub const POSITION: u32 = 0;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const WORKGROUP_ID: u32 = 26;
}

/// Accumulates SPIR-V instructions, serializing to a word stream.
#[allow(dead_code)]
pub struct SpvBuilder {
    next_id: u32,
    // Ordered sections (each as raw word streams).
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    // Constants and global variables share the same section in SPIR-V.
    constants_globals: Vec<u32>,
    functions: Vec<u32>,
    // Scalar-constant cache, keyed by (type id, value bits). SPIR-V requires a
    // given <type, value> constant to be defined once; emitting it twice makes
    // some drivers' shader compilers fault. Dedup keeps each unique.
    constant_cache: HashMap<(u32, u32), Id>,
}

#[allow(dead_code)]
impl SpvBuilder {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            capabilities: Vec::new(),
            extensions: Vec::new(),
            imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            exec_modes: Vec::new(),
            annotations: Vec::new(),
            types: Vec::new(),
            constants_globals: Vec::new(),
            functions: Vec::new(),
            constant_cache: HashMap::new(),
        }
    }

    pub fn fresh_id(&mut self) -> Id {
        let id = Id(self.next_id);
        self.next_id += 1;
        id
    }

    // ── Preamble ──────────────────────────────────────────────────────────────

    pub fn capability_shader(&mut self) {
        emit(&mut self.capabilities, op::CAPABILITY, &[cap::SHADER]);
    }

    pub fn capability_runtime_descriptor_array(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::RUNTIME_DESCRIPTOR_ARRAY],
        );
    }

    pub fn enable_atomic_float32_add_ext(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::ATOMIC_FLOAT32_ADD_EXT],
        );
        let name = encode_string("SPV_EXT_shader_atomic_float");
        emit_raw(&mut self.extensions, op::EXTENSION, &name);
    }

    /// Returns the ID of the GLSL.std.450 import.
    pub fn ext_inst_import_glsl(&mut self) -> Id {
        let id = self.fresh_id();
        let name = encode_string("GLSL.std.450");
        let mut words = vec![id.0];
        words.extend_from_slice(&name);
        emit_raw(&mut self.imports, op::EXT_INST_IMPORT, &words);
        id
    }

    pub fn memory_model_logical_glsl450(&mut self) {
        emit(&mut self.memory_model, op::MEMORY_MODEL, &[0, 1]);
    }

    pub fn entry_point_glcompute(&mut self, fn_id: Id, name: &str, interface: &[Id]) {
        let name_enc = encode_string(name);
        let mut words = vec![exec_model::GL_COMPUTE, fn_id.0];
        words.extend_from_slice(&name_enc);
        for &iface in interface {
            words.push(iface.0);
        }
        emit_raw(&mut self.entry_points, op::ENTRY_POINT, &words);
    }

    pub fn entry_point_vertex(&mut self, fn_id: Id, name: &str, interface: &[Id]) {
        let name_enc = encode_string(name);
        let mut words = vec![exec_model::VERTEX, fn_id.0];
        words.extend_from_slice(&name_enc);
        for &iface in interface {
            words.push(iface.0);
        }
        emit_raw(&mut self.entry_points, op::ENTRY_POINT, &words);
    }

    pub fn entry_point_fragment(&mut self, fn_id: Id, name: &str, interface: &[Id]) {
        let name_enc = encode_string(name);
        let mut words = vec![exec_model::FRAGMENT, fn_id.0];
        words.extend_from_slice(&name_enc);
        for &iface in interface {
            words.push(iface.0);
        }
        emit_raw(&mut self.entry_points, op::ENTRY_POINT, &words);
    }

    pub fn execution_mode_local_size(&mut self, fn_id: Id, x: u32, y: u32, z: u32) {
        emit(
            &mut self.exec_modes,
            op::EXECUTION_MODE,
            &[fn_id.0, 17, x, y, z],
        );
    }

    /// OriginUpperLeft execution mode (mode 7) — required for fragment shaders.
    pub fn execution_mode_origin_upper_left(&mut self, fn_id: Id) {
        emit(&mut self.exec_modes, op::EXECUTION_MODE, &[fn_id.0, 7]);
    }

    // ── Decorations ───────────────────────────────────────────────────────────

    pub fn decorate(&mut self, target: Id, decoration: u32, extra: &[u32]) {
        let mut words = vec![target.0, decoration];
        words.extend_from_slice(extra);
        emit_raw(&mut self.annotations, op::DECORATE, &words);
    }

    pub fn member_decorate(&mut self, struct_id: Id, member: u32, decoration: u32, extra: &[u32]) {
        let mut words = vec![struct_id.0, member, decoration];
        words.extend_from_slice(extra);
        emit_raw(&mut self.annotations, op::MEMBER_DECORATE, &words);
    }

    // ── Types ─────────────────────────────────────────────────────────────────

    pub fn type_void(&mut self) -> Id {
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_VOID, &[id.0]);
        id
    }

    pub fn type_bool(&mut self) -> Id {
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_BOOL, &[id.0]);
        id
    }

    pub fn type_int(&mut self, width: u32, signed: bool) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_INT,
            &[id.0, width, if signed { 1 } else { 0 }],
        );
        id
    }

    pub fn type_float(&mut self, width: u32) -> Id {
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_FLOAT, &[id.0, width]);
        id
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_VECTOR,
            &[id.0, component.0, count],
        );
        id
    }

    pub fn type_runtime_array(&mut self, elem: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_RUNTIME_ARRAY, &[id.0, elem.0]);
        id
    }

    pub fn type_array(&mut self, elem: Id, len: Id) -> Id {
        let id = self.fresh_id();
        // The length constant must precede OpTypeArray, so fixed arrays live in
        // the combined types/constants/globals section after that constant.
        emit(
            &mut self.constants_globals,
            op::TYPE_ARRAY,
            &[id.0, elem.0, len.0],
        );
        id
    }

    pub fn type_pointer_global(&mut self, storage_class: u32, pointee: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.constants_globals,
            op::TYPE_POINTER,
            &[id.0, storage_class, pointee.0],
        );
        id
    }

    pub fn type_struct(&mut self, members: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![id.0];
        for m in members {
            words.push(m.0);
        }
        emit_raw(&mut self.types, op::TYPE_STRUCT, &words);
        id
    }

    pub fn type_pointer(&mut self, storage_class: u32, pointee: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_POINTER,
            &[id.0, storage_class, pointee.0],
        );
        id
    }

    pub fn type_function(&mut self, ret: Id, params: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![id.0, ret.0];
        for p in params {
            words.push(p.0);
        }
        emit_raw(&mut self.types, op::TYPE_FUNCTION, &words);
        id
    }

    pub fn type_matrix(&mut self, col_ty: Id, col_count: u32) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_MATRIX,
            &[id.0, col_ty.0, col_count],
        );
        id
    }

    pub fn type_vector3_uint(&mut self, uint_id: Id) -> Id {
        self.type_vector(uint_id, 3)
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    pub fn constant_u32(&mut self, ty: Id, value: u32) -> Id {
        self.constant_bits(ty, value)
    }

    pub fn constant_f32(&mut self, ty: Id, value: f32) -> Id {
        self.constant_bits(ty, value.to_bits())
    }

    /// Emit (or reuse) an `OpConstant` of `ty` with the given 32-bit pattern.
    /// Deduplicates on `(ty, bits)` so a `<type, value>` constant is defined once.
    fn constant_bits(&mut self, ty: Id, bits: u32) -> Id {
        if let Some(&id) = self.constant_cache.get(&(ty.0, bits)) {
            return id;
        }
        let id = self.fresh_id();
        emit(
            &mut self.constants_globals,
            op::CONSTANT,
            &[ty.0, id.0, bits],
        );
        self.constant_cache.insert((ty.0, bits), id);
        id
    }

    // ── Global variables ──────────────────────────────────────────────────────

    pub fn global_variable(&mut self, ptr_ty: Id, storage_class: u32) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.constants_globals,
            op::VARIABLE,
            &[ptr_ty.0, id.0, storage_class],
        );
        id
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    pub fn begin_function(&mut self, ret_ty: Id, fn_id: Id, fn_ty: Id) {
        // None function control = 0
        emit(
            &mut self.functions,
            op::FUNCTION,
            &[ret_ty.0, fn_id.0, 0, fn_ty.0],
        );
    }

    pub fn end_function(&mut self) {
        emit(&mut self.functions, op::FUNCTION_END, &[]);
    }

    pub fn label(&mut self) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::LABEL, &[id.0]);
        id
    }

    /// Emit `OpLabel` with a pre-allocated ID (needed when the label ID must be
    /// known before the block starts, e.g. for branch targets).
    pub fn label_with_id(&mut self, id: Id) {
        emit(&mut self.functions, op::LABEL, &[id.0]);
    }

    pub fn op_return(&mut self) {
        emit(&mut self.functions, op::RETURN, &[]);
    }

    pub fn op_control_barrier(&mut self, execution_scope: Id, memory_scope: Id, semantics: Id) {
        emit(
            &mut self.functions,
            op::CONTROL_BARRIER,
            &[execution_scope.0, memory_scope.0, semantics.0],
        );
    }

    pub fn op_atomic_fadd_ext(
        &mut self,
        ty: Id,
        ptr: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::ATOMIC_FADD_EXT,
            &[ty.0, id.0, ptr.0, scope.0, semantics.0, value.0],
        );
        id
    }

    pub fn op_branch(&mut self, target: Id) {
        emit(&mut self.functions, op::BRANCH, &[target.0]);
    }

    pub fn op_branch_conditional(&mut self, cond: Id, true_label: Id, false_label: Id) {
        emit(
            &mut self.functions,
            op::BRANCH_CONDITIONAL,
            &[cond.0, true_label.0, false_label.0],
        );
    }

    pub fn op_selection_merge(&mut self, merge_label: Id) {
        // SelectionControl None = 0
        emit(
            &mut self.functions,
            op::SELECTION_MERGE,
            &[merge_label.0, 0],
        );
    }

    pub fn op_loop_merge(&mut self, merge_label: Id, continue_label: Id) {
        // LoopControl None = 0
        emit(
            &mut self.functions,
            op::LOOP_MERGE,
            &[merge_label.0, continue_label.0, 0],
        );
    }

    pub fn op_variable(&mut self, ptr_ty: Id, storage_class: u32) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::VARIABLE,
            &[ptr_ty.0, id.0, storage_class],
        );
        id
    }

    pub fn op_load(&mut self, ty: Id, ptr: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::LOAD, &[ty.0, id.0, ptr.0]);
        id
    }

    pub fn op_store(&mut self, ptr: Id, val: Id) {
        emit(&mut self.functions, op::STORE, &[ptr.0, val.0]);
    }

    pub fn op_access_chain(&mut self, ptr_ty: Id, base: Id, indices: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ptr_ty.0, id.0, base.0];
        for i in indices {
            words.push(i.0);
        }
        emit_raw(&mut self.functions, op::ACCESS_CHAIN, &words);
        id
    }

    pub fn op_composite_extract(&mut self, ty: Id, composite: Id, indices: &[u32]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ty.0, id.0, composite.0];
        words.extend_from_slice(indices);
        emit_raw(&mut self.functions, op::COMPOSITE_EXTRACT, &words);
        id
    }

    pub fn op_composite_construct(&mut self, ty: Id, components: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ty.0, id.0];
        for c in components {
            words.push(c.0);
        }
        emit_raw(&mut self.functions, op::COMPOSITE_CONSTRUCT, &words);
        id
    }

    pub fn op_iadd(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::IADD, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_fadd(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FADD, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_isub(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::ISUB, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_fsub(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FSUB, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_imul(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::IMUL, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_fmul(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FMUL, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_fdiv(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FDIV, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_udiv(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::UDIV, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_fnegate(&mut self, ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FNEGATE, &[ty.0, id.0, val.0]);
        id
    }

    pub fn op_snegate(&mut self, ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::SNEGATE, &[ty.0, id.0, val.0]);
        id
    }

    pub fn op_vector_times_scalar(&mut self, vec_ty: Id, vec: Id, scalar: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::VECTOR_TIMES_SCALAR,
            &[vec_ty.0, id.0, vec.0, scalar.0],
        );
        id
    }

    pub fn op_ult(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::ULESS_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ule(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::ULESS_THAN_EQ,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ugt(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::UGREATER_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_uge(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::UGREATER_THAN_EQ,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_lt(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_LESS_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_le(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_LESS_THAN_EQ,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_gt(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_GREATER_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_ge(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_GREATER_THAN_EQ,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_convert_u_to_f(&mut self, f32_ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::CONVERT_U_TO_F,
            &[f32_ty.0, id.0, val.0],
        );
        id
    }

    pub fn op_convert_f_to_u(&mut self, u32_ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::CONVERT_F_TO_U,
            &[u32_ty.0, id.0, val.0],
        );
        id
    }

    pub fn op_matrix_times_vector(&mut self, result_ty: Id, mat: Id, vec: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::MATRIX_TIMES_VECTOR,
            &[result_ty.0, id.0, mat.0, vec.0],
        );
        id
    }

    /// OpDot — dot product of two float vectors; result is a scalar f32.
    pub fn op_dot(&mut self, f32_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::DOT, &[f32_ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_iequal(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IEQUAL,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_inot_equal(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::INOT_EQUAL,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_eq(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_EQUAL,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_ford_ne(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::FORD_NOT_EQUAL,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_logical_and(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::LOGICAL_AND,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_logical_or(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::LOGICAL_OR,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    /// OpExtInst — call an extended instruction set function (e.g. GLSL.std.450).
    pub fn op_ext_inst(&mut self, ty: Id, set: Id, opcode: u32, operands: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ty.0, id.0, set.0, opcode];
        for o in operands {
            words.push(o.0);
        }
        emit_raw(&mut self.functions, op::EXT_INST, &words);
        id
    }

    pub fn op_slt(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SLESS_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    // ── Serialize ─────────────────────────────────────────────────────────────

    /// Serialize the module to a SPIR-V word stream ready to embed as `&[u32]`.
    pub fn finish(self) -> Vec<u32> {
        let bound = self.next_id;
        let mut out = vec![
            0x07230203, // magic
            0x00010300, // version 1.3
            0x7A656E67, // generator: 'zeng' (ZenGPU)
            bound, 0, // schema
        ];
        out.extend_from_slice(&self.capabilities);
        out.extend_from_slice(&self.extensions);
        out.extend_from_slice(&self.imports);
        out.extend_from_slice(&self.memory_model);
        out.extend_from_slice(&self.entry_points);
        out.extend_from_slice(&self.exec_modes);
        out.extend_from_slice(&self.annotations);
        out.extend_from_slice(&self.types);
        out.extend_from_slice(&self.constants_globals);
        out.extend_from_slice(&self.functions);
        out
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn emit(section: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    let word_count = (1 + operands.len()) as u32;
    section.push((word_count << 16) | opcode);
    section.extend_from_slice(operands);
}

fn emit_raw(section: &mut Vec<u32>, opcode: u32, words: &[u32]) {
    let word_count = (1 + words.len()) as u32;
    section.push((word_count << 16) | opcode);
    section.extend_from_slice(words);
}

/// Encode a null-terminated UTF-8 string as SPIR-V words (little-endian,
/// zero-padded to a 4-byte boundary).
pub fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let total = bytes.len() + 1; // +1 for null terminator
    let words = total.div_ceil(4);
    let mut out = vec![0u32; words];
    for (i, &b) in bytes.iter().enumerate() {
        out[i / 4] |= (b as u32) << (8 * (i % 4));
    }
    out
}
