//! SPIR-V binary builder for the ZSL subset: SSBOs, push constants, scalars, vectors,
//! matrices, arithmetic/logic.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub u32);

#[allow(dead_code)]
pub(crate) mod op {
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
    pub const ATOMIC_I_ADD: u32 = 234;
    pub const ATOMIC_AND: u32 = 240;
    pub const ATOMIC_OR: u32 = 241;
    pub const IADD: u32 = 128;
    pub const FADD: u32 = 129;
    pub const ISUB: u32 = 130;
    pub const FSUB: u32 = 131;
    pub const IMUL: u32 = 132;
    pub const FMUL: u32 = 133;
    pub const UDIV: u32 = 134;
    pub const SDIV: u32 = 135;
    pub const FDIV: u32 = 136;
    pub const UMOD: u32 = 137;
    pub const SREM: u32 = 138;
    pub const FREM: u32 = 140;
    pub const ULESS_THAN: u32 = 176;
    pub const ULESS_THAN_EQ: u32 = 178;
    pub const UGREATER_THAN: u32 = 172;
    pub const UGREATER_THAN_EQ: u32 = 174;
    pub const SGREATER_THAN_EQ: u32 = 175;
    pub const SLESS_THAN: u32 = 177;
    pub const SLESS_THAN_EQ: u32 = 179;
    pub const SGREATER_THAN: u32 = 173;
    pub const FORD_LESS_THAN: u32 = 184;
    pub const FORD_GREATER_THAN: u32 = 186;
    pub const FORD_LESS_THAN_EQ: u32 = 188;
    pub const FORD_GREATER_THAN_EQ: u32 = 190;
    pub const FLESS_THAN: u32 = 185;
    pub const IS_NAN: u32 = 156;
    pub const IS_INF: u32 = 157;
    pub const VECTOR_SHUFFLE: u32 = 79;
    pub const COMPOSITE_CONSTRUCT: u32 = 80;
    pub const TYPE_IMAGE: u32 = 25;
    pub const TYPE_SAMPLED_IMAGE: u32 = 27;
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u32 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u32 = 88;
    pub const IMAGE_FETCH: u32 = 95;
    pub const IMAGE_READ: u32 = 98;
    pub const IMAGE_WRITE: u32 = 99;
    pub const IMAGE: u32 = 100;
    pub const IMAGE_QUERY_SIZE_LOD: u32 = 103;
    pub const IMAGE_QUERY_SIZE: u32 = 104;
    pub const SELECT: u32 = 169;
    pub const KILL: u32 = 252;
    pub const COMPOSITE_EXTRACT: u32 = 81;
    pub const CONSTANT_NULL: u32 = 46;
    pub const TYPE_COOPERATIVE_MATRIX_KHR: u32 = 4456;
    pub const COOPERATIVE_MATRIX_LOAD_KHR: u32 = 4457;
    pub const COOPERATIVE_MATRIX_STORE_KHR: u32 = 4458;
    pub const COOPERATIVE_MATRIX_MUL_ADD_KHR: u32 = 4459;
    pub const CONVERT_F_TO_U: u32 = 109;
    pub const CONVERT_F_TO_S: u32 = 110;
    pub const CONVERT_S_TO_F: u32 = 111;
    pub const CONVERT_U_TO_F: u32 = 112;
    pub const CONVERT_F_TO_F: u32 = 115;
    pub const U_CONVERT: u32 = 113;
    pub const S_CONVERT: u32 = 114;
    pub const BITCAST: u32 = 124;
    pub const SNEGATE: u32 = 126;
    pub const FNEGATE: u32 = 127;
    pub const VECTOR_TIMES_SCALAR: u32 = 142;
    pub const TYPE_MATRIX: u32 = 24;
    pub const MATRIX_TIMES_VECTOR: u32 = 145;
    pub const DOT: u32 = 148;
    pub const LOGICAL_OR: u32 = 166;
    pub const LOGICAL_AND: u32 = 167;
    pub const LOGICAL_NOT: u32 = 168;
    pub const IEQUAL: u32 = 170;
    pub const INOT_EQUAL: u32 = 171;
    pub const FORD_EQUAL: u32 = 180;
    pub const FORD_NOT_EQUAL: u32 = 182;
    pub const SHIFT_RIGHT_LOGICAL: u32 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u32 = 195;
    pub const SHIFT_LEFT_LOGICAL: u32 = 196;
    pub const BITWISE_OR: u32 = 197;
    pub const BITWISE_XOR: u32 = 198;
    pub const BITWISE_AND: u32 = 199;
    pub const BITFIELD_SEXTRACT: u32 = 202;
    pub const BITFIELD_UEXTRACT: u32 = 203;
    pub const EXT_INST: u32 = 12;
    pub const GROUP_NON_UNIFORM_ALL: u32 = 334;
    pub const GROUP_NON_UNIFORM_ANY: u32 = 335;
    pub const GROUP_NON_UNIFORM_BROADCAST: u32 = 337;
    pub const GROUP_NON_UNIFORM_BALLOT: u32 = 339;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 345;
    pub const GROUP_NON_UNIFORM_SHUFFLE_XOR: u32 = 346;
    pub const GROUP_NON_UNIFORM_SHUFFLE_UP: u32 = 347;
    pub const GROUP_NON_UNIFORM_SHUFFLE_DOWN: u32 = 348;
    pub const GROUP_NON_UNIFORM_IADD: u32 = 349;
    pub const GROUP_NON_UNIFORM_FADD: u32 = 350;
    pub const GROUP_NON_UNIFORM_SMIN: u32 = 353;
    pub const GROUP_NON_UNIFORM_UMIN: u32 = 354;
    pub const GROUP_NON_UNIFORM_FMIN: u32 = 355;
    pub const GROUP_NON_UNIFORM_SMAX: u32 = 356;
    pub const GROUP_NON_UNIFORM_UMAX: u32 = 357;
    pub const GROUP_NON_UNIFORM_FMAX: u32 = 358;
    pub const GROUP_NON_UNIFORM_BITWISE_AND: u32 = 359;
    pub const GROUP_NON_UNIFORM_BITWISE_OR: u32 = 360;
}

#[allow(dead_code)]
pub mod deco {
    pub const BLOCK: u32 = 2;
    pub const ALIASED: u32 = 20;
    pub const BUFFER_BLOCK: u32 = 3;
    pub const COL_MAJOR: u32 = 5;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const BUILT_IN: u32 = 11;
    pub const NON_WRITABLE: u32 = 24;
    pub const LOCATION: u32 = 30;
    pub const INDEX: u32 = 32;
    pub const FLAT: u32 = 14;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

#[allow(dead_code)]
pub mod sc {
    pub const INPUT: u32 = 1;
    pub const OUTPUT: u32 = 3;
    pub const UNIFORM: u32 = 2;
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const STORAGE_BUFFER: u32 = 12;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const FUNCTION: u32 = 7;
    pub const WORKGROUP: u32 = 4;
}

#[allow(dead_code)]
pub(crate) mod cap {
    pub const SHADER: u32 = 1;
    pub const FLOAT16: u32 = 9;
    pub const INT8: u32 = 39;
    pub const STORAGE_BUFFER_16BIT_ACCESS: u32 = 4433;
    pub const STORAGE_BUFFER_8BIT_ACCESS: u32 = 4448;
    pub const COOPERATIVE_MATRIX_KHR: u32 = 6022;
    pub const VULKAN_MEMORY_MODEL: u32 = 5345;
    pub const VULKAN_MEMORY_MODEL_DEVICE_SCOPE: u32 = 5346;
    pub const GROUP_NON_UNIFORM: u32 = 61;
    pub const GROUP_NON_UNIFORM_VOTE: u32 = 62;
    pub const GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
    pub const GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    pub const GROUP_NON_UNIFORM_SHUFFLE_RELATIVE: u32 = 66;
    pub const IMAGE_QUERY: u32 = 50;
    pub const STORAGE_IMAGE_READ_WITHOUT_FORMAT: u32 = 55;
    pub const STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
    pub const RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;
    pub const ATOMIC_FLOAT32_ADD_EXT: u32 = 6033;
}

mod exec_model {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
}

pub mod builtin {
    pub const POSITION: u32 = 0;
    pub const FRAG_COORD: u32 = 15;
    pub const VERTEX_INDEX: u32 = 42;
    pub const INSTANCE_INDEX: u32 = 43;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const WORKGROUP_ID: u32 = 26;
    pub const SUBGROUP_SIZE: u32 = 36;
    pub const SUBGROUP_ID: u32 = 40;
    pub const SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
}

#[allow(dead_code)]
pub struct SpvBuilder {
    next_id: u32,
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    constants_globals: Vec<u32>,
    functions: Vec<u32>,
    constant_cache: HashMap<(u32, u32), Id>,
    type_constant_cache: HashMap<(u32, u32), Id>,
    scalar_type_cache: HashMap<(u32, u32, bool), Id>,
    capabilities_seen: Vec<u32>,
    /// The module has device-scope atomics.
    atomics: bool,
    /// Module-scope variables, for SPIR-V 1.4 entry point interfaces.
    globals: Vec<Id>,
    /// `(component, count)` of each vector type: SPIR-V forbids declaring one twice.
    vector_type_cache: HashMap<(u32, u32), Id>,
    /// An extension that needs SPIR-V 1.4 was enabled.
    needs_1_4: bool,
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
            scalar_type_cache: HashMap::new(),
            capabilities_seen: Vec::new(),
            constant_cache: HashMap::new(),
            type_constant_cache: HashMap::new(),
            atomics: false,
            globals: Vec::new(),
            vector_type_cache: HashMap::new(),
            needs_1_4: false,
        }
    }

    pub fn fresh_id(&mut self) -> Id {
        let id = Id(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn capability_shader(&mut self) {
        emit(&mut self.capabilities, op::CAPABILITY, &[cap::SHADER]);
    }

    pub fn capability_runtime_descriptor_array(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::RUNTIME_DESCRIPTOR_ARRAY],
        );
        let name = encode_string("SPV_EXT_descriptor_indexing");
        emit_raw(&mut self.extensions, op::EXTENSION, &name);
    }

    pub fn capability_image_query(&mut self) {
        emit(&mut self.capabilities, op::CAPABILITY, &[cap::IMAGE_QUERY]);
    }

    /// Storage images use `Unknown` format: one kernel serves every float format.
    pub fn capability_storage_image_without_format(&mut self) {
        for c in [
            cap::STORAGE_IMAGE_READ_WITHOUT_FORMAT,
            cap::STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
        ] {
            emit(&mut self.capabilities, op::CAPABILITY, &[c]);
        }
    }

    pub fn enable_atomic_float32_add_ext(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::ATOMIC_FLOAT32_ADD_EXT],
        );
        let name = encode_string("SPV_EXT_shader_atomic_float_add");
        emit_raw(&mut self.extensions, op::EXTENSION, &name);
    }

    /// Subgroup builtins are core in SPIR-V 1.3: no extension needed.
    pub fn capability_group_non_uniform(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM],
        );
    }

    pub fn capability_group_non_uniform_vote(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM_VOTE],
        );
    }

    pub fn capability_group_non_uniform_arithmetic(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM_ARITHMETIC],
        );
    }

    pub fn capability_group_non_uniform_ballot(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM_BALLOT],
        );
    }

    pub fn capability_group_non_uniform_shuffle(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM_SHUFFLE],
        );
    }

    pub fn capability_group_non_uniform_shuffle_relative(&mut self) {
        emit(
            &mut self.capabilities,
            op::CAPABILITY,
            &[cap::GROUP_NON_UNIFORM_SHUFFLE_RELATIVE],
        );
    }

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

    /// OriginUpperLeft (mode 7), required for fragment shaders.
    pub fn execution_mode_origin_upper_left(&mut self, fn_id: Id) {
        emit(&mut self.exec_modes, op::EXECUTION_MODE, &[fn_id.0, 7]);
    }

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
        if let Some(&id) = self.scalar_type_cache.get(&(op::TYPE_INT, width, signed)) {
            return id;
        }
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_INT,
            &[id.0, width, if signed { 1 } else { 0 }],
        );
        self.scalar_type_cache
            .insert((op::TYPE_INT, width, signed), id);
        id
    }

    pub fn type_float(&mut self, width: u32) -> Id {
        if let Some(&id) = self.scalar_type_cache.get(&(op::TYPE_FLOAT, width, false)) {
            return id;
        }
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_FLOAT, &[id.0, width]);
        self.scalar_type_cache
            .insert((op::TYPE_FLOAT, width, false), id);
        id
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        if let Some(&id) = self.vector_type_cache.get(&(component.0, count)) {
            return id;
        }
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_VECTOR,
            &[id.0, component.0, count],
        );
        self.vector_type_cache.insert((component.0, count), id);
        id
    }

    /// `OpTypeCooperativeMatrixKHR`: a subgroup-scope matrix of `component`. `scope`, `rows`,
    /// `cols` and `usage` are u32 constants.
    pub fn type_cooperative_matrix(
        &mut self,
        component: Id,
        scope: Id,
        rows: Id,
        cols: Id,
        usage: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_COOPERATIVE_MATRIX_KHR,
            &[id.0, component.0, scope.0, rows.0, cols.0, usage.0],
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

    /// Struct over a [`Self::type_array`] array: emitted in the globals section, after its member.
    pub fn type_struct_global(&mut self, members: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![id.0];
        for m in members {
            words.push(m.0);
        }
        emit_raw(&mut self.constants_globals, op::TYPE_STRUCT, &words);
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

    pub fn constant_u32(&mut self, ty: Id, value: u32) -> Id {
        self.constant_bits(ty, value)
    }

    pub fn constant_f32(&mut self, ty: Id, value: f32) -> Id {
        self.constant_bits(ty, value.to_bits())
    }

    /// Emits or reuses an `OpConstant`, deduplicated on `(ty, bits)`.
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

    pub fn global_variable(&mut self, ptr_ty: Id, storage_class: u32) -> Id {
        let id = self.fresh_id();
        self.globals.push(id);
        emit(
            &mut self.constants_globals,
            op::VARIABLE,
            &[ptr_ty.0, id.0, storage_class],
        );
        id
    }

    pub fn begin_function(&mut self, ret_ty: Id, fn_id: Id, fn_ty: Id) {
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

    /// `OpLabel` with a pre-allocated ID, for branch targets known before the block starts.
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
        self.atomics = true;
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::ATOMIC_FADD_EXT,
            &[ty.0, id.0, ptr.0, scope.0, semantics.0, value.0],
        );
        id
    }

    pub fn op_atomic_int(
        &mut self,
        opcode: u32,
        ty: Id,
        ptr: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Id {
        self.atomics = true;
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            opcode,
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
        emit(
            &mut self.functions,
            op::SELECTION_MERGE,
            &[merge_label.0, 0],
        );
    }

    pub fn op_loop_merge(&mut self, merge_label: Id, continue_label: Id) {
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

    pub fn type_constant_u32(&mut self, ty: Id, value: u32) -> Id {
        if let Some(&id) = self.type_constant_cache.get(&(ty.0, value)) {
            return id;
        }
        let id = self.fresh_id();
        emit(&mut self.types, op::CONSTANT, &[ty.0, id.0, value]);
        self.type_constant_cache.insert((ty.0, value), id);
        id
    }

    /// `OpConstantNull` of `ty` (also valid for cooperative matrices).
    pub fn constant_null(&mut self, ty: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.constants_globals,
            op::CONSTANT_NULL,
            &[ty.0, id.0],
        );
        id
    }

    /// `OpCooperativeMatrixLoadKHR`: `layout` and `stride` are u32 values.
    pub fn op_cooperative_matrix_load(&mut self, ty: Id, ptr: Id, layout: Id, stride: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::COOPERATIVE_MATRIX_LOAD_KHR,
            &[ty.0, id.0, ptr.0, layout.0, stride.0],
        );
        id
    }

    pub fn op_cooperative_matrix_store(&mut self, ptr: Id, object: Id, layout: Id, stride: Id) {
        emit(
            &mut self.functions,
            op::COOPERATIVE_MATRIX_STORE_KHR,
            &[ptr.0, object.0, layout.0, stride.0],
        );
    }

    /// `OpCooperativeMatrixMulAddKHR` with its `CooperativeMatrixOperands` mask (0 for float,
    /// signedness bits for integer matrices).
    pub fn op_cooperative_matrix_mul_add(&mut self, ty: Id, a: Id, b: Id, c: Id, flags: u32) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::COOPERATIVE_MATRIX_MUL_ADD_KHR,
            &[ty.0, id.0, a.0, b.0, c.0, flags],
        );
        id
    }

    pub fn capability(&mut self, capability: u32) {
        if self.capabilities_seen.contains(&capability) {
            return;
        }
        self.capabilities_seen.push(capability);
        emit(&mut self.capabilities, op::CAPABILITY, &[capability]);
    }

    pub fn extension(&mut self, name: &str) {
        if name == "SPV_KHR_workgroup_memory_explicit_layout" {
            self.needs_1_4 = true;
        }
        let words = encode_string(name);
        emit_raw(&mut self.extensions, op::EXTENSION, &words);
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

    pub fn op_vector_shuffle(&mut self, ty: Id, a: Id, lanes: &[u32]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ty.0, id.0, a.0, a.0];
        words.extend_from_slice(lanes);
        emit_raw(&mut self.functions, op::VECTOR_SHUFFLE, &words);
        id
    }

    pub fn op_select(&mut self, ty: Id, cond: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SELECT,
            &[ty.0, id.0, cond.0, a.0, b.0],
        );
        id
    }

    /// `OpKill`: ends the invocation. Terminates its block.
    pub fn op_kill(&mut self) {
        emit(&mut self.functions, op::KILL, &[]);
    }

    pub fn type_image_2d(&mut self, sampled: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_IMAGE,
            &[id.0, sampled.0, 1, 0, 0, 0, 1, 0],
        );
        id
    }

    pub fn type_storage_image_2d(&mut self, sampled: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.types,
            op::TYPE_IMAGE,
            &[id.0, sampled.0, 1, 0, 0, 0, 2, 0],
        );
        id
    }

    pub fn type_sampled_image(&mut self, image: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.types, op::TYPE_SAMPLED_IMAGE, &[id.0, image.0]);
        id
    }

    pub fn op_image_sample_implicit_lod(&mut self, ty: Id, image: Id, coord: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_SAMPLE_IMPLICIT_LOD,
            &[ty.0, id.0, image.0, coord.0],
        );
        id
    }

    pub fn op_image_sample_explicit_lod(&mut self, ty: Id, image: Id, coord: Id, lod: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_SAMPLE_EXPLICIT_LOD,
            &[ty.0, id.0, image.0, coord.0, 0x2, lod.0],
        );
        id
    }

    pub fn op_image(&mut self, ty: Id, sampled_image: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE,
            &[ty.0, id.0, sampled_image.0],
        );
        id
    }

    pub fn op_image_fetch(&mut self, ty: Id, image: Id, coord: Id, lod: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_FETCH,
            &[ty.0, id.0, image.0, coord.0, 0x2, lod.0],
        );
        id
    }

    pub fn op_image_read(&mut self, ty: Id, image: Id, coord: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_READ,
            &[ty.0, id.0, image.0, coord.0],
        );
        id
    }

    pub fn op_image_write(&mut self, image: Id, coord: Id, texel: Id) {
        emit(
            &mut self.functions,
            op::IMAGE_WRITE,
            &[image.0, coord.0, texel.0],
        );
    }

    pub fn op_image_query_size_lod(&mut self, ty: Id, image: Id, lod: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_QUERY_SIZE_LOD,
            &[ty.0, id.0, image.0, lod.0],
        );
        id
    }

    pub fn op_image_query_size(&mut self, ty: Id, image: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::IMAGE_QUERY_SIZE,
            &[ty.0, id.0, image.0],
        );
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

    pub fn op_umod(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::UMOD, &[ty.0, id.0, a.0, b.0]);
        id
    }

    /// Sign follows the dividend (C and Rust `%`), unlike `OpSMod`.
    pub fn op_srem(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::SREM, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_frem(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::FREM, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_sdiv(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::SDIV, &[ty.0, id.0, a.0, b.0]);
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

    pub fn op_sle(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SLESS_THAN_EQ,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_sgt(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SGREATER_THAN,
            &[bool_ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_sge(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SGREATER_THAN_EQ,
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

    pub fn op_is_nan(&mut self, bool_ty: Id, value: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::IS_NAN, &[bool_ty.0, id.0, value.0]);
        id
    }

    pub fn op_is_inf(&mut self, bool_ty: Id, value: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::IS_INF, &[bool_ty.0, id.0, value.0]);
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

    pub fn op_convert_s_to_f(&mut self, f32_ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::CONVERT_S_TO_F,
            &[f32_ty.0, id.0, val.0],
        );
        id
    }

    pub fn op_uconvert(&mut self, ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::U_CONVERT, &[ty.0, id.0, val.0]);
        id
    }

    pub fn op_sconvert(&mut self, ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::S_CONVERT, &[ty.0, id.0, val.0]);
        id
    }

    pub fn op_convert_f_to_f(&mut self, ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::CONVERT_F_TO_F,
            &[ty.0, id.0, val.0],
        );
        id
    }

    pub fn op_convert_f_to_s(&mut self, i32_ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::CONVERT_F_TO_S,
            &[i32_ty.0, id.0, val.0],
        );
        id
    }

    pub fn op_bitcast(&mut self, result_ty: Id, val: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::BITCAST,
            &[result_ty.0, id.0, val.0],
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

    pub fn op_logical_not(&mut self, bool_ty: Id, value: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::LOGICAL_NOT,
            &[bool_ty.0, id.0, value.0],
        );
        id
    }

    pub fn op_bitwise_and(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::BITWISE_AND,
            &[ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_bitwise_or(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(&mut self.functions, op::BITWISE_OR, &[ty.0, id.0, a.0, b.0]);
        id
    }

    pub fn op_bitwise_xor(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::BITWISE_XOR,
            &[ty.0, id.0, a.0, b.0],
        );
        id
    }

    /// Shift amount `b` must have the same integer type as `a`.
    pub fn op_shift_left_logical(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SHIFT_LEFT_LOGICAL,
            &[ty.0, id.0, a.0, b.0],
        );
        id
    }

    /// Zero-fills, so `u32` only.
    pub fn op_shift_right_logical(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SHIFT_RIGHT_LOGICAL,
            &[ty.0, id.0, a.0, b.0],
        );
        id
    }

    /// Sign-fills, for `i32`.
    pub fn op_shift_right_arithmetic(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::SHIFT_RIGHT_ARITHMETIC,
            &[ty.0, id.0, a.0, b.0],
        );
        id
    }

    pub fn op_bitfield_sextract(&mut self, ty: Id, base: Id, offset: Id, count: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::BITFIELD_SEXTRACT,
            &[ty.0, id.0, base.0, offset.0, count.0],
        );
        id
    }

    pub fn op_bitfield_uextract(&mut self, ty: Id, base: Id, offset: Id, count: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::BITFIELD_UEXTRACT,
            &[ty.0, id.0, base.0, offset.0, count.0],
        );
        id
    }

    pub fn op_ext_inst(&mut self, ty: Id, set: Id, opcode: u32, operands: &[Id]) -> Id {
        let id = self.fresh_id();
        let mut words = vec![ty.0, id.0, set.0, opcode];
        for o in operands {
            words.push(o.0);
        }
        emit_raw(&mut self.functions, op::EXT_INST, &words);
        id
    }

    pub fn op_group_non_uniform_any(&mut self, bool_ty: Id, scope: Id, predicate: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_ANY,
            &[bool_ty.0, id.0, scope.0, predicate.0],
        );
        id
    }

    pub fn op_group_non_uniform_all(&mut self, bool_ty: Id, scope: Id, predicate: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_ALL,
            &[bool_ty.0, id.0, scope.0, predicate.0],
        );
        id
    }

    pub fn op_group_non_uniform_ballot(&mut self, uvec4_ty: Id, scope: Id, predicate: Id) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_BALLOT,
            &[uvec4_ty.0, id.0, scope.0, predicate.0],
        );
        id
    }

    pub fn op_group_non_uniform_broadcast(
        &mut self,
        value_ty: Id,
        scope: Id,
        value: Id,
        other: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_BROADCAST,
            &[value_ty.0, id.0, scope.0, value.0, other.0],
        );
        id
    }

    pub fn op_group_non_uniform_shuffle(
        &mut self,
        value_ty: Id,
        scope: Id,
        value: Id,
        other: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_SHUFFLE,
            &[value_ty.0, id.0, scope.0, value.0, other.0],
        );
        id
    }

    pub fn op_group_non_uniform_shuffle_xor(
        &mut self,
        value_ty: Id,
        scope: Id,
        value: Id,
        other: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_SHUFFLE_XOR,
            &[value_ty.0, id.0, scope.0, value.0, other.0],
        );
        id
    }

    pub fn op_group_non_uniform_shuffle_up(
        &mut self,
        value_ty: Id,
        scope: Id,
        value: Id,
        other: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_SHUFFLE_UP,
            &[value_ty.0, id.0, scope.0, value.0, other.0],
        );
        id
    }

    pub fn op_group_non_uniform_shuffle_down(
        &mut self,
        value_ty: Id,
        scope: Id,
        value: Id,
        other: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            op::GROUP_NON_UNIFORM_SHUFFLE_DOWN,
            &[value_ty.0, id.0, scope.0, value.0, other.0],
        );
        id
    }

    /// All ten share the operand shape `[ResultType, ResultId, Scope, GroupOperation, Value]`.
    pub fn op_group_non_uniform_arith(
        &mut self,
        opcode: u32,
        value_ty: Id,
        scope: Id,
        group_operation: u32,
        value: Id,
    ) -> Id {
        let id = self.fresh_id();
        emit(
            &mut self.functions,
            opcode,
            &[value_ty.0, id.0, scope.0, group_operation, value.0],
        );
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

    pub fn finish(mut self) -> Vec<u32> {
        // Cooperative matrices need the Vulkan memory model (an extension before SPIR-V 1.5).
        if self
            .capabilities_seen
            .contains(&cap::COOPERATIVE_MATRIX_KHR)
        {
            self.capability(cap::VULKAN_MEMORY_MODEL);
            if self.atomics {
                self.capability(cap::VULKAN_MEMORY_MODEL_DEVICE_SCOPE);
            }
            self.extension("SPV_KHR_vulkan_memory_model");
            self.memory_model.clear();
            emit(&mut self.memory_model, op::MEMORY_MODEL, &[0, 3]);
        }
        // SPIR-V 1.4 entry points list every module-scope variable they use, not only inputs and
        // outputs. Listing them all is valid: every global belongs to the one entry point.
        let version = if self.needs_1_4 {
            self.list_all_globals_in_entry_points();
            0x00010400
        } else {
            0x00010300
        };
        let bound = self.next_id;
        let mut out = vec![0x07230203, version, 0x7A656E67, bound, 0];
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

impl SpvBuilder {
    /// Appends the module-scope variables missing from each `OpEntryPoint` interface.
    fn list_all_globals_in_entry_points(&mut self) {
        let mut out = Vec::with_capacity(self.entry_points.len() + self.globals.len());
        let mut at = 0;
        while at < self.entry_points.len() {
            let head = self.entry_points[at];
            let wc = (head >> 16) as usize;
            let mut words = self.entry_points[at..at + wc].to_vec();
            if head & 0xffff == op::ENTRY_POINT {
                // Operands: model, function, name (zero-terminated string), then interface ids.
                let name_words = words[3..]
                    .iter()
                    .position(|w| w >> 24 == 0)
                    .map_or(0, |p| p + 1);
                let listed: Vec<u32> = words[3 + name_words..].to_vec();
                for g in &self.globals {
                    if !listed.contains(&g.0) {
                        words.push(g.0);
                    }
                }
                words[0] = ((words.len() as u32) << 16) | op::ENTRY_POINT;
            }
            out.extend_from_slice(&words);
            at += wc;
        }
        self.entry_points = out;
    }
}

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

pub fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let total = bytes.len() + 1;
    let words = total.div_ceil(4);
    let mut out = vec![0u32; words];
    for (i, &b) in bytes.iter().enumerate() {
        out[i / 4] |= (b as u32) << (8 * (i % 4));
    }
    out
}
