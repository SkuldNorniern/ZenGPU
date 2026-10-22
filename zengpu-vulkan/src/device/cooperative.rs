//! Check cooperative matrix requirements before handing unsupported types to the driver.

use super::*;

#[derive(Clone, Copy, Debug)]
struct MatrixType {
    component: vk::ComponentTypeKHR,
    scope: u32,
    rows: u32,
    cols: u32,
    usage: u32,
}

impl MatrixType {
    fn matches(self, p: &vk::CooperativeMatrixPropertiesKHR<'_>) -> bool {
        self.scope == p.scope.as_raw() as u32
            && match self.usage {
                0 => (self.rows, self.cols, self.component) == (p.m_size, p.k_size, p.a_type),
                1 => (self.rows, self.cols, self.component) == (p.k_size, p.n_size, p.b_type),
                2 => {
                    (self.rows, self.cols) == (p.m_size, p.n_size)
                        && (self.component == p.c_type || self.component == p.result_type)
                }
                _ => false,
            }
    }
}

pub(super) fn validate(
    words: &[u32],
    supported: &[vk::CooperativeMatrixPropertiesKHR<'_>],
) -> Result<()> {
    let module = zengpu_spv::decode(words)
        .map_err(|e| GpuError::ShaderCompile(format!("cooperative matrix SPIR-V: {e}")))?;
    let mut components = HashMap::new();
    let mut constants = HashMap::new();
    let mut value_types = HashMap::new();
    for i in &module.instructions {
        if let (Some(id), Some(ty)) = (i.result_id, i.result_type) {
            value_types.insert(id, ty);
        }
        match (i.opcode, i.result_id, i.trailing.as_slice()) {
            (21, Some(id), [8, 1]) => {
                components.insert(id, vk::ComponentTypeKHR::SINT8);
            }
            (21, Some(id), [8, 0]) => {
                components.insert(id, vk::ComponentTypeKHR::UINT8);
            }
            (21, Some(id), [32, 1]) => {
                components.insert(id, vk::ComponentTypeKHR::SINT32);
            }
            (21, Some(id), [32, 0]) => {
                components.insert(id, vk::ComponentTypeKHR::UINT32);
            }
            (22, Some(id), [16]) => {
                components.insert(id, vk::ComponentTypeKHR::FLOAT16);
            }
            (22, Some(id), [32]) => {
                components.insert(id, vk::ComponentTypeKHR::FLOAT32);
            }
            (43 | 50, Some(id), [value]) => {
                constants.insert(id, *value);
            }
            _ => {}
        }
        // The lightweight SPIR-V decoder does not yet name these opcodes.
        if matches!(i.opcode, 57 | 83 | 245 | 4457 | 4459) {
            let count = (words[i.word_offset] >> 16) as usize;
            let body = &words[i.word_offset + 1..i.word_offset + count];
            if let [ty, id, ..] = body {
                value_types.insert(*id, *ty);
            }
        }
    }
    let invalid = || GpuError::ShaderCompile("unresolved cooperative matrix type".into());
    let mut matrices = HashMap::new();
    for i in &module.instructions {
        if i.opcode != 4456 {
            continue;
        }
        let count = (words[i.word_offset] >> 16) as usize;
        let body = &words[i.word_offset + 1..i.word_offset + count];
        let [id, component, scope, rows, cols, usage] = body else {
            return Err(invalid());
        };
        let ty = MatrixType {
            component: *components.get(component).ok_or_else(invalid)?,
            scope: *constants.get(scope).ok_or_else(invalid)?,
            rows: *constants.get(rows).ok_or_else(invalid)?,
            cols: *constants.get(cols).ok_or_else(invalid)?,
            usage: *constants.get(usage).ok_or_else(invalid)?,
        };
        if !supported.iter().any(|p| ty.matches(p)) {
            return Err(GpuError::Unsupported(format!(
                "cooperative matrix {:?} {}x{} use {} is not supported by this device",
                ty.component, ty.rows, ty.cols, ty.usage,
            )));
        }
        matrices.insert(*id, ty);
    }
    for i in &module.instructions {
        if i.opcode != 4459 {
            continue;
        }
        let count = (words[i.word_offset] >> 16) as usize;
        let body = &words[i.word_offset + 1..i.word_offset + count];
        let [result, _, a, b, c, rest @ ..] = body else {
            return Err(invalid());
        };
        let ty = |value| -> Result<MatrixType> {
            let id = value_types.get(value).ok_or_else(invalid)?;
            matrices.get(id).copied().ok_or_else(invalid)
        };
        let (a, b, c) = (ty(a)?, ty(b)?, ty(c)?);
        let result = *matrices.get(result).ok_or_else(invalid)?;
        let saturating = rest.first().copied().unwrap_or(0) & 0x10 != 0;
        if !supported.iter().any(|p| {
            a.matches(p)
                && b.matches(p)
                && c.matches(p)
                && result.matches(p)
                && c.component == p.c_type
                && result.component == p.result_type
                && saturating == (p.saturating_accumulation == vk::TRUE)
        }) {
            return Err(GpuError::Unsupported(
                "cooperative matrix multiply types do not match a supported device shape".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zengpu_spirv::{ZslShader, zsl};

    const MIXED: ZslShader = zsl!(
        @workgroup_size(32)
        kernel mixed(a: device buffer<f16>, b: device buffer<f16>, ia: device buffer<u32>, ib: device buffer<u32>, out: device mut buffer<f32>, oi: device mut buffer<i32>) {
            let fa: frag<f16, 16, 16, a> = frag_load_a(a, 0, 16)
            let fb: frag<f16, 16, 16, b> = frag_load_b(b, 0, 16)
            let acc: frag<f32, 16, 16, acc> = frag_zero_acc()
            frag_store(out, 0, 16, mma(fa, fb, acc))
            let qa: frag<i8, 16, 16, a> = frag_load_a(ia, 0, 16)
            let qb: frag<i8, 16, 16, b> = frag_load_b(ib, 0, 16)
            let qacc: frag<i32, 16, 16, acc> = frag_zero_acc()
            frag_store(oi, 0, 16, mma(qa, qb, qacc))
        }
    );

    fn properties(integer_k: u32) -> [vk::CooperativeMatrixPropertiesKHR<'static>; 2] {
        let float = vk::CooperativeMatrixPropertiesKHR {
            m_size: 16,
            n_size: 16,
            k_size: 16,
            a_type: vk::ComponentTypeKHR::FLOAT16,
            b_type: vk::ComponentTypeKHR::FLOAT16,
            c_type: vk::ComponentTypeKHR::FLOAT32,
            result_type: vk::ComponentTypeKHR::FLOAT32,
            scope: vk::ScopeKHR::SUBGROUP,
            ..Default::default()
        };
        let integer = vk::CooperativeMatrixPropertiesKHR {
            k_size: integer_k,
            a_type: vk::ComponentTypeKHR::SINT8,
            b_type: vk::ComponentTypeKHR::SINT8,
            c_type: vk::ComponentTypeKHR::SINT32,
            result_type: vk::ComponentTypeKHR::SINT32,
            ..float
        };
        [float, integer]
    }

    #[test]
    fn accepts_advertised_matrix_shapes() {
        validate(MIXED.spirv_words().unwrap(), &properties(16)).unwrap();
    }

    #[test]
    fn rejects_a_different_integer_k_before_driver_use() {
        assert!(
            matches!(validate(MIXED.spirv_words().unwrap(), &properties(32)),
                Err(GpuError::Unsupported(message)) if message.contains("SINT8")
            )
        );
    }

    #[test]
    fn rejects_a_saturating_only_multiply() {
        let mut supported = properties(16);
        supported[1].saturating_accumulation = vk::TRUE;
        assert!(matches!(validate(MIXED.spirv_words().unwrap(), &supported),
            Err(GpuError::Unsupported(message)) if message.contains("multiply types")
        ));
    }

    #[test]
    fn truncated_matrix_declarations_return_an_error() {
        let mut words = MIXED.spirv_words().unwrap().to_vec();
        let module = zengpu_spv::decode(&words).unwrap();
        let offset = module
            .instructions
            .iter()
            .find(|i| i.opcode == 4456)
            .unwrap()
            .word_offset;
        words.truncate(offset + 2);
        assert!(matches!(
            validate(&words, &properties(16)),
            Err(GpuError::ShaderCompile(_))
        ));
    }
}
