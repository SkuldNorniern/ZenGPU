//! `ModuleReflection`: push-constant layout and bindless table assignment of a compiled entry
//! point, in declaration order.
//!
//! Computed from an already-built module, not at proc-macro time.

use std::collections::HashMap;

use crate::ir::binding::{self, BindingInfo};
use crate::ir::layout::{self, LayoutInfo};
use crate::ir::{GraphicsModule, Module};

/// push-constant layout, plus the bindless SSBO table for buffer params.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamReflection {
    pub name: String,
    pub layout: LayoutInfo,
    /// `Some` for buffer params (their table), `None` for scalars.
    pub binding: Option<BindingInfo>,
}

/// params in declaration order, buffers first, then scalars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleReflection {
    pub params: Vec<ParamReflection>,
}

fn merge(layouts: Vec<(String, LayoutInfo)>, bindings: Vec<(String, BindingInfo)>) -> ModuleReflection {
    let mut bindings: HashMap<String, BindingInfo> = bindings.into_iter().collect();
    let params = layouts
        .into_iter()
        .map(|(name, layout)| {
            let binding = bindings.remove(&name);
            ParamReflection {
                name,
                layout,
                binding,
            }
        })
        .collect();
    ModuleReflection { params }
}

pub fn reflect_compute(module: &Module) -> ModuleReflection {
    merge(layout::compute_layout(module), binding::compute_bindings(module))
}

pub fn reflect_graphics(module: &GraphicsModule) -> ModuleReflection {
    merge(layout::graphics_layout(module), binding::graphics_bindings(module))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::{Shader, parse_zsl};

    fn compile(src: &str) -> Module {
        match parse_zsl(src).expect("parse") {
            Shader::Compute(m) => m,
            Shader::Graphics(_) => panic!("expected compute"),
        }
    }

    #[test]
    fn typed_buffers_reflection_has_layout_and_binding_for_every_buffer() {
        let src = r#"
            @workgroup_size(64)
            kernel typed(src_u: device buffer<u32>, src_i: device buffer<i32>, out_u: device mut buffer<u32>, out_i: device mut buffer<i32>, id: global_id) {
                out_u[id.x] = src_u[id.x]
                out_i[id.x] = src_i[id.x]
            }
        "#;
        let module = compile(src);
        let reflection = reflect_compute(&module);
        assert_eq!(reflection.params.len(), 4);
        for (i, p) in reflection.params.iter().enumerate() {
            assert_eq!(p.layout.offset, (i * 4) as u32);
            let binding = p.binding.expect("buffer param must have a binding");
            let expected = if p.name.ends_with('u') { 2 } else { 3 };
            assert_eq!(binding.binding, expected);
        }
    }

    #[test]
    fn scalar_params_have_layout_but_no_binding() {
        let src = r#"
            push P { n: u32, alpha: f32 }
            @workgroup_size(64)
            kernel saxpy(push: P, x: device buffer<f32>, y: device mut buffer<f32>, id: global_id) {
                let i = id.x
                if i < push.n {
                    y[i] = push.alpha * x[i] + y[i]
                }
            }
        "#;
        let module = compile(src);
        let reflection = reflect_compute(&module);
        let names: Vec<&str> = reflection.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["x", "y", "n", "alpha"]);
        assert!(reflection.params[0].binding.is_some());
        assert!(reflection.params[1].binding.is_some());
        assert!(reflection.params[2].binding.is_none());
        assert!(reflection.params[3].binding.is_none());
    }
}
