//! `ModuleReflection`: aggregates [`crate::ir::layout`]'s push-constant byte
//! layout and [`crate::ir::binding`]'s bindless-table assignment for a
//! compiled entry point into one queryable value, in declaration order.
//!
//! A pure Rust API over an already-built `Module`/`GraphicsModule` — nothing
//! computes this yet at proc-macro time (that wiring, so a `const SHADER:
//! ZslShader = zsl!(...)` carries reflection too, is a separate, later step).

use std::collections::HashMap;

use crate::ir::binding::{self, BindingInfo};
use crate::ir::layout::{self, LayoutInfo};
use crate::ir::{GraphicsModule, Module};

/// One param's full reflection: its push-constant layout, and — for buffer
/// params only — which bindless SSBO table it routes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamReflection {
    pub name: String,
    pub layout: LayoutInfo,
    /// `Some` for a buffer param (which bindless table it routes to);
    /// `None` for a scalar param (push-constant `layout` alone is enough).
    pub binding: Option<BindingInfo>,
}

/// Full reflection for one compiled entry point, params in declaration order
/// (buffer params first, then scalar params — see [`crate::ir::layout`]).
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

/// Reflect a compute [`Module`]'s entry point.
pub fn reflect_compute(module: &Module) -> ModuleReflection {
    merge(layout::compute_layout(module), binding::compute_bindings(module))
}

/// Reflect a graphics [`GraphicsModule`]'s entry point.
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
        assert!(reflection.params[0].binding.is_some()); // x: buffer
        assert!(reflection.params[1].binding.is_some()); // y: buffer
        assert!(reflection.params[2].binding.is_none()); // n: scalar
        assert!(reflection.params[3].binding.is_none()); // alpha: scalar
    }
}
