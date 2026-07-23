//! Seam between parsed IR and backend input.
//!
//! `ValidatedModule` has private fields: backends only get a module that passed [`validate`]. Today
//! `validate` checks nothing; the parser rejects bad source first. Real checks go here later.
//!
//! No backend takes `&ValidatedModule` yet.

use crate::frontend::parser::Shader;
use crate::frontend::diagnostic::Diagnostic;

/// A [`Shader`] that passed [`validate`]. Its fields are private, so only `validate` builds one.
pub struct ValidatedModule {
    shader: Shader,
}

impl ValidatedModule {
    pub fn shader(&self) -> &Shader {
        &self.shader
    }

    pub fn into_inner(self) -> Shader {
        self.shader
    }
}

/// Validate a parsed [`Shader`]. Always succeeds: the parser already checks everything it does.
pub fn validate(shader: Shader) -> Result<ValidatedModule, Diagnostic> {
    Ok(ValidatedModule { shader })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parser::parse_compute;

    #[test]
    fn validate_wraps_and_unwraps_a_parsed_module() {
        let module = parse_compute(
            r#"
            @workgroup_size(64)
            kernel noop(x: device mut buffer<f32>, id: global_id) {
                let i = id.x
                x[i] = x[i]
            }
        "#,
        )
        .expect("parse");
        let validated = validate(Shader::Compute(module)).expect("validate");
        match validated.shader() {
            Shader::Compute(_) => {}
            Shader::Graphics(_) => panic!("expected compute"),
        }
        match validated.into_inner() {
            Shader::Compute(_) => {}
            Shader::Graphics(_) => panic!("expected compute"),
        }
    }
}
