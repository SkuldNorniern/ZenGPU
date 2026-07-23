//! The seam between "parsed IR" and "what a backend is allowed to consume".
//!
//! `ValidatedModule` wraps an already-parsed [`crate::frontend::parser::Shader`]
//! behind private fields, so a backend can only ever receive a module that has
//! gone through [`validate`] — never an arbitrary freshly-parsed one. Today
//! [`validate`] performs no real checks; the parser (`crate::frontend::parser`)
//! already rejects structurally invalid source (unknown identifiers, read-only
//! buffer writes, type mismatches on bitwise operands, etc.) before a `Shader`
//! ever exists. This module exists so a later, much larger validation pass
//! (type checking against a unified type system, layout checks, capability
//! checks, uniformity analysis) has a stable seam to grow into without
//! changing `validate`'s signature or any of its callers again.
//!
//! No backend takes `&ValidatedModule` yet — that migration happens once
//! backend adapters exist to consume it.

use crate::frontend::parser::Shader;
use crate::frontend::diagnostic::Diagnostic;

/// A [`Shader`] that has passed [`validate`]. Fields are private: the only way
/// to obtain one is through `validate`, and the only way to get the `Shader`
/// back out is through [`ValidatedModule::into_inner`] or [`ValidatedModule::shader`].
pub struct ValidatedModule {
    shader: Shader,
}

impl ValidatedModule {
    /// Borrow the validated shader.
    pub fn shader(&self) -> &Shader {
        &self.shader
    }

    /// Consume the wrapper, returning the validated shader.
    pub fn into_inner(self) -> Shader {
        self.shader
    }
}

/// Validate a parsed [`Shader`], producing a [`ValidatedModule`].
///
/// Performs no checks today beyond what the parser itself already enforces —
/// this always succeeds. It exists as the seam a later, real validation pass
/// grows into (see the module docs above).
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
