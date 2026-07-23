//! Structured backend-lowering diagnostics.
//!
//! Distinct from `parser::ParseError` (front-end syntax errors): `Diagnostic` is
//! what a lowering backend (spirv/hip/cuda/msl/dx) returns when it cannot
//! represent an operation, so callers can eventually match on `code` instead of
//! parsing message text. Today's backends still return bare `String`; later
//! commits convert them to return `Diagnostic` directly.

/// A coarse classification of why a backend lowering failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCode {
    /// The construct is syntactically/structurally invalid (bad argument count,
    /// wrong operand shape) rather than merely unsupported by one backend.
    InvalidConstruct,
    /// The construct is well-formed ZSL but this backend/target cannot lower it
    /// (e.g. a type or operator with no mapping on this emitter).
    Unsupported,
    /// An expression or statement has a type this operation does not accept
    /// (e.g. `&&` on a non-bool operand).
    TypeMismatch,
    /// A buffer/shared/local access is invalid (wrong mutability, unknown name,
    /// writing a read-only buffer).
    InvalidAccess,
    /// A backend-internal invariant was violated; this indicates a compiler
    /// defect, not a user-source problem.
    Internal,
}

/// A structured backend diagnostic: what went wrong, why, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    /// Byte offset into the original ZSL source, when known. Backends that
    /// lower already-parsed IR without retained spans may leave this `None`.
    pub span: Option<usize>,
}

impl Diagnostic {
    /// Construct a diagnostic with no known source span.
    pub fn new(code: DiagnosticCode, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            span: None,
        }
    }

    /// Construct a diagnostic at a known byte offset.
    pub fn at(code: DiagnosticCode, message: impl Into<String>, span: usize) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            span: Some(span),
        }
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Diagnostic {}

/// Existing call sites return bare `String` errors; this lets `?` keep working
/// at call sites that haven't migrated to `Diagnostic` yet.
impl From<Diagnostic> for String {
    fn from(d: Diagnostic) -> String {
        d.message
    }
}

/// Wrap a legacy string error as an `Internal`-classified diagnostic. Used when
/// converting today's `Result<_, String>` backend signatures to `Diagnostic` in
/// a later commit; not called anywhere yet in this commit.
impl From<String> for Diagnostic {
    fn from(message: String) -> Diagnostic {
        Diagnostic::new(DiagnosticCode::Internal, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_display_shows_message() {
        let d = Diagnostic::new(DiagnosticCode::Unsupported, "no f64 on this target");
        assert_eq!(d.to_string(), "no f64 on this target");
    }

    #[test]
    fn diagnostic_round_trips_through_string() {
        let original = Diagnostic::at(DiagnosticCode::TypeMismatch, "bool required", 42);
        let s: String = original.clone().into();
        assert_eq!(s, "bool required");
        let back: Diagnostic = s.into();
        assert_eq!(back.code, DiagnosticCode::Internal);
        assert_eq!(back.message, "bool required");
    }
}
