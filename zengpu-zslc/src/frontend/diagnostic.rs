//! Backends return `Diagnostic` when they cannot represent an operation, so callers can match on
//! `code`. Some still return a bare `String`.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCode {
    /// Invalid construct (bad argument count, operand shape), not merely unsupported by
    /// one backend.
    InvalidConstruct,
    /// Valid ZSL that this backend/target cannot lower.
    Unsupported,
    /// Operand type not accepted (e.g. `&&` on a non-bool).
    TypeMismatch,
    /// Invalid buffer/shared/local access: mutability, unknown name, writing a read-only buffer.
    InvalidAccess,
    /// Backend invariant violated: a compiler defect, not a user error.
    Internal,
    /// Subgroup collective inside non-subgroup-uniform control flow.
    NonUniformControlFlow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    /// Byte offset into the ZSL source; `None` when the IR kept no spans.
    pub span: Option<usize>,
}

impl Diagnostic {
    pub fn new(code: DiagnosticCode, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            span: None,
        }
    }

    pub fn at(code: DiagnosticCode, message: impl Into<String>, span: usize) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            span: Some(span),
        }
    }
}

impl Display for Diagnostic {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.message)
    }
}

impl Error for Diagnostic {}

/// Lets `?` convert a bare `String` error into a `Diagnostic`.
impl From<Diagnostic> for String {
    fn from(d: Diagnostic) -> String {
        d.message
    }
}

/// Wraps a string error as an `Internal` diagnostic.
impl From<String> for Diagnostic {
    fn from(message: String) -> Diagnostic {
        Diagnostic::new(DiagnosticCode::Internal, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
