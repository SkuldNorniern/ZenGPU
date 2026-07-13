//! Structural SPIR-V validation.
//!
//! This is *not* a full SPIR-V validator — it has no type system and does not
//! check capability/decoration rules. It catches the structural defects that a
//! code generator (e.g. the ZSL lowering) is most likely to produce and that
//! make drivers fault inside `vkCreateShaderModule`/`vkCreateGraphicsPipelines`
//! with no diagnostic: dangling id references, use of id `0`, duplicate
//! definitions, ids past the declared bound, and a missing entry point.

use std::collections::HashMap;
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::decode::{self, DecodeError, Instruction, Module};

/// A single structural problem.
#[derive(Clone, Debug)]
pub struct Issue {
    pub word_offset: usize,
    pub opcode: u16,
    pub op_name: &'static str,
    pub message: String,
}

impl Display for Issue {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "word {}: {} (op {}): {}",
            self.word_offset, self.op_name, self.opcode, self.message
        )
    }
}

/// Result of validation: either a decode failure or a list of structural issues.
#[derive(Clone, Debug)]
pub enum ValidateError {
    Decode(DecodeError),
    Structural(Vec<Issue>),
}

impl Display for ValidateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::Structural(issues) => {
                writeln!(f, "{} structural issue(s):", issues.len())?;
                for issue in issues {
                    writeln!(f, "  {issue}")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for ValidateError {}

/// Validate a SPIR-V word stream structurally. `Ok(())` means no structural
/// defects were found (the module may still be semantically invalid).
pub fn validate(words: &[u32]) -> Result<(), ValidateError> {
    let module = decode::decode(words).map_err(ValidateError::Decode)?;
    let issues = validate_module(&module);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidateError::Structural(issues))
    }
}

/// Validate an already-decoded module, returning all structural issues found.
pub fn validate_module(module: &Module) -> Vec<Issue> {
    let mut issues = Vec::new();
    let bound = module.header.bound;

    // ── Pass 1: collect definitions, flag duplicates / out-of-bound ids ──────
    let mut def_count: HashMap<u32, usize> = HashMap::new();
    let mut has_entry_point = false;
    let mut has_function = false;

    for inst in &module.instructions {
        if inst.opcode == 15 {
            has_entry_point = true;
        }
        if inst.opcode == 54 {
            has_function = true;
        }
        if let Some(id) = inst.result_id {
            *def_count.entry(id).or_insert(0) += 1;
            if id == 0 {
                issues.push(issue(inst, "result id is 0 (reserved/invalid)".into()));
            }
            if id >= bound {
                issues.push(issue(
                    inst,
                    format!("result id %{id} >= declared bound {bound}"),
                ));
            }
        }
    }

    for (&id, &count) in &def_count {
        if count > 1 {
            issues.push(Issue {
                word_offset: 0,
                opcode: 0,
                op_name: "<module>",
                message: format!("id %{id} defined {count} times"),
            });
        }
    }

    // ── Pass 2: every referenced id must be defined (no dangling / id 0) ─────
    for inst in &module.instructions {
        if let Some(t) = inst.result_type {
            check_ref(&mut issues, inst, &def_count, t, "result type");
        }
        for id in inst.id_operands() {
            check_ref(&mut issues, inst, &def_count, id, "operand");
        }
    }

    // ── Module-level structural requirements ─────────────────────────────────
    if !has_entry_point {
        issues.push(module_issue("no OpEntryPoint in module"));
    }
    if !has_function {
        issues.push(module_issue("no OpFunction in module"));
    }

    issues
}

fn check_ref(
    issues: &mut Vec<Issue>,
    inst: &Instruction,
    defs: &HashMap<u32, usize>,
    id: u32,
    role: &str,
) {
    if id == 0 {
        issues.push(issue(inst, format!("{role} references id 0 (invalid)")));
    } else if !defs.contains_key(&id) {
        issues.push(issue(inst, format!("{role} references undefined id %{id}")));
    }
}

fn issue(inst: &Instruction, message: String) -> Issue {
    Issue {
        word_offset: inst.word_offset,
        opcode: inst.opcode,
        op_name: inst.info.name,
        message,
    }
}

fn module_issue(message: &str) -> Issue {
    Issue {
        word_offset: 0,
        opcode: 0,
        op_name: "<module>",
        message: message.to_string(),
    }
}
