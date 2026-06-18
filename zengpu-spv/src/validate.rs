//! Structural checks only: no type system, no capability or decoration rules. Catches what the ZSL
//! lowering most likely emits and what makes drivers fault silently: dangling ids, id 0, duplicate
//! definitions, ids past the bound, missing entry point.

use std::collections::HashMap;

use crate::decode::{self, DecodeError, Module};

#[derive(Clone, Debug)]
pub struct Issue {
    pub word_offset: usize,
    pub opcode: u16,
    pub op_name: &'static str,
    pub message: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "word {}: {} (op {}): {}",
            self.word_offset, self.op_name, self.opcode, self.message
        )
    }
}

#[derive(Clone, Debug)]
pub enum ValidateError {
    Decode(DecodeError),
    Structural(Vec<Issue>),
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::error::Error for ValidateError {}

/// `Ok` means no structural defects. The module may still be invalid.
pub fn validate(words: &[u32]) -> Result<(), ValidateError> {
    let module = decode::decode(words).map_err(ValidateError::Decode)?;
    let issues = validate_module(&module);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidateError::Structural(issues))
    }
}

pub fn validate_module(module: &Module) -> Vec<Issue> {
    let mut issues = Vec::new();
    let bound = module.header.bound;

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

    for inst in &module.instructions {
        if let Some(t) = inst.result_type {
            check_ref(&mut issues, inst, &def_count, t, "result type");
        }
        for id in inst.id_operands() {
            check_ref(&mut issues, inst, &def_count, id, "operand");
        }
    }

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
    inst: &crate::decode::Instruction,
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

fn issue(inst: &crate::decode::Instruction, message: String) -> Issue {
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
