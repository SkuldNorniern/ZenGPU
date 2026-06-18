//! Render a decoded module as SPIRV-Tools-style assembly text.

use crate::decode::{Instruction, Module};
use crate::opcodes::{self, Trailing};

/// Disassemble a SPIR-V word stream to text. On decode failure, returns the
/// error rendered as a single comment line so callers can log unconditionally.
pub fn disassemble(words: &[u32]) -> String {
    match crate::decode::decode(words) {
        Ok(module) => disassemble_module(&module),
        Err(e) => format!("; <decode error: {e}>\n"),
    }
}

/// Disassemble an already-decoded module.
pub fn disassemble_module(module: &Module) -> String {
    let (major, minor) = module.header.version_pair();
    let mut out = String::new();
    out.push_str(&format!(
        "; SPIR-V {major}.{minor}  bound={}  generator=0x{:08X}\n",
        module.header.bound, module.header.generator
    ));
    for inst in &module.instructions {
        out.push_str(&render_instruction(inst));
        out.push('\n');
    }
    out
}

fn render_instruction(inst: &Instruction) -> String {
    let mut line = String::new();

    if let Some(id) = inst.result_id {
        line.push_str(&format!("%{id:<4} = ", id = id));
    } else {
        line.push_str("        ");
    }

    line.push_str(inst.info.name);

    if let Some(t) = inst.result_type {
        line.push_str(&format!(" %{t}"));
    }

    line.push_str(&render_trailing(inst));
    line
}

fn render_trailing(inst: &Instruction) -> String {
    let t = &inst.trailing;
    let mut out = String::new();
    match inst.info.trailing {
        Trailing::EntryPoint => {
            // [model lit, fn id, name string…, interface ids…]
            if let Some(model) = t.first() {
                out.push_str(&format!(" {model}"));
            }
            if let Some(fn_id) = t.get(1) {
                out.push_str(&format!(" %{fn_id}"));
            }
            if let Some(name_words) = opcodes::entry_point_name_words(t) {
                let name = opcodes::decode_string(&t[2..]);
                out.push_str(&format!(" \"{name}\""));
                for id in t.iter().skip(2 + name_words) {
                    out.push_str(&format!(" %{id}"));
                }
            }
        }
        Trailing::ExtInstImport => {
            out.push_str(&format!(" \"{}\"", opcodes::decode_string(t)));
        }
        Trailing::ExtInst => {
            if let Some(set) = t.first() {
                out.push_str(&format!(" %{set}"));
            }
            if let Some(instr) = t.get(1) {
                out.push_str(&format!(" {instr}"));
            }
            for id in t.iter().skip(2) {
                out.push_str(&format!(" %{id}"));
            }
        }
        kind => {
            let ids = opcodes::id_operands(kind, t);
            for (i, w) in t.iter().enumerate() {
                // An operand is an id iff its value appears in the id set at the
                // matching structural position; for the simple kinds the id set
                // is positional so membership testing is adequate for rendering.
                if is_id_position(kind, i, t.len()) {
                    let _ = w;
                    out.push_str(&format!(" %{w}"));
                } else {
                    out.push_str(&format!(" {w}"));
                }
            }
            let _ = ids;
        }
    }
    out
}

/// Whether trailing operand index `i` (of `len` total) is an id-ref, for the
/// positional [`Trailing`] kinds (not the string-bearing ones).
fn is_id_position(kind: Trailing, i: usize, len: usize) -> bool {
    match kind {
        Trailing::Literals => false,
        Trailing::AllIds => true,
        Trailing::IdsFrom { from } => i >= from,
        Trailing::IdsAt { idx } => idx.contains(&i),
        // String-bearing kinds are handled by the caller, never here.
        Trailing::EntryPoint | Trailing::ExtInstImport | Trailing::ExtInst => {
            let _ = len;
            false
        }
    }
}
