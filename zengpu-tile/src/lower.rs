//! O0 lowering: one ZSL compute kernel per tile kernel, no schedule, one workgroup per output
//! tile. Threads run the body as scalar code. i8 operands are read as packed words and
//! reduced with `dot4_i8`.

use std::fmt::Write as _;

use crate::TileError;
use crate::ir::{Affine, Block, DType, Op, Stmt, Tile, TileKernel};

/// Workgroup size for tiles of at least this many elements.
const MAX_WORKGROUP: usize = 256;

pub(crate) fn zsl_source(kernel: &TileKernel) -> Result<(String, u32), TileError> {
    let tile_elems = kernel.tile[0] * kernel.tile[1];
    let workgroup = MAX_WORKGROUP.min(tile_elems.div_ceil(32) * 32);
    let iters = tile_elems.div_ceil(workgroup);

    let mut g = Gen {
        kernel,
        values: vec![None; kernel.values.len()],
        fresh: 0,
    };
    let mut body = String::new();
    for stmt in &kernel.body {
        g.stmt(&mut body, stmt)?;
    }

    let mut src = String::new();
    let params: Vec<String> = kernel
        .tensors
        .iter()
        .map(|t| {
            let mutability = if t.is_output { "mut " } else { "" };
            format!(
                "b_{}: device {mutability}buffer<{}>",
                t.name,
                buffer_elem(t.dtype)
            )
        })
        .collect();
    let _ = writeln!(src, "@workgroup_size({workgroup})");
    let _ = writeln!(src, "kernel {}({}) {{", kernel.name, params.join(", "));
    let _ = writeln!(src, "let gn = group_id().x");
    let _ = writeln!(src, "let gm = group_id().y");
    let _ = writeln!(src, "for it in 0..{iters} {{");
    let _ = writeln!(src, "let e = local_id().x + it * {workgroup}");
    let _ = writeln!(src, "if e < {tile_elems} {{");
    let _ = writeln!(src, "let tr = e / {}", kernel.tile[1]);
    let _ = writeln!(src, "let tc = e - tr * {}", kernel.tile[1]);
    src.push_str(&body);
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    let _ = writeln!(src, "}}");
    Ok((src, workgroup as u32))
}

/// i8 is stored packed in `u32` words.
fn buffer_elem(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 | DType::I8 => "u32",
    }
}

fn scalar(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 | DType::I8 => "u32",
    }
}

fn zero(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::F32 => "0.0",
        _ => "0",
    }
}

fn affine(a: Affine) -> String {
    let mut terms = Vec::new();
    if a.c0 != 0 || (a.c_m == 0 && a.c_n == 0 && a.c_k == 0) {
        terms.push(a.c0.to_string());
    }
    for (c, name) in [(a.c_m, "gm"), (a.c_n, "gn"), (a.c_k, "kt")] {
        match c {
            0 => {}
            1 => terms.push(name.to_string()),
            c => terms.push(format!("{c} * {name}")),
        }
    }
    format!("({})", terms.join(" + "))
}

/// `[1, cols]` values are the same in every row: the row is always 0.
fn row_arg(shape0: usize, r: &str) -> String {
    if shape0 == 1 {
        "0".to_string()
    } else {
        r.to_string()
    }
}

struct Gen<'a> {
    kernel: &'a TileKernel,
    /// Loads have no local. The consumer reads them on demand.
    values: Vec<Option<String>>,
    fresh: usize,
}

impl Gen<'_> {
    fn name(&mut self, prefix: &str) -> String {
        self.fresh += 1;
        format!("{prefix}{}", self.fresh)
    }

    fn stmt(&mut self, out: &mut String, stmt: &Stmt) -> Result<(), TileError> {
        match stmt {
            Stmt::Def(t, op) => self.def(out, *t, *op),
            Stmt::Store { block, value } => self.store(out, block, *value),
            Stmt::Loop { extent, step, body } => {
                let _ = writeln!(out, "for kt in 0..{} {{", extent.div_ceil(*step));
                for s in body {
                    self.stmt(out, s)?;
                }
                let _ = writeln!(out, "}}");
                Ok(())
            }
        }
    }

    fn def(&mut self, out: &mut String, t: Tile, op: Op) -> Result<(), TileError> {
        let dtype = self.kernel.values[t.0].dtype;
        match op {
            Op::Load(_) => {}
            Op::Zeros => {
                let v = format!("v{}", t.0);
                let _ = writeln!(out, "let {v}: {} = {}", scalar(dtype), zero(dtype));
                self.values[t.0] = Some(v);
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                let ea = self.elem_row(out, a, "tr", "tc")?;
                let eb = self.elem_row(out, b, "tr", "tc")?;
                let sym = if matches!(op, Op::Add(..)) { "+" } else { "*" };
                let v = format!("v{}", t.0);
                let _ = writeln!(out, "let {v}: {} = ({ea} {sym} {eb})", scalar(dtype));
                self.values[t.0] = Some(v);
            }
            Op::Cast(a, to) => {
                let src_dtype = self.kernel.values[a.0].dtype;
                let ea = self.elem_row(out, a, "tr", "tc")?;
                let v = format!("v{}", t.0);
                let needs_float = matches!(src_dtype, DType::I32 | DType::U32) && to == DType::F32;
                let expr = if needs_float {
                    format!("f32({ea})")
                } else {
                    ea
                };
                let _ = writeln!(out, "let {v}: {} = {expr}", scalar(to));
                self.values[t.0] = Some(v);
            }
            Op::Dot { a, b, acc } => {
                let acc_var = self.values[acc.0]
                    .clone()
                    .ok_or_else(|| lower_err("dot accumulator has no local"))?;
                let k_len = self.kernel.values[a.0].shape[1];
                let int8 = self.kernel.values[a.0].dtype == DType::I8;
                let (index, count, step) = if int8 {
                    let g = self.name("g");
                    (g, k_len / 4, 4)
                } else {
                    (self.name("kk"), k_len, 1)
                };
                let _ = writeln!(out, "for {index} in 0..{count} {{");
                let idx = if step == 1 {
                    index.clone()
                } else {
                    format!("({index} * 4)")
                };
                let ea = self.elem(out, a, "tr", &idx)?;
                let eb = self.elem(out, b, "tc", &idx)?;
                if int8 {
                    let _ = writeln!(out, "{acc_var} = dot4_i8({ea}, {eb}, {acc_var})");
                } else {
                    let _ = writeln!(out, "{acc_var} = {acc_var} + {ea} * {eb}");
                }
                let _ = writeln!(out, "}}");
                self.values[t.0] = Some(acc_var);
            }
        }
        Ok(())
    }

    fn elem_row(
        &mut self,
        out: &mut String,
        t: Tile,
        r: &str,
        c: &str,
    ) -> Result<String, TileError> {
        let shape0 = self.kernel.values[t.0].shape[0];
        let r = row_arg(shape0, r);
        self.elem(out, t, &r, c)
    }

    /// `r` and `c` are block-local ZSL u32 expressions. May emit statements into `out`.
    fn elem(&mut self, out: &mut String, t: Tile, r: &str, c: &str) -> Result<String, TileError> {
        let value = &self.kernel.values[t.0];
        match value.def {
            Op::Load(block) => Ok(self.load(out, block, r, c)),
            _ => self.values[t.0]
                .clone()
                .ok_or_else(|| lower_err("value used before it is defined")),
        }
    }

    /// Out-of-range reads return 0. For i8, returns the packed word holding column `c`.
    fn load(&mut self, out: &mut String, block: Block, r: &str, c: &str) -> String {
        let decl = &self.kernel.tensors[block.tensor.0];
        let v = self.name("ld");
        let (ty, zero) = (scalar(decl.dtype), zero(decl.dtype));
        let _ = writeln!(out, "let {v}: {ty} = {zero}");
        let gr = format!("({} * {} + {r})", affine(block.rows), block.shape[0]);
        let gc = format!("({} * {} + {c})", affine(block.cols), block.shape[1]);
        let (rows, cols) = (decl.shape[0], decl.shape[1]);
        let index = if decl.dtype == DType::I8 {
            format!("{gr} * {} + {gc} / 4", cols / 4)
        } else {
            format!("{gr} * {cols} + {gc}")
        };
        let _ = writeln!(out, "if {gr} < {rows} && {gc} < {cols} {{");
        let _ = writeln!(out, "{v} = b_{}[{index}]", decl.name);
        let _ = writeln!(out, "}}");
        v
    }

    fn store(&mut self, out: &mut String, block: &Block, value: Tile) -> Result<(), TileError> {
        let (tr, tc) = ("tr", "tc");
        let v = self.elem(out, value, tr, tc)?;
        let decl = &self.kernel.tensors[block.tensor.0];
        let gr = format!("({} * {} + {tr})", affine(block.rows), block.shape[0]);
        let gc = format!("({} * {} + {tc})", affine(block.cols), block.shape[1]);
        let (rows, cols) = (decl.shape[0], decl.shape[1]);
        let _ = writeln!(out, "if {gr} < {rows} && {gc} < {cols} {{");
        let _ = writeln!(out, "b_{}[{gr} * {cols} + {gc}] = {v}", decl.name);
        let _ = writeln!(out, "}}");
        Ok(())
    }
}

fn lower_err(msg: &str) -> TileError {
    TileError::Lower(msg.to_string())
}
