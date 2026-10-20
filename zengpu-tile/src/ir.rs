//! Kernels over tile blocks, no threads or lanes. Blocks reach tensors through affine maps of the
//! grid and k.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::TileError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    /// Signed byte, one per element in buffers (read as i32 in code).
    I8,
    I32,
    U32,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::U32 => "u32",
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile(pub(crate) usize);

/// Block index `c0 + c_m * grid_m + c_n * grid_n + c_k * k`, not an element index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Affine {
    pub c0: i64,
    pub c_m: i64,
    pub c_n: i64,
    pub c_k: i64,
}

impl Affine {
    pub const fn constant(c: i64) -> Self {
        Self {
            c0: c,
            c_m: 0,
            c_n: 0,
            c_k: 0,
        }
    }

    pub const fn grid_m() -> Self {
        Self {
            c0: 0,
            c_m: 1,
            c_n: 0,
            c_k: 0,
        }
    }

    pub const fn grid_n() -> Self {
        Self {
            c0: 0,
            c_m: 0,
            c_n: 1,
            c_k: 0,
        }
    }

    pub const fn k() -> Self {
        Self {
            c0: 0,
            c_m: 0,
            c_n: 0,
            c_k: 1,
        }
    }

    pub const fn uses_k(self) -> bool {
        self.c_k != 0
    }
}

impl Display for Affine {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut terms = Vec::new();
        for (c, name) in [(self.c_m, "gm"), (self.c_n, "gn"), (self.c_k, "k")] {
            match c {
                0 => {}
                1 => terms.push(name.to_string()),
                c => terms.push(format!("{c}*{name}")),
            }
        }
        if self.c0 != 0 || terms.is_empty() {
            terms.insert(0, self.c0.to_string());
        }
        f.write_str(&terms.join("+"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Block {
    pub tensor: Tensor,
    pub rows: Affine,
    pub cols: Affine,
    /// Edge blocks are masked: loads read 0, stores skip.
    pub shape: [usize; 2],
}

impl Block {
    pub const fn new(tensor: Tensor, rows: Affine, cols: Affine, shape: [usize; 2]) -> Self {
        Self {
            tensor,
            rows,
            cols,
            shape,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Zeros,
    Load(Block),
    /// Right operand may be a `[1, cols]` row broadcast.
    Add(Tile, Tile),
    Mul(Tile, Tile),
    /// Values of dtype f16 stay f32 until stored.
    Cast(Tile, DType),
    /// `acc + a * b^T`: `a` is `[BM, BK]`, `b` is `[BN, BK]`, `acc` is `[BM, BN]`.
    Dot { a: Tile, b: Tile, acc: Tile },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Def(Tile, Op),
    Store { block: Block, value: Tile },
    Loop {
        extent: usize,
        step: usize,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorDecl {
    pub name: String,
    pub dtype: DType,
    pub shape: [usize; 2],
    pub is_output: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValueDecl {
    pub dtype: DType,
    pub shape: [usize; 2],
    pub def: Op,
    /// Loop nesting depth at definition (0 outside, 1 inside the k loop).
    pub depth: usize,
    /// Set once an accumulator has been consumed by `dot`.
    pub consumed: bool,
}

/// Validated tile kernel. Build with `TileKernel::builder`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileKernel {
    pub(crate) name: String,
    pub(crate) tile: [usize; 2],
    pub(crate) tensors: Vec<TensorDecl>,
    pub(crate) values: Vec<ValueDecl>,
    pub(crate) body: Vec<Stmt>,
}

impl TileKernel {
    pub fn builder(name: &str, tile: [usize; 2]) -> KernelBuilder {
        KernelBuilder::new(name, tile)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tile(&self) -> [usize; 2] {
        self.tile
    }

    /// `[ceil(M / BM), ceil(N / BN)]` from the output shape. Outputs must share it.
    pub fn grid(&self) -> [usize; 2] {
        let out = self
            .tensors
            .iter()
            .find(|t| t.is_output)
            .map(|t| t.shape)
            .unwrap_or([0, 0]);
        [out[0].div_ceil(self.tile[0]), out[1].div_ceil(self.tile[1])]
    }
}

impl Display for TileKernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(
            f,
            "tile v0 kernel {} tile [{}, {}]",
            self.name, self.tile[0], self.tile[1]
        )?;
        for t in &self.tensors {
            let role = if t.is_output { "output" } else { "input" };
            writeln!(
                f,
                "{role} {} {} [{}, {}]",
                t.name, t.dtype, t.shape[0], t.shape[1]
            )?;
        }
        for stmt in &self.body {
            write_stmt(f, stmt, 0, self)?;
        }
        Ok(())
    }
}

fn write_stmt(f: &mut Formatter<'_>, stmt: &Stmt, indent: usize, k: &TileKernel) -> FmtResult {
    let pad = "  ".repeat(indent);
    match stmt {
        Stmt::Def(t, op) => {
            let text = match op {
                Op::Zeros => {
                    let v = &k.values[t.0];
                    format!("zeros {} [{}, {}]", v.dtype, v.shape[0], v.shape[1])
                }
                Op::Load(b) => format!(
                    "load {} [{}, {}] shape=[{}, {}]",
                    k.tensors[b.tensor.0].name, b.rows, b.cols, b.shape[0], b.shape[1]
                ),
                Op::Add(a, b) => format!("add %{} %{}", a.0, b.0),
                Op::Mul(a, b) => format!("mul %{} %{}", a.0, b.0),
                Op::Cast(a, to) => format!("cast %{} {to}", a.0),
                Op::Dot { a, b, acc } => format!("dot %{} %{} %{}", a.0, b.0, acc.0),
            };
            writeln!(f, "{pad}%{} = {text}", t.0)
        }
        Stmt::Store { block, value } => writeln!(
            f,
            "{pad}store {} [{}, {}] shape=[{}, {}] %{}",
            k.tensors[block.tensor.0].name,
            block.rows,
            block.cols,
            block.shape[0],
            block.shape[1],
            value.0
        ),
        Stmt::Loop { extent, step, body } => {
            writeln!(f, "{pad}loop k 0..{extent} step {step} {{")?;
            for s in body {
                write_stmt(f, s, indent + 1, k)?;
            }
            writeln!(f, "{pad}}}")
        }
    }
}

/// The first invalid op is recorded and reported by `build`.
#[derive(Debug)]
pub struct KernelBuilder {
    name: String,
    tile: [usize; 2],
    tensors: Vec<TensorDecl>,
    values: Vec<ValueDecl>,
    /// Scope stack: 0 is the kernel body, 1 the k loop body.
    scopes: Vec<Vec<Stmt>>,
    error: Option<String>,
}

impl KernelBuilder {
    fn new(name: &str, tile: [usize; 2]) -> Self {
        let mut b = Self {
            name: name.to_string(),
            tile,
            tensors: Vec::new(),
            values: Vec::new(),
            scopes: vec![Vec::new()],
            error: None,
        };
        if !is_ident(name) {
            b.fail(format!("kernel name `{name}` is not an identifier"));
        }
        if tile.contains(&0) {
            b.fail(format!("tile shape {tile:?} has a zero dimension"));
        }
        b
    }

    fn fail(&mut self, msg: String) {
        if self.error.is_none() {
            self.error = Some(msg);
        }
    }

    fn depth(&self) -> usize {
        self.scopes.len() - 1
    }

    fn push(&mut self, stmt: Stmt) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(stmt);
        }
    }

    fn define(&mut self, dtype: DType, shape: [usize; 2], def: Op) -> Tile {
        self.values.push(ValueDecl {
            dtype,
            shape,
            def,
            depth: self.depth(),
            consumed: false,
        });
        let t = Tile(self.values.len() - 1);
        self.push(Stmt::Def(t, def));
        t
    }

    fn is_tile_or_row(&self, shape: [usize; 2]) -> bool {
        shape == self.tile || (shape[0] == 1 && shape[1] == self.tile[1])
    }

    fn tensor(&mut self, name: &str, dtype: DType, shape: [usize; 2], is_output: bool) -> Tensor {
        if !is_ident(name) {
            self.fail(format!("tensor name `{name}` is not an identifier"));
        }
        if shape.contains(&0) {
            self.fail(format!("tensor `{name}` has a zero dimension"));
        }
        if dtype == DType::I8 && is_output {
            self.fail(format!("output `{name}`: i8 is input-only"));
        }
        if self.tensors.iter().any(|t| t.name == name) {
            self.fail(format!("duplicate tensor name `{name}`"));
        }
        self.tensors.push(TensorDecl {
            name: name.to_string(),
            dtype,
            shape,
            is_output,
        });
        Tensor(self.tensors.len() - 1)
    }

    pub fn input(&mut self, name: &str, dtype: DType, shape: [usize; 2]) -> Tensor {
        self.tensor(name, dtype, shape, false)
    }

    pub fn output(&mut self, name: &str, dtype: DType, shape: [usize; 2]) -> Tensor {
        if let Some(prev) = self.tensors.iter().find(|t| t.is_output).map(|t| t.shape)
            && prev != shape
        {
            self.fail(format!(
                "output `{name}` shape {shape:?} differs from {prev:?}"
            ));
        }
        self.tensor(name, dtype, shape, true)
    }

    pub fn zeros(&mut self, dtype: DType, shape: [usize; 2]) -> Tile {
        if shape != self.tile {
            self.fail(format!(
                "zeros shape {shape:?} must equal tile {:?}",
                self.tile
            ));
        }
        if dtype == DType::I8 {
            self.fail("zeros cannot be i8".to_string());
        }
        self.define(dtype, shape, Op::Zeros)
    }

    fn check_block(&mut self, block: &Block, want_output: bool) -> Option<DType> {
        let decl = self.tensors.get(block.tensor.0).cloned();
        let Some(decl) = decl else {
            self.fail("block names an unknown tensor".to_string());
            return None;
        };
        if decl.is_output != want_output {
            let role = if want_output { "an output" } else { "an input" };
            self.fail(format!("`{}` must be {role}", decl.name));
        }
        let coeffs = [block.rows, block.cols].map(|a| [a.c0, a.c_m, a.c_n, a.c_k]);
        if coeffs.iter().flatten().any(|&c| c < 0) {
            self.fail(format!(
                "block index map of `{}` has a negative term",
                decl.name
            ));
        }
        if (block.rows.uses_k() || block.cols.uses_k()) && self.depth() == 0 {
            self.fail(format!(
                "block of `{}` uses k outside the k loop",
                decl.name
            ));
        }
        if block.shape.contains(&0) {
            self.fail(format!("block of `{}` has a zero dimension", decl.name));
        }
        Some(decl.dtype)
    }

    pub fn load(&mut self, block: Block) -> Tile {
        let dtype = self.check_block(&block, false).unwrap_or(DType::F32);
        self.define(dtype, block.shape, Op::Load(block))
    }

    fn use_value(&mut self, t: Tile) -> Option<ValueDecl> {
        let Some(v) = self.values.get(t.0).cloned() else {
            self.fail("unknown tile value".to_string());
            return None;
        };
        if v.consumed {
            self.fail(format!("tile %{} used after dot consumed it", t.0));
        }
        if v.depth > self.depth() {
            self.fail(format!(
                "tile %{} used outside the k loop it was defined in",
                t.0
            ));
        }
        Some(v)
    }

    pub fn add(&mut self, a: Tile, b: Tile) -> Tile {
        self.binary(a, b, Op::Add)
    }

    pub fn mul(&mut self, a: Tile, b: Tile) -> Tile {
        self.binary(a, b, Op::Mul)
    }

    fn binary(&mut self, a: Tile, b: Tile, make: fn(Tile, Tile) -> Op) -> Tile {
        let (va, vb) = (self.use_value(a), self.use_value(b));
        let (Some(va), Some(vb)) = (va, vb) else {
            return self.define(DType::F32, self.tile, Op::Zeros);
        };
        if va.shape != self.tile || !self.is_tile_or_row(vb.shape) {
            self.fail(format!(
                "elementwise shapes {:?} and {:?} do not match tile {:?}",
                va.shape, vb.shape, self.tile
            ));
        }
        if va.dtype != vb.dtype {
            self.fail(format!(
                "elementwise dtypes {} and {} differ",
                va.dtype, vb.dtype
            ));
        }
        if va.dtype == DType::I8 {
            self.fail("elementwise ops on i8 are not supported".to_string());
        }
        self.define(va.dtype, va.shape, make(a, b))
    }

    /// Supported: same type, f16 <-> f32, f32 -> i32, i32 -> f32, u32 -> f32. f32 -> f16 rounds to
    /// binary16 here, so later ops see the rounded value.
    pub fn cast(&mut self, a: Tile, to: DType) -> Tile {
        let Some(va) = self.use_value(a) else {
            return self.define(to, self.tile, Op::Cast(a, to));
        };
        let ok = va.dtype == to
            || matches!(
                (va.dtype, to),
                (DType::F16, DType::F32)
                    | (DType::F32, DType::F16)
                    | (DType::F32, DType::I32)
                    | (DType::I32, DType::F32)
                    | (DType::U32, DType::F32)
            );
        if !ok || !self.is_tile_or_row(va.shape) {
            self.fail(format!(
                "cast {} -> {to} on shape {:?} is not supported",
                va.dtype, va.shape
            ));
        }
        self.define(to, va.shape, Op::Cast(a, to))
    }

    /// `acc` is consumed: use the returned tile. Operands must be loads.
    pub fn dot(&mut self, a: Tile, b: Tile, acc: Tile) -> Tile {
        let (va, vb, vacc) = (self.use_value(a), self.use_value(b), self.use_value(acc));
        let (Some(va), Some(vb), Some(vacc)) = (va, vb, vacc) else {
            return self.define(DType::F32, self.tile, Op::Zeros);
        };
        let (rows, cols) = (self.tile[0], self.tile[1]);
        let is_load = |t: Tile| matches!(self.values[t.0].def, Op::Load(_));
        if !is_load(a) || !is_load(b) {
            self.fail("dot operands must be load results".to_string());
        }
        if va.shape[0] != rows || vb.shape[0] != cols || va.shape[1] != vb.shape[1] {
            self.fail(format!(
                "dot shapes {:?} x {:?} do not fit tile {:?}",
                va.shape, vb.shape, self.tile
            ));
        }
        let acc_ok = match (va.dtype, vb.dtype, vacc.dtype) {
            (DType::F16, DType::F16, DType::F32) | (DType::F32, DType::F32, DType::F32) => true,
            (DType::I8, DType::I8, DType::I32) => va.shape[1] % 4 == 0,
            _ => false,
        };
        if !acc_ok {
            self.fail(format!(
                "dot {} x {} into {} is not supported (f16 x f16 -> f32, f32 x f32 -> f32, i8 x i8 -> i32)",
                va.dtype, vb.dtype, vacc.dtype
            ));
        }
        let is_acc = matches!(self.values[acc.0].def, Op::Zeros | Op::Dot { .. });
        if vacc.shape != self.tile || !is_acc {
            self.fail(
                "dot accumulator must be a zeros or dot result with the tile shape".to_string(),
            );
        }
        self.values[acc.0].consumed = true;
        let t = self.define(vacc.dtype, vacc.shape, Op::Dot { a, b, acc });
        self.values[t.0].depth = vacc.depth;
        t
    }

    /// Value dtype must equal the output dtype (use `cast` to convert).
    pub fn store(&mut self, block: Block, value: Tile) {
        let dtype = self.check_block(&block, true);
        let Some(vv) = self.use_value(value) else {
            return;
        };
        if block.shape != self.tile || vv.shape != self.tile {
            self.fail(format!(
                "store shapes {:?} and {:?} must equal tile {:?}",
                block.shape, vv.shape, self.tile
            ));
        }
        if let Some(d) = dtype
            && d != vv.dtype
        {
            self.fail(format!(
                "store of {} into {d} output needs an explicit cast",
                vv.dtype
            ));
        }
        self.push(Stmt::Store { block, value });
    }

    /// Loops do not nest. The body sees the index via `Affine::k`.
    pub fn loop_k(&mut self, k: usize, bk: usize, body: impl FnOnce(&mut Self)) {
        if self.depth() != 0 {
            self.fail("k loops do not nest".to_string());
            return;
        }
        if k == 0 || bk == 0 {
            self.fail("k loop extent and step must be nonzero".to_string());
            return;
        }
        self.scopes.push(Vec::new());
        body(self);
        let inner = self.scopes.pop().unwrap_or_default();
        for s in &inner {
            if let Stmt::Def(_, Op::Load(b)) = s {
                let name = self.tensors[b.tensor.0].name.clone();
                if (b.cols.uses_k() && b.shape[1] != bk) || (b.rows.uses_k() && b.shape[0] != bk) {
                    self.fail(format!(
                        "block of `{name}` in the k loop must be {bk} wide along k"
                    ));
                }
            }
        }
        self.push(Stmt::Loop {
            extent: k,
            step: bk,
            body: inner,
        });
    }

    pub fn build(self) -> Result<TileKernel, TileError> {
        if let Some(msg) = self.error {
            return Err(TileError::Build(msg));
        }
        if !self.tensors.iter().any(|t| t.is_output) {
            return Err(TileError::Build("kernel has no output".to_string()));
        }
        let body = self.scopes.into_iter().next().unwrap_or_default();
        Ok(TileKernel {
            name: self.name,
            tile: self.tile,
            tensors: self.tensors,
            values: self.values,
            body,
        })
    }
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
