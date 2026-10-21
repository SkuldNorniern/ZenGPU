use zengpu_hal::{
    BackendPreference, Bindings, BufferHandle, ComputePipelineDesc, ComputePipelineHints,
    DispatchOp, GpuDevice, PipelineHandle, PipelineStats, ShaderDesc, ShaderHandle,
};

use crate::TileError;
use crate::ir::DType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSource {
    pub text: String,
    pub entry: &'static str,
}

/// Binding order: inputs, then outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBinding {
    pub name: String,
    pub dtype: DType,
    pub shape: [usize; 2],
    pub is_output: bool,
}

/// Per backend: a failed lowering errors on use only.
#[derive(Debug)]
pub struct CompiledTile {
    pub(crate) name: String,
    pub(crate) zsl: String,
    pub(crate) grid: [u32; 3],
    pub(crate) workgroup: [u32; 3],
    pub(crate) bindings: Vec<TileBinding>,
    /// `u32x4` aliases of the tensors, after the tensor params. Each shares its
    /// tensor's host buffer.
    pub(crate) aliases: Vec<usize>,
    /// Split-K partial workspace bytes, 0 when unsplit. Passed after the bindings.
    pub(crate) workspace: u64,
    /// Split-K reduction: sums the partials, then runs the epilogue.
    pub(crate) reduce: Option<Box<CompiledTile>>,
    /// `naive`, `staged` or `mma`: the code shape the schedule resolved to.
    pub(crate) mode: &'static str,
    /// Lanes per subgroup the code needs, when it uses the matrix unit.
    pub(crate) subgroup: Option<u32>,
    pub(crate) notes: Vec<String>,
    pub(crate) spirv: Result<Vec<u32>, TileError>,
    pub(crate) hip: Result<BackendSource, TileError>,
    pub(crate) cuda: Result<BackendSource, TileError>,
    pub(crate) msl: Result<BackendSource, TileError>,
}

impl CompiledTile {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn zsl(&self) -> &str {
        &self.zsl
    }

    /// Workgroups per dispatch, `[grid_n, grid_m, split_k]`.
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// Workspace bytes, 0 when unsplit.
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace
    }

    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    pub fn bindings(&self) -> &[TileBinding] {
        &self.bindings
    }

    pub fn schedule_name(&self) -> &'static str {
        self.mode
    }

    /// One note per fallback. Empty when the schedule applied as written.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn spirv(&self) -> Result<&[u32], TileError> {
        self.spirv.as_ref().map(Vec::as_slice).map_err(Clone::clone)
    }

    pub fn hip(&self) -> Result<&BackendSource, TileError> {
        self.hip.as_ref().map_err(Clone::clone)
    }

    pub fn cuda(&self) -> Result<&BackendSource, TileError> {
        self.cuda.as_ref().map_err(Clone::clone)
    }

    pub fn msl(&self) -> Result<&BackendSource, TileError> {
        self.msl.as_ref().map_err(Clone::clone)
    }

    pub fn load(&self, device: &dyn GpuDevice) -> Result<TileProgram, TileError> {
        let (shader, entry) = match device.backend() {
            BackendPreference::Hip => {
                let src = self.hip()?;
                (
                    gpu(device.create_shader(ShaderDesc::hip(&src.text)))?,
                    src.entry,
                )
            }
            BackendPreference::Vulkan => {
                let bytes: Vec<u8> = self.spirv()?.iter().flat_map(|w| w.to_le_bytes()).collect();
                (
                    gpu(device.create_shader(ShaderDesc::spirv(&bytes)))?,
                    "main",
                )
            }
            BackendPreference::Metal => {
                let src = self.msl()?;
                (
                    gpu(device.create_shader(ShaderDesc::msl(&src.text)))?,
                    src.entry,
                )
            }
            other => {
                return Err(TileError::Unsupported(format!(
                    "backend {other:?} has no tile runner"
                )));
            }
        };
        let desc = ComputePipelineDesc {
            shader,
            entry,
            block: self.workgroup,
        };
        let hints = ComputePipelineHints {
            subgroup_size: self.subgroup,
            capture_stats: true,
        };
        match device.create_compute_pipeline_with(desc, hints) {
            Ok(pipeline) => Ok(TileProgram {
                shader,
                pipeline,
                stats: device.pipeline_stats(pipeline),
                grid: self.grid,
                slots: self.bindings.len(),
                aliases: self.aliases.clone(),
                workspace: self.workspace,
                reduce: match &self.reduce {
                    Some(r) => Some(Box::new(r.load(device)?)),
                    None => None,
                },
            }),
            Err(e) => {
                device.destroy_shader(shader);
                Err(TileError::Gpu(e.to_string()))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TileProgram {
    shader: ShaderHandle,
    pipeline: PipelineHandle,
    /// Main pass resource report, if the device gives one.
    stats: Option<PipelineStats>,
    grid: [u32; 3],
    /// Tensor bindings. Buffers follow them, then the workspace if any.
    slots: usize,
    aliases: Vec<usize>,
    workspace: u64,
    reduce: Option<Box<TileProgram>>,
}

impl TileProgram {
    /// `None` on a device without a report.
    pub fn stats(&self) -> Option<PipelineStats> {
        self.stats
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace
    }

    /// Runs the program and waits. `buffers` follow `CompiledTile::bindings`, then the
    /// workspace if nonzero.
    pub fn dispatch(
        &self,
        device: &dyn GpuDevice,
        buffers: &[BufferHandle],
    ) -> Result<(), TileError> {
        self.dispatch_many(device, buffers, 1)
    }

    /// Runs the program `count` times as one submission with one wait. Split schedules reduce after
    /// each main pass.
    pub fn dispatch_many(
        &self,
        device: &dyn GpuDevice,
        buffers: &[BufferHandle],
        count: usize,
    ) -> Result<(), TileError> {
        let passes = self.passes();
        let slots = passes
            .iter()
            .map(|p| p.slot_indices(buffers))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ops: Vec<DispatchOp<'_>> = Vec::with_capacity(count * passes.len());
        for _ in 0..count {
            for (pass, slots) in passes.iter().zip(&slots) {
                ops.push(DispatchOp {
                    pipeline: pass.pipeline,
                    bindings: Bindings {
                        buffers: slots,
                        ..Bindings::default()
                    },
                    grid: pass.grid,
                });
            }
        }
        gpu(device.dispatch_batch(&ops))
    }

    /// The program, then its reduction pass if it has one.
    fn passes(&self) -> Vec<&TileProgram> {
        let mut out = vec![self];
        if let Some(r) = &self.reduce {
            out.push(r);
        }
        out
    }

    /// Shader slots: bindings, aliases (reusing their tensor's buffer), then the workspace.
    fn slot_indices(&self, buffers: &[BufferHandle]) -> Result<Vec<u32>, TileError> {
        let want = self.slots + usize::from(self.workspace > 0);
        if buffers.len() != want {
            return Err(TileError::Unsupported(format!(
                "dispatch got {} buffers, kernel takes {want}",
                buffers.len(),
            )));
        }
        let mut slots: Vec<u32> = buffers[..self.slots].iter().map(|b| b.index()).collect();
        slots.extend(self.aliases.iter().map(|&t| buffers[t].index()));
        if self.workspace > 0 {
            slots.push(buffers[self.slots].index());
        }
        Ok(slots)
    }

    pub fn destroy(self, device: &dyn GpuDevice) {
        if let Some(r) = self.reduce {
            r.destroy(device);
        }
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }
}

fn gpu<T>(r: zengpu_hal::Result<T>) -> Result<T, TileError> {
    r.map_err(|e| TileError::Gpu(e.to_string()))
}
