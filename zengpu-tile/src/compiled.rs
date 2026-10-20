
use zengpu_hal::{
    BackendPreference, Bindings, BufferHandle, ComputePipelineDesc, GpuDevice, PipelineHandle,
    ShaderDesc, ShaderHandle,
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

    /// Workgroup count per dispatch, as `[grid_n, grid_m, 1]`.
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    pub fn bindings(&self) -> &[TileBinding] {
        &self.bindings
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
        match device.create_compute_pipeline(desc) {
            Ok(pipeline) => Ok(TileProgram {
                shader,
                pipeline,
                grid: self.grid,
                slots: self.bindings.len(),
            }),
            Err(e) => {
                device.destroy_shader(shader);
                Err(TileError::Gpu(e.to_string()))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileProgram {
    shader: ShaderHandle,
    pipeline: PipelineHandle,
    grid: [u32; 3],
    slots: usize,
}

impl TileProgram {
    /// Runs synchronously.
    ///
    /// `buffers` are in `bindings` order: inputs, then outputs.
    pub fn dispatch(
        &self,
        device: &dyn GpuDevice,
        buffers: &[BufferHandle],
    ) -> Result<(), TileError> {
        if buffers.len() != self.slots {
            return Err(TileError::Unsupported(format!(
                "dispatch got {} buffers, kernel takes {}",
                buffers.len(),
                self.slots
            )));
        }
        let slots: Vec<u32> = buffers.iter().map(|b| b.index()).collect();
        let bindings = Bindings {
            buffers: &slots,
            ..Bindings::default()
        };
        gpu(device.dispatch(self.pipeline, bindings, self.grid))
    }

    pub fn destroy(self, device: &dyn GpuDevice) {
        device.destroy_pipeline(self.pipeline);
        device.destroy_shader(self.shader);
    }
}

fn gpu<T>(r: zengpu_hal::Result<T>) -> Result<T, TileError> {
    r.map_err(|e| TileError::Gpu(e.to_string()))
}
