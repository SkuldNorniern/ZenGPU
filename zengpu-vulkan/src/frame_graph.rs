use std::collections::VecDeque;

use ash::vk;
use zengpu_hal::Result;

use crate::depth_target::DepthTarget;
use crate::offscreen::OffscreenTarget;
use crate::swapchain::{DeviceContext, Swapchain};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ResourceId(usize);

#[derive(Copy, Clone, Debug)]
pub enum AttachmentUsage {
    /// Pass writes this as a color attachment.
    ///
    /// The render pass must use `initialLayout = UNDEFINED`, `finalLayout =
    /// COLOR_ATTACHMENT_OPTIMAL`. Later readers get a barrier.
    ColorWrite,
    /// Pass samples this in a fragment shader. The graph barriers from the last write layout
    /// to `SHADER_READ_ONLY_OPTIMAL`.
    ShaderSample,
    /// Pass writes this as a depth-stencil attachment.
    ///
    /// The render pass must use `initialLayout = UNDEFINED`, `finalLayout =
    /// DEPTH_STENCIL_ATTACHMENT_OPTIMAL`. The graph barriers if it was used differently before.
    DepthWrite,
}

struct FrameResource {
    image: vk::Image,
    _view: vk::ImageView,
    _format: vk::Format,
    _extent: vk::Extent2D,
    initial_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
}

struct PassDef {
    attachments: Vec<(ResourceId, AttachmentUsage)>,
    record: Box<dyn Fn(vk::CommandBuffer) -> Result<()> + 'static>,
}

/// Per-frame render graph: build one each frame, then `execute`.
///
/// Tracks layouts and inserts barriers for resources that change layout between passes.
/// Passes are topologically sorted by dependencies. The `mark_present` resource gets a
/// barrier to `PRESENT_SRC_KHR`.
///
/// The record closures begin and end their render passes themselves. The graph creates no render
/// passes or framebuffers.
pub struct FrameGraph {
    resources: Vec<FrameResource>,
    passes: Vec<PassDef>,
    present_resource: Option<ResourceId>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self { resources: Vec::new(), passes: Vec::new(), present_resource: None }
    }

    /// Register a color resource. `initial_layout` is the layout before the first pass; `UNDEFINED`
    /// for transient or new images.
    pub fn add_resource(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
        format: vk::Format,
        extent: vk::Extent2D,
        initial_layout: vk::ImageLayout,
    ) -> ResourceId {
        let id = ResourceId(self.resources.len());
        self.resources.push(FrameResource {
            image,
            _view: view,
            _format: format,
            _extent: extent,
            initial_layout,
            aspect: vk::ImageAspectFlags::COLOR,
        });
        id
    }

    /// Register a depth resource, for `AttachmentUsage::DepthWrite`. Use `UNDEFINED` when building
    /// per frame: `loadOp = CLEAR` discards previous content.
    pub fn add_depth_resource(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
        format: vk::Format,
        extent: vk::Extent2D,
        initial_layout: vk::ImageLayout,
    ) -> ResourceId {
        let id = ResourceId(self.resources.len());
        self.resources.push(FrameResource {
            image,
            _view: view,
            _format: format,
            _extent: extent,
            initial_layout,
            aspect: vk::ImageAspectFlags::DEPTH,
        });
        id
    }

    /// Register a color `OffscreenTarget`. Same as `add_resource` with `initial_layout
    /// = UNDEFINED`.
    pub fn add_offscreen(&mut self, target: &OffscreenTarget) -> ResourceId {
        let id = ResourceId(self.resources.len());
        self.resources.push(FrameResource {
            image: target.image(),
            _view: target.view(),
            _format: target.raw_format(),
            _extent: target.raw_extent(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            aspect: vk::ImageAspectFlags::COLOR,
        });
        id
    }

    /// Register a `DepthTarget`. Same as `add_depth_resource` with `initial_layout = UNDEFINED`.
    pub fn add_depth(&mut self, target: &DepthTarget) -> ResourceId {
        let id = ResourceId(self.resources.len());
        self.resources.push(FrameResource {
            image: target.image(),
            _view: target.view(),
            _format: crate::DEPTH_FORMAT,
            _extent: target.raw_extent(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            aspect: vk::ImageAspectFlags::DEPTH,
        });
        id
    }

    /// Register swapchain image `index`. Same as `add_resource` with `initial_layout = UNDEFINED`.
    pub fn add_swapchain_image(&mut self, sc: &Swapchain, index: u32) -> ResourceId {
        let i = index as usize;
        let id = ResourceId(self.resources.len());
        self.resources.push(FrameResource {
            image: sc.images()[i],
            _view: sc.image_views()[i],
            _format: sc.raw_format(),
            _extent: sc.raw_extent(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            aspect: vk::ImageAspectFlags::COLOR,
        });
        id
    }

    /// Add a render pass. `attachments` lists the resources it reads or writes. `record` begins and
    /// ends the render pass itself.
    pub fn add_pass<F>(
        &mut self,
        attachments: &[(ResourceId, AttachmentUsage)],
        record: F,
    ) where
        F: Fn(vk::CommandBuffer) -> Result<()> + 'static,
    {
        self.passes.push(PassDef {
            attachments: attachments.to_vec(),
            record: Box::new(record),
        });
    }

    /// Marks the resource to barrier to `PRESENT_SRC_KHR` after all passes (the final
    /// present target).
    pub fn mark_present(&mut self, res: ResourceId) {
        self.present_resource = Some(res);
    }

    /// Records all passes into `cmd`: topological sort, barriers where a layout must change, each
    /// pass's `record`, then the present barrier.
    pub fn execute(&self, cmd: vk::CommandBuffer, ctx: &DeviceContext) -> Result<()> {
        let dev = ctx.device();
        let order = self.topo_sort();
        let mut layouts: Vec<vk::ImageLayout> =
            self.resources.iter().map(|r| r.initial_layout).collect();

        for pass_idx in &order {
            let pass = &self.passes[*pass_idx];

            let mut image_barriers: Vec<vk::ImageMemoryBarrier> = Vec::new();
            let mut src_stages = vk::PipelineStageFlags::empty();
            let mut dst_stages = vk::PipelineStageFlags::empty();

            for &(ResourceId(res_idx), usage) in &pass.attachments {
                let cur = layouts[res_idx];
                let needed = target_layout(usage);

                if cur == needed
                    || (cur == vk::ImageLayout::UNDEFINED
                        && matches!(
                            usage,
                            AttachmentUsage::ColorWrite | AttachmentUsage::DepthWrite
                        ))
                {
                    continue;
                }

                let (ss, sa) = src_info(cur);
                let (ds, da) = dst_info(usage);
                src_stages |= ss;
                dst_stages |= ds;
                image_barriers.push(vk::ImageMemoryBarrier {
                    src_access_mask: sa,
                    dst_access_mask: da,
                    old_layout: cur,
                    new_layout: needed,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: self.resources[res_idx].image,
                    subresource_range: subresource(self.resources[res_idx].aspect),
                    ..Default::default()
                });
            }

            if !image_barriers.is_empty() {
                if src_stages.is_empty() {
                    src_stages = vk::PipelineStageFlags::TOP_OF_PIPE;
                }
                if dst_stages.is_empty() {
                    dst_stages = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                }
                unsafe {
                    dev.cmd_pipeline_barrier(
                        cmd,
                        src_stages,
                        dst_stages,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &image_barriers,
                    );
                }
            }

            (pass.record)(cmd)?;

            for &(ResourceId(res_idx), usage) in &pass.attachments {
                layouts[res_idx] = post_layout(usage);
            }
        }

        if let Some(ResourceId(res_idx)) = self.present_resource {
            let cur = layouts[res_idx];
            if cur != vk::ImageLayout::PRESENT_SRC_KHR {
                let (ss, sa) = src_info(cur);
                unsafe {
                    dev.cmd_pipeline_barrier(
                        cmd,
                        ss,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier {
                            src_access_mask: sa,
                            dst_access_mask: vk::AccessFlags::empty(),
                            old_layout: cur,
                            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: self.resources[res_idx].image,
                            subresource_range: subresource(self.resources[res_idx].aspect),
                            ..Default::default()
                        }],
                    );
                }
            }
        }

        Ok(())
    }

    fn topo_sort(&self) -> Vec<usize> {
        let n = self.passes.len();
        let r = self.resources.len();

        let mut writers: Vec<Vec<usize>> = vec![Vec::new(); r];
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); r];

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for &(ResourceId(res_idx), usage) in &pass.attachments {
                match usage {
                    AttachmentUsage::ColorWrite | AttachmentUsage::DepthWrite => {
                        writers[res_idx].push(pass_idx)
                    }
                    AttachmentUsage::ShaderSample => readers[res_idx].push(pass_idx),
                }
            }
        }

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_deg: Vec<usize> = vec![0; n];

        for res_idx in 0..r {
            for &w in &writers[res_idx] {
                for &rdr in &readers[res_idx] {
                    if w != rdr && !adj[w].contains(&rdr) {
                        adj[w].push(rdr);
                        in_deg[rdr] += 1;
                    }
                }
            }
        }

        let mut queue: VecDeque<usize> =
            (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        order
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn target_layout(usage: AttachmentUsage) -> vk::ImageLayout {
    match usage {
        AttachmentUsage::ColorWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        AttachmentUsage::ShaderSample => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AttachmentUsage::DepthWrite => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    }
}

fn post_layout(usage: AttachmentUsage) -> vk::ImageLayout {
    match usage {
        AttachmentUsage::ColorWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        AttachmentUsage::ShaderSample => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AttachmentUsage::DepthWrite => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    }
}

fn src_info(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => {
            (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty())
        }
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            (vk::PipelineStageFlags::FRAGMENT_SHADER, vk::AccessFlags::SHADER_READ)
        }
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => {
            (vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::empty())
        }
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

fn dst_info(usage: AttachmentUsage) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match usage {
        AttachmentUsage::ColorWrite => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        AttachmentUsage::ShaderSample => {
            (vk::PipelineStageFlags::FRAGMENT_SHADER, vk::AccessFlags::SHADER_READ)
        }
        AttachmentUsage::DepthWrite => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        ),
    }
}

fn subresource(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
