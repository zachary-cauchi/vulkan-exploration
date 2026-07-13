pub mod shaders;

use std::sync::Arc;

use tracing::debug;
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{CommandBufferUsage, CopyBufferInfo},
    memory::allocator::MemoryTypeFilter,
    pipeline::{
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderModule,
};

use crate::{error::CrateResult, vk::device::VkDevice};

pub fn example_allocate_memory_buffer(device: VkDevice) -> CrateResult<()> {
    debug!("Allocating some data.");

    let subbuffer = device.alloc_host_data(
        BufferUsage::UNIFORM_BUFFER,
        MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        [13, 37],
    )?;

    debug!(
        "Allocated. Subbuffer: {:?}, contents: {:?}",
        subbuffer,
        *subbuffer.read()?
    );

    *subbuffer.write()? = [37, 13];

    debug!("Wrote to subbuffer. New contents: {:?}", *subbuffer.read()?);

    Ok(())
}

pub fn example_copy_between_buffers(device: VkDevice) -> CrateResult<()> {
    debug!("Initialising source buffer.");

    let src_content = [1, 2, 4, 8];
    let src = device.alloc_host_data(
        BufferUsage::TRANSFER_SRC,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        src_content,
    )?;

    debug!("Initialising destination buffer.");

    let dst_content = [0; 4];
    let dst = device.alloc_host_data(
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        dst_content,
    )?;

    debug!("Creating primary command buffer.");

    let mut primary_cmd_buffer_builder =
        device.primary_cmd_buffer(CommandBufferUsage::OneTimeSubmit)?;

    primary_cmd_buffer_builder.copy_buffer(CopyBufferInfo::buffers(src.clone(), dst.clone()))?;

    let primary_cmd_buffer = primary_cmd_buffer_builder.build()?;

    debug!("Command buffer ready. Sending to device.");

    device.send_to_device(primary_cmd_buffer)?;

    let new_src_content = *src.read()?;
    let new_dst_content = *dst.read()?;

    debug!(
        "Buffer contents after operation - Source: {new_src_content:?}, Dest: {new_dst_content:?}"
    );

    Ok(())
}

pub fn example_perform_compute(device: VkDevice) -> CrateResult<()> {
    debug!("Compute shader example entered.");

    let data_list = 0..65536u32;

    debug!("Allocating buffer from iter.");

    let data_buffer = device.alloc_host_iter(
        BufferUsage::STORAGE_BUFFER,
        MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        data_list,
    )?;

    debug!("Loading shader onto device.");

    let shader: Arc<ShaderModule> = shaders::mul_by_12::load(device.device())?;

    debug!("Loading shader entrypoint.");

    // SAFETY: This will always be Some because the shader has the `main` entrypoint at compile-time.
    let entry = shader.entry_point("main").unwrap();

    debug!("Entrypoint set. Creating pipeline stage.");

    let stage = PipelineShaderStageCreateInfo::new(entry);

    debug!("Created stage.");

    let layout = PipelineLayout::new(
        device.device(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.device())?,
    )?;

    debug!("Layout created.");

    let compute_pipeline = ComputePipeline::new(
        device.device(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    debug!("Compute pipeline ready. Pipeline: {compute_pipeline:?}");

    Ok(())
}
