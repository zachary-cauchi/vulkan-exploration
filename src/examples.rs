use tracing::debug;
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{CommandBufferUsage, CopyBufferInfo},
    memory::allocator::MemoryTypeFilter,
    sync,
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
