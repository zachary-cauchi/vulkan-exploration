pub mod shaders;

use std::sync::Arc;

use image::Rgba;
use tracing::{debug, warn};
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{
        ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo,
    },
    descriptor_set::WriteDescriptorSet,
    format::{ClearColorValue, Format},
    image::{ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::MemoryTypeFilter,
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderModule,
};

use crate::{
    error::{CrateError, CrateResult},
    vk::device::VkDevice,
};

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

    let layout = device.auto_pipeline_layout([&stage])?;

    debug!("Layout created.");

    let compute_pipeline = ComputePipeline::new(
        device.device(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    debug!("Compute pipeline initialised. Pipeline: {compute_pipeline:?}");

    let descriptor_set_layouts = compute_pipeline.layout().set_layouts();
    // Our compute pipeline only has the one compute shader stage in its layout, so get the descriptor set for that stage.
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .ok_or_else(|| CrateError::missing_data("Empty descriptor set."))?;

    // Bind the data_buffer to binding 0.
    let data_buffer_binding = WriteDescriptorSet::buffer(0, data_buffer.clone());

    let descriptor_set =
        device.descriptor_set(descriptor_set_layout.clone(), [data_buffer_binding], [])?;

    debug!("Prepared descriptor set. Set: {descriptor_set:?}");

    debug!("Creating command buffer to dispatch compute pipeline.");

    let mut cmd_buffer_builder = device.primary_cmd_buffer(CommandBufferUsage::OneTimeSubmit)?;

    let work_group_counts = [1024, 1, 1];

    cmd_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )?;

    // SAFETY: There is no 'safe' way to dispatch programs outside the host to the physical device.
    // As such, we have to trust we did everything right to have some confidence in its safety.
    unsafe {
        cmd_buffer_builder.dispatch(work_group_counts)?;
    }

    let cmd_buffer = cmd_buffer_builder.build()?;

    debug!("Command buffer primed. Sending to physical device.");

    device.send_to_device(cmd_buffer)?;

    let shader_output = data_buffer.read()?;
    let shader_successful = shader_output
        .iter()
        .enumerate()
        .all(|(i, n)| *n == i as u32 * 12);

    match shader_successful {
        true => debug!("Shader computed all values correctly!"),
        false => warn!("Shader did not compute all values correctly."),
    }

    Ok(())
}

pub fn example_image(device: VkDevice) -> CrateResult<()> {
    const DIM_X: u32 = 1024;
    const DIM_Y: u32 = 1024;
    const DIM_Z: u32 = 1;
    const BUFFER_SIZE: u32 = DIM_X * DIM_Y * DIM_Z * 4;
    const IMG_SAVE_PATH: &str = "out/image.png";

    let dimensions: [u32; 3] = [DIM_X, DIM_Y, DIM_Z];

    debug!("Creating source image.");

    let src_image = device.new_image(
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: dimensions,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        MemoryTypeFilter::PREFER_DEVICE,
    )?;

    debug!("Done. Creating destination buffer for image exporting.");

    let buffer_iter = (0..BUFFER_SIZE).map(|_| 0u8);
    let img_dest_buffer = device.alloc_host_iter(
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        buffer_iter,
    )?;

    debug!("Creating command buffer to clear image then copy to destination buffer.");

    let mut builder = device.primary_cmd_buffer(CommandBufferUsage::OneTimeSubmit)?;

    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([1.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(src_image.clone())
        })?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            src_image.clone(),
            img_dest_buffer.clone(),
        ))?;

    let cmd_buffer = builder.build()?;

    debug!("Sending command buffer to physical device.");

    device.send_to_device(cmd_buffer)?;

    debug!("Commands executed. Retrieving image.");

    let buffer_content = img_dest_buffer.read()?;
    let final_image = image::ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..])
        .ok_or_else(|| CrateError::bad_arguments("Failed to parse raw image."))?;

    debug!("Image parsed successfully. Saving to '{IMG_SAVE_PATH}'.");

    final_image.save(IMG_SAVE_PATH)?;

    Ok(())
}

pub fn example_mandelbrot_compute(device: VkDevice) -> CrateResult<()> {
    const DIM_X: u32 = 1024;
    const DIM_Y: u32 = 1024;
    const DIM_Z: u32 = 1;
    const BUFFER_SIZE: u32 = DIM_X * DIM_Y * DIM_Z * 4;
    const IMG_SAVE_PATH: &str = "out/mandelbrot.png";

    debug!("Creating image and image view.");

    let dimensions = [DIM_X, DIM_Y, DIM_Z];
    let image = device.new_image(
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: dimensions,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        MemoryTypeFilter::PREFER_DEVICE,
    )?;

    let view = ImageView::new_default(image.clone())?;

    debug!("Creating destination buffer.");

    let buffer_iter = (0..BUFFER_SIZE).map(|_| 0u8);
    let img_dest_buffer = device.alloc_host_iter(
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        buffer_iter,
    )?;

    debug!("Loading and setting up mandelbrot shader.");

    let shader: Arc<ShaderModule> = shaders::mandelbrot::load(device.device())?;
    // SAFETY: This will always be Some because the shader has the `main` entrypoint at compile-time.
    let entry = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = device.auto_pipeline_layout([&stage])?;

    let compute_pipeline = ComputePipeline::new(
        device.device(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    debug!("Compute pipeline initialised.");

    let descriptor_set_layouts = compute_pipeline.layout().set_layouts();
    // Our compute pipeline only has the one compute shader stage in its layout, so get the descriptor set for that stage.
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .ok_or_else(|| CrateError::missing_data("Empty descriptor set."))?;

    // Bind the data_buffer to binding 0.
    let data_buffer_binding = WriteDescriptorSet::image_view(0, view.clone());

    let descriptor_set =
        device.descriptor_set(descriptor_set_layout.clone(), [data_buffer_binding], [])?;

    debug!("Descriptor set initialised. Building command buffer.");

    let work_group_counts = [DIM_X / 8, DIM_Y / 8, 1];
    let mut cmd_buffer_builder = device.primary_cmd_buffer(CommandBufferUsage::OneTimeSubmit)?;

    cmd_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )?;

    // SAFETY: There is no 'safe' way to dispatch programs outside the host to the physical device.
    // As such, we have to trust we did everything right to have some confidence in its safety.
    unsafe {
        cmd_buffer_builder.dispatch(work_group_counts)?;
    }

    cmd_buffer_builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
        image.clone(),
        img_dest_buffer.clone(),
    ))?;

    let cmd_buffer = cmd_buffer_builder.build()?;

    device.send_to_device(cmd_buffer)?;

    let buffer_content = img_dest_buffer.read()?;
    let final_image =
        image::ImageBuffer::<Rgba<u8>, _>::from_raw(DIM_X, DIM_Y, &buffer_content[..])
            .ok_or_else(|| CrateError::bad_arguments("Failed to parse raw image."))?;

    debug!("Image parsed successfully. Saving to {IMG_SAVE_PATH}.");

    final_image.save(IMG_SAVE_PATH)?;

    Ok(())
}
