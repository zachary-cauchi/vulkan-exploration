use tracing::{debug, info};
use vulkano::device::QueueFlags;

use crate::{
    error::CrateResult,
    examples::{
        example_allocate_memory_buffer, example_copy_between_buffers, example_graphics_pipeline,
        example_image, example_mandelbrot_compute, example_perform_compute,
    },
    vk::VkContext,
};

pub fn run() -> CrateResult<()> {
    let vk_context = VkContext::new()?;

    info!("Enumerating physical devices.");

    vk_context.print_devices_info()?;

    info!("Creating graphics device.");

    let device = vk_context.create_device(QueueFlags::GRAPHICS)?;

    debug!("Device: {device:?}");

    example_allocate_memory_buffer(device.clone())?;

    example_copy_between_buffers(device.clone())?;

    example_perform_compute(device.clone())?;

    example_image(device.clone())?;

    example_mandelbrot_compute(device.clone())?;

    example_graphics_pipeline(device)?;

    Ok(())
}
