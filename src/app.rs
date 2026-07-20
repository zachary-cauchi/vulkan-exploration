use tracing::{debug, info};
use winit::event_loop::EventLoop;

use crate::{
    error::CrateResult,
    examples::{
        example_allocate_memory_buffer, example_copy_between_buffers, example_graphics_pipeline,
        example_image, example_mandelbrot_compute, example_perform_compute,
    },
    vk::VkContext,
};

pub fn run() -> CrateResult<()> {
    let event_loop = EventLoop::new()?;
    let mut vk_context = VkContext::new(&event_loop)?;

    info!("Enumerating physical devices.");

    vk_context.print_devices_info()?;

    info!("Creating graphics device.");

    vk_context.run_app(event_loop)?;

    let device = vk_context.state().get_device().unwrap();

    debug!("Device: {device:?}");

    example_allocate_memory_buffer(device.clone())?;

    example_copy_between_buffers(device.clone())?;

    example_perform_compute(device.clone())?;

    example_image(device.clone())?;

    example_mandelbrot_compute(device.clone())?;

    example_graphics_pipeline(device)?;

    info!("Examples finished.");

    Ok(())
}
