use std::sync::Arc;

use tracing::{debug, error, info};
use vulkano::{
    device::{DeviceExtensions, QueueFlags},
    swapchain::Surface,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    platform::wayland::ActiveEventLoopExtWayland,
    window::{Window, WindowAttributes, WindowId},
};

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

    let required_device_extensions = DeviceExtensions {
        khr_display_swapchain: true,
        ..Default::default()
    };

    vk_context.run_app(event_loop)?;

    // debug!("Device: {device:?}");

    // example_allocate_memory_buffer(device.clone())?;

    // example_copy_between_buffers(device.clone())?;

    // example_perform_compute(device.clone())?;

    // example_image(device.clone())?;

    // example_mandelbrot_compute(device.clone())?;

    // example_graphics_pipeline(device)?;

    info!("Examples finished.");

    Ok(())
}
