use tracing::{debug, info};
use vulkano::device::QueueFlags;

use crate::{error::CrateResult, vk::VkContext};

pub fn run() -> CrateResult<()> {
    let vk_context = VkContext::new()?;

    info!("Enumerating physical devices.");

    vk_context.print_devices_info()?;

    info!("Creating graphics device.");

    let device = vk_context.create_device(QueueFlags::GRAPHICS)?;

    debug!("Device: {device:?}");

    Ok(())
}
