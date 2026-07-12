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

    debug!("Allocating some data.");

    let subbuffer = device.alloc_host_data([13, 37])?;

    debug!(
        "Allocated. Subbuffer: {:?}, contents: {:?}",
        subbuffer,
        *subbuffer.read()?
    );

    *subbuffer.write()? = [37, 13];

    debug!("Wrote to subbuffer. New contents: {:?}", *subbuffer.read()?);
    Ok(())
}
