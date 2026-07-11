use tracing::{debug, info, warn};
use vulkano::{
    VulkanLibrary,
    instance::{Instance, InstanceCreateInfo},
};

use crate::error::{CrateError, CrateResult};

pub fn run() -> CrateResult<()> {
    let vk_lib = VulkanLibrary::new()?;
    let info = InstanceCreateInfo::application_from_cargo_toml();

    let vk_instance = Instance::new(vk_lib, info)?;

    info!("Enumerating physical devices.");

    let mut devices = vk_instance.enumerate_physical_devices()?.peekable();

    if devices.peek().is_none() {
        return Err(CrateError::NoCompatibleDevice);
    }

    for device in devices {
        let name = &device.properties().device_name;
        let device_type = device.properties().device_type;
        let driver_name = device
            .properties()
            .driver_info
            .as_ref()
            .map_or("Not found", |s| s.as_str());
        let max_api = device.api_version();

        debug!(
            "Found device. Name: '{name}', Type: '{device_type:?}', driver: '{driver_name}', Max API version: '{max_api}'",
        );

        // TODO Requires 'khr_display' extension enabled in instance.
        // match device.display_properties() {
        //     Ok(displays) => debug!("Displays: {:?}", displays),
        //     Err(e) => warn!(
        //         "Could not get displays for device. Error: {}",
        //         CrateError::from(e)
        //     ),
        // }
    }

    Ok(())
}
