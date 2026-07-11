use std::{collections::HashMap, sync::Arc};

use tracing::{debug, info, instrument};
use vulkano::{
    VulkanLibrary,
    device::{QueueFlags, physical::PhysicalDevice},
    instance::{Instance, InstanceCreateInfo},
};

use crate::error::{CrateError, CrateResult};

pub fn run() -> CrateResult<()> {
    let vk_lib = VulkanLibrary::new()?;
    let info = InstanceCreateInfo::application_from_cargo_toml();

    let vk_instance = Instance::new(vk_lib, info)?;

    info!("Enumerating physical devices.");

    print_devices_info(&vk_instance)?;

    Ok(())
}

fn print_devices_info(vk_instance: &Arc<Instance>) -> CrateResult<()> {
    let mut devices = vk_instance.enumerate_physical_devices()?.peekable();

    if devices.peek().is_none() {
        return Err(CrateError::NoCompatibleDevice);
    }

    for device in devices {
        print_device_info(device);
    }

    Ok(())
}

#[instrument(name = "device", skip_all, fields(name = device.properties().device_name))]
fn print_device_info(device: Arc<PhysicalDevice>) {
    let device_type = device.properties().device_type;
    let driver_name = device
        .properties()
        .driver_info
        .as_ref()
        .map_or("Not found", |s| s.as_str());
    let max_api = device.api_version();

    debug!(
        "Found device. Type: '{device_type:?}', driver: '{driver_name}', Max API version: '{max_api}'",
    );

    let mut family_caps: HashMap<QueueFlags, (usize, usize)> = HashMap::new();

    for (i, queue_family) in device.queue_family_properties().iter().enumerate() {
        debug!(
            "Family {i} - Queue count: {}, flags: {:?}",
            queue_family.queue_count, queue_family.queue_flags
        );

        let caps = [
            QueueFlags::COMPUTE,
            QueueFlags::GRAPHICS,
            QueueFlags::TRANSFER,
        ];

        for cap in caps {
            if queue_family.queue_flags.intersects(cap) {
                let entry = family_caps.entry(cap).or_default();
                entry.0 += 1;
                entry.1 += queue_family.queue_count as usize;
            }
        }
    }

    for (cap, (family_count, queue_count)) in family_caps.into_iter() {
        debug!(
            "{} queue families ({} total queues) support the '{:?}' feature.",
            family_count, queue_count, cap
        );
    }

    // TODO Requires 'khr_display' extension enabled in instance.
    // match device.display_properties() {
    //     Ok(displays) => debug!("Displays: {:?}", displays),
    //     Err(e) => warn!(
    //         "Could not get displays for device. Error: {}",
    //         CrateError::from(e)
    //     ),
    // }
}
