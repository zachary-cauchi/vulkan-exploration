pub mod device;

use std::{collections::HashMap, sync::Arc};

use tracing::{debug, instrument};
use vulkano::{
    VulkanLibrary,
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags, physical::PhysicalDevice},
    instance::{Instance, InstanceCreateInfo},
};

use crate::{
    error::{CrateError, CrateResult},
    vk::device::VkDevice,
};

#[derive(Clone)]
pub struct VkContext {
    instance: Arc<Instance>,
}

impl VkContext {
    pub fn new() -> CrateResult<Self> {
        let vk_lib = VulkanLibrary::new()?;
        let info = InstanceCreateInfo::application_from_cargo_toml();

        let vk_instance = Instance::new(vk_lib, info)?;

        Ok(Self {
            instance: vk_instance,
        })
    }

    pub fn print_devices_info(&self) -> CrateResult<()> {
        let mut devices = self.instance.enumerate_physical_devices()?.peekable();

        if devices.peek().is_none() {
            return Err(CrateError::NoCompatibleDevice);
        }

        for device in devices {
            print_device_info(device);
        }

        Ok(())
    }

    #[instrument(skip(self))]
    pub fn create_device(&self, required_caps: QueueFlags) -> CrateResult<VkDevice> {
        if required_caps.is_empty() {
            return Err(CrateError::bad_arguments(
                "No queue capabilities specified.",
            ));
        }

        debug!("Selecting physical device.");

        // Get the best device for the required queue capabilities.
        // The device with the largest queue family meeting this requirement is selected.
        let (physical_device, queue_family_index) = self
            .instance
            .enumerate_physical_devices()?
            .filter_map(|pd| {
                let qf_index = pd
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .filter(|(_, qf)| qf.queue_flags.contains(required_caps))
                    .max_by_key(|(_, qf)| qf.queue_count)?
                    .0;

                Some((pd, qf_index))
            })
            .max_by_key(|(_, qf_index)| *qf_index)
            .ok_or(CrateError::NoCompatibleDevice)?;

        debug!("Physical device selected. Using queue family at index {queue_family_index}.");

        let (device, queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        debug!("Device created.");

        VkDevice::new(device, queue_family_index as u32, queues.collect())
    }
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
