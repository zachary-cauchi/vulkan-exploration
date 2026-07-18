pub mod device;

use std::{collections::HashMap, sync::Arc};

use tracing::{debug, error, info, instrument};
use vulkano::{
    Version, VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::{Surface, Swapchain},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    error::{CrateError, CrateResult},
    vk::device::VkDevice,
};

#[derive(Clone)]
pub struct VkContext {
    instance: Arc<Instance>,
    window_resized: bool,
    recreate_swapchain: bool,
    surface: Option<Arc<Surface>>,
    window: Option<Arc<Window>>,
    active_device: Option<Arc<VkDevice>>,
}

impl VkContext {
    pub fn new(event_loop: &EventLoop<()>) -> CrateResult<Self> {
        let vk_lib = VulkanLibrary::new()?;
        let required_extensions = Surface::required_extensions(&event_loop)?;

        let info = InstanceCreateInfo {
            application_name: Some(env!("CARGO_PKG_NAME").to_owned()),
            application_version: Version {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
            },
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        };

        let vk_instance = Instance::new(vk_lib, info)?;

        Ok(Self {
            instance: vk_instance,
            recreate_swapchain: true,
            window_resized: true,
            window: None,
            surface: None,
            active_device: None,
        })
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> CrateResult<()> {
        let attributes = WindowAttributes::default();

        let window = Arc::new(event_loop.create_window(attributes)?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;

        debug!("Window created. Id: {:?}", window.id());

        self.window.replace(window);
        self.surface.replace(surface);

        Ok(())
    }

    pub fn run_app(&mut self, event_loop: EventLoop<()>) -> CrateResult<()> {
        info!("Running app.");

        event_loop.run_app(self)?;

        info!("App stopped.");

        Ok(())
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
    pub fn create_device(
        &self,
        required_caps: QueueFlags,
        required_extensions: &DeviceExtensions,
    ) -> CrateResult<VkDevice> {
        if required_caps.is_empty() {
            return Err(CrateError::bad_arguments("No queue capabilities specified"));
        }

        let Some((surface, window)) = self.surface.clone().zip(self.window.clone()) else {
            return Err(CrateError::missing_data(
                "Surface and window not yet initialised",
            ));
        };

        debug!(
            "Selecting physical device. Required queue capabilities: ({:?}), required device extensions: {:?}",
            required_caps, required_extensions
        );

        // Get the best device for the required queue capabilities.
        // The device with the largest queue family meeting this requirement is selected.
        let (physical_device, queue_family_index) = self
            .instance
            .enumerate_physical_devices()?
            // First filter by required extensions.
            .filter(|pd| pd.supported_extensions().contains(&required_extensions))
            .filter_map(|pd| {
                // Get the queue family index with the greatest queue quantity for the required queue features
                // and surface support.
                let qf_index = pd
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .filter(|(i, qf)| match pd.surface_support(*i as u32, &surface) {
                        Ok(supported) => supported && qf.queue_flags.contains(required_caps),
                        Err(e) => {
                            error!("Error checking queue family {i} surface support. Error: {e}");
                            false
                        }
                    })
                    .max_by_key(|(_, qf)| qf.queue_count)?
                    .0;

                Some((pd, qf_index))
            })
            .max_by_key(|(pd, qf_index)| {
                let mut device_score = *qf_index;

                device_score *= match pd.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 4,
                    PhysicalDeviceType::IntegratedGpu => 3,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 1,
                    _ => 0,
                };

                device_score
            })
            .ok_or(CrateError::NoCompatibleDevice)?;

        debug!("Physical device selected. Using queue family at index {queue_family_index}.");

        let (device, queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                enabled_extensions: *required_extensions,
                ..Default::default()
            },
        )?;

        debug!("Device created.");

        VkDevice::new(
            device,
            window,
            surface,
            queue_family_index as u32,
            queues.collect(),
        )
    }
}

impl ApplicationHandler for VkContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        debug!("Resuming app.");

        if self.window.is_some() && self.surface.is_some() {
            debug!("Window and surface already initialised.");
            return;
        }

        if let Err(e) = self.create_window(event_loop) {
            error!("Failed to resume. Error: {e}");
            return event_loop.exit();
        }

        let required_caps = QueueFlags::GRAPHICS;
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };

        match self.create_device(required_caps, &required_device_extensions) {
            Ok(d) => self.active_device = Some(Arc::new(d)),
            Err(e) => {
                error!("Failed to resume. Error: {e}");
                return event_loop.exit();
            }
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(_) = self.window.as_ref().filter(|w| w.id() == window_id) else {
            error!("No window with ID {window_id:?} found. Not processing event {event:?}");
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(s) => {
                debug!("Window resize to {s:?}.");
                self.window_resized = true;
            }
            WindowEvent::RedrawRequested => {
                debug!("Redraw requested.");
                // window.request_redraw();
            }
            event => debug!("Unimplemented window event {event:?}"),
        }
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
