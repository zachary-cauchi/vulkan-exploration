pub mod device;

use std::{collections::HashMap, sync::Arc, time::Duration};

use tracing::{debug, error, info, instrument, trace, warn};
use vulkano::{
    Version, VulkanError, VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, acquire_next_image},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    error::{CrateError, CrateResult},
    examples,
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
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
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
            recreate_swapchain: false,
            window_resized: false,
            window: None,
            surface: None,
            active_device: None,
            swapchain: None,
            swapchain_images: vec![],
        })
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> CrateResult<()> {
        let attributes = WindowAttributes::default().with_title("Vulkan exploration");

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

        let Some(surface) = self.surface.clone() else {
            return Err(CrateError::missing_data(
                "Vulkan surface not yet initialised",
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
            .filter(|pd| pd.supported_extensions().contains(required_extensions))
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

        VkDevice::new(device, queue_family_index as u32, queues.collect())
    }

    fn create_swapchain(&mut self, device: Arc<Device>) -> CrateResult<()> {
        let window = self
            .window
            .clone()
            .ok_or_else(|| CrateError::missing_data("Window is uninitialised"))?;
        let surface = self
            .surface
            .clone()
            .ok_or_else(|| CrateError::missing_data("Surface is uninitialised"))?;

        let physical_device = device.physical_device();
        let caps = physical_device.surface_capabilities(&surface, Default::default())?;

        // SAFETY: Already knwon to be `Some` from above check.
        let dimensions = window.inner_size();

        let composite_alpha = caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .ok_or_else(|| {
                CrateError::missing_data("No supported compose alpha mode in surface")
            })?;

        let image_format = physical_device.surface_formats(&surface, Default::default())?[0].0;

        debug!("Creating swapchain");

        let (swapchain, swapchain_images) = Swapchain::new(
            device,
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )?;

        self.swapchain.replace(swapchain);
        self.swapchain_images = swapchain_images;

        Ok(())
    }

    pub fn build_framebuffers(
        &self,
        render_pass: Arc<RenderPass>,
    ) -> CrateResult<Vec<Arc<Framebuffer>>> {
        let framebuffers = self
            .swapchain_images
            .iter()
            .map(|img| {
                let view = ImageView::new_default(img.clone())?;
                let info = FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                };
                Ok(Framebuffer::new(render_pass.clone(), info)?)
            })
            .collect::<CrateResult<Vec<_>>>()?;

        Ok(framebuffers)
    }

    pub fn recreate_swapchain(&mut self, window: Arc<Window>) -> CrateResult<()> {
        let swapchain = self
            .swapchain
            .clone()
            .ok_or_else(|| CrateError::missing_data("Swapchain not initialised"))?;

        debug!("Recreating swapchain.");

        let new_dimensions = window.inner_size();

        let (new_swapchain, new_images) = swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..swapchain.create_info()
        })?;

        self.swapchain.replace(new_swapchain);

        self.swapchain_images.clear();
        self.swapchain_images = new_images;

        Ok(())
    }

    fn redraw(&mut self, window: Arc<Window>) -> CrateResult<()> {
        if self.recreate_swapchain {
            self.recreate_swapchain(window)
                .inspect_err(|e| error!("Failed to recreate swapchain. Error: {e}"))?;
        }

        let device = self
            .active_device
            .clone()
            .ok_or_else(|| CrateError::missing_data("No device initialised."))?;

        // SAFETY: Swapchain asserted to be `Some` above.
        let swapchain = self.swapchain().unwrap();

        debug!("Issuing redraw.");

        let cmd_buffers = examples::example_framebuffers_pipelines(self)?;
        let acquire_timeout = Duration::from_secs(1);

        let (image_index, is_suboptimal, future) =
            acquire_next_image(swapchain.clone(), Some(acquire_timeout))?;

        // // SAFETY: The vector is guaranteed to us by vulkano to not be empty and the index to always be valid.
        // let next_image = self.swapchain_images[next_image_stats.image_index as usize].clone();

        if is_suboptimal {
            warn!(
                "Acquired image ({}) is suboptimal. This shouldn't happen since we just recreated the swapchain.",
                image_index
            );
            self.recreate_swapchain = true;
            return Ok(());
        }

        debug!("Acquired image ({}).", image_index);

        let cmd_buffer = cmd_buffers[image_index as usize].clone();

        let res =
            device.send_to_device_get_swapchain_image(cmd_buffer, swapchain, image_index, future);

        match res {
            Ok(_) => {}
            Err(CrateError::VkError(VulkanError::OutOfDate)) => {
                warn!("Command buffer failed: Out of date.");
                self.recreate_swapchain = true;
                return Ok(());
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    pub fn active_device(&self) -> Option<Arc<VkDevice>> {
        self.active_device.clone()
    }

    pub fn window(&self) -> Option<Arc<Window>> {
        self.window.clone()
    }

    pub fn swapchain(&self) -> Option<Arc<Swapchain>> {
        self.swapchain.clone()
    }
}

impl ApplicationHandler for VkContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        debug!("Resuming app.");

        if self.window.is_some() && self.surface.is_some() {
            debug!("Window and surface already initialised.");
            return;
        };

        if let Err(e) = self.create_window(event_loop) {
            error!("Failed to resume. Error: {e}");
            return event_loop.exit();
        }

        let required_caps = QueueFlags::GRAPHICS;
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };

        let vk_device = match self.create_device(required_caps, &required_device_extensions) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                error!("Failed to resume. Error: {e}");
                return event_loop.exit();
            }
        };

        let device = vk_device.device();

        self.active_device = Some(vk_device);

        if let Err(e) = self.create_swapchain(device) {
            error!("Failed to create swapchain. Error: {e}");
            event_loop.exit()
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone().filter(|w| w.id() == window_id) else {
            error!("No window with ID {window_id:?} found. Not processing event {event:?}");
            return;
        };

        match event {
            WindowEvent::Destroyed | WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                debug!("Window resize to {s:?}.");
                self.recreate_swapchain = true;
                self.window_resized = true;
            }
            WindowEvent::RedrawRequested => {
                debug!("Redraw requested.");
                if let Err(e) = self.redraw(window) {
                    error!("Error during screen redraw. Error: {e}");
                    event_loop.exit()
                }
            }
            event => trace!("Unimplemented window event {event:?}"),
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
