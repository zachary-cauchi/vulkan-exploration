pub mod device;
pub mod state;

use std::{collections::HashMap, sync::Arc, time::Duration};

use tracing::{debug, error, info, instrument, trace, warn};
use vulkano::{
    Version, VulkanError, VulkanLibrary,
    device::{DeviceExtensions, QueueFlags, physical::PhysicalDevice},
    image::view::ImageView,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Surface, acquire_next_image},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use crate::{
    error::{CrateError, CrateResult},
    examples,
    vk::state::VkState,
};

#[derive(Clone)]
pub struct VkContext {
    instance: Arc<Instance>,
    window_resized: bool,
    recreate_swapchain: bool,
    state: VkState,
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
        let state = VkState::new(vk_instance.clone());

        Ok(Self {
            instance: vk_instance,
            recreate_swapchain: false,
            window_resized: false,
            state,
        })
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

    pub fn build_framebuffers(
        &self,
        render_pass: Arc<RenderPass>,
    ) -> CrateResult<Vec<Arc<Framebuffer>>> {
        let framebuffers = self
            .state
            .get_swapchain_images()
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

    fn redraw(&mut self) -> CrateResult<()> {
        if self.recreate_swapchain {
            self.state
                .recreate_swapchain()
                .inspect_err(|e| error!("Failed to recreate swapchain. Error: {e}"))?;
            self.recreate_swapchain = false;
        }

        // SAFETY: Swapchain asserted to be `Some` above.
        let swapchain = self.state.get_swapchain().unwrap();

        debug!("Issuing redraw.");

        let cmd_buffers = examples::example_framebuffers_pipelines(self)?;
        let acquire_timeout = Duration::from_secs(1);

        let (image_index, is_suboptimal, future) =
            acquire_next_image(swapchain.clone(), Some(acquire_timeout))?;

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

        let result =
            self.state
                .execute_cmd_buffer_against_swapchain(cmd_buffer, future, image_index);

        match result {
            Ok(_) => Ok(()),
            Err(CrateError::VkError(VulkanError::OutOfDate)) => {
                warn!("Command buffer failed: Out of date.");
                self.recreate_swapchain = true;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn state(&self) -> &VkState {
        &self.state
    }
}

impl ApplicationHandler for VkContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        debug!("Resuming app.");

        if self.state.get_window().is_ok() && self.state.get_surface().is_ok() {
            debug!("Window and surface already initialised.");
            return;
        };

        if let Err(e) = self.state.create_window(event_loop) {
            error!("Failed to resume. Error: {e}");
            return event_loop.exit();
        }

        let required_caps = QueueFlags::GRAPHICS;
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };

        if let Err(e) = self
            .state
            .create_device(required_caps, &required_device_extensions)
        {
            error!("Failed to resume. Error: {e}");
            return event_loop.exit();
        };

        if let Err(e) = self.state.create_swapchain() {
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
        match self.state.get_window() {
            Ok(w) if w.id() == window_id => {}
            Ok(_) => {
                error!("No window with ID {window_id:?} found. Not processing event {event:?}");
                return;
            }
            Err(e) => {
                error!("Failed to handle window event. Error: {e}");
                return;
            }
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
                if let Err(e) = self.redraw() {
                    error!("Error during screen redraw. Error: {e}");
                    event_loop.exit()
                } else {
                    debug!("Redraw complete.");
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
