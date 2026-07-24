use std::{cell::Cell, sync::Arc};

use tracing::{debug, error, instrument, warn};
use vulkano::{
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    image::{Image, ImageUsage},
    instance::Instance,
    swapchain::{PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo},
    sync::{
        self, GpuFuture,
        future::{FenceSignalFuture, JoinFuture},
    },
};
use winit::{
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

use crate::{
    error::{CrateError, CrateResult},
    vk::device::VkDevice,
};

type StoredSwapchainFuture = FenceSignalFuture<
    PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>,
>;

#[derive(Clone)]
pub struct VkState {
    instance: Arc<Instance>,
    surface: Option<Arc<Surface>>,
    window: Option<Arc<Window>>,
    device: Option<Arc<VkDevice>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_fences: Vec<Option<Arc<StoredSwapchainFuture>>>,
    prev_frame: Cell<usize>,
}

impl VkState {
    pub fn new(instance: Arc<Instance>) -> Self {
        Self {
            instance,
            surface: None,
            window: None,
            device: None,
            swapchain: None,
            swapchain_images: vec![],
            swapchain_fences: vec![],
            prev_frame: Cell::new(0),
        }
    }

    pub fn get_device(&self) -> CrateResult<Arc<VkDevice>> {
        self.device
            .clone()
            .ok_or_else(|| CrateError::missing_data("Device is uninitialised"))
    }

    pub fn get_window(&self) -> CrateResult<Arc<Window>> {
        self.window
            .clone()
            .ok_or_else(|| CrateError::missing_data("Window is uninitialised"))
    }

    pub fn get_surface(&self) -> CrateResult<Arc<Surface>> {
        self.surface
            .clone()
            .ok_or_else(|| CrateError::missing_data("Surface is uninitialised"))
    }

    pub fn get_swapchain(&self) -> CrateResult<Arc<Swapchain>> {
        self.swapchain
            .clone()
            .ok_or_else(|| CrateError::missing_data("Swapchain is uninitialised"))
    }

    pub fn get_swapchain_images(&self) -> &[Arc<Image>] {
        self.swapchain_images.as_slice()
    }

    pub fn get_image_fence(&self, index: usize) -> CrateResult<Box<dyn GpuFuture>> {
        // Ensure the current image fence (if present) is completed to avoid race conditions
        // With accidentally using the image before it's released.
        if let Some(image_fence) = &self.swapchain_fences[index] {
            image_fence.wait(None)?;
        }

        // If the previous image is still being processed (`Some`), return that future.
        // Else, create a general device-sync future.
        let prev_future = match self.swapchain_fences[self.prev_frame.get()].clone() {
            None => {
                let mut now = sync::now(self.get_device()?.device());
                now.cleanup_finished();
                now.boxed()
            }
            Some(fence) => fence.boxed(),
        };

        self.prev_frame.set(index);

        Ok(prev_future)
    }

    pub fn execute_cmd_buffer_against_swapchain(
        &mut self,
        cmd_buffer: Arc<PrimaryAutoCommandBuffer>,
        swapchain_future: SwapchainAcquireFuture,
        image_index: u32,
    ) -> CrateResult<()> {
        let device = self.get_device()?;
        let swapchain = self.get_swapchain()?;

        // Get the next image future to wait upon before issuing cmd buffer executions.
        // Syncing with the device every frame can be costly, so instead of waiting on the device,
        // continue execution from the previous frame future (which will have already synced with the device).
        // We can't reuse resources from an in-progress frame in another frame anyways,
        // so we don't lose anything by doing this.
        let future = self
            .get_image_fence(image_index as usize)?
            .join(swapchain_future);

        // Send the command buffer to the GPU and present a swapchain image afterwards.
        let res = device
            .with_future(future)
            .execute(cmd_buffer)?
            .present_swapchain(swapchain, image_index)?
            .signal_fence_and_flush();

        match res {
            Ok(new_future) => {
                let future = Arc::new(new_future.unwrap_future());
                self.swapchain_fences[image_index as usize] = Some(future);
                Ok(())
            }
            Err(e) => {
                self.swapchain_fences[image_index as usize] = None;
                Err(e)
            }
        }
    }

    pub fn create_window(&mut self, event_loop: &ActiveEventLoop) -> CrateResult<()> {
        let attributes = WindowAttributes::default().with_title("Vulkan exploration");

        let window = Arc::new(event_loop.create_window(attributes)?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;

        debug!("Window created. Id: {:?}", window.id());

        self.window.replace(window);
        self.surface.replace(surface);

        Ok(())
    }

    #[instrument(skip(self))]
    pub fn create_device(
        &mut self,
        required_caps: QueueFlags,
        required_extensions: &DeviceExtensions,
    ) -> CrateResult<()> {
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

        let vk_device = Arc::new(VkDevice::new(
            device,
            queue_family_index as u32,
            queues.collect(),
        )?);

        self.device = Some(vk_device);

        Ok(())
    }

    pub fn create_swapchain(&mut self) -> CrateResult<()> {
        let window = self.get_window()?;
        let surface = self.get_surface()?;
        let device = self.get_device()?.device();

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
        self.swapchain_fences = vec![None; self.swapchain_images.len()];
        self.prev_frame.set(0);

        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> CrateResult<()> {
        let window = self.get_window()?;
        let swapchain = self.get_swapchain()?;

        debug!("Recreating swapchain.");

        let new_dimensions = window.inner_size();

        let (new_swapchain, new_images) = swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..swapchain.create_info()
        })?;

        self.swapchain_images.clear();
        self.swapchain_images = new_images;
        self.swapchain.replace(new_swapchain);

        Ok(())
    }
}
