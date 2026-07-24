use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        CopyDescriptorSet, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
    },
    device::{Device, Queue},
    image::{Image, ImageCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        PipelineLayout, PipelineShaderStageCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    swapchain::{PresentFuture, Swapchain, SwapchainPresentInfo},
    sync::{
        self, GpuFuture,
        fence::Fence,
        future::{FenceSignalFuture, JoinFuture, NowFuture},
    },
};

use crate::error::{CrateError, CrateResult};

#[derive(Debug, Clone)]
pub struct VkDevice {
    device: Arc<Device>,
    queue_family_index: u32,
    queues: Vec<Arc<Queue>>,
    mem_allocator: Arc<StandardMemoryAllocator>,
    cmd_allocator: Arc<StandardCommandBufferAllocator>,
    desc_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VkDevice {
    pub(super) fn new(
        device: Arc<Device>,
        queue_family_index: u32,
        queues: Vec<Arc<Queue>>,
    ) -> CrateResult<Self> {
        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let desc_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Ok(Self {
            device,
            queues,
            queue_family_index,
            mem_allocator,
            cmd_allocator,
            desc_allocator,
        })
    }

    /// Demo function to allocate some data.
    /// Creates an on-device, host-accessible subbuffer for accessing the data.
    pub fn alloc_host_data<D>(
        &self,
        buffer_type: BufferUsage,
        mem_type: MemoryTypeFilter,
        data: D,
    ) -> CrateResult<Subbuffer<D>>
    where
        D: BufferContents,
    {
        let subbuffer = Buffer::from_data(
            self.mem_allocator.clone(),
            BufferCreateInfo {
                usage: buffer_type,
                // usage: buffer_type,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: mem_type,
                // memory_type_filter: mem_type,
                ..Default::default()
            },
            data,
        )?;

        Ok(subbuffer)
    }

    /// Demo function to allocate some data from an iterator of finite, known size.
    /// Creates an on-device, host-accessible subbuffer for accessing the data.
    pub fn alloc_host_iter<D, I>(
        &self,
        buffer_type: BufferUsage,
        mem_type: MemoryTypeFilter,
        data_iter: I,
    ) -> CrateResult<Subbuffer<[D]>>
    where
        D: BufferContents,
        I: IntoIterator<Item = D>,
        I::IntoIter: ExactSizeIterator,
    {
        let buffer = Buffer::from_iter(
            self.mem_allocator.clone(),
            BufferCreateInfo {
                usage: buffer_type,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: mem_type,
                ..Default::default()
            },
            data_iter,
        )?;

        Ok(buffer)
    }

    pub fn primary_cmd_buffer(
        &self,
        usage: CommandBufferUsage,
    ) -> CrateResult<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let builder = AutoCommandBufferBuilder::primary(
            self.cmd_allocator.clone(),
            self.queue_family_index,
            usage,
        )?;

        Ok(builder)
    }

    /// Sends one command buffer to the GPU. Inefficient due to only queueing one item for sending before flushing.
    pub fn send_to_device(&self, cmd_buffer: Arc<PrimaryAutoCommandBuffer>) -> CrateResult<()> {
        // SAFETY: Already confirmed at initialisation to have at least one element.
        let queue = self.queues.first().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(queue.clone(), cmd_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        Ok(())
    }

    pub fn descriptor_set(
        &self,
        layout: Arc<DescriptorSetLayout>,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> CrateResult<Arc<DescriptorSet>> {
        let descriptor_set = DescriptorSet::new(
            self.desc_allocator.clone(),
            layout,
            descriptor_writes,
            descriptor_copies,
        )?;

        Ok(descriptor_set)
    }

    pub fn new_image(
        &self,
        create_info: ImageCreateInfo,
        mem_type: MemoryTypeFilter,
    ) -> CrateResult<Arc<Image>> {
        let image = Image::new(
            self.mem_allocator.clone(),
            create_info,
            AllocationCreateInfo {
                memory_type_filter: mem_type,
                ..Default::default()
            },
        )?;

        Ok(image)
    }

    pub fn new_fence(&self) -> CrateResult<Fence> {
        let fence = Fence::from_pool(self.device.clone())?;
        Ok(fence)
    }

    pub fn auto_pipeline_layout<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> CrateResult<Arc<PipelineLayout>> {
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        Ok(layout)
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Synchronise with the physical device, returning the created future.
    /// Also performs general cleanup to free unused resources.
    pub fn sync_with_device(self: &Arc<Self>) -> DeviceFuture<NowFuture> {
        let mut future = sync::now(self.device());

        future.cleanup_finished();

        DeviceFuture {
            device: self.clone(),
            future,
        }
    }

    /// Create a device future with the given future.
    pub fn with_future<Fut>(self: &Arc<Self>, future: Fut) -> DeviceFuture<Fut>
    where
        Fut: GpuFuture,
    {
        DeviceFuture {
            device: self.clone(),
            future,
        }
    }
}

/// Wrapper around a `GpuFuture`.
pub struct DeviceFuture<Fut> {
    device: Arc<VkDevice>,
    future: Fut,
}

impl<Fut> DeviceFuture<Fut> {
    pub fn unwrap_future(self) -> Fut {
        self.future
    }
}

impl<Fut> DeviceFuture<Fut>
where
    Fut: GpuFuture,
{
    /// Join another future to this one. Future execution will continue after both futures have completed.
    pub fn join<F>(self, next_future: F) -> DeviceFuture<JoinFuture<Fut, F>>
    where
        F: GpuFuture,
    {
        DeviceFuture {
            device: self.device,
            future: self.future.join(next_future),
        }
    }

    /// Execute the supplied command buffer on the first device queue.
    pub fn execute(
        self,
        cmd_buffer: Arc<PrimaryAutoCommandBuffer>,
    ) -> CrateResult<DeviceFuture<CommandBufferExecFuture<Fut>>> {
        // SAFETY: Already confirmed at initialisation to have at least one element.
        let queue = self.device.queues.first().unwrap().clone();

        Ok(DeviceFuture {
            device: self.device,
            future: self.future.then_execute(queue, cmd_buffer)?,
        })
    }

    /// Present a swapchain image after execution completes.
    /// This must be called on or after a command buffer future is created (by calling `execute at least once`).
    pub fn present_swapchain(
        self,
        swapchain: Arc<Swapchain>,
        image_index: u32,
    ) -> CrateResult<DeviceFuture<PresentFuture<Fut>>> {
        let queue = self
            .future
            .queue()
            .ok_or_else(|| CrateError::missing_data("Current future has no attached queue."))?;

        let future = self.future.then_swapchain_present(
            queue,
            SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
        );

        Ok(DeviceFuture {
            device: self.device,
            future,
        })
    }

    /// Triggers signalling of a fence, followed by a flush.
    pub fn signal_fence_and_flush(self) -> CrateResult<DeviceFuture<FenceSignalFuture<Fut>>> {
        Ok(DeviceFuture {
            device: self.device,
            future: self.future.then_signal_fence_and_flush()?,
        })
    }

    pub fn boxed(self) -> DeviceFuture<Box<dyn GpuFuture>>
    where
        Fut: 'static,
    {
        DeviceFuture {
            device: self.device,
            future: self.future.boxed(),
        }
    }
}

impl<Fut> DeviceFuture<FenceSignalFuture<Fut>>
where
    Fut: GpuFuture,
{
    /// Consumes this future, waiting without timeout for it to complete.
    pub fn wait(self) -> CrateResult<()> {
        self.future.wait(None)?;
        Ok(())
    }
}
