use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        CopyDescriptorSet, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

use crate::error::CrateResult;

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

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
}
