use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::error::CrateResult;

#[derive(Debug)]
pub struct VkDevice {
    device: Arc<Device>,
    queues: Vec<Arc<Queue>>,
    mem_allocator: Arc<StandardMemoryAllocator>,
}

impl VkDevice {
    pub(super) fn new(device: Arc<Device>, queues: Vec<Arc<Queue>>) -> CrateResult<Self> {
        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(Self {
            device,
            queues,
            mem_allocator,
        })
    }

    /// Demo function to allocate some data.
    /// Creates an on-device, host-accessible subbuffer for accessing the data.
    pub fn alloc_host_data<D>(&self, data: D) -> CrateResult<Subbuffer<D>>
    where
        D: BufferContents,
    {
        let subbuffer = Buffer::from_data(
            self.mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )?;

        Ok(subbuffer)
    }
}
