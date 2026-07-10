use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

use crate::error::CrateResult;

pub fn run() -> CrateResult<()> {
    let vk_lib = VulkanLibrary::new()?;
    let info = InstanceCreateInfo::application_from_cargo_toml();

    let vk_instance = Instance::new(vk_lib, info)?;

    Ok(())
}
