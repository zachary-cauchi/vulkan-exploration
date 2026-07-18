use std::{error::Error, fmt::Display};

use image::ImageError;
use vulkano::{
    LoadingError, Validated, ValidationError, VulkanError, buffer::AllocateBufferError,
    command_buffer::CommandBufferExecError, image::AllocateImageError,
    pipeline::layout::IntoPipelineLayoutCreateInfoError, swapchain::FromWindowError,
    sync::HostAccessError,
};
use winit::{
    error::{EventLoopError, OsError},
    raw_window_handle::HandleError,
};

#[derive(Debug)]
pub enum CrateError {
    VkLoadingErr(LoadingError),
    VkError(VulkanError),
    VkValidationError(ValidationError),
    VkAlloc(AllocateBufferError),
    VkHostAccess(HostAccessError),
    VkCmdBufferExec(CommandBufferExecError),
    VkPipelineInfo(IntoPipelineLayoutCreateInfoError),
    VkImage(AllocateImageError),
    VkWindowSetup(FromWindowError),
    WinitEventLoop(EventLoopError),
    WinitHandle(HandleError),
    WinitOsErr(OsError),
    ImageProcessing(ImageError),
    BadArguments(String),
    MissingData(String),
    NoCompatibleDevice,
}

impl CrateError {
    pub fn bad_arguments(msg: impl Into<String>) -> Self {
        Self::BadArguments(msg.into())
    }

    pub fn missing_data(msg: impl Into<String>) -> Self {
        Self::MissingData(msg.into())
    }
}

impl Display for CrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VkLoadingErr(err) => write!(f, "Vulkan initialisation error ({err})"),
            Self::VkError(err) => write!(f, "Vulkan error ({err})"),
            Self::VkValidationError(err) => write!(f, "Vulkan validation layer error ({err})"),
            Self::VkAlloc(err) => write!(f, "Vulkan buffer allocation error ({err})"),
            Self::VkHostAccess(err) => write!(f, "Vulkan host memory access error ({err})"),
            Self::VkCmdBufferExec(err) => write!(f, "Vulkan command buffer exec error ({err})"),
            Self::VkPipelineInfo(err) => write!(f, "Vulkan pipeline layout creation error ({err})"),
            Self::VkImage(err) => write!(f, "Vulkan image allocation error ({err})"),
            Self::VkWindowSetup(err) => write!(f, "Vulkan window setup error ({err})"),
            Self::WinitEventLoop(err) => write!(f, "Winit event-handler error ({err})"),
            Self::WinitHandle(err) => write!(f, "Winit handle error ({err})"),
            Self::WinitOsErr(err) => write!(f, "Winit OS error ({err})"),
            Self::ImageProcessing(err) => write!(f, "Error in image-handling library ({err})"),
            Self::BadArguments(msg) => write!(f, "Bad arguments supplied ({msg})"),
            Self::MissingData(msg) => write!(f, "Missing expected data ({msg})"),
            Self::NoCompatibleDevice => f.write_str("No compatible Vulkan GPU device found"),
        }
    }
}

impl Error for CrateError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::VkLoadingErr(err) => Some(err),
            Self::VkError(err) => Some(err),
            Self::VkAlloc(err) => Some(err),
            Self::VkHostAccess(err) => Some(err),
            Self::VkValidationError(err) => Some(err),
            Self::VkCmdBufferExec(err) => Some(err),
            Self::VkPipelineInfo(err) => Some(err),
            Self::VkImage(err) => Some(err),
            Self::VkWindowSetup(err) => Some(err),
            Self::WinitEventLoop(err) => Some(err),
            Self::WinitHandle(err) => Some(err),
            Self::WinitOsErr(err) => Some(err),
            Self::ImageProcessing(err) => Some(err),
            Self::BadArguments(_) | Self::MissingData(_) | Self::NoCompatibleDevice => None,
        }
    }
}

impl From<LoadingError> for CrateError {
    fn from(value: LoadingError) -> Self {
        Self::VkLoadingErr(value)
    }
}

impl From<VulkanError> for CrateError {
    fn from(value: VulkanError) -> Self {
        Self::VkError(value)
    }
}

impl From<AllocateBufferError> for CrateError {
    fn from(value: AllocateBufferError) -> Self {
        Self::VkAlloc(value)
    }
}

impl From<HostAccessError> for CrateError {
    fn from(value: HostAccessError) -> Self {
        Self::VkHostAccess(value)
    }
}

impl From<ValidationError> for CrateError {
    fn from(value: ValidationError) -> Self {
        Self::VkValidationError(value)
    }
}

impl From<CommandBufferExecError> for CrateError {
    fn from(value: CommandBufferExecError) -> Self {
        Self::VkCmdBufferExec(value)
    }
}

impl From<IntoPipelineLayoutCreateInfoError> for CrateError {
    fn from(value: IntoPipelineLayoutCreateInfoError) -> Self {
        Self::VkPipelineInfo(value)
    }
}

impl From<AllocateImageError> for CrateError {
    fn from(value: AllocateImageError) -> Self {
        Self::VkImage(value)
    }
}

impl From<FromWindowError> for CrateError {
    fn from(value: FromWindowError) -> Self {
        Self::VkWindowSetup(value)
    }
}

impl<E> From<Validated<E>> for CrateError
where
    E: Into<CrateError> + Error + 'static,
{
    fn from(value: Validated<E>) -> Self {
        match value {
            Validated::Error(e) => e.into(),
            Validated::ValidationError(e) => e.into(),
        }
    }
}

impl<E> From<Box<E>> for CrateError
where
    E: Clone + Into<CrateError>,
{
    fn from(value: Box<E>) -> Self {
        value.as_ref().clone().into()
    }
}

impl From<ImageError> for CrateError {
    fn from(value: ImageError) -> Self {
        Self::ImageProcessing(value)
    }
}

impl From<EventLoopError> for CrateError {
    fn from(value: EventLoopError) -> Self {
        Self::WinitEventLoop(value)
    }
}

impl From<HandleError> for CrateError {
    fn from(value: HandleError) -> Self {
        Self::WinitHandle(value)
    }
}

impl From<OsError> for CrateError {
    fn from(value: OsError) -> Self {
        Self::WinitOsErr(value)
    }
}

pub type CrateResult<T> = Result<T, CrateError>;
