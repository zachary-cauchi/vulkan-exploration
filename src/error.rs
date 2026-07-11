use std::{error::Error, fmt::Display};

use vulkano::{LoadingError, Validated, VulkanError};

#[derive(Debug)]
pub enum CrateError {
    VkLoadingErr(LoadingError),
    VkError(VulkanError),
    VkValidationError(String),
    BadArguments(String),
    NoCompatibleDevice,
}

impl CrateError {
    pub fn bad_arguments(msg: impl Into<String>) -> Self {
        Self::BadArguments(msg.into())
    }
}

impl Display for CrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VkLoadingErr(err) => write!(f, "Vulkan initialisation error ({err})"),
            Self::VkError(err) => write!(f, "Vulkan error ({err})"),
            Self::VkValidationError(err) => write!(f, "Vulkan validation layer error ({err})"),
            Self::NoCompatibleDevice => f.write_str("No compatible Vulkan GPU device found"),
            Self::BadArguments(msg) => write!(f, "Bad arguments supplied ({msg})"),
        }
    }
}

impl Error for CrateError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::VkLoadingErr(err) => Some(err),
            Self::VkError(err) => Some(err),
            Self::VkValidationError(_) => None,
            Self::NoCompatibleDevice => None,
            Self::BadArguments(_) => None,
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

impl<E> From<Validated<E>> for CrateError
where
    E: Into<CrateError> + Error + 'static,
{
    fn from(value: Validated<E>) -> Self {
        let mut safe_to_unwrap = false;
        let value = value.map(|e| {
            safe_to_unwrap = true;
            e
        });

        if safe_to_unwrap {
            // SAFETY: Already confirmed it will unwrap to a value.
            value.unwrap().into()
        } else {
            // SAFETY: Implementation always returns `Some`.
            let val_err = value.source().unwrap();
            Self::VkValidationError(format!("{val_err:?}"))
        }
    }
}

pub type CrateResult<T> = Result<T, CrateError>;
