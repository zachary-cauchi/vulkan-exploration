use std::{error::Error, fmt::Display};

use vulkano::{LoadingError, Validated, VulkanError};

#[derive(Debug)]
pub enum CrateError {
    VkLoadingErr(LoadingError),
    VkError(VulkanError),
    VkValidationError(String),
}

impl Display for CrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VkLoadingErr(err) => write!(f, "Vulkan initialisation error ({err})"),
            Self::VkError(err) => write!(f, "Vulkan error ({err})"),
            Self::VkValidationError(err) => write!(f, "Vulkan validation layer error ({err})"),
        }
    }
}

impl Error for CrateError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::VkLoadingErr(err) => Some(err),
            Self::VkError(err) => Some(err),
            Self::VkValidationError(_) => None,
        }
    }
}

impl From<LoadingError> for CrateError {
    fn from(value: LoadingError) -> Self {
        Self::VkLoadingErr(value)
    }
}

impl From<Validated<VulkanError>> for CrateError {
    fn from(value: Validated<VulkanError>) -> Self {
        let mut safe_to_unwrap = false;
        let value = value.map(|e| {
            safe_to_unwrap = true;
            e
        });

        if safe_to_unwrap {
            // SAFETY: Already confirmed to unwrap to a value.
            Self::VkError(value.unwrap())
        } else {
            // SAFETY: Implementation always returns `Some`.
            let val_err = value.source().unwrap();
            Self::VkValidationError(format!("{val_err:?}"))
        }
    }
}

pub type CrateResult<T> = Result<T, CrateError>;
