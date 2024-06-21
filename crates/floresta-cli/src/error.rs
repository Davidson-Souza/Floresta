use thiserror::Error as ThisError;
use std::error::Error;

#[derive(Debug, ThisError)]
pub enum FlorestaError {
    #[error("Unexpected error: {0}")]
    Unexpected(Box<dyn Error + Send + Sync>),
}