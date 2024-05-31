use floresta_cli::rpc_types::Error as RpcError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Invalid network specified")]
    InvalidNetwork,
    #[error("RPC client error: {0}")]
    CustomRpcError(#[from] reqwest::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Floresta RPC error: {0}")]
    FlorestaRpcError(#[from] floresta_cli::rpc_types::Error),
    #[error("Internal API error: {0}")]
    ApiError(String),
}

impl From<serde_json::Error> for AppError {
    fn from(error: serde_json::Error) -> Self {
        AppError::SerializationError(error)
    }
}
impl From<RpcError> for AppError {
    fn from(error: RpcError) -> Self {
        AppError::FlorestaRpcError(error)
    }
}
