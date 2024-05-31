use thiserror::Error;
use bitcoin::{address::error, consensus::encode::Error as BitcoinEncodeError};

#[derive(Error, Debug)]
pub enum WatchOnlyError {
    #[error("Wallet not initialized")]
    WalletNotInitialized,
    #[error("Transaction not found")]
    TransactionNotFound,
    #[error(transparent)]
    DbError(#[from] DbError),

}
#[derive(Error, Debug)]
pub enum DbError{
    #[error("Poisoned lock")]
    PoisonedLock,
    #[error(transparent)]
    KvError(#[from] kv::Error),
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("WalletNotInitialized")]
    WalletNotInitialized,
    #[error(transparent)]
    DeserializeError(#[from] BitcoinEncodeError),
    #[error("TransactionNotFound")]
    TransactionNotFound,
}