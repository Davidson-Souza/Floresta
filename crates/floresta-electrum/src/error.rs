// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error;
use core::fmt;

use floresta_common::impl_error_from;
use floresta_domain::node_interface::NodeError;
use floresta_domain::wallet::error::WatchOnlyError;

#[derive(Debug)]
pub enum Error {
    /// The parameter is invalid.
    InvalidParams,
    /// The JSON string is invalid.
    Parsing(serde_json::Error),
    /// A blockchain error has occurred.
    Blockchain(Box<dyn error::Error + Send + 'static>),
    /// An IO error has occurred.
    Io(std::io::Error),
    /// A Mempool error has occurred.
    Mempool(Box<dyn error::Error + Send + Sync + 'static>),
    /// The node is unresponsive.
    NodeHandle(NodeError),
    /// Wallet error
    WalletError(WatchOnlyError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParams => write!(f, "The provided parameter is invalid"),
            Self::Parsing(e) => write!(f, "Invalid JSON string: {e}"),
            Self::Blockchain(e) => write!(f, "Blockchain error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Mempool(e) => writeln!(f, "Mempool error: {e}"),
            Self::NodeHandle(e) => write!(f, "The node is unresponsive: {e}"),
            Self::WalletError(e) => write!(f, "Wallet error: {e}"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::InvalidParams => None,
            Self::Parsing(e) => Some(e),
            Self::Blockchain(e) => Some(e.as_ref()),
            Self::Io(e) => Some(e),
            Self::Mempool(e) => Some(e.as_ref()),
            Self::NodeHandle(e) => Some(e),
            Self::WalletError(e) => Some(e),
        }
    }
}

impl_error_from!(Error, serde_json::Error, Parsing);
impl_error_from!(Error, std::io::Error, Io);
impl_error_from!(Error, WatchOnlyError, WalletError);
