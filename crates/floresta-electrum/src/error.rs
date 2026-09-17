// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error;
use core::fmt;

use floresta_common::impl_error_from;
use tokio::sync::oneshot;

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
    Mempool(Box<dyn error::Error + Send + 'static>),
    /// The node is unresponsive.
    NodeHandle(oneshot::error::RecvError),
    /// Compact-filter synchronization or rescan failed.
    CompactFilters(String),

    /// Historical rescans are unavailable because compact filters are disabled.
    CompactFiltersDisabled,
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
            Self::CompactFilters(error) => write!(f, "Compact-filter error: {error}"),
            Self::CompactFiltersDisabled => write!(f, "Compact filters are disabled"),
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
            Self::CompactFilters(_) | Self::CompactFiltersDisabled => None,
        }
    }
}

impl_error_from!(Error, serde_json::Error, Parsing);
impl_error_from!(Error, std::io::Error, Io);
impl_error_from!(Error, oneshot::error::RecvError, NodeHandle);
