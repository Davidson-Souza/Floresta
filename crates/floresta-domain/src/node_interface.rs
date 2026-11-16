// SPDX-License-Identifier: MIT OR Apache-2.0

//! Network-independent access to node data, peers, and configuration.

use core::error::Error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use core::future::Future;

use bitcoin::Block;
use bitcoin::BlockHash;
#[cfg(feature = "mempool")]
use bitcoin::Transaction;
#[cfg(feature = "mempool")]
use bitcoin::Txid;
use bitcoin::p2p::message_filter::CFHeaders;

#[cfg(feature = "mempool")]
use crate::mempool::MempoolError;

/// Errors returned when interacting with a node.
///
/// Internal failures retain only their diagnostic message.
#[derive(Debug)]
pub enum NodeError {
    /// The requested block or transaction was not found.
    NotFound,

    #[cfg(feature = "mempool")]
    /// A transaction was rejected by the mempool.
    Mempool(MempoolError),

    /// An implementation-specific failure occurred while handling the request.
    Internal(String),
}

impl Display for NodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "The requested block or transaction was not found"),
            #[cfg(feature = "mempool")]
            Self::Mempool(error) => write!(f, "Mempool error: {error}"),
            Self::Internal(error) => write!(f, "Internal node error: {error}"),
        }
    }
}

impl Error for NodeError {}

#[cfg(feature = "mempool")]
impl From<MempoolError> for NodeError {
    fn from(error: MempoolError) -> Self {
        Self::Mempool(error)
    }
}

/// Chain data that consumers can request without depending on the wire crate.
pub trait ChainMethods {
    /// Fetches a block from the network by its hash.
    ///
    /// Returns [`NodeError::NotFound`] if the block is unavailable.
    fn get_block(&self, block: BlockHash) -> impl Future<Output = Result<Block, NodeError>> + Send;

    /// Returns compact-filter headers for the requested block range.
    fn get_cfilters_headers(
        &self,
        start_height: u32,
        stop_hash: BlockHash,
    ) -> impl Future<Output = Result<CFHeaders, NodeError>> + Send;
}

/// Mempool transactions that consumers can access without depending on the wire crate.
#[cfg(feature = "mempool")]
pub trait MempoolMethods {
    /// Broadcasts a transaction, returning its ID or the reason it was rejected.
    fn broadcast_transaction(
        &self,
        transaction: Transaction,
    ) -> impl Future<Output = Result<Txid, NodeError>> + Send;

    /// Gets a transaction from the mempool by its ID.
    ///
    /// This function will return a transaction from the mempool if it exists. If the transaction
    /// is not in the mempool (because it doesn't exist or because it's already been mined), this
    /// function will return [`NodeError::NotFound`].
    fn get_mempool_transaction(
        &self,
        txid: Txid,
    ) -> impl Future<Output = Result<Transaction, NodeError>> + Send;
}

/// Methods for interacting with our peers.
pub trait NetworkMethods {
    /// The implementation-specific address used to identify and connect to peers.
    type Address;

    /// Implementation-specific information about a connected peer.
    type PeerInfo;

    /// Implementation-specific address manager connection statistics.
    type ConnectionStats;

    /// Connects to a specified address and port.
    /// This function will return a boolean indicating whether the connection was successful. It
    /// may be called multiple times, and may use hostnames or IP addresses.
    fn add_peer(
        &self,
        addr: Self::Address,
        v2transport: bool,
    ) -> impl Future<Output = Result<bool, NodeError>> + Send;

    /// Removes a peer from the node's peer list.
    /// This function will return a boolean indicating whether the peer was successfully removed.
    /// It may be called multiple times, and may use hostnames or IP addresses.
    fn remove_peer(
        &self,
        addr: Self::Address,
    ) -> impl Future<Output = Result<bool, NodeError>> + Send;

    /// Immediately disconnect from a peer.
    ///
    /// Returns a bool indicating whether the disconnection was successful.
    fn disconnect_peer(
        &self,
        addr: Self::Address,
    ) -> impl Future<Output = Result<bool, NodeError>> + Send;

    /// Attempts to connect to a peer once.
    ///
    /// This function will try to connect to the peer once, but will not add it to the node's
    /// peer list. It will return a boolean indicating whether the connection was successful.
    /// It may be called multiple times, and may use hostnames or IP addresses.
    fn onetry_peer(
        &self,
        addr: Self::Address,
        v2transport: bool,
    ) -> impl Future<Output = Result<bool, NodeError>> + Send;

    /// Gets information about all connected peers.
    ///
    /// This function will return a list of [`Self::PeerInfo`] values, each of which contains
    /// information about a single peer.
    fn get_peer_info(&self) -> impl Future<Output = Result<Vec<Self::PeerInfo>, NodeError>> + Send;

    /// Returns the number of peers currently connected to the node.
    fn get_connection_count(&self) -> impl Future<Output = Result<usize, NodeError>> + Send;

    /// Pings all connected peers to check if they are alive.
    fn ping(&self) -> impl Future<Output = Result<bool, NodeError>> + Send;

    /// Returns address manager statistics broken down by network.
    fn get_addrman_info(
        &self,
    ) -> impl Future<Output = Result<Self::ConnectionStats, NodeError>> + Send;
}

/// Methods used to interact with the node's configuration.
pub trait NodeConfigMethods {
    /// The implementation-specific configuration of the running node.
    type Config;

    /// Gets the current configuration from the running node.
    fn get_config(&self) -> impl Future<Output = Result<Self::Config, NodeError>> + Send;
}

/// A trait defining what methods our node can expose.
#[cfg(feature = "mempool")]
pub trait NodeMethods:
    ChainMethods + MempoolMethods + NetworkMethods + NodeConfigMethods + Send + 'static
{
}

#[cfg(feature = "mempool")]
impl<T> NodeMethods for T where
    T: ChainMethods + MempoolMethods + NetworkMethods + NodeConfigMethods + Send + 'static
{
}
