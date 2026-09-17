// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node interface definition trait.
//!
//! Our node runs as a task that owns it and manages all its state internally. It processes data as
//! it arrives, updates our local state, and makes sure we keep doing progress. This architecture
//! is very nice because it requires minimal synchronization, provides near-perfect encapsulation,
//! improves testing and makes debug easier. The only problem is: if you are not allowed to own or
//! share the node, how do you communicate with it?
//!
//! The answer is: The node handle! You can get one by calling `get_handle`, and use it to send and
//! receive messages to/from the node. You can use it request blocks, transactions, cfilters,
//! proofs, ask the node to connect with someone, ask the node to disconnect from some peer, etc.
//!
//! This module defines the common interface used by the node handle, the actual implementation is
//! under `node_handle.rs`. We do this to make our testing easier, since we can mock a node while
//! testing other modules, and to allow people to reuse other crates without wire: simply
//! re-implement the relevant parts of node interface and you are fine!

use bitcoin::Transaction;
use bitcoin::Txid;
use floresta_domain::mempool::MempoolError;
use serde::Serialize;

use super::UtreexoNodeConfig;
use super::node::ConnectionKind;
use super::node::PeerStatus;
use super::transport::TransportProtocol;
use crate::address_man::ConnectionStats;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;

#[derive(Debug, Clone, Serialize)]
/// A struct representing a peer connected to the node.
///
/// This struct contains information about a peer connected to the node, like its address, the
/// services it provides, the user agent it's using, the height of the blockchain it's currently
/// at, its state and the kind of connection it has with the node.
pub struct PeerInfo {
    pub id: u32,
    #[serde(serialize_with = "serialize_addr")]
    pub address: BitcoinSocketAddr,
    pub services: String,
    #[serde(rename = "servicesnames")]
    pub services_names: Vec<String>,
    #[serde(rename = "relaytxes")]
    pub relay_txs: bool,
    pub user_agent: String,
    pub inbound: bool,
    pub bip152_hb_to: bool,
    pub bip152_hb_from: bool,
    pub initial_height: u32,
    #[serde(rename = "timeoffset")]
    pub time_offset: i64,
    pub state: PeerStatus,
    pub kind: ConnectionKind,
    pub permissions: Vec<String>,
    pub transport_protocol: TransportProtocol,
}

pub use floresta_common::ChainMethods;
/// Mempool-oriented methods.
///
/// These methods allows users to fetch or update mempool transtactions to/from the network.
pub trait MempoolMethods {
    type Error: core::error::Error;

    fn broadcast_transaction(
        &self,
        transaction: Transaction,
    ) -> impl Future<Output = Result<Result<Txid, MempoolError>, Self::Error>>;

    /// Gets a transaction from the mempool by its ID.
    ///
    /// This function will return a transaction from the mempool if it exists. If the transaction
    /// is not in the mempool (because it doesn't exist or because it's already been mined), this
    /// function will return `None`.
    fn get_mempool_transaction(
        &self,
        txid: Txid,
    ) -> impl Future<Output = Result<Option<Transaction>, Self::Error>>;
}

/// Methods for interacting with our peers.
pub trait NetworkMethods {
    type Error: core::error::Error;

    /// Connects to a specified address and port.
    /// This function will return a boolean indicating whether the connection was successful. It
    /// may be called multiple times, and may use hostnames or IP addresses.
    fn add_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> impl Future<Output = Result<bool, Self::Error>>;

    /// Removes a peer from the node's peer list.
    /// This function will return a boolean indicating whether the peer was successfully removed.
    /// It may be called multiple times, and may use hostnames or IP addresses.
    fn remove_peer(
        &self,
        addr: BitcoinSocketAddr,
    ) -> impl Future<Output = Result<bool, Self::Error>>;

    /// Immediately disconnect from a peer.
    ///
    /// Returns a bool indicating whether the disconnection was successful.
    fn disconnect_peer(
        &self,
        addr: BitcoinSocketAddr,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;

    /// Attempts to connect to a peer once.
    ///
    /// This function will try to connect to the peer once, but will not add it to the node's
    /// peer list. It will return a boolean indicating whether the connection was successful.
    /// It may be called multiple times, and may use hostnames or IP addresses.
    fn onetry_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> impl Future<Output = Result<bool, Self::Error>>;

    /// Gets information about all connected peers.
    ///
    /// This function will return a list of `PeerInfo` structs, each of which contains information
    /// about a single peer.
    fn get_peer_info(&self) -> impl Future<Output = Result<Vec<PeerInfo>, Self::Error>>;

    /// Returns the number of peers currently connected to the node
    fn get_connection_count(&self) -> impl Future<Output = Result<usize, Self::Error>>;

    /// Pings all connected peers to check if they are alive.
    fn ping(&self) -> impl Future<Output = Result<bool, Self::Error>>;

    /// Returns address manager statistics broken down by network.
    fn get_addrman_info(&self) -> impl Future<Output = Result<ConnectionStats, Self::Error>>;
}

/// Methods used to interact with the node's configuration.
pub trait NodeConfigMethods {
    type Error: core::error::Error;

    /// Get the current [`UtreexoNodeConfig`] from the running node.
    fn get_config(&self) -> impl Future<Output = Result<UtreexoNodeConfig, Self::Error>>;
}

/// A trait defining what methods our node can expose.
pub trait NodeMethods:
    ChainMethods + MempoolMethods + NetworkMethods + NodeConfigMethods + Send + 'static
{
}

impl<T> NodeMethods for T where
    T: ChainMethods + MempoolMethods + NetworkMethods + NodeConfigMethods + Send + 'static
{
}

fn serialize_addr<S>(local_addr: &BitcoinSocketAddr, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&local_addr.to_string())
}
