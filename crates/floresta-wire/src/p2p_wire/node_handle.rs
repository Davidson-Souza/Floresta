// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node interface implementation.
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
//! This module actually implement the [`crate::node_interface::NodeMethods`] trait, and
//! contains the actual type you will use when interacting with a real node.

use std::time::Instant;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::bip158::BlockFilter;
use bitcoin::p2p::message_filter::CFCheckpt;
use bitcoin::p2p::message_filter::CFHeaders;
use floresta_domain::mempool::MempoolError;
use rustreexo::proof::Proof;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tokio::sync::oneshot::error::RecvError;

use super::UtreexoNodeConfig;
use super::node::NodeNotification;
use crate::address_man::ConnectionStats;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::node_interface::MempoolMethods;
use crate::node_interface::NetworkMethods;
use crate::node_interface::NodeConfigMethods;
use crate::node_interface::PeerInfo;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// A request that can be made to the node.
///
/// While the node is running, consumers may want to request some useful data, like block data,
/// mempool transactions or tell the node to connect with some given peers. This struct represents
/// all the possible requests that can be made to the node as well as the data that needs to be
/// sent along with the request.
pub enum UserRequest {
    /// Request the [`UtreexoNodeConfig`] of the node.
    Config,

    /// Get a block by its hash.
    ///
    /// This will cause network requests to be made to fetch the block data.
    Block(BlockHash),

    /// Get the Utreexo proof and LeafData for a block by its hash.
    UtreexoProof(BlockHash),

    /// Get an unconfirmed transaction from the mempool by its ID.
    MempoolTransaction(Txid),

    /// Return information about all connected peers.
    GetPeerInfo,

    /// Return the number of connected peers.
    GetConnectionCount,

    /// Add a peer to the node's peer list.
    ///
    /// This function will add this peer to a special list of peers such that, if we lose the
    /// connection, we will keep trying to connect to it until we succeed.
    Add((BitcoinSocketAddr, bool)),

    /// Removes a node from the node's peer list.
    ///
    /// This function will remove a node that was added with [`UserRequest::Add`]. This will **not**
    /// disconnect the peer, but if it disconnects, it will not be reconnected again.
    Remove(BitcoinSocketAddr),

    /// Attempts to connect to a peer once.
    ///
    /// Different from [`UserRequest::Add`], this function will try to connect to the peer once, but
    /// will not add it to the node's added peers list.
    Onetry((BitcoinSocketAddr, bool)),

    /// Attempt to disconnect from a peer.
    Disconnect(BitcoinSocketAddr),

    /// Ping all connected peers to check if they are alive.
    Ping,

    /// Adds a transaction to mempool and advertises it
    SendTransaction(Transaction),

    /// Return address manager statistics.
    GetAddrManInfo,

    /// Request compact filter headers from a peer, starting from a given height, until a stop hash
    /// is reached.
    GetCFilterHeaders {
        /// The first height we are requesting filters for.
        start_height: u32,

        /// The last block where we wish filters for.
        ///
        /// The remote node will send min(height(stop_hash), 2_000) headers on each request.
        stop_hash: BlockHash,
    },

    /// Request basic compact block filters for consecutive blocks.
    GetCFilters {
        /// The height of the first requested block.
        start_height: u32,

        /// Ordered hashes of every requested block.
        block_hashes: Vec<BlockHash>,
    },

    /// Request BIP157 filter-header checkpoints through a block.
    GetCFCheckpt {
        /// The final block in the checkpoint chain.
        stop_hash: BlockHash,
    },
}

#[derive(Debug)]
/// A response that can be sent back to the user.
///
/// When the user makes a request to the node, the node will respond with some data. This enum
/// represents all the possible responses that the node can send back to the user.
pub enum NodeResponse {
    /// The [`UtreexoNodeConfig`] of the node.
    Config(UtreexoNodeConfig),

    /// A response containing a block, if we could fetch it.
    Block(Option<Block>),

    /// A response containing a Utreexo proof, if we could fetch it.
    UtreexoProof(Option<Proof>),

    /// A response containing a transaction from the mempool, if we could fetch it.
    MempoolTransaction(Option<Transaction>),

    /// A response containing a list of peer information.
    GetPeerInfo(Vec<PeerInfo>),

    /// The number of connected peers
    GetConnectionCount(usize),

    /// A response indicating whether a peer was successfully added.
    Add(bool),

    /// A response indicating whether a peer was successfully removed.
    Remove(bool),

    // A response indicating whether a peer was successfully disconnected from.
    Disconnect(bool),

    /// A response indicating whether a peer was successfully connected once.
    Onetry(bool),

    /// A response indicating whether the ping was successful.
    Ping(bool),

    /// Transaction broadcast
    TransactionBroadcastResult(Result<Txid, MempoolError>),

    /// Address manager statistics.
    GetAddrManInfo(ConnectionStats),

    /// Received compact block filter headers.
    CFilterHeaders(CFHeaders),

    /// Received basic compact block filters.
    CFilters(Vec<BlockFilter>),

    /// Received BIP157 filter-header checkpoints.
    CFCheckpt(CFCheckpt),
}

#[derive(Debug)]
pub struct RequestData {
    pub time: Instant,
    pub resolve: oneshot::Sender<Option<NodeResponse>>,
    pub req: UserRequest,
}

#[derive(Debug, Clone)]
/// A struct representing the interface to the node.
///
/// This struct will be used by consumers to interact with the node. You may have as many of it as
/// you need, and you can use it to send requests to the node and get responses back.
pub struct NodeHandle {
    node_sender: UnboundedSender<NodeNotification>,
}

impl NodeHandle {
    pub fn new(node_sender: UnboundedSender<NodeNotification>) -> Self {
        Self { node_sender }
    }

    /// Sends a request to the node.
    ///
    /// This is an internal utility function that will be used to send requests to the node. It will
    /// send the request to the node and return a oneshot receiver that will be used to get the
    /// response back.
    async fn send_request(
        &self,
        request: UserRequest,
    ) -> Result<NodeResponse, oneshot::error::RecvError> {
        let (tx, rx) = oneshot::channel();
        let _ = self
            .node_sender
            .send(NodeNotification::FromUser(request, tx)); // Send the request to the node

        rx.await
    }
}

impl floresta_common::ChainMethods for NodeHandle {
    type Error = RecvError;

    async fn get_block(&self, block: BlockHash) -> Result<Option<Block>, Self::Error> {
        let val = self.send_request(UserRequest::Block(block)).await?;

        extract_variant!(Block, val);
    }

    /// Returns a list of Compact Block Filters headers for the requested block range.
    async fn get_cfilters_headers(
        &self,
        start_height: u32,
        stop_hash: BlockHash,
    ) -> Result<CFHeaders, oneshot::error::RecvError> {
        let val = self
            .send_request(UserRequest::GetCFilterHeaders {
                start_height,
                stop_hash,
            })
            .await?;

        extract_variant!(CFilterHeaders, val)
    }

    async fn get_cfilter(
        &self,
        start_height: u32,
        block_hashes: Vec<BlockHash>,
    ) -> Result<Vec<BlockFilter>, Self::Error> {
        if block_hashes.is_empty() {
            return Ok(Vec::new());
        }

        let val = self
            .send_request(UserRequest::GetCFilters {
                start_height,
                block_hashes,
            })
            .await?;

        extract_variant!(CFilters, val)
    }

    async fn get_cfcheckpt(&self, stop_hash: BlockHash) -> Result<CFCheckpt, Self::Error> {
        let val = self
            .send_request(UserRequest::GetCFCheckpt { stop_hash })
            .await?;

        extract_variant!(CFCheckpt, val)
    }
}

impl MempoolMethods for NodeHandle {
    type Error = RecvError;

    async fn broadcast_transaction(
        &self,
        transaction: Transaction,
    ) -> Result<Result<Txid, MempoolError>, Self::Error> {
        let val = self
            .send_request(UserRequest::SendTransaction(transaction))
            .await?;

        extract_variant!(TransactionBroadcastResult, val)
    }

    async fn get_mempool_transaction(
        &self,
        txid: Txid,
    ) -> Result<Option<Transaction>, Self::Error> {
        let val = self
            .send_request(UserRequest::MempoolTransaction(txid))
            .await?;

        extract_variant!(MempoolTransaction, val);
    }
}

impl NetworkMethods for NodeHandle {
    type Error = RecvError;

    async fn add_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> Result<bool, Self::Error> {
        let val = self
            .send_request(UserRequest::Add((addr, v2transport)))
            .await?;

        extract_variant!(Add, val);
    }

    async fn remove_peer(&self, addr: BitcoinSocketAddr) -> Result<bool, Self::Error> {
        let val = self.send_request(UserRequest::Remove(addr)).await?;
        extract_variant!(Remove, val);
    }

    async fn disconnect_peer(&self, addr: BitcoinSocketAddr) -> Result<bool, Self::Error> {
        let val = self.send_request(UserRequest::Disconnect(addr)).await?;

        extract_variant!(Disconnect, val);
    }

    async fn onetry_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> Result<bool, Self::Error> {
        let val = self
            .send_request(UserRequest::Onetry((addr, v2transport)))
            .await?;
        extract_variant!(Onetry, val);
    }

    async fn get_peer_info(&self) -> Result<Vec<PeerInfo>, Self::Error> {
        let val = self.send_request(UserRequest::GetPeerInfo).await?;

        extract_variant!(GetPeerInfo, val);
    }

    async fn get_connection_count(&self) -> Result<usize, Self::Error> {
        let val = self.send_request(UserRequest::GetConnectionCount).await?;

        extract_variant!(GetConnectionCount, val);
    }

    async fn ping(&self) -> Result<bool, Self::Error> {
        let val = self.send_request(UserRequest::Ping).await?;

        extract_variant!(Ping, val)
    }

    async fn get_addrman_info(&self) -> Result<ConnectionStats, Self::Error> {
        let val = self.send_request(UserRequest::GetAddrManInfo).await?;

        extract_variant!(GetAddrManInfo, val)
    }
}

impl NodeConfigMethods for NodeHandle {
    type Error = RecvError;

    async fn get_config(&self) -> Result<UtreexoNodeConfig, Self::Error> {
        let val = self.send_request(UserRequest::Config).await?;

        extract_variant!(Config, val)
    }
}

macro_rules! extract_variant {
    ($variant:ident, $var:ident) => {
        if let NodeResponse::$variant(val) = $var {
            return Ok(val);
        } else {
            panic!("Unexpected variant");
        }
    };
}

use extract_variant;

#[cfg(test)]
mod tests {
    use bitcoin::hashes::Hash;
    use floresta_common::ChainMethods;
    use tokio::sync::mpsc::unbounded_channel;

    use super::*;

    #[tokio::test]
    async fn requests_compact_filters_as_one_batch() {
        let (node_sender, mut node_receiver) = unbounded_channel();
        let handle = NodeHandle::new(node_sender);
        let block_hashes = vec![
            BlockHash::from_byte_array([1; 32]),
            BlockHash::from_byte_array([2; 32]),
        ];
        let expected_hashes = block_hashes.clone();
        let request =
            tokio::spawn(async move { handle.get_cfilter(42, block_hashes).await.unwrap() });

        let NodeNotification::FromUser(user_request, responder) =
            node_receiver.recv().await.unwrap()
        else {
            panic!("expected a user request");
        };
        assert_eq!(
            user_request,
            UserRequest::GetCFilters {
                start_height: 42,
                block_hashes: expected_hashes,
            }
        );

        let filters = vec![BlockFilter::new(&[1]), BlockFilter::new(&[2])];
        responder
            .send(NodeResponse::CFilters(filters.clone()))
            .unwrap();
        assert_eq!(request.await.unwrap(), filters);
    }
}
