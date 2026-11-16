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
//! This module implements [`floresta_domain::node_interface::NodeMethods`], and
//! contains the actual type you will use when interacting with a real node.

use std::time::Instant;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::p2p::message_filter::CFHeaders;
use floresta_domain::mempool::MempoolError;
use floresta_domain::node_interface::ChainMethods;
use floresta_domain::node_interface::MempoolMethods;
use floresta_domain::node_interface::NetworkMethods;
use floresta_domain::node_interface::NodeConfigMethods;
use floresta_domain::node_interface::NodeError;
use rustreexo::proof::Proof;
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

use super::UtreexoNodeConfig;
use super::node::ConnectionKind;
use super::node::NodeNotification;
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
    async fn send_request(&self, request: UserRequest) -> Result<NodeResponse, NodeError> {
        let (tx, rx) = oneshot::channel();
        let _ = self
            .node_sender
            .send(NodeNotification::FromUser(request, tx)); // Send the request to the node

        rx.await
            .map_err(|error| NodeError::Internal(error.to_string()))
    }
}

impl ChainMethods for NodeHandle {
    async fn get_block(&self, block: BlockHash) -> Result<Block, NodeError> {
        let val = self.send_request(UserRequest::Block(block)).await?;

        let NodeResponse::Block(block) = val else {
            panic!("Unexpected variant");
        };
        block.ok_or(NodeError::NotFound)
    }

    /// Returns a list of Compact Block Filters headers for the requested block range.
    async fn get_cfilters_headers(
        &self,
        start_height: u32,
        stop_hash: BlockHash,
    ) -> Result<CFHeaders, NodeError> {
        let val = self
            .send_request(UserRequest::GetCFilterHeaders {
                start_height,
                stop_hash,
            })
            .await?;

        extract_variant!(CFilterHeaders, val)
    }
}

impl MempoolMethods for NodeHandle {
    async fn broadcast_transaction(&self, transaction: Transaction) -> Result<Txid, NodeError> {
        let val = self
            .send_request(UserRequest::SendTransaction(transaction))
            .await?;

        let NodeResponse::TransactionBroadcastResult(result) = val else {
            panic!("Unexpected variant");
        };
        result.map_err(NodeError::from)
    }

    async fn get_mempool_transaction(&self, txid: Txid) -> Result<Transaction, NodeError> {
        let val = self
            .send_request(UserRequest::MempoolTransaction(txid))
            .await?;

        let NodeResponse::MempoolTransaction(transaction) = val else {
            panic!("Unexpected variant");
        };
        transaction.ok_or(NodeError::NotFound)
    }
}

impl NetworkMethods for NodeHandle {
    type Address = BitcoinSocketAddr;
    type PeerInfo = PeerInfo;
    type ConnectionStats = ConnectionStats;

    async fn add_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> Result<bool, NodeError> {
        let val = self
            .send_request(UserRequest::Add((addr, v2transport)))
            .await?;

        extract_variant!(Add, val);
    }

    async fn remove_peer(&self, addr: BitcoinSocketAddr) -> Result<bool, NodeError> {
        let val = self.send_request(UserRequest::Remove(addr)).await?;
        extract_variant!(Remove, val);
    }

    async fn disconnect_peer(&self, addr: BitcoinSocketAddr) -> Result<bool, NodeError> {
        let val = self.send_request(UserRequest::Disconnect(addr)).await?;

        extract_variant!(Disconnect, val);
    }

    async fn onetry_peer(
        &self,
        addr: BitcoinSocketAddr,
        v2transport: bool,
    ) -> Result<bool, NodeError> {
        let val = self
            .send_request(UserRequest::Onetry((addr, v2transport)))
            .await?;
        extract_variant!(Onetry, val);
    }

    async fn get_peer_info(&self) -> Result<Vec<PeerInfo>, NodeError> {
        let val = self.send_request(UserRequest::GetPeerInfo).await?;

        extract_variant!(GetPeerInfo, val);
    }

    async fn get_connection_count(&self) -> Result<usize, NodeError> {
        let val = self.send_request(UserRequest::GetConnectionCount).await?;

        extract_variant!(GetConnectionCount, val);
    }

    async fn ping(&self) -> Result<bool, NodeError> {
        let val = self.send_request(UserRequest::Ping).await?;

        extract_variant!(Ping, val)
    }

    async fn get_addrman_info(&self) -> Result<ConnectionStats, NodeError> {
        let val = self.send_request(UserRequest::GetAddrManInfo).await?;

        extract_variant!(GetAddrManInfo, val)
    }
}

impl NodeConfigMethods for NodeHandle {
    type Config = UtreexoNodeConfig;

    async fn get_config(&self) -> Result<UtreexoNodeConfig, NodeError> {
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

fn serialize_addr<S>(local_addr: &BitcoinSocketAddr, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&local_addr.to_string())
}

#[cfg(test)]
mod tests {
    use bitcoin::Network;
    use bitcoin::blockdata::constants::genesis_block;
    use tokio::sync::mpsc::unbounded_channel;

    use super::*;

    #[tokio::test]
    async fn missing_data_is_distinct_from_a_stopped_node() {
        let (sender, mut receiver) = unbounded_channel();
        let handle = NodeHandle::new(sender);
        let block = genesis_block(Network::Regtest);
        let txid = block.txdata[0].compute_txid();
        let responder = async {
            let Some(NodeNotification::FromUser(_, reply)) = receiver.recv().await else {
                panic!("Expected a block request");
            };
            reply.send(NodeResponse::Block(None)).unwrap();
            let Some(NodeNotification::FromUser(_, reply)) = receiver.recv().await else {
                panic!("Expected a transaction request");
            };
            reply.send(NodeResponse::MempoolTransaction(None)).unwrap();
        };
        let requests = async {
            assert!(matches!(
                handle.get_block(block.block_hash()).await,
                Err(NodeError::NotFound)
            ));
            assert!(matches!(
                handle.get_mempool_transaction(txid).await,
                Err(NodeError::NotFound)
            ));
        };
        tokio::join!(requests, responder);

        drop(receiver);
        let error = handle.get_block(block.block_hash()).await.unwrap_err();
        assert!(matches!(error, NodeError::Internal(_)));
    }

    #[tokio::test]
    async fn broadcast_rejection_preserves_the_domain_error() {
        let (sender, mut receiver) = unbounded_channel();
        let handle = NodeHandle::new(sender);
        let transaction = genesis_block(Network::Regtest).txdata.remove(0);
        let responder = async {
            let Some(NodeNotification::FromUser(_, reply)) = receiver.recv().await else {
                panic!("Expected a broadcast request");
            };
            reply
                .send(NodeResponse::TransactionBroadcastResult(Err(
                    MempoolError::ConflictingTransaction,
                )))
                .unwrap();
        };
        let (result, ()) = tokio::join!(handle.broadcast_transaction(transaction), responder);
        let error = result.unwrap_err();
        assert!(matches!(
            error,
            NodeError::Mempool(MempoolError::ConflictingTransaction)
        ));
    }
}
