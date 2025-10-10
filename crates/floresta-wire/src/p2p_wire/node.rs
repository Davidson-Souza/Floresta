//! Main file for this blockchain. A node is the central task that runs and handles important
//! events, such as new blocks, peer connection/disconnection, new addresses, etc.
//! A node should not care about peer-specific messages, peers'll handle things like pings.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Debug;
use std::net::IpAddr;
use std::net::SocketAddr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::ServiceFlags;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::Txid;
use floresta_chain::proof_util;
use floresta_chain::proof_util::UtreexoLeafError;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::ChainBackend;
use floresta_chain::CompactLeafData;
use floresta_common::service_flags;
use floresta_common::service_flags::UTREEXO;
use floresta_common::FractionAvg;
use floresta_compact_filters::flat_filters_store::FlatFiltersStore;
use floresta_compact_filters::network_filters::NetworkFilters;
use rand::seq::SliceRandom;
use rustreexo::accumulator::proof::Proof;
use serde::Deserialize;
use serde::Serialize;
use tokio::net::tcp::WriteHalf;
use tokio::spawn;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use super::address_man::AddressMan;
use super::address_man::AddressState;
use super::address_man::LocalAddress;
use super::block_proof::Bitmap;
use super::error::AddrParseError;
use super::error::WireError;
use super::mempool::Mempool;
use super::mempool::MempoolProof;
use super::node_context::NodeContext;
use super::node_interface::NodeInterface;
use super::node_interface::NodeResponse;
use super::node_interface::PeerInfo;
use super::node_interface::UserRequest;
use super::peer::create_actors;
use super::peer::Peer;
use super::peer::PeerMessages;
use super::peer::Version;
use super::running_node::RunningNode;
use super::socks::Socks5StreamBuilder;
use super::transport;
use super::transport::TransportProtocol;
use super::UtreexoNodeConfig;
use crate::block_proof::UtreexoProof;
use crate::node_context::PeerId;
use crate::p2p_wire::macros::try_and_debug;
use crate::p2p_wire::macros::try_and_error;
use crate::p2p_wire::macros::try_and_warn;

/// How long before we consider using alternative ways to find addresses,
/// such as hard-coded peers
const HARDCODED_ADDRESSES_GRACE_PERIOD: Duration = Duration::from_secs(60);

/// How long before we try to get addresses from DNS seeds again (5 minutes)
const DNS_SEED_RETRY_PERIOD: Duration = Duration::from_secs(5 * 60);

#[derive(Debug)]
pub enum NodeNotification {
    DnsSeedAddresses(Vec<LocalAddress>),
    FromPeer(u32, PeerMessages),
    FromUser(UserRequest, oneshot::Sender<NodeResponse>),
}

#[derive(Debug, Clone, PartialEq, Hash)]
/// Sent from node to peers, usually to request something
pub enum NodeRequest {
    /// Request the full block data for one or more blocks
    GetBlock(Vec<BlockHash>),

    /// Asks peer for headers
    GetHeaders(Vec<BlockHash>),

    /// Ask for other peers addresses
    GetAddresses,

    /// Asks this peer to shutdown
    Shutdown,

    /// Sends a transaction to peers
    BroadcastTransaction(Txid),

    /// Ask for an unconfirmed transaction
    MempoolTransaction(Txid),

    /// Sends know addresses to our peers
    SendAddresses(Vec<AddrV2Message>),

    /// Requests the peer to send us the utreexo state for a given block
    GetUtreexoState((BlockHash, u32)),

    /// Requests the peer to send us the compact block filters for blocks
    /// starting at a given block hash and height.
    GetFilter((BlockHash, u32)),

    /// Sends a ping to the peer to check if it's alive
    Ping,

    /// Ask for the peer to send us the block proof for a given block
    ///
    /// The first bitmap tells which proof hashes do we want, and the second
    /// which leaf data the peer should send us.
    ///
    /// Proof hashes are the hashes needed to reconstruct the proof, while
    /// leaf data are the actual data of the leaves (i.e., the txouts).
    GetBlockProof((BlockHash, Bitmap, Bitmap)),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) enum InflightRequests {
    /// Requests the peer to send us the next block headers in their main chain
    Headers,

    /// Requests the peer to send us the utreexo state for a given peer
    UtreexoState(PeerId),

    /// Requests the peer to send us the block data for a given block hash
    Blocks(BlockHash),

    /// We've opened a connection with a peer, and are waiting for them to complete the handshake.
    Connect(PeerId),

    /// Requests the peer to send us the compact filters for blocks
    GetFilters,

    /// Requests the peer to send us the utreexo proof for a given block
    UtreexoProof(BlockHash),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ConnectionKind {
    Feeler,
    Regular(ServiceFlags),
    Extra,
}

impl Serialize for ConnectionKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ConnectionKind::Feeler => serializer.serialize_str("feeler"),
            ConnectionKind::Regular(_) => serializer.serialize_str("regular"),
            ConnectionKind::Extra => serializer.serialize_str("extra"),
        }
    }
}

#[derive(Debug, Clone)]
/// Local information kept about each peer
pub struct LocalPeerView {
    /// Average message times from this peer
    ///
    /// This is measured in milliseconds, and it's recorded every time we get
    /// a response from a peer
    pub(crate) message_times: FractionAvg,

    /// The state in which this peer is, e.g., awaiting handshake, ready, banned, etc.
    pub(crate) state: PeerStatus,

    /// An id identifying this peer's address in our address manager
    pub(crate) address_id: u32,

    /// A channel used to send requests to this peer
    pub(crate) channel: UnboundedSender<NodeRequest>,

    /// Services this peer claims to support
    pub(crate) services: ServiceFlags,

    /// A version string that identifies which software this peer is running
    pub(crate) user_agent: String,

    /// This peer's IP address
    pub(crate) address: IpAddr,

    /// The port we used to connect to this peer
    pub(crate) port: u16,

    /// The last time we received a message from this peer
    pub(crate) _last_message: Instant,

    /// The kind of connection we have with this peer
    ///
    /// We use different connections with different goals, e.g., feeler connections,
    /// regular connections, extra connections to find about new tips, etc.
    pub(crate) kind: ConnectionKind,

    /// The latest height this peer has announced to us
    pub(crate) height: u32,

    /// The banscore of this peer
    ///
    /// This is a score kept for each peer, every time this peer misbehaves, we
    /// increase this score. If the score reaches a certain threshold, we ban
    /// the peer.
    pub(crate) banscore: u32,

    /// The transport protocol this peer is using (v1 or v2)
    pub(crate) transport_protocol: TransportProtocol,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RescanStatus {
    InProgress(u32),
    Completed(Instant),
    None,
}

impl Default for RunningNode {
    fn default() -> Self {
        RunningNode {
            last_address_rearrange: Instant::now(),
            last_invs: HashMap::default(),
            inflight_filters: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
/// A block that is currently being downloaded or pending processing
///
/// To download a block, we first request the block itself, and then we
/// request the proof and leaf data for it. This struct holds the data
/// we already have. We may also keep it around, as we may receive blocks
/// out of order, so while we wait for the previous blocks to finish download,
/// we keep the blocks that are already downloaded as an [`InflightBlock`].
pub(crate) struct InflightBlock {
    /// The peer that sent the block
    pub peer: PeerId,

    /// The block itself
    pub block: Block,

    /// The udata associated with the block, if any
    pub leaf_data: Option<Vec<CompactLeafData>>,

    /// The proof associated with the block, if any
    pub proof: Option<Proof>,
}

pub struct NodeCommon<Chain: ChainBackend> {
    // 1. Core Blockchain and Transient Data
    pub(crate) chain: Chain,
    pub(crate) blocks: HashMap<BlockHash, InflightBlock>,
    pub(crate) mempool: Arc<tokio::sync::Mutex<Mempool>>,
    pub(crate) block_filters: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
    pub(crate) last_filter: BlockHash,

    // 2. Peer Management
    pub(crate) peer_id_count: u32,
    pub(crate) peer_ids: Vec<u32>,
    pub(crate) peers: HashMap<u32, LocalPeerView>,
    pub(crate) peer_by_service: HashMap<ServiceFlags, Vec<u32>>,
    pub(crate) max_banscore: u32,
    pub(crate) address_man: AddressMan,
    pub(crate) added_peers: Vec<AddedPeerInfo>,

    // 3. Internal Communication
    pub(crate) node_rx: UnboundedReceiver<NodeNotification>,
    pub(crate) node_tx: UnboundedSender<NodeNotification>,

    // 4. Networking Configuration
    pub(crate) socks5: Option<Socks5StreamBuilder>,
    pub(crate) fixed_peer: Option<LocalAddress>,

    // 5. Time and Event Tracking
    pub(crate) inflight: HashMap<InflightRequests, (u32, Instant)>,
    pub(crate) inflight_user_requests:
        HashMap<UserRequest, (u32, Instant, oneshot::Sender<NodeResponse>)>,
    pub(crate) last_tip_update: Instant,
    pub(crate) last_connection: Instant,
    pub(crate) last_peer_db_dump: Instant,
    pub(crate) last_block_request: u32,
    pub(crate) last_get_address_request: Instant,
    pub(crate) last_broadcast: Instant,
    pub(crate) last_send_addresses: Instant,
    pub(crate) block_sync_avg: FractionAvg,
    pub(crate) last_feeler: Instant,
    pub(crate) startup_time: Instant,
    pub(crate) last_dns_seed_call: Instant,

    // 6. Configuration and Metadata
    pub(crate) config: UtreexoNodeConfig,
    pub(crate) datadir: String,
    pub(crate) network: Network,
    pub(crate) kill_signal: Arc<tokio::sync::RwLock<bool>>,
}

/// The main node that operates while florestad is up.
///
/// [`UtreexoNode`] aims to be modular where `Chain` can be any implementation
/// of a [`ChainBackend`].
///
/// `Context` refers to which state the [`UtreexoNode`] is on, being
/// [`RunningNode`], [`SyncNode`], and [`ChainSelector`]. Defaults to
/// [`RunningNode`] which automatically transitions between contexts.
///
/// [`SyncNode`]: super::sync_node::SyncNode
/// [`ChainSelector`]: super::chain_selector::ChainSelector
pub struct UtreexoNode<Chain: ChainBackend, Context = RunningNode> {
    pub(crate) common: NodeCommon<Chain>,
    pub(crate) context: Context,
}

impl<Chain: ChainBackend, T> Deref for UtreexoNode<Chain, T> {
    fn deref(&self) -> &Self::Target {
        &self.common
    }
    type Target = NodeCommon<Chain>;
}

impl<T, Chain: ChainBackend> DerefMut for UtreexoNode<Chain, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

#[derive(Debug, Clone)]
/// A simple struct of added peers, used to track the ones we added manually by `addnode <ip:port> add` command.
pub struct AddedPeerInfo {
    /// The address of the peer
    pub(crate) address: AddrV2,

    /// The port of the peer
    pub(crate) port: u16,

    /// Whether we should allow V1 fallback for this connection
    pub(crate) v1_fallback: bool,
}

#[derive(Debug, PartialEq, Clone, Copy, Deserialize, Serialize)]
pub enum PeerStatus {
    Awaiting,
    Ready,
    Banned,
}

impl<T, Chain> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
{
    pub fn new(
        config: UtreexoNodeConfig,
        chain: Chain,
        mempool: Arc<Mutex<Mempool>>,
        block_filters: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
        kill_signal: Arc<tokio::sync::RwLock<bool>>,
        address_man: AddressMan,
    ) -> Result<Self, WireError> {
        let (node_tx, node_rx) = unbounded_channel();
        let socks5 = config.proxy.map(Socks5StreamBuilder::new);

        let fixed_peer = config
            .fixed_peer
            .as_ref()
            .map(|address| Self::resolve_connect_host(address, Self::get_port(config.network)))
            .transpose()?;

        Ok(UtreexoNode {
            common: NodeCommon {
                last_dns_seed_call: Instant::now(),
                startup_time: Instant::now(),
                block_sync_avg: FractionAvg::new(0, 0),
                last_filter: chain.get_block_hash(0).unwrap(),
                block_filters,
                inflight: HashMap::new(),
                inflight_user_requests: HashMap::new(),
                peer_id_count: 0,
                peers: HashMap::new(),
                last_block_request: chain.get_validation_index().expect("Invalid chain"),
                chain,
                peer_ids: Vec::new(),
                peer_by_service: HashMap::new(),
                mempool,
                network: config.network,
                node_rx,
                node_tx,
                address_man,
                last_tip_update: Instant::now(),
                last_connection: Instant::now(),
                last_peer_db_dump: Instant::now(),
                last_broadcast: Instant::now(),
                last_feeler: Instant::now(),
                blocks: HashMap::new(),
                last_get_address_request: Instant::now(),
                last_send_addresses: Instant::now(),
                datadir: config.datadir.clone(),
                max_banscore: config.max_banscore,
                socks5,
                fixed_peer,
                config,
                kill_signal,
                added_peers: Vec::new(),
            },
            context: T::default(),
        })
    }

    /// Checks whether some of our inflight requests have timed out.
    ///
    /// This function will check if any of our inflight requests have timed out, and if so,
    /// it will remove them from the inflight list and increase the banscore of the peer that
    /// sent the request. It will also resend the request to another peer.
    pub(crate) fn check_for_timeout(&mut self) -> Result<(), WireError> {
        let now = Instant::now();

        let timed_out_fn = |req: &InflightRequests, time: &Instant| match req {
            InflightRequests::Connect(_)
                if now.duration_since(*time).as_secs() > T::CONNECTION_TIMEOUT =>
            {
                Some(req.clone())
            }

            _ if now.duration_since(*time).as_secs() > T::REQUEST_TIMEOUT => Some(req.clone()),

            _ => None,
        };

        let timed_out = self
            .inflight
            .iter()
            .filter_map(|(req, (_, time))| timed_out_fn(req, time))
            .collect::<Vec<_>>();

        for req in timed_out {
            let Some((peer, _)) = self.inflight.remove(&req) else {
                continue;
            };

            if let InflightRequests::Connect(_) = req {
                // ignore the output as it might fail due to the task being cancelled
                let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
                self.peers.remove(&peer);
                continue;
            }

            debug!("Request timed out: {req:?}");
            self.increase_banscore(peer, 1)?;
            self.redo_inflight_request(req)?;
        }

        Ok(())
    }

    /// Returns a handle to the node interface that we can use to request data from our
    /// node. This struct is thread safe, so we can use it from multiple threads and have
    /// multiple handles. It also doesn't require a mutable reference to the node, or any
    /// synchronization mechanism.
    pub fn get_handle(&self) -> NodeInterface {
        NodeInterface::new(self.common.node_tx.clone())
    }

    /// Handles getpeerinfo requests, returning a list of all connected peers and some useful
    /// information about it.
    fn handle_get_peer_info(&self, responder: oneshot::Sender<NodeResponse>) {
        let mut peers = Vec::new();
        for peer in self.peer_ids.iter() {
            peers.push(self.get_peer_info(peer));
        }

        let peers = peers.into_iter().flatten().collect();
        try_and_debug!("Can't send answer for getpeerinfo request", responder.send(NodeResponse::GetPeerInfo(peers)));
    }

    // Helper function to resolve an IpAddr to AddrV2
    // This is a little bit of a hack while rust-bitcoin
    // do not have an `from` or `into` that do IpAddr <> AddrV2
    fn to_addr_v2(&self, addr: IpAddr) -> AddrV2 {
        match addr {
            IpAddr::V4(addr) => AddrV2::Ipv4(addr),
            IpAddr::V6(addr) => AddrV2::Ipv6(addr),
        }
    }

    /// Handles addnode-RPC `Add` requests, adding a new peer to the `added_peers` list. This means
    /// the peer is marked as a "manually added peer". We then try to connect to it, or retry later.
    pub fn handle_addnode_add_peer(
        &mut self,
        addr: IpAddr,
        port: u16,
        v2_transport: bool,
    ) -> Result<(), WireError> {
        // See https://github.com/bitcoin/bitcoin/blob/8309a9747a8df96517970841b3648937d05939a3/src/net.cpp#L3558
        debug!("Adding node {addr}:{port}");
        let address = self.to_addr_v2(addr);

        // Check if the peer already exists
        if self
            .added_peers
            .iter()
            .any(|info| address == info.address && port == info.port)
        {
            return Err(WireError::PeerAlreadyExists(addr, port));
        }

        // Add a simple reference to the peer
        self.added_peers.push(AddedPeerInfo {
            address,
            port,
            v1_fallback: !v2_transport,
        });

        // Implementation detail for `addnode`: on bitcoin-core, the node doesn't connect immediately
        // after adding a peer, it just adds it to the `added_peers` list. Here we do almost the same,
        // but we do an early connection attempt to the peer, so we can start communicating with.
        self.maybe_open_connection_with_added_peers()
    }

    /// Handles remove node requests, removing a peer from the node.
    ///
    /// Removes a node from the `added_peers` list but does not
    /// disconnect the node if it was already connected.  It only ensures
    /// that the node is no longer treated as a manually added node
    /// (i.e., it won't be reconnected if disconnected).
    ///
    /// If someone wants to remove a peer, it should be done using the
    /// `disconnectnode`.
    pub fn handle_addnode_remove_peer(&mut self, addr: IpAddr, port: u16) -> Result<(), WireError> {
        //
        // (TODO) Make `disconnectnode`` command.
        debug!("Trying to remove peer {addr}:{port}");

        let address = self.to_addr_v2(addr);
        let index = self
            .added_peers
            .iter()
            .position(|info| address == info.address && port == info.port);

        match index {
            Some(peer_id) => self.added_peers.remove(peer_id),
            None => return Err(WireError::PeerNotFoundAtAddress(addr, port)),
        };

        Ok(())
    }

    /// Handles addnode onetry requests, connecting to the node and this will try to connect to the given address and port.
    /// If it's successful, it will add the node to the peers list, but not to the added_peers list (e.g., it won't be reconnected if disconnected).
    pub fn handle_addnode_onetry_peer(
        &mut self,
        addr: IpAddr,
        port: u16,
        v2_transport: bool,
    ) -> Result<(), WireError> {
        debug!("Creating an one-try connection with {addr}:{port}");

        // Check if the peer already exists
        if self
            .peers
            .iter()
            .any(|(_, peer)| addr == peer.address && port == peer.port)
        {
            return Err(WireError::PeerAlreadyExists(addr, port));
        }

        let kind = ConnectionKind::Regular(ServiceFlags::NONE);
        let peer_id = self.peer_id_count;
        let address = LocalAddress::new(
            self.to_addr_v2(addr),
            0,
            AddressState::NeverTried,
            ServiceFlags::NONE,
            port,
            peer_id as usize,
        );

        // Return true if exists or false if anything fails during connection
        // We allow V1 fallback iff the `v2` flag is not set
        self.open_connection(kind, peer_id as usize, address, !v2_transport)
    }

    /// Sends the same request to all connected peers
    ///
    /// This function is best-effort, meaning that some peers may not receive the request if they
    /// are disconnected or if there is an error sending the request. We intentionally won't
    /// propagate the error to the caller, as this would request an early return from the function,
    /// which would prevent us from sending the request to the peers the comes after the first
    /// erroing one.
    fn broadcast_to_peers(&mut self, request: NodeRequest) {
        for peer in self.peers.values() {
            if peer.state != PeerStatus::Ready {
                continue;
            }

            if let Err(err) = peer.channel.send(request.clone()) {
                warn!("Failed to send request to peer {}: {err}", peer.address);
            }
        }
    }

    /// Actually perform the user request
    ///
    /// These are requests made by some consumer of `floresta-wire` using the [`NodeInterface`], and may
    /// be a mempool transaction, a block, or a connection request.
    pub(crate) fn perform_user_request(
        &mut self,
        user_req: UserRequest,
        responder: oneshot::Sender<NodeResponse>,
    ) {
        if self.inflight.len() >= RunningNode::MAX_INFLIGHT_REQUESTS {
            return;
        }

        debug!("Performing user request {user_req:?}");

        let req = match user_req {
            UserRequest::Config => {
                let config = self.common.config.clone();
                let _ = responder.send(NodeResponse::Config(config));

                return;
            }
            UserRequest::Ping => {
                self.broadcast_to_peers(NodeRequest::Ping);
                try_and_debug!("Can't ping peer", responder.send(NodeResponse::Ping(true)));
                return;
            }
            UserRequest::Block(block_hash) => NodeRequest::GetBlock(vec![block_hash]),
            UserRequest::UtreexoProof(block_hash) => {
                NodeRequest::GetBlockProof((block_hash, Bitmap::default(), Bitmap::default()))
            }
            UserRequest::MempoolTransaction(txid) => NodeRequest::MempoolTransaction(txid),
            UserRequest::GetPeerInfo => {
                self.handle_get_peer_info(responder);
                return;
            }
            UserRequest::Add((addr, port, v2transport)) => {
                let node_response = match self.handle_addnode_add_peer(addr, port, v2transport) {
                    Ok(_) => {
                        info!("Added peer {addr}:{port}");
                        NodeResponse::Add(true)
                    }
                    Err(err) => {
                        warn!("{err:?}");
                        NodeResponse::Add(false)
                    }
                };

                let _ = responder.send(node_response);
                return;
            }
            UserRequest::Remove((addr, port)) => {
                let node_response = match self.handle_addnode_remove_peer(addr, port) {
                    Ok(_) => {
                        info!("Removed peer {addr}:{port}");
                        NodeResponse::Remove(true)
                    }
                    Err(err) => {
                        warn!("{err:?}");
                        NodeResponse::Remove(false)
                    }
                };

                let _ = responder.send(node_response);
                return;
            }
            UserRequest::Onetry((addr, port, v2transport)) => {
                let node_response = match self.handle_addnode_onetry_peer(addr, port, v2transport) {
                    Ok(_) => {
                        info!("Connected to peer {addr}:{port}");
                        NodeResponse::Onetry(true)
                    }
                    Err(err) => {
                        warn!("{err:?}");
                        NodeResponse::Onetry(false)
                    }
                };

                let _ = responder.send(node_response);
                return;
            }
        };

        let peer = self.send_to_fastest_peer(req, ServiceFlags::NONE);
        if let Ok(peer) = peer {
            self.inflight_user_requests
                .insert(user_req, (peer, Instant::now(), responder));
        }
    }

    /// Check if this block request is made by a user through the user interface and answer it
    /// back to the user if so.
    ///
    /// This function will return the given block if it isn't a user request. This is to avoid cloning
    /// the block.
    pub(crate) fn check_is_user_block_and_reply(
        &mut self,
        block: Block,
    ) -> Result<Option<Block>, WireError> {
        // If this block is a request made through the user interface, send it back to the
        // user.
        if let Some(request) = self
            .inflight_user_requests
            .remove(&UserRequest::Block(block.block_hash()))
        {
            debug!("answering user request for block {}", block.block_hash());
            request
                .2
                .send(NodeResponse::Block(Some(block)))
                .map_err(|_| WireError::ResponseSendError)?;

            return Ok(None);
        }

        Ok(Some(block))
    }

    /// Returns the fastest peer that supports the given service flags and is initialized.
    fn get_fastest_peer(&self, service: ServiceFlags) -> Option<(&u32, &LocalPeerView)> {
        let should_send = |(_, peer): &(&PeerId, &LocalPeerView)| {
            peer.services.has(service) && peer.state == PeerStatus::Ready
        };

        self.peers
            .iter()
            .filter(should_send)
            .min_by_key(|(_, peer)| peer.message_times.value().round() as u64)
    }

    /// Sends a request to the fastest initialized peer that supports
    /// required service flags.
    pub(crate) fn send_to_fastest_peer(
        &self,
        request: NodeRequest,
        required_service: ServiceFlags,
    ) -> Result<PeerId, WireError> {
        let (peer_id, peer) = self
            .get_fastest_peer(required_service)
            .ok_or(WireError::NoPeersAvailable)?;

        peer.channel.send(request)?;

        Ok(*peer_id)
    }

    pub(crate) fn attach_proof(
        &mut self,
        uproof: UtreexoProof,
        peer: PeerId,
    ) -> Result<(), WireError> {
        debug!("Received utreexo proof for block {}", uproof.block_hash);
        self.inflight
            .remove(&InflightRequests::UtreexoProof(uproof.block_hash));

        let Some(block) = self.blocks.get_mut(&uproof.block_hash) else {
            warn!(
                "Received utreexo proof for block {}, but we don't have it",
                uproof.block_hash
            );
            self.increase_banscore(peer, 5)?;

            return Ok(());
        };

        let proof = Proof {
            hashes: uproof.proof_hashes,
            targets: uproof.targets,
        };

        block.proof = Some(proof);
        block.leaf_data = Some(uproof.leaf_data);

        Ok(())
    }

    pub(crate) fn request_block_proof(
        &mut self,
        block: Block,
        peer: PeerId,
    ) -> Result<(), WireError> {
        let block_hash = block.block_hash();
        self.inflight.remove(&InflightRequests::Blocks(block_hash));

        // Reply and return early if it's a user-requested block. Else continue handling it.
        let Some(block) = self.check_is_user_block_and_reply(block)? else {
            return Ok(());
        };

        if block.txdata.len() == 1 {
            // This is an empty block, so there's no proof for it
            let inflight_block = InflightBlock {
                leaf_data: Some(Vec::new()),
                proof: Some(Proof::default()),
                block,
                peer,
            };

            self.blocks.insert(block_hash, inflight_block);
            return Ok(());
        }

        let inflight_block = InflightBlock {
            leaf_data: None,
            proof: None,
            block,
            peer,
        };

        debug!(
            "Received block {} from peer {}",
            block_hash, inflight_block.peer
        );

        self.send_to_fastest_peer(
            NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
            UTREEXO.into(),
        )?;

        self.inflight.insert(
            InflightRequests::UtreexoProof(block_hash),
            (peer, Instant::now()),
        );

        self.blocks.insert(block_hash, inflight_block);

        Ok(())
    }

    /// Processes ready blocks in order, stopping at the tip or the first missing block/proof.
    /// Call again when new blocks or proofs arrive.
    pub(crate) fn process_pending_blocks(&mut self) -> Result<(), WireError>
    where
        Chain::Error: From<UtreexoLeafError>,
    {
        loop {
            let best_block = self.chain.get_best_block()?.0;
            let next_block = self.chain.get_validation_index()? + 1;
            if next_block > best_block {
                // If we are at the best block, we don't need to process any more blocks
                return Ok(());
            }

            let next_block_hash = self.chain.get_block_hash(next_block)?;

            let Some(block) = self.blocks.get(&next_block_hash) else {
                // If we don't have the next block, we can't process it
                return Ok(());
            };

            if block.proof.is_none() {
                // If the block doesn't have a proof, we can't process it
                return Ok(());
            }

            let start = Instant::now();
            self.process_block(next_block, next_block_hash)?;

            let elapsed = start.elapsed().as_secs();
            self.block_sync_avg.add(elapsed);

            #[cfg(feature = "metrics")]
            {
                use metrics::get_metrics;

                let avg = self.block_sync_avg.value();
                let metrics = get_metrics();
                metrics.avg_block_processing_time.set(avg);
            }
        }
    }

    /// Actually process a block that is ready to be processed.
    ///
    /// This function will take the next block in our chain, process its proof and validate it.
    /// If everything is correct, it will connect the block to our chain.
    fn process_block(&mut self, block_height: u32, block_hash: BlockHash) -> Result<(), WireError>
    where
        Chain::Error: From<UtreexoLeafError>,
    {
        debug!("processing block {block_hash}");

        let inflight_block = self
            .blocks
            .remove(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        let leaf_data = inflight_block
            .leaf_data
            .ok_or(WireError::LeafDataNotFound)?;

        let proof = inflight_block.proof.ok_or(WireError::BlockProofNotFound)?;
        let block = inflight_block.block;
        let peer = inflight_block.peer;

        let (del_hashes, inputs) =
            proof_util::process_proof(&leaf_data, &block.txdata, block_height, |h| {
                self.chain.get_block_hash(h)
            })?;

        if let Err(e) = self.chain.connect_block(&block, proof, inputs, del_hashes) {
            error!(
                "Invalid block {:?} received by peer {} reason: {:?}",
                block.header, peer, e
            );

            if let BlockchainError::BlockValidation(e) = e {
                // Because the proof isn't committed to the block, we can't invalidate
                // it if the proof is invalid. Any other error should cause the block
                // to be invalidated.
                match e {
                    BlockValidationErrors::InvalidCoinbase(_)
                    | BlockValidationErrors::UtxoNotFound(_)
                    | BlockValidationErrors::ScriptValidationError(_)
                    | BlockValidationErrors::NullPrevOut
                    | BlockValidationErrors::EmptyInputs
                    | BlockValidationErrors::EmptyOutputs
                    | BlockValidationErrors::ScriptError
                    | BlockValidationErrors::BlockTooBig
                    | BlockValidationErrors::NotEnoughPow
                    | BlockValidationErrors::TooManyCoins
                    | BlockValidationErrors::BadMerkleRoot
                    | BlockValidationErrors::BadWitnessCommitment
                    | BlockValidationErrors::NotEnoughMoney
                    | BlockValidationErrors::FirstTxIsNotCoinbase
                    | BlockValidationErrors::BadCoinbaseOutValue
                    | BlockValidationErrors::EmptyBlock
                    | BlockValidationErrors::BadBip34
                    | BlockValidationErrors::BIP94TimeWarp
                    | BlockValidationErrors::UnspendableUTXO
                    | BlockValidationErrors::CoinbaseNotMatured => {
                        try_and_error!("Can't invalidate block", self.chain.invalidate_block(block.block_hash()));
                    }
                    BlockValidationErrors::InvalidProof => {}
                    BlockValidationErrors::BlockExtendsAnOrphanChain
                    | BlockValidationErrors::BlockDoesntExtendTip => {
                        // for some reason, we've tried to connect a block that doesn't extend the
                        // tip
                        self.last_block_request = self.chain.get_validation_index().unwrap_or(0);
                        // this is our mistake, don't punish the peer
                        return Ok(());
                    }
                }
            }

            warn!(
                "Block {} from peer {peer} is invalid, banning peer",
                block.block_hash()
            );

            // Disconnect the peer and ban it.
            if let Some(peer) = self.peers.get(&peer).cloned() {
                self.address_man
                    .update_set_state(peer.address_id as usize, AddressState::Banned(T::BAN_TIME));
            }

            self.send_to_peer(peer, NodeRequest::Shutdown)?;
            return Err(WireError::PeerMisbehaving);
        }

        self.last_tip_update = Instant::now();
        Ok(())
    }

    // TODO(@luisschwab): get rid of this once
    // https://github.com/rust-bitcoin/rust-bitcoin/pull/4639 makes it into a release.
    fn get_port(network: Network) -> u16 {
        match network {
            Network::Bitcoin => 8333,
            Network::Signet => 38333,
            Network::Testnet => 18333,
            Network::Testnet4 => 48333,
            Network::Regtest => 18444,
        }
    }

    /// Register a message on `self.inflights` and record the time taken to respond to it.
    ///
    /// We need this information for two purposes:
    /// 1. To calculate the average time taken to respond to messages from peers, which we use
    ///    to select the fastest peer when sending requests.
    /// 2. If `metrics` feature is enabled, we record the time taken for all peers on a histogram,
    ///    and expose it as a prometheus metric.
    pub(crate) fn register_message_time(
        &mut self,
        notification: &PeerMessages,
        peer: PeerId,
    ) -> Option<()> {
        let now = Instant::now();

        let when = match notification {
            PeerMessages::Block(block) => {
                let inflight = self
                    .inflight
                    .get(&InflightRequests::Blocks(block.block_hash()))?;

                inflight.1
            }

            PeerMessages::Ready(_) => {
                let inflight = self.inflight.get(&InflightRequests::Connect(peer))?;
                inflight.1
            }

            PeerMessages::Headers(_) => {
                let inflight = self.inflight.get(&InflightRequests::Headers)?;
                inflight.1
            }

            PeerMessages::BlockFilter((_, _)) => {
                let inflight = self.inflight.get(&InflightRequests::GetFilters)?;
                inflight.1
            }

            PeerMessages::UtreexoState(_) => {
                let inflight = self.inflight.get(&InflightRequests::UtreexoState(peer))?;
                inflight.1
            }

            _ => return None,
        };

        let elapsed = now.duration_since(when);
        if let Some(peer) = self.peers.get_mut(&peer) {
            peer.message_times.add(elapsed.as_millis() as u64);
        }

        #[cfg(feature = "metrics")]
        {
            use metrics::get_metrics;
            let metrics = get_metrics();

            metrics.message_times.observe(elapsed.as_secs_f64());
        }

        Some(())
    }

    /// Resolves a string address into a LocalAddress
    ///
    /// This function should get an address in the format `<address>[<:port>]` and return a
    /// usable [`LocalAddress`]. It can be an ipv4, ipv6 or a hostname. In case of hostnames,
    /// we resolve them using the system's DNS resolver and return an ip address. Errors if
    /// the provided address is invalid, or we can't resolve it.
    ///
    /// TODO: Allow for non-clearnet addresses like onion services and i2p.
    fn resolve_connect_host(
        address: &str,
        default_port: u16,
    ) -> Result<LocalAddress, AddrParseError> {
        // ipv6
        if address.starts_with('[') {
            if !address.contains(']') {
                return Err(AddrParseError::InvalidIpv6);
            }

            let mut split = address.trim_end().split(']');
            let hostname = split.next().ok_or(AddrParseError::InvalidIpv6)?;
            let port = split
                .next()
                .filter(|x| !x.is_empty())
                .map(|port| {
                    port.trim_start_matches(':')
                        .parse()
                        .map_err(|_e| AddrParseError::InvalidPort)
                })
                .transpose()?
                .unwrap_or(default_port);

            let hostname = hostname.trim_start_matches('[');
            let ip = hostname.parse().map_err(|_e| AddrParseError::InvalidIpv6)?;
            return Ok(LocalAddress::new(
                AddrV2::Ipv6(ip),
                0,
                AddressState::NeverTried,
                ServiceFlags::NONE,
                port,
                rand::random(),
            ));
        }

        // ipv4 - it's hard to differentiate between ipv4 and hostname without an actual regex
        // simply try to parse it as an ip address and if it fails, assume it's a hostname

        // this breaks the necessity of feature gate on windows
        let mut address = address;
        if address.is_empty() {
            address = "127.0.0.1"
        }

        let mut split = address.split(':');
        let ip = split
            .next()
            .ok_or(AddrParseError::InvalidIpv4)?
            .parse()
            .map_err(|_e| AddrParseError::InvalidIpv4);

        match ip {
            Ok(ip) => {
                let port = split
                    .next()
                    .map(|port| port.parse().map_err(|_e| AddrParseError::InvalidPort))
                    .transpose()?
                    .unwrap_or(default_port);

                if split.next().is_some() {
                    return Err(AddrParseError::Inconclusive);
                }

                let id = rand::random();
                Ok(LocalAddress::new(
                    AddrV2::Ipv4(ip),
                    0,
                    AddressState::NeverTried,
                    ServiceFlags::NONE,
                    port,
                    id,
                ))
            }

            Err(_) => {
                let mut split = address.split(':');
                let hostname = split.next().ok_or(AddrParseError::InvalidHostname)?;
                let port = split
                    .next()
                    .map(|port| port.parse().map_err(|_e| AddrParseError::InvalidPort))
                    .transpose()?
                    .unwrap_or(default_port);

                if split.next().is_some() {
                    return Err(AddrParseError::Inconclusive);
                }

                let ip = dns_lookup::lookup_host(hostname)
                    .map_err(|_e| AddrParseError::InvalidHostname)?;
                let id = rand::random();
                let ip = match ip[0] {
                    std::net::IpAddr::V4(ip) => AddrV2::Ipv4(ip),
                    std::net::IpAddr::V6(ip) => AddrV2::Ipv6(ip),
                };

                Ok(LocalAddress::new(
                    ip,
                    0,
                    AddressState::NeverTried,
                    ServiceFlags::NONE,
                    port,
                    id,
                ))
            }
        }
    }

    pub(crate) fn get_peer_info(&self, peer: &u32) -> Option<PeerInfo> {
        let peer = self.peers.get(peer)?;
        Some(PeerInfo {
            address: SocketAddr::new(peer.address, peer.port),
            services: peer.services,
            user_agent: peer.user_agent.clone(),
            initial_height: peer.height,
            state: peer.state,
            kind: peer.kind,
            transport_protocol: peer.transport_protocol,
        })
    }

    #[cfg(feature = "metrics")]
    pub(crate) fn update_peer_metrics(&self) {
        use metrics::get_metrics;

        let metrics = get_metrics();
        metrics.peer_count.set(self.peer_ids.len() as f64);
    }

    pub(crate) fn handle_disconnection(&mut self, peer: u32, idx: usize) -> Result<(), WireError> {
        if let Some(p) = self.peers.remove(&peer) {
            std::mem::drop(p.channel);
            if matches!(p.kind, ConnectionKind::Regular(_)) && p.state == PeerStatus::Ready {
                info!("Peer disconnected: {peer}");
            }

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            match p.state {
                PeerStatus::Ready => {
                    self.address_man
                        .update_set_state(idx, AddressState::Tried(now));
                }
                PeerStatus::Awaiting => {
                    self.address_man
                        .update_set_state(idx, AddressState::Failed(now));
                }
                PeerStatus::Banned => {
                    self.address_man
                        .update_set_state(idx, AddressState::Banned(RunningNode::BAN_TIME));
                }
            }
        }

        self.peer_ids.retain(|&id| id != peer);
        for (_, v) in self.peer_by_service.iter_mut() {
            v.retain(|&id| id != peer);
        }

        let inflight = self
            .inflight
            .clone()
            .into_iter()
            .filter(|(_k, v)| v.0 == peer)
            .collect::<Vec<_>>();

        for req in inflight {
            self.inflight.remove(&req.0);
            self.redo_inflight_request(req.0.clone())?;
        }

        #[cfg(feature = "metrics")]
        self.update_peer_metrics();

        Ok(())
    }

    /// Asks all utreexo peers for proofs of blocks that we have, but haven't received proofs
    /// for yet, and don't have any GetProofs inflight. This may be caused by a peer disconnecting
    /// while we didn't have more utreexo peers to redo the request.
    pub(crate) fn ask_for_missed_proofs(&mut self) -> Result<(), WireError> {
        // If we have no peers, we can't ask for proofs
        if !self.has_utreexo_peers() {
            return Ok(());
        }

        let pending_blocks = self
            .blocks
            .iter()
            .filter_map(|(hash, block)| {
                if block.proof.is_some() && block.leaf_data.is_some() {
                    return None;
                }

                if !self
                    .inflight
                    .contains_key(&InflightRequests::UtreexoProof(*hash))
                {
                    return Some(*hash);
                }

                None
            })
            .collect::<Vec<_>>();

        for block_hash in pending_blocks {
            let peer = self.send_to_fastest_peer(
                NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
                service_flags::UTREEXO.into(),
            )?;

            self.inflight.insert(
                InflightRequests::UtreexoProof(block_hash),
                (peer, Instant::now()),
            );
        }

        Ok(())
    }

    pub(crate) fn redo_inflight_request(&mut self, req: InflightRequests) -> Result<(), WireError> {
        match req {
            InflightRequests::UtreexoProof(block_hash) => {
                if !self.has_utreexo_peers() {
                    return Ok(());
                }

                if !self.blocks.contains_key(&block_hash) {
                    // If we don't have the block anymore, we can't ask for the proof
                    return Ok(());
                }

                if self
                    .inflight
                    .contains_key(&InflightRequests::UtreexoProof(block_hash))
                {
                    // If we already have an inflight request for this block, we don't need to redo it
                    return Ok(());
                }

                let peer = self.send_to_fastest_peer(
                    NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
                    service_flags::UTREEXO.into(),
                )?;

                self.inflight.insert(
                    InflightRequests::UtreexoProof(block_hash),
                    (peer, Instant::now()),
                );
            }

            InflightRequests::Blocks(block) => {
                self.request_blocks(vec![block])?;
            }
            InflightRequests::Headers => {
                let peer = self.send_to_fastest_peer(
                    NodeRequest::GetHeaders(vec![]),
                    service_flags::UTREEXO.into(),
                )?;

                self.inflight
                    .insert(InflightRequests::Headers, (peer, Instant::now()));
            }
            InflightRequests::UtreexoState(_) => {
                let peer = self.send_to_fastest_peer(
                    NodeRequest::GetUtreexoState((self.chain.get_block_hash(0).unwrap(), 0)),
                    service_flags::UTREEXO.into(),
                )?;
                self.inflight
                    .insert(InflightRequests::UtreexoState(peer), (peer, Instant::now()));
            }
            InflightRequests::GetFilters => {
                if !self.has_compact_filters_peer() {
                    return Ok(());
                }
                let peer = self.send_to_fastest_peer(
                    NodeRequest::GetFilter((self.chain.get_block_hash(0).unwrap(), 0)),
                    ServiceFlags::COMPACT_FILTERS,
                )?;

                self.inflight
                    .insert(InflightRequests::GetFilters, (peer, Instant::now()));
            }
            InflightRequests::Connect(_) => {
                // We don't need to do anything here
            }
        }

        Ok(())
    }

    fn is_peer_good(peer: &LocalPeerView, needs: ServiceFlags) -> bool {
        if peer.state == PeerStatus::Banned {
            return false;
        }

        peer.services.has(needs)
    }

    pub(crate) fn handle_peer_ready(
        &mut self,
        peer: u32,
        version: &Version,
    ) -> Result<(), WireError> {
        self.inflight.remove(&InflightRequests::Connect(peer));
        if version.kind == ConnectionKind::Feeler {
            self.peers.entry(peer).and_modify(|p| {
                p.state = PeerStatus::Ready;
            });

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            self.send_to_peer(peer, NodeRequest::Shutdown)?;
            self.address_man
                .update_set_service_flag(version.address_id, version.services)
                .update_set_state(version.address_id, AddressState::Tried(now));

            return Ok(());
        }

        if version.kind == ConnectionKind::Extra {
            let locator = self.chain.get_block_locator()?;
            self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;

            self.inflight
                .insert(InflightRequests::Headers, (peer, Instant::now()));

            return Ok(());
        }

        info!(
            "New peer id={} version={} blocks={} services={}",
            version.id, version.user_agent, version.blocks, version.services
        );

        if let Some(peer_data) = self.common.peers.get_mut(&peer) {
            peer_data.state = PeerStatus::Ready;
            peer_data.services = version.services;
            peer_data.user_agent.clone_from(&version.user_agent);
            peer_data.height = version.blocks;
            peer_data.transport_protocol = version.transport_protocol;

            // If this peer doesn't have basic services, we disconnect it
            if let ConnectionKind::Regular(needs) = version.kind {
                if !Self::is_peer_good(peer_data, needs) {
                    info!(
                        "Disconnecting peer {peer} for not having the required services. has={} needs={}", peer_data.services, needs
                    );
                    peer_data.channel.send(NodeRequest::Shutdown)?;
                    self.address_man.update_set_state(
                        version.address_id,
                        AddressState::Tried(
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        ),
                    );

                    self.address_man
                        .update_set_service_flag(version.address_id, version.services);

                    return Ok(());
                }
            };

            if peer_data.services.has(service_flags::UTREEXO.into()) {
                self.common
                    .peer_by_service
                    .entry(service_flags::UTREEXO.into())
                    .or_default()
                    .push(peer);
            }

            if peer_data.services.has(ServiceFlags::COMPACT_FILTERS) {
                self.common
                    .peer_by_service
                    .entry(ServiceFlags::COMPACT_FILTERS)
                    .or_default()
                    .push(peer);
            }

            if peer_data.services.has(ServiceFlags::from(1 << 25)) {
                self.common
                    .peer_by_service
                    .entry(ServiceFlags::from(1 << 25))
                    .or_default()
                    .push(peer);
            }

            // We can request historical blocks from this peer
            if peer_data.services.has(ServiceFlags::NETWORK) {
                self.common
                    .peer_by_service
                    .entry(ServiceFlags::NETWORK)
                    .or_default()
                    .push(peer);
            }

            self.address_man
                .update_set_state(version.address_id, AddressState::Connected)
                .update_set_service_flag(version.address_id, version.services);

            self.peer_ids.push(peer);
        }

        #[cfg(feature = "metrics")]
        self.update_peer_metrics();
        Ok(())
    }

    pub(crate) fn send_to_peer(&self, peer_id: u32, req: NodeRequest) -> Result<(), WireError> {
        if let Some(peer) = &self.peers.get(&peer_id) {
            if peer.state == PeerStatus::Awaiting {
                return Ok(());
            }
            peer.channel.send(req)?;
        }
        Ok(())
    }

    /// Increases the "banscore" of a peer.
    ///
    /// This is a always increasing number that, if reaches our `max_banscore` setting,
    /// will cause our peer to be banned for one BANTIME.
    /// The amount of each increment is given by factor, and it's calibrated for each misbehaving
    /// action that a peer may incur in.
    pub(crate) fn increase_banscore(&mut self, peer_id: u32, factor: u32) -> Result<(), WireError> {
        let Some(peer) = self.common.peers.get_mut(&peer_id) else {
            return Ok(());
        };

        peer.banscore += factor;

        // This peer is misbehaving too often, ban it
        let is_missbehaving = peer.banscore >= self.common.max_banscore;
        // extra peers should be banned immediately
        let is_extra = peer.kind == ConnectionKind::Extra;

        if is_missbehaving || is_extra {
            warn!("banning peer {peer_id} for misbehaving");
            peer.channel.send(NodeRequest::Shutdown)?;
            peer.state = PeerStatus::Banned;
            return Ok(());
        }

        debug!("increasing banscore for peer {peer_id}");

        Ok(())
    }

    pub(crate) fn has_utreexo_peers(&self) -> bool {
        !self
            .peer_by_service
            .get(&service_flags::UTREEXO.into())
            .unwrap_or(&Vec::new())
            .is_empty()
    }

    pub(crate) fn has_compact_filters_peer(&self) -> bool {
        self.peer_by_service
            .get(&ServiceFlags::COMPACT_FILTERS)
            .map(|peers| !peers.is_empty())
            .unwrap_or(false)
    }

    #[inline]
    pub(crate) fn send_to_random_peer(
        &mut self,
        req: NodeRequest,
        required_service: ServiceFlags,
    ) -> Result<u32, WireError> {
        if self.peers.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }

        let peers = match required_service {
            ServiceFlags::NONE => &self.peer_ids,
            _ => self
                .peer_by_service
                .get(&required_service)
                .ok_or(WireError::NoPeersAvailable)?,
        };

        if peers.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }

        let peer = peers
            .choose(&mut rand::thread_rng())
            .expect("infallible: we checked that peers isn't empty");

        self.peers
            .get(peer)
            .ok_or(WireError::NoPeersAvailable)?
            .channel
            .send(req)
            .map_err(WireError::ChannelSend)?;

        Ok(*peer)
    }

    /// Fetch peers from DNS seeds, sending a `NodeNotification` with found ones. Returns
    /// immediately after spawning a background blocking task that performs the work.
    pub(crate) fn get_peers_from_dns(&self) -> Result<(), WireError> {
        let node_sender = self.node_tx.clone();
        let network = self.network;

        let proxy_addr = self.socks5.as_ref().map(|proxy| {
            let addr = proxy.address;
            info!("Asking for DNS peers via the SOCKS5 proxy: {addr}");
            addr
        });

        tokio::task::spawn_blocking(move || {
            let dns_seeds = floresta_chain::get_chain_dns_seeds(network);
            let mut addresses = Vec::new();

            let default_port = Self::get_port(network);
            for seed in dns_seeds {
                let _addresses = AddressMan::get_seeds_from_dns(&seed, default_port, proxy_addr);

                if let Ok(_addresses) = _addresses {
                    addresses.extend(_addresses);
                }
            }

            info!(
                "Fetched {} peer addresses from all DNS seeds",
                addresses.len()
            );
            node_sender
                .send(NodeNotification::DnsSeedAddresses(addresses))
                .unwrap();
        });

        Ok(())
    }

    pub(crate) fn init_peers(&mut self) -> Result<(), WireError> {
        let anchors = self.common.address_man.start_addr_man(self.datadir.clone());

        if !self.config.disable_dns_seeds {
            self.get_peers_from_dns()?;
            self.last_dns_seed_call = Instant::now();
        }

        for address in anchors {
            self.open_connection(
                ConnectionKind::Regular(UTREEXO.into()),
                address.id,
                address,
                // Using V1 transport fallback as utreexo nodes have limited support
                true,
            )?;
        }

        Ok(())
    }

    pub(crate) fn shutdown(&mut self) {
        info!("Shutting down node...");
        try_and_warn!("Couldn't save anchor utreexo peers", self.save_utreexo_peers());
        for peer in self.peer_ids.iter() {
            try_and_debug!("Can't ask peer to stop", self.send_to_peer(*peer, NodeRequest::Shutdown));
        }
        try_and_debug!("Can't save peers to disk", self.save_peers());
        try_and_debug!("Failure to flush our chain", self.chain.flush());
    }

    pub(crate) async fn handle_broadcast(&self) -> Result<(), WireError> {
        for (_, peer) in self.peers.iter() {
            if peer.services.has(ServiceFlags::from(1 << 24)) {
                continue;
            }

            let transactions = self.chain.get_unbroadcasted();

            for transaction in transactions {
                let txid = transaction.compute_txid();
                let mut mempool = self.mempool.lock().await;

                if self.network == Network::Regtest {
                    match mempool.try_prove(&transaction, &self.chain) {
                        Ok(proof) => {
                            let MempoolProof {
                                proof,
                                target_hashes,
                                leaves,
                            } = proof;

                            let leaves = transaction
                                .input
                                .iter()
                                .map(|input| input.previous_output)
                                .zip(leaves.into_iter())
                                .collect::<Vec<_>>();

                            let targets = proof.targets.clone();
                            try_and_error!("Can't accept transaction to mempool", mempool.accept_to_mempool(
                                transaction,
                                proof,
                                &leaves,
                                &target_hashes,
                                &targets,
                            ));
                        }
                        Err(e) => {
                            error!(
                                "Could not prove tx {} because: {:?}",
                                transaction.compute_txid(),
                                e
                            );
                        }
                    }

                    peer.channel
                        .send(NodeRequest::BroadcastTransaction(txid))
                        .map_err(WireError::ChannelSend)?;
                }

                let stale = self.mempool.lock().await.get_stale();
                for tx in stale {
                    peer.channel
                        .send(NodeRequest::BroadcastTransaction(tx))
                        .map_err(WireError::ChannelSend)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn ask_for_addresses(&mut self) -> Result<(), WireError> {
        let _ = self.send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)?;
        Ok(())
    }

    pub(crate) fn save_peers(&self) -> Result<(), WireError> {
        self.address_man
            .dump_peers(&self.datadir)
            .map_err(WireError::Io)
    }

    /// Saves the utreexo peers to disk so we can reconnect with them later
    pub(crate) fn save_utreexo_peers(&self) -> Result<(), WireError> {
        let peers: &Vec<u32> = self
            .peer_by_service
            .get(&service_flags::UTREEXO.into())
            .ok_or(WireError::NoUtreexoPeersAvailable)?;
        let peers_usize: Vec<usize> = peers.iter().map(|&peer| peer as usize).collect();
        if peers_usize.is_empty() {
            warn!("No connected Utreexo peers to save to disk");
            return Ok(());
        }
        info!("Saving utreexo peers to disk...");
        self.address_man
            .dump_utreexo_peers(&self.datadir, &peers_usize)
            .map_err(WireError::Io)
    }

    /// Checks whether is necessary to fetch peers from DNS seeds.
    ///
    /// If the last DNS lookup was more than 5 minutes ago, and we still
    /// don't have any connected peers, we retry another DNS lookup.
    fn maybe_ask_for_dns_peers(&mut self) {
        if self.config.disable_dns_seeds {
            return;
        }

        if !self.peers.is_empty() {
            return;
        }

        let last_dns_request = self.last_dns_seed_call.elapsed();
        // don't ask too often
        if last_dns_request < DNS_SEED_RETRY_PERIOD {
            return;
        }

        self.last_dns_seed_call = Instant::now();

        info!("We've been running for a while and we don't have any peers, asking for DNS peers");
        try_and_debug!("Can't get peers from DNS seeds", self.get_peers_from_dns());
    }

    /// If we don't have any peers, we use the hardcoded addresses.
    ///
    ///
    /// This is only done if we don't have any peers for a long time, or we
    /// can't find a Utreexo peer in a context we need them. This function
    /// won't do anything if `--connect` was used
    fn maybe_use_hadcoded_addresses(&mut self, needs_utreexo: bool) {
        if self.fixed_peer.is_some() {
            return;
        }

        let has_peers = !self.peers.is_empty();
        // Return if we have peers and utreexo isn't needed OR we have utreexo peers
        if has_peers && (!needs_utreexo || self.has_utreexo_peers()) {
            return;
        }

        let mut wait = HARDCODED_ADDRESSES_GRACE_PERIOD;
        if needs_utreexo {
            // This gives some extra time for the node to try connections after chain selection
            wait += Duration::from_secs(60);
        }

        if self.startup_time.elapsed() < wait {
            return;
        }

        info!("No peers found, using hardcoded addresses");
        let net = self.network;
        self.address_man.add_fixed_addresses(net);
    }

    pub(crate) fn maybe_open_connection_with_added_peers(&mut self) -> Result<(), WireError> {
        if self.added_peers.is_empty() {
            return Ok(());
        }
        let peers_count = self.peer_id_count;
        for added_peer in self.added_peers.clone() {
            let matching_peer = self.peers.values().find(|peer| {
                self.to_addr_v2(peer.address) == added_peer.address && peer.port == added_peer.port
            });

            if matching_peer.is_none() {
                let address = LocalAddress::new(
                    added_peer.address.clone(),
                    0,
                    AddressState::Tried(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    ),
                    ServiceFlags::NONE,
                    added_peer.port,
                    peers_count as usize,
                );

                // Finally, open the connection with the node
                self.open_connection(
                    ConnectionKind::Regular(ServiceFlags::NONE),
                    peers_count as usize,
                    address,
                    added_peer.v1_fallback,
                )?
            }
        }
        Ok(())
    }

    pub(crate) fn maybe_open_connection(
        &mut self,
        required_service: ServiceFlags,
    ) -> Result<(), WireError> {
        // If the user passes in a `--connect` cli argument, we only connect with
        // that particular peer.
        if self.fixed_peer.is_some() && !self.peers.is_empty() {
            return Ok(());
        }

        // If we've tried getting some connections, but the addresses we have are not
        // working. Try getting some more addresses from DNS
        self.maybe_ask_for_dns_peers();
        let needs_utreexo = required_service.has(service_flags::UTREEXO.into());
        self.maybe_use_hadcoded_addresses(needs_utreexo);

        // try to connect with manually added peers
        self.maybe_open_connection_with_added_peers()?;

        let connection_kind = ConnectionKind::Regular(required_service);
        if self.peers.len() < T::MAX_OUTGOING_PEERS {
            self.create_connection(connection_kind)?;
        }

        Ok(())
    }

    pub(crate) fn open_feeler_connection(&mut self) -> Result<(), WireError> {
        // No feeler if `-connect` is set
        if self.fixed_peer.is_some() {
            return Ok(());
        }
        self.create_connection(ConnectionKind::Feeler)?;
        Ok(())
    }

    pub(crate) fn request_blocks(&mut self, blocks: Vec<BlockHash>) -> Result<(), WireError> {
        let should_request = |block: &BlockHash| {
            let is_inflight = self
                .inflight
                .contains_key(&InflightRequests::Blocks(*block));
            let is_pending = self.blocks.contains_key(block);

            !(is_inflight || is_pending)
        };

        let blocks: Vec<_> = blocks.into_iter().filter(should_request).collect();
        // if there's no block to request, don't propagate any message
        if blocks.is_empty() {
            return Ok(());
        }

        let peer = self
            .send_to_fastest_peer(NodeRequest::GetBlock(blocks.clone()), ServiceFlags::NETWORK)?;

        for block in blocks.iter() {
            self.inflight
                .insert(InflightRequests::Blocks(*block), (peer, Instant::now()));
        }

        Ok(())
    }

    pub(crate) fn create_connection(&mut self, kind: ConnectionKind) -> Result<(), WireError> {
        let required_services = match kind {
            ConnectionKind::Regular(services) => services,
            _ => ServiceFlags::NONE,
        };

        let address = self
            .fixed_peer
            .as_ref()
            .map(|addr| (0, addr.clone()))
            .or_else(|| {
                self.address_man.get_address_to_connect(
                    required_services,
                    matches!(kind, ConnectionKind::Feeler),
                )
            });

        let Some((peer_id, address)) = address else {
            // No peers with the desired services are known, load hardcoded addresses
            let net = self.network;
            self.address_man.add_fixed_addresses(net);

            return Err(WireError::NoAddressesAvailable);
        };

        debug!("attempting connection with address={address:?} kind={kind:?}",);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Defaults to failed, if the connection is successful, we'll update the state
        self.address_man
            .update_set_state(peer_id, AddressState::Failed(now));

        // Don't connect to the same peer twice
        let is_connected = |(_, peer_addr): (_, &LocalPeerView)| {
            peer_addr.address == address.get_net_address() && peer_addr.port == address.get_port()
        };

        if self.common.peers.iter().any(is_connected) {
            return Err(WireError::PeerAlreadyExists(
                address.get_net_address(),
                address.get_port(),
            ));
        }

        // We allow V1 fallback only if the cli option was set, it's a --connect peer
        // or if we are connecting to a utreexo peer, since utreexod doesn't support V2 yet.
        let is_fixed = self.fixed_peer.is_some();
        let allow_v1 = self.config.allow_v1_fallback
            || kind == ConnectionKind::Regular(UTREEXO.into())
            || is_fixed;

        self.open_connection(kind, peer_id, address, allow_v1)?;

        Ok(())
    }

    /// Opens a new connection that doesn't require a proxy and includes the functionalities of create_outbound_connection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn open_non_proxy_connection(
        kind: ConnectionKind,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
        mempool: Arc<Mutex<Mempool>>,
        network: Network,
        node_tx: UnboundedSender<NodeNotification>,
        user_agent: String,
        allow_v1_fallback: bool,
    ) -> Result<(), WireError> {
        let address = (address.get_net_address(), address.get_port());

        let (transport_reader, transport_writer, transport_protocol) =
            transport::connect(address, network, allow_v1_fallback).await?;

        let (cancellation_sender, cancellation_receiver) = oneshot::channel();
        let (actor_receiver, actor) = create_actors(transport_reader);
        tokio::spawn(async move {
            tokio::select! {
                _ = cancellation_receiver => {}
                _ = actor.run() => {}
            }
        });

        // Use create_peer function instead of manually creating the peer
        Peer::<WriteHalf>::create_peer(
            peer_id_count,
            mempool,
            node_tx.clone(),
            requests_rx,
            peer_id,
            kind,
            actor_receiver,
            transport_writer,
            user_agent,
            cancellation_sender,
            transport_protocol,
        );

        Ok(())
    }

    /// Opens a connection through a socks5 interface
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn open_proxy_connection(
        proxy: SocketAddr,
        kind: ConnectionKind,
        mempool: Arc<Mutex<Mempool>>,
        network: Network,
        node_tx: UnboundedSender<NodeNotification>,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
        user_agent: String,
        allow_v1_fallback: bool,
    ) -> Result<(), WireError> {
        let (transport_reader, transport_writer, transport_protocol) =
            transport::connect_proxy(proxy, address, network, allow_v1_fallback).await?;

        let (cancellation_sender, cancellation_receiver) = oneshot::channel();
        let (actor_receiver, actor) = create_actors(transport_reader);
        tokio::spawn(async move {
            tokio::select! {
                _ = cancellation_receiver => {}
                _ = actor.run() => {}
            }
        });

        Peer::<WriteHalf>::create_peer(
            peer_id_count,
            mempool,
            node_tx,
            requests_rx,
            peer_id,
            kind,
            actor_receiver,
            transport_writer,
            user_agent,
            cancellation_sender,
            transport_protocol,
        );
        Ok(())
    }

    /// Creates a new outgoing connection with `address`.
    ///
    /// `kind` may or may not be a [`ConnectionKind::Feeler`], a special connection type
    /// that is used to learn about good peers, but are not kept after handshake
    /// (others are [`ConnectionKind::Regular`] and [`ConnectionKind::Extra`]).
    ///
    /// We will always try to open a V2 connection first. If the `allow_v1_fallback` is set,
    /// we may retry the connection with the old V1 protocol if the V2 connection fails.
    /// We don't open the connection here, we create a [`Peer`] actor that will try to open
    /// a connection with the given address and kind. If it succeeds, it will send a
    /// [`PeerMessages::Ready`] to the node after handshaking.
    pub(crate) fn open_connection(
        &mut self,
        kind: ConnectionKind,
        peer_id: usize,
        address: LocalAddress,
        allow_v1_fallback: bool,
    ) -> Result<(), WireError> {
        let (requests_tx, requests_rx) = unbounded_channel();
        if let Some(ref proxy) = self.socks5 {
            spawn(timeout(
                Duration::from_secs(10),
                Self::open_proxy_connection(
                    proxy.address,
                    kind,
                    self.mempool.clone(),
                    self.network,
                    self.node_tx.clone(),
                    peer_id,
                    address.clone(),
                    requests_rx,
                    self.peer_id_count,
                    self.config.user_agent.clone(),
                    allow_v1_fallback,
                ),
            ));
        } else {
            spawn(timeout(
                Duration::from_secs(10),
                Self::open_non_proxy_connection(
                    kind,
                    peer_id,
                    address.clone(),
                    requests_rx,
                    self.peer_id_count,
                    self.mempool.clone(),
                    self.network,
                    self.node_tx.clone(),
                    self.config.user_agent.clone(),
                    allow_v1_fallback,
                ),
            ));
        }

        let peer_count: u32 = self.peer_id_count;

        self.inflight.insert(
            InflightRequests::Connect(peer_count),
            (peer_count, Instant::now()),
        );

        self.peers.insert(
            peer_count,
            LocalPeerView {
                message_times: FractionAvg::new(0, 0),
                address: address.get_net_address(),
                port: address.get_port(),
                user_agent: "".to_string(),
                state: PeerStatus::Awaiting,
                channel: requests_tx,
                services: ServiceFlags::NONE,
                _last_message: Instant::now(),
                kind,
                address_id: peer_id as u32,
                height: 0,
                banscore: 0,
                // Will be downgraded to V1 if the V2 handshake fails, and we allow fallback
                transport_protocol: TransportProtocol::V2,
            },
        );

        match kind {
            ConnectionKind::Feeler => self.last_feeler = Instant::now(),
            ConnectionKind::Regular(_) => self.last_connection = Instant::now(),
            _ => {}
        }

        // Increment peer_id count and the list of peer ids
        // so we can get information about connected or
        // added peers when requesting with getpeerinfo command
        self.peer_id_count += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use floresta_chain::pruned_utreexo::partial_chain::PartialChainState;

    use crate::node::UtreexoNode;
    use crate::running_node::RunningNode;

    fn check_address_resolving(address: &str, port: u16, should_succeed: bool, description: &str) {
        let result =
            UtreexoNode::<PartialChainState, RunningNode>::resolve_connect_host(address, port);
        if should_succeed {
            assert!(result.is_ok(), "Failed: {description}");
        } else {
            assert!(result.is_err(), "Unexpected success: {description}");
        }
    }

    #[test]
    fn test_parse_address() {
        // IPv6 Tests
        check_address_resolving("[::1]", 8333, true, "Valid IPv6 without port");
        check_address_resolving("[::1", 8333, false, "Invalid IPv6 format");
        check_address_resolving("[::1]:8333", 8333, true, "Valid IPv6 with port");
        check_address_resolving(
            "[::1]:8333:8333",
            8333,
            false,
            "Invalid IPv6 with multiple ports",
        );

        // IPv4 Tests
        check_address_resolving("127.0.0.1", 8333, true, "Valid IPv4 without port");
        check_address_resolving("321.321.321.321", 8333, false, "Invalid IPv4 format");
        check_address_resolving("127.0.0.1:8333", 8333, true, "Valid IPv4 with port");
        check_address_resolving(
            "127.0.0.1:8333:8333",
            8333,
            false,
            "Invalid IPv4 with multiple ports",
        );

        // Hostname Tests
        check_address_resolving("example.com", 8333, true, "Valid hostname without port");
        check_address_resolving("example", 8333, false, "Invalid hostname");
        check_address_resolving("example.com:8333", 8333, true, "Valid hostname with port");
        check_address_resolving(
            "example.com:8333:8333",
            8333,
            false,
            "Invalid hostname with multiple ports",
        );

        // Edge Cases
        // This could fail on windows but doesnt since inside `resolve_connect_host` we specificate empty addresses as localhost for all OS`s.
        check_address_resolving("", 8333, true, "Empty string address");
        check_address_resolving(
            " 127.0.0.1:8333 ",
            8333,
            false,
            "Address with leading/trailing spaces",
        );
        check_address_resolving("127.0.0.1:0", 0, true, "Valid address with port 0");
        check_address_resolving(
            "127.0.0.1:65535",
            65535,
            true,
            "Valid address with maximum port",
        );
        check_address_resolving(
            "127.0.0.1:65536",
            65535,
            false,
            "Valid address with out-of-range port",
        )
    }
}
