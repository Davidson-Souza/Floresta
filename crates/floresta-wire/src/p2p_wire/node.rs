//! Main file for this blockchain. A node is the central task that runs and handles important
//! events, such as new blocks, peer connection/disconnection, new addresses, etc.
//! A node should not care about peer-specific messages, peers'll handle things like pings.
use std::collections::HashMap;
use std::fmt::Debug;
use std::net::IpAddr;
use std::net::SocketAddr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::time::timeout;
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tokio::spawn;
use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::utreexo::UtreexoBlock;
use bitcoin::p2p::ServiceFlags;
use bitcoin::BlockHash;
use bitcoin::Txid;
use floresta_chain::pruned_utreexo::chainparams::get_chain_dns_seeds;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::Network;
use futures::Future;
use log::info;
use log::warn;

use super::address_man::AddressMan;
use super::address_man::AddressState;
use super::address_man::LocalAddress;
use super::error::WireError;
use super::mempool::Mempool;
use super::node_context::NodeContext;
use super::node_interface::NodeInterface;
use super::node_interface::PeerInfo;
use super::node_interface::UserRequest;
use super::peer::Peer;
use super::peer::PeerMessages;
use super::peer::Version;
use super::running_node::RunningNode;
use super::socks::Socks5Addr;
use super::socks::Socks5Error;
use super::socks::Socks5StreamBuilder;
use crate::node_context::PeerId;


#[derive(Debug)]
pub enum NodeNotification {
    FromPeer(u32, PeerMessages),
}

#[derive(Debug, Clone, PartialEq, Hash)]
/// Sent from node to peers, usually to request something
pub enum NodeRequest {
    /// Get this block's data
    GetBlock((Vec<BlockHash>, bool)),
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
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) enum InflightRequests {
    Headers,
    Blocks(BlockHash),
    RescanBlock(BlockHash),
    UserRequest(UserRequest),
    Connect(u32),
}

#[derive(Debug, Clone)]
pub struct LocalPeerView {
    state: PeerStatus,
    pub(crate) address_id: u32,
    channel: UnboundedSender<NodeRequest>,
    services: ServiceFlags,
    user_agent: String,
    address: IpAddr,
    port: u16,
    _last_message: Instant,
    feeler: bool,
    height: u32,
    banscore: u32,
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
            last_rescan_request: RescanStatus::None,
            last_feeler: Instant::now(),
            last_address_rearrange: Instant::now(),
            user_requests: Arc::new(NodeInterface {
                requests: Mutex::new(Vec::new()),
            }),
        }
    }
}

pub struct NodeCommon<Chain: BlockchainInterface + UpdatableChainstate> {
    pub(crate) peer_id_count: u32,
    pub(crate) last_headers_request: Instant,
    pub(crate) last_tip_update: Instant,
    pub(crate) last_connection: Instant,
    pub(crate) last_peer_db_dump: Instant,
    pub(crate) last_broadcast: Instant,
    pub(crate) last_send_addresses: Instant,
    pub(crate) last_block_request: u32,
    pub(crate) network: Network,
    pub(crate) last_get_address_request: Instant,
    pub(crate) utreexo_peers: Vec<u32>,
    pub(crate) peer_ids: Vec<u32>,
    pub(crate) peers: HashMap<u32, LocalPeerView>,
    pub(crate) chain: Arc<Chain>,
    pub(crate) blocks: HashMap<BlockHash, (PeerId, UtreexoBlock)>,
    pub(crate) inflight: HashMap<InflightRequests, (u32, Instant)>,
    pub(crate) node_rx: UnboundedReceiver<NodeNotification>,
    pub(crate) node_tx: UnboundedSender<NodeNotification>,
    pub(crate) mempool: Arc<RwLock<Mempool>>,
    pub(crate) datadir: String,
    pub(crate) address_man: AddressMan,
    pub(crate) max_banscore: u32,
    pub(crate) socks5: Option<Socks5StreamBuilder>,
    pub(crate) fixed_peer: Option<LocalAddress>,
}

pub struct UtreexoNode<Context, Chain: BlockchainInterface + UpdatableChainstate>(
    pub(crate) NodeCommon<Chain>,
    pub(crate) Context,
);

impl<Chain: BlockchainInterface + UpdatableChainstate, T> Deref for UtreexoNode<T, Chain> {
    fn deref(&self) -> &Self::Target {
        &self.0
    }
    type Target = NodeCommon<Chain>;
}

impl<T, Chain: BlockchainInterface + UpdatableChainstate> DerefMut for UtreexoNode<T, Chain> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum PeerStatus {
    Awaiting,
    Ready,
}

impl<T, Chain> UtreexoNode<T, Chain>
where
    T: 'static + Default + NodeContext,
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    pub fn new(
        chain: Arc<Chain>,
        mempool: Arc<RwLock<Mempool>>,
        network: Network,
        datadir: String,
        proxy: Option<SocketAddr>,
        max_banscore: Option<u32>,
        fixed_peer: Option<LocalAddress>,
    ) -> Self {
        let (node_tx, node_rx) = unbounded_channel();
        let socks5 = proxy.map(Socks5StreamBuilder::new);
        UtreexoNode(
            NodeCommon {
                inflight: HashMap::new(),
                peer_id_count: 0,
                peers: HashMap::new(),
                last_block_request: chain.get_validation_index().expect("Invalid chain"),
                chain,
                peer_ids: Vec::new(),
                utreexo_peers: Vec::new(),
                mempool,
                network,
                node_rx,
                node_tx,
                address_man: AddressMan::default(),
                last_headers_request: Instant::now(),
                last_tip_update: Instant::now(),
                last_connection: Instant::now(),
                last_peer_db_dump: Instant::now(),
                last_broadcast: Instant::now(),
                blocks: HashMap::new(),
                last_get_address_request: Instant::now(),
                last_send_addresses: Instant::now(),
                datadir,
                socks5,
                max_banscore: max_banscore.unwrap_or(50),
                fixed_peer,
            },
            T::default(),
        )
    }

    pub(crate) fn get_peer_info(&self, peer: &u32) -> Option<PeerInfo> {
        let peer = self.peers.get(peer)?;
        Some(PeerInfo {
            address: format!("{}:{}", peer.address, peer.port),
            services: peer.services.to_string(),
            user_agent: peer.user_agent.clone(),
            initial_height: peer.height,
        })
    }
    pub(crate) fn handle_disconnection(&mut self, peer: u32, idx: usize) -> Result<(), WireError> {
        if let Some(p) = self.peers.remove(&peer) {
            p.channel.closed();
            if !p.feeler && p.state == PeerStatus::Ready {
                info!("Peer disconnected: {}", peer);
            }
        }
        self.peer_ids.retain(|&id| id != peer);
        self.utreexo_peers.retain(|&id| id != peer);

        self.address_man.update_set_state(
            idx,
            AddressState::Tried(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
        );
        Ok(())
    }
    pub(crate) async fn handle_peer_ready(
        &mut self,
        peer: u32,
        version: &Version,
    ) -> Result<(), WireError> {
        if version.feeler {
            self.send_to_peer(peer, NodeRequest::Shutdown).await?;
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
        info!(
            "New peer id={} version={} blocks={}",
            version.id, version.user_agent, version.blocks
        );
        self.inflight.remove(&InflightRequests::Connect(peer));

        if let Some(peer_data) = self.peers.get_mut(&peer) {
            // This peer doesn't have basic services, so we disconnect it
            if !version
                .services
                .has(ServiceFlags::NETWORK | ServiceFlags::WITNESS)
            {
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                self.peers.remove(&peer);
                return Ok(());
            }
            peer_data.state = PeerStatus::Ready;
            peer_data.services = version.services;
            peer_data.user_agent = version.user_agent.clone();
            peer_data.height = version.blocks;
            self.address_man
                .update_set_state(version.address_id, AddressState::Connected)
                .update_set_service_flag(version.address_id, version.services);
            if version.services.has(ServiceFlags::from(1 << 24)) {
                self.utreexo_peers.push(peer);
            }
            self.peer_ids.push(peer);
        }
        Ok(())
    }

    pub(crate) fn get_default_port(&self) -> u16 {
        match self.network {
            Network::Bitcoin => 8333,
            Network::Testnet => 18333,
            Network::Signet => 38333,
            Network::Regtest => 18444,
        }
    }

    pub(crate) async fn send_to_peer(
        &self,
        peer_id: u32,
        req: NodeRequest,
    ) -> Result<(), WireError> {
        if let Some(peer) = &self.peers.get(&peer_id) {
            if peer.state == PeerStatus::Ready {
                peer.channel
                    .send(req)
                    .map_err(WireError::ChannelSend)?;
            }
        }
        Ok(())
    }

    /// Increses the "banscore" of a peer.
    ///
    /// This is a always increasing number that, if reaches our `max_banscore` setting,
    /// will cause our peer to be banned for one BANTIME.
    /// The amount of each increment is given by factor, and it's callibrated for each misbehaving
    /// action that a peer may incur in.
    pub(crate) async fn increase_banscore(
        &mut self,
        peer_id: u32,
        factor: u32,
    ) -> Result<(), WireError> {
        let Some(peer) = self.0.peers.get_mut(&peer_id) else {
            return Ok(());
        };
        peer.banscore += factor;
        // This peer is misbehaving too often, ban it
        if peer.banscore >= self.0.max_banscore {
            warn!("banning peer {} for misbehaving", peer_id);
            let _ = peer.channel.send(NodeRequest::Shutdown);
            self.0.address_man.update_set_state(
                peer.address_id as usize,
                AddressState::Banned(RunningNode::BAN_TIME),
            );

            // remove all inflight requests for that peer
            let peer_req = self
                .inflight
                .keys()
                .filter(|k| self.inflight.get(k).unwrap().0 == peer_id)
                .cloned()
                .collect::<Vec<_>>();
            for peer in peer_req {
                self.inflight.remove_entry(&peer);
            }
        }

        Ok(())
    }

    #[inline]
    pub(crate) async fn send_to_random_peer(
        &mut self,
        req: NodeRequest,
        required_services: ServiceFlags,
    ) -> Result<u32, WireError> {
        if self.peers.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }
        for _ in 0..10 {
            let idx = if required_services.has(ServiceFlags::UTREEXO) {
                if self.utreexo_peers.is_empty() {
                    return Err(WireError::NoUtreexoPeersAvailable);
                }
                let idx = rand::random::<usize>() % self.utreexo_peers.len();
                *self
                    .utreexo_peers
                    .get(idx)
                    .expect("node is in the interval 0..utreexo_peers.len(), but is not here?")
            } else {
                if self.peer_ids.is_empty() {
                    return Err(WireError::NoPeersAvailable);
                }
                let idx = rand::random::<usize>() % self.peer_ids.len();
                *self
                    .peer_ids
                    .get(idx)
                    .expect("node is in the interval 0..utreexo_peers.len(), but is not here?")
            };
            if let Some(peer) = self.peers.get(&idx) {
                if peer.state != PeerStatus::Ready {
                    continue;
                }
                peer.channel
                    .send(req.clone())
                    .map_err(WireError::ChannelSend)?;
                return Ok(idx);
            }
        }
        self.create_connection(false).await;
        Err(WireError::NoPeerToSendRequest)
    }

    pub(crate) async fn init_peers(&mut self) -> Result<(), WireError> {
        let anchors = self
            .0
            .address_man
            .start_addr_man(
                self.datadir.clone(),
                self.get_default_port(),
                self.network,
                &get_chain_dns_seeds(self.network),
            )
            .map_err(WireError::Io)?;
        for address in anchors {
            self.open_connection(false, address.id, address).await;
        }
        Ok(())
    }

    pub(crate) async fn shutdown(&mut self) {
        info!("Shutting down node");
        for peer in self.peer_ids.iter() {
            try_and_log!(self.send_to_peer(*peer, NodeRequest::Shutdown).await);
        }
        try_and_log!(self.save_peers());
        try_and_log!(self.chain.flush());
    }
    pub(crate) async fn ask_block(&mut self) -> Result<(), WireError> {
        let blocks = self.get_blocks_to_download()?;
        self.request_blocks(blocks).await
    }
    pub(crate) async fn handle_broadcast(&self) -> Result<(), WireError> {
        for (_, peer) in self.peers.iter() {
            if peer.services.has(ServiceFlags::from(1 << 24)) {
                continue;
            }

            let transactions = self.chain.get_unbroadcasted();

            for transaction in transactions {
                let txid = transaction.txid();
                self.mempool.write().await.accept_to_mempool(transaction);
                peer.channel
                    .send(NodeRequest::BroadcastTransaction(txid))
                    .map_err(WireError::ChannelSend)?;
            }
            let stale = self.mempool.write().await.get_stale();
            for tx in stale {
                peer.channel
                    .send(NodeRequest::BroadcastTransaction(tx))
                    .map_err(WireError::ChannelSend)?;
            }
        }
        Ok(())
    }
    pub(crate) async fn ask_for_addresses(&mut self) -> Result<(), WireError> {
        let _ = self
            .send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)
            .await?;
        Ok(())
    }
    pub(crate) fn save_peers(&self) -> Result<(), WireError> {
        self.address_man
            .dump_peers(&self.datadir)
            .map_err(WireError::Io)
    }
    pub(crate) fn get_blocks_to_download(&mut self) -> Result<Vec<BlockHash>, WireError> {
        let mut blocks = Vec::new();
        let tip = self.chain.get_height()?;

        for i in (self.last_block_request + 1)..=(self.last_block_request + 10) {
            if i > tip {
                break;
            }
            self.last_block_request += 1;
            let hash = self.chain.get_block_hash(i)?;
            blocks.push(hash);
        }

        Ok(blocks)
    }

    pub(crate) async fn maybe_open_connection(&mut self) -> Result<(), WireError> {
        // If the user passes in a `--connect` cli argument, we only connect with
        // that particular peer.
        if self.fixed_peer.is_some() && !self.peers.is_empty() {
            return Ok(());
        }
        if self.peers.len() < T::MAX_OUTGOING_PEERS {
            self.create_connection(false).await;
        }
        Ok(())
    }
    pub(crate) async fn open_feeler_connection(&mut self) -> Result<(), WireError> {
        // No feeler if `-connect` is set
        if self.fixed_peer.is_some() {
            return Ok(());
        }
        self.create_connection(true).await;
        Ok(())
    }

    pub(crate) async fn request_blocks(&mut self, blocks: Vec<BlockHash>) -> Result<(), WireError> {
        let blocks: Vec<_> = blocks
            .into_iter()
            .filter(|block| {
                !self
                    .inflight
                    .contains_key(&InflightRequests::Blocks(*block))
            })
            .collect();

        let peer = self
            .send_to_random_peer(
                NodeRequest::GetBlock((blocks.clone(), true)),
                ServiceFlags::UTREEXO,
            )
            .await?;

        for block in blocks.iter() {
            self.inflight
                .insert(InflightRequests::Blocks(*block), (peer, Instant::now()));
        }

        Ok(())
    }

    pub(crate) async fn create_connection(&mut self, feeler: bool) -> Option<()> {
        // We should try to keep at least two utreexo connections
        let required_services = if self.utreexo_peers.len() < 2 {
            ServiceFlags::NETWORK | ServiceFlags::WITNESS | ServiceFlags::from(1 << 24)
        } else {
            ServiceFlags::NETWORK | ServiceFlags::WITNESS
        };

        let (peer_id, address) = match &self.fixed_peer {
            Some(address) => (0, address.clone()),
            None => self
                .address_man
                .get_address_to_connect(required_services, feeler)?,
        };

        self.address_man
            .update_set_state(peer_id, AddressState::Connected);

        // Don't connect to the same peer twice
        if self
            .0
            .peers
            .iter()
            .any(|peers| peers.1.address == address.get_net_address())
        {
            return None;
        }

        self.open_connection(feeler, peer_id, address).await;
        Some(())
    }
    /// Opens a new connection that doesn't require a proxy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_non_proxy_connection(
        feeler: bool,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
        mempool: Arc<RwLock<Mempool>>,
        network: bitcoin::Network,
        node_tx: UnboundedSender<NodeNotification>,
    ) -> impl Future<Output = ()> + Send + 'static {
        Peer::create_outbound_connection(
            peer_id_count,
            (address.get_net_address(), address.get_port()),
            mempool,
            network,
            node_tx,
            requests_rx,
            peer_id,
            feeler,
        )
    }
    /// Opens a connection through a socks5 interface
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn open_proxy_connection(
        proxy: SocketAddr,
        feeler: bool,
        mempool: Arc<RwLock<Mempool>>,
        network: bitcoin::Network,
        node_tx: UnboundedSender<NodeNotification>,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
    ) -> Result<(), Socks5Error> {
        let addr = match address.get_address() {
            AddrV2::Cjdns(addr) => Socks5Addr::Ipv6(addr),
            AddrV2::I2p(addr) => Socks5Addr::Domain(addr.into()),
            AddrV2::Ipv4(addr) => Socks5Addr::Ipv4(addr),
            AddrV2::Ipv6(addr) => Socks5Addr::Ipv6(addr),
            AddrV2::TorV2(addr) => Socks5Addr::Domain(addr.into()),
            AddrV2::TorV3(addr) => Socks5Addr::Domain(addr.into()),
            AddrV2::Unknown(_, _) => {
                return Err(Socks5Error::InvalidAddress);
            }
        };

        let proxy = TcpStream::connect(proxy).await?;
        let stream = Socks5StreamBuilder::connect(proxy, addr, address.get_port()).await?;
        Peer::create_peer_from_transport(
            stream,
            peer_id_count,
            mempool,
            network,
            node_tx,
            requests_rx,
            peer_id,
            feeler,
        );
        Ok(())
    }

    /// Creates a new outgoing connection with `address`. Connection may or may not be feeler,
    /// a special connection type that is used to learn about good peers, but are not kept afer
    /// handshake.
    pub(crate) async fn open_connection(
        &mut self,
        feeler: bool,
        peer_id: usize,
        address: LocalAddress,
    ) {
        let (requests_tx, requests_rx) = unbounded_channel();
        if let Some(ref proxy) = self.socks5 {
            spawn(timeout(
                Duration::from_secs(10),
                Self::open_proxy_connection(
                    proxy.address,
                    feeler,
                    self.mempool.clone(),
                    self.network.into(),
                    self.node_tx.clone(),
                    peer_id,
                    address.clone(),
                    requests_rx,
                    self.peer_id_count,
                ),
            ));
        } else {
            spawn(Self::open_non_proxy_connection(
                feeler,
                peer_id,
                address.clone(),
                requests_rx,
                self.peer_id_count,
                self.mempool.clone(),
                self.network.into(),
                self.node_tx.clone(),
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
                address: address.get_net_address(),
                port: address.get_port(),
                user_agent: "".to_string(),
                state: PeerStatus::Awaiting,
                channel: requests_tx,
                services: ServiceFlags::NONE,
                _last_message: Instant::now(),
                feeler,
                address_id: peer_id as u32,
                height: 0,
                banscore: 0,
            },
        );

        self.peer_id_count += 1;
    }
}

/// Run a task and log any errors that might occur.
macro_rules! try_and_log {
    ($what:expr) => {
        let result = $what;

        if let Err(error) = result {
            log::error!("{}:{} - {:?}", line!(), file!(), error);
        }
    };
}
macro_rules! periodic_job {
    ($what:expr, $timer:expr, $interval:ident, $context:ty) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            try_and_log!($what);
            $timer = Instant::now();
        }
    };
    ($what:expr, $timer:expr, $interval:ident, $context:ty, $no_log:literal) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            $what;
            $timer = Instant::now();
        }
    };
}
pub(crate) use periodic_job;
pub(crate) use try_and_log;
