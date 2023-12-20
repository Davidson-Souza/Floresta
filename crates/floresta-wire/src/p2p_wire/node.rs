//! Main file for this blockchain. A node is the central task that runs and handles important
//! events, such as new blocks, peer connection/disconnection, new addresses, etc.
//! A node should not care about peer-specific messages, peers'll handle things like pings.
use super::{
    address_man::{AddressMan, AddressState, LocalAddress},
    error::WireError,
    mempool::Mempool,
    node_context::{IBDNode, NodeContext, RunningNode},
    node_interface::{NodeInterface, NodeResponse, PeerInfo, UserRequest},
    peer::{Peer, PeerMessages, Version},
    socks::{Socks5Addr, Socks5Error, Socks5StreamBuilder},
};

use async_std::{
    channel::{self, bounded, Receiver, SendError, Sender},
    future::timeout,
    net::TcpStream,
    // sync::RwLock,
    task::spawn,
};
use bitcoin::{
    hashes::{sha256, Hash},
    network::{
        address::{AddrV2, AddrV2Message},
        constants::ServiceFlags,
        message_blockdata::Inventory,
        utreexo::{UData, UtreexoBlock},
    },
    BlockHash, BlockHeader, OutPoint, Transaction, TxOut, Txid,
};
use floresta_chain::{
    pruned_utreexo::{
        chainparams::get_chain_dns_seeds, udata::proof_util, BlockchainInterface,
        UpdatableChainstate,
    },
    BlockValidationErrors, BlockchainError, Network,
};
use futures::Future;
use log::{debug, error, info, trace, warn};
use rustreexo::accumulator::{node_hash::NodeHash, proof::Proof};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::{IpAddr, SocketAddr},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

/// Max number of simultaneous connections we initiates we are willing to hold
const MAX_OUTGOING_PEERS: usize = 10;

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

#[derive(Default, PartialEq)]
enum NodeState {
    #[default]
    WaitingPeer,
    DownloadHeaders,
    DownloadBlocks,
    Running,
}
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum InflightRequests {
    Headers,
    Blocks(BlockHash),
    RescanBlock(BlockHash),
    UserRequest(UserRequest),
    Connect(u32),
}
#[derive(Debug, Clone)]
pub struct LocalPeerView {
    state: PeerStatus,
    address_id: u32,
    channel: Sender<NodeRequest>,
    services: ServiceFlags,
    user_agent: String,
    address: IpAddr,
    port: u16,
    _last_message: Instant,
    feeler: bool,
    height: u32,
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
            last_block_check: Instant::now(),
        }
    }
}

pub struct NodeCommon<Chain: BlockchainInterface + UpdatableChainstate> {
    peer_id_count: u32,
    last_headers_request: Instant,
    last_tip_update: Instant,
    last_connection: Instant,
    last_peer_db_dump: Instant,
    last_broadcast: Instant,
    last_send_addresses: Instant,
    last_block_request: u32,
    network: Network,
    last_get_address_request: Instant,
    utreexo_peers: Vec<u32>,
    peer_ids: Vec<u32>,
    peers: HashMap<u32, LocalPeerView>,
    chain: Arc<Chain>,
    inflight: HashMap<InflightRequests, (u32, Instant)>,
    node_rx: Receiver<NodeNotification>,
    node_tx: Sender<NodeNotification>,
    state: NodeState,
    mempool: Arc<RwLock<Mempool>>,
    datadir: String,
    address_man: AddressMan,
    socks5: Option<Socks5StreamBuilder>,
}

pub struct UtreexoNode<Context, Chain: BlockchainInterface + UpdatableChainstate>(
    NodeCommon<Chain>,
    Context,
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
    ShutingDown,
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
    ) -> Self {
        let (node_tx, node_rx) = channel::unbounded();
        let socks5 = proxy.map(Socks5StreamBuilder::new);
        UtreexoNode(
            NodeCommon {
                inflight: HashMap::new(),
                state: NodeState::WaitingPeer,
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
                last_get_address_request: Instant::now(),
                last_send_addresses: Instant::now(),
                datadir,
                socks5,
            },
            T::default(),
        )
    }
    fn get_peer_info(&self, peer: &u32) -> Option<PeerInfo> {
        let peer = self.peers.get(peer)?;
        Some(PeerInfo {
            address: format!("{}:{}", peer.address, peer.port),
            services: peer.services.to_string(),
            user_agent: peer.user_agent.clone(),
            initial_height: peer.height,
        })
    }
    fn handle_disconnection(&mut self, peer: u32, idx: usize) -> Result<(), WireError> {
        if let Some(p) = self.peers.remove(&peer) {
            p.channel.close();
            if !p.feeler && p.state == PeerStatus::Ready {
                info!("Peer disconnected: {}", peer);
            }
        }
        self.peer_ids.retain(|&id| id != peer);
        self.utreexo_peers.retain(|&id| id != peer);

        if self.peer_ids.is_empty() || self.utreexo_peers.is_empty() {
            self.state = NodeState::WaitingPeer;
        }
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
    async fn handle_peer_ready(&mut self, peer: u32, version: &Version) -> Result<(), WireError> {
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
            if version.services.has(ServiceFlags::NODE_UTREEXO) {
                self.utreexo_peers.push(peer);
            }
            self.peer_ids.push(peer);
        }
        Ok(())
    }
    fn get_default_port(&self) -> u16 {
        match self.network {
            Network::Bitcoin => 8333,
            Network::Testnet => 18333,
            Network::Signet => 38333,
            Network::Regtest => 18444,
        }
    }

    #[allow(clippy::type_complexity)]
    fn process_proof(
        udata: &UData,
        transactions: &[Transaction],
        chain: &Arc<Chain>,
        block_hash: &BlockHash,
    ) -> Result<(Proof, Vec<sha256::Hash>, HashMap<OutPoint, TxOut>), WireError> {
        let targets = udata.proof.targets.iter().map(|target| target.0).collect();
        let hashes = udata
            .proof
            .hashes
            .iter()
            .map(|hash| NodeHash::Some(hash.into_inner()))
            .collect();
        let proof = Proof::new(targets, hashes);
        let mut hashes = Vec::new();
        let mut leaves_iter = udata.leaves.iter().cloned();
        let mut tx_iter = transactions.iter();

        let mut inputs = HashMap::new();
        tx_iter.next(); // Skip coinbase

        for tx in tx_iter {
            let txid = tx.txid();
            for (vout, out) in tx.output.iter().enumerate() {
                inputs.insert(
                    OutPoint {
                        txid,
                        vout: vout as u32,
                    },
                    out.clone(),
                );
            }

            for input in tx.input.iter() {
                if !inputs.contains_key(&input.previous_output) {
                    if let Some(leaf) = leaves_iter.next() {
                        let height = leaf.header_code >> 1;
                        let hash = chain.get_block_hash(height)?;
                        let leaf = proof_util::reconstruct_leaf_data(&leaf, input, hash)
                            .expect("Invalid proof");
                        // Coinbase can only be spent after a certain amount of confirmations
                        if leaf.header_code & 1 == 1
                            && !chain.is_coinbase_mature(height, *block_hash)?
                        {
                            return Err(WireError::CoinbaseNotMatured);
                        }
                        hashes.push(leaf._get_leaf_hashes());
                        inputs.insert(leaf.prevout, leaf.utxo);
                    }
                }
            }
        }

        Ok((proof, hashes, inputs))
    }
    async fn send_to_peer(&self, peer_id: u32, req: NodeRequest) -> Result<(), WireError> {
        if let Some(peer) = &self.peers.get(&peer_id) {
            if peer.state == PeerStatus::Ready {
                peer.channel
                    .send(req)
                    .await
                    .map_err(WireError::ChannelSend)?;
            }
        }
        Ok(())
    }
    async fn check_for_timeout(&mut self) -> Result<(), WireError> {
        let mut timed_out = Vec::new();
        for request in self.inflight.keys() {
            let (_, time) = self.inflight.get(request).unwrap();
            if time.elapsed() > Duration::from_secs(T::REQUEST_TIMEOUT) {
                timed_out.push(request.clone());
            }
        }
        let mut removed_peers = HashSet::new();
        let mut to_request = Vec::new();
        let mut rescan_blocks = Vec::new();
        for request in timed_out {
            let Some((peer, _)) = self.inflight.remove(&request) else {
                continue;
            };
            match request {
                InflightRequests::Blocks(block) => to_request.push(block),
                InflightRequests::RescanBlock(block) => rescan_blocks.push(block),
                InflightRequests::Headers => {
                    self.send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)
                        .await?;
                    self.last_headers_request = Instant::now();
                }
                InflightRequests::UserRequest(_) => {}
                InflightRequests::Connect(peer) => {
                    self.send_to_peer(peer, NodeRequest::Shutdown).await?
                }
            }

            if !removed_peers.contains(&peer) {
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                removed_peers.insert(peer);
                if let Some(peer) = self.peers.get_mut(&peer) {
                    info!("Peer {} timed out request, shuting down", peer.address);
                    peer.state = PeerStatus::ShutingDown;
                }
            }
        }
        for hash in rescan_blocks {
            let peer = self
                .send_to_random_peer(
                    NodeRequest::GetBlock((vec![hash], false)),
                    ServiceFlags::NONE,
                )
                .await?;
            self.inflight
                .insert(InflightRequests::RescanBlock(hash), (peer, Instant::now()));
        }
        self.request_blocks(to_request).await?;
        Ok(())
    }
    #[inline]
    async fn send_to_random_peer(
        &mut self,
        req: NodeRequest,
        required_services: ServiceFlags,
    ) -> Result<u32, WireError> {
        if self.peers.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }
        for _ in 0..10 {
            let idx = if required_services.has(ServiceFlags::NODE_UTREEXO) {
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
                    .await
                    .map_err(WireError::ChannelSend)?;
                return Ok(idx);
            }
        }
        self.create_connection(false).await;
        Err(WireError::NoPeerToSendRequest)
    }

    async fn init_peers(&mut self) -> Result<(), WireError> {
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

    async fn shutdown(&mut self) {
        info!("Shutting down node");
        for peer in self.peer_ids.iter() {
            try_and_log!(self.send_to_peer(*peer, NodeRequest::Shutdown).await);
        }
        try_and_log!(self.save_peers());
        try_and_log!(self.chain.flush());
    }
    async fn ask_block(&mut self) -> Result<(), WireError> {
        let blocks = self.get_blocks_to_download()?;
        self.request_blocks(blocks).await
    }
    async fn handle_broadcast(&self) -> Result<(), WireError> {
        for (_, peer) in self.peers.iter() {
            if peer.services.has(ServiceFlags::NODE_UTREEXO) {
                continue;
            }

            let transactions = self.chain.get_unbroadcasted();

            for transaction in transactions {
                let txid = transaction.txid();
                self.mempool.write().await.accept_to_mempool(transaction);
                peer.channel
                    .send(NodeRequest::BroadcastTransaction(txid))
                    .await
                    .map_err(WireError::ChannelSend)?;
            }
            let stale = self.mempool.write().await.get_stale();
            for tx in stale {
                peer.channel
                    .send(NodeRequest::BroadcastTransaction(tx))
                    .await
                    .map_err(WireError::ChannelSend)?;
            }
        }
        Ok(())
    }
    async fn ask_for_addresses(&mut self) -> Result<(), WireError> {
        let _ = self
            .send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)
            .await?;
        Ok(())
    }
    fn save_peers(&self) -> Result<(), WireError> {
        self.address_man
            .dump_peers(&self.datadir)
            .map_err(WireError::Io)
    }
    fn get_blocks_to_download(&mut self) -> Result<Vec<BlockHash>, WireError> {
        let mut blocks = Vec::new();
        let tip = self.chain.get_height()?;

        for i in (self.last_block_request + 1)..=(self.last_block_request + 100) {
            if i > tip {
                break;
            }
            self.last_block_request += 1;
            let hash = self.chain.get_block_hash(i)?;
            blocks.push(hash);
        }

        Ok(blocks)
    }

    async fn maybe_open_connection(&mut self) -> Result<(), WireError> {
        if self.peers.len() < MAX_OUTGOING_PEERS {
            self.create_connection(false).await;
        }
        Ok(())
    }
    async fn open_feeler_connection(&mut self) -> Result<(), WireError> {
        self.create_connection(true).await;
        Ok(())
    }

    async fn request_blocks(&mut self, blocks: Vec<BlockHash>) -> Result<(), WireError> {
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
                ServiceFlags::NODE_UTREEXO,
            )
            .await?;
        for block in blocks.iter() {
            self.inflight
                .insert(InflightRequests::Blocks(*block), (peer, Instant::now()));
        }
        Ok(())
    }

    async fn create_connection(&mut self, feeler: bool) -> Option<()> {
        // We should try to keep at least two utreexo connections
        let required_services = if self.utreexo_peers.len() < 2 {
            ServiceFlags::NETWORK | ServiceFlags::WITNESS | ServiceFlags::NODE_UTREEXO
        } else {
            ServiceFlags::NETWORK | ServiceFlags::WITNESS
        };
        let (peer_id, address) = self
            .address_man
            .get_address_to_connect(required_services, feeler)?;
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
    fn open_non_proxy_connection(
        feeler: bool,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: Receiver<NodeRequest>,
        peer_id_count: u32,
        mempool: Arc<RwLock<Mempool>>,
        network: bitcoin::Network,
        node_tx: Sender<NodeNotification>,
    ) -> impl Future<Output = ()> + Send + 'static {
        Peer::<TcpStream>::create_outbound_connection(
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
    async fn open_proxy_connection(
        proxy: SocketAddr,
        feeler: bool,
        mempool: Arc<RwLock<Mempool>>,
        network: bitcoin::Network,
        node_tx: Sender<NodeNotification>,
        peer_id: usize,
        address: LocalAddress,
        requests_rx: Receiver<NodeRequest>,
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
    async fn open_connection(&mut self, feeler: bool, peer_id: usize, address: LocalAddress) {
        let (requests_tx, requests_rx) = bounded(1024);
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
            Self::open_non_proxy_connection(
                feeler,
                peer_id,
                address.clone(),
                requests_rx,
                self.peer_id_count,
                self.mempool.clone(),
                self.network.into(),
                self.node_tx.clone(),
            )
            .await;
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
            },
        );

        self.peer_id_count += 1;
    }
}

/// An IBD node, should be used to get your chainstate up-to-date with the network, but
/// returns as soon as there's no more blocks to download.
impl<Chain> UtreexoNode<IBDNode, Chain>
where
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    async fn handle_block(chain: &Arc<Chain>, block: UtreexoBlock) -> Result<(), WireError> {
        let (proof, del_hashes, inputs) = Self::process_proof(
            &block.udata.unwrap(),
            &block.block.txdata,
            chain,
            &block.block.block_hash(),
        )?;
        try_and_log!(chain
            .connect_block(&block.block, proof, inputs, del_hashes)
            .map_err(|e| {
                if let BlockchainError::BlockValidation(_) = &e {
                    try_and_log!(chain.invalidate_block(block.block.block_hash()));
                }
                error!(
                    "Error while connecting block {}: {e:?}",
                    block.block.block_hash()
                );
                e
            }));
        Ok(())
    }
    async fn handle_headers(&mut self, headers: Vec<BlockHeader>) -> Result<(), WireError> {
        if headers.is_empty() {
            // Start downloading blocks
            self.chain.flush()?;
            self.state = NodeState::DownloadBlocks;
            return Ok(());
        }
        self.last_headers_request = Instant::now();
        info!(
            "Downloading headers at height={} hash={}",
            self.chain.get_best_block()?.0 + 1,
            headers[0].block_hash()
        );
        for header in headers {
            self.chain.accept_header(header)?;
        }
        if self.inflight.contains_key(&InflightRequests::Headers) {
            return Ok(());
        }
        let locator = self.chain.get_block_locator()?;
        let peer = self
            .send_to_random_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)
            .await?;

        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));
        Ok(())
    }
    async fn maybe_request_headers(&mut self) -> Result<(), WireError> {
        if self.state != NodeState::DownloadHeaders {
            return Ok(());
        }
        info!("Asking for headers");
        let locator = self
            .chain
            .get_block_locator()
            .expect("Could not create locator");
        self.send_to_random_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)
            .await?;
        self.last_headers_request = Instant::now();
        Ok(())
    }
    pub async fn run(&mut self, stop_signal: &Arc<RwLock<bool>>) -> Result<(), WireError> {
        self.create_connection(false).await;
        self.last_headers_request = Instant::now();
        loop {
            while let Ok(notification) =
                timeout(Duration::from_millis(10), self.node_rx.recv()).await
            {
                try_and_log!(self.handle_notification(notification).await);
            }

            if *stop_signal.read().await {
                break;
            }

            if self.state == NodeState::WaitingPeer {
                try_and_log!(self.maybe_open_connection().await);
            }
            self.last_tip_update = Instant::now();

            // If we don't have any peers, then we can't do anything
            if self.state == NodeState::WaitingPeer {
                continue;
            }
            // We download blocks in parallel, sometimes we get them out of order, so we need to
            // process them in order. If we got all blocks but the first one, we can't process
            // them yet. Once we get it, we can process all blocks we have.
            if self.state == NodeState::DownloadBlocks {
                self.process_queued_blocks().await.or_else(|err| {
                    // This usually means we just processed all blocks, and we are done.
                    if matches!(err, WireError::Blockchain(BlockchainError::BlockNotPresent)) {
                        info!("Finished downloading blocks");
                        self.chain.toggle_ibd(false);
                        Ok(())
                    } else {
                        Err(err)
                    }
                })?;
            }

            if !self.chain.is_in_idb() {
                break;
            }
            // If we are downloading blocks, we need to request more if we have space.
            let currently_inflight = self.0.inflight.len() + self.1.blocks.len();
            if self.state == NodeState::DownloadBlocks
                && currently_inflight < IBDNode::MAX_INFLIGHT_REQUESTS
            {
                let blocks = self.get_blocks_to_download()?;
                if blocks.is_empty() {
                    info!("Finished downloading blocks");
                    self.chain.toggle_ibd(false);

                    break;
                }
                self.request_blocks(blocks).await?;
            }

            self.check_for_timeout().await?;

            periodic_job!(
                self.maybe_request_headers().await,
                self.last_headers_request,
                IBD_REQUEST_BLOCKS_AGAIN,
                IBDNode
            );
        }
        Ok(())
    }
    async fn process_queued_blocks(&mut self) -> Result<(), WireError> {
        let mut hash = self
            .chain
            .get_block_hash(self.chain.get_validation_index()? + 1)?;

        while let Some(block) = self.1.blocks.remove(&hash) {
            Self::handle_block(&self.chain, block).await?;
            hash = self
                .chain
                .get_block_hash(self.chain.get_validation_index()? + 1)?;
        }
        Ok(())
    }

    async fn handle_notification(
        &mut self,
        notification: Result<NodeNotification, async_std::channel::RecvError>,
    ) -> Result<(), WireError> {
        match notification? {
            NodeNotification::FromPeer(peer, message) => match message {
                PeerMessages::NewBlock(block) => {
                    trace!("We got and inv with block {block} but we are on IBD, ignoring it");
                }
                PeerMessages::Block(block) => {
                    // Remove from inflight, since we just got it.
                    if self
                        .inflight
                        .remove(&InflightRequests::Blocks(block.block.block_hash()))
                        .is_none()
                    {
                        // We didn't request this block, so we should disconnect the peer.
                        if let Some(peer) = self.peers.get(&peer).cloned() {
                            self.address_man.update_set_state(
                                peer.address_id as usize,
                                AddressState::Banned(IBDNode::BAN_TIME),
                            );
                        }
                        error!(
                            "Peer {peer} sent us block {} which we didn't request",
                            block.block.block_hash()
                        );

                        self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                        return Err(WireError::PeerMisbehaving);
                    }
                    // We may receive blocks out of order, so we store them in a map until we
                    // receive all the previous ones.
                    let height = self.chain.get_validation_index()? + 1;
                    if self.0.chain.get_block_hash(height)? == block.block.block_hash() {
                        Self::handle_block(&self.chain, block).await?;
                    } else {
                        self.1.blocks.insert(block.block.block_hash(), block);
                    }

                    let currently_inflight = self.inflight.len() + self.1.blocks.len();
                    if self.state == NodeState::DownloadBlocks
                        && currently_inflight < IBDNode::MAX_INFLIGHT_REQUESTS
                    {
                        let blocks = self.get_blocks_to_download()?;
                        self.request_blocks(blocks).await?;
                    }
                }
                PeerMessages::Headers(headers) => {
                    self.inflight.remove(&InflightRequests::Headers);
                    return self.handle_headers(headers).await;
                }
                PeerMessages::Ready(version) => {
                    self.handle_peer_ready(peer, &version).await?;

                    if version.services.has(ServiceFlags::NODE_UTREEXO)
                        && matches!(self.state, NodeState::WaitingPeer)
                        && !self.inflight.contains_key(&InflightRequests::Headers)
                    {
                        try_and_log!(
                            self.send_to_peer(
                                peer,
                                NodeRequest::GetHeaders(
                                    self.chain.get_block_locator().expect("Can get locators"),
                                )
                            )
                            .await
                        );
                        self.state = NodeState::DownloadHeaders;
                    }
                }

                PeerMessages::Disconnected(idx) => {
                    self.handle_disconnection(peer, idx)?;

                    if self.peer_ids.is_empty() || self.utreexo_peers.is_empty() {
                        self.state = NodeState::WaitingPeer;
                    }
                }
                PeerMessages::Addr(addresses) => {
                    let addresses: Vec<_> =
                        addresses.iter().cloned().map(|addr| addr.into()).collect();
                    self.address_man.push_addresses(&addresses);
                }
                _ => {}
            },
        }
        Ok(())
    }
}

impl<Chain> UtreexoNode<RunningNode, Chain>
where
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    /// Returns a handle to the node interface that we can use to request data from our
    /// node. This struct is thread safe, so we can use it from multiple threads and have
    /// multiple handles. It also doesn't require a mutable reference to the node, or any
    /// synchronization mechanism.
    pub fn get_handle(&self) -> Arc<NodeInterface> {
        self.1.user_requests.clone()
    }
    #[allow(clippy::result_large_err)]
    fn check_request_timeout(&mut self) -> Result<(), SendError<NodeResponse>> {
        let mutex = self.1.user_requests.requests.lock().unwrap();
        let mut to_remove = Vec::new();
        for req in mutex.iter() {
            if req.time.elapsed() > Duration::from_secs(10) {
                to_remove.push(req.req);
            }
        }
        drop(mutex);
        for request in to_remove {
            self.1.user_requests.send_answer(request, None);
        }

        Ok(())
    }
    async fn handle_user_request(&mut self) {
        let mut requests = Vec::new();

        for request in self.1.user_requests.requests.lock().unwrap().iter() {
            if !self
                .inflight
                .contains_key(&InflightRequests::UserRequest(request.req))
            {
                requests.push(request.req);
            }
        }
        self.perform_user_request(requests).await;
    }
    fn handle_get_peer_info(&self) {
        let mut peers = Vec::new();
        for peer in self.peer_ids.iter() {
            peers.push(self.get_peer_info(peer));
        }
        let peers = peers.into_iter().flatten().collect();
        self.1.user_requests.send_answer(
            UserRequest::GetPeerInfo,
            Some(NodeResponse::GetPeerInfo(peers)),
        );
    }
    /// In some edge cases, we may get a block header, but not the block itself. This
    /// function checks if we have the block, and if not, requests it.
    async fn ask_missed_block(&mut self) -> Result<(), WireError> {
        let best_block = self.chain.get_best_block()?.0;
        let validation_index = self.chain.get_validation_index()?;

        if best_block == validation_index {
            return Ok(());
        }
        for block in validation_index..=best_block {
            let block = self.0.chain.get_block_hash(block)?;
            self.request_blocks(vec![block]).await?;
        }
        Ok(())
    }
    async fn perform_user_request(&mut self, user_req: Vec<UserRequest>) {
        for user_req in user_req {
            let req = match user_req {
                UserRequest::Block(block) => NodeRequest::GetBlock((vec![block], false)),
                UserRequest::UtreexoBlock(block) => NodeRequest::GetBlock((vec![block], true)),
                UserRequest::MempoolTransaction(txid) => NodeRequest::MempoolTransaction(txid),
                UserRequest::GetPeerInfo => {
                    self.handle_get_peer_info();
                    continue;
                }
                UserRequest::Connect((addr, port)) => {
                    let addr_v2 = match addr {
                        IpAddr::V4(addr) => AddrV2::Ipv4(addr),
                        IpAddr::V6(addr) => AddrV2::Ipv6(addr),
                    };
                    let id = rand::random::<usize>();
                    let local_addr =
                        LocalAddress::new(addr_v2, 0, AddressState::NeverTried, 0.into(), port, id);
                    self.open_connection(false, 0, local_addr).await;
                    self.1.user_requests.send_answer(
                        UserRequest::Connect((addr, port)),
                        Some(NodeResponse::Connect(true)),
                    );
                    continue;
                }
            };
            let peer = self.send_to_random_peer(req, ServiceFlags::NONE).await;
            if let Ok(peer) = peer {
                self.inflight.insert(
                    InflightRequests::UserRequest(user_req),
                    (peer, Instant::now()),
                );
            }
        }
    }
    async fn send_addresses(&mut self) -> Result<(), WireError> {
        let addresses = self
            .address_man
            .get_addresses_to_send()
            .into_iter()
            .map(|(addr, time, services, port)| AddrV2Message {
                services,
                addr,
                port,
                time: time as u32,
            })
            .collect();

        self.send_to_random_peer(NodeRequest::SendAddresses(addresses), ServiceFlags::NONE)
            .await?;
        Ok(())
    }
    pub async fn run(mut self, kill_signal: &Arc<RwLock<bool>>) {
        try_and_log!(self.init_peers().await);

        // Use this node state to Initial Block download
        let mut ibd = UtreexoNode(self.0, IBDNode::default());
        try_and_log!(UtreexoNode::<IBDNode, Chain>::run(&mut ibd, kill_signal).await);

        // Then take the final state and run the node
        self = UtreexoNode(ibd.0, self.1);

        let _ = self
            .send_to_random_peer(
                NodeRequest::GetHeaders(self.chain.get_block_locator().expect("Can get locators")),
                ServiceFlags::NONE,
            )
            .await;

        loop {
            while let Ok(notification) =
                timeout(Duration::from_millis(1), self.node_rx.recv()).await
            {
                try_and_log!(self.handle_notification(notification).await);
            }

            if *kill_signal.read().await {
                self.shutdown().await;
                break;
            }

            // Jobs that don't need a connected peer

            // Check whether we are in a stale tip
            periodic_job!(
                self.check_for_stale_tip().await,
                self.last_tip_update,
                ASSUME_STALE,
                RunningNode
            );
            // Save our peers db
            periodic_job!(
                self.save_peers(),
                self.last_peer_db_dump,
                PEER_DB_DUMP_INTERVAL,
                RunningNode
            );
            // Rework our address database
            periodic_job!(
                self.address_man.rearrange_buckets(),
                self.1.last_address_rearrange,
                ADDRESS_REARRANGE_INTERVAL,
                RunningNode,
                true
            );
            // Perhaps we need more connections
            periodic_job!(
                self.maybe_open_connection().await,
                self.last_connection,
                TRY_NEW_CONNECTION,
                RunningNode
            );

            // Requests using the node handle
            try_and_log!(self.check_request_timeout());
            self.handle_user_request().await;

            // Check if some of our peers have timed out a request
            try_and_log!(self.check_for_timeout().await);

            // Those jobs bellow needs a connected peer to work
            if self.state == NodeState::WaitingPeer {
                continue;
            }
            // Check if we haven't missed any block
            periodic_job!(
                self.ask_missed_block().await,
                self.1.last_block_check,
                BLOCK_CHECK_INTERVAL,
                RunningNode
            );
            // Aks our peers for new addresses
            periodic_job!(
                self.ask_for_addresses().await,
                self.last_get_address_request,
                ASK_FOR_PEERS_INTERVAL,
                RunningNode
            );
            // Open new feeler connection periodically
            periodic_job!(
                self.open_feeler_connection().await,
                self.1.last_feeler,
                FEELER_INTERVAL,
                RunningNode
            );
            // Try broadcast transactions
            periodic_job!(
                self.handle_broadcast().await,
                self.last_broadcast,
                BROADCAST_DELAY,
                RunningNode
            );
            // Send our addresses to our peers
            periodic_job!(
                self.send_addresses().await,
                self.last_send_addresses,
                SEND_ADDRESSES_INTERVAL,
                RunningNode
            );
            try_and_log!(self.ask_block().await);
            try_and_log!(self.request_rescan_block().await);
        }
    }
    async fn request_rescan_block(&mut self) -> Result<(), WireError> {
        let tip = self.chain.get_height().unwrap();
        if self.inflight.len() + 10 > RunningNode::MAX_INFLIGHT_REQUESTS {
            return Ok(());
        }
        // We use a grace period to avoid looping at the end of rescan
        if let RescanStatus::Completed(time) = self.1.last_rescan_request {
            if time.elapsed() > Duration::from_secs(60) {
                self.1.last_rescan_request = RescanStatus::None;
            }
        }
        if self.1.last_rescan_request == RescanStatus::None
            && self.chain.get_rescan_index().is_some()
        {
            self.1.last_rescan_request =
                RescanStatus::InProgress(self.chain.get_rescan_index().unwrap());
        }
        if let RescanStatus::InProgress(height) = self.1.last_rescan_request {
            for i in (height + 1)..=(height + 10) {
                if i > tip {
                    self.1.last_rescan_request = RescanStatus::Completed(Instant::now());
                    break;
                }
                self.1.last_rescan_request = RescanStatus::InProgress(i);
                let hash = self.chain.get_block_hash(i)?;
                let peer = self
                    .send_to_random_peer(
                        NodeRequest::GetBlock((vec![hash], false)),
                        ServiceFlags::NONE,
                    )
                    .await?;
                self.inflight
                    .insert(InflightRequests::RescanBlock(hash), (peer, Instant::now()));
            }
        }

        Ok(())
    }
    /// This function checks how many time has passed since our last tip update, if it's
    /// been more than 15 minutes, try to update it.
    async fn check_for_stale_tip(&mut self) -> Result<(), WireError> {
        warn!("Potential stale tip detected, trying extra peers");
        self.create_connection(false).await;
        self.send_to_random_peer(
            NodeRequest::GetHeaders(self.chain.get_block_locator().unwrap()),
            ServiceFlags::NONE,
        )
        .await?;
        Ok(())
    }
    async fn handle_new_block(&mut self) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::Headers) {
            return Ok(());
        }
        let locator = self.0.chain.get_block_locator().unwrap();

        let peer = self
            .send_to_random_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)
            .await?;
        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }
    /// This function is called every time we get a Block message from a peer.
    /// This block may be a rescan block, a user request or a new block that we
    /// need to process.
    async fn handle_block_data(&mut self, block: UtreexoBlock, peer: u32) -> Result<(), WireError> {
        // Rescan block, a block that the wallet is interested in to check if it contains
        // any transaction that we are interested in.
        if self
            .inflight
            .remove(&InflightRequests::RescanBlock(block.block.block_hash()))
            .is_some()
        {
            self.request_rescan_block().await?;
            return Ok(self.chain.process_rescan_block(&block.block)?);
        }
        // If this block is a request made through the user interface, send it back to the
        // user.
        if self
            .inflight
            .remove(&InflightRequests::UserRequest(UserRequest::Block(
                block.block.block_hash(),
            )))
            .is_some()
        {
            if block.udata.is_some() {
                self.1.user_requests.send_answer(
                    UserRequest::UtreexoBlock(block.block.block_hash()),
                    Some(NodeResponse::UtreexoBlock(block)),
                );
                return Ok(());
            }
            self.1.user_requests.send_answer(
                UserRequest::Block(block.block.block_hash()),
                Some(NodeResponse::Block(block.block)),
            );
            return Ok(());
        }

        // If none of the above, it means that this block is a new block that we need to
        // process.

        // Check if we actually requested this block. If a peer sends a block we didn't
        // request, we should disconnect it.
        if self
            .inflight
            .remove(&InflightRequests::Blocks(block.block.block_hash()))
            .is_none()
        {
            // We didn't request this block, so we should disconnect the peer.
            if let Some(peer) = self.peers.get(&peer).cloned() {
                self.address_man.update_set_state(
                    peer.address_id as usize,
                    AddressState::Banned(RunningNode::BAN_TIME),
                );
            }
            error!(
                "Peer {peer} sent us block {} which we didn't request",
                block.block.block_hash()
            );
            self.send_to_peer(peer, NodeRequest::Shutdown).await?;
            return Err(WireError::PeerMisbehaving);
        }

        let validation_index = self.chain.get_validation_index()?;
        let validation_hash = self.chain.get_block_hash(validation_index)?;

        // We've downloaded a block that's not the next we need, ignore it for now
        if validation_hash != block.block.header.prev_blockhash {
            return Ok(());
        }

        let (proof, del_hashes, inputs) = Self::process_proof(
            &block.udata.unwrap(),
            &block.block.txdata,
            &self.chain,
            &block.block.block_hash(),
        )?;

        if let Err(e) = self
            .chain
            .connect_block(&block.block, proof, inputs, del_hashes)
        {
            error!("Invalid block received by peer {} reason: {:?}", peer, e);

            if let BlockchainError::BlockValidation(e) = e {
                // Because the proof isn't committed to the block, we can't invalidate
                // it if the proof is invalid. Any other error should cause the block
                // to be invalidated.
                match e {
                    BlockValidationErrors::InvalidTx(_)
                    | BlockValidationErrors::NotEnoughPow
                    | BlockValidationErrors::BadMerkleRoot
                    | BlockValidationErrors::BadWitnessCommitment
                    | BlockValidationErrors::NotEnoughMoney
                    | BlockValidationErrors::FirstTxIsnNotCoinbase
                    | BlockValidationErrors::BadCoinbaseOutValue
                    | BlockValidationErrors::EmptyBlock
                    | BlockValidationErrors::BlockExtendsAnOrphanChain
                    | BlockValidationErrors::BadBip34 => {
                        self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                        try_and_log!(self.chain.invalidate_block(block.block.block_hash()));
                    }
                    BlockValidationErrors::InvalidProof => {}
                }
            }

            // Disconnect the peer and ban it.
            if let Some(peer) = self.peers.get(&peer).cloned() {
                self.address_man.update_set_state(
                    peer.address_id as usize,
                    AddressState::Banned(RunningNode::BAN_TIME),
                );
            }
            self.send_to_peer(peer, NodeRequest::Shutdown).await?;
            return Err(WireError::PeerMisbehaving);
        }
        // Remove confirmed transactions from the mempool.
        let mempool_delta = self.mempool.write().await.consume_block(&block.block);
        debug!(
            "Block {} accepted, confirmed transactions: {:?}",
            block.block.block_hash(),
            mempool_delta
        );
        self.last_tip_update = Instant::now();
        Ok(())
    }
    async fn handle_notification(
        &mut self,
        notification: Result<NodeNotification, async_std::channel::RecvError>,
    ) -> Result<(), WireError> {
        match notification? {
            NodeNotification::FromPeer(peer, message) => match message {
                PeerMessages::NewBlock(block) => {
                    trace!("We got an inv with block {block} requesting it");
                    self.handle_new_block().await?;
                }
                PeerMessages::Block(block) => {
                    trace!(
                        "Got data for block {} from peer {peer}",
                        block.block.block_hash()
                    );
                    self.handle_block_data(block, peer).await?;
                }
                PeerMessages::Headers(headers) => {
                    self.inflight.remove(&InflightRequests::Headers);
                    for header in headers.iter() {
                        self.chain.accept_header(*header)?;
                    }
                }
                PeerMessages::Ready(version) => {
                    self.handle_peer_ready(peer, &version).await?;
                    if version.services.has(ServiceFlags::NODE_UTREEXO) {
                        self.state = NodeState::Running;
                    }
                }
                PeerMessages::Disconnected(idx) => {
                    self.handle_disconnection(peer, idx)?;
                }
                PeerMessages::Addr(addresses) => {
                    let addresses: Vec<_> =
                        addresses.iter().cloned().map(|addr| addr.into()).collect();
                    self.address_man.push_addresses(&addresses);
                }
                PeerMessages::NotFound(inv) => match inv {
                    Inventory::Error => {}
                    Inventory::Block(block)
                    | Inventory::WitnessBlock(block)
                    | Inventory::UtreexoBlock(block)
                    | Inventory::UtreexoWitnessBlock(block)
                    | Inventory::CompactBlock(block) => {
                        self.1
                            .user_requests
                            .send_answer(UserRequest::Block(block), None);
                    }

                    Inventory::WitnessTransaction(tx) | Inventory::Transaction(tx) => {
                        self.1
                            .user_requests
                            .send_answer(UserRequest::MempoolTransaction(tx), None);
                    }
                    _ => {}
                },
                PeerMessages::Transaction(tx) => {
                    self.1.user_requests.send_answer(
                        UserRequest::MempoolTransaction(tx.txid()),
                        Some(NodeResponse::MempoolTransaction(tx)),
                    );
                }
            },
        }
        Ok(())
    }
}

/// Run a task and log any errors that might occur.
macro_rules! try_and_log {
    ($what: expr) => {
        let result = $what;

        if let Err(error) = result {
            log::error!("{}:{} - {:?}", line!(), file!(), error);
        }
    };
}
macro_rules! periodic_job {
    ($what: expr, $timer: expr, $interval: ident, $context: ty) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            try_and_log!($what);
            $timer = Instant::now();
        }
    };
    ($what: expr, $timer: expr, $interval: ident, $context: ty, $no_log: literal) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            $what;
            $timer = Instant::now();
        }
    };
}
pub(crate) use periodic_job;
pub(crate) use try_and_log;
