// SPDX-License-Identifier: MIT OR Apache-2.0

use core::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Network;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::ChainBackend;
use floresta_common::Ema;
use floresta_common::service_flags;
use floresta_common::try_and_log;
use floresta_domain::mempool::MempoolBase;
use rand::RngExt;
use tokio::net::tcp::WriteHalf;
use tokio::spawn;
use tokio::sync::Mutex;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::debug;
use tracing::info;

use super::ConnectionKind;
use super::InflightRequests;
use super::LocalPeerView;
use super::NodeNotification;
use super::NodeRequest;
use super::PeerStatus;
use super::UtreexoNode;
use crate::TransportProtocol;
use crate::address_man::AddressMan;
use crate::address_man::AddressState;
use crate::address_man::LocalAddress;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::node_context::NodeContext;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::Peer;
use crate::p2p_wire::peer::create_actors;
use crate::p2p_wire::transport;

/// How long before we consider using alternative ways to find addresses,
/// such as hard-coded peers
const HARDCODED_ADDRESSES_GRACE_PERIOD: Duration = Duration::from_secs(60);

/// The minimum amount of time between address fetching requests from DNS seeds (one hour).
const DNS_SEED_REQUEST_INTERVAL: Duration = Duration::from_secs(60 * 60);

impl<T, Chain> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
{
    // === CONNECTION CREATION ===

    /// Create a new outgoing connection, selecting an appropriate peer address.
    ///
    /// If fixed peers are set via the `--connect` CLI argument, their connection
    /// kind will always be coerced to [`ConnectionKind::Manual`] and the first
    /// not-yet-connected fixed peer is selected. Otherwise, an address is
    /// selected from the [`AddressMan`] based on the required [`ServiceFlags`]
    /// for the given `connection_kind`.
    ///
    /// If no address is available and the kind is not [`ConnectionKind::Manual`],
    /// hardcoded addresses are loaded into the [`AddressMan`] as a fallback.
    pub(crate) fn create_connection(
        &mut self,
        mut conn_kind: ConnectionKind,
    ) -> Result<(), WireError> {
        // Set the fixed peer's connection kind to manual, if set.
        if self.has_fixed_peers() {
            conn_kind = ConnectionKind::Manual;
        }

        // Get the peer's `ServiceFlags`.
        let required_services = match conn_kind {
            ConnectionKind::Regular(services) => services,
            ConnectionKind::Feeler | ConnectionKind::Extra | ConnectionKind::Manual => {
                ServiceFlags::NONE
            }
        };

        // Pick the first fixed peer that we are not already connected to, or
        // fall back to fetching a new address from the address manager when no
        // fixed peers were configured.
        let candidate_peer = if self.has_fixed_peers() {
            self.fixed_peers
                .iter()
                .find(|addr| {
                    self.peers.values().all(|p| {
                        p.address.as_bitcoin_socket_addr() != addr.as_bitcoin_socket_addr()
                    })
                })
                .map(|addr| (0, addr.clone()))
        } else {
            self.address_man.get_address_to_connect(
                required_services,
                matches!(conn_kind, ConnectionKind::Feeler),
            )
        };

        // Load hardcoded addresses to the address manager if no fixed or manual peers exist.
        let Some((peer_id, peer_address)) = candidate_peer else {
            if !matches!(conn_kind, ConnectionKind::Manual) {
                let net = self.network;
                self.address_man.add_fixed_addresses(net);
            }

            return Err(WireError::NoAddressesAvailable);
        };

        debug!("Attempting connection with address={peer_address:?} kind={conn_kind:?}",);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Defaults to failed, if the connection is successful, we'll update the state
        self.address_man
            .update_set_state(peer_id, AddressState::Failed(now));

        // Don't connect to the same peer twice
        if self
            .common
            .peers
            .values()
            .any(|p| p.address == peer_address)
        {
            return Err(WireError::PeerAlreadyExists(peer_address));
        }

        // Only allow P2PV1 fallback if the peer's connection kind is manual,
        // or if the `--allow-v1-fallback` CLI argument was set.
        let allow_v1_fallback =
            matches!(conn_kind, ConnectionKind::Manual) || self.config.allow_v1_fallback;

        // Open a connection to the peer.
        self.open_connection(conn_kind, peer_address, allow_v1_fallback)?;

        Ok(())
    }

    pub(crate) fn open_feeler_connection(&mut self) -> Result<(), WireError> {
        // No feeler if `--connect` is set
        if self.has_fixed_peers() {
            return Ok(());
        }

        for _ in 0..T::NEW_CONNECTIONS_BATCH_SIZE {
            self.create_connection(ConnectionKind::Feeler)?;
        }

        Ok(())
    }

    /// Creates a new outgoing connection with `address`.
    ///
    /// `kind` may or may not be a [`ConnectionKind::Feeler`], a special connection type
    /// that is used to learn about good peers, but are not kept after handshake
    /// (others are [`ConnectionKind::Regular`], [`ConnectionKind::Manual`] and [`ConnectionKind::Extra`]).
    ///
    /// We will always try to open a V2 connection first. If the `allow_v1_fallback` is set,
    /// we may retry the connection with the old V1 protocol if the V2 connection fails.
    /// We don't open the connection here, we create a [`Peer`] actor that will try to open
    /// a connection with the given address and kind. If it succeeds, it will send a
    /// [`PeerMessages::Ready`](crate::p2p_wire::peer::PeerMessages) to the node after handshaking.
    pub(crate) fn open_connection(
        &mut self,
        kind: ConnectionKind,
        peer_address: LocalAddress,
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
                    peer_address.clone(),
                    requests_rx,
                    self.peer_id_count,
                    self.config.user_agent.clone(),
                    self.chain
                        .get_best_block()
                        .expect("infallible in ChainState")
                        .0,
                    allow_v1_fallback,
                ),
            ));
        } else {
            spawn(timeout(
                Duration::from_secs(10),
                Self::open_non_proxy_connection(
                    kind,
                    peer_address.clone(),
                    requests_rx,
                    self.peer_id_count,
                    self.mempool.clone(),
                    self.network,
                    self.node_tx.clone(),
                    self.config.user_agent.clone(),
                    self.chain
                        .get_best_block()
                        .expect("infallible in ChainState")
                        .0,
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
                message_times: Ema::with_half_life_50(),
                address: peer_address,
                user_agent: "".to_string(),
                state: PeerStatus::Awaiting,
                channel: requests_tx,
                services: ServiceFlags::NONE,
                _last_message: Instant::now(),
                kind,
                height: 0,
                time_offset: 0,
                banscore: 0,
                // Will be downgraded to V1 if the V2 handshake fails, and we allow fallback
                transport_protocol: TransportProtocol::V2,
            },
        );

        match kind {
            ConnectionKind::Feeler => self.last_feeler = Instant::now(),
            ConnectionKind::Regular(_) => self.last_connection = Instant::now(),
            // Note: Creating a manual peer intentionally doesn't affect the `last_connection`
            // timer, since they don't necessarily follow our connection logic, and we may still
            // need more utreexo/CBS peers
            //
            // Extra connections are also not taken into account here because they will probably be
            // short-lived.
            _ => {}
        }

        // Increment peer_id count and the list of peer ids
        // so we can get information about connected or
        // added peers when requesting with getpeerinfo command
        self.peer_id_count += 1;
        Ok(())
    }

    /// Opens a new connection that doesn't require a proxy and includes the functionalities of create_outbound_connection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn open_non_proxy_connection(
        kind: ConnectionKind,
        peer_address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
        mempool: Arc<Mutex<dyn MempoolBase>>,
        network: Network,
        node_tx: UnboundedSender<NodeNotification>,
        our_user_agent: String,
        our_best_block: u32,
        allow_v1_fallback: bool,
    ) -> Result<(), WireError> {
        let ip_addr = peer_address
            .get_net_address()
            .ok_or(WireError::UnreachableNetwork)?;
        let address = (ip_addr, peer_address.get_port());

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
        let handle = Peer::<WriteHalf>::create_peer(
            peer_id_count,
            peer_address,
            mempool,
            node_tx.clone(),
            requests_rx,
            kind,
            actor_receiver,
            transport_writer,
            our_user_agent,
            our_best_block,
            cancellation_sender,
            transport_protocol,
        );

        // We drop the returned join handle because we will never join on the actor
        drop(handle);

        Ok(())
    }

    /// Opens a connection through a socks5 interface
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn open_proxy_connection(
        proxy: SocketAddr,
        kind: ConnectionKind,
        mempool: Arc<Mutex<dyn MempoolBase>>,
        network: Network,
        node_tx: UnboundedSender<NodeNotification>,
        peer_address: LocalAddress,
        requests_rx: UnboundedReceiver<NodeRequest>,
        peer_id_count: u32,
        our_user_agent: String,
        our_best_block: u32,
        allow_v1_fallback: bool,
    ) -> Result<(), WireError> {
        let (transport_reader, transport_writer, transport_protocol) =
            transport::connect_proxy(proxy, peer_address.clone(), network, allow_v1_fallback)
                .await?;

        let (cancellation_sender, cancellation_receiver) = oneshot::channel();
        let (actor_receiver, actor) = create_actors(transport_reader);
        tokio::spawn(async move {
            tokio::select! {
                _ = cancellation_receiver => {}
                _ = actor.run() => {}
            }
        });

        let handle = Peer::<WriteHalf>::create_peer(
            peer_id_count,
            peer_address,
            mempool,
            node_tx,
            requests_rx,
            kind,
            actor_receiver,
            transport_writer,
            our_user_agent,
            our_best_block,
            cancellation_sender,
            transport_protocol,
        );

        // We drop the returned join handle because we will never join on the actor
        drop(handle);

        Ok(())
    }

    // === BOOTSTRAPPING ===

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
            let default_port = BitcoinSocketAddr::default_p2p_port(network);
            let dns_seeds = floresta_chain::get_chain_dns_seeds(network);

            let mut addresses = Vec::new();
            for seed in &dns_seeds {
                if let Ok(got) = AddressMan::get_seeds_from_dns(seed, default_port, proxy_addr) {
                    addresses.extend(got);
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

    /// Check whether it's necessary to request more addresses from DNS seeds.
    ///
    /// Perform another address request from DNS seeds if we still don't have enough addresses
    /// on the [`AddressMan`] and the last address request from DNS seeds was over 2 minutes ago.
    fn maybe_ask_dns_seed_for_addresses(&mut self) {
        let enough_addresses = self.address_man.enough_addresses();

        // Skip if address fetching from DNS seeds is disabled,
        // or if the [`AddressMan`] has enough addresses in its database.
        if self.config.disable_dns_seeds || enough_addresses {
            return;
        }

        // Don't ask for peers too often.
        let last_dns_request = self.last_dns_seed_call.elapsed();
        if last_dns_request < DNS_SEED_REQUEST_INTERVAL {
            return;
        }

        self.last_dns_seed_call = Instant::now();

        info!(
            "Floresta has been running for a while without enough addresses, requesting more from DNS seeds"
        );
        try_and_log!(self.get_peers_from_dns());
    }

    /// If we don't have any peers, we use the hardcoded addresses.
    ///
    ///
    /// This is only done if we don't have any peers for a long time, or we
    /// can't find a Utreexo peer in a context we need them. This function
    /// won't do anything if `--connect` was used
    fn maybe_use_hardcoded_addresses(&mut self) {
        if self.has_fixed_peers() {
            return;
        }

        if self.used_fixed_addresses {
            return;
        }

        if self.address_man.enough_addresses() {
            return;
        }

        let wait = HARDCODED_ADDRESSES_GRACE_PERIOD;
        if self.startup_time.elapsed() < wait {
            return;
        }

        self.used_fixed_addresses = true;

        info!("No peers found, using hardcoded addresses");
        let net = self.network;
        self.address_man.add_fixed_addresses(net);
    }

    pub(crate) fn init_peers(&mut self) -> Result<(), WireError> {
        let anchors = self.common.address_man.start_addr_man(&self.common.datadir);
        let enough_addresses = self.common.address_man.enough_addresses();

        if !self.config.disable_dns_seeds && !enough_addresses {
            self.get_peers_from_dns()?;
            self.last_dns_seed_call = Instant::now();
        }

        for address in anchors {
            self.open_connection(
                ConnectionKind::Regular(service_flags::UTREEXO.into()),
                address,
                // Using V1 transport fallback as utreexo nodes have limited support
                true,
            )?;
        }

        Ok(())
    }

    // === MAINTENANCE ===

    pub(crate) fn maybe_open_connection(
        &mut self,
        required_service: ServiceFlags,
    ) -> Result<(), WireError> {
        // try to connect with manually added peers
        self.maybe_open_connection_with_added_peers()?;
        if self.connected_peers() >= T::MAX_OUTGOING_PEERS {
            return Ok(());
        }

        let connection_kind = ConnectionKind::Regular(required_service);

        // If the user passes in `--connect` cli arguments, we only connect with
        // those peers. Try to (re)connect as many as we are missing.
        if self.has_fixed_peers() {
            let missing = self.fixed_peers.len().saturating_sub(self.peers.len());
            for _ in 0..missing {
                if let Err(e) = self.create_connection(connection_kind) {
                    debug!("Failed to connect to fixed peer: {e:?}");
                }
            }
            return Ok(());
        }

        // If we've tried getting some connections, but the addresses we have are not
        // working. Try getting some more addresses from DNS
        self.maybe_ask_dns_seed_for_addresses();
        self.maybe_use_hardcoded_addresses();

        for _ in 0..T::NEW_CONNECTIONS_BATCH_SIZE {
            // Ignore the error so we don't break out of the loop
            let _ = self.create_connection(connection_kind);
        }

        Ok(())
    }

    /// Try disconnecting the slowest non-protected-service peer.
    ///
    /// For peers that don't have any of the `protected_services`, we disconnect the
    /// highest-latency one when any of these hold:
    /// - Its latency is more than 1.5x the median latency across eligible peers.
    /// - `always` is true.
    /// - Or randomly, with 5% probability.
    pub(crate) fn maybe_disconnect_slowest_peer(
        &self,
        protected_services: &[ServiceFlags],
        always: bool,
    ) -> Result<(), WireError> {
        /// Require at least 5 samples to compute the median latency. Fewer samples won't
        /// output a meaningful median value.
        const MIN_SAMPLES: usize = 5;
        /// If the slowest peer has higher latency than 1.5x the median latency, disconnect it.
        /// This is the same value as the `libbitcoin` default `allowed_deviation`.
        const ALLOWED_DEVIATION: f64 = 1.5;

        // Use latency samples only from regular, non-protected-service peers.
        let is_eligible_peer = |peer: &LocalPeerView| {
            peer.is_regular_peer() && !peer.has_any_service(protected_services)
        };

        let mut samples: Vec<_> = self
            .peers
            .iter()
            .filter_map(|(&peer_id, peer)| {
                let latency = peer.message_times.value()?;

                is_eligible_peer(peer).then_some((peer_id, latency))
            })
            .collect();

        if samples.len() < MIN_SAMPLES {
            return Ok(());
        }

        // Sort by latency, then get the median time and the slowest peer
        samples.sort_by(|a, b| a.1.total_cmp(&b.1));

        let (_, median_latency) = samples[samples.len() / 2];
        let (slowest_peer_id, slowest_latency) = samples[samples.len() - 1];

        let should_disconnect = always
            || slowest_latency > ALLOWED_DEVIATION * median_latency
            || rand::rng().random_ratio(1, 20);

        if !should_disconnect {
            return Ok(());
        }

        info!("Disconnecting slowest non-protected peer: {slowest_peer_id}");
        self.send_to_peer(slowest_peer_id, NodeRequest::Shutdown)
    }

    pub(crate) fn maybe_open_connection_with_added_peers(&mut self) -> Result<(), WireError> {
        if self.added_peers.is_empty() {
            return Ok(());
        }
        let peers_count = self.peer_id_count;
        for added_peer in self.added_peers.clone() {
            let matching_peer = self
                .peers
                .values()
                .find(|peer| *peer.address.as_bitcoin_socket_addr() == added_peer.address);

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
                    peers_count as usize,
                );

                // Finally, open the connection with the node
                self.open_connection(ConnectionKind::Manual, address, added_peer.v1_fallback)?
            }
        }
        Ok(())
    }
}
