// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Transaction;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::message_blockdata::Inventory;
use floresta_chain::ChainBackend;
use floresta_common::service_flags;
use floresta_common::service_flags_strings;
use floresta_common::try_and_log;
use rand::prelude::IteratorRandom;
use rand::seq::IndexedRandom;
use tracing::debug;
use tracing::info;
use tracing::warn;

use super::ConnectionKind;
use super::InflightRequests;
use super::LocalPeerView;
use super::NodeRequest;
use super::PeerStatus;
use super::UtreexoNode;
use crate::address_man::AddressState;
use crate::address_man::LocalAddress;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::block_proof::Bitmap;
use crate::node::running_ctx::RunningNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_handle::NodeResponse;
use crate::node_handle::UserRequest;
use crate::node_interface::PeerInfo;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::peer::Version;

#[derive(Debug, Clone)]
/// A simple struct of added peers, used to track the ones we added manually by `addnode <ip:port> add` command.
pub struct AddedPeerInfo {
    /// The address of the peer
    pub(crate) address: BitcoinSocketAddr,

    /// Whether we should allow V1 fallback for this connection
    pub(crate) v1_fallback: bool,
}

impl<T, Chain> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
{
    // === SENDING TO PEERS ===

    /// Peers ready for ordinary requests; feelers and temporary connections have dedicated work.
    fn ready_peers(&self, service: ServiceFlags) -> impl Iterator<Item = (PeerId, &LocalPeerView)> {
        self.peers
            .iter()
            .filter(move |(_, peer)| {
                peer.state == PeerStatus::Ready
                    && peer.is_long_lived()
                    && peer.services.has(service)
            })
            .map(|(&id, peer)| (id, peer))
    }

    /// Returns the median response time among eligible peers, in milliseconds.
    pub(crate) fn median_peer_latency(&self, service: ServiceFlags) -> Option<f64> {
        let mut times = self
            .ready_peers(service)
            .filter_map(|(_, peer)| peer.message_times.value())
            .collect::<Vec<_>>();
        times.sort_by(f64::total_cmp);
        times.get(times.len() / 2).copied()
    }

    /// Returns the arithmetic mean response time among eligible peers.
    pub(crate) fn average_peer_latency(&self, service: ServiceFlags) -> Option<Duration> {
        let (total_ms, peers) = self
            .ready_peers(service)
            .filter_map(|(_, peer)| peer.message_times.value())
            .fold((0.0, 0_u32), |(total, count), latency| {
                (total + latency, count + 1)
            });
        (peers != 0).then(|| Duration::from_secs_f64(total_ms / f64::from(peers) / 1_000.0))
    }

    /// Picks a ready peer, weighted by inverse general response latency.
    ///
    /// A peer with half the latency is twice as likely to receive the next request.
    fn choose_peer_by_latency(&self, service: ServiceFlags) -> Option<PeerId> {
        let initial_time = self.median_peer_latency(service).unwrap_or(1_000.0);
        let candidates: Vec<_> = self.ready_peers(service).collect();
        let (id, _) = candidates
            .choose_weighted(&mut rand::rng(), |(_, peer)| {
                // Keep the weight finite even if the measured duration is zero.
                1.0 / peer.message_times.value().unwrap_or(initial_time).max(1e-9)
            })
            .ok()?;
        Some(*id)
    }

    /// Picks the least-loaded block provider, breaking ties by general response latency.
    fn choose_block_peer(&self, service: ServiceFlags) -> Option<PeerId> {
        // Count current assignments once; replies, retries and disconnects already update this map.
        let mut pending: HashMap<PeerId, usize> = HashMap::new();
        for (request, (peer, _)) in &self.inflight {
            if matches!(request, InflightRequests::Blocks(_)) {
                *pending.entry(*peer).or_default() += 1;
            }
        }

        let initial_time = self.median_peer_latency(service).unwrap_or(1_000.0);
        self.ready_peers(service)
            .map(|(id, peer)| {
                (
                    id,
                    pending.get(&id).copied().unwrap_or(0),
                    peer.message_times.value().unwrap_or(initial_time),
                )
            })
            .min_by(|(_, pending_a, latency_a), (_, pending_b, latency_b)| {
                pending_a
                    .cmp(pending_b)
                    .then_with(|| latency_a.total_cmp(latency_b))
            })
            .map(|(id, _, _)| id)
    }

    /// Whether the node was configured with a fixed peer list (via `--connect`).
    ///
    /// When this is true, we connect *only* to those peers and skip the usual
    /// peer-discovery paths (address manager, feelers, hardcoded fallback).
    pub(crate) fn has_fixed_peers(&self) -> bool {
        !self.fixed_peers.is_empty()
    }

    /// Returns how many connected peers we have.
    ///
    /// This function will only count peers that completed handshake and are ready
    /// to be used.
    pub(crate) fn connected_peers(&self) -> usize {
        self.peers
            .values()
            .filter(|p| p.state == PeerStatus::Ready && p.is_long_lived())
            .count()
    }

    /// Balances block requests by outstanding blocks; other requests use latency weighting.
    ///
    /// Faster block providers clear their outstanding work sooner and earn more refills.
    ///
    /// Returns an error if no ready peer has `required_service` or if sending the request failed.
    pub(crate) fn send_to_fast_peer(
        &self,
        request: NodeRequest,
        required_service: ServiceFlags,
    ) -> Result<PeerId, WireError> {
        let peer_id = match request {
            NodeRequest::GetBlock(..) => self.choose_block_peer(required_service),
            _ => self.choose_peer_by_latency(required_service),
        }
        .ok_or(WireError::NoPeersAvailable)?;

        self.peers[&peer_id].channel.send(request)?;

        Ok(peer_id)
    }

    /// Sends to a uniformly chosen ready, long-lived peer supporting `required_service`.
    pub(crate) fn send_to_random_peer(
        &self,
        req: NodeRequest,
        required_service: ServiceFlags,
    ) -> Result<PeerId, WireError> {
        let (peer_id, peer) = self
            .ready_peers(required_service)
            .choose(&mut rand::rng())
            .ok_or(WireError::NoPeersAvailable)?;

        peer.channel.send(req)?;
        Ok(peer_id)
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

    /// Sends the same request to all connected peers
    ///
    /// This function is best-effort, meaning that some peers may not receive the request if they
    /// are disconnected or if there is an error sending the request. We intentionally won't
    /// propagate the error to the caller, as this would request an early return from the function,
    /// which would prevent us from sending the request to the peers the comes after the first
    /// erroing one.
    pub(crate) fn broadcast_to_peers(&mut self, request: NodeRequest) {
        for peer in self.peers.values() {
            if peer.state != PeerStatus::Ready {
                continue;
            }

            if let Err(err) = peer.channel.send(request.clone()) {
                warn!("Failed to send request to peer {}: {err}", peer.address);
            }
        }
    }

    pub(crate) fn ask_for_addresses(&mut self) -> Result<(), WireError> {
        let _ = self.send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)?;
        Ok(())
    }

    // === PEER LIFECYCLE ===

    fn is_peer_good(peer: &LocalPeerView, needs: ServiceFlags) -> bool {
        if peer.state == PeerStatus::Banned {
            return false;
        }

        peer.services.has(needs)
    }

    pub(crate) fn handle_peer_ready(
        &mut self,
        peer: u32,
        mut version: Version,
    ) -> Result<(), WireError> {
        self.inflight.remove(&InflightRequests::Connect(peer));

        // Mark this peer as ready to communicate with
        self.peers.entry(peer).and_modify(|p| {
            p.state = PeerStatus::Ready;
        });

        // Ask for new addresses to populate our address manager
        self.send_to_peer(peer, NodeRequest::GetAddresses)?;
        self.inflight
            .insert(InflightRequests::GetAddresses, (peer, Instant::now()));

        let good_peers_count = self.connected_peers();
        if good_peers_count > T::MAX_OUTGOING_PEERS {
            // We allow utreexo, extra and manual peers to bypass our connection limits
            let is_utreexo_peer = matches!(version.kind, ConnectionKind::Regular(services) if services.has(service_flags::UTREEXO.into()));
            let is_manual_peer = version.kind == ConnectionKind::Manual;
            let is_extra = version.kind == ConnectionKind::Extra;

            if !(is_utreexo_peer || is_manual_peer || is_extra) {
                debug!(
                    "Already have {} peers, disconnecting peer to avoid blowing up our max of {}",
                    good_peers_count,
                    T::MAX_OUTGOING_PEERS
                );

                // If a peer exceeds our max, just turn them into a feeler so we can receive their
                // AddrV2 message and then disconnect.
                self.peers.entry(peer).and_modify(|p| {
                    p.kind = ConnectionKind::Feeler;
                });

                version.kind = ConnectionKind::Feeler;
            }
        }

        if version.kind == ConnectionKind::Feeler {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

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
            peer_data.services = version.services;
            peer_data.user_agent.clone_from(&version.user_agent);
            peer_data.height = version.blocks;
            peer_data.time_offset = version.time_offset;
            peer_data.transport_protocol = version.transport_protocol;

            // If this peer doesn't have basic services, we disconnect it
            if let ConnectionKind::Regular(needs) = version.kind {
                if !Self::is_peer_good(peer_data, needs) {
                    info!(
                        "Disconnecting peer {peer} for not having the required services. has={} needs={}",
                        peer_data.services, needs
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

            if peer_data
                .services
                .has(service_flags::UTREEXO_ARCHIVE.into())
            {
                self.common
                    .peer_by_service
                    .entry(service_flags::UTREEXO_ARCHIVE.into())
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

    /// Handles a NOTFOUND inventory by completing any matching inflight user request with `None`.
    pub(crate) fn handle_notfound_msg(&mut self, inv: Inventory) -> Result<(), WireError> {
        match inv {
            Inventory::Error => {}

            Inventory::Block(block)
            | Inventory::WitnessBlock(block)
            | Inventory::CompactBlock(block) => {
                if let Some(request) = self
                    .inflight_user_requests
                    .remove(&UserRequest::Block(block))
                {
                    request
                        .2
                        .send(NodeResponse::Block(None))
                        .map_err(|_| WireError::ResponseSendError)?;
                }
            }

            Inventory::WitnessTransaction(tx) | Inventory::Transaction(tx) => {
                if let Some(request) = self
                    .inflight_user_requests
                    .remove(&UserRequest::MempoolTransaction(tx))
                {
                    request
                        .2
                        .send(NodeResponse::MempoolTransaction(None))
                        .map_err(|_| WireError::ResponseSendError)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Handles an incoming mempool transaction by completing any matching inflight user request.
    pub(crate) fn handle_tx_msg(&mut self, tx: Transaction) -> Result<(), WireError> {
        let txid = tx.compute_txid();
        debug!("saw a mempool transaction with txid={txid}");

        if let Some(request) = self
            .inflight_user_requests
            .remove(&UserRequest::MempoolTransaction(txid))
        {
            request
                .2
                .send(NodeResponse::MempoolTransaction(Some(tx)))
                .map_err(|_| WireError::ResponseSendError)?;
        }

        Ok(())
    }

    /// Handles peer messages where behavior is common to all node contexts, returning `Some` only
    /// for peer messages that require context-specific handling.
    pub(crate) fn handle_peer_msg_common(
        &mut self,
        msg: PeerMessages,
        peer: PeerId,
    ) -> Result<Option<PeerMessages>, WireError> {
        match msg {
            PeerMessages::Addr(addresses) => {
                self.handle_addresses_from_peer(peer, addresses)?;
                Ok(None)
            }
            PeerMessages::NotFound(inv) => {
                self.handle_notfound_msg(inv)?;
                Ok(None)
            }
            PeerMessages::Transaction(tx) => {
                self.handle_tx_msg(tx)?;
                Ok(None)
            }
            PeerMessages::UtreexoState(_) => {
                warn!("Utreexo state received from peer {peer}, but we didn't ask");
                self.increase_banscore(peer, 5)?;
                Ok(None)
            }
            PeerMessages::CFHeaders(cfheaders) => {
                let req = self.inflight_user_requests.iter().find_map(|(req, _)| {
                    if let UserRequest::GetCFilterHeaders { stop_hash, .. } = req {
                        if *stop_hash == cfheaders.stop_hash {
                            return Some(req.clone());
                        }
                    }

                    None
                });

                match req {
                    Some(req) => {
                        let final_req = self.inflight_user_requests.remove(&req).unwrap();
                        let _ = final_req.2.send(NodeResponse::CFilterHeaders(cfheaders));
                    }

                    None => {
                        warn!("Peer {peer} sent us cfheaders, but we didn't request it");
                        self.increase_banscore(peer, 5)?;
                    }
                }

                Ok(None)
            }
            _ => Ok(Some(msg)),
        }
    }

    pub(crate) fn handle_disconnection(&mut self, peer: u32, idx: usize) -> Result<(), WireError> {
        if let Some(p) = self.peers.remove(&peer) {
            if p.is_long_lived() && p.state == PeerStatus::Ready {
                info!("Peer disconnected: {peer}");
            }

            std::mem::drop(p.channel);

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
        for v in self.peer_by_service.values_mut() {
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

            if let Err(e) = self.redo_inflight_request(&req.0) {
                // CRITICAL: never drop the request, so we retry it later
                self.inflight.insert(req.0, req.1);
                return Err(e);
            }
        }

        #[cfg(feature = "metrics")]
        self.update_peer_metrics();

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

        // Manual connections are exempt from being punished
        if peer.is_manual_peer() {
            return Ok(());
        }

        peer.banscore += factor;

        // This peer is misbehaving too often, ban it
        let is_misbehaving = peer.banscore >= self.common.max_banscore;
        // extra peers should be banned immediately
        let is_extra = peer.kind == ConnectionKind::Extra;

        if is_misbehaving || is_extra {
            warn!("banning peer {peer_id} for misbehaving");
            self.disconnect_and_ban(peer_id)?;
            return Ok(());
        }

        debug!("increasing banscore for peer {peer_id}");

        Ok(())
    }

    /// Disconnects a peer and bans it for `T::BAN_TIME`.
    pub(crate) fn disconnect_and_ban(&mut self, peer: PeerId) -> Result<(), WireError> {
        if let Some(peer) = self.peers.get_mut(&peer) {
            // Manual connections are exempt from being punished
            if peer.is_manual_peer() {
                return Ok(());
            }

            // `handle_disconnection` will mark the address as banned when `Peer` object return
            peer.state = PeerStatus::Banned;
        }

        self.send_to_peer(peer, NodeRequest::Shutdown)?;
        Ok(())
    }

    /// Tries to randomly disconnect up to `n` non-protected-feature peers.
    pub(crate) fn disconnect_random_peers(&self, n: usize, protected_services: &[ServiceFlags]) {
        let mut rng = rand::rng();

        let peers = self
            .peers
            .values()
            .filter(|peer| peer.is_regular_peer() && !peer.has_any_service(protected_services))
            .sample(&mut rng, n);

        for peer in peers {
            let _ = peer.channel.send(NodeRequest::Shutdown);
        }
    }

    /// Checks whether some of our inflight requests have timed out.
    ///
    /// This function will check if any of our inflight requests have timed out,
    /// and if so it will remove them from the inflight list, apply the timeout
    /// penalty, and resend the request to a peer.
    pub(crate) fn check_for_timeout(&mut self) -> Result<(), WireError> {
        let now = Instant::now();
        let average_peer_time = self.average_peer_latency(ServiceFlags::NETWORK);
        let block_timeout = self.context.block_request_timeout(average_peer_time);

        let timed_out_fn = |req: &InflightRequests, time: &Instant| {
            let age = now.duration_since(*time);
            match req {
                InflightRequests::Connect(_)
                    if age > Duration::from_secs(T::CONNECTION_TIMEOUT) =>
                {
                    Some(req.clone())
                }
                InflightRequests::Blocks(_) if age > block_timeout => Some(req.clone()),
                _ if age > Duration::from_secs(T::REQUEST_TIMEOUT) => Some(req.clone()),
                _ => None,
            }
        };

        let timed_out = self
            .inflight
            .iter()
            .filter_map(|(req, (_, time))| timed_out_fn(req, time))
            .collect::<Vec<_>>();

        let mut retry_error = None;
        let mut block_timeouts_by_peer: HashMap<PeerId, usize> = HashMap::new();
        let mut other_timeouts = 0;
        let mut retry_failures = 0;
        let mut oldest_timeout_secs = 0;
        for req in timed_out {
            let Some((peer, time)) = self.inflight.remove(&req) else {
                continue;
            };

            if matches!(req, InflightRequests::Blocks(_)) {
                *block_timeouts_by_peer.entry(peer).or_default() += 1;
            } else {
                other_timeouts += 1;
            }
            // Include all expired request kinds in the oldest age and retry-failure totals.
            oldest_timeout_secs =
                oldest_timeout_secs.max(now.saturating_duration_since(time).as_secs());

            // If a feeler connection times out, we ban them at the first message
            if let Some(peer_data) = self.peers.get(&peer) {
                if peer_data.kind == ConnectionKind::Feeler {
                    debug!("Feeler peer {peer} timed out request");
                    self.send_to_peer(peer, NodeRequest::Shutdown)?;
                    self.peers.remove(&peer);
                    continue;
                }
            }

            if let InflightRequests::Connect(_) = req {
                // ignore the output as it might fail due to the task being cancelled
                let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
                self.peers.remove(&peer);
                continue;
            }
            debug!("Request timed out: {req:?}");
            if matches!(
                req,
                InflightRequests::Blocks(_)
                    | InflightRequests::Headers
                    | InflightRequests::GetFilters
                    | InflightRequests::UtreexoState(_)
            ) && let Some(peer) = self.peers.get_mut(&peer)
            {
                // Silence is at least this slow; score it now even if no response ever arrives.
                let timeout = if matches!(req, InflightRequests::Blocks(_)) {
                    block_timeout
                } else {
                    Duration::from_secs(T::REQUEST_TIMEOUT)
                };
                peer.message_times.add(timeout.as_secs_f64() * 1_000.0);
            }
            // Adaptive download probing may delay honest block responses.
            if T::PENALIZE_BLOCK_TIMEOUT || !matches!(req, InflightRequests::Blocks(_)) {
                try_and_log!(self.increase_banscore(peer, 1));
            }

            let exclude_timed_out_peer =
                !T::PENALIZE_BLOCK_TIMEOUT && matches!(req, InflightRequests::Blocks(_));
            if exclude_timed_out_peer {
                // SwiftSync retries are latency probes, not peer punishments. Exclude this
                // assignment before selecting a replacement; disconnect cleanup runs later.
                let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
                if let Some(peer) = self.peers.get_mut(&peer) {
                    peer.state = PeerStatus::Awaiting;
                }
            } else {
                self.send_to_peer(peer, NodeRequest::Shutdown)?;
            }

            if let Err(e) = self.redo_inflight_request(&req) {
                retry_failures += 1;
                // CRITICAL: never drop the request, so we retry it later.
                //
                // Note that this failed `redo_inflight_request` will make us punish the
                // unresponsive peer again on each tick while retries keep failing. We
                // could keep track of requests that timed-out and failed to be retried,
                // to punish them once, but this is uncommon and punishments are small.
                self.inflight.insert(req, (peer, time));
                // One failed dispatch must not hide the remaining peers' timeouts.
                retry_error = Some(e);
            }
        }

        // Retries replace request timestamps, so report the expired assignments before losing them.
        if !block_timeouts_by_peer.is_empty() {
            let block_timeouts = block_timeouts_by_peer.values().sum::<usize>();
            let mut block_timeouts_by_peer = block_timeouts_by_peer.into_iter().collect::<Vec<_>>();
            block_timeouts_by_peer.sort_unstable();
            info!(
                "Block request timeouts: blocks={block_timeouts} other_requests={other_timeouts} \
                 retry_failures={retry_failures} oldest_age_secs={oldest_timeout_secs} \
                 dynamic_timeout_ms={:.1} average_peer_ms={:?} \
                 block_timeouts_by_peer={block_timeouts_by_peer:?}",
                block_timeout.as_secs_f64() * 1_000.0,
                average_peer_time.map(|time| time.as_secs_f64() * 1_000.0),
            );
        }

        retry_error.map_or(Ok(()), Err)
    }

    pub(crate) fn handle_addresses_from_peer(
        &mut self,
        peer: u32,
        addresses: Vec<AddrV2Message>,
    ) -> Result<(), WireError> {
        /// Don't allow addresses whose timestamp are more than 10 minutes into the future, or an
        /// attacker could make us drop all addresses from our address manager with their own.
        const ADDRESS_TIME_CUTOFF: Duration = Duration::from_secs(600); // 10 minutes

        debug!("Got {} addresses from peer {}", addresses.len(), peer);
        self.inflight.remove(&InflightRequests::GetAddresses);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System clock isn't working");

        let cutoff = now + ADDRESS_TIME_CUTOFF;
        let good_addresses: Vec<_> = addresses
            .into_iter()
            .filter(|addr| u64::from(addr.time) < cutoff.as_secs())
            .map(|addr| addr.into())
            .collect();

        self.address_man.push_addresses(&good_addresses);

        // For feeler peers, we ask for addresses to populate our address manager,
        // then disconnect them
        let Some(peer_data) = self.peers.get(&peer) else {
            return Ok(());
        };

        if matches!(peer_data.kind, ConnectionKind::Feeler) {
            self.send_to_peer(peer, NodeRequest::Shutdown)?;
        }

        Ok(())
    }

    pub(crate) fn redo_inflight_request(
        &mut self,
        req: &InflightRequests,
    ) -> Result<(), WireError> {
        match req {
            InflightRequests::UtreexoProof(block_hash) => {
                if !self.has_utreexo_peers() {
                    return Ok(());
                }

                if !self.blocks.contains_key(block_hash) {
                    // If we don't have the block anymore, we can't ask for the proof
                    return Ok(());
                }

                if self.inflight.contains_key(req) {
                    // If we already have an inflight request for this block, we don't need to redo it
                    return Ok(());
                }

                let peer = self.send_to_fast_peer(
                    NodeRequest::GetBlockProof((*block_hash, Bitmap::new(), Bitmap::new())),
                    service_flags::UTREEXO.into(),
                )?;

                self.inflight.insert(
                    InflightRequests::UtreexoProof(*block_hash),
                    (peer, Instant::now()),
                );
            }

            InflightRequests::Blocks(block) => {
                self.request_blocks(vec![*block])?;
            }

            InflightRequests::Headers => {
                let locator = self.chain.get_block_locator()?;
                let peer =
                    self.send_to_fast_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)?;

                self.inflight
                    .insert(InflightRequests::Headers, (peer, Instant::now()));
            }

            InflightRequests::UtreexoState(_) => {
                let peer = self.send_to_fast_peer(
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
                let peer = self.send_to_fast_peer(
                    NodeRequest::GetFilter((self.chain.get_block_hash(0).unwrap(), 0)),
                    ServiceFlags::COMPACT_FILTERS,
                )?;

                self.inflight
                    .insert(InflightRequests::GetFilters, (peer, Instant::now()));
            }

            InflightRequests::Connect(_) | InflightRequests::GetAddresses => {
                // We don't need to do anything here
            }
        }

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

    // === METRICS AND HELPERS ===

    /// Records general response latency, only for the current requester.
    ///
    /// Batched blocks and filters use each reply's elapsed time from the shared request timestamp.
    pub(crate) fn register_message_time(
        &mut self,
        notification: &PeerMessages,
        peer: PeerId,
        read_at: Instant,
    ) -> Option<()> {
        let request = match notification {
            PeerMessages::Block(block) => InflightRequests::Blocks(block.block_hash()),
            PeerMessages::Ready(_) => InflightRequests::Connect(peer),
            PeerMessages::Headers(_) => InflightRequests::Headers,
            PeerMessages::BlockFilter(_) => InflightRequests::GetFilters,
            PeerMessages::UtreexoState(_) => InflightRequests::UtreexoState(peer),
            _ => return None,
        };
        let (request_peer, sent_at) = self.inflight.get(&request)?;

        // A late reply must not use another peer's retry timestamp.
        if *request_peer != peer {
            return None;
        }

        // A reply may be read just before a retry, so `sent_at` will be
        // overridden with the retry instant, which makes `read_at < sent_at`.
        // Skip that stale sample instead of recording zero latency.
        let elapsed = read_at.checked_duration_since(*sent_at)?.as_secs_f64();
        if let Some(peer) = self.peers.get_mut(&peer) {
            peer.message_times.add(elapsed * 1_000.0); // milliseconds
        }

        #[cfg(feature = "metrics")]
        {
            use floresta_metrics::get_metrics;
            let metrics = get_metrics();

            metrics.message_times.observe(elapsed);
        }

        Some(())
    }

    #[cfg(feature = "metrics")]
    pub(crate) fn update_peer_metrics(&self) {
        use floresta_metrics::get_metrics;

        let metrics = get_metrics();
        metrics.peer_count.set(self.peer_ids.len() as f64);
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

    pub(crate) fn get_peer_info(&self, peer_id: &u32) -> Option<PeerInfo> {
        let peer = self.peers.get(peer_id)?;
        Some(PeerInfo {
            id: *peer_id,
            address: peer.address.as_bitcoin_socket_addr().clone(),
            services: format!("{:016x}", peer.services.to_u64()),
            services_names: service_flags_strings(&peer.services),
            relay_txs: false,
            user_agent: peer.user_agent.clone(),
            inbound: false,
            bip152_hb_to: false,
            bip152_hb_from: false,
            initial_height: peer.height,
            time_offset: peer.time_offset,
            state: peer.state,
            kind: peer.kind,
            permissions: Vec::new(),
            transport_protocol: peer.transport_protocol,
        })
    }

    // === ADDNODE ===

    /// Handles addnode-RPC `Add` requests, adding a new peer to the `added_peers` list. This means
    /// the peer is marked as a "manually added peer". We then try to connect to it, or retry later.
    pub fn handle_addnode_add_peer(
        &mut self,
        peer_address: BitcoinSocketAddr,
        v2_transport: bool,
    ) -> Result<(), WireError> {
        // See https://github.com/bitcoin/bitcoin/blob/8309a9747a8df96517970841b3648937d05939a3/src/net.cpp#L3558

        // Add this address to our address manager for later
        // assume it has the bare-minimum services, otherwise `push_addresses` will ignore it
        let mut local_address = LocalAddress::from(peer_address.clone());
        debug!("Adding node {}", local_address);

        local_address.set_services(ServiceFlags::NETWORK_LIMITED | ServiceFlags::WITNESS);

        // Check if the peer already exists
        if self
            .added_peers
            .iter()
            .any(|peer_info| peer_address == peer_info.address)
        {
            return Err(WireError::PeerAlreadyExists(local_address));
        }

        self.address_man.push_addresses(&[local_address]);

        // Add a simple reference to the peer
        self.added_peers.push(AddedPeerInfo {
            address: peer_address,
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
    pub fn handle_addnode_remove_peer(&mut self, addr: BitcoinSocketAddr) -> Result<(), WireError> {
        debug!("Trying to remove peer {addr}");

        let index = self
            .added_peers
            .iter()
            .position(|info| addr == info.address)
            .ok_or(WireError::PeerNotFoundAtAddress(addr))?;

        self.added_peers.remove(index);

        Ok(())
    }

    /// Handles the node request for immediate disconnection from a peer.
    pub fn handle_disconnect_peer(&mut self, addr: BitcoinSocketAddr) -> Result<(), WireError> {
        // Get the peer's index in the [`AddressMan`]'s list, if it exists.
        let peer_id = self
            .peers
            .iter()
            .find_map(|(&id, peer)| (*peer.address.as_bitcoin_socket_addr() == addr).then_some(id))
            .ok_or(WireError::PeerNotFoundAtAddress(addr))?;

        self.send_to_peer(peer_id, NodeRequest::Shutdown)
    }

    /// Handles addnode onetry requests, connecting to the node and this will try to connect to the given address and port.
    /// If it's successful, it will add the node to the peers list, but not to the added_peers list (e.g., it won't be reconnected if disconnected).
    pub fn handle_addnode_onetry_peer(
        &mut self,
        peer_address: BitcoinSocketAddr,
        v2_transport: bool,
    ) -> Result<(), WireError> {
        let kind = ConnectionKind::Manual;

        // Add this address to our address manager for later
        // assume it has the bare-minimum services, otherwise `push_addresses` will ignore it
        let mut local_address = LocalAddress::from(peer_address.clone());
        debug!("Creating an one-try connection with {local_address}");

        // Check if the peer already exists
        if self
            .peers
            .iter()
            .any(|(_, peer_info)| *peer_info.address.as_bitcoin_socket_addr() == peer_address)
        {
            return Err(WireError::PeerAlreadyExists(local_address));
        }

        local_address.set_services(ServiceFlags::NETWORK_LIMITED | ServiceFlags::WITNESS);

        self.address_man.push_addresses(&[local_address.clone()]);
        // Return true if exists or false if anything fails during connection
        // We allow V1 fallback iff the `v2` flag is not set
        self.open_connection(kind, local_address, !v2_transport)
    }
}
