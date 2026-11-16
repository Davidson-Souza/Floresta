// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::BlockHash;
use bitcoin::Transaction;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::message_blockdata::Inventory;
use floresta_chain::ChainBackend;
use floresta_common::service_flags;
use floresta_common::service_flags_strings;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::IteratorRandom;
use rand::seq::IndexedRandom;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use super::ConnectionKind;
use super::InflightRequests;
use super::LocalPeerView;
use super::NodeRequest;
use super::PeerStatus;
use super::UtreexoNode;
use crate::address_man::AddressMan;
use crate::address_man::AddressState;
use crate::address_man::LocalAddress;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::block_proof::Bitmap;
use crate::node::running_ctx::RunningNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_handle::NodeResponse;
use crate::node_handle::PeerInfo;
use crate::node_handle::UserRequest;
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

    /// Picks a `Ready` peer supporting `service`, biased toward lower message latency.
    ///
    /// Each candidate weight is computed as `lowest_time / time_i`. For instance, if we have two
    /// candidates with latencies of 50ms and 100ms, weights are 1.0 and 0.5 respectively, and the
    /// probability of being chosen is 2/3 and 1/3. If `exclude` is `Some`, that peer is never
    /// considered a candidate.
    fn choose_peer_by_latency(
        &self,
        service: ServiceFlags,
        exclude: Option<PeerId>,
    ) -> Option<(&PeerId, &LocalPeerView)> {
        // Epsilon is a small positive floor for `f64`. If by any chance a peer has extremely low
        // message latency, we clamp it to `EPS` so `lowest_time / time_i` stays finite and stable.
        const EPS: f64 = 1e-9;

        let candidates: Vec<(&PeerId, &LocalPeerView, f64)> = self
            .peers
            .iter()
            .filter(|(id, peer)| {
                Some(**id) != exclude
                    && peer.services.has(service)
                    && peer.state == PeerStatus::Ready
            })
            .filter_map(|(id, peer)| {
                // Get the average message latency from each peer
                let Some(t) = peer.message_times.value() else {
                    error!("Peer {peer:?} has no message times");
                    return None;
                };
                Some((id, peer, t.max(EPS)))
            })
            .collect();

        // Fastest observed time among candidates. Returns `None` if no candidate is found.
        let lowest_time = candidates.iter().map(|(_, _, t)| *t).reduce(f64::min)?;

        let weights: Vec<f64> = candidates
            .iter()
            .map(|(_, _, time)| lowest_time / time)
            .collect();

        let dist = WeightedIndex::new(&weights).ok()?;
        let idx = dist.sample(&mut rand::rng());

        let (id, peer, _) = candidates[idx];
        Some((id, peer))
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
            .filter(|p| p.state == PeerStatus::Ready && p.kind.is_long_lived())
            .count()
    }

    /// Sends a request to an initialized peer that supports `required_service`, chosen via a
    /// latency-weighted distribution (lower latency => more likely).
    ///
    /// Returns an error if no ready peer has `required_service` or if sending the request failed.
    pub(crate) fn send_to_fast_peer(
        &self,
        request: NodeRequest,
        required_service: ServiceFlags,
    ) -> Result<PeerId, WireError> {
        self.send_to_fast_peer_inner(request, required_service, None)
    }

    /// Same as [`Self::send_to_fast_peer`], but never picks the `excluded` peer.
    ///
    /// Returns [`WireError::NoPeersAvailable`] if no other eligible peer exists.
    pub(crate) fn send_to_fast_peer_excluding(
        &self,
        request: NodeRequest,
        required_service: ServiceFlags,
        excluded: PeerId,
    ) -> Result<PeerId, WireError> {
        self.send_to_fast_peer_inner(request, required_service, Some(excluded))
    }

    fn send_to_fast_peer_inner(
        &self,
        request: NodeRequest,
        required_service: ServiceFlags,
        exclude: Option<PeerId>,
    ) -> Result<PeerId, WireError> {
        let (peer_id, peer) = self
            .choose_peer_by_latency(required_service, exclude)
            .ok_or(WireError::NoPeersAvailable)?;

        peer.channel.send(request)?;

        Ok(*peer_id)
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
            .choose(&mut rand::rng())
            .expect("infallible: we checked that peers isn't empty");

        self.peers
            .get(peer)
            .ok_or(WireError::NoPeersAvailable)?
            .channel
            .send(req)
            .map_err(WireError::ChannelSend)?;

        Ok(*peer)
    }

    #[inline]
    pub(crate) fn send_to_random_peer_with_predicate(
        &mut self,
        req: NodeRequest,
        required_service: ServiceFlags,
        predicate: impl Fn(&LocalPeerView) -> bool,
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

        let (peer_id, peer) = peers
            .iter()
            .filter_map(|peer_id| self.peers.get(peer_id).map(|peer| (peer_id, peer)))
            .filter(|(_, peer)| predicate(peer))
            .choose(&mut rand::rng())
            .ok_or(WireError::NoPeersAvailable)?;

        peer.channel.send(req).map_err(WireError::ChannelSend)?;

        Ok(*peer_id)
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

            if request.is_broadcast_transaction() && !peer.kind.is_full_relay() {
                continue;
            }

            if let Err(err) = peer.channel.send(request.clone()) {
                warn!("Failed to send request to peer {}: {err}", peer.address);
            }
        }
    }

    pub(crate) fn ask_for_addresses(&mut self) -> Result<(), WireError> {
        self.send_to_random_peer_with_predicate(
            NodeRequest::GetAddresses,
            ServiceFlags::NONE,
            |peer| peer.state == PeerStatus::Ready && peer.kind.is_full_relay(),
        )?;

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

        let good_peers_count = self.connected_peers();
        if good_peers_count > T::MAX_OUTGOING_PEERS {
            // We allow utreexo, extra and manual peers to bypass our connection limits
            let is_utreexo_peer = matches!(version.kind, ConnectionKind::OutboundFullRelay(services) if services.has(service_flags::UTREEXO.into()));
            let is_manual_peer = version.kind.is_manual();
            let is_extra = version.kind == ConnectionKind::Extra;

            if !(is_utreexo_peer || is_manual_peer || is_extra) {
                debug!(
                    "Already have {} peers, treating peer {peer} as addr-fetch before disconnecting to avoid blowing up our max of {}",
                    good_peers_count,
                    T::MAX_OUTGOING_PEERS
                );

                // If a peer exceeds our max, just turn them into an addr-fetch
                // so we can receive their AddrV2 message and then disconnect.
                self.peers.entry(peer).and_modify(|p| {
                    p.kind = ConnectionKind::AddrFetch;
                });

                version.kind = ConnectionKind::AddrFetch;
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

            debug!(
                "Feeler peer {peer} completed handshake; marking address tried and disconnecting"
            );
            self.send_to_peer(peer, NodeRequest::Shutdown)?;
            return Ok(());
        }

        if version.kind == ConnectionKind::AddrFetch {
            debug!("Addr-fetch peer {peer} completed handshake; requesting addresses");
            self.send_to_peer(peer, NodeRequest::GetAddresses)?;
            self.inflight
                .insert(InflightRequests::GetAddresses, (peer, Instant::now()));

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
            if let Some(peer_data) = self.common.peers.get_mut(&peer) {
                peer_data.services = version.services;
                peer_data.user_agent.clone_from(&version.user_agent);
                peer_data.height = version.blocks;
                peer_data.transport_protocol = version.transport_protocol;
            }

            let locator = self.chain.get_block_locator()?;
            debug!("Extra peer {peer} completed handshake; requesting headers");
            self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;

            self.inflight
                .insert(InflightRequests::Headers, (peer, Instant::now()));

            return Ok(());
        }

        if !version.kind.is_block_relay_only() {
            debug!(
                "Requesting addresses from peer {peer} kind={:?}",
                version.kind
            );
            self.send_to_peer(peer, NodeRequest::GetAddresses)?;
            self.inflight
                .insert(InflightRequests::GetAddresses, (peer, Instant::now()));
        }

        info!(
            "New peer id={} version={} blocks={} services={}",
            version.id, version.user_agent, version.blocks, version.services
        );

        if let Some(peer_data) = self.common.peers.get_mut(&peer) {
            peer_data.services = version.services;
            peer_data.address.set_services(version.services);
            peer_data.user_agent.clone_from(&version.user_agent);
            peer_data.height = version.blocks;
            peer_data.time_offset = version.time_offset;
            peer_data.transport_protocol = version.transport_protocol;

            // If this peer doesn't have basic services, we disconnect it
            if let ConnectionKind::OutboundFullRelay(needs)
            | ConnectionKind::BlockRelayOnly(needs) = version.kind
            {
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

            if version.kind.is_block_relay_only() {
                let locator = self.chain.get_block_locator()?;
                self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;

                self.inflight
                    .insert(InflightRequests::Headers, (peer, Instant::now()));
            }
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
        // Ignore queued replies from removed peers, even if retrying their requests failed.
        // They must not complete requests now assigned to another peer.
        // Still handle `Disconnected` so each node context can finish its cleanup.
        if !self.peers.contains_key(&peer) && !matches!(msg, PeerMessages::Disconnected(_)) {
            return Ok(None);
        }

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
            if p.kind.is_long_lived() && p.state == PeerStatus::Ready {
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

        // User requests have no automatic retry. Dropping their response senders
        // gives callers an error instead of leaving them waiting for this peer.
        self.inflight_user_requests
            .retain(|_, (requested_peer, _, _)| *requested_peer != peer);

        let inflight = self
            .inflight
            .clone()
            .into_iter()
            .filter(|(_k, v)| v.0 == peer)
            .collect::<Vec<_>>();

        let mut retry_error = None;
        for req in inflight {
            self.inflight.remove(&req.0);

            if let Err(e) = self.redo_inflight_request(&req.0) {
                // CRITICAL: never drop the request, so we retry it later
                self.inflight.insert(req.0, req.1);
                // Retry the remaining requests before returning an error.
                retry_error = Some(e);
            }
        }

        #[cfg(feature = "metrics")]
        self.update_peer_metrics();

        retry_error.map_or(Ok(()), Err)
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
        if peer.kind.is_manual() {
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

    /// Punishes a peer for sending us a full block we never requested.
    pub(crate) fn punish_unrequested_block(
        &mut self,
        peer: PeerId,
        block_hash: BlockHash,
    ) -> Result<(), WireError> {
        warn!("Received block {block_hash} from peer {peer}, but we didn't ask for it");
        self.increase_banscore(peer, 5)
    }

    /// Disconnects a peer and bans it for `T::BAN_TIME`.
    pub(crate) fn disconnect_and_ban(&mut self, peer: PeerId) -> Result<(), WireError> {
        if let Some(peer) = self.peers.get_mut(&peer) {
            // Manual connections are exempt from being punished
            if peer.kind.is_manual() {
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
            .filter(|peer| peer.kind.is_regular() && !peer.has_any_service(protected_services))
            .sample(&mut rng, n);

        for peer in peers {
            let _ = peer.channel.send(NodeRequest::Shutdown);
        }
    }

    /// Checks whether some of our inflight requests have timed out.
    ///
    /// Disconnects unresponsive peers without banning them and retries their requests.
    /// Manual peers keep their connection when requests time out.
    /// Address timeouts only disconnect feeler peers.
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

        let timed_out: Vec<_> = self
            .inflight
            .iter()
            .filter_map(|(req, (_, time))| timed_out_fn(req, time))
            .collect();

        let mut retry_error = None;
        for req in timed_out {
            let Some(&(peer, time)) = self.inflight.get(&req) else {
                continue;
            };

            // This request was timed out when collected, but an earlier iteration
            // disconnecting the same peer may have retried it. So re-check timeouts.
            if timed_out_fn(&req, &time).is_none() {
                continue;
            }

            // Remove it now to avoid retrying it both during disconnection and below.
            self.inflight.remove(&req);

            debug!("Request timed out: {req:?}");
            if let Some(peer_data) = self.peers.get(&peer) {
                // Do not disconnect peers on missing address replies, except for
                // short-lived connections.
                if matches!(req, InflightRequests::GetAddresses) && !peer_data.kind.is_short_lived()
                {
                    continue;
                }

                // Disconnect peers on timeout unless they are manual (and completed handshake).
                if !peer_data.kind.is_manual() || matches!(req, InflightRequests::Connect(_)) {
                    let idx = peer_data.address.id;
                    let _ = self.send_to_peer(peer, NodeRequest::Shutdown);

                    // Remove the peer before retrying, without waiting for its task to stop.
                    // It cannot be selected again, and the common handler discards late replies.
                    if let Err(e) = self.handle_disconnection(peer, idx) {
                        retry_error = Some(e);
                    }
                }
            }

            if let Err(e) = self.redo_inflight_request(&req) {
                // CRITICAL: never drop the request, so we retry it later, on the next tick.
                // Keep the old timestamp so it stays expired.
                self.inflight.insert(req, (peer, time));
                // Retry the remaining requests before returning an error.
                retry_error = Some(e);
            }
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

        // Addr-fetch peers are short-lived: after we receive their addresses,
        // we disconnect them.
        let Some(peer_data) = self.peers.get(&peer) else {
            return Ok(());
        };

        if matches!(peer_data.kind, ConnectionKind::AddrFetch) {
            debug!("Addr-fetch peer {peer} returned addresses; disconnecting");
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
                let Some(filters) = self.block_filters.as_ref() else {
                    return Ok(());
                };

                let peer = self.send_to_fast_peer(
                    NodeRequest::GetFilter((self.last_filter, filters.get_height()? + 1)),
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
        let utreexo_peers: Vec<LocalAddress> = self
            .peers
            .values()
            .filter(|peer| {
                peer.state == PeerStatus::Ready && peer.services.has(service_flags::UTREEXO.into())
            })
            .map(|peer| peer.address.clone())
            .collect();
        if utreexo_peers.is_empty() {
            warn!("No connected Utreexo peers to save to disk");
            return Ok(());
        }
        info!("Saving utreexo peers to disk...");
        AddressMan::dump_utreexo_peers(&self.datadir, &utreexo_peers).map_err(WireError::Io)
    }

    // === METRICS AND HELPERS ===

    /// Records response latency for the peer currently assigned to the request.
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

        // A reply may be read just before a retry to the same peer, so `sent_at`
        // will be overridden with the retry instant, which makes `read_at < sent_at`.
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
