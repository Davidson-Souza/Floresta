// SPDX-License-Identifier: MIT OR Apache-2.0

//! After a node catches-up with the network, we can start listening for new blocks, handing any
//! request our user might make and keep our peers alive. This mode requires way less bandwidth and
//! CPU to run, being bound by the number of blocks found in a given period.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;

use bitcoin::BlockHash;
use bitcoin::bip158::BlockFilter;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2Message;
use floresta_chain::ThreadSafeChain;
use floresta_chain::proof_util;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::pruned_utreexo::partial_chain::PartialChainState;
use floresta_common::service_flags;
use floresta_common::try_and_log;
use floresta_common::try_and_warn;
use rand::rng;
use rand::seq::IteratorRandom;
use rustreexo::stump::Stump;
use tokio::time;
use tokio::time::MissedTickBehavior;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::node::ConnectionKind;
use crate::node::InflightRequests;
use crate::node::MAX_ADDRV2_ADDRESSES;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::PeerStatus;
use crate::node::UtreexoNode;
use crate::node::WitnessMode;
use crate::node::chain_selector_ctx::ChainSelector;
use crate::node::periodic_job;
use crate::node::swift_sync_ctx::SwiftSync;
use crate::node::sync_ctx::SyncNode;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;

/// How long do we keep the last invs to compute peer scores
const MAX_INV_RETENTION_TIME: Duration = Duration::from_secs(60 * 60); // One hour

/// Don't hold more than this many invs in `last_invs`
const MAX_LAST_INVS: usize = 144; // Around one day worth of blocks

/// To prevent disk filling attacks, we forbid reorgs that are over two days deep.
const MAX_REORG_DEPTH: u32 = 288; // The expected amount of blocks mined in two days

#[derive(Debug, Clone)]
pub struct RunningNode {
    pub(crate) last_address_rearrange: Instant,
    /// To find peers with a good connectivity, keep track of what peers sent us an inv message
    /// for a block, in the first 5 seconds after we get the first inv message. If we ever decide
    /// to disconnect a peer, we should disconnect the ones that didn't send us an inv message
    /// in a timely manner, but keep the ones that notified us of a new blocks the fastest.
    /// We also keep the moment we received the first inv message
    pub(crate) last_invs: HashMap<BlockHash, (Instant, Vec<PeerId>)>,
    pub(crate) inflight_filters: BTreeMap<u32, BlockFilter>,
}

impl NodeContext for RunningNode {
    const REQUEST_TIMEOUT: u64 = 2 * 60;
}

impl Default for RunningNode {
    fn default() -> Self {
        Self {
            last_address_rearrange: Instant::now(),
            last_invs: HashMap::default(),
            inflight_filters: BTreeMap::new(),
        }
    }
}

impl<Chain> UtreexoNode<Chain, RunningNode>
where
    Chain: ThreadSafeChain + Clone,
    WireError: From<Chain::Error>,
    Chain::Error: From<proof_util::UtreexoLeafError>,
{
    fn send_addresses(&mut self) -> Result<(), WireError> {
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
            .take(MAX_ADDRV2_ADDRESSES)
            .collect();

        self.send_to_random_peer_with_predicate(
            NodeRequest::SendAddresses(addresses),
            ServiceFlags::NONE,
            |peer| peer.state == PeerStatus::Ready && peer.kind.is_full_relay(),
        )?;

        Ok(())
    }

    /// Every time we restart the node, we'll be a few blocks behind the tip. This function
    /// will start a sync node that will request, download and validate all blocks from the
    /// last validation index to the tip. This function will block until the sync node is
    /// finished.
    ///
    /// On the first startup, if we use either assumeutreexo or pow fraud proofs, this function
    /// will only download the blocks that are after the one that got assumed. So, for PoW fraud
    /// proofs, this means the last 100 blocks, and for assumeutreexo, this means however many
    /// blocks from the hard-coded value in the config file.
    pub async fn catch_up(self) -> Result<Self, WireError> {
        let swift_sync = UtreexoNode {
            common: self.common,
            context: SwiftSync::default(),
        };

        // Propagate chainstate errors instead of falling back to proof sync with uncertain state
        let swift_sync = swift_sync.run(|_| {}).await?;

        if *swift_sync.kill_signal.read().await {
            // Return to the caller shutdown path instead of starting proof sync
            return Ok(Self {
                common: swift_sync.common,
                context: self.context,
            });
        }

        let swift_sync_failed = swift_sync.was_aborted();

        // Finish IBD with regular utreexo sync
        let mut sync = UtreexoNode {
            common: swift_sync.common,
            context: SyncNode::default(),
        };

        // If SwiftSync couldn't complete, we need to validate all blocks from scratch
        if swift_sync_failed {
            // Clear the inflight requests and in-memory blocks to start from genesis
            sync.inflight.clear();
            sync.blocks.clear();
            assert_eq!(sync.unprocessed_blocks(), 0);
        }

        let sync = sync.run(|_| {}).await;

        // Once we are synced, peer diversity is the priority, as we must be able to discover
        // newly mined blocks. However, the peer list that we have built during IBD is biased
        // towards low-latency peers (often geographically close to our node).
        //
        // Keep half the running-mode peer limit to leave room for fresh peers.
        let peers_to_keep = RunningNode::MAX_OUTGOING_PEERS / 2;
        let peers_to_disconnect = sync.connected_peers().saturating_sub(peers_to_keep);
        let protected_services = &[service_flags::UTREEXO.into()];
        sync.disconnect_random_peers(peers_to_disconnect, protected_services);

        Ok(Self {
            common: sync.common,
            context: self.context,
        })
    }

    /// This function is called periodically to check if we have:
    /// - 10 connections
    /// - At least one utreexo peer
    /// - At least one compact filters peer
    ///
    /// If we are missing the special peers but have 10 connections, we should disconnect one
    /// random peer and try to connect to a utreexo and a compact filters peer.
    fn check_connections(&mut self) -> Result<(), WireError> {
        // retry the added peers connections
        self.maybe_open_connection_with_added_peers()?;

        // if we have 10 connections, but not a single utreexo or CBF one, disconnect one random
        // peer and create a utreexo and CBS connection
        if !self.has_utreexo_peers() {
            if self.connected_peers() >= RunningNode::MAX_OUTGOING_PEERS {
                self.peers
                    .values()
                    .filter(|peer| peer.kind.is_regular())
                    .choose(&mut rng())
                    .and_then(|p| p.channel.send(NodeRequest::Shutdown).ok());
            }

            self.maybe_open_connection(service_flags::UTREEXO.into())?;
        }

        if self.block_filters.is_some() && !self.has_compact_filters_peer() {
            if self.connected_peers() >= RunningNode::MAX_OUTGOING_PEERS {
                self.peers
                    .values()
                    .filter(|peer| peer.kind.is_regular())
                    .choose(&mut rng())
                    .and_then(|p| p.channel.send(NodeRequest::Shutdown).ok());
            }

            self.maybe_open_connection(ServiceFlags::COMPACT_FILTERS)?;
        }

        self.maybe_open_connection(ServiceFlags::NONE)?;

        Ok(())
    }

    /// If either PoW fraud proofs or assumeutreexo are enabled, we will "skip" IBD for all
    /// historical blocks. This allow us to start the node faster, making it usable in a few
    /// minutes. If you still want to validate all blocks, you can enable the backfill option.
    ///
    /// This function will spawn a background task that will download and validate all blocks
    /// that got assumed. After completion, the task will shutdown and the node will continue
    /// running normally. If we ever assume an invalid chain, the node will [halt and catch fire].
    ///
    /// [halt and catch fire]: https://en.wikipedia.org/wiki/Halt_and_Catch_Fire_(computing)
    pub fn backfill(&self, done_flag: std::sync::mpsc::Sender<()>) -> Result<bool, WireError> {
        // try finding the last state of the sync node
        let state = std::fs::read(self.config.datadir.join(".sync_node_state"));
        // try to recover from the disk state, if it exists. Otherwise, start from genesis
        let (chain, end) = match state {
            Ok(state) => {
                // if this file is empty, this means we've finished backfilling
                if state.is_empty() {
                    return Ok(false);
                }

                let acc = Stump::deserialize(&state[..(state.len() - 8)]).unwrap();
                let tip = u32::from_le_bytes(
                    state[(state.len() - 8)..(state.len() - 4)]
                        .try_into()
                        .unwrap(),
                );

                let end = u32::from_le_bytes(state[(state.len() - 4)..].try_into().unwrap());
                info!("Recovering backfill node from state tip={tip}, end={tip}");

                (
                    self.chain
                        .get_partial_chain(tip, end, acc)
                        .expect("Failed to get partial chain"),
                    end,
                )
            }
            Err(_) => {
                // if the file doesn't exist or got corrupted, start from genesis
                let end = self
                    .chain
                    .get_validation_index()
                    .expect("can get the validation index");
                (
                    self.chain
                        .get_partial_chain(0, end, Stump::default())
                        .unwrap(),
                    end,
                )
            }
        };

        let backfill = UtreexoNode::<PartialChainState, SyncNode>::new(
            self.config.clone(),
            chain,
            self.mempool.clone(),
            None,
            self.kill_signal.clone(),
            self.address_man.clone(),
        )
        .unwrap();

        let datadir = self.config.datadir.clone();
        let outer_chain = self.chain.clone();

        let fut = UtreexoNode::<PartialChainState, SyncNode>::run(
            backfill,
            move |chain: &PartialChainState| {
                if chain.has_invalid_blocks() {
                    panic!("We assumed a chain with invalid blocks, something went really wrong");
                }

                done_flag.send(()).unwrap();

                // we haven't finished the backfill yet, save the current state for the next run
                if chain.is_in_ibd() {
                    let acc = chain.get_acc();
                    let tip = chain.get_height().unwrap();
                    let mut ser_acc = Vec::new();
                    acc.serialize(&mut ser_acc).unwrap();
                    ser_acc.extend_from_slice(&tip.to_le_bytes());
                    ser_acc.extend_from_slice(&end.to_le_bytes());
                    std::fs::write(datadir.join(".sync_node_state"), ser_acc)
                        .expect("Failed to write sync node state");
                    return;
                }

                // empty the file if we're done
                std::fs::write(datadir.join(".sync_node_state"), Vec::new())
                    .expect("Failed to write sync node state");

                for block in chain.list_valid_blocks() {
                    outer_chain
                        .mark_block_as_valid(block.block_hash())
                        .expect("Failed to mark block as valid");
                }

                info!("Backfilling task shutting down...");
            },
        );

        tokio::task::spawn(fut);
        Ok(true)
    }

    /// Runs the P2P node through initial synchronization and normal operation.
    ///
    /// The node initializes its peers, selects the best header chain, optionally starts historical
    /// block backfill and catches the validated chain state up to the network tip. It then enters
    /// the [`RunningNode`] event loop, where it processes peer and user messages and performs
    /// periodic maintenance until shutdown is requested.
    ///
    /// On an orderly shutdown, the node closes its peer connections, persists its state and uses
    /// `stop_signal` to notify the owner that shutdown is complete. If startup terminates early,
    /// the sender is dropped and the corresponding receiver is closed.
    pub async fn run(mut self, stop_signal: tokio::sync::oneshot::Sender<()>) {
        try_and_warn!(self.init_peers());

        // Use this node state to Initial Block download
        let mut ibd = UtreexoNode {
            common: self.common,
            context: ChainSelector::default(),
        };

        try_and_log!(UtreexoNode::<Chain, ChainSelector>::run(&mut ibd).await);

        self = Self {
            common: ibd.common,
            context: self.context,
        };

        if *self.kill_signal.read().await {
            self.shutdown();
            try_and_log!(stop_signal.send(()));
            return;
        }

        // download blocks from the network before our validation index, probably because we've
        // assumed it somehow.
        let (sender, recv) = std::sync::mpsc::channel();
        let is_backfilling = match self.config.backfill {
            true => {
                info!("Starting backfill task...");
                self.backfill(sender)
                    .expect("Failed to spawn backfill thread")
            }
            false => false,
        };

        // Catch up with the network, downloading blocks from our last validation index to the tip
        let kill_signal = self.kill_signal.clone();
        info!("Catching up with the network...");
        self = match self.catch_up().await {
            Ok(node) => node,
            Err(e) => {
                error!("An error happened while trying to catch-up with the network: {e:?}");
                *kill_signal.write().await = true;
                return;
            }
        };

        if *self.kill_signal.read().await {
            self.shutdown();
            match stop_signal.send(()) {
                Ok(_) => {}
                Err(e) => error!("Stop signal receiver already dropped: {e:?}"),
            }
            return;
        }

        self.last_block_request = self.chain.get_validation_index().unwrap_or(0);
        if let Some(ref cfilters) = self.block_filters {
            self.last_filter = self
                .chain
                .get_block_hash(cfilters.get_height().unwrap_or(1))
                .unwrap();
        }

        self.last_block_request = self.chain.get_validation_index().unwrap_or(0);
        if let Some(ref cfilters) = self.block_filters {
            self.last_filter = self
                .chain
                .get_block_hash(cfilters.get_height().unwrap_or(1))
                .unwrap();
        }

        let mut ticker = time::interval(RunningNode::MAINTENANCE_TICK);
        // If we fall behind, don't "catch up" by running maintenance repeatedly
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        info!("starting running node...");
        loop {
            tokio::select! {
                biased;

                // Maintenance runs only on tick but has priority
                _ = ticker.tick() => match self.maintenance_tick().await {
                    LoopControl::Continue => {},
                    LoopControl::Break => break,
                },

                // Handle messages as soon as we find any, otherwise sleep until maintenance
                msg = self.node_rx.recv() => {
                    let Some(notification) = msg else {
                        break;
                    };
                    try_and_log!(self.handle_notification(notification).await);

                    // Drain all queued messages
                    while let Ok(notification) = self.node_rx.try_recv() {
                        try_and_log!(self.handle_notification(notification).await);
                    }
                }
            }
        }

        // ignore the error here because if the backfill task already
        // finished, this channel will be closed
        if is_backfilling {
            let _ = recv.recv();
        }

        self.shutdown();
        let _ = stop_signal.send(());
    }

    /// Performs the periodic maintenance tasks, including checking for the cancel signal, peer
    /// connections, and inflight request timeouts.
    ///
    /// Returns `LoopControl::Break` if we need to stop the node due to the kill signal being set.
    async fn maintenance_tick(&mut self) -> LoopControl {
        if *self.kill_signal.read().await {
            return LoopControl::Break;
        }

        // Jobs that don't need a connected peer
        try_and_log!(self.process_pending_blocks());

        // Save our peers db
        periodic_job!(
            self.last_peer_db_dump => self.save_peers(),
            RunningNode::PEER_DB_DUMP_INTERVAL,
        );

        // Rework our address database
        periodic_job!(
            self.context.last_address_rearrange => self.address_man.rearrange_buckets(),
            RunningNode::ADDRESS_REARRANGE_INTERVAL,
            no_log,
        );

        // Perhaps we need more connections
        periodic_job!(
            self.last_connection => self.check_connections(),
            RunningNode::TRY_NEW_CONNECTION,
            no_log,
        );

        // Check if some of our peers have timed out a request
        try_and_log!(self.check_for_timeout());

        // Open new feeler connection periodically
        periodic_job!(
            self.last_feeler => self.open_feeler_connection(),
            RunningNode::FEELER_INTERVAL,
            no_log,
        );

        // The jobs below need a connected peer to work
        if self.peer_ids.is_empty() {
            return LoopControl::Continue;
        }

        // Ask our peers for new addresses
        periodic_job!(
            self.last_get_address_request => self.ask_for_addresses(),
            RunningNode::ASK_FOR_PEERS_INTERVAL,
        );

        // Send our addresses to our peers
        periodic_job!(
            self.last_send_addresses => self.send_addresses(),
            RunningNode::SEND_ADDRESSES_INTERVAL,
        );

        // Check whether we are in a stale tip
        periodic_job!(
            self.last_tip_update => self.check_for_stale_tip(),
            RunningNode::ASSUME_STALE,
        );

        // tries to download filters from the network
        try_and_log!(self.download_filters());
        try_and_log!(self.ask_missed_block());

        // requests that need a utreexo peer
        if !self.has_utreexo_peers() {
            return LoopControl::Continue;
        }

        try_and_log!(self.ask_for_missed_proofs());

        LoopControl::Continue
    }

    fn download_filters(&mut self) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::GetFilters) {
            return Ok(());
        }

        if !self.has_compact_filters_peer() {
            return Ok(());
        }

        let Some(ref filters) = self.block_filters else {
            return Ok(());
        };

        let mut height = filters.get_height()?;
        let best_height = self.chain.get_height()?;

        if height == 0 {
            let user_height = self.config.filter_start_height.unwrap_or(1);

            height = if user_height < 0 {
                best_height.saturating_sub(user_height.unsigned_abs())
            } else {
                user_height as u32
            };

            height = height.saturating_sub(1);
            filters.save_height(height)?;
        }

        if height >= best_height {
            return Ok(());
        }

        info!("Downloading filters from height {}", filters.get_height()?);
        let stop = if height + 500 > best_height {
            best_height
        } else {
            height + 500
        };

        let stop_hash = self.chain.get_block_hash(stop)?;
        self.last_filter = stop_hash;

        let peer = self.send_to_fast_peer(
            NodeRequest::GetFilter((stop_hash, height + 1)),
            ServiceFlags::COMPACT_FILTERS,
        )?;

        self.inflight
            .insert(InflightRequests::GetFilters, (peer, Instant::now()));

        Ok(())
    }

    fn ask_missed_block(&mut self) -> Result<(), WireError> {
        let tip = self.chain.get_height().unwrap();
        let next = self.chain.get_validation_index().unwrap();
        if tip == next {
            return Ok(());
        }

        let mut blocks = Vec::new();
        for i in (next + 1)..=tip {
            let hash = self.chain.get_block_hash(i)?;
            // already requested
            if self.inflight.contains_key(&InflightRequests::Blocks(hash)) {
                continue;
            }

            if blocks.len() >= RunningNode::MAX_INFLIGHT_REQUESTS {
                break;
            }

            // already downloaded
            if self.blocks.contains_key(&hash) {
                continue;
            }

            blocks.push(hash);
        }

        if blocks.is_empty() {
            return Ok(());
        }

        self.request_blocks(blocks)?;
        Ok(())
    }

    /// If we think our tip is stale, we may disconnect one peer and try to get a new one.
    /// In this process, if the extra peer gives us a new block, we should drop one of our
    /// already connected peers to keep the number of connections stable. This function
    /// decides which peer to drop based on whether they've timely inv-ed us about the last
    /// 6 blocks.
    fn get_peer_score(&self, peer: PeerId) -> u32 {
        let mut score = 0;
        for block in self.context.last_invs.keys() {
            if self.context.last_invs[block].1.contains(&peer) {
                score += 1;
            }
        }

        score
    }

    /// This function checks how many time has passed since our last tip update, if it's
    /// been more than 15 minutes, try to update it.
    fn check_for_stale_tip(&mut self) -> Result<(), WireError> {
        warn!("Potential stale tip detected, trying extra peers");

        // this catches an edge-case where all our utreexo peers are gone, and the GetData
        // times-out. That yields an error, but doesn't ask the block again. Our last_block_request
        // will be pointing to a block that will never arrive, so we basically deadlock.
        self.last_block_request = self.chain.get_validation_index().unwrap();
        // update this or we'll get this warning every second after 15 minutes without a block,
        // until we get a new block.
        self.last_tip_update = Instant::now();
        self.create_connection(ConnectionKind::Extra)?;
        self.send_to_random_peer(
            NodeRequest::GetHeaders(self.chain.get_block_locator().unwrap()),
            ServiceFlags::NONE,
        )?;
        Ok(())
    }

    fn handle_new_block(&mut self, block: BlockHash, peer: u32) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::Headers) {
            return Ok(());
        }

        if self.chain.get_block_header(&block).is_ok() {
            return Ok(());
        }

        let locator = self.chain.get_block_locator().unwrap();
        self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;

        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }

    async fn handle_notification(
        &mut self,
        notification: NodeNotification,
    ) -> Result<(), WireError> {
        match notification {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder).await;
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }

            NodeNotification::FromPeer(peer, message, time) => {
                self.register_message_time(&message, peer, time);

                let Some(unhandled) = self.handle_peer_msg_common(message, peer)? else {
                    return Ok(());
                };

                match unhandled {
                    PeerMessages::UtreexoProof(uproof) => {
                        self.attach_proof(uproof, peer)?;
                        self.process_pending_blocks()?;
                    }

                    PeerMessages::NewBlock(block) => {
                        debug!("We got an inv with block {block} requesting it");
                        // This shouldn't happen under normal operation, but prevents worse-case
                        // scenarios.
                        if self.context.last_invs.len() >= MAX_LAST_INVS {
                            self.context.last_invs.drain();
                        }

                        match self.context.last_invs.get_mut(&block) {
                            Some((when, peers)) => {
                                if peers.contains(&peer) {
                                    return Ok(());
                                }

                                // if it's been less than 5 seconds since we got the first inv message
                                // for this block, we should mark as this peer sent us in a timely manner
                                if when.elapsed() < Duration::from_secs(5) {
                                    peers.push(peer);
                                }
                            }

                            None => {
                                self.context.last_invs.retain(|_, (when, _)| {
                                    when.elapsed() <= MAX_INV_RETENTION_TIME
                                });
                                self.context
                                    .last_invs
                                    .insert(block, (Instant::now(), vec![peer]));
                            }
                        }

                        // Don't request a block twice
                        let already_requested =
                            self.inflight.contains_key(&InflightRequests::Blocks(block))
                                || self.blocks.contains_key(&block);
                        if already_requested {
                            return Ok(());
                        }

                        if self.chain.get_block_header(&block).is_ok() {
                            return Ok(());
                        }

                        self.handle_new_block(block, peer)?;
                    }

                    PeerMessages::Block(block) => {
                        if let Some(block) = self.handle_received_block(block, peer)? {
                            self.request_block_proof(block, peer)?;
                        }
                    }

                    PeerMessages::Headers(headers) => {
                        debug!(
                            "Got headers from peer {peer} with {} headers",
                            headers.len()
                        );
                        self.inflight.remove(&InflightRequests::Headers);

                        let peer_info = self.peers.get(&peer).cloned().expect("Peer not found");
                        let is_extra = matches!(peer_info.kind, ConnectionKind::Extra);

                        if is_extra {
                            // if this is an extra peer, and the headers message is empty, disconnect it
                            if headers.is_empty() {
                                self.increase_banscore(peer, 5)?;
                                return Ok(());
                            }

                            // this peer got us a new block, we should disconnect one of our regular peers
                            // and keep this one as block-relay-only.
                            let peer_to_disconnect = self
                                .peers
                                .iter()
                                // Don't disconnect manual connections
                                .filter(|(_, info)| info.kind.is_regular())
                                .min_by_key(|(k, _)| self.get_peer_score(**k))
                                .map(|(peer, _)| *peer);

                            // disconnect the peer with the lowest score
                            if let Some(peer_to_disconnect) = peer_to_disconnect {
                                debug!(
                                    "Disconnecting peer {peer_to_disconnect} after extra peer promotion"
                                );
                                self.send_to_peer(peer_to_disconnect, NodeRequest::Shutdown)?;
                            }

                            // update the peer info
                            debug!(
                                "Extra peer {peer} returned useful headers; keeping it as block-relay-only"
                            );
                            self.peers.entry(peer).and_modify(|info| {
                                info.kind = ConnectionKind::BlockRelayOnly(peer_info.services);
                            });

                            if !self.peer_ids.contains(&peer) {
                                self.peer_ids.push(peer);
                            }
                        }

                        for header in headers.iter() {
                            let block_hash = header.block_hash();

                            // Don't request a block twice
                            let already_requested = self
                                .inflight
                                .contains_key(&InflightRequests::Blocks(block_hash))
                                || self.blocks.contains_key(&block_hash);

                            if already_requested {
                                continue;
                            }

                            if self.chain.get_block_header(&block_hash).is_ok() {
                                continue;
                            }

                            let Some(previous_height) =
                                self.chain.get_block_height(&header.prev_blockhash)?
                            else {
                                // Orphan chain
                                self.disconnect_and_ban(peer)?;
                                return Ok(());
                            };

                            let (best_height, _) = self.chain.get_best_block()?;
                            if best_height.saturating_sub(MAX_REORG_DEPTH) > previous_height {
                                // peer has a super deep reorg, this could be a disk fill attack
                                warn!(
                                    "Peer {peer} is trying to reorg a very deep block, might be a disk fill attack. Banning it"
                                );
                                self.disconnect_and_ban(peer)?;
                                return Ok(());
                            }

                            self.chain.accept_header(*header)?;

                            // `accept_header` may change the best chain.
                            let (_, best_hash) = self.chain.get_best_block()?;

                            // Don't request blocks outside the best chain.
                            if block_hash != best_hash {
                                continue;
                            }

                            self.send_to_peer(
                                peer,
                                NodeRequest::GetBlock(vec![header.block_hash()], WitnessMode::Full),
                            )?;

                            self.inflight.insert(
                                InflightRequests::Blocks(header.block_hash()),
                                (peer, Instant::now()),
                            );
                        }
                    }

                    PeerMessages::Ready(version) => {
                        debug!(
                            "handshake with peer={peer} succeeded feeler={:?}",
                            version.kind
                        );
                        self.handle_peer_ready(peer, version)?;
                    }

                    PeerMessages::Disconnected(idx) => {
                        self.handle_disconnection(peer, idx)?;
                    }

                    PeerMessages::BlockFilter((hash, filter)) => {
                        debug!("Got a block filter for block {hash} from peer {peer}");

                        if let Some(filters) = self.common.block_filters.as_ref() {
                            let mut current_height = filters.get_height()?;
                            let Some(this_height) = self.chain.get_block_height(&hash)? else {
                                warn!("Filter for block {hash} received, but we don't have it");
                                return Ok(());
                            };

                            if current_height + 1 != this_height {
                                self.context.inflight_filters.insert(this_height, filter);
                                return Ok(());
                            }

                            filters.push_filter(filter, current_height + 1)?;
                            current_height += 1;

                            while let Some(filter) =
                                self.context.inflight_filters.remove(&(current_height))
                            {
                                filters.push_filter(filter, current_height)?;
                                current_height += 1;
                            }

                            filters.save_height(current_height)?;
                            let current_hash = self.chain.get_block_hash(current_height)?;
                            if self.last_filter == current_hash
                                && self.context.inflight_filters.is_empty()
                            {
                                self.inflight.remove(&InflightRequests::GetFilters);
                                self.download_filters()?;
                            }
                        }
                    }

                    _ => unreachable!(
                        "Error: `handle_peer_msg_common` should have handled remaining PeerMessages"
                    ),
                }
            }

            msg @ NodeNotification::FromWorker { .. } => {
                error!("Received a notification from the worker thread {msg:?}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::time::Instant;

    use bitcoin::Network;
    use bitcoin::p2p::ServiceFlags;
    use floresta_chain::AssumeValidArg;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::FlatChainStoreConfig;
    use floresta_chain::pruned_utreexo::UpdatableChainstate;
    use floresta_common::Ema;
    use floresta_mempool::Mempool;
    use tokio::sync::Mutex;
    use tokio::sync::RwLock;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::unbounded_channel;

    use crate::TransportProtocol;
    use crate::UtreexoNodeConfig;
    use crate::address_man::AddressMan;
    use crate::address_man::AddressState;
    use crate::address_man::LocalAddress;
    use crate::node::ConnectionKind;
    use crate::node::InflightRequests;
    use crate::node::LocalPeerView;
    use crate::node::NodeNotification;
    use crate::node::NodeRequest;
    use crate::node::PeerStatus;
    use crate::node::UtreexoNode;
    use crate::node::WitnessMode;
    use crate::node::running_ctx::RunningNode;
    use crate::p2p_wire::peer::PeerMessages;
    use crate::p2p_wire::peer::Version;
    use crate::p2p_wire::tests::signet_headers;

    type TestChain = Arc<ChainState<FlatChainStore>>;
    type TestNode = UtreexoNode<TestChain, RunningNode>;

    fn peer_services() -> ServiceFlags {
        ServiceFlags::NETWORK | ServiceFlags::WITNESS | ServiceFlags::NETWORK_LIMITED
    }

    fn test_node() -> TestNode {
        let datadir = format!("./tmp-db/{}.running_node", rand::random::<u32>());
        fs::create_dir_all(&datadir).expect("test datadir should be created");

        let chainstore = FlatChainStore::new(FlatChainStoreConfig::new(&datadir))
            .expect("chain store should be created");
        let chain = Arc::new(
            ChainState::open(chainstore, Network::Signet, AssumeValidArg::Disabled)
                .expect("chain state should open"),
        );

        let mut headers = signet_headers();
        headers.remove(0);
        chain
            .accept_header(headers[0])
            .expect("test chain should accept first signet header");

        let config = UtreexoNodeConfig {
            network: Network::Signet,
            datadir: datadir.into(),
            ..Default::default()
        };

        UtreexoNode::<TestChain, RunningNode>::new(
            config,
            chain,
            Arc::new(Mutex::new(Mempool::new(1000))),
            None,
            Arc::new(RwLock::new(false)),
            AddressMan::new(None, &[]),
        )
        .expect("test node should be created")
    }

    fn local_address(peer_id: u32) -> LocalAddress {
        LocalAddress::new(
            format!("192.0.2.{}:38333", peer_id + 1)
                .parse()
                .expect("test address should parse"),
            0,
            AddressState::NeverTried,
            peer_services(),
            peer_id as usize,
        )
    }

    fn insert_peer(
        node: &mut TestNode,
        peer_id: u32,
        kind: ConnectionKind,
    ) -> UnboundedReceiver<NodeRequest> {
        let (sender, receiver) = unbounded_channel();

        node.peers.insert(
            peer_id,
            LocalPeerView {
                message_times: Ema::with_half_life_50(),
                state: PeerStatus::Ready,
                channel: sender,
                services: peer_services(),
                user_agent: "/test:0.0.0/".to_string(),
                address: local_address(peer_id),
                _last_message: Instant::now(),
                kind,
                height: 1,
                time_offset: 0,
                banscore: 0,
                transport_protocol: TransportProtocol::V2,
            },
        );

        receiver
    }

    /// Creates a test node with a single ready peer of the given kind.
    fn test_node_with_peer(
        peer_id: u32,
        kind: ConnectionKind,
    ) -> (TestNode, UnboundedReceiver<NodeRequest>) {
        let mut node = test_node();
        let receiver = insert_peer(&mut node, peer_id, kind);

        (node, receiver)
    }

    fn peer_version(peer_id: u32, kind: ConnectionKind) -> Version {
        Version {
            user_agent: "/test:0.0.0/".to_string(),
            protocol_version: 70016,
            blocks: 1,
            id: peer_id,
            address_id: peer_id as usize,
            services: peer_services(),
            time_offset: 0,
            kind,
            transport_protocol: TransportProtocol::V2,
        }
    }

    #[test]
    fn feeler_peer_disconnects_after_handshake_without_requesting_addresses() {
        let peer = 0;
        let (mut node, mut receiver) = test_node_with_peer(peer, ConnectionKind::Feeler);

        node.handle_peer_ready(peer, peer_version(peer, ConnectionKind::Feeler))
            .expect("feeler peer should become ready");

        assert_eq!(
            receiver
                .try_recv()
                .expect("feeler peer should be disconnected"),
            NodeRequest::Shutdown
        );
        assert!(
            receiver.try_recv().is_err(),
            "feeler peer should not receive address requests"
        );
    }

    #[tokio::test]
    async fn addr_fetch_peer_requests_addresses_then_disconnects() {
        let peer = 0;
        let (mut node, mut receiver) = test_node_with_peer(peer, ConnectionKind::AddrFetch);

        node.handle_peer_ready(peer, peer_version(peer, ConnectionKind::AddrFetch))
            .expect("addr-fetch peer should become ready");

        assert_eq!(
            receiver
                .try_recv()
                .expect("addr-fetch peer should request addresses"),
            NodeRequest::GetAddresses
        );
        assert!(
            receiver.try_recv().is_err(),
            "addr-fetch peer should wait for addresses before disconnecting"
        );

        node.handle_notification(NodeNotification::FromPeer(
            peer,
            PeerMessages::Addr(Vec::new()),
            Instant::now(),
        ))
        .await
        .expect("addr-fetch address response should be handled");

        assert_eq!(
            receiver
                .try_recv()
                .expect("addr-fetch peer should disconnect after addresses"),
            NodeRequest::Shutdown
        );
    }

    #[tokio::test]
    async fn extra_peer_with_useful_headers_replaces_regular_peer_as_block_relay_only() {
        let mut node = test_node();

        let regular_peer = 0;
        let extra_peer = 1;
        let mut regular_peer_receiver = insert_peer(
            &mut node,
            regular_peer,
            ConnectionKind::OutboundFullRelay(ServiceFlags::NONE),
        );
        let mut extra_peer_receiver = insert_peer(&mut node, extra_peer, ConnectionKind::Extra);

        node.peer_ids.push(regular_peer);
        node.inflight
            .insert(InflightRequests::Headers, (extra_peer, Instant::now()));

        let headers = signet_headers();
        let useful_header = headers[2];
        node.handle_notification(NodeNotification::FromPeer(
            extra_peer,
            PeerMessages::Headers(vec![useful_header]),
            Instant::now(),
        ))
        .await
        .expect("extra peer headers should be handled");

        assert_eq!(
            regular_peer_receiver
                .try_recv()
                .expect("regular peer should be disconnected"),
            NodeRequest::Shutdown
        );
        assert!(matches!(
            node.peers
                .get(&extra_peer)
                .expect("extra peer should still exist")
                .kind,
            ConnectionKind::BlockRelayOnly(services) if services == peer_services()
        ));
        assert!(node.peer_ids.contains(&extra_peer));
        assert_eq!(
            extra_peer_receiver
                .try_recv()
                .expect("block-relay-only peer should be asked for the new block"),
            NodeRequest::GetBlock(vec![useful_header.block_hash()], WitnessMode::Full)
        );
    }
}
