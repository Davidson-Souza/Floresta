// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node that downloads and validates the blockchain.

use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::p2p::ServiceFlags;
use floresta_chain::ThreadSafeChain;
use floresta_chain::proof_util;
use floresta_chain::pruned_utreexo::IBDState;
use floresta_common::service_flags;
use floresta_common::try_and_log;
use tokio::time;
use tokio::time::MissedTickBehavior;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::node::ConnectionKind;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::UtreexoNode;
use crate::node::periodic_job;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;

/// [`SyncNode`] is a node that downloads and validates the blockchain.
/// This node implements:
///     - `NodeContext`
///     - `UtreexoNode<SyncNode, Chain>`
///
/// see [node_context](crates/floresta-wire/src/p2p_wire/node_context.rs) and [node.rs](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
#[derive(Clone, Debug, Default)]
pub struct SyncNode {}

impl NodeContext for SyncNode {
    /// Get the required [services](ServiceFlags) for the [`SyncNode`].
    ///
    /// The [`SyncNode`] is active during IBD, and therefore requires that peers support:
    ///   * `NETWORK`: the peer is capable of serving the entire blockchain.
    ///   * `WITNESS`: the peer is capable of serving blocks and transactions with witness data.
    ///   * `UTREEXO_ARCHIVE`: the peer is capable of serving inclusion proofs for the entire blockchain.
    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::NETWORK | ServiceFlags::WITNESS | service_flags::UTREEXO_ARCHIVE.into()
    }

    const REQUEST_TIMEOUT: u64 = 60 * 2; // 2 minutes
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default

    // A more conservative value than the default of 1 second, since we'll have many peer messages
    const MAINTENANCE_TICK: Duration = Duration::from_secs(5);
}

/// Node methods for a [`UtreexoNode`] where its Context is a [`SyncNode`].
/// See [node](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
impl<Chain> UtreexoNode<Chain, SyncNode>
where
    Chain: ThreadSafeChain,
    WireError: From<Chain::Error>,
    Chain::Error: From<proof_util::UtreexoLeafError>,
{
    /// Computes the next blocks to request, and sends a GETDATA request
    ///
    /// We send block requests in batches of four, and we can always have two
    /// such batches inflight. Therefore, we can have at most eight inflight
    /// blocks.
    ///
    /// This function sends exactly one GETDATA, therefore ask for four blocks.
    /// It will compute the next blocks we need, given our tip, validation index,
    /// inflight requests and cached blocks. We then select a random peer and send
    /// the request.
    ///
    /// TODO: Be smarter when selecting peers to send, like taking in consideration
    /// already inflight blocks and latency.
    fn get_blocks_to_download(&mut self) {
        let max_inflight_blocks = SyncNode::BLOCKS_PER_GETDATA * SyncNode::MAX_CONCURRENT_GETDATA;
        let inflight_blocks = self
            .inflight
            .keys()
            .filter(|inflight| matches!(inflight, InflightRequests::Blocks(_)))
            .count();

        let unprocessed_blocks = inflight_blocks + self.blocks.len();

        // if we do a request, this will be the new inflight blocks count
        let next_unprocessed_count = unprocessed_blocks + SyncNode::BLOCKS_PER_GETDATA;

        // if this request would make our inflight queue too long, postpone it
        if next_unprocessed_count > max_inflight_blocks {
            return;
        }

        let mut blocks = Vec::with_capacity(SyncNode::BLOCKS_PER_GETDATA);
        for _ in 0..SyncNode::BLOCKS_PER_GETDATA {
            let next_block = self.last_block_request + 1;
            let validation_index = self.chain.get_validation_index().unwrap();
            if next_block <= validation_index {
                self.last_block_request = validation_index;
            }

            let next_block = self.chain.get_block_hash(next_block);
            match next_block {
                Ok(next_block) => {
                    blocks.push(next_block);
                    self.last_block_request += 1;
                }

                Err(_) => {
                    // this is likely because we've reached the end of the chain
                    // and we've got a `BlockNotPresent` error.
                    break;
                }
            }
        }

        try_and_log!(self.request_blocks(blocks));
    }

    fn ask_for_missed_blocks(&mut self) -> Result<(), WireError> {
        let next_request = self.chain.get_validation_index()? + 1;
        let last_block_requested = self.last_block_request;

        // we accumulate the hashes of all blocks in [next_request, last_block_requested] here
        // and pass it to request_blocks, which will filter inflight and pending blocks out.
        let mut range_blocks = Vec::new();

        for request_height in next_request..=last_block_requested {
            let block_hash = self.chain.get_block_hash(request_height)?;
            range_blocks.push(block_hash);
        }

        self.request_blocks(range_blocks)
    }

    /// This function will periodically check our connections, to ensure that:
    ///   - we have enough utreexo peers to download proofs from (at least 2)
    ///   - we have enough peers to download blocks from (at most `MAX_OUTGOING_PEERS`)
    ///   - we disconnect from peers that are too slow, which could otherwise slow down our IBD
    fn check_connections(&mut self) -> Result<(), WireError> {
        // With `--connect`, the user pins us to a specific peer set; don't search
        // for extra utreexo or network peers. `maybe_open_connection` still
        // reconnects any fixed peer that has dropped.
        if self.has_fixed_peers() {
            return self.maybe_open_connection(service_flags::UTREEXO.into());
        }

        let utreexo_peers = self
            .peer_by_service
            .get(&service_flags::UTREEXO.into())
            .map_or(0, |peers| peers.len());

        // We are looking for at least two utreexo peers to diversify our requests
        let needs_utreexo = utreexo_peers < 2;

        if self.connected_peers() >= SyncNode::MAX_OUTGOING_PEERS {
            // If we need more utreexo peers, always disconnect the slowest peer
            let always_disconnect = needs_utreexo;
            // Don't disconnect from utreexo peers in any case
            let protected_services = &[service_flags::UTREEXO.into()];

            self.maybe_disconnect_slowest_peer(protected_services, always_disconnect)?;
        }

        if needs_utreexo {
            info!("Not enough utreexo peers (we have {utreexo_peers}), opening a new connection");
            self.maybe_open_connection(service_flags::UTREEXO.into())?;
        }

        self.maybe_open_connection(ServiceFlags::NETWORK)
    }

    /// Starts the sync node by updating the last block requested and starting the main loop.
    /// This loop to the following tasks, in order:
    ///     - Receives messages from our peers through the node_tx channel.
    ///     - Handles the message received.
    ///     - Checks if the kill signal is set, if so, breaks the loop.
    ///     - Checks if the chain is in IBD and disables it if it's not (e.g. if the chain is synced).
    ///     - Checks if our tip is obsolete and requests a new one, creating a new connection.
    ///     - Handles timeouts for inflight requests.
    ///     - If were low on inflights, requests new blocks to validate.
    pub async fn run(mut self, done_cb: impl FnOnce(&Chain)) -> Self {
        info!("Starting sync node...");
        self.last_block_request = self.chain.get_validation_index().unwrap();

        let mut ticker = time::interval(SyncNode::MAINTENANCE_TICK);
        // If we fall behind, don't "catch up" by running maintenance repeatedly
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

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
                    let Some(msg) = msg else {
                        break;
                    };
                    try_and_log!(self.handle_message(msg).await);

                    // Drain all queued messages
                    while let Ok(msg) = self.node_rx.try_recv() {
                        try_and_log!(self.handle_message(msg).await);
                    }
                    if *self.kill_signal.read().await {
                        break;
                    }
                }
            }
        }

        done_cb(&self.chain);
        self
    }

    /// Performs the periodic maintenance tasks, including checking for the cancel signal, peer
    /// connections, and inflight request timeouts.
    ///
    /// Returns `LoopControl::Break` if we need to break the main `SyncNode` loop, either because
    /// the kill signal was set or because the chain is synced.
    async fn maintenance_tick(&mut self) -> LoopControl {
        if *self.kill_signal.read().await {
            return LoopControl::Break;
        }

        let validation_index = self
            .chain
            .get_validation_index()
            .expect("validation index block should present");

        let (best_block, tip_hash) = self
            .chain
            .get_best_block()
            .expect("best block should present");

        if validation_index == best_block {
            // Skip the tip-age check at genesis: the genesis timestamp (~2009) would
            // always look stale and trap regtest nodes in IBD indefinitely.
            let stale = best_block > 0 && {
                let tip_time = self
                    .chain
                    .get_block_header(&tip_hash)
                    .expect("tip header is always present")
                    .time;
                let now: u32 = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);
                now.saturating_sub(tip_time) > self.config.max_tip_age_secs
            };

            if stale {
                warn!(
                    "validation caught up to a stale tip; chain store may be stuck, staying in IBD"
                );
            } else {
                info!("IBD is finished, switching to normal operation mode");
                self.chain.update_ibd(IBDState::Done);
                return LoopControl::Break;
            }
        }

        periodic_job!(
            self.last_connection => self.check_connections(),
            SyncNode::TRY_NEW_CONNECTION,
            no_log,
        );

        // Open new feeler connection periodically
        periodic_job!(
            self.last_feeler => self.open_feeler_connection(),
            SyncNode::FEELER_INTERVAL,
            no_log,
        );

        try_and_log!(self.check_for_timeout());

        let assume_stale = Instant::now()
            .duration_since(self.common.last_tip_update)
            .as_secs()
            > SyncNode::ASSUME_STALE;

        if assume_stale {
            try_and_log!(self.create_connection(ConnectionKind::Extra));
            self.last_tip_update = Instant::now();
            return LoopControl::Continue;
        }

        try_and_log!(self.process_pending_blocks());
        if !self.has_utreexo_peers() {
            return LoopControl::Continue;
        }

        // Ask for missed blocks or proofs if they are no longer inflight or pending
        try_and_log!(self.ask_for_missed_blocks());
        try_and_log!(self.ask_for_missed_proofs());

        self.get_blocks_to_download();
        LoopControl::Continue
    }

    /// Process a message from a peer and handle it accordingly between the variants of [`PeerMessages`].
    async fn handle_message(&mut self, msg: NodeNotification) -> Result<(), WireError> {
        match msg {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder).await;
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }

            NodeNotification::FromPeer(peer, notification, time) => {
                self.register_message_time(&notification, peer, time);

                let Some(unhandled) = self.handle_peer_msg_common(notification, peer)? else {
                    return Ok(());
                };

                match unhandled {
                    PeerMessages::Block(block) => {
                        let hash = block.block_hash();
                        if self.blocks.contains_key(&hash) {
                            debug!(
                                "Received block {hash} from peer {peer}, but we already have it"
                            );
                            return Ok(());
                        }

                        self.request_block_proof(block, peer)?;

                        self.process_pending_blocks()?;
                        self.get_blocks_to_download();
                    }

                    PeerMessages::Ready(version) => {
                        try_and_log!(self.handle_peer_ready(peer, version));
                    }

                    PeerMessages::Disconnected(idx) => {
                        try_and_log!(self.handle_disconnection(peer, idx));
                    }

                    PeerMessages::UtreexoProof(uproof) => {
                        self.attach_proof(uproof, peer)?;
                        self.process_pending_blocks()?;
                        self.get_blocks_to_download();
                    }

                    _ => {}
                }
            }
        }

        Ok(())
    }
}
