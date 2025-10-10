//! A node that downloads and validates the blockchain.

use std::time::Duration;
use std::time::Instant;

use bitcoin::p2p::message_blockdata::Inventory;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::proof_util;
use floresta_chain::ThreadSafeChain;
use floresta_common::service_flags;
use floresta_common::service_flags::UTREEXO;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tokio::time::timeout;
use tracing::debug;
use tracing::info;
use tracing::warn;

use super::error::WireError;
use super::node_interface::UserRequest;
use super::peer::PeerMessages;
use crate::node::ConnectionKind;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_interface::NodeResponse;
use crate::p2p_wire::macros::periodic_job;
use crate::p2p_wire::macros::try_and_debug;
use crate::p2p_wire::macros::try_and_error;
use crate::p2p_wire::macros::try_and_warn;

/// [`SyncNode`] is a node that downloads and validates the blockchain.
/// This node implements:
///     - `NodeContext`
///     - `UtreexoNode<SyncNode, Chain>`
///
/// see [node_context](crates/floresta-wire/src/p2p_wire/node_context.rs) and [node.rs](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
#[derive(Clone, Debug, Default)]
pub struct SyncNode {}

impl NodeContext for SyncNode {
    fn get_required_services(&self) -> bitcoin::p2p::ServiceFlags {
        ServiceFlags::WITNESS | service_flags::UTREEXO.into() | ServiceFlags::NETWORK
    }

    const TRY_NEW_CONNECTION: u64 = 30; // 30 seconds
    const REQUEST_TIMEOUT: u64 = 10 * 60; // 10 minutes
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default
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
        if self.has_utreexo_peers() == false {
            return;
        }

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

        try_and_error!("Can't request blocks", self.request_blocks(blocks));
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
    ///   - if some of peers are too slow, and potentially stalling our block download (TODO)
    fn check_connections(&mut self) -> Result<(), WireError> {
        let total_peers = self.peers.len();
        let utreexo_peers = self
            .peer_by_service
            .get(&UTREEXO.into())
            .map_or(0, |peers| peers.len());

        if utreexo_peers < 2 && total_peers >= SyncNode::MAX_OUTGOING_PEERS {
            // if we have more than the maximum number of outgoing peers, disconnect
            // some non-utreexo peers.
            //
            // FIXME: We should actually disconnect the slowest non-utreexo peer, to
            // make sure we can download blocks faster.
            let mut non_utreexo_peers = Vec::new();
            for (id, peer) in self.peers.iter() {
                if !peer.services.has(UTREEXO.into()) {
                    non_utreexo_peers.push(*id);
                }
            }

            let peer_to_disconnect = *non_utreexo_peers
                .choose(&mut thread_rng())
                .expect("infallible: the `if` clause implies we have some non-utreexo peers");

            info!("Disconnecting non-utreexo peer {peer_to_disconnect} to open up more space for utreexo peers");
            self.send_to_peer(peer_to_disconnect, NodeRequest::Shutdown)?;
        }

        if utreexo_peers < 2 {
            info!("Not enough utreexo peers (we have {utreexo_peers}), opening a new connection");
            self.maybe_open_connection(UTREEXO.into())?;
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

        loop {
            while let Ok(Some(msg)) = timeout(Duration::from_secs(1), self.node_rx.recv()).await {
                try_and_error!("Failure to handle notification from peer", self.handle_message(msg));
            }

            if *self.kill_signal.read().await {
                break;
            }

            let validation_index = self
                .chain
                .get_validation_index()
                .expect("validation index block should present");

            let best_block = self
                .chain
                .get_best_block()
                .expect("best block should present")
                .0;

            if validation_index == best_block {
                info!("IBD is finished, switching to normal operation mode");
                self.chain.toggle_ibd(false);
                break;
            }

            periodic_job!(
                self.check_connections(),
                self.last_connection,
                TRY_NEW_CONNECTION,
                SyncNode
            );

            // Open new feeler connection periodically
            periodic_job!(
                self.open_feeler_connection(),
                self.last_feeler,
                FEELER_INTERVAL,
                SyncNode
            );

            try_and_warn!("Failure checking for timeout", self.check_for_timeout());
            let assume_stale = Instant::now()
                .duration_since(self.common.last_tip_update)
                .as_secs()
                > SyncNode::ASSUME_STALE;

            if assume_stale {
                try_and_warn!("Potential stable tip detected, trying new peers", self.create_connection(ConnectionKind::Extra));
                self.last_tip_update = Instant::now();
                continue;
            }

            try_and_error!("An error occurred while processing a pending block", self.process_pending_blocks());
            if !self.has_utreexo_peers() {
                continue;
            }

            // Ask for missed blocks or proofs if they are no longer inflight or pending
            try_and_debug!("Error while asking for missed block", self.ask_for_missed_blocks());
            try_and_debug!("Error while asking for missed proofs", self.ask_for_missed_proofs());

            self.get_blocks_to_download();
        }

        done_cb(&self.chain);
        self
    }

    /// Process a message from a peer and handle it accordingly between the variants of [`PeerMessages`].
    fn handle_message(&mut self, msg: NodeNotification) -> Result<(), WireError> {
        match msg {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder);
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }

            NodeNotification::FromPeer(peer, notification) => {
                self.register_message_time(&notification, peer);

                match notification {
                    PeerMessages::Block(block) => {
                        if self.blocks.contains_key(&block.block_hash()) {
                            debug!(
                                "Received block {} from peer {}, but we already have it",
                                block.block_hash(),
                                peer
                            );

                            return Ok(());
                        }

                        self.request_block_proof(block, peer)?;

                        self.process_pending_blocks()?;
                        self.get_blocks_to_download();
                    }

                    PeerMessages::Ready(version) => {
                        try_and_warn!("Failure while initializing state for peer", self.handle_peer_ready(peer, &version));
                    }

                    PeerMessages::Disconnected(idx) => {
                        try_and_warn!("Failure to update a peer state after disconnection", self.handle_disconnection(peer, idx));
                    }

                    PeerMessages::Addr(addresses) => {
                        debug!("Got {} addresses from peer {}", addresses.len(), peer);
                        let addresses: Vec<_> =
                            addresses.into_iter().map(|addr| addr.into()).collect();

                        self.address_man.push_addresses(&addresses);
                    }

                    PeerMessages::NotFound(inv) => match inv {
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
                    },

                    PeerMessages::Transaction(tx) => {
                        debug!("saw a mempool transaction with txid={}", tx.compute_txid());
                        if let Some(request) = self
                            .inflight_user_requests
                            .remove(&UserRequest::MempoolTransaction(tx.compute_txid()))
                        {
                            request
                                .2
                                .send(NodeResponse::MempoolTransaction(Some(tx)))
                                .map_err(|_| WireError::ResponseSendError)?;
                        }
                    }

                    PeerMessages::UtreexoState(_) => {
                        warn!("Utreexo state received from peer {peer}, but we didn't ask",);
                        self.increase_banscore(peer, 5)?;
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
