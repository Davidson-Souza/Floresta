// SPDX-License-Identifier: MIT

//! A module that connects with multiple peers and finds the best chain.
//!
//! # The theory
//!
//! In Bitcoin, the history of transactions processed by the network is defined by a sequence of
//! blocks, chainned by their cryptographic hash. A block commits the hash for the block right
//! before it. Therefore, if we pick any given block, there's exactly one history leading to the
//! very first block, that commits to no one. However, if you go in the other way, starting at the
//! first block and going up, there may not be only one history. Multiple blocks may commit to the
//! same parent. We need a way to pick just one such chain, among all others.
//!
//! To do that, we use the most work rule, sometimes called "Nakamoto Consensus" after Bitcoin's
//! creator, Satoshi Nakamoto. Every block has to solve a probabilistic challenge of finding a
//! combination of data that hashes to a value smaller than a network-agreed value. Because hash
//! functions are pseudorandom, one must make certain amount of hashes (on average) before finding a
//! valid one. If we define the amount of hashes needed to find a block as this block's "work",
//! by adding-up the work in each of a chain's blocks, we arrive with the `chainwork`. The Nakamoto
//! consensus consists in taking the chain with most work as the best one.
//!
//! This works because anyone in the network will compute the same amount of work and pick the same
//! one, regardless of where and when. Because work is a intrinsic and deterministic property of a
//! block, everyone comparing the same chain, be on earth, on mars; in 2020 or 2100, they will
//! choose the exact same chain, always.
//!
//! The most critial part of syncing-up a Bitcoin node is making sure you know about the most-work
//! chain. If someone can eclypse you, they can make you start following a chain that only you and
//! the attacker care about. If you get paid in this chain, you can't pay someone else outside this
//! chain, because they will be following other chains. Luckly, we only need one honest peer, to
//! find the best-work chain and avoid any attacker to fools us into accepting payments in a "fake
//! Bitcoin"
//!
//! # Implementation
//!
//! In Floresta, we try to pick a good balance between data downloaded and security. We could
//! simply download all chains from all peers and pick the most work one. But each header is
//! 80 bytes-long, with ~800k blocks, that's arround 60 MBs. If we have 10 peers, that's 600MBs
//! (excluding overhead by the p2p messages). Moreover, it's very uncommon to actually have peers
//! in different chains. So we can optmistically download all headers from one random peer, and
//! then check with the others if they aggree. If they have another chain for us, we download that
//! chain, and pick whichever has more work.
//!
//! Most likely we'll only download one chain and all peers will agree with it. Then we can start
//! downloading the actual blocks and validating them.
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use tokio::time::timeout;
use tokio::sync::RwLock;
use bitcoin::block::Header;
use bitcoin::BlockHash;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use log::info;

use super::error::WireError;
use super::peer::PeerMessages;
use crate::address_man::AddressState;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;

#[derive(Debug, Default, Clone)]
/// A p2p driver that attemps to connect with multiple peers, ask which chain are them following
/// and download and verify the headers, **not** the actual blocks. This is the first part of a
/// loger IBD pipeline.
/// The actual blocks should be downloaded by a SyncPeer.
pub struct ChainSelector {
    /// The state we are in
    state: ChainSelectorState,
    /// To save in bandwi****, we download headers from only one peer, and then look for forks
    /// afterwards. This is the peer we are using during this phase
    sync_peer: PeerId,
    /// Peers that already sent us a message we are waiting for
    done_peers: HashSet<PeerId>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ChainSelectorState {
    #[default]
    /// We are opening connection with some peers
    CreatingConnections,
    /// We are downloading headers from only one peer, assuming this peer is honest
    DownloadingHeaders,
    /// We've downloaded all headers, and now we are checking with our peers if they
    /// have an alternative tip with more PoW. Very unlikely, but we shouldn't trust
    /// only one peer...
    LookingForForks,
    /// We've downloaded all headers
    Done,
}

impl NodeContext for ChainSelector {
    const REQUEST_TIMEOUT: u64 = 10; // Ban peers stalling our IBD
    const TRY_NEW_CONNECTION: u64 = 10; // Try creating connections more aggressively
}

impl<Chain> UtreexoNode<ChainSelector, Chain>
where
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    async fn handle_headers(
        &mut self,
        peer: PeerId,
        headers: Vec<Header>,
    ) -> Result<(), WireError> {
        if headers.is_empty() {
            self.empty_headers_message(peer).await?;
            return Ok(());
        }

        info!(
            "Downloading headers from peer={peer} at height={} hash={}",
            self.chain.get_best_block()?.0 + 1,
            headers[0].block_hash()
        );

        for header in headers.iter() {
            if let Err(e) = self.chain.accept_header(*header) {
                log::error!("Error while downloading headers from peer={peer} err={e}");

                self.send_to_peer(peer, NodeRequest::Shutdown).await?;

                let peer = self.peers.get(&peer).unwrap();
                self.0.address_man.update_set_state(
                    peer.address_id as usize,
                    AddressState::Banned(ChainSelector::BAN_TIME),
                );
            }
        }

        self.request_headers(headers.last().unwrap().block_hash())
            .await
    }

    /// If we get an empty `haders` message, our next action depends on which state are
    /// we in:
    ///   - If we are downloading headers for the first time, this means we've just
    ///     finished and should go to the next phase
    ///   - If we are checking with our peer if they have an alternative tip, this peer
    ///     has send all blocks they have. Once all peers have finished, we just pick the
    ///     most PoW chain among all chains that we got
    async fn empty_headers_message(&mut self, peer: PeerId) -> Result<(), WireError> {
        match self.1.state {
            ChainSelectorState::DownloadingHeaders => {
                self.poke_peers().await?;
                self.1.state = ChainSelectorState::LookingForForks;
            }
            ChainSelectorState::LookingForForks => {
                self.1.done_peers.insert(peer);
                for peer in self.0.peer_ids.iter() {
                    // at least one peer haven't finished
                    if !self.1.done_peers.contains(peer) {
                        break;
                    }
                }
                self.1.state = ChainSelectorState::Done;
            }
            _ => {}
        }

        Ok(())
    }

    /// Ask for headers, given a tip
    ///
    /// This function will send a `getheaders` request to our peers, assuming this
    /// peer is following a chain with `tip` inside it. We use this in case some of
    /// our peer is in a fork, so we can learn about all blocks in that fork and
    /// compare the candidate chains to pick the best one.
    async fn request_headers(&mut self, tip: BlockHash) -> Result<(), WireError> {
        let locator = self
            .chain
            .get_block_locator_for_tip(tip)
            .unwrap_or_default();
        self.send_to_peer(self.1.sync_peer, NodeRequest::GetHeaders(locator))
            .await?;

        let peer = self.1.sync_peer;
        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }

    /// Checks if some request has timed-out.
    ///
    /// If it does, we disconnect and ban this peer
    async fn check_for_timeout(&mut self) -> Result<(), WireError> {
        let mut failed = vec![];

        for (request, (peer, instant)) in self.inflight.clone() {
            if instant.elapsed().as_secs() > ChainSelector::REQUEST_TIMEOUT {
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                failed.push(request)
            }
        }

        for request in failed {
            self.inflight.remove(&request);
        }

        Ok(())
    }

    /// Sends a `getheaders` to all our peers
    ///
    /// After we download all blocks from one peer, we ask our peers if they
    /// agree with our sync peer on what is the best chain. If they are in a fork,
    /// we'll download that fork and compare with our own chain. We should always pick
    /// the most PoW one.
    async fn poke_peers(&self) -> Result<(), WireError> {
        let locator = self.chain.get_block_locator().unwrap();
        for peer in self.0.peer_ids.iter() {
            let get_headers = NodeRequest::GetHeaders(locator.clone());
            self.send_to_peer(*peer, get_headers).await?;
        }

        Ok(())
    }

    pub async fn run(&mut self, stop_signal: &Arc<RwLock<bool>>) -> Result<(), WireError> {
        self.create_connection(false).await;

        info!("Starting ibd, selecting the best chain");

        loop {
            while let Some(notification) =
                timeout(Duration::from_millis(10), self.node_rx.recv()).await.unwrap()
            {
                try_and_log!(self.handle_notification(Some(notification)).await);
            }

            periodic_job!(
                self.maybe_open_connection().await,
                self.last_connection,
                TRY_NEW_CONNECTION,
                ChainSelector
            );

            if self.1.state == ChainSelectorState::CreatingConnections {
                // If we have enough peers, try to download headers
                if self.peer_ids.len() >= 2 {
                    let new_sync_peer = rand::random::<usize>() % self.peer_ids.len();
                    self.1.sync_peer = *self.peer_ids.get(new_sync_peer).unwrap();

                    try_and_log!(self.request_headers(self.chain.get_best_block()?.1).await);

                    self.1.state = ChainSelectorState::DownloadingHeaders;
                }
            }

            // We downloaded all headers in the most-pow chain, and all our peers agree
            // this is the most-pow chain, we're done!
            if self.1.state == ChainSelectorState::Done {
                self.chain.mark_chain_as_valid()?;
                break;
            }

            try_and_log!(self.check_for_timeout().await);

            if *stop_signal.read().await {
                break;
            }
        }

        Ok(())
    }

    async fn handle_notification(
        &mut self,
        notification: Option<NodeNotification>,
    ) -> Result<(), WireError> {
        match notification {
            Some(NodeNotification::FromPeer(peer, message)) => match message {
                PeerMessages::Headers(headers) => {
                    self.inflight.remove(&InflightRequests::Headers);
                    return self.handle_headers(peer, headers).await;
                }

                PeerMessages::Ready(version) => {
                    self.handle_peer_ready(peer, &version).await?;
                }

                PeerMessages::Disconnected(idx) => {
                    if peer == self.1.sync_peer {
                        self.1.state = ChainSelectorState::CreatingConnections;
                    }
                    self.handle_disconnection(peer, idx)?;
                }

                PeerMessages::Addr(addresses) => {
                    let addresses: Vec<_> =
                        addresses.iter().cloned().map(|addr| addr.into()).collect();
                    self.address_man.push_addresses(&addresses);
                }

                _ => {}
            },

            None => {}
        }
        Ok(())
    }
}
