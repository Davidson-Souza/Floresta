// SPDX-License-Identifier: MIT OR Apache-2.0

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
//! The most critical part of syncing-up a Bitcoin node is making sure you know about the most-work
//! chain. If someone can eclypse you, they can make you start following a chain that only you and
//! the attacker care about. If you get paid in this chain, you can't pay someone else outside this
//! chain, because they will be following other chains. Luckily, we only need one honest peer, to
//! find the best-work chain and avoid any attacker to fools us into accepting payments in a "fake
//! Bitcoin"
//!
//! # Implementation
//!
//! Floresta runs the disk-free PRESYNC phase independently against every connected peer. The
//! first peer to demonstrate the network's minimum chainwork supplies full header hashes at
//! 100,000-block boundaries. Requests to the other presync peers are then aborted.
//!
//! The checkpoint ranges are redownloaded concurrently through distinct peers chosen by the
//! normal latency-weighted selector. A complete range is inserted directly only after its ending
//! checkpoint hash is reproduced; completed ranges are committed in height order. Once every
//! range and the remaining headers have arrived, Floresta asks all peers whether they agree and
//! downloads competing forks before choosing the most-work chain.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::block::Header;
use bitcoin::network::Network;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::ChainBackend;
use floresta_chain::CompactLeafData;
use floresta_chain::proof_util;
use floresta_chain::pruned_utreexo::IBDState;
use floresta_chain::pruned_utreexo::consensus::Consensus;
use floresta_common::service_flags;
use floresta_common::try_and_log;
use rand::rng;
use rand::seq::IndexedRandom;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::proof::Proof;
use rustreexo::stump::Stump;
use tokio::time;
use tokio::time::MissedTickBehavior;
use tokio::time::timeout;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::address_man::AddressState;
use crate::block_proof::Bitmap;
use crate::node::InflightBlock;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node::periodic_job;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::headers_sync::HeaderCheckpoint;
use crate::p2p_wire::headers_sync::HeaderRange;
use crate::p2p_wire::headers_sync::HeadersRangeDownload;
use crate::p2p_wire::headers_sync::HeadersSyncState;
use crate::p2p_wire::headers_sync::PresyncPhase;
use crate::p2p_wire::headers_sync::RangeProgress;
use crate::p2p_wire::peer::PeerMessages;

#[derive(Debug, Default, Clone)]
/// A p2p driver that attempts to connect with multiple peers, ask which chain are them following
/// and download and verify the headers, **not** the actual blocks. This is the first part of a
/// logger IBD pipeline.
/// The actual blocks should be downloaded by a SyncPeer.
pub struct ChainSelector {
    /// The state we are in
    state: ChainSelectorState,

    /// Peers that already sent us a message we are waiting for
    done_peers: HashSet<PeerId>,

    /// Keep track each peer's tip
    tip_cache: HashMap<PeerId, BlockHash>,

    /// Independent PRESYNC state machines for peers still racing.
    presync_states: HashMap<PeerId, HeadersSyncState>,

    /// Outstanding `getheaders` requests, tracked independently by peer.
    headers_requests: HashMap<PeerId, Instant>,

    /// Peer pinned for ordinary header download after parallel range redownload completes.
    headers_sync_peer: Option<PeerId>,

    /// Checkpoint ranges not yet assigned to a peer.
    redownload_pending: VecDeque<HeaderRange>,

    /// One independently verified checkpoint range per busy peer.
    redownload_active: HashMap<PeerId, HeadersRangeDownload>,

    /// Completed ranges waiting for all lower ranges before direct insertion.
    redownload_completed: BTreeMap<u32, Vec<Header>>,

    /// First height not yet inserted through `push_headers`.
    redownload_next_height: Option<u32>,

    /// Final checkpoint that concurrent redownload must reproduce.
    redownload_target: Option<HeaderCheckpoint>,

    /// Peers being disconnected after an invalid or timed-out range response.
    unavailable_headers_peers: HashSet<PeerId>,

    /// Peers with one outstanding response from an aborted PRESYNC request.
    stale_presync_responses: HashSet<PeerId>,

    /// Whether every checkpoint range was verified and inserted.
    redownload_complete: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ChainSelectorState {
    #[default]
    /// We are opening connection with some peers
    CreatingConnections,
    /// We are racing PRESYNC peers, redownloading checkpoint ranges concurrently, or fetching the
    /// remaining headers from one fast peer.
    DownloadingHeaders,
    /// We've downloaded all headers, and now we are checking with our peers if they
    /// have an alternative tip with more PoW. Very unlikely, but we shouldn't trust
    /// only one peer...
    LookingForForks(Instant),
    /// We've downloaded all headers
    Done,
}

pub enum FindAccResult {
    Found(Vec<u8>),
    KeepLooking(Vec<(PeerId, Vec<u8>)>),
}

/// Helper enum to express the different possibilities under `find_who_is_lying`
pub enum PeerCheck {
    /// One peer is lying
    OneLying(PeerId),

    /// Both peers are lying
    BothLying,

    /// One peer is unresponsive
    UnresponsivePeer(PeerId),

    /// Both peers are unresponsive
    BothUnresponsivePeers,
}

/// Maximum age in seconds a chain tip may have before the node refuses to leave IBD.
///
/// Mirrors Bitcoin Core's [`DEFAULT_MAX_TIP_AGE`](https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainstatemanager_opts.h#L24).
pub(crate) const DEFAULT_MAX_TIP_AGE: u32 = 24 * 60 * 60;

impl NodeContext for ChainSelector {
    const REQUEST_TIMEOUT: u64 = 60; // Ban peers stalling our IBD

    // Since we don't have any peers when chain selection starts, we use a more aggressive batch
    // size to make sure we get to our `MAX_OUTGOING_CONNECTIONS` ASAP
    const NEW_CONNECTIONS_BATCH_SIZE: usize = 12;

    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::NETWORK
            | service_flags::UTREEXO.into()
            | service_flags::UTREEXO_ARCHIVE.into()
    }
}

impl<Chain> UtreexoNode<Chain, ChainSelector>
where
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
    Chain::Error: From<proof_util::UtreexoLeafError>,
{
    /// This function is called every time we get a `Headers` message from a peer.
    /// It will validate the headers and add them to our chain, if they are valid.
    /// If we get an empty headers message, we'll check what to do next, depending on
    /// our current state. We may poke our peers to see if they have an alternative tip,
    /// or we may just finish the IBD, if no one have an alternative tip.
    async fn handle_headers(
        &mut self,
        peer: PeerId,
        headers: Vec<Header>,
        received_at: Instant,
    ) -> Result<(), WireError> {
        // A peer may still owe one response to a PRESYNC request that was aborted when another
        // peer won. Drain it without consuming a newer range request assigned to the same peer.
        if self.context.stale_presync_responses.remove(&peer) {
            debug!("Ignoring aborted presync response from peer={peer}");
            return Ok(());
        }

        // Fork agreement is deliberately unreachable until all checkpoint ranges and remaining
        // headers have been downloaded.
        if matches!(self.context.state, ChainSelectorState::LookingForForks(_)) {
            if headers.is_empty() {
                return self.empty_headers_message(peer).await;
            }

            for header in &headers {
                if let Err(error) = self.chain.accept_header(*header) {
                    error!("Error while accepting fork header from peer={peer}: {error}");
                    self.disconnect_and_ban(peer)?;
                    return Ok(());
                }
            }

            let last = headers
                .last()
                .expect("non-empty headers response")
                .block_hash();
            self.context.tip_cache.insert(peer, last);
            self.last_tip_update = Instant::now();
            let locator = self
                .chain
                .get_block_locator_for_tip(last)
                .unwrap_or_default();
            self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;
            return Ok(());
        }

        if self.context.state != ChainSelectorState::DownloadingHeaders {
            debug!("Ignoring headers from peer={peer} before header sync starts");
            return Ok(());
        }

        let is_presync = self.context.presync_states.contains_key(&peer);
        let is_range_download = self.context.redownload_active.contains_key(&peer);
        let is_normal_download =
            self.context.redownload_complete && self.context.headers_sync_peer == Some(peer);
        if !is_presync && !is_range_download && !is_normal_download {
            debug!("Ignoring headers from peer={peer} without assigned header work");
            return Ok(());
        }

        let Some(sent_at) = self.context.headers_requests.remove(&peer) else {
            debug!("Ignoring unsolicited headers from peer={peer}");
            return Ok(());
        };
        let elapsed = received_at.saturating_duration_since(sent_at).as_secs_f64();
        if let Some(peer_data) = self.peers.get_mut(&peer) {
            peer_data.message_times.add(elapsed * 1_000.0);
        }

        if is_presync {
            if headers.is_empty() {
                // Regtest has no minimum-work gate. No returned headers means the actual chain is
                // already complete, so agreement may begin immediately.
                if self.network == Network::Regtest {
                    self.context.presync_states.clear();
                    self.context.headers_requests.clear();
                    self.context.redownload_complete = true;
                    self.context.headers_sync_peer = Some(peer);
                    return self.empty_headers_message(peer).await;
                }

                self.context.presync_states.remove(&peer);
                self.context.unavailable_headers_peers.insert(peer);
                let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
                if self.context.presync_states.is_empty() {
                    self.context.state = ChainSelectorState::CreatingConnections;
                }
                return Ok(());
            }

            let (result, phase_after) = {
                let state = self
                    .context
                    .presync_states
                    .get_mut(&peer)
                    .expect("presync assignment checked above");
                let result = state.process_presync(&headers);
                (result, state.phase().clone())
            };

            if !result.success {
                error!("Peer={peer} failed headers presync; disconnecting and banning");
                self.context.presync_states.remove(&peer);
                self.disconnect_and_ban(peer)?;
                return Ok(());
            }

            match phase_after {
                PresyncPhase::Presync => {
                    if let Some(hash) = self
                        .context
                        .presync_states
                        .get(&peer)
                        .and_then(HeadersSyncState::next_locator_hash)
                    {
                        self.send_tracked_headers_request(peer, vec![hash], None)?;
                    }
                }
                PresyncPhase::Redownload => {
                    let ranges = self
                        .context
                        .presync_states
                        .get(&peer)
                        .and_then(HeadersSyncState::redownload_ranges)
                        .expect("redownload phase has checkpoint ranges");
                    info!(
                        "Peer={peer} won headers presync with {} checkpoint ranges",
                        ranges.len()
                    );
                    self.begin_parallel_redownload(ranges)?;
                }
                PresyncPhase::Aborted => {
                    self.context.presync_states.remove(&peer);
                    self.context.unavailable_headers_peers.insert(peer);
                    let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
                }
            }
            return Ok(());
        }

        if is_range_download {
            if headers.is_empty() {
                self.fail_redownload_range(peer)?;
                return Ok(());
            }

            let (range, progress) = {
                let download = self
                    .context
                    .redownload_active
                    .get_mut(&peer)
                    .expect("range assignment checked above");
                (download.range(), download.process(&headers))
            };

            match progress {
                RangeProgress::InProgress => self.request_active_range(peer)?,
                RangeProgress::Complete(headers) => {
                    self.context.redownload_active.remove(&peer);
                    self.context
                        .redownload_completed
                        .insert(range.start.height + 1, headers);
                    self.commit_completed_redownload_ranges()?;
                    self.dispatch_redownload_ranges()?;
                }
                RangeProgress::Invalid => self.fail_redownload_range(peer)?,
            }
            return Ok(());
        }

        if headers.is_empty() {
            return self.empty_headers_message(peer).await;
        }

        // Minimum work and every checkpoint range are already verified. Continue from the
        // committed tip with the selected fast peer using normal header validation.
        for header in &headers {
            if let Err(error) = self.chain.accept_header(*header) {
                error!("Error while downloading headers from peer={peer}: {error}");
                self.context.unavailable_headers_peers.insert(peer);
                // Keep the selected slot occupied until the disconnect notification arrives.
                self.context.headers_requests.insert(peer, Instant::now());
                self.disconnect_and_ban(peer)?;
                return Ok(());
            }
        }
        let last = headers
            .last()
            .expect("non-empty headers response")
            .block_hash();
        self.context.tip_cache.insert(peer, last);
        self.last_tip_update = Instant::now();
        if !self.context.unavailable_headers_peers.contains(&peer) {
            self.request_headers_from_peer(peer, last)?;
        }
        Ok(())
    }

    /// Parses a serialized Utreexo accumulator into a [`Stump`].
    ///
    /// An empty slice yields [`Stump::default`]. Otherwise the payload must contain
    /// an 8-byte little-endian leaf count followed by `leaves.count_ones()` root
    /// hashes of 32 bytes each. All other lengths are rejected.
    fn parse_acc(acc: &[u8]) -> Result<Stump, WireError> {
        if acc.is_empty() {
            return Ok(Stump::default());
        }
        let Some((leaf_count, roots_bytes)) = acc.split_first_chunk::<8>() else {
            return Err(WireError::PeerMisbehaving);
        };
        let leaves = u64::from_le_bytes(*leaf_count);

        let expected_roots = leaves.count_ones() as usize;
        let expected_roots_len = expected_roots * 32;
        if roots_bytes.len() != expected_roots_len {
            return Err(WireError::PeerMisbehaving);
        }

        let mut roots = Vec::with_capacity(expected_roots);
        for chunk in roots_bytes.chunks_exact(32) {
            let mut root = [0u8; 32];
            root.copy_from_slice(chunk);
            roots.push(BitcoinNodeHash::from(root));
        }

        Ok(Stump { leaves, roots })
    }

    /// Sends a request to two peers and wait for their response
    ///
    /// This function will send a `GetUtreexoState` request to two peers and wait for their
    /// response. If both peers respond, it will return the accumulator from both peers.
    /// If only one peer responds, it will return the accumulator from that peer and `None`
    /// for the other. If no peer responds, it will return `None` for both.
    /// We use this during the cut-and-choose protocol, to find where they disagree.
    async fn grab_both_peers_version(
        &mut self,
        peer1: PeerId,
        peer2: PeerId,
        block_hash: BlockHash,
        block_height: u32,
    ) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), WireError> {
        self.send_to_peer(
            peer1,
            NodeRequest::GetUtreexoState((block_hash, block_height)),
        )?;

        self.send_to_peer(
            peer2,
            NodeRequest::GetUtreexoState((block_hash, block_height)),
        )?;

        let mut peer1_version = None;
        let mut peer2_version = None;
        for _ in 0..2 {
            if let Ok(Some(NodeNotification::FromPeer(
                peer,
                PeerMessages::UtreexoState(state),
                _,
            ))) = timeout(Duration::from_secs(60), self.node_rx.recv()).await
            {
                if peer == peer1 {
                    peer1_version = Some(state);
                } else if peer == peer2 {
                    peer2_version = Some(state);
                }
            }
        }

        Ok((peer1_version, peer2_version))
    }

    /// Find which peer is lying about what the accumulator state is at a given point
    ///
    /// This function will ask peers their accumulator for a given block, and check whether
    /// they agree or not. If they don't, we cut the search in half and keep looking for the
    /// fork point. Once we find the last agreed accumulator, we ask for the block and proof
    /// that comes after it, update the accumulator from that point, and find who is lying.
    ///
    /// If successful returns the [PeerCheck] enum, representing whether peers are:
    ///
    /// - Lying
    /// - Unresponsive
    async fn find_who_is_lying(
        &mut self,
        peer1: PeerId,
        peer2: PeerId,
    ) -> Result<PeerCheck, WireError> {
        let (mut height, mut hash) = self.chain.get_best_block()?;
        let mut prev_height = 0;
        // we first narrow down the possible fork point to a couple of blocks, looking
        // for all blocks in a linear search would be too slow
        loop {
            // ask both peers for the utreexo state
            let (peer1_acc, peer2_acc) = self
                .grab_both_peers_version(peer1, peer2, hash, height)
                .await?;

            // if a peer is unresponsive, we opt for an early return
            let (peer1_acc, peer2_acc) = match (peer1_acc, peer2_acc) {
                (Some(acc1), Some(acc2)) => (acc1, acc2),
                (None, Some(_)) => return Ok(PeerCheck::UnresponsivePeer(peer1)),
                (Some(_), None) => return Ok(PeerCheck::UnresponsivePeer(peer2)),
                (None, None) => return Ok(PeerCheck::BothUnresponsivePeers),
            };

            // if we have different states, we need to keep looking until we find the
            // fork point
            let interval = height.abs_diff(prev_height);
            prev_height = height;

            if interval < 5 {
                break;
            }

            if peer1_acc == peer2_acc {
                // if they're equal, then the disagreement is in a newer block
                height += interval / 2;
            } else {
                // if they're different, then the disagreement is in an older block
                height -= interval / 2;
            }

            hash = self.chain.get_block_hash(height).unwrap();
        }
        info!("Fork point is around height={height} hash={hash}");
        // at the end, this variable should hold the last block where they agreed
        let mut fork = 0;

        // Getting the acc for the block on which we landed on
        let (peer1_acc, peer2_acc) = self
            .grab_both_peers_version(peer1, peer2, hash, height)
            .await?;

        // Initializing the agree bool for the block on which we landed on
        let agree = peer1_acc == peer2_acc;

        if agree {
            height += 1;
        } else {
            height -= 1;
        }

        loop {
            // keep asking blocks until we find the fork point
            let (peer1_acc, peer2_acc) = self
                .grab_both_peers_version(peer1, peer2, hash, height)
                .await?;

            // as we go, we'll approach the fork from two possible sides: we came from the side
            // they disagree, and therefore the point of inflection is the first block they agree.
            // on the other hand, if are agreeing, and we find they disagreeing, the last block
            // they've agreed on is the previous one (not the current one)
            match agree {
                true => {
                    // they agreed in the last block, so the fork is in the next one
                    if peer1_acc != peer2_acc {
                        fork = height - 1;
                    }
                }

                false => {
                    // they disagreed in the last block and now agree, the last block is the fork
                    if peer1_acc == peer2_acc {
                        fork = height;
                    }
                }
            }

            if fork != 0 {
                break;
            }

            // if we still don't know where the fork is, we need to keep looking
            if agree {
                // if they agree on this current block, we need to look in the next one
                height += 1;
            } else {
                // if they disagree on this current block, we need to look in the previous one
                height -= 1;
            }
            hash = self.chain.get_block_hash(height)?;
        }
        hash = self.chain.get_block_hash(fork)?;

        let acc = self
            .grab_both_peers_version(peer1, peer2, hash, fork)
            .await?;

        let agreed = match acc {
            (Some(acc1), Some(_)) => Self::parse_acc(&acc1)?,
            (Some(acc1), None) => Self::parse_acc(&acc1)?,
            (None, Some(acc2)) => Self::parse_acc(&acc2)?,
            (None, None) => return Ok(PeerCheck::BothUnresponsivePeers),
        };

        hash = self.chain.get_block_hash(fork + 1)?;

        // now we know where the fork is, we need to check who is lying
        let (peer1_acc, peer2_acc) = self
            .grab_both_peers_version(peer1, peer2, hash, fork + 1)
            .await?;

        // if a peer is unresponsive, we opt for an early return
        let (peer1_acc, peer2_acc) = match (peer1_acc, peer2_acc) {
            (Some(acc1), Some(acc2)) => (acc1, acc2),
            (None, Some(_)) => return Ok(PeerCheck::UnresponsivePeer(peer1)),
            (Some(_), None) => return Ok(PeerCheck::UnresponsivePeer(peer2)),
            (None, None) => return Ok(PeerCheck::BothUnresponsivePeers),
        };

        let block_hash = self.chain.get_block_hash(fork + 1)?;

        let inflight_block = match self.get_block_and_proof(peer1, block_hash).await {
            Err(WireError::PeerTimeout) => return Ok(PeerCheck::UnresponsivePeer(peer1)),
            res => res?,
        };

        let (leaf_data, proof, _) = inflight_block
            .aux_data
            .expect("Block proof and leaf data should be present");

        let acc1 = self.update_acc(agreed, &inflight_block.block, proof, &leaf_data, fork + 1)?;

        let peer1_acc = Self::parse_acc(&peer1_acc)?;
        let peer2_acc = Self::parse_acc(&peer2_acc)?;

        if peer1_acc != acc1 && peer2_acc != acc1 {
            return Ok(PeerCheck::BothLying);
        }

        if peer1_acc != acc1 {
            return Ok(PeerCheck::OneLying(peer1));
        }

        Ok(PeerCheck::OneLying(peer2))
    }

    /// Requests a block and its proof from a peer
    ///
    /// If you need to see a peer's version of a given block, you can use this method
    /// to request a block from a specific peer.
    async fn get_block_and_proof(
        &mut self,
        peer: PeerId,
        block_hash: BlockHash,
    ) -> Result<InflightBlock, WireError> {
        self.send_to_peer(peer, NodeRequest::GetBlock(vec![block_hash]))?;

        let timeout = Instant::now() + Duration::from_secs(60);
        let mut block = None;
        loop {
            if Instant::now() > timeout {
                return Err(WireError::PeerTimeout);
            }

            let Some(NodeNotification::FromPeer(id, message, _)) = self.node_rx.recv().await else {
                // Keep waiting until peer message is read or timeout
                continue;
            };

            if id != peer {
                continue;
            }

            match message {
                // STEP 1: Receive the block and ask for the proof
                PeerMessages::Block(recv_block) => {
                    if recv_block.block_hash() != block_hash {
                        error!("peer {peer} sent us a block we didn't request");
                        self.disconnect_and_ban(peer)?;
                        return Err(WireError::PeerMisbehaving);
                    }

                    // Check if the block was maliciously mutated by our peer
                    let is_mutated = Consensus::check_merkle_root(&recv_block).is_none()
                        || !recv_block.check_witness_commitment();

                    if is_mutated {
                        error!(
                            "Peer {peer} sent us a mutated block {}",
                            recv_block.block_hash()
                        );
                        self.disconnect_and_ban(peer)?;
                        return Err(WireError::PeerMisbehaving);
                    }

                    block = Some(recv_block);
                    // ask for the proof. Sending two empty bitmaps means we want the full
                    // proof and all leaf data
                    self.send_to_peer(
                        peer,
                        NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
                    )?;
                }

                // STEP 2: Receive the proof and return the `InflightBlock`
                PeerMessages::UtreexoProof(uproof) => {
                    let Some(block) = block else {
                        error!("peer {peer} sent us a proof without sending the block first");
                        self.disconnect_and_ban(peer)?;
                        return Err(WireError::PeerMisbehaving);
                    };

                    let proof = Proof {
                        hashes: uproof.proof_hashes,
                        targets: uproof.targets,
                    };

                    return Ok(InflightBlock {
                        peer,
                        block,
                        aux_data: Some((uproof.leaf_data, proof, peer)),
                    });
                }
                _ => {}
            }
        }
    }

    /// Updates a Stump, with the data from a block and its proof
    fn update_acc(
        &self,
        acc: Stump,
        block: &Block,
        proof: Proof,
        leaf_data: &[CompactLeafData],
        height: u32,
    ) -> Result<Stump, WireError> {
        let (del_hashes, _) = proof_util::process_proof(leaf_data, &block.txdata, height, |h| {
            self.chain.get_block_hash(h)
        })?;

        Ok(self
            .chain
            .update_acc(acc, block, height, proof, del_hashes)?)
    }

    /// Finds the accumulator for one block
    ///
    /// This method will find what the accumulator looks like for a block with (height, hash).
    /// Check-out [this](https://blog.dlsouza.lol/2023/09/28/pow-fraud-proof.html) post
    /// to learn how the cut-and-choose protocol works
    async fn find_accumulator_for_block(
        &mut self,
        height: u32,
        hash: BlockHash,
    ) -> Result<Stump, WireError> {
        let mut candidate_accs = Vec::new();

        match self.find_accumulator_for_block_step(hash, height).await {
            Ok(FindAccResult::Found(acc)) => {
                // everyone agrees. Just parse the accumulator and finish-up
                let acc = Self::parse_acc(&acc)?;
                return Ok(acc);
            }
            Ok(FindAccResult::KeepLooking(mut accs)) => {
                accs.sort();
                accs.dedup();
                candidate_accs = accs;
            }
            _ => {}
        }

        let mut invalid_accs = HashSet::new();
        for peer in candidate_accs.windows(2) {
            if invalid_accs.contains(&peer[0].1) || invalid_accs.contains(&peer[1].1) {
                continue;
            }
            let (peer1, peer2) = (peer[0].0, peer[1].0);

            let liar_state = self.find_who_is_lying(peer1, peer2).await?;

            match liar_state {
                PeerCheck::OneLying(liar) => {
                    self.disconnect_and_ban(liar)?;
                    if liar == peer1 {
                        invalid_accs.insert(peer[0].1.clone());
                        continue;
                    }
                    invalid_accs.insert(peer[1].1.clone());
                }
                PeerCheck::UnresponsivePeer(dead_peer) => {
                    self.disconnect_and_ban(dead_peer)?;
                }
                PeerCheck::BothUnresponsivePeers => {
                    self.disconnect_and_ban(peer1)?;
                    self.disconnect_and_ban(peer2)?;
                }
                PeerCheck::BothLying => {
                    self.disconnect_and_ban(peer1)?;
                    self.disconnect_and_ban(peer2)?;

                    invalid_accs.insert(peer[0].1.clone());
                    invalid_accs.insert(peer[1].1.clone());
                }
            }
        }
        //filter out the invalid accs
        candidate_accs.retain(|acc| !invalid_accs.contains(&acc.1));
        //we should have only one candidate left
        assert_eq!(candidate_accs.len(), 1);

        Self::parse_acc(&candidate_accs.pop().unwrap().1)
    }

    /// If we get an empty `headers` message, our next action depends on which state are
    /// we in:
    ///   - If we are downloading headers for the first time, this means we've just
    ///     finished and should go to the next phase
    ///   - If we are checking with our peer if they have an alternative tip, this peer
    ///     has send all blocks they have. Once all peers have finished, we just pick the
    ///     most PoW chain among all chains that we got
    async fn empty_headers_message(&mut self, peer: PeerId) -> Result<(), WireError> {
        match self.context.state {
            ChainSelectorState::DownloadingHeaders => {
                let (best_height, tip_hash) = self.chain.get_best_block()?;

                // Skip the tip-age check at genesis: the genesis timestamp (~2009) is always
                // stale and would trap regtest or a cold-start node in IBD indefinitely.
                if best_height > 0 {
                    let tip_time = self.chain.get_block_header(&tip_hash)?.time;
                    let now: u32 = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs() as u32)
                        .unwrap_or(0);
                    let tip_age = now.saturating_sub(tip_time);

                    if tip_age > self.config.max_tip_age_secs {
                        warn!(
                            tip_age,
                            "header sync reached a stale tip; staying in IBD until the chain advances"
                        );
                        self.last_tip_update = Instant::now();
                        return Ok(());
                    }
                }

                self.context.done_peers.clear();
                self.context.headers_sync_peer = None;
                self.context.presync_states.clear();
                self.context.headers_requests.clear();
                self.context.redownload_pending.clear();
                self.context.redownload_active.clear();
                self.context.redownload_completed.clear();
                self.context.redownload_next_height = None;
                self.context.redownload_target = None;
                self.context.redownload_complete = true;

                info!("Finished downloading headers from peer={peer}, checking if our peers agree");
                self.poke_peers()?;
                self.context.state = ChainSelectorState::LookingForForks(Instant::now());
                self.context.done_peers.insert(peer);
            }
            ChainSelectorState::LookingForForks(_) => {
                self.context.done_peers.insert(peer);
                for peer in self
                    .common
                    .peer_ids
                    .iter()
                    .filter(|peer| !self.context.unavailable_headers_peers.contains(peer))
                {
                    // At least one usable peer has not finished.
                    if !self.context.done_peers.contains(peer) {
                        return Ok(());
                    }
                }

                if let Some(assume_utreexo) = self.common.config.assume_utreexo.as_ref() {
                    self.context.state = ChainSelectorState::Done;
                    // already assumed the chain
                    if self.chain.get_validation_index().unwrap() >= assume_utreexo.height {
                        return Ok(());
                    }
                    info!(
                        "Assuming chain with height={} tip={}",
                        assume_utreexo.height, assume_utreexo.block_hash
                    );
                    let acc = Stump {
                        leaves: assume_utreexo.leaves,
                        roots: assume_utreexo.roots.clone(),
                    };
                    self.chain
                        .mark_chain_as_assumed(acc, assume_utreexo.block_hash)?;
                    return Ok(());
                }

                let has_peers = self
                    .peer_by_service
                    .contains_key(&service_flags::UTREEXO_ARCHIVE.into());

                if self.config.pow_fraud_proofs && has_peers {
                    self.check_tips().await?;
                }

                self.context.state = ChainSelectorState::Done;
            }
            _ => {}
        }

        Ok(())
    }

    async fn is_our_chain_invalid(&mut self, other_tip: BlockHash) -> Result<(), WireError> {
        let fork = self.chain.get_fork_point(other_tip)?;
        let fork_height = self.chain.get_block_height(&fork)?.unwrap_or(0);

        let peers = self
            .peer_by_service
            .get(&service_flags::UTREEXO.into())
            .ok_or(WireError::NoPeersAvailable)?;

        let rand_peer = *peers
            .choose(&mut rng())
            .ok_or(WireError::NoPeersAvailable)?;

        let block = self.get_block_and_proof(rand_peer, fork).await?;
        let (leaf_data, proof, _) = block
            .aux_data
            .expect("Block proof and leaf data should be present");

        let (del_hashes, inputs) =
            proof_util::process_proof(&leaf_data, &block.block.txdata, fork_height, |h| {
                self.chain.get_block_hash(h)
            })?;

        let acc = self.find_accumulator_for_block(fork_height, fork).await?;
        let is_valid = self
            .chain
            .validate_block(&block.block, proof, inputs, del_hashes, acc);

        if is_valid.is_err() {
            let best_block = self.chain.get_best_block()?.1;
            self.ban_peers_on_tip(best_block)?;

            self.chain.switch_chain(other_tip)?;
            self.chain.invalidate_block(fork)?;
            return Ok(());
        }

        // our chain's block is valid, therefore there's no reason for anyone be in this fork
        self.ban_peers_on_tip(other_tip)?;
        Ok(())
    }

    fn ban_peers_on_tip(&mut self, tip: BlockHash) -> Result<(), WireError> {
        for peer in self.common.peers.clone() {
            if self.context.tip_cache.get(&peer.0).copied().eq(&Some(tip)) {
                self.address_man.update_set_state(
                    peer.1.address.id,
                    AddressState::Banned(ChainSelector::BAN_TIME),
                );
                self.disconnect_and_ban(peer.0)?;
            }
        }

        Ok(())
    }

    async fn check_tips(&mut self) -> Result<(), WireError> {
        let (height, _) = self.chain.get_best_block()?;
        let validation_index = self.chain.get_validation_index()?;
        if (validation_index + 100) < height {
            let mut tips = self.chain.get_chain_tips()?;
            let (height, hash) = self.chain.get_best_block()?;
            let acc = self.find_accumulator_for_block(height, hash).await?;

            // only one tip, our peers are following the same chain
            if tips.len() == 1 {
                info!(
                    "Assuming chain with {} blocks",
                    self.chain.get_best_block()?.0
                );

                self.context.state = ChainSelectorState::Done;
                self.chain.mark_chain_as_assumed(acc, tips[0]).unwrap();
                self.chain.update_ibd(IBDState::Done);
            }
            // if we have more than one tip, we need to check if our best chain has an invalid block
            tips.remove(0); // no need to check our best one
            for tip in tips {
                self.is_our_chain_invalid(tip).await?;
            }

            return Ok(());
        }

        info!("chain close enough to tip, not asking for utreexo state");
        self.context.state = ChainSelectorState::Done;
        Ok(())
    }

    /// Sends a tracked ordinary or checkpoint-bounded `getheaders` request.
    fn send_tracked_headers_request(
        &mut self,
        peer: PeerId,
        locator: Vec<BlockHash>,
        stop_hash: Option<BlockHash>,
    ) -> Result<(), WireError> {
        let request = match stop_hash {
            Some(stop_hash) => NodeRequest::GetHeadersRange { locator, stop_hash },
            None => NodeRequest::GetHeaders(locator),
        };
        self.send_to_peer(peer, request)?;
        self.context.headers_requests.insert(peer, Instant::now());
        Ok(())
    }

    fn request_headers_from_peer(&mut self, peer: PeerId, tip: BlockHash) -> Result<(), WireError> {
        let locator = self
            .chain
            .get_block_locator_for_tip(tip)
            .unwrap_or_default();
        self.send_tracked_headers_request(peer, locator, None)
    }

    /// Starts one independent PRESYNC state machine for every available connected peer.
    fn start_headers_presync(&mut self) -> Result<(), WireError> {
        let (start_height, start_hash) = self.chain.get_best_block()?;
        let start_bits = self.chain.get_block_header(&start_hash)?.bits;
        let locator = self
            .chain
            .get_block_locator_for_tip(start_hash)
            .unwrap_or_default();
        let peers = self.peer_ids.clone();

        self.context.presync_states.clear();
        self.context.headers_requests.clear();
        self.context.headers_sync_peer = None;
        self.context.redownload_pending.clear();
        self.context.redownload_active.clear();
        self.context.redownload_completed.clear();
        self.context.redownload_next_height = None;
        self.context.redownload_target = None;
        self.context.redownload_complete = false;
        self.inflight.remove(&InflightRequests::Headers);

        for peer in peers {
            if self.context.unavailable_headers_peers.contains(&peer) {
                continue;
            }
            if let Err(error) = self.send_tracked_headers_request(peer, locator.clone(), None) {
                debug!("Failed to start headers presync with peer={peer}: {error}");
                continue;
            }
            self.context.presync_states.insert(
                peer,
                HeadersSyncState::new(start_height, start_hash, start_bits, self.network),
            );
        }

        if self.context.presync_states.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }

        info!(
            "Started headers presync with {} peers",
            self.context.presync_states.len()
        );
        self.context.state = ChainSelectorState::DownloadingHeaders;
        Ok(())
    }

    /// Aborts the remaining PRESYNC requests and starts bounded ranges on distinct fast peers.
    fn begin_parallel_redownload(&mut self, ranges: Vec<HeaderRange>) -> Result<(), WireError> {
        let outstanding = self
            .context
            .headers_requests
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for peer in outstanding {
            self.context.headers_requests.remove(&peer);
            self.context.stale_presync_responses.insert(peer);
        }
        self.context.presync_states.clear();
        self.context.headers_sync_peer = None;
        self.context.redownload_active.clear();
        self.context.redownload_completed.clear();
        self.context.redownload_complete = false;

        let Some(first) = ranges.first().copied() else {
            self.context.redownload_complete = true;
            return self.start_remaining_headers();
        };
        let target = ranges.last().expect("checked as non-empty").end;
        self.context.redownload_next_height = Some(first.start.height + 1);
        self.context.redownload_target = Some(target);
        self.context.redownload_pending = ranges.into();

        info!(
            ranges = self.context.redownload_pending.len(),
            target_height = target.height,
            "Starting parallel checkpoint-range redownload"
        );
        self.dispatch_redownload_ranges()
    }

    /// Fills every idle peer slot with the next checkpoint range.
    fn dispatch_redownload_ranges(&mut self) -> Result<(), WireError> {
        loop {
            let Some(range) = self.context.redownload_pending.front().copied() else {
                return Ok(());
            };

            let mut excluded = self
                .context
                .redownload_active
                .keys()
                .copied()
                .collect::<HashSet<_>>();
            excluded.extend(self.context.unavailable_headers_peers.iter().copied());
            excluded.extend(
                self.peers
                    .keys()
                    .filter(|peer| !self.peer_ids.contains(peer))
                    .copied(),
            );

            let request = NodeRequest::GetHeadersRange {
                locator: vec![range.start.hash],
                stop_hash: range.end.hash,
            };
            let peer =
                match self.send_to_fast_peer_excluding(request, ServiceFlags::NONE, &excluded) {
                    Ok(peer) => peer,
                    Err(WireError::NoPeersAvailable) => return Ok(()),
                    Err(error) => return Err(error),
                };

            self.context.redownload_pending.pop_front();
            self.context
                .redownload_active
                .insert(peer, HeadersRangeDownload::new(range, self.network));
            self.context.headers_requests.insert(peer, Instant::now());
            debug!(
                "Assigned header range {}..={} to peer={peer}",
                range.start.height, range.end.height
            );
        }
    }

    fn request_active_range(&mut self, peer: PeerId) -> Result<(), WireError> {
        let download = self
            .context
            .redownload_active
            .get(&peer)
            .ok_or(WireError::PeerNotFound)?;
        self.send_tracked_headers_request(
            peer,
            vec![download.next_locator_hash()],
            Some(download.stop_hash()),
        )
    }

    /// Requeues an incomplete range and removes its peer from this scheduling round.
    fn fail_redownload_range(&mut self, peer: PeerId) -> Result<(), WireError> {
        if let Some(download) = self.context.redownload_active.remove(&peer) {
            self.context.redownload_pending.push_front(download.range());
        }
        self.context.headers_requests.remove(&peer);
        self.context.unavailable_headers_peers.insert(peer);
        let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
        self.dispatch_redownload_ranges()
    }

    /// Inserts completed ranges in height order through the prevalidated fast path.
    fn commit_completed_redownload_ranges(&mut self) -> Result<(), WireError> {
        let Some(mut next_height) = self.context.redownload_next_height else {
            return Ok(());
        };

        while let Some(headers) = self.context.redownload_completed.remove(&next_height) {
            let count = headers.len() as u32;
            self.chain.push_headers(headers, next_height)?;
            next_height += count;
        }
        self.context.redownload_next_height = Some(next_height);

        let complete = self
            .context
            .redownload_target
            .is_some_and(|target| next_height == target.height + 1)
            && self.context.redownload_pending.is_empty()
            && self.context.redownload_active.is_empty()
            && self.context.redownload_completed.is_empty();
        if !complete {
            return Ok(());
        }

        let target = self
            .context
            .redownload_target
            .expect("completion requires a target");
        if self.chain.get_block_hash(target.height)? != target.hash {
            error!("Direct header insertion did not reproduce the presync checkpoint");
            return Err(WireError::PeerMisbehaving);
        }

        self.context.redownload_complete = true;
        info!(
            height = target.height,
            hash = %target.hash,
            "Finished parallel checkpoint-range redownload"
        );
        self.start_remaining_headers()
    }

    /// Uses the normal latency-weighted selector for headers after checkpoint redownload.
    fn start_remaining_headers(&mut self) -> Result<(), WireError> {
        if self.context.headers_sync_peer.is_some() {
            return Ok(());
        }
        let (_, tip) = self.chain.get_best_block()?;
        let locator = self
            .chain
            .get_block_locator_for_tip(tip)
            .unwrap_or_default();
        let mut excluded = self.context.unavailable_headers_peers.clone();
        excluded.extend(
            self.peers
                .keys()
                .filter(|peer| !self.peer_ids.contains(peer))
                .copied(),
        );
        let peer = self.send_to_fast_peer_excluding(
            NodeRequest::GetHeaders(locator),
            ServiceFlags::NONE,
            &excluded,
        )?;
        self.context.headers_sync_peer = Some(peer);
        self.context.headers_requests.insert(peer, Instant::now());
        Ok(())
    }

    fn check_headers_request_timeouts(&mut self) -> Result<(), WireError> {
        let now = Instant::now();
        let timed_out = self
            .context
            .headers_requests
            .iter()
            .filter_map(|(&peer, &sent_at)| {
                (now.saturating_duration_since(sent_at).as_secs() > ChainSelector::REQUEST_TIMEOUT)
                    .then_some(peer)
            })
            .collect::<Vec<_>>();

        for peer in timed_out {
            warn!("Headers request to peer={peer} timed out; disconnecting it");
            let is_selected_peer = self.context.headers_sync_peer == Some(peer);
            self.context.headers_requests.remove(&peer);

            if let Some(download) = self.context.redownload_active.remove(&peer) {
                self.context.redownload_pending.push_front(download.range());
            } else if self.context.presync_states.remove(&peer).is_some() {
                self.context.stale_presync_responses.insert(peer);
            } else if is_selected_peer {
                // Ordinary header download remains pinned until this peer actually disconnects.
                self.context.headers_requests.insert(peer, now);
            }

            self.context.unavailable_headers_peers.insert(peer);
            let _ = self.send_to_peer(peer, NodeRequest::Shutdown);
        }

        self.dispatch_redownload_ranges()?;
        if self.context.redownload_complete && self.context.headers_sync_peer.is_none() {
            let _ = self.start_remaining_headers();
        } else if !self.context.redownload_complete
            && self.context.presync_states.is_empty()
            && self.context.redownload_pending.is_empty()
            && self.context.redownload_active.is_empty()
        {
            self.context.state = ChainSelectorState::CreatingConnections;
        }
        Ok(())
    }

    /// Rechecks a stale completed tip with the same fast peer after the normal timeout.
    fn maybe_retry_stale_tip(&mut self) -> Result<(), WireError> {
        if !self.context.redownload_complete || !self.context.headers_requests.is_empty() {
            return Ok(());
        }
        if self.context.headers_sync_peer.is_none() {
            return self.start_remaining_headers();
        }
        if self.last_tip_update.elapsed().as_secs() <= ChainSelector::REQUEST_TIMEOUT {
            return Ok(());
        }

        let peer = self.context.headers_sync_peer.expect("checked as present");
        let (_, tip) = self.chain.get_best_block()?;
        self.request_headers_from_peer(peer, tip)
    }

    /// Sends a `getheaders` to all our peers
    ///
    /// After we download all blocks from one peer, we ask our peers if they
    /// agree with our sync peer on what is the best chain. If they are in a fork,
    /// we'll download that fork and compare with our own chain. We should always pick
    /// the most PoW one.
    fn poke_peers(&self) -> Result<(), WireError> {
        let locator = self.chain.get_block_locator().unwrap();
        for peer in self
            .common
            .peer_ids
            .iter()
            .filter(|peer| !self.context.unavailable_headers_peers.contains(peer))
        {
            let get_headers = NodeRequest::GetHeaders(locator.clone());
            self.send_to_peer(*peer, get_headers)?;
        }

        Ok(())
    }

    pub async fn run(&mut self) -> Result<(), WireError> {
        info!("Starting IBD, selecting the best chain");

        let mut ticker = time::interval(ChainSelector::MAINTENANCE_TICK);
        // If we fall behind, don't "catch up" by running maintenance repeatedly
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;

                // Maintenance runs only on tick but has priority
                _ = ticker.tick() => match self.maintenance_tick().await? {
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

        Ok(())
    }

    /// Whether we have enough peers to start downloading headers
    fn can_start_headers_sync(&self) -> bool {
        let connected_peers = self.connected_peers();
        if self.network == Network::Regtest && connected_peers >= 1 {
            return true;
        }

        if self.has_fixed_peers() {
            return connected_peers >= 1;
        }

        connected_peers >= ChainSelector::MAX_OUTGOING_PEERS
    }

    /// Performs the periodic maintenance tasks, including checking for the cancel signal, peer
    /// connections, and inflight request timeouts.
    ///
    /// Returns `LoopControl::Break` if we need to break the main `ChainSelector` loop, either
    /// because the kill signal was set or because the header chain is synced.
    async fn maintenance_tick(&mut self) -> Result<LoopControl, WireError> {
        if *self.kill_signal.read().await {
            return Ok(LoopControl::Break);
        }

        // Checks if we need to open a new connection
        periodic_job!(
            self.last_connection => self.maybe_open_connection(ServiceFlags::NONE),
            ChainSelector::TRY_NEW_CONNECTION,
            no_log,
        );

        // Open new feeler connection periodically
        periodic_job!(
            self.last_feeler => self.open_feeler_connection(),
            ChainSelector::FEELER_INTERVAL,
            no_log,
        );

        if let ChainSelectorState::LookingForForks(start) = self.context.state {
            if start.elapsed().as_secs() > ChainSelector::REQUEST_TIMEOUT {
                self.context.state = ChainSelectorState::LookingForForks(Instant::now());
                self.poke_peers()?;
            }
        }

        if self.context.state == ChainSelectorState::CreatingConnections {
            // Once enough peers are ready, race all of them through independent presyncs.
            if self.can_start_headers_sync() {
                try_and_log!(self.start_headers_presync());
            }
        }

        if self.context.state == ChainSelectorState::DownloadingHeaders {
            try_and_log!(self.maybe_retry_stale_tip());
            try_and_log!(self.check_headers_request_timeouts());
        }

        // We downloaded all headers in the most-pow chain, and all our peers agree
        // this is the most-pow chain, we're done!
        if self.context.state == ChainSelectorState::Done {
            self.chain.update_ibd(IBDState::DownloadingBlocks);
            try_and_log!(self.chain.flush());
            return Ok(LoopControl::Break);
        }

        try_and_log!(self.check_for_timeout());

        Ok(LoopControl::Continue)
    }

    async fn find_accumulator_for_block_step(
        &mut self,
        block: BlockHash,
        height: u32,
    ) -> Result<FindAccResult, WireError> {
        for peer_id in self.common.peer_ids.iter() {
            let peer = self.peers.get(peer_id).unwrap();
            if peer.services.has(service_flags::UTREEXO_ARCHIVE.into()) {
                self.send_to_peer(*peer_id, NodeRequest::GetUtreexoState((block, height)))?;
                self.common.inflight.insert(
                    InflightRequests::UtreexoState(*peer_id),
                    (*peer_id, Instant::now()),
                );
            }
        }

        if self.inflight.is_empty() {
            return Err(WireError::NoPeersAvailable);
        }

        let mut peer_accs = Vec::new();
        loop {
            // wait for all peers to respond or timeout after 1 minute
            if self.inflight.is_empty() {
                break;
            }

            if let Ok(Some(message)) = timeout(Duration::from_secs(60), self.node_rx.recv()).await {
                match message {
                    NodeNotification::DnsSeedAddresses(addresses) => {
                        self.address_man.push_addresses(&addresses);
                    }

                    NodeNotification::FromPeer(peer, message, _) => {
                        if let PeerMessages::UtreexoState(state) = message {
                            self.inflight.remove(&InflightRequests::UtreexoState(peer));
                            info!("got state {state:?}");
                            peer_accs.push((peer, state));
                        }
                    }

                    NodeNotification::FromUser(request, responder) => {
                        self.perform_user_request(request, responder).await;
                    }
                }
            }

            for inflight in self.inflight.clone().iter() {
                if inflight.1.1.elapsed().as_secs() > 60 {
                    self.inflight.remove(inflight.0);
                }
            }
        }

        if peer_accs.len() == 1 {
            warn!("Only one peers with the UTREEXO_FILTER service flag");
            return Ok(FindAccResult::Found(peer_accs.pop().unwrap().1));
        }

        let mut accs = HashSet::new();
        for (_, acc) in peer_accs.iter() {
            accs.insert(acc);
        }

        // if all peers have the same state, we can assume it's the correct one
        if accs.len() == 1 {
            return Ok(FindAccResult::Found(peer_accs.pop().unwrap().1));
        }

        // if we have different states, we need to keep looking until we find the
        // fork point
        Ok(FindAccResult::KeepLooking(peer_accs))
    }

    async fn handle_notification(
        &mut self,
        notification: NodeNotification,
    ) -> Result<(), WireError> {
        match notification {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder).await;
            }

            NodeNotification::FromPeer(peer, notification, time) => {
                self.handle_peer_notification(notification, peer, time)
                    .await?;
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }
        }
        Ok(())
    }

    async fn handle_peer_notification(
        &mut self,
        notification: PeerMessages,
        peer: PeerId,
        time: Instant,
    ) -> Result<(), WireError> {
        self.register_message_time(&notification, peer, time);

        let Some(unhandled) = self.handle_peer_msg_common(notification, peer)? else {
            return Ok(());
        };

        match unhandled {
            PeerMessages::Headers(headers) => {
                if self
                    .inflight
                    .get(&InflightRequests::Headers)
                    .is_some_and(|(request_peer, _)| *request_peer == peer)
                {
                    self.inflight.remove(&InflightRequests::Headers);
                }
                return self.handle_headers(peer, headers, time).await;
            }

            PeerMessages::Ready(version) => {
                self.handle_peer_ready(peer, version)?;
                if matches!(self.context.state, ChainSelectorState::LookingForForks(_)) {
                    let locator = self.chain.get_block_locator().unwrap();
                    self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;
                } else if self.context.state == ChainSelectorState::DownloadingHeaders
                    && !self.context.redownload_pending.is_empty()
                {
                    self.dispatch_redownload_ranges()?;
                }
            }

            PeerMessages::Disconnected(idx) => {
                let was_sync_peer = self.context.headers_sync_peer == Some(peer);
                self.context.headers_requests.remove(&peer);
                self.context.stale_presync_responses.remove(&peer);
                self.context.unavailable_headers_peers.remove(&peer);
                self.context.presync_states.remove(&peer);
                if let Some(download) = self.context.redownload_active.remove(&peer) {
                    self.context.redownload_pending.push_front(download.range());
                }
                if was_sync_peer {
                    self.context.headers_sync_peer = None;
                }

                self.handle_disconnection(peer, idx)?;

                if self.context.state == ChainSelectorState::DownloadingHeaders {
                    self.dispatch_redownload_ranges()?;
                    if self.context.redownload_complete && was_sync_peer {
                        let _ = self.start_remaining_headers();
                    } else if !self.context.redownload_complete
                        && self.context.presync_states.is_empty()
                        && self.context.redownload_pending.is_empty()
                        && self.context.redownload_active.is_empty()
                    {
                        self.context.state = ChainSelectorState::CreatingConnections;
                    }
                }

                if self.peers.is_empty() {
                    self.context.state = ChainSelectorState::CreatingConnections;
                }
            }

            // During chain selection we don't ask for blocks, unless it's an explicit
            // user request made through the node handle. If it isn't, we punish this
            // peer for sending an unrequested block.
            PeerMessages::Block(block) => {
                let block = self.check_is_user_block_and_reply(block)?;

                if block.is_some() {
                    error!("peer {peer} sent us a block we didn't request");
                    self.increase_banscore(peer, 5)?;
                }
            }

            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use bitcoin::CompactTarget;
    use bitcoin::Network;
    use bitcoin::TxMerkleNode;
    use bitcoin::block::Version;
    use bitcoin::hashes::Hash;
    use floresta_chain::AssumeValidArg;
    use floresta_chain::BlockchainInterface;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::FlatChainStoreConfig;
    use floresta_common::Ema;
    use floresta_mempool::Mempool;
    use rustreexo::node_hash::BitcoinNodeHash;
    use tokio::sync::Mutex;
    use tokio::sync::RwLock;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::unbounded_channel;

    use super::*;
    use crate::UtreexoNodeConfig;
    use crate::address_man::AddressMan;
    use crate::node::ConnectionKind;
    use crate::node::LocalPeerView;
    use crate::node::NodeRequest;
    use crate::node::PeerStatus;
    use crate::p2p_wire::transport::TransportProtocol;

    type TestNode = UtreexoNode<Arc<ChainState<FlatChainStore>>, ChainSelector>;

    fn parse_acc(acc: &[u8]) -> Result<Stump, WireError> {
        TestNode::parse_acc(acc)
    }

    fn serialize_acc(leaves: u64, root_count: usize) -> Vec<u8> {
        let mut acc = leaves.to_le_bytes().to_vec();
        acc.extend(std::iter::repeat_n(0u8, 32 * root_count));
        acc
    }

    #[test]
    fn test_parse_acc() {
        let empty_wire = &[] as &[u8];
        let empty_stump = parse_acc(empty_wire).expect("empty wire should parse");
        assert_eq!(empty_stump, Stump::default());

        let zero_leaves_wire = [0u8; 8];
        let zero_leaves_stump =
            parse_acc(&zero_leaves_wire).expect("zero-leaves wire should parse");
        assert_eq!(zero_leaves_stump, Stump::default());

        let header_truncated_min_wire = vec![0u8; 1];
        let header_truncated_min = parse_acc(&header_truncated_min_wire);
        assert!(matches!(
            header_truncated_min,
            Err(WireError::PeerMisbehaving)
        ));

        let header_truncated_max_wire = vec![0u8; 7];
        let header_truncated_max = parse_acc(&header_truncated_max_wire);
        assert!(matches!(
            header_truncated_max,
            Err(WireError::PeerMisbehaving)
        ));

        let roots_missing_wire = 1u64.to_le_bytes().to_vec();
        let roots_missing = parse_acc(&roots_missing_wire);
        assert!(matches!(roots_missing, Err(WireError::PeerMisbehaving)));

        let mut roots_one_byte_short_wire = 1u64.to_le_bytes().to_vec();
        roots_one_byte_short_wire.extend([0u8; 31]);
        let roots_one_byte_short = parse_acc(&roots_one_byte_short_wire);
        assert!(matches!(
            roots_one_byte_short,
            Err(WireError::PeerMisbehaving)
        ));

        let roots_excess_count_wire = serialize_acc(8, 2);
        let roots_excess_count = parse_acc(&roots_excess_count_wire);
        assert!(matches!(
            roots_excess_count,
            Err(WireError::PeerMisbehaving)
        ));

        let mut roots_trailing_byte_wire = serialize_acc(1, 1);
        roots_trailing_byte_wire.push(0);
        let roots_trailing_byte = parse_acc(&roots_trailing_byte_wire);
        assert!(matches!(
            roots_trailing_byte,
            Err(WireError::PeerMisbehaving)
        ));

        let leaves_8_one_root_wire = serialize_acc(8, 1);
        let leaves_8_one_root_stump =
            parse_acc(&leaves_8_one_root_wire).expect("leaves=8 one-root wire should parse");
        assert_eq!(
            leaves_8_one_root_stump,
            Stump {
                leaves: 8,
                roots: vec![BitcoinNodeHash::from([0u8; 32])],
            },
        );

        let mut leaves_1_nonzero_root_wire = 1u64.to_le_bytes().to_vec();
        leaves_1_nonzero_root_wire.extend([1u8; 32]);
        let leaves_1_nonzero_root_stump = parse_acc(&leaves_1_nonzero_root_wire)
            .expect("leaves=1 nonzero root wire should parse");
        assert_eq!(
            leaves_1_nonzero_root_stump,
            Stump {
                leaves: 1,
                roots: vec![BitcoinNodeHash::from([1u8; 32])],
            },
        );

        let leaves_3_two_roots_wire = serialize_acc(3, 2);
        let leaves_3_two_roots_stump =
            parse_acc(&leaves_3_two_roots_wire).expect("leaves=3 two-root wire should parse");
        assert_eq!(
            leaves_3_two_roots_stump,
            Stump {
                leaves: 3,
                roots: vec![
                    BitcoinNodeHash::from([0u8; 32]),
                    BitcoinNodeHash::from([0u8; 32]),
                ],
            },
        );
    }

    fn test_node(network: Network, name: &str) -> TestNode {
        let datadir = format!("./tmp-db/{}.cs_{name}", rand::random::<u32>());
        let chainstore = FlatChainStore::new(FlatChainStoreConfig::new(&datadir)).unwrap();
        let chain =
            Arc::new(ChainState::open(chainstore, network, AssumeValidArg::Disabled).unwrap());

        TestNode::new(
            UtreexoNodeConfig {
                network,
                max_tip_age_secs: u32::MAX,
                ..Default::default()
            },
            chain,
            Arc::new(Mutex::new(Mempool::new(1000))),
            None,
            Arc::new(RwLock::new(false)),
            AddressMan::new(None, &[]),
        )
        .unwrap()
    }

    fn add_test_peer(
        node: &mut TestNode,
        peer: PeerId,
        latency_ms: f64,
    ) -> UnboundedReceiver<NodeRequest> {
        let (sender, receiver) = unbounded_channel();
        let mut message_times = Ema::with_half_life_50();
        message_times.add(latency_ms);
        node.peers.insert(
            peer,
            LocalPeerView {
                message_times,
                address: format!("127.0.0.1:{}", 8333 + peer).parse().unwrap(),
                services: ServiceFlags::NONE,
                user_agent: format!("test_peer_{peer}"),
                height: 0,
                time_offset: 0,
                state: PeerStatus::Ready,
                channel: sender,
                kind: ConnectionKind::Regular(ServiceFlags::NONE),
                banscore: 0,
                _last_message: Instant::now(),
                transport_protocol: TransportProtocol::V2,
            },
        );
        node.peer_ids.push(peer);
        receiver
    }

    fn make_header(prev: BlockHash, time: u32, bits: CompactTarget) -> Header {
        (0u32..=u32::MAX)
            .map(|nonce| Header {
                version: Version::from_consensus(1),
                prev_blockhash: prev,
                merkle_root: TxMerkleNode::all_zeros(),
                time,
                bits,
                nonce,
            })
            .find(|header| header.validate_pow(header.target()).is_ok())
            .expect("regtest target should produce a valid nonce")
    }

    fn make_headers(
        prev: BlockHash,
        first_time: u32,
        bits: CompactTarget,
        count: usize,
    ) -> Vec<Header> {
        let mut headers = Vec::with_capacity(count);
        let mut prev = prev;
        for offset in 0..count {
            let header = make_header(prev, first_time + offset as u32, bits);
            prev = header.block_hash();
            headers.push(header);
        }
        headers
    }

    fn assert_getheaders(receiver: &mut UnboundedReceiver<NodeRequest>) {
        assert!(matches!(
            receiver.try_recv(),
            Ok(NodeRequest::GetHeaders(_) | NodeRequest::GetHeadersRange { .. })
        ));
    }

    fn assert_range_request(receiver: &mut UnboundedReceiver<NodeRequest>) {
        assert!(matches!(
            receiver.try_recv(),
            Ok(NodeRequest::GetHeadersRange { .. })
        ));
    }

    #[test]
    fn presync_starts_for_every_connected_peer() {
        let mut node = test_node(Network::Regtest, "presync_all_peers");
        let mut peer_one = add_test_peer(&mut node, 1, 10.0);
        let mut peer_two = add_test_peer(&mut node, 2, 20.0);

        node.start_headers_presync().unwrap();

        assert_eq!(node.context.presync_states.len(), 2);
        assert_eq!(node.context.headers_requests.len(), 2);
        assert_eq!(node.context.headers_sync_peer, None);
        assert_getheaders(&mut peer_one);
        assert_getheaders(&mut peer_two);
    }

    #[tokio::test]
    async fn first_presync_peer_stops_other_presync_requests() {
        const WINNER: PeerId = 1;
        const OTHER: PeerId = 2;

        let mut node = test_node(Network::Regtest, "presync_first_wins");
        let mut winner_requests = add_test_peer(&mut node, WINNER, 20.0);
        let mut other_requests = add_test_peer(&mut node, OTHER, 30.0);
        node.start_headers_presync().unwrap();
        assert_getheaders(&mut winner_requests);
        assert_getheaders(&mut other_requests);

        let (_, genesis) = node.chain.get_best_block().unwrap();
        let genesis_header = node.chain.get_block_header(&genesis).unwrap();
        let header = make_header(genesis, genesis_header.time + 1, genesis_header.bits);
        node.handle_headers(WINNER, vec![header], Instant::now())
            .await
            .unwrap();

        assert!(node.context.presync_states.is_empty());
        assert!(node.context.stale_presync_responses.contains(&OTHER));
        assert_eq!(node.context.redownload_active.len(), 1);
        assert!(node.context.redownload_pending.is_empty());
        assert_eq!(node.context.headers_requests.len(), 1);
        assert_eq!(node.context.headers_sync_peer, None);
        assert!(!node.context.redownload_complete);
        assert_eq!(node.chain.get_best_block().unwrap().0, 0);

        let range_requests = [winner_requests.try_recv(), other_requests.try_recv()]
            .into_iter()
            .filter(|request| matches!(request, Ok(NodeRequest::GetHeadersRange { .. })))
            .count();
        assert_eq!(range_requests, 1);
    }

    #[tokio::test]
    async fn checkpoint_ranges_download_concurrently_and_commit_in_order() {
        let mut node = test_node(Network::Regtest, "parallel_redownload");
        let mut peer_one = add_test_peer(&mut node, 1, 10.0);
        let mut peer_two = add_test_peer(&mut node, 2, 20.0);
        let mut peer_three = add_test_peer(&mut node, 3, 30.0);
        node.context.state = ChainSelectorState::DownloadingHeaders;

        let (_, genesis) = node.chain.get_best_block().unwrap();
        let genesis_header = node.chain.get_block_header(&genesis).unwrap();
        let headers = make_headers(genesis, genesis_header.time + 1, genesis_header.bits, 6);
        let checkpoints = [
            HeaderCheckpoint {
                height: 0,
                hash: genesis,
                bits: genesis_header.bits,
            },
            HeaderCheckpoint {
                height: 2,
                hash: headers[1].block_hash(),
                bits: headers[1].bits,
            },
            HeaderCheckpoint {
                height: 4,
                hash: headers[3].block_hash(),
                bits: headers[3].bits,
            },
            HeaderCheckpoint {
                height: 6,
                hash: headers[5].block_hash(),
                bits: headers[5].bits,
            },
        ];
        let ranges = checkpoints
            .windows(2)
            .map(|pair| HeaderRange {
                start: pair[0],
                end: pair[1],
            })
            .collect();

        node.context.stale_presync_responses.insert(1);
        node.begin_parallel_redownload(ranges).unwrap();
        assert_eq!(node.context.redownload_active.len(), 3);
        assert_eq!(node.context.headers_requests.len(), 3);
        assert_range_request(&mut peer_one);
        assert_range_request(&mut peer_two);
        assert_range_request(&mut peer_three);

        let peer_one_range = node.context.redownload_active[&1].range();
        node.handle_headers(1, vec![headers[0]], Instant::now())
            .await
            .unwrap();
        assert_eq!(node.context.redownload_active[&1].range(), peer_one_range);
        assert!(node.context.headers_requests.contains_key(&1));

        // Complete the high ranges first. They stay in memory until the missing lower range
        // arrives, then all three are inserted through push_headers in height order.
        let mut assignments = node
            .context
            .redownload_active
            .iter()
            .map(|(&peer, download)| (peer, download.range()))
            .collect::<Vec<_>>();
        assignments.sort_by_key(|item| std::cmp::Reverse(item.1.start.height));
        for (peer, range) in assignments {
            let start = range.start.height as usize;
            let end = range.end.height as usize;
            node.handle_headers(peer, headers[start..end].to_vec(), Instant::now())
                .await
                .unwrap();
        }

        assert!(node.context.redownload_complete);
        assert!(node.context.redownload_active.is_empty());
        assert!(node.context.redownload_completed.is_empty());
        assert_eq!(
            node.chain.get_best_block().unwrap(),
            (6, headers[5].block_hash())
        );

        let sync_peer = node
            .context
            .headers_sync_peer
            .expect("normal header download should use a fast peer");
        match sync_peer {
            1 => assert_getheaders(&mut peer_one),
            2 => assert_getheaders(&mut peer_two),
            3 => assert_getheaders(&mut peer_three),
            _ => panic!("unexpected sync peer"),
        }
        assert_eq!(node.context.state, ChainSelectorState::DownloadingHeaders);

        node.handle_headers(sync_peer, Vec::new(), Instant::now())
            .await
            .unwrap();
        assert!(matches!(
            node.context.state,
            ChainSelectorState::LookingForForks(_)
        ));
    }

    #[tokio::test]
    async fn disconnected_range_is_reassigned_to_an_idle_peer() {
        let mut node = test_node(Network::Regtest, "parallel_failover");
        let mut peer_one = add_test_peer(&mut node, 1, 10.0);
        let mut peer_two = add_test_peer(&mut node, 2, 20.0);
        node.context.state = ChainSelectorState::DownloadingHeaders;

        let (_, genesis) = node.chain.get_best_block().unwrap();
        let genesis_header = node.chain.get_block_header(&genesis).unwrap();
        let headers = make_headers(genesis, genesis_header.time + 1, genesis_header.bits, 2);
        let range = HeaderRange {
            start: HeaderCheckpoint {
                height: 0,
                hash: genesis,
                bits: genesis_header.bits,
            },
            end: HeaderCheckpoint {
                height: 2,
                hash: headers[1].block_hash(),
                bits: headers[1].bits,
            },
        };

        node.begin_parallel_redownload(vec![range]).unwrap();
        let first_peer = *node
            .context
            .redownload_active
            .keys()
            .next()
            .expect("range should be assigned");
        match first_peer {
            1 => assert_range_request(&mut peer_one),
            2 => assert_range_request(&mut peer_two),
            _ => panic!("unexpected range peer"),
        }

        let address_id = node.peers[&first_peer].address.id;
        node.handle_peer_notification(
            PeerMessages::Disconnected(address_id),
            first_peer,
            Instant::now(),
        )
        .await
        .unwrap();

        let replacement = *node
            .context
            .redownload_active
            .keys()
            .next()
            .expect("range should be reassigned");
        assert_ne!(replacement, first_peer);
        assert_eq!(node.context.redownload_active[&replacement].range(), range);
        match replacement {
            1 => assert_range_request(&mut peer_one),
            2 => assert_range_request(&mut peer_two),
            _ => panic!("unexpected replacement peer"),
        }
    }
}
