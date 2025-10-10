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
//! The most critical part of syncing-up a Bitcoin node is making sure you know about the most-work
//! chain. If someone can eclypse you, they can make you start following a chain that only you and
//! the attacker care about. If you get paid in this chain, you can't pay someone else outside this
//! chain, because they will be following other chains. Luckily, we only need one honest peer, to
//! find the best-work chain and avoid any attacker to fools us into accepting payments in a "fake
//! Bitcoin"
//!
//! # Implementation
//!
//! In Floresta, we try to pick a good balance between data downloaded and security. We could
//! simply download all chains from all peers and pick the most work one. But each header is
//! 80 bytes-long, with ~800k blocks, that's around 60 MBs. If we have 10 peers, that's 600MBs
//! (excluding overhead by the p2p messages). Moreover, it's very uncommon to actually have peers
//! in different chains. So we can optmistically download all headers from one random peer, and
//! then check with the others if they agree. If they have another chain for us, we download that
//! chain, and pick whichever has more work.
//!
//! Most likely we'll only download one chain and all peers will agree with it. Then we can start
//! downloading the actual blocks and validating them.

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;

use bitcoin::block::Header;
use bitcoin::consensus::deserialize;
use bitcoin::p2p::message_blockdata::Inventory;
use bitcoin::p2p::ServiceFlags;
use bitcoin::Block;
use bitcoin::BlockHash;
use floresta_chain::proof_util;
use floresta_chain::ChainBackend;
use floresta_chain::CompactLeafData;
use floresta_common::service_flags;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use rustreexo::accumulator::proof::Proof;
use rustreexo::accumulator::stump::Stump;
use tokio::time::timeout;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use super::error::WireError;
use super::node_interface::UserRequest;
use super::peer::PeerMessages;
use crate::address_man::AddressState;
use crate::block_proof::Bitmap;
use crate::node::InflightBlock;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_interface::NodeResponse;
use crate::p2p_wire::macros::periodic_job;
use crate::p2p_wire::macros::try_and_error;
use crate::p2p_wire::macros::try_and_warn;

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

impl NodeContext for ChainSelector {
    const REQUEST_TIMEOUT: u64 = 60; // Ban peers stalling our IBD
    const TRY_NEW_CONNECTION: u64 = 10; // Try creating connections more aggressively

    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::NETWORK | service_flags::UTREEXO.into() | service_flags::UTREEXO_FILTER.into()
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
                error!("Error while downloading headers from peer={peer} err={e}");

                self.send_to_peer(peer, NodeRequest::Shutdown)?;

                let peer = self.peers.get(&peer).unwrap();
                self.common.address_man.update_set_state(
                    peer.address_id as usize,
                    AddressState::Banned(ChainSelector::BAN_TIME),
                );
            }
        }

        let last = headers.last().unwrap().block_hash();
        self.context
            .tip_cache
            .entry(peer)
            .and_modify(|e| *e = last)
            .or_insert(last);

        self.request_headers(last)
    }

    /// Takes a serialized accumulator and parses it into a Stump
    fn parse_acc(mut acc: Vec<u8>) -> Result<Stump, WireError> {
        if acc.is_empty() {
            return Ok(Stump::default());
        }
        let leaves = deserialize(acc.drain(0..8).as_slice()).unwrap_or(0);
        let mut roots = Vec::new();
        while !acc.is_empty() {
            let slice = acc.drain(0..32);
            let mut root = [0u8; 32];
            root.copy_from_slice(&slice.collect::<Vec<u8>>());
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
            if let Ok(Some(NodeNotification::FromPeer(peer, PeerMessages::UtreexoState(state)))) =
                timeout(Duration::from_secs(60), self.node_rx.recv()).await
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
            (Some(acc1), Some(_)) => Self::parse_acc(acc1)?,
            (Some(acc1), None) => Self::parse_acc(acc1)?,
            (None, Some(acc2)) => Self::parse_acc(acc2)?,
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

        let proof = inflight_block.proof.expect("Block proof should be present");
        let leaf_data = inflight_block
            .leaf_data
            .expect("Leaf data should be present");

        let acc1 = self.update_acc(agreed, inflight_block.block, proof, &leaf_data, fork + 1)?;

        let peer1_acc = Self::parse_acc(peer1_acc)?;
        let peer2_acc = Self::parse_acc(peer2_acc)?;

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

            let Some(NodeNotification::FromPeer(id, message)) = self.node_rx.recv().await else {
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
                        self.increase_banscore(peer, self.max_banscore)?;
                        return Err(WireError::PeerMisbehaving);
                    }

                    // Check if the blocks was maliciously mutated by our peer
                    let is_mutated =
                        !(recv_block.check_merkle_root() && recv_block.check_witness_commitment());

                    if is_mutated {
                        error!(
                            "Peer {peer} sent us a mutated block {}",
                            recv_block.block_hash()
                        );
                        self.increase_banscore(peer, self.config.max_banscore)?;
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
                        self.increase_banscore(peer, self.config.max_banscore)?;
                        return Err(WireError::PeerMisbehaving);
                    };

                    let proof = Proof {
                        hashes: uproof.proof_hashes,
                        targets: uproof.targets,
                    };

                    return Ok(InflightBlock {
                        block,
                        proof: Some(proof),
                        leaf_data: Some(uproof.leaf_data),
                        peer,
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
        block: Block,
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
                let acc = Self::parse_acc(acc)?;
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
                    self.send_to_peer(liar, NodeRequest::Shutdown)?;
                    if liar == peer1 {
                        invalid_accs.insert(peer[0].1.clone());
                        continue;
                    }
                    invalid_accs.insert(peer[1].1.clone());
                }
                PeerCheck::UnresponsivePeer(dead_peer) => {
                    self.send_to_peer(dead_peer, NodeRequest::Shutdown)?;
                }
                PeerCheck::BothUnresponsivePeers => {
                    self.send_to_peer(peer1, NodeRequest::Shutdown)?;
                    self.send_to_peer(peer2, NodeRequest::Shutdown)?;
                }
                PeerCheck::BothLying => {
                    self.send_to_peer(peer1, NodeRequest::Shutdown)?;
                    self.send_to_peer(peer2, NodeRequest::Shutdown)?;
                    invalid_accs.insert(peer[0].1.clone());
                    invalid_accs.insert(peer[1].1.clone());
                }
            }
        }
        //filter out the invalid accs
        candidate_accs.retain(|acc| !invalid_accs.contains(&acc.1));
        //we should have only one candidate left
        assert_eq!(candidate_accs.len(), 1);

        Self::parse_acc(candidate_accs.pop().unwrap().1)
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
                info!("Finished downloading headers from peer={peer}, checking if our peers agree");
                self.poke_peers()?;
                self.context.state = ChainSelectorState::LookingForForks(Instant::now());
                self.context.done_peers.insert(peer);
            }
            ChainSelectorState::LookingForForks(_) => {
                self.context.done_peers.insert(peer);
                for peer in self.common.peer_ids.iter() {
                    // at least one peer haven't finished
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
                    .contains_key(&ServiceFlags::from(1 << 25));

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
            .choose(&mut thread_rng())
            .ok_or(WireError::NoPeersAvailable)?;

        let block = self.get_block_and_proof(rand_peer, fork).await?;
        let leaf_data = block.leaf_data.expect("Leaf data should be present");
        let proof = block.proof.expect("Block proof should be present");
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
                    peer.1.address_id as usize,
                    AddressState::Banned(ChainSelector::BAN_TIME),
                );
                self.send_to_peer(peer.0, NodeRequest::Shutdown)?;
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
                self.chain.toggle_ibd(false);
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

    /// Ask for headers, given a tip
    ///
    /// This function will send a `getheaders` request to our peers, assuming this
    /// peer is following a chain with `tip` inside it. We use this in case some of
    /// our peer is in a fork, so we can learn about all blocks in that fork and
    /// compare the candidate chains to pick the best one.
    fn request_headers(&mut self, tip: BlockHash) -> Result<(), WireError> {
        let locator = self
            .chain
            .get_block_locator_for_tip(tip)
            .unwrap_or_default();

        let peer =
            self.send_to_fastest_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)?;

        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }

    /// Sends a `getheaders` to all our peers
    ///
    /// After we download all blocks from one peer, we ask our peers if they
    /// agree with our sync peer on what is the best chain. If they are in a fork,
    /// we'll download that fork and compare with our own chain. We should always pick
    /// the most PoW one.
    fn poke_peers(&self) -> Result<(), WireError> {
        let locator = self.chain.get_block_locator().unwrap();
        for peer in self.common.peer_ids.iter() {
            let get_headers = NodeRequest::GetHeaders(locator.clone());
            self.send_to_peer(*peer, get_headers)?;
        }

        Ok(())
    }

    pub async fn run(&mut self) -> Result<(), WireError> {
        info!("Starting IBD, selecting the best chain");

        loop {
            while let Ok(notification) = timeout(Duration::from_secs(1), self.node_rx.recv()).await
            {
                match notification {
                    Some(NodeNotification::FromUser(request, responder)) => {
                        self.perform_user_request(request, responder);
                    }

                    Some(NodeNotification::FromPeer(peer, notification)) => {
                        try_and_warn!("Failure to process notification from peer", self.handle_peer_notification(notification, peer).await);
                    }

                    Some(NodeNotification::DnsSeedAddresses(addresses)) => {
                        self.address_man.push_addresses(&addresses);
                    }

                    None => {
                        break;
                    }
                }

                // Shutdown if needed while in the notifications loop
                if *self.kill_signal.read().await {
                    self.shutdown();
                    return Ok(());
                }
            }

            // Checks if we need to open a new connection
            periodic_job!(
                self.maybe_open_connection(ServiceFlags::NONE),
                self.last_connection,
                TRY_NEW_CONNECTION,
                ChainSelector
            );

            // Open new feeler connection periodically
            periodic_job!(
                self.open_feeler_connection(),
                self.last_feeler,
                FEELER_INTERVAL,
                ChainSelector
            );

            if let ChainSelectorState::LookingForForks(start) = self.context.state {
                if start.elapsed().as_secs() > 30 {
                    self.context.state = ChainSelectorState::LookingForForks(Instant::now());
                    self.poke_peers()?;
                }
            }

            if self.context.state == ChainSelectorState::CreatingConnections {
                // If we have enough peers, try to download headers
                if !self.peer_ids.is_empty() {
                    try_and_error!("Failure sending get headers", self.request_headers(self.chain.get_best_block()?.1));
                    self.context.state = ChainSelectorState::DownloadingHeaders;
                }
            }

            // We downloaded all headers in the most-pow chain, and all our peers agree
            // this is the most-pow chain, we're done!
            if self.context.state == ChainSelectorState::Done {
                try_and_warn!("Failure flushing chain", self.chain.flush());
                break;
            }

            try_and_warn!("Failure checking for timeout", self.check_for_timeout());
            if *self.kill_signal.read().await {
                self.shutdown();
                break;
            }
        }

        Ok(())
    }

    async fn find_accumulator_for_block_step(
        &mut self,
        block: BlockHash,
        height: u32,
    ) -> Result<FindAccResult, WireError> {
        for peer_id in self.common.peer_ids.iter() {
            let peer = self.peers.get(peer_id).unwrap();
            if peer.services.has(ServiceFlags::from(1 << 25)) {
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

                    NodeNotification::FromPeer(peer, message) => {
                        if let PeerMessages::UtreexoState(state) = message {
                            self.inflight.remove(&InflightRequests::UtreexoState(peer));
                            info!("got state {state:?}");
                            peer_accs.push((peer, state));
                        }
                    }

                    NodeNotification::FromUser(request, responder) => {
                        self.perform_user_request(request, responder);
                    }
                }
            }

            for inflight in self.inflight.clone().iter() {
                if inflight.1 .1.elapsed().as_secs() > 60 {
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

    async fn handle_peer_notification(
        &mut self,
        notification: PeerMessages,
        peer: PeerId,
    ) -> Result<(), WireError> {
        self.register_message_time(&notification, peer);

        match notification {
            PeerMessages::Headers(headers) => {
                self.inflight.remove(&InflightRequests::Headers);
                return self.handle_headers(peer, headers).await;
            }

            PeerMessages::Ready(version) => {
                self.handle_peer_ready(peer, &version)?;
                if matches!(self.context.state, ChainSelectorState::LookingForForks(_)) {
                    let locator = self.chain.get_block_locator().unwrap();
                    self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;
                }
            }

            PeerMessages::Disconnected(idx) => {
                if self.peers.is_empty() {
                    self.context.state = ChainSelectorState::CreatingConnections;
                }
                self.handle_disconnection(peer, idx)?;
            }

            PeerMessages::Addr(addresses) => {
                let addresses: Vec<_> = addresses.iter().cloned().map(|addr| addr.into()).collect();
                self.address_man.push_addresses(&addresses);
            }

            PeerMessages::Block(block) => {
                // During chain selection we don't ask for blocks, unless it's an explicit
                // user request made through the node handle. If it isn't, we punish this
                // peer for sending an unrequested block.
                if self.check_is_user_block_and_reply(block)?.is_some() {
                    error!("peer {peer} sent us a block we didn't request");
                    self.increase_banscore(peer, 5)?;
                }
            }

            PeerMessages::NotFound(inv) => match inv {
                Inventory::Error => {}

                Inventory::Block(block)
                | Inventory::WitnessBlock(block)
                | Inventory::CompactBlock(block) => {
                    let request = self
                        .inflight_user_requests
                        .remove(&UserRequest::Block(block));
                    if let Some(request) = request {
                        request
                            .2
                            .send(NodeResponse::Block(None))
                            .map_err(|_| WireError::ResponseSendError)?;
                    }
                }

                Inventory::WitnessTransaction(tx) | Inventory::Transaction(tx) => {
                    let request = self
                        .inflight_user_requests
                        .remove(&UserRequest::MempoolTransaction(tx));
                    if let Some(request) = request {
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

                let request = self
                    .inflight_user_requests
                    .remove(&UserRequest::MempoolTransaction(tx.compute_txid()));
                if let Some(request) = request {
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

            _ => {}
        }
        Ok(())
    }
}
