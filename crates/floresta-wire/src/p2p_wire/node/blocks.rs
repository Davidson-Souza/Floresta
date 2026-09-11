// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::ChainBackend;
use floresta_chain::CompactLeafData;
use floresta_chain::extensions::BlockExt;
use floresta_chain::proof_util;
use floresta_chain::proof_util::UtreexoLeafError;
use floresta_common::service_flags;
use floresta_common::try_and_log;
use rustreexo::proof::Proof;
use tracing::debug;
use tracing::error;
use tracing::warn;

use super::InflightRequests;
use super::NodeRequest;
use super::UtreexoNode;
use crate::block_proof::Bitmap;
use crate::block_proof::UtreexoProof;
use crate::node::WitnessMode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_handle::NodeResponse;
use crate::node_handle::UserRequest;
use crate::p2p_wire::error::WireError;

/// The leaf data, utreexo proof and the peer that sent them.
type UtreexoData = (Vec<CompactLeafData>, Proof, PeerId);

#[derive(Debug)]
/// A block that is currently being downloaded or pending processing
///
/// To download a block, we first request the block itself, and then we
/// request the proof and leaf data for it. This struct holds the data
/// we already have. We may also keep it around, as we may receive blocks
/// out of order, so while we wait for the previous blocks to finish download,
/// we keep the blocks that are already downloaded as an [`InflightBlock`].
pub(crate) struct InflightBlock {
    /// The peer that sent the block.
    pub peer: PeerId,

    /// The block itself.
    pub block: Arc<Block>,

    /// Auxiliary data needed for validating this block. Currently, it includes utreexo
    /// leaf data (previous UTXOs spent in the block), the corresponding accumulator
    /// inclusion proof, and the peer id that provided them.
    pub aux_data: Option<UtreexoData>,

    /// If this block is currently being processed in a worker, this is the start time.
    pub processing_since: Option<Instant>,
}

impl InflightBlock {
    /// Creates a new `InflightBlock` from a block and the associated peer id.
    ///
    /// If the block doesn't spend any output (i.e., coinbase transaction only) this method adds
    /// empty auxiliary data, which marks this inflight block as ready to process. Blocks with
    /// transactions require [`UtreexoData`] (see [`InflightBlock::add_utreexo_data`]).
    fn new(block: Arc<Block>, peer: PeerId) -> Self {
        let aux_data = match block.txdata.len() {
            1 => Some((Vec::new(), Proof::default(), peer)),
            _ => None, // we need auxiliary data for the txs
        };

        Self {
            peer,
            block,
            aux_data,
            processing_since: None,
        }
    }

    /// Attaches the auxiliary utreexo data to this `InflightBlock`.
    fn add_utreexo_data(&mut self, leaf_data: Vec<CompactLeafData>, proof: Proof, peer: PeerId) {
        self.aux_data = Some((leaf_data, proof, peer));
    }
}

/// Outcome of [`UtreexoNode::enforce_block_structure_check_and_retry_user_request`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockStructureCheckOutcome {
    /// Caller should continue to user-reply/storage handling. This does not by itself
    /// mean the block is valid: authentic consensus-invalid blocks also continue under
    /// existing semantics.
    Continue,
    /// The block was bad and the user request was retried with another peer; the
    /// caller must stop processing this block.
    UserRequestRetried,
}

impl<T, Chain> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
{
    /// Returns `true` only if we can request `BLOCKS_PER_GETDATA` without exceeding the maximum
    /// unprocessed blocks allowed.
    pub(crate) fn can_request_more_blocks(&self) -> bool {
        let max_inflight_blocks = T::BLOCKS_PER_GETDATA * T::MAX_CONCURRENT_GETDATA;

        // If we do a GETDATA request, this will be the new unprocessed count
        let next_unprocessed = self.unprocessed_blocks() + T::BLOCKS_PER_GETDATA;

        next_unprocessed <= max_inflight_blocks
    }

    /// Returns the number of blocks awaiting processing (in memory or requested).
    pub(crate) fn unprocessed_blocks(&self) -> usize {
        let blocks_in_mem = self.blocks.len();
        let requested_blocks = self
            .inflight
            .keys()
            .filter(|inflight| matches!(inflight, InflightRequests::Blocks(_)))
            .count();

        blocks_in_mem + requested_blocks
    }

    pub(crate) fn request_blocks(&mut self, blocks: Vec<BlockHash>) -> Result<(), WireError> {
        let should_request = |block: &BlockHash| {
            let is_inflight = self
                .inflight
                .contains_key(&InflightRequests::Blocks(*block));
            let is_pending = self.blocks.contains_key(block);

            !(is_inflight || is_pending)
        };

        let blocks: Vec<_> = blocks.into_iter().filter(should_request).collect();
        // if there's no block to request, don't propagate any message
        if blocks.is_empty() {
            return Ok(());
        }

        let block_req = NodeRequest::GetBlock(blocks.clone(), self.witness_mode);
        let peer = self.send_to_fast_peer(block_req, ServiceFlags::NETWORK)?;

        for block in blocks.iter() {
            self.inflight
                .insert(InflightRequests::Blocks(*block), (peer, Instant::now()));
        }

        Ok(())
    }

    /// Validates the structure of the block using
    /// [`Block::check_block_structure`].
    ///
    /// If validation fails, [`Self::handle_block_error`] is called to handle the
    /// error and apply the appropriate network consensus.
    ///
    /// `invalidate_chain`: when `true` and the block is consensus-invalid, it is invalidated in
    /// our chain with [`UpdatableChainstate::invalidate_block`]. When `false`, the chain is left
    /// untouched.
    ///
    /// [`Block::check_block_structure`]: floresta_chain::extensions::BlockExt::check_block_structure
    /// [`UpdatableChainstate::invalidate_block`]: floresta_chain::pruned_utreexo::UpdatableChainstate::invalidate_block
    pub(crate) fn enforce_block_structure_check(
        &mut self,
        block: &Block,
        peer: PeerId,
        invalidate_chain: bool,
    ) -> Result<(), WireError> {
        let Err(e) = block.check_block_structure() else {
            return Ok(());
        };

        self.handle_block_error(
            e.into(),
            Arc::new(block.clone()),
            peer,
            None,
            invalidate_chain,
        )
    }

    /// Enforces the block structure consensus with [`Self::enforce_block_structure_check`],
    /// banning the peer if the block is bad.
    ///
    /// Returns [`BlockStructureCheckOutcome::UserRequestRetried`] if the block was bad and
    /// shouldn't be processed any further because it was a user request retried with
    /// another peer, keeping the request open. Otherwise returns
    /// [`BlockStructureCheckOutcome::Continue`], meaning the caller should continue to
    /// user-reply/storage handling; when the block was bad and no retry happened, a
    /// `WireError` is returned instead.
    ///
    /// When no other peer is available to retry with, the user request is failed with
    /// a `None` block instead of being left open forever, since user requests have no
    /// timeout.
    pub(crate) fn enforce_block_structure_check_and_retry_user_request(
        &mut self,
        block: &Block,
        peer: PeerId,
    ) -> Result<BlockStructureCheckOutcome, WireError> {
        let block_hash = block.block_hash();
        let is_user_request = self
            .inflight_user_requests
            .contains_key(&UserRequest::Block(block_hash));

        // User-requested fetches must not mutate our chain state
        let Err(e) = self.enforce_block_structure_check(block, peer, !is_user_request) else {
            return Ok(BlockStructureCheckOutcome::Continue);
        };

        if is_user_request {
            // Retry the block elsewhere; the user request stays open
            // and the inflight entry re-arms the timeout machinery.
            // The offending peer is excluded from the retry: manual peers
            // are exempt from bans, so it stays eligible and could be
            // picked again, sending another mutated block and retriggering
            // this retry indefinitely.
            match self.send_to_fast_peer_excluding(
                NodeRequest::GetBlock(vec![block_hash], WitnessMode::Full),
                ServiceFlags::NETWORK,
                peer,
            ) {
                Ok(new_peer) => {
                    let now = Instant::now();
                    self.inflight
                        .insert(InflightRequests::Blocks(block_hash), (new_peer, now));

                    // Disconnection handling drops user requests by their registered peer, and the
                    // banned offender disconnects right after the ban, so the request must track
                    // the replacement peer or it would be lost before the replacement replies.
                    if let Some(request) = self
                        .inflight_user_requests
                        .get_mut(&UserRequest::Block(block_hash))
                    {
                        request.0 = new_peer;
                        request.1 = now;
                    }
                    return Ok(BlockStructureCheckOutcome::UserRequestRetried);
                }
                Err(err) => warn!("couldn't retry block {block_hash} with another peer: {err}"),
            }

            // We couldn't retry the request with the other peers, so we respond
            // to the user with `None`.
            if let Some(request) = self
                .inflight_user_requests
                .remove(&UserRequest::Block(block_hash))
            {
                request
                    .2
                    .send(NodeResponse::Block(None))
                    .map_err(|_| WireError::ResponseSendError)?;
            }
        }

        Err(e)
    }

    /// Handles a block received from a peer at the notification level.
    ///
    /// Removes the matching [`InflightRequests::Blocks`] entry and checks for an
    /// outstanding [`UserRequest::Block`]. Blocks that were neither requested nor
    /// user-requested are punished via [`Self::punish_unrequested_block`] and dropped
    /// before validation, so the structure check cannot touch the chain.
    pub(crate) fn handle_received_block(
        &mut self,
        block: Block,
        peer: PeerId,
    ) -> Result<Option<Block>, WireError> {
        let block_hash = block.block_hash();
        let was_requested = self
            .inflight
            .remove(&InflightRequests::Blocks(block_hash))
            .is_some();
        let is_user_request = self
            .inflight_user_requests
            .contains_key(&UserRequest::Block(block_hash));

        // A block we never asked for must not reach block validation: discard it before
        // the structure check can touch the chain (e.g. `invalidate_block`).
        if !was_requested && !is_user_request {
            self.punish_unrequested_block(peer, block_hash)?;
            return Ok(None);
        }

        // Enforce the block structure consensus (which may ban the peer) and check
        // whether there is an outstanding user request before proceeding.
        match self.enforce_block_structure_check_and_retry_user_request(&block, peer)? {
            BlockStructureCheckOutcome::UserRequestRetried => return Ok(None),
            BlockStructureCheckOutcome::Continue => {}
        }

        // Reply and return early if it's a user-requested block. Else continue handling it.
        self.check_is_user_block_and_reply(block)
    }

    /// Stores an accepted block and requests its Utreexo proof when needed.
    ///
    /// The block must have been accepted by [`Self::handle_received_block`]: requested
    /// gating, structure validation, user-request retry and user replies all happen
    /// there. Blocks with only a coinbase need no proof and are immediately ready.
    pub(crate) fn request_block_proof(
        &mut self,
        block: Block,
        peer: PeerId,
    ) -> Result<(), WireError> {
        let block_hash = block.block_hash();
        let txdata_len = block.txdata.len();
        debug!("Received block {block_hash} from peer {peer}, with {txdata_len} txs");

        self.blocks
            .insert(block_hash, InflightBlock::new(Arc::new(block), peer));

        // We only need auxiliary utreexo data if there are non-coinbase transactions
        if txdata_len != 1 {
            let utreexo_peer = self.send_to_fast_peer(
                NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
                service_flags::UTREEXO.into(),
            )?;

            self.inflight.insert(
                InflightRequests::UtreexoProof(block_hash),
                (utreexo_peer, Instant::now()),
            );
        }

        Ok(())
    }

    pub(crate) fn attach_proof(
        &mut self,
        uproof: UtreexoProof,
        peer: PeerId,
    ) -> Result<(), WireError> {
        debug!("Received utreexo proof for block {}", uproof.block_hash);
        self.inflight
            .remove(&InflightRequests::UtreexoProof(uproof.block_hash));

        let Some(block) = self.blocks.get_mut(&uproof.block_hash) else {
            warn!(
                "Received utreexo proof for block {}, but we don't have it",
                uproof.block_hash
            );
            self.increase_banscore(peer, 5)?;

            return Ok(());
        };

        let proof = Proof {
            hashes: uproof.proof_hashes,
            targets: uproof.targets,
        };

        // Add the proof and leaf data, together with the peer id that sent them
        block.add_utreexo_data(uproof.leaf_data, proof, peer);

        Ok(())
    }

    /// Asks all utreexo peers for proofs of blocks that we have, but haven't received proofs
    /// for yet, and don't have any GetProofs inflight. This may be caused by a peer disconnecting
    /// while we didn't have more utreexo peers to redo the request.
    pub(crate) fn ask_for_missed_proofs(&mut self) -> Result<(), WireError> {
        // If we have no peers, we can't ask for proofs
        if !self.has_utreexo_peers() {
            return Ok(());
        }

        let pending_blocks = self
            .blocks
            .iter()
            .filter_map(|(hash, block)| {
                if block.aux_data.is_some() {
                    return None;
                }

                if !self
                    .inflight
                    .contains_key(&InflightRequests::UtreexoProof(*hash))
                {
                    return Some(*hash);
                }

                None
            })
            .collect::<Vec<_>>();

        for block_hash in pending_blocks {
            let peer = self.send_to_fast_peer(
                NodeRequest::GetBlockProof((block_hash, Bitmap::new(), Bitmap::new())),
                service_flags::UTREEXO.into(),
            )?;

            self.inflight.insert(
                InflightRequests::UtreexoProof(block_hash),
                (peer, Instant::now()),
            );
        }

        Ok(())
    }

    /// Processes ready blocks in order, stopping at the tip or the first missing block/proof.
    /// Call again when new blocks or proofs arrive.
    pub(crate) fn process_pending_blocks(&mut self) -> Result<(), WireError>
    where
        Chain::Error: From<UtreexoLeafError>,
    {
        loop {
            let best_block = self.chain.get_best_block()?.0;
            let next_block = self.chain.get_validation_index()? + 1;
            if next_block > best_block {
                // If we are at the best block, we don't need to process any more blocks
                return Ok(());
            }

            let next_block_hash = self.chain.get_block_hash(next_block)?;

            let Some(block) = self.blocks.get(&next_block_hash) else {
                // If we don't have the next block, we can't process it
                return Ok(());
            };

            if block.aux_data.is_none() {
                // If the block doesn't have a proof, we can't process it
                return Ok(());
            }

            let start = Instant::now();
            self.process_block(next_block, next_block_hash)?;

            let elapsed = start.elapsed().as_secs_f64();
            self.block_sync_avg.add(elapsed);

            #[cfg(feature = "metrics")]
            {
                use floresta_metrics::get_metrics;

                let avg = self.block_sync_avg.value().expect("at least one sample");
                let metrics = get_metrics();
                metrics.avg_block_processing_time.set(avg);
            }
        }
    }

    /// Actually process a block that is ready to be processed.
    ///
    /// This function will take the next block in our chain, process its proof and validate it.
    /// If everything is correct, it will connect the block to our chain.
    fn process_block(&mut self, block_height: u32, block_hash: BlockHash) -> Result<(), WireError>
    where
        Chain::Error: From<UtreexoLeafError>,
    {
        debug!("processing block {block_hash}");

        let inflight = self
            .blocks
            .remove(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        let block = inflight.block;
        let peer = inflight.peer;
        let (leaf_data, proof, utreexo_peer) =
            inflight.aux_data.ok_or(WireError::BlockProofNotFound)?;

        // The leaf data supplied by the utreexo peer is consumed here, before it is
        // authenticated, so errors must go through the same handling as `connect_block` ones;
        // otherwise a misbehaving peer would go unpunished and the block would silently stall.
        let (del_hashes, inputs) = match proof_util::process_proof(
            &leaf_data,
            &block.txdata,
            block_height,
            |h| self.chain.get_block_hash(h),
            |h| self.chain.get_mtp_by_height(h),
        ) {
            Ok(processed) => processed,
            Err(err) => {
                return self.handle_process_block_error(err.into(), block, peer, utreexo_peer);
            }
        };

        if let Err(err) = self.chain.connect_block(&block, proof, inputs, del_hashes) {
            return self.handle_process_block_error(
                WireError::Blockchain(err),
                block,
                peer,
                utreexo_peer,
            );
        }

        self.last_tip_update = Instant::now();
        Ok(())
    }

    /// Handles an error raised while processing a block, either by [`proof_util::process_proof`]
    /// or [`UpdatableChainstate::connect_block`].
    ///
    /// Non-blockchain errors are propagated unchanged; blockchain ones are forwarded to
    /// [`Self::handle_block_error`], which blames and punishes the responsible peer.
    ///
    /// [`UpdatableChainstate::connect_block`]: floresta_chain::pruned_utreexo::UpdatableChainstate::connect_block
    fn handle_process_block_error(
        &mut self,
        err: WireError,
        block: Arc<Block>,
        block_peer: PeerId,
        utreexo_peer: PeerId,
    ) -> Result<(), WireError> {
        let WireError::Blockchain(chain_err) = err else {
            Err(err)?
        };

        // Blocks reaching `process_block` are chain blocks: user-requested blocks are consumed
        // earlier, in `handle_received_block`.
        self.handle_block_error(chain_err, block, block_peer, Some(utreexo_peer), true)
    }

    /// Handles chain errors caused by peer-supplied block or Utreexo data.
    ///
    /// Disconnects and bans the responsible peer, if any. Local failures are not punished,
    /// and consensus-invalid blocks are only invalidated.
    ///
    /// `utreexo_peer` is `None` when no Utreexo peer was involved yet (see #1280).
    ///
    /// `invalidate_chain`: when `true` and the block is consensus-invalid, it is invalidated in
    /// our chain with [`UpdatableChainstate::invalidate_block`]. When `false`, the chain is left
    /// untouched.
    ///
    /// [`UpdatableChainstate::invalidate_block`]: floresta_chain::pruned_utreexo::UpdatableChainstate::invalidate_block
    fn handle_block_error(
        &mut self,
        chain_err: BlockchainError,
        block: Arc<Block>,
        block_peer: PeerId,
        utreexo_peer: Option<PeerId>,
        invalidate_chain: bool,
    ) -> Result<(), WireError> {
        // Return early if the error is not from block validation (e.g., a database error)
        let Some(e) = Self::block_validation_err(chain_err) else {
            return Ok(());
        };

        let block_hash = block.block_hash();

        let blamed_peer = match utreexo_peer {
            Some(utreexo_peer) => self.blame_peer_for_block_error(
                &e,
                block,
                block_peer,
                utreexo_peer,
                invalidate_chain,
            ),
            None => {
                self.blame_block_peer_for_block_error(&e, block_peer, block_hash, invalidate_chain)
            }
        };

        let Some(blamed_peer) = blamed_peer else {
            return Ok(());
        };

        error!(
            "Validation failed for block {block_hash}, received by peer {blamed_peer}. Reason: {e}"
        );

        // Disconnect the responsible peer and ban it.
        self.disconnect_and_ban(blamed_peer)?;
        Err(WireError::PeerMisbehaving)
    }

    /// Returns the inner block-validation error, if any.
    pub(crate) fn block_validation_err(e: BlockchainError) -> Option<BlockValidationErrors> {
        match e {
            BlockchainError::TransactionError(tx_err) => Some(tx_err.error),
            BlockchainError::BlockValidation(block_err) => Some(block_err),
            BlockchainError::AccumulatorError(_)
            | BlockchainError::InvalidUtreexoProof
            | BlockchainError::UtreexoLeaf(_) => Some(BlockValidationErrors::InvalidUtreexoProof),
            _ => None,
        }
    }

    /// Handles the different block validation errors that can happen when connecting a block.
    ///
    /// Utreexo errors are checked first, since only the proof peer can cause them; everything
    /// else is delegated to [`Self::blame_block_peer_for_block_error`].
    ///
    /// Returns the peer id that caused this error, if any.
    fn blame_peer_for_block_error(
        &mut self,
        e: &BlockValidationErrors,
        block: Arc<Block>,
        block_peer: PeerId,
        utreexo_peer: PeerId,
        invalidate_chain: bool,
    ) -> Option<PeerId> {
        let hash = block.block_hash();

        match e {
            // The utreexo peer sent us an invalid utreexo proof. Block is not yet processed.
            BlockValidationErrors::InvalidUtreexoProof => {
                self.blocks
                    .insert(hash, InflightBlock::new(block, block_peer));

                warn!("Proof for block {hash} is invalid, banning peer {utreexo_peer}");
                Some(utreexo_peer)
            }

            // The utreexo peer sent us incomplete leaf data. A same-block spend falls through
            // below: the block is invalid, not the proof.
            BlockValidationErrors::UtxoNotFound(outpoint)
                if !block.is_invalid_missing_input(*outpoint) =>
            {
                self.blocks
                    .insert(hash, InflightBlock::new(block, block_peer));

                warn!("Leaf data for block {hash} is invalid, banning peer {utreexo_peer}");
                Some(utreexo_peer)
            }

            // Every other error kind is either the block peer's fault or ours.
            _ => self.blame_block_peer_for_block_error(e, block_peer, hash, invalidate_chain),
        }
    }

    /// Finds whether the peer that sent us the block is responsible for a block validation
    /// error:
    ///
    /// - Consensus-invalid blocks are invalidated in our chain, but no peer is banned.
    /// - Mutated blocks are blamed on the block peer, without invalidating the block.
    /// - A block that doesn't extend our tip is our mistake, so no peer is blamed.
    ///
    /// Utreexo errors are attributed by the caller ([`Self::blame_peer_for_block_error`]).
    /// This match is exhaustive, so a new error variant must be classified here.
    ///
    /// `invalidate_chain`: when `true` and the block is consensus-invalid, it is invalidated in
    /// our chain with [`UpdatableChainstate::invalidate_block`]. When `false`, the chain is left
    /// untouched.
    ///
    /// Returns the peer id to blame, if any.
    ///
    /// [`UpdatableChainstate::invalidate_block`]: floresta_chain::pruned_utreexo::UpdatableChainstate::invalidate_block
    fn blame_block_peer_for_block_error(
        &mut self,
        e: &BlockValidationErrors,
        block_peer: PeerId,
        hash: BlockHash,
        invalidate_chain: bool,
    ) -> Option<PeerId> {
        match e {
            // The block is invalid, so we have to invalidate it in our chain.
            BlockValidationErrors::InvalidCoinbase(_)
            | BlockValidationErrors::ScriptValidationError(_)
            | BlockValidationErrors::NullPrevOut
            | BlockValidationErrors::DuplicateInput
            | BlockValidationErrors::EmptyInputs
            | BlockValidationErrors::EmptyOutputs
            | BlockValidationErrors::ScriptError
            | BlockValidationErrors::BlockTooBig
            | BlockValidationErrors::TooManySigops
            | BlockValidationErrors::NotEnoughPow
            | BlockValidationErrors::BadDiffBits
            | BlockValidationErrors::BadVersion
            | BlockValidationErrors::TooManyCoins
            | BlockValidationErrors::NotEnoughMoney
            | BlockValidationErrors::FirstTxIsNotCoinbase
            | BlockValidationErrors::BadCoinbaseOutValue
            | BlockValidationErrors::EmptyBlock
            | BlockValidationErrors::BadBip34
            | BlockValidationErrors::BIP94TimeWarp
            | BlockValidationErrors::UnspendableUTXO
            | BlockValidationErrors::NonFinalTransaction
            | BlockValidationErrors::InvalidSignetBlockSolution
            | BlockValidationErrors::CoinbaseNotMatured
            | BlockValidationErrors::UtxoNotFound(_)
            | BlockValidationErrors::BadRelativeLockTime => {
                if invalidate_chain {
                    try_and_log!(self.chain.invalidate_block(hash));
                }

                warn!("Block {hash} is invalid");
                None
            }

            // This block's txdata doesn't match the txid or wtxid merkle root. This can be a
            // mutated block, so we can't invalidate it since the original txdata may be valid.
            BlockValidationErrors::BadMerkleRoot | BlockValidationErrors::BadWitnessCommitment => {
                Some(block_peer)
            }

            // We've tried to connect a block that doesn't extend the tip.
            BlockValidationErrors::BlockExtendsAnOrphanChain
            | BlockValidationErrors::BlockDoesntExtendTip => {
                self.last_block_request = self.chain.get_validation_index().unwrap_or(0);

                // This is our mistake, don't punish any peer
                None
            }

            // Routed to the proof peer by `blame_peer_for_block_error`; kept so this match
            // stays exhaustive.
            BlockValidationErrors::InvalidUtreexoProof => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::Txid;
    use bitcoin::hashes::Hash;
    use floresta_chain::CompactLeafData;
    use floresta_chain::ScriptPubKeyKind;
    use floresta_chain::TransactionError;
    use floresta_chain::extensions::BlockExt;
    use floresta_chain::proof_util::LeafErrorKind;
    use floresta_chain::proof_util::UtreexoLeafError;
    use floresta_common::Ema;
    use floresta_mempool::Mempool;
    use rustreexo::stump::StumpError;
    use tokio::sync::Mutex;
    use tokio::sync::RwLock;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::unbounded_channel;

    use super::*;
    use crate::UtreexoNodeConfig;
    use crate::address_man::AddressMan;
    use crate::node::ConnectionKind;
    use crate::node::LocalPeerView;
    use crate::node::PeerStatus;
    use crate::node::sync_ctx::SyncNode;
    use crate::p2p_wire::tests::utils::BlockDefect;
    use crate::p2p_wire::tests::utils::mock_chain::MockChain;
    use crate::p2p_wire::tests::utils::synthetic_block;
    use crate::p2p_wire::transport::TransportProtocol;
    /// The node under test: a [`UtreexoNode`] running on top of a [`MockChain`].
    type MockNode = UtreexoNode<MockChain, SyncNode>;

    /// The peer that sent us the block under test.
    const BLOCK_PEER: PeerId = 0;

    /// The peer that sent us the utreexo proof and leaf data for the block under test.
    const UTREEXO_PEER: PeerId = 1;

    /// Everything a test needs: the node under test, its mock chain and the peer
    /// channel receivers. The receivers must be kept alive for the whole test,
    /// otherwise `disconnect_and_ban` fails while sending `Shutdown` to a peer.
    struct TestSetup {
        node: MockNode,
        chain: MockChain,
        _peer_rxs: Vec<UnboundedReceiver<NodeRequest>>,
    }

    /// Creates a node backed by a [`MockChain`] and two ready peers, with no
    /// requests in flight. The chain reports `validation_index` as its
    /// validation index.
    fn setup_test_with_validation_index(validation_index: u32) -> TestSetup {
        let chain = MockChain::with_validation_index(validation_index);

        let config = UtreexoNodeConfig {
            network: Network::Regtest,
            pow_fraud_proofs: false,
            datadir: PathBuf::from("./tmp-db/unit-test"),
            user_agent: "unit_test".to_string(),
            ..Default::default()
        };

        let mempool = Arc::new(Mutex::new(Mempool::new(1000)));
        let kill_signal = Arc::new(RwLock::new(false));
        let mut node: MockNode = UtreexoNode::new(
            config,
            chain.clone(),
            mempool,
            None,
            kill_signal,
            AddressMan::new(None, &[]),
        )
        .expect("building a node over a mock chain cannot fail");

        let mut peer_rxs = Vec::new();
        for peer_id in [BLOCK_PEER, UTREEXO_PEER] {
            let (tx, rx) = unbounded_channel();
            peer_rxs.push(rx);

            node.peers.insert(
                peer_id,
                LocalPeerView {
                    message_times: Ema::with_half_life_50(),
                    address: "127.0.0.1:8333".parse().expect("valid address"),
                    services: ServiceFlags::NETWORK,
                    user_agent: "unit_test".to_string(),
                    height: 0,
                    time_offset: 0,
                    state: PeerStatus::Ready,
                    channel: tx,
                    kind: ConnectionKind::OutboundFullRelay(ServiceFlags::NETWORK),
                    banscore: 0,
                    _last_message: Instant::now(),
                    transport_protocol: TransportProtocol::V2,
                },
            );
        }

        TestSetup {
            node,
            chain,
            _peer_rxs: peer_rxs,
        }
    }

    /// A coinbase-only block: it passes the merkle-root and witness-commitment
    /// checks, and counts as ready to process when wrapped in an [`InflightBlock`].
    fn dummy_block() -> Arc<Block> {
        Arc::new(synthetic_block(BlockDefect::None))
    }

    fn assert_peer_state(node: &MockNode, peer: PeerId, state: PeerStatus) {
        assert_eq!(node.peers.get(&peer).expect("peer exists").state, state);
    }

    /// Errors that make the block consensus-invalid: the block is invalidated, no peer banned.
    fn invalid_block_errors() -> Vec<BlockValidationErrors> {
        vec![
            BlockValidationErrors::InvalidCoinbase("bad coinbase".into()),
            BlockValidationErrors::ScriptValidationError("script error".into()),
            BlockValidationErrors::NullPrevOut,
            BlockValidationErrors::DuplicateInput,
            BlockValidationErrors::EmptyInputs,
            BlockValidationErrors::EmptyOutputs,
            BlockValidationErrors::ScriptError,
            BlockValidationErrors::BlockTooBig,
            BlockValidationErrors::TooManySigops,
            BlockValidationErrors::NotEnoughPow,
            BlockValidationErrors::BadDiffBits,
            BlockValidationErrors::TooManyCoins,
            BlockValidationErrors::NotEnoughMoney,
            BlockValidationErrors::FirstTxIsNotCoinbase,
            BlockValidationErrors::BadCoinbaseOutValue,
            BlockValidationErrors::EmptyBlock,
            BlockValidationErrors::BadBip34,
            BlockValidationErrors::BIP94TimeWarp,
            BlockValidationErrors::UnspendableUTXO,
            BlockValidationErrors::NonFinalTransaction,
            BlockValidationErrors::CoinbaseNotMatured,
            BlockValidationErrors::BadRelativeLockTime,
        ]
    }

    /// Errors that may come from a mutated block: the block peer is to blame, but
    /// the block isn't invalidated, since the original (unmutated) one may be valid.
    fn mutated_block_errors() -> Vec<BlockValidationErrors> {
        vec![
            BlockValidationErrors::BadMerkleRoot,
            BlockValidationErrors::BadWitnessCommitment,
        ]
    }

    /// Errors that mean *we* tried to connect a block in the wrong place: no peer
    /// is to blame, and we go back to asking blocks from our validation index.
    fn orphan_chain_errors() -> Vec<BlockValidationErrors> {
        vec![
            BlockValidationErrors::BlockExtendsAnOrphanChain,
            BlockValidationErrors::BlockDoesntExtendTip,
        ]
    }

    /// Utreexo errors that only the peer that sent the proof can cause.
    ///
    /// `UtxoNotFound` isn't listed: it may also be a same-block spend, so it has its own tests.
    fn utreexo_errors() -> Vec<BlockValidationErrors> {
        vec![BlockValidationErrors::InvalidUtreexoProof]
    }

    /// Expected outcome of [`UtreexoNode::blame_peer_for_block_error`] for each error kind.
    #[derive(Debug, PartialEq, Eq)]
    enum ExpectedBlame {
        /// Consensus-invalid block: invalidate it, without blaming any peer.
        InvalidatesBlock,

        /// Mutated block: blame the block peer, without invalidating the block.
        BlamesBlockPeer,

        /// Error caused by the proof peer: blame it and keep the block for another proof.
        BlamesUtreexoPeer,

        /// Block connected in the wrong place: no blame, restart from our validation index.
        RestartsFromValidationIndex,
    }

    /// The expected blame consensus for every `BlockValidationErrors` variant.
    ///
    /// Exhaustive on purpose: a new variant fails to compile until it's classified here.
    fn expected_blame(e: &BlockValidationErrors, block: &Block) -> ExpectedBlame {
        use BlockValidationErrors::*;

        match e {
            InvalidCoinbase(_)
            | InvalidSignetBlockSolution
            | ScriptValidationError(_)
            | NullPrevOut
            | DuplicateInput
            | EmptyInputs
            | EmptyOutputs
            | ScriptError
            | BlockTooBig
            | TooManySigops
            | NotEnoughPow
            | BadDiffBits
            | TooManyCoins
            | NotEnoughMoney
            | FirstTxIsNotCoinbase
            | BadCoinbaseOutValue
            | EmptyBlock
            | BadBip34
            | BIP94TimeWarp
            | UnspendableUTXO
            | NonFinalTransaction
            | CoinbaseNotMatured
            | BadVersion
            | BadRelativeLockTime => ExpectedBlame::InvalidatesBlock,

            // An output spent before being created in the same block is a consensus
            // violation. Any other missing output is the utreexo peer's fault.
            UtxoNotFound(outpoint) => {
                if block.is_invalid_missing_input(*outpoint) {
                    ExpectedBlame::InvalidatesBlock
                } else {
                    ExpectedBlame::BlamesUtreexoPeer
                }
            }

            BadMerkleRoot | BadWitnessCommitment => ExpectedBlame::BlamesBlockPeer,

            BlockExtendsAnOrphanChain | BlockDoesntExtendTip => {
                ExpectedBlame::RestartsFromValidationIndex
            }

            InvalidUtreexoProof => ExpectedBlame::BlamesUtreexoPeer,
        }
    }

    #[test]
    fn test_blame_block_peer_for_block_error_invalid_block_errors_invalidate_block() {
        let hash = dummy_block().block_hash();

        for error in invalid_block_errors() {
            let mut setup = setup_test_with_validation_index(0);
            let blamed = setup
                .node
                .blame_block_peer_for_block_error(&error, BLOCK_PEER, hash, true);

            // Consensus-invalid blocks are invalidated, but no peer is banned for them
            assert_eq!(blamed, None, "blame consensus for {error:?}");
            assert_eq!(
                setup.chain.invalidated_blocks(),
                vec![hash],
                "consensus-invalid blocks must be invalidated, case: {error:?}"
            );
        }
    }

    #[test]
    fn test_blame_block_peer_for_block_error_skips_invalidation_when_disabled() {
        let hash = dummy_block().block_hash();
        let mut setup = setup_test_with_validation_index(0);

        let blamed = setup.node.blame_block_peer_for_block_error(
            &BlockValidationErrors::BlockTooBig,
            BLOCK_PEER,
            hash,
            false,
        );

        // Still no peer is blamed, and the chain is not invalidated when `invalidate_chain` is off
        assert_eq!(blamed, None);
        assert!(setup.chain.invalidated_blocks().is_empty());
    }

    #[test]
    fn test_blame_block_peer_for_block_error_mutated_block_errors_blame_block_peer() {
        let hash = dummy_block().block_hash();

        for error in mutated_block_errors() {
            let mut setup = setup_test_with_validation_index(0);
            let blamed = setup
                .node
                .blame_block_peer_for_block_error(&error, BLOCK_PEER, hash, true);

            assert_eq!(blamed, Some(BLOCK_PEER), "blame consensus for {error:?}");

            // The unmutated block may still be valid, so we can't invalidate it
            assert!(
                setup.chain.invalidated_blocks().is_empty(),
                "mutated blocks must not be invalidated, case: {error:?}"
            );
        }
    }

    #[test]
    fn test_blame_block_peer_for_block_error_orphan_chain_errors_blame_no_peer() {
        let hash = dummy_block().block_hash();

        for error in orphan_chain_errors() {
            let mut setup = setup_test_with_validation_index(7);
            setup.node.last_block_request = 100;

            let blamed = setup
                .node
                .blame_block_peer_for_block_error(&error, BLOCK_PEER, hash, true);

            // This is our own mistake, don't punish any peer
            assert_eq!(blamed, None, "blame consensus for {error:?}");

            // We go back to asking for blocks from our validation index
            assert_eq!(setup.node.last_block_request, 7, "case: {error:?}");
            assert!(
                setup.chain.invalidated_blocks().is_empty(),
                "case: {error:?}"
            );
        }
    }

    #[test]
    fn test_blame_block_peer_for_block_error_utreexo_errors_blame_no_peer() {
        let hash = dummy_block().block_hash();

        for error in utreexo_errors() {
            let mut setup = setup_test_with_validation_index(0);
            let blamed = setup
                .node
                .blame_block_peer_for_block_error(&error, BLOCK_PEER, hash, true);

            // Utreexo errors are the utreexo peer's fault, not the block peer's
            assert_eq!(blamed, None, "blame consensus for {error:?}");
            assert!(
                setup.chain.invalidated_blocks().is_empty(),
                "case: {error:?}"
            );
        }
    }

    /// One instance of every `BlockValidationErrors` variant.
    fn all_block_errors(block: &Block) -> Vec<BlockValidationErrors> {
        let mut all = invalid_block_errors();
        all.extend(mutated_block_errors());
        all.extend(orphan_chain_errors());
        all.extend(utreexo_errors());

        // `UtxoNotFound` blames the proof peer, unless the output is from this same block.
        all.push(BlockValidationErrors::UtxoNotFound(OutPoint::null()));
        all.push(BlockValidationErrors::UtxoNotFound(OutPoint {
            txid: block.txdata[0].compute_txid(),
            vout: 0,
        }));

        all
    }

    #[test]
    fn test_blame_peer_for_block_error_covers_all_variants() {
        let block = dummy_block();
        let hash = block.block_hash();

        // `expected_blame` is exhaustive; here we check the node follows it.
        for error in all_block_errors(&block) {
            let mut setup = setup_test_with_validation_index(0);
            let blamed = setup.node.blame_peer_for_block_error(
                &error,
                block.clone(),
                BLOCK_PEER,
                UTREEXO_PEER,
                true,
            );

            let (expected_blamed, expected_invalidation, expected_kept) =
                match expected_blame(&error, &block) {
                    ExpectedBlame::InvalidatesBlock => (None, true, false),
                    ExpectedBlame::BlamesBlockPeer => (Some(BLOCK_PEER), false, false),
                    ExpectedBlame::BlamesUtreexoPeer => (Some(UTREEXO_PEER), false, true),
                    ExpectedBlame::RestartsFromValidationIndex => (None, false, false),
                };

            assert_eq!(blamed, expected_blamed, "blame consensus for {error:?}");
            assert_eq!(
                !setup.chain.invalidated_blocks().is_empty(),
                expected_invalidation,
                "invalidation consensus for {error:?}"
            );
            assert_eq!(
                setup.node.blocks.contains_key(&hash),
                expected_kept,
                "block kept for another proof, case: {error:?}"
            );
        }
    }

    #[test]
    fn test_blame_peer_for_block_error_block_error_blames_block_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let blamed = setup.node.blame_peer_for_block_error(
            &BlockValidationErrors::BadMerkleRoot,
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
            true,
        );

        // Block errors are delegated to the block peer's blame consensus
        assert_eq!(blamed, Some(BLOCK_PEER));

        // The block peer is to blame, so the block isn't kept around for a retry
        assert!(!setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_blame_peer_for_block_error_invalid_utreexo_proof_blames_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let blamed = setup.node.blame_peer_for_block_error(
            &BlockValidationErrors::InvalidUtreexoProof,
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
            true,
        );

        assert_eq!(blamed, Some(UTREEXO_PEER));

        // An invalid proof doesn't make the block invalid, so we must not invalidate it
        assert!(setup.chain.invalidated_blocks().is_empty());

        // The block is put back in the pending map, so another peer can prove it
        let inflight = setup.node.blocks.get(&hash).expect("block re-inserted");
        assert_eq!(inflight.peer, BLOCK_PEER);
    }

    #[test]
    fn test_blame_peer_for_block_error_utxo_not_found_blames_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let blamed = setup.node.blame_peer_for_block_error(
            &BlockValidationErrors::UtxoNotFound(OutPoint::null()),
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
            true,
        );

        assert_eq!(blamed, Some(UTREEXO_PEER));

        // Incomplete leaf data blames the proof peer, never the block.
        assert!(
            setup.chain.invalidated_blocks().is_empty(),
            "incomplete leaf data must not invalidate the block"
        );

        let inflight = setup.node.blocks.get(&hash).expect("block re-inserted");
        assert_eq!(inflight.peer, BLOCK_PEER);
    }

    #[test]
    fn test_blame_peer_for_block_error_same_block_spend_invalidates_block() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        // An output spent before being created in the same block is a consensus violation.
        let outpoint = OutPoint {
            txid: block.txdata[0].compute_txid(),
            vout: 0,
        };

        let blamed = setup.node.blame_peer_for_block_error(
            &BlockValidationErrors::UtxoNotFound(outpoint),
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
            true,
        );

        assert_eq!(blamed, None);
        assert_eq!(setup.chain.invalidated_blocks(), vec![hash]);

        // The block is invalid, so it isn't kept around for a retry
        assert!(!setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_handle_block_error_transaction_error_invalidates_block() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let chain_err = BlockchainError::TransactionError(TransactionError {
            txid: Txid::all_zeros(),
            error: BlockValidationErrors::InvalidCoinbase("bad coinbase".into()),
        });

        let result = setup
            .node
            .handle_block_error(chain_err, block, BLOCK_PEER, None, true);

        // The tx error is unwrapped into the block error, which only invalidates the block.
        assert!(result.is_ok());
        assert_eq!(setup.chain.invalidated_blocks(), vec![hash]);
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
    }

    #[test]
    fn test_handle_block_error_block_validation_blames_block_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();

        let chain_err = BlockchainError::BlockValidation(BlockValidationErrors::BadMerkleRoot);
        let result = setup
            .node
            .handle_block_error(chain_err, block, BLOCK_PEER, None, true);

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
    }

    #[test]
    fn test_handle_block_error_invalid_utreexo_proof_blames_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let result = setup.node.handle_block_error(
            BlockchainError::InvalidUtreexoProof,
            block,
            BLOCK_PEER,
            Some(UTREEXO_PEER),
            true,
        );

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);

        // The block is put back in the pending map, so another peer can prove it
        assert!(setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_handle_block_error_accumulator_error_blames_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let chain_err =
            BlockchainError::AccumulatorError(StumpError::Io(std::io::ErrorKind::Other));
        let result =
            setup
                .node
                .handle_block_error(chain_err, block, BLOCK_PEER, Some(UTREEXO_PEER), true);

        // Accumulator errors are mapped to an invalid utreexo proof
        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
        assert!(setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_handle_block_error_utreexo_leaf_error_blames_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let leaf_error = UtreexoLeafError {
            leaf: CompactLeafData {
                header_code: 0,
                amount: 0,
                spk_ty: ScriptPubKeyKind::Other(vec![].into()),
            },
            txid: Txid::all_zeros(),
            vin: 0,
            kind: LeafErrorKind::EmptyStack,
        };

        let result = setup.node.handle_block_error(
            BlockchainError::UtreexoLeaf(leaf_error),
            block,
            BLOCK_PEER,
            Some(UTREEXO_PEER),
            true,
        );

        // Leaf reconstruction errors are mapped to an invalid utreexo proof
        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
        assert!(setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_handle_block_error_non_validation_error_returns_ok() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();

        // Local failures (e.g. database errors) are not blamed on any peer
        let result = setup.node.handle_block_error(
            BlockchainError::BlockNotPresent,
            block,
            BLOCK_PEER,
            Some(UTREEXO_PEER),
            true,
        );

        assert!(result.is_ok());
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
    }

    #[test]
    fn test_handle_block_error_no_blamed_peer_returns_ok() {
        let block = dummy_block();

        // With and without a utreexo peer involved, the outcome is the same:
        // the error is our own fault (the block doesn't extend our tip), so
        // no peer is punished and we go back to our validation index.
        for utreexo_peer in [Some(UTREEXO_PEER), None] {
            let mut setup = setup_test_with_validation_index(7);
            setup.node.last_block_request = 100;

            let result = setup.node.handle_block_error(
                BlockchainError::BlockValidation(BlockValidationErrors::BlockDoesntExtendTip),
                block.clone(),
                BLOCK_PEER,
                utreexo_peer,
                true,
            );

            assert!(result.is_ok(), "case utreexo_peer={utreexo_peer:?}");
            assert_eq!(
                setup.node.last_block_request, 7,
                "case utreexo_peer={utreexo_peer:?}"
            );
            assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
            assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
        }
    }

    #[test]
    fn test_handle_process_block_error_non_blockchain_error_is_propagated() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();

        let result = setup.node.handle_process_block_error(
            WireError::BlockNotFound,
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
        );

        // Non-blockchain errors are returned unchanged, with no peer punished
        assert!(matches!(result, Err(WireError::BlockNotFound)));
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
    }

    #[test]
    fn test_handle_process_block_error_blockchain_utreexo_error_bans_utreexo_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();
        let hash = block.block_hash();

        let result = setup.node.handle_process_block_error(
            WireError::Blockchain(BlockchainError::InvalidUtreexoProof),
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
        );

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);

        // The block is put back in the pending map, so another peer can prove it
        assert!(setup.node.blocks.contains_key(&hash));
    }

    #[test]
    fn test_handle_process_block_error_blockchain_block_error_bans_block_peer() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();

        let result = setup.node.handle_process_block_error(
            WireError::Blockchain(BlockchainError::BlockValidation(
                BlockValidationErrors::BadMerkleRoot,
            )),
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
        );

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Banned);
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
    }

    #[test]
    fn test_handle_process_block_error_blockchain_no_blame_returns_ok() {
        let mut setup = setup_test_with_validation_index(0);
        let block = dummy_block();

        let result = setup.node.handle_process_block_error(
            WireError::Blockchain(BlockchainError::BlockValidation(
                BlockValidationErrors::BlockDoesntExtendTip,
            )),
            block,
            BLOCK_PEER,
            UTREEXO_PEER,
        );

        // Nobody's fault: no peer is punished
        assert!(result.is_ok());
        assert_peer_state(&setup.node, BLOCK_PEER, PeerStatus::Ready);
        assert_peer_state(&setup.node, UTREEXO_PEER, PeerStatus::Ready);
    }
}
