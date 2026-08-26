// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::ChainBackend;
use floresta_chain::CompactLeafData;
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
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
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
    pub block: Block,

    /// Auxiliary data needed for validating this block. Currently, it includes utreexo
    /// leaf data (previous UTXOs spent in the block), the corresponding accumulator
    /// inclusion proof, and the peer id that provided them.
    pub aux_data: Option<UtreexoData>,
}

impl InflightBlock {
    /// Creates a new `InflightBlock` from a block and the associated peer id.
    ///
    /// If the block doesn't spend any output (i.e., coinbase transaction only) this method adds
    /// empty auxiliary data, which marks this inflight block as ready to process. Blocks with
    /// transactions require [`UtreexoData`] (see [`InflightBlock::add_utreexo_data`]).
    fn new(block: Block, peer: PeerId) -> Self {
        let aux_data = match block.txdata.len() {
            1 => Some((Vec::new(), Proof::default(), peer)),
            _ => None, // we need auxiliary data for the txs
        };

        Self {
            peer,
            block,
            aux_data,
        }
    }

    /// Attaches the auxiliary utreexo data to this `InflightBlock`.
    fn add_utreexo_data(&mut self, leaf_data: Vec<CompactLeafData>, proof: Proof, peer: PeerId) {
        self.aux_data = Some((leaf_data, proof, peer));
    }
}

impl<T, Chain> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
    Chain: ChainBackend + 'static,
    WireError: From<Chain::Error>,
{
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

        let peer =
            self.send_to_fast_peer(NodeRequest::GetBlock(blocks.clone()), ServiceFlags::NETWORK)?;

        for block in blocks.iter() {
            self.inflight
                .insert(InflightRequests::Blocks(*block), (peer, Instant::now()));
        }

        Ok(())
    }

    pub(crate) fn request_block_proof(
        &mut self,
        block: Block,
        peer: PeerId,
    ) -> Result<(), WireError> {
        let block_hash = block.block_hash();
        self.inflight.remove(&InflightRequests::Blocks(block_hash));

        // Reply and return early if it's a user-requested block. Else continue handling it.
        let Some(block) = self.check_is_user_block_and_reply(block)? else {
            return Ok(());
        };

        let txdata_len = block.txdata.len();
        debug!("Received block {block_hash} from peer {peer}, with {txdata_len} txs");

        self.blocks
            .insert(block_hash, InflightBlock::new(block, peer));

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
                use metrics::get_metrics;

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

        let (del_hashes, inputs) =
            proof_util::process_proof(&leaf_data, &block.txdata, block_height, |h| {
                self.chain.get_block_hash(h)
            })?;

        if let Err(chain_err) = self.chain.connect_block(&block, proof, inputs, del_hashes) {
            error!(
                "Validation failed for block with {:?}, received by peer {peer}. Reason: {chain_err}",
                block.header,
            );

            // Return early if the error is not from block validation (e.g., a database error)
            let Some(e) = Self::block_validation_err(chain_err) else {
                return Ok(());
            };

            return match self.handle_validation_errors(e, block, peer, utreexo_peer) {
                // Disconnect the responsible peer and ban it.
                Some(blamed_peer) => {
                    self.disconnect_and_ban(blamed_peer)?;
                    Err(WireError::PeerMisbehaving)
                }
                None => Ok(()),
            };
        }

        self.last_tip_update = Instant::now();
        Ok(())
    }

    /// Returns the inner [`BlockValidationErrors`] of this chain error, if any.
    fn block_validation_err(e: BlockchainError) -> Option<BlockValidationErrors> {
        match e {
            BlockchainError::TransactionError(tx_err) => Some(tx_err.error),
            BlockchainError::BlockValidation(block_err) => Some(block_err),
            // TODO: we need clearer error definitions for utreexo failures
            BlockchainError::AccumulatorError(_) | BlockchainError::InvalidUtreexoProof => {
                Some(BlockValidationErrors::InvalidUtreexoProof)
            }
            _ => None,
        }
    }

    /// Handles the different block validation errors that can happen when connecting a block.
    ///
    /// Returns the peer id that caused this error, since it could be block or utreexo-related.
    fn handle_validation_errors(
        &mut self,
        e: BlockValidationErrors,
        block: Block,
        block_peer: PeerId,
        utreexo_peer: PeerId,
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

            // The utreexo peer sent us incomplete leaf data. Block is not yet processed.
            BlockValidationErrors::UtxoNotFound(_) => {
                self.blocks
                    .insert(hash, InflightBlock::new(block, block_peer));

                warn!("Leaf data for block {hash} is invalid, banning peer {utreexo_peer}");
                Some(utreexo_peer)
            }

            // The block is invalid, so we have to invalidate it in our chain.
            BlockValidationErrors::ContextFreeValidation(_)
            | BlockValidationErrors::InvalidCoinbase(_)
            | BlockValidationErrors::ScriptValidationError(_)
            | BlockValidationErrors::NullPrevOut
            | BlockValidationErrors::DuplicateInput
            | BlockValidationErrors::EmptyInputs
            | BlockValidationErrors::EmptyOutputs
            | BlockValidationErrors::ScriptError
            | BlockValidationErrors::BlockTooBig
            | BlockValidationErrors::NotEnoughPow
            | BlockValidationErrors::TooManyCoins
            | BlockValidationErrors::NotEnoughMoney
            | BlockValidationErrors::FirstTxIsNotCoinbase
            | BlockValidationErrors::BadCoinbaseOutValue
            | BlockValidationErrors::EmptyBlock
            | BlockValidationErrors::BadBip34
            | BlockValidationErrors::BIP94TimeWarp
            | BlockValidationErrors::UnspendableUTXO
            | BlockValidationErrors::NonFinalTransaction
            | BlockValidationErrors::CoinbaseNotMatured => {
                try_and_log!(self.chain.invalidate_block(hash));

                warn!("Block {hash} is invalid, banning peer {block_peer}");
                Some(block_peer)
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
        }
    }
}
