// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module is centered around the `ChainState` type, defining it and providing
//! implementations for the [BlockchainInterface] and [UpdatableChainstate] traits.
//!
//! Consequently, `ChainState` serves as the blockchain backend for our node and is
//! the highest-level type in `floresta-chain`. It is responsible for:
//!
//! - Keeping track of the chain state, and using a [ChainStore] for persisted storage
//! - Correctly updating the state with the help of the `consensus.rs` functions
//! - Interfacing with other components, and providing data about the current view of the chain
//!
//! The primary methods for updating our state are [ChainState::accept_header], which constructs
//! a chain of headers, and [ChainState::connect_block], which verifies the corresponding blocks.
//!
//! Key definitions:
//! - [ChainState]: The high-level chain backend
//! - [BlockConsumer]: Trait for receiving new block notifications

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::fmt::format;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::cmp::min;
use core::ops::Add;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::Target;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::Work;
use bitcoin::block::Header as BlockHeader;
use bitcoin::blockdata::constants::DIFFCHANGE_INTERVAL;
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use floresta_common::Channel;
#[cfg(feature = "metrics")]
use floresta_metrics as metrics;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::proof::Proof;
use rustreexo::stump::Stump;
use spin::RwLock;
use tracing::debug;
use tracing::info;
use tracing::warn;

use super::BlockchainInterface;
use super::UpdatableChainstate;
use super::chain_state_builder::BlockchainBuilderError;
use super::chain_state_builder::ChainStateBuilder;
use super::chainparams::ChainParams;
use super::chainstore::ChainStoreWarning;
use super::chainstore::DiskBlockHeader;
use super::consensus::Consensus;
use super::error::BlockValidationErrors;
use super::error::BlockchainError;
use super::partial_chain::PartialChainState;
use super::partial_chain::PartialChainStateInner;
use crate::BestChain;
use crate::ChainStore;
use crate::extensions::HeaderExt;
use crate::extensions::WorkExt;
use crate::prelude::*;
use crate::pruned_utreexo::IBDState;
use crate::pruned_utreexo::utxo_data::UtxoData;
use crate::read_lock;
use crate::write_lock;

/// Trait for components that need to receive notifications about new blocks.
pub trait BlockConsumer: Sync + Send + 'static {
    /// Return true if this consumer wants the set of spent UTXOs.
    fn wants_spent_utxos(&self) -> bool;

    /// Called whenever a valid block is connected. `spent_utxos` is None unless consumer `wants_spent_utxos()`.
    fn on_block(
        &self,
        block: &Block,
        height: u32,
        spent_utxos: Option<&HashMap<OutPoint, UtxoData>>,
    );
}

impl BlockConsumer for Channel<(Block, u32)> {
    fn wants_spent_utxos(&self) -> bool {
        false
    }

    fn on_block(
        &self,
        block: &Block,
        height: u32,
        _spent_utxos: Option<&HashMap<OutPoint, UtxoData>>,
    ) {
        self.send((block.to_owned(), height));
    }
}

impl BlockConsumer for Channel<(Block, u32, HashMap<OutPoint, UtxoData>)> {
    fn wants_spent_utxos(&self) -> bool {
        true
    }

    fn on_block(
        &self,
        block: &Block,
        height: u32,
        spent_utxos: Option<&HashMap<OutPoint, UtxoData>>,
    ) {
        let spent_utxos = spent_utxos.expect("Safe to unwrap because wants_spent_utxos()");
        self.send((block.to_owned(), height, spent_utxos.to_owned()));
    }
}

/// Internal state of the blockchain managed by `ChainState`.
pub struct ChainStateInner<PersistedState: ChainStore> {
    /// The acc we use for validation.
    acc: Stump,
    /// All data is persisted here.
    chainstore: PersistedState,
    /// Best known block, cached in a specific field to faster access.
    best_block: BestChain,
    /// We may have multiple modules that needs to receive and process blocks as they come, to
    /// be notified of new blocks, a module should implement the [BlockConsumer] trait, and
    /// subscribe by passing an [Arc] of itself to chainstate.
    /// When a new block is accepted (as valid) we call `on_block` from [BlockConsumer].
    /// If a module just wants pass in a channel, `Sender` implements [BlockConsumer], and can
    /// be used during subscription (just keep the `Receiver` side.
    subscribers: Vec<Arc<dyn BlockConsumer>>,
    /// Fee estimation for 1, 10 and 20 blocks
    fee_estimation: (f64, f64, f64),
    /// What is our current IBD state?
    ibd: IBDState,
    /// Parameters for the chain and functions that verify the chain.
    consensus: Consensus,
    /// Assume valid is a Core-specific config that tells the node to not validate signatures
    /// in blocks before this one. Note that we only skip signature validation, everything else
    /// is still validated.
    assume_valid: Option<BlockHash>,
}

/// The high-level chain backend managing the blockchain state.
///
/// `ChainState` is responsible for:
/// - Keeping track of the chain state with the help of a `ChainStore` for persisted storage.
/// - Correctly updating the state using consensus functions.
/// - Interfacing with other components and providing data about the current view of the chain.
pub struct ChainState<PersistedState: ChainStore> {
    inner: RwLock<ChainStateInner<PersistedState>>,
}

#[derive(Debug, Copy, Clone)]
/// Represents the argument for the assume-valid configuration.
///
/// This enum indicates the state of the assume-valid configuration,
/// which defines whether we should validate the scripts for blocks before this one.
/// You can either disable it, use a value provided by floresta or use your own value.
pub enum AssumeValidArg {
    /// Do not assume any script are valid, check every single one from genesis. This should make IBD considerably slower, but in theory has the best security model
    Disabled,
    /// Use the hard-coded value provided by Floresta. In this case, you trust that the Floresta repository faces enough scrutiny and review, and therefore the value can be trusted.
    Hardcoded,
    /// Provide your own value, moving the trust assumption to your program.
    UserInput(BlockHash),
}

impl<PersistedState: ChainStore> ChainState<PersistedState> {
    fn maybe_reindex(&self, potential_tip: &DiskBlockHeader) -> Result<(), BlockchainError> {
        if let DiskBlockHeader::HeadersOnly(_, height) = potential_tip {
            let best_height = self.get_best_block()?.0;

            if *height > best_height {
                warn!(
                    "Found block with height {height}, while the current best chain is at {best_height}. Reindexing."
                );
                self.reindex_chain()?;
            }
        }

        Ok(())
    }

    /// Just adds headers to the chainstate, without validating them.
    pub fn push_headers(
        &self,
        headers: Vec<BlockHeader>,
        height: u32,
    ) -> Result<(), BlockchainError> {
        for (offset, &header) in headers.iter().enumerate() {
            let disk_height = height + offset as u32;
            let disk_header = DiskBlockHeader::FullyValid(header, disk_height);
            let hash = disk_header.block_hash();

            self.update_header_and_index(&disk_header, hash, disk_height)?;
        }

        Ok(())
    }

    fn update_header(&self, header: &DiskBlockHeader) -> Result<(), BlockchainError> {
        Ok(write_lock!(self).chainstore.save_header(header)?)
    }

    fn update_header_and_index(
        &self,
        header: &DiskBlockHeader,
        hash: BlockHash,
        height: u32,
    ) -> Result<(), BlockchainError> {
        let mut inner = write_lock!(self);

        inner.chainstore.save_header(header)?;
        inner.chainstore.update_block_index(height, hash)?;

        Ok(())
    }

    fn validate_header(&self, block_header: &BlockHeader) -> Result<BlockHash, BlockchainError> {
        // As in Core, the header's own proof of work comes first: it is the cheap check, and the
        // ones below read the chain.
        let block_hash = block_header
            .validate_pow(block_header.target())
            .map_err(|_| BlockValidationErrors::NotEnoughPow)?;

        let prev_block = self.get_disk_block_header(&block_header.prev_blockhash)?;
        let height = prev_block
            .height()
            .ok_or(BlockValidationErrors::BlockExtendsAnOrphanChain)?
            + 1;

        // Like Core's `bad-diffbits`, the compact bits must match exactly: a harder target
        // than the required one is invalid too.
        let expected_target = self.get_next_required_work(&prev_block, height, block_header)?;
        if block_header.bits != expected_target.to_compact_lossy() {
            Err(BlockValidationErrors::BadDiffBits)?;
        }

        self.check_bip94_block(block_header, height)?;

        // Like Core's `bad-version`: old versions are rejected once their BIP activates
        let min_version =
            Consensus::min_header_version(&read_lock!(self).consensus.parameters.params, height);
        if block_header.version.to_consensus() < min_version {
            Err(BlockValidationErrors::BadVersion)?;
        }

        Ok(block_hash)
    }

    #[inline]
    /// Whether a node is the genesis block for this net
    fn is_genesis(&self, header: &BlockHeader) -> bool {
        header.block_hash() == self.chain_params().genesis.block_hash()
    }

    #[inline]
    /// Returns the ancestor of a given block
    fn get_ancestor(&self, header: &BlockHeader) -> Result<DiskBlockHeader, BlockchainError> {
        self.get_disk_block_header(&header.prev_blockhash)
    }

    /// Computes the work of a fork branch.
    ///
    /// This function should only be used for forks. It will make O(n) header lookups, since we
    /// can't use the index for fork blocks. Calling it on a big branch will cause it to use a
    /// lot of CPU.
    ///
    /// If you want to get the work for a main chain block, use `get_branch_work` instead.
    fn get_fork_work(
        &self,
        new_tip: BlockHeader,
        fork_point: BlockHash,
    ) -> Result<Work, BlockchainError> {
        let mut header = new_tip;
        let mut work = self.get_work(fork_point)?;

        while header.block_hash() != fork_point {
            work = work + header.work();
            header = *self.get_ancestor(&header)?;
        }

        Ok(work)
    }

    /// Returns the cumulative work in this branch.
    ///
    /// This function is more optimized CPU-wise. It will take (n/2016) header lookups to compute
    /// the work of a branch starting a `header`.
    ///
    /// It does not work with fork headers, see `get_fork_work` for that.
    fn get_branch_work(&self, header: BlockHeader) -> Result<Work, BlockchainError> {
        let block_height = self
            .get_block_height(&header.block_hash())?
            .ok_or(BlockchainError::BlockNotPresent)?;

        let mut total_chainwork = Work::from_be_bytes([0u8; 32]);
        for epoch_start_height in (0..=block_height).step_by(2016) {
            // Calculate the number of blocks in this epoch
            let epoch_end_height = min(epoch_start_height + 2015, block_height);
            let blocks_in_epoch = epoch_end_height - epoch_start_height + 1;

            // Get the block hash and header at the start of the epoch
            let epoch_block_hash = self.get_block_hash(epoch_start_height)?;

            let epoch_block_header = self.get_block_header(&epoch_block_hash)?;

            let epoch_chainwork = epoch_block_header
                .work()
                .multiply_work_by_u32(blocks_in_epoch)?;

            total_chainwork = total_chainwork.add(epoch_chainwork);
        }

        Ok(total_chainwork)
    }

    /// Checks if a branch is valid (i.e. all ancestors are known)
    fn check_branch(&self, branch_tip: &BlockHeader) -> Result<(), BlockchainError> {
        let mut header = *branch_tip;

        while !self.is_genesis(&header) {
            let _header = self.get_ancestor(&header)?;
            if let DiskBlockHeader::Orphan(block) = _header {
                return Err(BlockchainError::InvalidTip(format(format_args!(
                    "Block {} doesn't have a known ancestor (i.e an orphan block)",
                    block.block_hash()
                ))));
            }
            header = *_header;
        }

        Ok(())
    }

    /// Returns the depth of a branch (i.e. how many blocks are in the branch)
    fn get_chain_depth(&self, branch_tip: &BlockHeader) -> Result<u32, BlockchainError> {
        let mut header = *branch_tip;

        let mut counter = 0;
        while !self.is_genesis(&header) {
            header = *self.get_ancestor(&header)?;
            counter += 1;
        }

        Ok(counter)
    }

    /// Mark the current index as active, because we are in the middle of a reorg
    fn mark_chain_as_active(
        &self,
        new_tip: &BlockHeader,
        fork_point: BlockHash,
    ) -> Result<(), BlockchainError> {
        let mut height = self.get_chain_depth(new_tip)?;
        let mut header = DiskBlockHeader::HeadersOnly(*new_tip, height);

        while !self.is_genesis(&header) && header.block_hash() != fork_point {
            let disk_header = DiskBlockHeader::HeadersOnly(*header, height);
            let hash = disk_header.block_hash();

            self.update_header_and_index(&disk_header, hash, height)?;

            header = self.get_ancestor(&header)?;
            height -= 1;
        }

        Ok(())
    }

    /// Mark the current index as inactive, either because we found an invalid ancestor,
    /// or we are in the middle of reorg
    fn mark_chain_as_inactive(
        &self,
        old_tip: &BlockHeader,
        fork_point: BlockHash,
    ) -> Result<(), BlockchainError> {
        let mut header = *old_tip;
        let mut height = self.get_chain_depth(old_tip)?;

        while !self.is_genesis(&header) && header.block_hash() != fork_point {
            self.update_header(&DiskBlockHeader::InFork(header, height))?;

            header = *self.get_ancestor(&header)?;
            height -= 1;
        }

        Ok(())
    }

    /// Finds where in the current index, a given branch forks out.
    fn find_fork_point(&self, header: &BlockHeader) -> Result<BlockHeader, BlockchainError> {
        let mut header = *self.get_ancestor(header)?;
        let inner = read_lock!(self);
        while !self.is_genesis(&header) {
            match inner.chainstore.get_header(&header.block_hash())? {
                Some(DiskBlockHeader::HeadersOnly(block, _)) => {
                    return Ok(block);
                }
                Some(DiskBlockHeader::FullyValid(block, _)) => {
                    return Ok(block);
                }
                Some(DiskBlockHeader::InFork(block, _)) => {
                    header = *self.get_ancestor(&block)?;
                    continue;
                }
                Some(DiskBlockHeader::AssumedValid(block, _)) => {
                    return Ok(block);
                }
                Some(DiskBlockHeader::Orphan(header)) => {
                    return Err(BlockchainError::InvalidTip(format(format_args!(
                        "Block {} doesn't have a known ancestor (i.e an orphan block)",
                        header.block_hash()
                    ))));
                }
                Some(DiskBlockHeader::InvalidChain(header)) => {
                    return Err(BlockchainError::InvalidTip(format(format_args!(
                        "Block {} is invalid",
                        header.block_hash()
                    ))));
                }
                None => {
                    return Err(BlockchainError::InvalidTip(format(format_args!(
                        "Block {} isn't in our storage",
                        header.block_hash()
                    ))));
                }
            }
        }
        Err(BlockchainError::InvalidTip(
            "Couldn't find a fork point".to_string(),
        ))
    }

    /// Returns the acc we must validate from after reorging, given the new `validation_index`.
    fn reorg_acc(&self, validation_index: BlockHash) -> Result<Stump, BlockchainError> {
        let height = self
            .get_block_height(&validation_index)?
            .ok_or(BlockchainError::BlockNotPresent)?;

        // Genesis is the only block we take as valid without ever validating it, so it's the only
        // one that legitimately has no roots saved. Its accumulator is the empty one.
        if height == 0 {
            return Ok(Stump::new());
        }

        self.get_roots_for_block(height)?
            .ok_or(BlockchainError::BadValidationIndex)
    }

    // This method should only be called after we validate the new branch
    fn reorg(&self, new_tip: BlockHeader) -> Result<(), BlockchainError> {
        let current_best_block = self.get_block_header(&self.get_best_block()?.1)?;
        let fork_point = self.find_fork_point(&new_tip)?;

        let validation_index = self.get_last_valid_block(&new_tip)?;
        let depth = self.get_chain_depth(&new_tip)?;
        let acc = self.reorg_acc(validation_index)?;

        self.mark_chain_as_inactive(&current_best_block, fork_point.block_hash())?;
        self.mark_chain_as_active(&new_tip, fork_point.block_hash())?;

        self.change_active_chain(&new_tip, validation_index, depth, acc);

        Ok(())
    }

    /// Changes the active chain to the new branch during a reorg
    fn change_active_chain(
        &self,
        new_tip: &BlockHeader,
        last_valid: BlockHash,
        depth: u32,
        acc: Stump,
    ) {
        let mut inner = self.inner.write();
        inner.best_block.best_block = new_tip.block_hash();
        inner.best_block.validation_index = last_valid;
        inner.best_block.depth = depth;
        inner.acc = acc;
    }

    /// Grabs the last block we validated in this branch. We don't validate a fork, unless it
    /// becomes the best chain. This function technically finds out what is the last common block
    /// between two branches.
    fn get_last_valid_block(&self, header: &BlockHeader) -> Result<BlockHash, BlockchainError> {
        let mut header = *header;

        while !self.is_genesis(&header) {
            let _header = self.get_ancestor(&header)?;

            match _header {
                DiskBlockHeader::FullyValid(_, _) | DiskBlockHeader::AssumedValid(_, _) => {
                    return Ok(_header.block_hash());
                }
                DiskBlockHeader::Orphan(_) => {
                    return Err(BlockchainError::InvalidTip(format(format_args!(
                        "Block {} doesn't have a known ancestor (i.e an orphan block)",
                        _header.block_hash()
                    ))));
                }
                DiskBlockHeader::HeadersOnly(_, _) | DiskBlockHeader::InFork(_, _) => {}
                DiskBlockHeader::InvalidChain(_) => {
                    return Err(BlockchainError::InvalidTip(format(format_args!(
                        "Block {} is in an invalid chain",
                        _header.block_hash()
                    ))));
                }
            }

            header = *_header;
        }

        Ok(header.block_hash())
    }

    /// If we get a header that doesn't build on top of our best chain, it may cause a reorganization.
    /// We check this here.
    fn maybe_reorg(&self, branch_tip: BlockHeader) -> Result<(), BlockchainError> {
        let current_tip = self.get_block_header(&self.get_best_block()?.1)?;
        self.check_branch(&branch_tip)?;

        let fork_point = self.find_fork_point(&branch_tip)?;
        let current_work = self.get_branch_work(current_tip)?;
        let new_work = self.get_fork_work(branch_tip, fork_point.block_hash())?;

        // If the new branch has more work, it becomes the new best chain
        if new_work > current_work {
            self.reorg(branch_tip)?;
            return Ok(());
        }
        // If the new branch has less work, we just store it as an alternative branch
        // that might become the best chain in the future.
        self.push_alt_tip(&branch_tip)?;

        let parent_height = self.get_ancestor(&branch_tip)?.try_height()?;

        self.update_header(&DiskBlockHeader::InFork(branch_tip, parent_height + 1))?;

        Ok(())
    }

    /// Stores a new tip for a branch that is not the best one
    fn push_alt_tip(&self, branch_tip: &BlockHeader) -> Result<(), BlockchainError> {
        let ancestor = self.get_ancestor(branch_tip);
        let ancestor = match ancestor {
            Ok(ancestor) => Some(ancestor),
            Err(BlockchainError::BlockNotPresent) => None,
            Err(e) => return Err(e),
        };
        let mut inner = write_lock!(self);
        if let Some(ancestor) = ancestor {
            let ancestor_hash = ancestor.block_hash();
            if let Some(idx) = inner
                .best_block
                .alternative_tips
                .iter()
                .position(|hash| ancestor_hash == *hash)
            {
                inner.best_block.alternative_tips.remove(idx);
            }
        }
        inner
            .best_block
            .alternative_tips
            .push(branch_tip.block_hash());
        Ok(())
    }

    /// Returns the chain_params struct for the current network
    fn chain_params(&self) -> ChainParams {
        let inner = read_lock!(self);
        // We clone the parameters here, because we don't want to hold the lock for too long
        inner.consensus.parameters.clone()
    }

    fn get_header_by_height(&self, height: u32) -> Result<DiskBlockHeader, BlockchainError> {
        read_lock!(self)
            .chainstore
            .get_header_by_height(height)?
            .ok_or(BlockchainError::BlockNotPresent)
    }

    /// Returns the Median Time Past (MTP) of `header`, following its `prev_blockhash` chain.
    fn median_time_past(&self, header: BlockHeader) -> Result<u32, BlockchainError> {
        header.median_time_past_with(|current_header| {
            self.get_disk_block_header(&current_header.prev_blockhash)
                .map(|header| *header)
        })
    }

    /// Returns the Median Time Past (MTP) of `header`'s parent, resolved by `prev_blockhash`.
    fn previous_median_time_past(&self, header: &BlockHeader) -> Result<u32, BlockchainError> {
        let previous_header = *self.get_disk_block_header(&header.prev_blockhash)?;
        self.median_time_past(previous_header)
    }

    fn notify(&self, block: &Block, height: u32, inputs: Option<&HashMap<OutPoint, UtxoData>>) {
        let inner = self.inner.read();
        for client in &inner.subscribers {
            if client.wants_spent_utxos() {
                client.on_block(block, height, inputs);
            } else {
                client.on_block(block, height, None);
            }
        }
    }

    fn new(mut chainstore: PersistedState, network: Network, assume_valid: AssumeValidArg) -> Self {
        let parameters = network.into();
        let genesis = genesis_block(&parameters);

        chainstore
            .save_header(&DiskBlockHeader::FullyValid(genesis.header, 0))
            .expect("Error while saving genesis");

        chainstore
            .update_block_index(0, genesis.block_hash())
            .expect("Error updating index");

        let assume_valid = ChainParams::get_assume_valid(network, assume_valid);

        Self {
            inner: RwLock::new(ChainStateInner {
                chainstore,
                acc: Stump::new(),
                best_block: BestChain {
                    best_block: genesis.block_hash(),
                    depth: 0,
                    validation_index: genesis.block_hash(),
                    alternative_tips: Vec::new(),
                },
                subscribers: Vec::new(),
                fee_estimation: (1_f64, 1_f64, 1_f64),
                ibd: IBDState::HeadersSync,
                consensus: Consensus { parameters },
                assume_valid,
            }),
        }
    }

    /// Fetches a `DiskBlockHeader` from the chain store given its block hash. Returns an error if
    /// it's not present or if the database operation failed.
    fn get_disk_block_header(&self, hash: &BlockHash) -> Result<DiskBlockHeader, BlockchainError> {
        read_lock!(self)
            .chainstore
            .get_header(hash)?
            .ok_or(BlockchainError::BlockNotPresent)
    }

    /// Returns the parsed accumulator for a given block height, if they are present.
    fn get_roots_for_block(&self, height: u32) -> Result<Option<Stump>, BlockchainError> {
        let acc = { write_lock!(self).chainstore.load_roots_for_block(height)? };

        let Some(acc) = acc else {
            return Ok(None);
        };

        let mut acc = acc.as_slice();
        let acc = Stump::deserialize(&mut acc)?;
        Ok(Some(acc))
    }

    /// Re-indexes the chain if we find ourselves in an undefined state
    ///
    /// Here, we have to find what's the best chain we have, then figure out
    /// how many blocks we have validated. We then need to take extra care
    /// to align our accumulator with the validation index. If the validation_index
    /// diverges from the inner acc by even one block, all proofs will be invalid.
    ///
    /// We start at the alleged validation index, and then we look for the last
    /// accumulator we have. If we don't have the accumulator for the validation index,
    /// we roll back our chain to the last accumulator we have, and then re-validate
    /// them
    fn reindex_chain(&self) -> Result<(), BlockchainError> {
        // Figure out what's the best chain we have
        let best_chain = self.find_best_chain();

        // This is what we've found as the best chain, now let's figure out
        // what is the most recent accumulator we have
        let validation_index_height = self
            .get_block_height(&best_chain.validation_index)?
            .unwrap_or(0);

        debug!(
            "Re-indexing chain, validation index height: {}, best block: {}, depth: {}",
            validation_index_height, best_chain.best_block, best_chain.depth
        );

        // This will be the height of the last accumulator we have
        // Find the last height <= validation_index_height that has an accumulator
        let mut found = None;

        for h in (1..=validation_index_height).rev() {
            if let Some(acc) = self.get_roots_for_block(h)? {
                found = Some((h, acc));
                break;
            }
        }

        if let Some((last_acc_height, acc)) = found {
            // If the last acc height is lower than the validation index height, roll back our
            // database state
            for height in (last_acc_height + 1)..=validation_index_height {
                let header = self.get_header_by_height(height)?;
                self.update_header(&DiskBlockHeader::HeadersOnly(*header, height))?;
            }

            let last_acc_header = self.get_header_by_height(last_acc_height)?;
            let mut inner = write_lock!(self);

            inner.acc = acc;
            inner.best_block = best_chain.clone();
            inner.best_block.validation_index = last_acc_header.block_hash();
        } else {
            // We may reindex as well during chain selection (no acc nor validation index)
            write_lock!(self).best_block = best_chain.clone();
        }

        debug!(
            "Re-indexing complete, new validation index: {} new tip: {}",
            best_chain.validation_index, best_chain.best_block
        );

        Ok(())
    }

    /// Reconstructs the `BestChain` data based on the database indexes and headers. This is useful
    /// to recover from an invalid state or data corruption
    fn find_best_chain(&self) -> BestChain {
        let get_disk_block_hash =
            |height: u32| -> Result<Option<BlockHash>, PersistedState::Error> {
                read_lock!(self).chainstore.get_block_hash(height)
            };

        let mut best_block = get_disk_block_hash(0)
            // TODO: handle possible Err
            .expect("TODO handle chainstore error")
            .expect("expected genesis block, found None");

        let mut depth = 0;
        let mut validation_index = best_block;
        let mut next_height = depth + 1;

        // Iteratively fetch the disk header given the next height
        while let Ok(Some(block_hash)) = get_disk_block_hash(next_height) {
            match self.get_disk_block_header(&block_hash) {
                Ok(DiskBlockHeader::FullyValid(_, height)) => {
                    assert_eq!(height, next_height);
                    validation_index = block_hash;
                }
                Ok(DiskBlockHeader::HeadersOnly(_, height))
                | Ok(DiskBlockHeader::AssumedValid(_, height)) => {
                    assert_eq!(height, next_height);
                }
                _ => break,
            }

            best_block = block_hash;
            depth = next_height;
            next_height += 1;
        }

        BestChain {
            best_block,
            depth,
            validation_index,
            alternative_tips: Vec::new(),
        }
    }

    fn load_chain_state(
        mut chainstore: PersistedState,
        network: Network,
        assume_valid: AssumeValidArg,
    ) -> Result<Self, BlockchainError> {
        let best_block = chainstore
            .load_height()?
            .ok_or(BlockchainError::ChainNotInitialized)?;

        debug!("Loaded best block: {best_block:#?}");

        let validation_index_height = chainstore
            .get_header(&best_block.validation_index)?
            .ok_or(BlockchainError::BlockNotPresent)?
            .try_height()?;

        let loaded_acc = chainstore.load_roots_for_block(validation_index_height)?;
        let acc = Self::deserialize_accumulator(loaded_acc)?;

        let inner = ChainStateInner {
            acc,
            best_block,
            chainstore,
            fee_estimation: (1_f64, 1_f64, 1_f64),
            subscribers: Vec::new(),
            ibd: IBDState::HeadersSync,
            consensus: Consensus {
                parameters: network.into(),
            },
            assume_valid: ChainParams::get_assume_valid(network, assume_valid),
        };

        info!(
            "ChainState loaded at height: {}, checking if we have all blocks",
            inner.best_block.depth,
        );

        let chainstate = Self {
            inner: RwLock::new(inner),
        };

        // Check the integrity of our chain
        chainstate.check_chain_integrity()?;
        Ok(chainstate)
    }

    /// Opens an existing chain state or creates one from genesis if the store is uninitialized.
    ///
    /// This is the main entry point for instantiating a [`ChainState`]. If the store already
    /// has data, the state is loaded from it. Otherwise, the chain is initialized from the
    /// genesis block for the given network.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be read or if the persisted state is corrupted.
    pub fn open(
        chainstore: PersistedState,
        network: Network,
        assume_valid: AssumeValidArg,
    ) -> Result<Self, BlockchainError> {
        if chainstore.load_height()?.is_some() {
            return Self::load_chain_state(chainstore, network, assume_valid);
        }
        Ok(Self::new(chainstore, network, assume_valid))
    }

    /// Checks whether our database got a file-level corruption, and if so, reindex.
    ///
    /// This protects us from fs corruption, like random bit-flips or power loss.
    fn check_db_integrity(&self) -> Result<(), BlockchainError> {
        let res = {
            let inner = read_lock!(self);
            inner.chainstore.check_integrity()
        };

        if res.is_err() {
            warn!("We had a data corruption in our database: {res:?}. Reindexing.");
            self.reindex_chain()?;
        }

        Ok(())
    }

    fn check_chain_integrity(&self) -> Result<(), BlockchainError> {
        self.check_db_integrity()?;
        let (best_height, best_hash) = self.get_best_block()?;

        // make sure our index is right for the latest block
        let best_disk_height = self.get_disk_block_header(&best_hash)?.try_height()?;

        if best_height != best_disk_height {
            warn!(
                "Best height {best_height} != disk header height {best_disk_height}. Reindexing."
            );
            self.reindex_chain()?;
            return Ok(());
        }

        // make sure our validation index is pointing to a valid block
        let validation_index = match self.get_validation_index() {
            Ok(idx) => idx,
            Err(e) => {
                warn!("Error with our validation index: {e}. Reindexing.");
                self.reindex_chain()?;
                return Ok(());
            }
        };

        let last_valid_block = self.get_block_hash(validation_index)?;
        let last_valid_header = self.get_disk_block_header(&last_valid_block)?;

        if !matches!(last_valid_header, DiskBlockHeader::FullyValid(_, _)) {
            warn!(
                "The validation index header is not `FullyValid`: {last_valid_header:?}, fetched with validation index {validation_index}. Reindexing."
            );
            self.reindex_chain()?;
            return Ok(());
        }

        // make sure we don't have valid blocks that are after our validation index
        let next_height = validation_index + 1;
        if next_height > best_height {
            // We don't have a next block, so we are good
            return Ok(());
        }

        let next_block_hash = self.get_block_hash(next_height)?;
        let next_header = self.get_disk_block_header(&next_block_hash)?;

        if matches!(next_header, DiskBlockHeader::FullyValid(_, _)) {
            warn!(
                "We have a valid block at height {next_height}, but our validation index is {validation_index}",
            );
            self.reindex_chain()?;
        }

        Ok(())
    }

    /// Tries to deserialize an accumulator returning an empty [`Stump`] `acc` is None
    fn deserialize_accumulator(acc: Option<Vec<u8>>) -> Result<Stump, BlockchainError> {
        let acc = match acc {
            Some(acc) => acc,
            None => return Ok(Stump::new()),
        };

        let mut acc = acc.as_slice();
        Stump::deserialize(&mut acc).map_err(BlockchainError::AccumulatorError)
    }

    fn update_view(
        &self,
        height: u32,
        block: &BlockHeader,
        acc: Stump,
    ) -> Result<(), BlockchainError> {
        let mut inner = write_lock!(self);

        inner
            .chainstore
            .save_header(&DiskBlockHeader::FullyValid(*block, height))?;
        inner
            .chainstore
            .update_block_index(height, block.block_hash())?;

        // save roots for this block
        let mut roots = Vec::new();
        acc.serialize(&mut roots)?;

        inner.chainstore.save_roots_for_block(roots, height)?;

        // Updates our local view of the network
        inner.acc = acc;
        inner.best_block.valid_block(block.block_hash());

        Ok(())
    }

    fn update_tip(&self, best_block: BlockHash, height: u32) {
        let mut inner = write_lock!(self);
        inner.best_block.best_block = best_block;
        inner.best_block.depth = height;
    }

    fn verify_script(&self, height: u32) -> Result<bool, PersistedState::Error> {
        let inner = self.inner.read();
        match inner.assume_valid {
            Some(hash) => {
                match inner.chainstore.get_header(&hash)? {
                    // If the assume-valid block is in the best chain, only verify scripts if we are higher
                    Some(DiskBlockHeader::HeadersOnly(_, assume_h))
                    | Some(DiskBlockHeader::FullyValid(_, assume_h)) => Ok(height > assume_h),
                    // Assume-valid is not in the best chain, so verify all the scripts
                    _ => Ok(true),
                }
            }
            None => Ok(true),
        }
    }
    pub fn acc(&self) -> Stump {
        read_lock!(self).acc.to_owned()
    }
    /// Returns the next required work for the next block, usually it's just the last block's target
    /// but if we are in a retarget period, it's calculated from the last 2016 blocks.
    fn get_next_required_work(
        &self,
        last_block: &BlockHeader,
        next_height: u32,
        next_header: &BlockHeader,
    ) -> Result<Target, BlockchainError> {
        let params: ChainParams = self.chain_params();

        // Outside a retarget height, testnets may allow min-difficulty blocks
        if next_height % 2016 != 0 {
            if !params.params.allow_min_difficulty_blocks {
                return Ok(last_block.target());
            }

            // Special testnet rule, if a block takes more than 20 minutes to mine, we can
            // mine a block with diff 1
            if u64::from(last_block.time) + params.params.pow_target_spacing * 2
                < u64::from(next_header.time)
            {
                return Ok(params.params.max_attainable_target);
            }

            // Otherwise, require the target of the last block that wasn't mined under that rule,
            // stopping at the last retarget block
            let min_difficulty_bits = params.params.max_attainable_target.to_compact_lossy();
            let mut header = *last_block;
            let mut height = next_height - 1;
            while height % 2016 != 0 && header.bits == min_difficulty_bits {
                header = *self.get_disk_block_header(&header.prev_blockhash)?;
                height -= 1;
            }

            return Ok(header.target());
        }

        // Regtest don't have retarget
        if !params.params.no_pow_retargeting {
            // Core reads both ends of the epoch from the new header's own ancestors
            // (`pindexLast->GetAncestor`), and the parent may be on a fork.
            let first_block = self.get_ancestor_at_height(
                last_block,
                next_height - 1,
                next_height - DIFFCHANGE_INTERVAL,
            )?;

            let target =
                Consensus::calc_next_work_required(last_block, &first_block, self.chain_params());

            if target < params.params.max_attainable_target {
                return Ok(target);
            }

            return Ok(params.params.max_attainable_target);
        }

        Ok(last_block.target())
    }

    /// Returns the ancestor of `header` at `target_height`, where `header` is at `header_height`.
    ///
    /// The height index only knows the best chain, so it is used when `header` is on it. Otherwise
    /// this follows `prev_blockhash`, which is also right for a header on a fork.
    fn get_ancestor_at_height(
        &self,
        header: &BlockHeader,
        header_height: u32,
        target_height: u32,
    ) -> Result<BlockHeader, BlockchainError> {
        if self.get_block_hash(header_height).ok() == Some(header.block_hash()) {
            return Ok(*self.get_header_by_height(target_height)?);
        }

        let mut ancestor = *header;
        for _ in target_height..header_height {
            ancestor = *self.get_disk_block_header(&ancestor.prev_blockhash)?;
        }

        Ok(ancestor)
    }

    /// Check timestamp against prev for difficulty-adjustment blocks to prevent timewarp attacks.
    /// Must only be called for non-genesis block headers.
    fn check_bip94_block(&self, block: &BlockHeader, height: u32) -> Result<(), BlockchainError> {
        let params = self.chain_params();
        // Difficulty adjustment window
        let window = params.params.miner_confirmation_window;

        if !params.enforce_bip94 || height % window != 0 {
            return Ok(());
        }

        let prev_header = self.get_disk_block_header(&block.prev_blockhash)?;
        Ok(Consensus::check_bip94_time(block, &prev_header)?)
    }

    /// Validates the block without checking whether the inputs are present in the UTXO set. This
    /// function contains the core validation logic.
    ///
    /// The methods `BlockchainInterface::validate_block` and `UpdatableChainstate::connect_block`
    /// call this and additionally verify the inclusion proof (i.e., they perform full validation).
    pub fn validate_block_no_acc(
        &self,
        block: &Block,
        height: u32,
        inputs: HashMap<OutPoint, UtxoData>,
    ) -> Result<(), BlockchainError> {
        let consensus = read_lock!(self).consensus.clone();
        consensus.check_block(block, height)?;

        // Validate block transactions
        let subsidy = consensus.get_subsidy(height);
        let lock_time_cutoff = consensus.block_lock_time_cutoff(height, &block.header, || {
            self.previous_median_time_past(&block.header)
        })?;
        #[cfg(feature = "bitcoinkernel")]
        let flags = consensus
            .parameters
            .get_validation_flags(height, block.block_hash());
        #[cfg(not(feature = "bitcoinkernel"))]
        let flags = consensus.parameters.get_sigop_flags(block.block_hash());
        let verify_script = self.verify_script(height)?;

        Consensus::verify_block_transactions(
            height,
            lock_time_cutoff,
            inputs,
            &block.txdata,
            subsidy,
            verify_script,
            flags,
        )?;
        Ok(())
    }
}

impl<PersistedState: ChainStore> BlockchainInterface for ChainState<PersistedState> {
    type Error = BlockchainError;

    fn get_params(&self) -> bitcoin::params::Params {
        self.chain_params().params
    }

    fn get_chain_params(&self) -> ChainParams {
        self.chain_params()
    }

    fn get_work(&self, tip: BlockHash) -> Result<Work, Self::Error> {
        let header = self.get_block_header(&tip)?;
        self.get_branch_work(header)
    }

    fn acc(&self) -> Stump {
        read_lock!(self).acc.to_owned()
    }

    fn size_on_disk(&self) -> Result<u64, Self::Error> {
        Ok(read_lock!(self).chainstore.size_on_disk()?)
    }

    fn get_mtp_by_height(&self, height: u32) -> Result<u32, Self::Error> {
        let header = *self.get_header_by_height(height)?;
        self.median_time_past(header)
    }

    fn get_fork_point(&self, block: BlockHash) -> Result<BlockHash, Self::Error> {
        let fork_point = self.find_fork_point(&self.get_block_header(&block)?)?;
        Ok(fork_point.block_hash())
    }

    fn update_acc(
        &self,
        acc: Stump,
        block: &Block,
        height: u32,
        proof: Proof,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<Stump, Self::Error> {
        Consensus::update_acc(&acc, block, height, proof, del_hashes)
    }

    fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error> {
        let inner = read_lock!(self);
        let mut tips = Vec::new();

        tips.push(inner.best_block.best_block);
        tips.extend(inner.best_block.alternative_tips.iter());

        Ok(tips)
    }

    fn validate_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
        acc: Stump,
    ) -> Result<(), Self::Error> {
        // Convert to BitcoinNodeHashes, from rustreexo
        let del_hashes: Vec<_> = del_hashes
            .into_iter()
            .map(|hash| BitcoinNodeHash::Some(hash.to_byte_array()))
            .collect();

        if !acc.verify(&proof, &del_hashes)? {
            Err(BlockValidationErrors::InvalidUtreexoProof)?;
        }

        let height = self
            .get_disk_block_header(&block.block_hash())?
            .try_height()?;

        self.validate_block_no_acc(block, height, inputs)
    }

    fn get_block_locator_for_tip(&self, tip: BlockHash) -> Result<Vec<BlockHash>, BlockchainError> {
        let mut hashes = Vec::new();
        let height = self.get_disk_block_header(&tip)?.try_height()?;

        let mut index = height;
        let mut current_height = height;
        let mut current_header = self.get_disk_block_header(&tip)?;
        let mut step = 1;

        while index > 0 {
            while current_height > index {
                current_header = self.get_ancestor(&current_header)?;
                current_height -= 1;
            }

            if hashes.len() >= 10 {
                step *= 2;
            }

            hashes.push(current_header.block_hash());

            if index > step {
                index -= step;
            } else {
                break;
            }
        }

        // genesis
        hashes.push(self.get_block_hash(0)?);
        Ok(hashes)
    }

    fn is_in_ibd(&self) -> bool {
        self.inner.read().ibd != IBDState::Done
    }

    fn get_block_height(&self, hash: &BlockHash) -> Result<Option<u32>, Self::Error> {
        self.get_disk_block_header(hash)
            .map(|header| header.height())
    }

    fn get_assume_valid_height(&self) -> Result<Option<u32>, Self::Error> {
        let inner = read_lock!(self);
        let Some(hash) = inner.assume_valid else {
            return Ok(None);
        };
        let height = match inner.chainstore.get_header(&hash)? {
            Some(DiskBlockHeader::HeadersOnly(_, height))
            | Some(DiskBlockHeader::FullyValid(_, height))
            | Some(DiskBlockHeader::AssumedValid(_, height)) => height,
            // `InFork`, `Orphan` or `InvalidChain`
            _ => return Ok(None),
        };

        // Cross-check that the AssumeValid block is on the current best chain.
        if height > inner.best_block.depth {
            return Ok(None);
        }
        if inner.chainstore.get_block_hash(height)? != Some(hash) {
            return Ok(None);
        }

        Ok(Some(height))
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
        read_lock!(self)
            .chainstore
            .get_block_hash(height)?
            .ok_or(BlockchainError::BlockNotPresent)
    }

    fn get_tx(&self, _txid: &Txid) -> Result<Option<Transaction>, Self::Error> {
        unimplemented!("This chainstate doesn't hold any tx")
    }

    fn get_height(&self) -> Result<u32, Self::Error> {
        let inner = read_lock!(self);
        Ok(inner.best_block.depth)
    }

    fn estimate_fee(&self, target: usize) -> Result<f64, Self::Error> {
        let inner = read_lock!(self);
        if target == 1 {
            Ok(inner.fee_estimation.0)
        } else if target == 10 {
            Ok(inner.fee_estimation.1)
        } else {
            Ok(inner.fee_estimation.2)
        }
    }

    fn get_block(&self, _hash: &BlockHash) -> Result<Block, Self::Error> {
        unimplemented!("This chainstate doesn't hold full blocks")
    }

    fn get_best_block(&self) -> Result<(u32, BlockHash), Self::Error> {
        let inner = read_lock!(self);
        Ok((inner.best_block.depth, inner.best_block.best_block))
    }

    fn get_block_header(&self, hash: &BlockHash) -> Result<BlockHeader, Self::Error> {
        let inner = read_lock!(self);
        if let Some(header) = inner.chainstore.get_header(hash)? {
            return Ok(*header);
        }
        Err(BlockchainError::BlockNotPresent)
    }

    fn subscribe(&self, tx: Arc<dyn BlockConsumer>) {
        let mut inner = self.inner.write();
        inner.subscribers.push(tx);
    }

    fn get_block_locator(&self) -> Result<Vec<BlockHash>, BlockchainError> {
        let top_height = self.get_height()?;
        let mut indexes = Vec::new();
        let mut step = 1;
        let mut index = top_height;
        while index > 0 {
            if indexes.len() >= 10 {
                step *= 2;
            }
            indexes.push(index);
            if index > step {
                index -= step;
            } else {
                break;
            }
        }
        indexes.push(0);
        let hashes = indexes
            .iter()
            .flat_map(|idx| self.get_block_hash(*idx))
            .collect();

        Ok(hashes)
    }

    fn get_validation_index(&self) -> Result<u32, Self::Error> {
        let inner = self.inner.read();
        let validation = inner.best_block.validation_index;
        let header = self.get_disk_block_header(&validation)?;
        // The last validated disk header can only be FullyValid
        if let DiskBlockHeader::FullyValid(_, height) = header {
            return Ok(height);
        }

        Err(BlockchainError::BadValidationIndex)
    }

    fn is_coinbase_mature(&self, height: u32, block: BlockHash) -> Result<bool, Self::Error> {
        let chain_params = self.chain_params();
        let current_height = self.get_disk_block_header(&block)?.try_height()?;

        Ok(height + chain_params.coinbase_maturity <= current_height)
    }

    fn ibd_state(&self) -> IBDState {
        let inner = read_lock!(self);
        inner.ibd
    }

    fn get_warnings(&self) -> Vec<ChainStoreWarning> {
        read_lock!(self).chainstore.get_warnings()
    }
}

impl<PersistedState: ChainStore> UpdatableChainstate for ChainState<PersistedState> {
    fn switch_chain(&self, new_tip: BlockHash) -> Result<(), BlockchainError> {
        let new_tip = self.get_block_header(&new_tip)?;
        self.reorg(new_tip)
    }

    fn get_acc(&self) -> Stump {
        self.acc()
    }

    fn mark_block_as_valid(&self, block: BlockHash) -> Result<(), BlockchainError> {
        let header = self.get_disk_block_header(&block)?;
        let height = header.try_height()?;

        let new_header = DiskBlockHeader::FullyValid(*header, height);
        self.update_header(&new_header)
    }

    fn mark_chain_as_assumed(
        &self,
        acc: Stump,
        assumed_hash: BlockHash,
    ) -> Result<bool, BlockchainError> {
        let assumed_header = self.get_disk_block_header(&assumed_hash)?;

        let mut header = assumed_header;
        let mut hash = assumed_hash;
        loop {
            if self.is_genesis(&header) {
                break;
            }

            let height = header.try_height()?;
            self.update_header_and_index(
                &DiskBlockHeader::FullyValid(*header, height),
                hash,
                height,
            )?;

            // Move to the previous block
            header = self.get_ancestor(&header)?;
            hash = header.block_hash();
        }

        // Update the tip and accumulator data with our assumed tip
        self.update_view(assumed_header.try_height()?, &assumed_header, acc)?;
        self.flush()?;

        Ok(true)
    }

    fn invalidate_block(&self, block: BlockHash) -> Result<(), BlockchainError> {
        let height = self.get_disk_block_header(&block)?.try_height()?;
        let current_height = self.get_height()?;

        // Mark all blocks after this one as invalid
        for h in height..=current_height {
            let hash = self.get_block_hash(h)?;
            let header = self.get_block_header(&hash)?;
            let new_header = DiskBlockHeader::InvalidChain(header);
            self.update_header(&new_header)?;
        }
        // Row back to our previous state. Note that acc doesn't actually change in this case
        // only the currently best known block.
        self.update_tip(
            self.get_ancestor(&self.get_block_header(&block)?)?
                .block_hash(),
            height - 1,
        );
        Ok(())
    }

    fn update_ibd(&self, ibd_state: IBDState) {
        let mut inner = write_lock!(self);
        inner.ibd = ibd_state;
    }

    fn connect_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<u32, BlockchainError> {
        let header = self.get_disk_block_header(&block.block_hash())?;
        let height = match header {
            DiskBlockHeader::FullyValid(_, height) => {
                let validation_index = self.get_validation_index()?;

                // If this block is not our validation index, but the caller is trying to connect
                // it, this is a logical error, and we will have spurious errors, specially with
                // invalid proof. They don't mean the block is invalid, just that we are using the
                // wrong accumulator, since we are not processing the right block.
                if height != validation_index {
                    Err(BlockValidationErrors::BlockDoesntExtendTip)?;
                }

                // If this block is our validation index, but it's fully valid, this clearly means
                // there was some corruption of our state. If we don't process this block, we will
                // be stuck forever.
                //
                // Note: You may think "just kick the validation index one block further and we are
                // good". But this is not the case, because we still need to update our
                // accumulator. Otherwise, the next block will always have an invalid proof
                // (because the accumulator is not updated).
                height
            },

            // Our called tried to connect_block on a block that is not the next one in our chain
            DiskBlockHeader::Orphan(_)
            | DiskBlockHeader::AssumedValid(_, _) // this will be validated by a partial chain
            | DiskBlockHeader::InFork(_, _)
            | DiskBlockHeader::InvalidChain(_) => return Err(BlockValidationErrors::BlockExtendsAnOrphanChain)?,

            DiskBlockHeader::HeadersOnly(_, height) => {
                let validation_index = self.get_validation_index()?;

                // In case of a `HeadersOnly` block, we need to check if the height is
                // the next one after the validation index. If not, we would be trying to
                // connect a block where our accumulator isn't the right one. So the proof will
                // always be invalid.
                if height != validation_index + 1 {
                    Err(BlockValidationErrors::BlockDoesntExtendTip)?;
                }

                height
            }
        };

        // Clone inputs only if a subscriber wants spent utxos
        let inputs_for_notifications = self
            .inner
            .read()
            .subscribers
            .iter()
            .any(|subscriber| subscriber.wants_spent_utxos())
            .then(|| inputs.clone());

        let acc = Consensus::update_acc(&self.acc(), block, height, proof, del_hashes)?;
        self.validate_block_no_acc(block, height, inputs)?;

        self.update_view(height, &block.header, acc)?;

        info!(
            "New tip! hash={} height={height} tx_count={}",
            block.block_hash(),
            block.txdata.len()
        );

        #[cfg(feature = "metrics")]
        metrics::get_metrics().block_height.set(height.into());

        if !self.is_in_ibd() || height % 100_000 == 0 {
            self.flush()?;
        }

        // Notify others we have a new block
        self.notify(block, height, inputs_for_notifications.as_ref());
        Ok(height)
    }

    fn handle_transaction(&self) -> Result<(), BlockchainError> {
        unimplemented!("This chain_state has no mempool")
    }

    fn flush(&self) -> Result<(), BlockchainError> {
        let mut inner = write_lock!(self);
        let best_block = inner.best_block.clone();

        inner.chainstore.save_height(&best_block)?;
        inner.chainstore.flush()?;

        Ok(())
    }

    fn accept_header(&self, header: BlockHeader) -> Result<(), BlockchainError> {
        let disk_header = self.get_disk_block_header(&header.block_hash());

        match disk_header {
            Err(e @ BlockchainError::Database(_)) => {
                // If there's a database error we don't know if we already
                // have the header or not
                return Err(e);
            }
            Ok(found) => {
                // Possibly reindex to recompute the best_block field
                self.maybe_reindex(&found)?;
                // We already have this header
                return Ok(());
            }
            _ => (),
        }
        // The best block we know of
        let best_block = self.get_best_block()?;

        // Do validation in this header
        let block_hash = self.validate_header(&header)?;

        // Update our current tip
        if header.prev_blockhash == best_block.1 {
            let height = best_block.0 + 1;
            debug!("Header builds on top of our best chain");

            write_lock!(self).best_block.new_block(block_hash, height);
            let disk_header = DiskBlockHeader::HeadersOnly(header, height);

            self.update_header_and_index(&disk_header, block_hash, height)?;
        } else {
            debug!("Header not in the best chain");

            self.maybe_reorg(header)?;
        }

        Ok(())
    }

    fn get_root_hashes(&self) -> Vec<BitcoinNodeHash> {
        let inner = read_lock!(self);
        inner.acc.roots.clone()
    }

    fn get_partial_chain(
        &self,
        initial_height: u32,
        final_height: u32,
        acc: Stump,
    ) -> Result<super::partial_chain::PartialChainState, BlockchainError> {
        let blocks = (0..=final_height)
            .map(|height| {
                let hash = self
                    .get_block_hash(height)
                    .expect("Block should be present");
                *self
                    .get_disk_block_header(&hash)
                    .expect("Block should be present")
            })
            .collect();

        let inner = PartialChainStateInner {
            error: None,
            blocks,
            consensus: Consensus {
                parameters: self.chain_params(),
            },
            current_acc: acc,
            final_height,
            assume_valid: false,
            current_height: initial_height,
        };

        Ok(PartialChainState(UnsafeCell::new(inner)))
    }
}

impl<T: ChainStore> TryFrom<ChainStateBuilder<T>> for ChainState<T> {
    type Error = BlockchainBuilderError;

    fn try_from(mut builder: ChainStateBuilder<T>) -> Result<Self, Self::Error> {
        let inner = ChainStateInner {
            acc: builder.acc().unwrap_or_default(),
            chainstore: builder.chainstore()?,
            best_block: builder.best_block()?,
            assume_valid: builder.assume_valid(),
            ibd: builder.ibd_state(),
            subscribers: Vec::new(),
            fee_estimation: (1_f64, 1_f64, 1_f64),
            consensus: Consensus {
                parameters: builder.chain_params()?,
            },
        };

        let inner = RwLock::new(inner);
        Ok(Self { inner })
    }
}

#[macro_export]
/// Grabs a RwLock for reading
macro_rules! read_lock {
    ($obj:ident) => {
        $obj.inner.read()
    };
}

#[macro_export]
/// Grabs a RwLock for writing
macro_rules! write_lock {
    ($obj:ident) => {
        $obj.inner.write()
    };
}

#[cfg(all(test, feature = "flat-chainstore"))]
mod test {
    use std::format;
    use std::fs::File;
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::vec::Vec;

    use bitcoin::Block;
    use bitcoin::BlockHash;
    use bitcoin::CompactTarget;
    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Sequence;
    use bitcoin::Transaction;
    use bitcoin::TxMerkleNode;
    use bitcoin::Work;
    use bitcoin::absolute::LockTime;
    use bitcoin::block::Header as BlockHeader;
    use bitcoin::block::Version as HeaderVersion;
    use bitcoin::consensus::Decodable;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::constants::DIFFCHANGE_INTERVAL;
    use bitcoin::constants::genesis_block;
    use bitcoin::hashes::Hash;
    use bitcoin::opcodes::OP_TRUE;
    use bitcoin::params::Params;
    use bitcoin::script::Builder;
    use bitcoin::transaction::Version as TransactionVersion;
    use floresta_common::assert_ok;
    use floresta_common::bhash;
    use rand::RngExt;
    use rustreexo::node_hash::BitcoinNodeHash;
    use rustreexo::proof::Proof;
    use rustreexo::stump::Stump;

    use super::BlockchainInterface;
    use super::ChainParams;
    use super::ChainState;
    use super::DiskBlockHeader;
    use super::UpdatableChainstate;
    use crate::AssumeValidArg;
    use crate::BlockchainError;
    use crate::ChainStore;
    use crate::CompactLeafData;
    use crate::FlatChainStore;
    use crate::FlatChainStoreConfig;
    use crate::ScriptPubKeyKind;
    use crate::extensions::WorkExt;
    use crate::prelude::HashMap;
    use crate::proof_util::process_proof;
    use crate::pruned_utreexo::consensus::Consensus;
    use crate::pruned_utreexo::error::BlockValidationErrors;
    use crate::pruned_utreexo::utxo_data::UtxoData;
    use crate::txin;
    use crate::txout;

    const DEFAULT_TEST_CHAINSTORE_SIZE: usize = 32_768;
    const TEST_FORK_FILE_SIZE: usize = 10_000;
    const EASIEST_REGTEST_TARGET_BITS: u32 = 0x207f_ffff;
    fn setup_test_chain(
        network: Network,
        assume_valid_arg: AssumeValidArg,
        header_capacity: Option<usize>,
    ) -> ChainState<FlatChainStore> {
        let test_id = rand::random::<u64>();
        let capacity = header_capacity.unwrap_or(DEFAULT_TEST_CHAINSTORE_SIZE);
        let config = FlatChainStoreConfig {
            block_index_size: Some(capacity),
            headers_file_size: Some(capacity),
            fork_file_size: Some(TEST_FORK_FILE_SIZE), // Will be rounded up to 16,384
            cache_size: Some(10),
            file_permission: Some(0o660),
            path: format!("./tmp-db/{test_id}/").into(),
        };

        let chainstore = FlatChainStore::new(config).unwrap();
        ChainState::open(chainstore, network, assume_valid_arg).unwrap()
    }

    #[test]
    fn test_assume_valid_height_uses_configured_block() {
        let genesis = genesis_block(Network::Regtest).block_hash();
        for (setting, expected) in [
            (AssumeValidArg::Disabled, None),
            (AssumeValidArg::Hardcoded, Some(0)),
            (AssumeValidArg::UserInput(genesis), Some(0)),
            (AssumeValidArg::UserInput(BlockHash::all_zeros()), None),
        ] {
            // Arc must forward the setting too, as the wire node shares its chainstate
            let chain = std::sync::Arc::new(setup_test_chain(Network::Regtest, setting, None));
            assert_eq!(chain.get_assume_valid_height().unwrap(), expected);
        }
    }

    #[test]
    fn test_assume_valid_height_requires_current_best_chain() {
        let genesis = genesis_block(Network::Regtest);
        let header = BlockHeader {
            prev_blockhash: genesis.block_hash(),
            time: genesis.header.time + 1,
            ..genesis.header
        };
        let hash = header.block_hash();
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::UserInput(hash), None);
        {
            let mut inner = write_lock!(chain);
            inner.best_block.best_block = hash;
            inner.best_block.depth = 1;
            inner.chainstore.update_block_index(1, hash).unwrap();
        }

        let active = DiskBlockHeader::HeadersOnly(header, 1);
        // Only active-chain header states are eligible
        for (stored, expected) in [
            (DiskBlockHeader::FullyValid(header, 1), Some(1)),
            (active, Some(1)),
            (DiskBlockHeader::AssumedValid(header, 1), Some(1)),
            (DiskBlockHeader::InFork(header, 1), None),
            (DiskBlockHeader::Orphan(header), None),
            (DiskBlockHeader::InvalidChain(header), None),
        ] {
            write_lock!(chain).chainstore.save_header(&stored).unwrap();
            assert_eq!(chain.get_assume_valid_height().unwrap(), expected);
        }

        let other_header = BlockHeader {
            time: header.time + 1,
            ..header
        };

        #[rustfmt::skip]
        let cases = [
            // Different block at the same height
            (DiskBlockHeader::HeadersOnly(other_header, 1), other_header.block_hash(), 1),
            // Simulate a rollback to genesis, retaining the AssumeValid header at height 1
            (active, genesis.block_hash(), 0),
        ];
        for (selected, tip_hash, depth) in cases {
            {
                let mut inner = write_lock!(chain);
                inner.chainstore.save_header(&active).unwrap();
                inner.chainstore.save_header(&selected).unwrap();
                inner
                    .chainstore
                    .update_block_index(1, selected.block_hash())
                    .unwrap();
                inner.best_block.best_block = tip_hash;
                inner.best_block.depth = depth;
            }
            assert_eq!(chain.get_assume_valid_height().unwrap(), None);
        }
    }

    fn anyone_can_spend_script() -> ScriptBuf {
        let mut script = ScriptBuf::new();
        script.push_opcode(OP_TRUE);
        script
    }

    fn test_coinbase(height: u32, sequence: Sequence, lock_time: LockTime) -> Transaction {
        let script_sig = Builder::new()
            .push_int(i64::from(height))
            .push_int(0)
            .into_script();

        Transaction {
            version: TransactionVersion::TWO,
            lock_time,
            input: vec![txin!(OutPoint::null(), script_sig, sequence)],
            output: vec![txout!(0, ScriptBuf::new_op_return([0x0, 0x1]))],
        }
    }

    fn block_with_transactions(height: u32, txdata: Vec<Transaction>) -> Block {
        let mut block = Block {
            header: BlockHeader {
                version: HeaderVersion::TWO,
                prev_blockhash: genesis_block(Network::Regtest).block_hash(),
                merkle_root: TxMerkleNode::all_zeros(),
                time: 1_716_400_000 + height,
                bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
                nonce: 0,
            },
            txdata,
        };
        block.header.merkle_root = block.compute_merkle_root().expect("block txs");
        block
    }

    fn block_with_coinbase(height: u32, coinbase: Transaction) -> Block {
        block_with_transactions(height, vec![coinbase])
    }

    fn test_outpoint(vout: u32) -> OutPoint {
        OutPoint {
            txid: genesis_block(Network::Regtest).txdata[0].compute_txid(),
            vout,
        }
    }

    /// Builds a two-input test transaction with the given lock time and sequences,
    /// plus its referenced input UTXOs.
    fn test_spend(
        lock_time: LockTime,
        first_sequence: Sequence,
        second_sequence: Sequence,
    ) -> (Transaction, HashMap<OutPoint, UtxoData>) {
        let first_prevout = test_outpoint(0);
        let second_prevout = test_outpoint(1);
        let transaction = Transaction {
            version: TransactionVersion::TWO,
            lock_time,
            input: vec![
                txin!(first_prevout, ScriptBuf::new(), first_sequence),
                txin!(second_prevout, ScriptBuf::new(), second_sequence),
            ],
            output: vec![txout!(1, ScriptBuf::new_op_return([0x0, 0x1]))],
        };
        let input_utxo = UtxoData {
            txout: txout!(1, anyone_can_spend_script()),
            is_coinbase: false,
            creation_height: 0,
            creation_time: 0,
        };
        let inputs = HashMap::from([
            (first_prevout, input_utxo.clone()),
            (second_prevout, input_utxo),
        ]);

        (transaction, inputs)
    }

    fn decode_block_and_inputs(
        block_file: File,
        stxos_file: File,
    ) -> (Block, HashMap<OutPoint, UtxoData>) {
        let block_bytes = zstd::decode_all(block_file).unwrap();
        let block: Block = deserialize(&block_bytes).unwrap();

        // Get utxos spent in the block
        let stxos_bytes = zstd::decode_all(stxos_file).unwrap();
        let mut stxos: Vec<UtxoData> =
            serde_json::from_slice(&stxos_bytes).expect("Failed to deserialize JSON");

        let inputs = block
            .txdata
            .iter()
            .skip(1) // Skip the coinbase transaction
            .flat_map(|tx| &tx.input)
            .map(|txin| (txin.previous_output, stxos.remove(0)))
            .collect();

        assert!(stxos.is_empty(), "Moved all stxos to the inputs map");

        (block, inputs)
    }

    /// Stores 11 synthetic headers ending at `tip_height` for Median Time Past (MTP).
    /// Returns the most-recent synthetic block hash.
    fn store_mtp_headers(
        chain: &ChainState<FlatChainStore>,
        network: Network,
        tip_height: u32,
        time: u32,
    ) -> BlockHash {
        assert!(tip_height > 10, "Need at least 11 headers for MTP");
        let genesis = genesis_block(network);
        let mut prev_hash = genesis.block_hash();

        for height in (tip_height - 10)..=tip_height {
            let header = BlockHeader {
                version: HeaderVersion::NO_SOFT_FORK_SIGNALLING,
                prev_blockhash: prev_hash,
                merkle_root: genesis.header.merkle_root,
                time,
                bits: CompactTarget::from_consensus(0x1702_8c74),
                nonce: height,
            };
            prev_hash = header.block_hash();

            write_lock!(chain)
                .chainstore
                .save_header(&DiskBlockHeader::FullyValid(header, height))
                .unwrap();
            write_lock!(chain)
                .chainstore
                .update_block_index(height, prev_hash)
                .unwrap();
        }

        prev_hash
    }

    fn assert_malleated_leaf_rejects_proof(malleate_script: bool) {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let mut parent = genesis_block(Network::Regtest).block_hash();
        let mut funding_outpoint = OutPoint::null();

        // Only height 1 adds a spendable output, so its proof has target 0 and no siblings.
        for height in 1..=100 {
            let mut coinbase = test_coinbase(height, Sequence::MAX, LockTime::ZERO);
            if height == 1 {
                coinbase.output = vec![txout!(1, anyone_can_spend_script())];
                funding_outpoint = OutPoint::new(coinbase.compute_txid(), 0);
            }
            let mut block = block_with_coinbase(height, coinbase);
            block.header.prev_blockhash = parent;
            while block.header.validate_pow(block.header.target()).is_err() {
                block.header.nonce += 1;
            }
            chain.accept_header(block.header).unwrap();
            chain
                .connect_block(&block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
            parent = block.block_hash();
        }

        let spend = Transaction {
            version: TransactionVersion::TWO,
            lock_time: LockTime::ZERO,
            input: vec![txin!(funding_outpoint, ScriptBuf::new(), Sequence::MAX)],
            output: vec![txout!(1, ScriptBuf::new_op_return([0]))],
        };
        let mut block = block_with_transactions(
            101,
            vec![test_coinbase(101, Sequence::MAX, LockTime::ZERO), spend],
        );
        block.header.prev_blockhash = parent;
        while block.header.validate_pow(block.header.target()).is_err() {
            block.header.nonce += 1;
        }
        chain.accept_header(block.header).unwrap();

        let honest_leaf = CompactLeafData {
            header_code: (1 << 1) | 1,
            amount: 1,
            spk_ty: ScriptPubKeyKind::Other(anyone_can_spend_script().into_bytes().into()),
        };
        let mut malicious_leaf = honest_leaf.clone();
        if malleate_script {
            malicious_leaf.spk_ty =
                ScriptPubKeyKind::Other(ScriptBuf::new_op_return([42]).into_bytes().into());
        } else {
            // Claim the otherwise mature coinbase was created in the previous block.
            malicious_leaf.header_code = (100 << 1) | 1;
        }
        let proof = Proof {
            targets: vec![0],
            hashes: Vec::new(),
        };
        let (bad_hashes, bad_inputs) = process_proof(
            &[malicious_leaf],
            &block.txdata,
            101,
            |height| chain.get_block_hash(height),
            |height| chain.get_mtp_by_height(height),
        )
        .unwrap();

        // These unauthenticated inputs would fail consensus checks if checked first.
        let validation_error = chain
            .validate_block_no_acc(&block, 101, bad_inputs.clone())
            .unwrap_err();
        assert!(
            matches!(validation_error, BlockchainError::TransactionError(_)),
            "expected a transaction failure from malleated leaf data: {validation_error:?}"
        );

        let acc = chain.acc();
        let error = chain
            .connect_block(&block, proof.clone(), bad_inputs, bad_hashes)
            .unwrap_err();
        // The wire layer retries and bans the proof peer for AccumulatorError, whereas
        // TransactionError would invalidate this perfectly valid block.
        assert!(
            matches!(error, BlockchainError::AccumulatorError(_)),
            "must reject the proof, not the block: {error:?}"
        );
        assert_eq!(chain.acc(), acc);
        assert_eq!(chain.get_validation_index().unwrap(), 100);
        assert!(matches!(
            chain.get_disk_block_header(&block.block_hash()).unwrap(),
            DiskBlockHeader::HeadersOnly(_, 101)
        ));

        // An honest peer can supply the original leaf and connect the exact same block.
        let (hashes, inputs) = process_proof(
            &[honest_leaf],
            &block.txdata,
            101,
            |height| chain.get_block_hash(height),
            |height| chain.get_mtp_by_height(height),
        )
        .unwrap();
        assert_eq!(
            chain.connect_block(&block, proof, inputs, hashes).unwrap(),
            101
        );
        assert_eq!(chain.get_validation_index().unwrap(), 101);
        assert!(matches!(
            chain.get_disk_block_header(&block.block_hash()).unwrap(),
            DiskBlockHeader::FullyValid(_, 101)
        ));
    }

    #[test]
    fn malleated_creation_height_rejects_proof_not_block() {
        assert_malleated_leaf_rejects_proof(false);
    }

    #[test]
    #[cfg(feature = "bitcoinkernel")]
    fn malleated_other_script_rejects_proof_not_block() {
        assert_malleated_leaf_rejects_proof(true);
    }

    #[test]
    fn reject_non_final_block_transaction() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let height = 1;

        let invalid = block_with_coinbase(
            height,
            test_coinbase(
                height,
                Sequence::ENABLE_LOCKTIME_NO_RBF,
                LockTime::from_height(height).unwrap(),
            ),
        );
        let valid =
            block_with_coinbase(height, test_coinbase(height, Sequence::MAX, LockTime::ZERO));

        match chain.validate_block_no_acc(&invalid, height, HashMap::new()) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::NonFinalTransaction)) => {}
            other => panic!("expected NonFinalTransaction, got {other:?}"),
        }

        chain
            .validate_block_no_acc(&valid, height, HashMap::new())
            .expect("block transactions are valid and final");
    }

    #[test]
    fn future_lock_time_requires_all_input_sequences_to_be_final() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let height = 1;
        let future_lock_time = LockTime::from_height(height + 1).unwrap();
        let coinbase = || test_coinbase(height, Sequence::MAX, LockTime::ZERO);

        let (transaction, inputs) = test_spend(future_lock_time, Sequence::MAX, Sequence::MAX);
        let block = block_with_transactions(height, vec![coinbase(), transaction]);

        chain
            .validate_block_no_acc(&block, height, inputs)
            .expect("all input sequences are final");

        let (transaction, inputs) = test_spend(
            future_lock_time,
            Sequence::MAX,
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );
        let block = block_with_transactions(height, vec![coinbase(), transaction]);

        match chain.validate_block_no_acc(&block, height, inputs) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::NonFinalTransaction)) => {}
            other => panic!("expected NonFinalTransaction, got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "this test is very slow in debug mode")]
    fn test_validate_many_inputs_block() {
        const HEIGHT: u32 = 367891;
        const BLOCKS: usize = (HEIGHT + 1) as usize;

        let block_file = File::open("./testdata/block_367891/raw.zst").unwrap();
        let stxos_file = File::open("./testdata/block_367891/spent_utxos.zst").unwrap();
        let (block, inputs) = decode_block_and_inputs(block_file, stxos_file);

        assert_eq!(
            block.block_hash(),
            bhash!("000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970"),
        );

        // Check whether the block validation passes or not
        let chain = setup_test_chain(Network::Bitcoin, AssumeValidArg::Disabled, Some(BLOCKS));
        chain
            .validate_block_no_acc(&block, HEIGHT, inputs)
            .expect("Block must be valid");
    }

    #[test]
    fn test_reject_block_with_too_many_sigops() {
        const HEIGHT: u32 = 783_426;
        const BLOCKS: usize = (HEIGHT + 1) as usize;

        let block_file = File::open("./testdata/invalid_block_783426/raw.zst").unwrap();
        let stxos_file = File::open("./testdata/invalid_block_783426/spent_utxos.zst").unwrap();
        let (mut block, inputs) = decode_block_and_inputs(block_file, stxos_file);

        assert_eq!(
            block.block_hash(),
            bhash!("00000000000000000002ec935e245f8ae70fc68cc828f05bf4cfa002668599e4"),
        );

        let chain = setup_test_chain(Network::Bitcoin, AssumeValidArg::Disabled, Some(BLOCKS));
        read_lock!(chain)
            .consensus
            .check_block(&block, HEIGHT)
            .expect("context-free checks must pass");

        let prev_hash = store_mtp_headers(&chain, Network::Bitcoin, HEIGHT - 1, block.header.time);
        block.header.prev_blockhash = prev_hash;
        match chain.validate_block_no_acc(&block, HEIGHT, inputs) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::TooManySigops)) => {}
            other => panic!("expected TooManySigops, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_full_block() {
        const HEIGHT: u32 = 866342;
        const BLOCKS: usize = (HEIGHT + 1) as usize;

        let block_file = File::open("./testdata/block_866342/raw.zst").unwrap();
        let stxos_file = File::open("./testdata/block_866342/spent_utxos.zst").unwrap();
        let (mut block, inputs) = decode_block_and_inputs(block_file, stxos_file);

        assert_eq!(
            block.block_hash(),
            bhash!("000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1"),
        );

        // Check whether the block validation passes or not
        let chain = setup_test_chain(Network::Bitcoin, AssumeValidArg::Disabled, Some(BLOCKS));
        let prev_hash = store_mtp_headers(&chain, Network::Bitcoin, HEIGHT - 1, block.header.time);
        block.header.prev_blockhash = prev_hash;
        chain
            .validate_block_no_acc(&block, HEIGHT, inputs)
            .expect("Block must be valid");
    }

    fn store_linked_headers(chain: &ChainState<FlatChainStore>, times: &[u32]) -> Vec<BlockHeader> {
        let genesis = genesis_block(Network::Regtest);
        let mut prev_hash = genesis.block_hash();
        let mut headers = Vec::with_capacity(times.len());

        for (height, &time) in (1u32..).zip(times) {
            let header = BlockHeader {
                version: HeaderVersion::TWO,
                prev_blockhash: prev_hash,
                merkle_root: genesis.header.merkle_root,
                time,
                bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
                nonce: height,
            };
            prev_hash = header.block_hash();
            write_lock!(chain)
                .chainstore
                .save_header(&DiskBlockHeader::FullyValid(header, height))
                .unwrap();
            write_lock!(chain)
                .chainstore
                .update_block_index(height, prev_hash)
                .unwrap();
            headers.push(header);
        }

        headers
    }

    fn median_time(mut times: Vec<u32>) -> u32 {
        times.sort_unstable();
        times[times.len() / 2]
    }

    #[test]
    fn chain_state_median_time_past_sorts_recent_headers() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let times = [111, 101, 109, 107, 103, 105, 110, 102, 108, 104, 106];
        let headers = store_linked_headers(&chain, &times);

        assert_eq!(
            chain.median_time_past(headers[10]).unwrap(),
            median_time(times.to_vec())
        );
    }

    #[test]
    fn chain_state_median_time_past_requires_connected_headers() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let missing_prev = BlockHeader {
            version: HeaderVersion::TWO,
            prev_blockhash: genesis_block(Network::Regtest).block_hash(),
            merkle_root: genesis_block(Network::Regtest).header.merkle_root,
            time: 100,
            bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
            nonce: 1,
        };
        let header = BlockHeader {
            version: HeaderVersion::TWO,
            prev_blockhash: missing_prev.block_hash(),
            merkle_root: genesis_block(Network::Regtest).header.merkle_root,
            time: 200,
            bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
            nonce: 2,
        };

        assert!(matches!(
            chain.median_time_past(header),
            Err(BlockchainError::BlockNotPresent)
        ));
    }

    #[test]
    fn chain_state_previous_median_time_past_uses_parent_hash() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let times = [211, 201, 209, 207, 203, 205, 210, 202, 208, 204, 206];
        let headers = store_linked_headers(&chain, &times);

        let candidate = BlockHeader {
            version: HeaderVersion::TWO,
            prev_blockhash: headers[10].block_hash(),
            merkle_root: genesis_block(Network::Regtest).header.merkle_root,
            time: 100,
            bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
            nonce: 12,
        };

        assert_eq!(
            chain.previous_median_time_past(&candidate).unwrap(),
            median_time(times.to_vec())
        );
    }

    const TESTNET_POW_LIMIT_BITS: u32 = 0x1d00_ffff;
    const TESTNET_HARD_BITS: u32 = 0x1c00_ffff;

    /// Stores headers at heights `1..=bits.len()` on `network`, spaced 10 minutes from genesis.
    /// Returns them indexed by height, genesis included.
    fn store_headers_with_bits(
        chain: &ChainState<FlatChainStore>,
        network: Network,
        bits: &[u32],
    ) -> Vec<BlockHeader> {
        let genesis = genesis_block(network).header;
        let mut headers = vec![genesis];

        for (height, &bits) in (1u32..).zip(bits) {
            let header = BlockHeader {
                version: HeaderVersion::TWO,
                prev_blockhash: headers[height as usize - 1].block_hash(),
                merkle_root: genesis.merkle_root,
                time: genesis.time + 600 * height,
                bits: CompactTarget::from_consensus(bits),
                nonce: height,
            };
            write_lock!(chain)
                .chainstore
                .save_header(&DiskBlockHeader::FullyValid(header, height))
                .unwrap();
            write_lock!(chain)
                .chainstore
                .update_block_index(height, header.block_hash())
                .unwrap();
            headers.push(header);
        }

        headers
    }

    fn required_bits(
        chain: &ChainState<FlatChainStore>,
        last: &BlockHeader,
        next_height: u32,
        next_time: u32,
    ) -> u32 {
        let next = BlockHeader {
            time: next_time,
            ..*last
        };
        chain
            .get_next_required_work(last, next_height, &next)
            .unwrap()
            .to_compact_lossy()
            .to_consensus()
    }

    #[test]
    fn min_difficulty_exception_does_not_apply_at_retarget_height() {
        let chain = setup_test_chain(Network::Testnet, AssumeValidArg::Disabled, None);
        let headers = store_headers_with_bits(
            &chain,
            Network::Testnet,
            &[TESTNET_HARD_BITS; DIFFCHANGE_INTERVAL as usize - 1],
        );
        let last = headers[DIFFCHANGE_INTERVAL as usize - 1];

        // The retarget Core computes for a period of 2015 * 600 seconds
        // (checked against btclib's `next_bits_required`)
        let retargeted = 0x1c00_ffde;
        assert_eq!(
            required_bits(&chain, &last, DIFFCHANGE_INTERVAL, last.time + 20 * 60 + 1),
            retargeted
        );
        assert_eq!(
            required_bits(&chain, &last, DIFFCHANGE_INTERVAL, last.time + 10 * 60),
            retargeted
        );
    }

    #[test]
    fn min_difficulty_walks_back_to_last_regular_block() {
        let chain = setup_test_chain(Network::Testnet, AssumeValidArg::Disabled, None);
        let mut bits = [TESTNET_HARD_BITS; 12];
        bits[9..].fill(TESTNET_POW_LIMIT_BITS); // heights 10, 11 and 12
        let headers = store_headers_with_bits(&chain, Network::Testnet, &bits);

        // Within 20 minutes: the target of height 9, not of the min-difficulty tip
        let last = headers[12];
        assert_eq!(
            required_bits(&chain, &last, 13, last.time + 10 * 60),
            TESTNET_HARD_BITS
        );
        // Later than 20 minutes: the exception applies
        assert_eq!(
            required_bits(&chain, &last, 13, last.time + 20 * 60 + 1),
            TESTNET_POW_LIMIT_BITS
        );
        // A regular tip is its own answer
        assert_eq!(
            required_bits(&chain, &headers[9], 10, headers[9].time + 10 * 60),
            TESTNET_HARD_BITS
        );
    }

    #[test]
    fn min_difficulty_walk_stops_at_retarget_block() {
        let chain = setup_test_chain(Network::Testnet, AssumeValidArg::Disabled, None);
        let bits = [TESTNET_POW_LIMIT_BITS; 3];
        let headers = store_headers_with_bits(&chain, Network::Testnet, &bits);

        let last = headers[3];
        assert_eq!(
            required_bits(&chain, &last, 4, last.time + 10 * 60),
            TESTNET_POW_LIMIT_BITS
        );
    }

    /// A chain of `network` with `2015` headers at the pow limit, and a header at height 2015
    /// that is a sibling of the best tip, stored as a fork and mined an hour earlier.
    fn chain_with_fork_at_height_2015(
        network: Network,
    ) -> (ChainState<FlatChainStore>, Vec<BlockHeader>, BlockHeader) {
        let chain = setup_test_chain(network, AssumeValidArg::Disabled, None);
        let headers = store_headers_with_bits(&chain, network, &[TESTNET_POW_LIMIT_BITS; 2015]);

        let fork = BlockHeader {
            time: headers[2015].time - 3600,
            nonce: 1,
            ..headers[2015]
        };
        write_lock!(chain)
            .chainstore
            .save_header(&DiskBlockHeader::InFork(fork, 2015))
            .unwrap();

        (chain, headers, fork)
    }

    #[test]
    fn retarget_is_computed_from_the_parents_own_ancestors() {
        let (chain, headers, fork) = chain_with_fork_at_height_2015(Network::Testnet);
        let child = BlockHeader {
            prev_blockhash: fork.block_hash(),
            time: fork.time + 600,
            ..fork
        };

        // Core: `pindexLast->GetAncestor`, so the epoch ends at the fork, not at the best tip
        let on_fork = Consensus::calc_next_work_required(&fork, &headers[0], chain.chain_params());
        let on_best =
            Consensus::calc_next_work_required(&headers[2015], &headers[0], chain.chain_params());
        assert_ne!(on_fork, on_best);

        assert_eq!(
            chain
                .get_next_required_work(&fork, DIFFCHANGE_INTERVAL, &child)
                .unwrap()
                .to_compact_lossy(),
            on_fork.to_compact_lossy()
        );
    }

    #[test]
    fn bip94_time_is_checked_against_the_parent_on_a_fork() {
        let (chain, _, fork) = chain_with_fork_at_height_2015(Network::Testnet4);
        let child = BlockHeader {
            prev_blockhash: fork.block_hash(),
            time: fork.time + 10,
            ..fork
        };

        // The best tip is an hour later than the fork, which would make this child too early
        assert!(chain.check_bip94_block(&child, DIFFCHANGE_INTERVAL).is_ok());
    }

    /// A regtest child of genesis with the given bits and a valid proof of work.
    fn mined_regtest_child(bits: u32) -> BlockHeader {
        let genesis = genesis_block(Network::Regtest).header;
        let mut header = BlockHeader {
            version: HeaderVersion::TWO,
            prev_blockhash: genesis.block_hash(),
            merkle_root: genesis.merkle_root,
            time: genesis.time + 600,
            bits: CompactTarget::from_consensus(bits),
            nonce: 0,
        };
        while header.validate_pow(header.target()).is_err() {
            header.nonce += 1;
        }
        header
    }

    #[test]
    fn accept_header_rejects_harder_bits_than_required() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let genesis = genesis_block(Network::Regtest);

        // Regtest requires 0x207fffff. This target is 128 times smaller (about 256 hashes to
        // mine, against 2) and the proof of work is valid for it, but Core rejects it with
        // `bad-diffbits`.
        let header = mined_regtest_child(0x2000_ffff);

        assert!(matches!(
            chain.accept_header(header),
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::BadDiffBits
            ))
        ));
        assert_eq!(chain.get_best_block().unwrap(), (0, genesis.block_hash()));
        assert!(chain.get_disk_block_header(&header.block_hash()).is_err());
    }

    #[test]
    fn accept_header_accepts_exactly_the_required_bits() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);

        let header = mined_regtest_child(EASIEST_REGTEST_TARGET_BITS);

        chain.accept_header(header).unwrap();
        assert_eq!(chain.get_best_block().unwrap(), (1, header.block_hash()));
    }

    #[test]
    fn header_pow_is_checked_before_the_required_bits() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let genesis = genesis_block(Network::Regtest).header;

        // Wrong bits, and a proof of work that does not meet them: Core reports `high-hash`
        // from `CheckBlockHeader` before it looks at the bits.
        let mut header = BlockHeader {
            version: HeaderVersion::TWO,
            prev_blockhash: genesis.block_hash(),
            merkle_root: genesis.merkle_root,
            time: genesis.time + 600,
            bits: CompactTarget::from_consensus(0x2000_ffff),
            nonce: 0,
        };
        while header.validate_pow(header.target()).is_ok() {
            header.nonce += 1;
        }

        assert!(matches!(
            chain.accept_header(header),
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::NotEnoughPow
            ))
        ));
    }

    /// A regtest child of `parent` with the given version and a valid proof of work.
    fn mined_regtest_child_of(parent: &BlockHeader, version: i32) -> BlockHeader {
        let mut header = BlockHeader {
            version: HeaderVersion::from_consensus(version),
            prev_blockhash: parent.block_hash(),
            merkle_root: parent.merkle_root,
            time: parent.time + 600,
            bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
            nonce: 0,
        };
        while header.validate_pow(header.target()).is_err() {
            header.nonce += 1;
        }
        header
    }

    /// Checks that `accept_header` rejects version `min_version - 1` with `BadVersion` at
    /// `height` on regtest, and accepts `min_version`.
    fn assert_min_version_at(height: u32, min_version: i32) {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let headers = store_headers_with_bits(
            &chain,
            Network::Regtest,
            &vec![EASIEST_REGTEST_TARGET_BITS; height as usize - 1],
        );
        let parent = headers.last().unwrap();
        chain.update_tip(parent.block_hash(), height - 1);

        let too_old = mined_regtest_child_of(parent, min_version - 1);
        assert!(matches!(
            chain.accept_header(too_old),
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::BadVersion
            ))
        ));
        assert!(chain.get_disk_block_header(&too_old.block_hash()).is_err());

        let header = mined_regtest_child_of(parent, min_version);
        chain.accept_header(header).unwrap();
        assert_eq!(
            chain.get_best_block().unwrap(),
            (height, header.block_hash())
        );
    }

    #[test]
    fn accept_header_enforces_the_bip66_minimum_version() {
        assert_min_version_at(Params::new(Network::Regtest).bip66_height, 3);
    }

    #[test]
    fn accept_header_enforces_the_bip65_minimum_version() {
        // On regtest BIP34 activates after both, so BIP66 and BIP65 are checked on their own
        let params = Params::new(Network::Regtest);
        assert!(params.bip66_height < params.bip65_height);
        assert!(params.bip65_height < params.bip34_height);
        assert_min_version_at(params.bip65_height, 4);
    }

    #[test]
    fn accept_header_takes_any_version_before_the_activations() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let genesis = genesis_block(Network::Regtest).header;

        let v0 = mined_regtest_child_of(&genesis, 0);
        chain.accept_header(v0).unwrap();
        let negative = mined_regtest_child_of(&v0, -1);
        chain.accept_header(negative).unwrap();
        assert_eq!(chain.get_best_block().unwrap(), (2, negative.block_hash()));
    }

    #[test]
    fn ancestor_at_height_follows_a_fork() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Disabled, None);
        let headers = store_headers_with_bits(&chain, Network::Regtest, &[0x207f_ffff; 5]);

        // A fork from height 2, with different headers at heights 3, 4 and 5
        let mut forks = vec![headers[2]];
        for height in 3..=5_u32 {
            let parent = forks.last().unwrap();
            let fork = BlockHeader {
                prev_blockhash: parent.block_hash(),
                nonce: 100 + height,
                ..headers[height as usize]
            };
            write_lock!(chain)
                .chainstore
                .save_header(&DiskBlockHeader::InFork(fork, height))
                .unwrap();
            forks.push(fork);
        }
        let (fork3, fork5) = (forks[1], forks[3]);
        assert_ne!(fork3, headers[3]);

        assert_eq!(chain.get_ancestor_at_height(&fork5, 5, 3).unwrap(), fork3);
        assert_eq!(
            chain.get_ancestor_at_height(&fork5, 5, 2).unwrap(),
            headers[2]
        );
        // On the best chain it reads the height index
        assert_eq!(
            chain.get_ancestor_at_height(&headers[5], 5, 3).unwrap(),
            headers[3]
        );
    }

    #[test]
    fn accept_mainnet_headers() {
        // Accepts the first 10,237 mainnet headers
        let file = include_bytes!("../../testdata/headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(Cursor::new(file)).unwrap();
        let mut buffer = uncompressed.as_slice();

        let mut headers = Vec::new();
        let chain = setup_test_chain(Network::Bitcoin, AssumeValidArg::Hardcoded, None);

        while let Ok(header) = BlockHeader::consensus_decode(&mut buffer) {
            chain.accept_header(header).unwrap();
            headers.push(header);
        }
        assert_eq!(headers.len(), 10_237);

        // Check that our `BestChain` is correct
        assert_eq!(
            chain.get_best_block().unwrap(),
            (
                10_236,
                bhash!("000000004f74d42205b9d7ae1cb5c1591e723894a71358fce73b7e0919628161")
            ),
        );
        assert_eq!(chain.get_validation_index().unwrap(), 0);

        // Check all headers that we have
        for i in 0..=10_236 {
            let hash = chain.get_block_hash(i).unwrap();
            let header = chain.get_disk_block_header(&hash).unwrap();

            match header {
                DiskBlockHeader::FullyValid(h, height) if i == 0 => {
                    assert_eq!(height, 0);
                    assert_eq!(h.block_hash(), hash);
                    assert_eq!(
                        hash,
                        bhash!("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f")
                    );
                }
                // Non-genesis blocks have not been validated, only accepted in the best chain
                DiskBlockHeader::HeadersOnly(h, height) => {
                    assert_eq!(height, i);
                    assert_eq!(h.block_hash(), hash);
                }
                _ => panic!("Header at height {i} should be `HeadersOnly`: {header:?}"),
            }
            assert_eq!(&*header, headers.get(i as usize).unwrap());
        }

        // Check that we don't have more headers (nor indexed hashes)
        for i in 10_237..11_000 {
            match chain.get_block_hash(i) {
                Err(BlockchainError::BlockNotPresent) => {}
                _ => panic!("Should not have found a hash for height {i}"),
            }
            match chain.get_header_by_height(i) {
                Err(BlockchainError::BlockNotPresent) => {}
                _ => panic!("Should not have found a header at height {i}"),
            }
        }
    }

    #[test]
    fn accept_first_signet_headers() {
        // Accepts the first 2016 signet headers
        let file = include_bytes!("../../testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(Cursor::new(file)).unwrap();
        let mut buffer = uncompressed.as_slice();

        let chain = setup_test_chain(Network::Signet, AssumeValidArg::Hardcoded, None);
        while let Ok(header) = BlockHeader::consensus_decode(&mut buffer) {
            chain.accept_header(header).unwrap();
        }
    }

    #[test]
    fn test_calc_next_work_required() {
        let first_block: BlockHeader = deserialize_hex("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a008f4d5fae77031e8ad22203").unwrap();
        let last_block: BlockHeader = deserialize_hex("00000020dec6741f7dc5df6661bcb2d3ec2fceb14bd0e6def3db80da904ed1eeb8000000d1f308132e6a72852c04b059e92928ea891ae6d513cd3e67436f908c804ec7be51df535fae77031e4d00f800").unwrap();

        let next_target = Consensus::calc_next_work_required(
            &last_block,
            &first_block,
            ChainParams::from(Network::Signet),
        );

        assert_eq!(0x1e012fa7, next_target.to_compact_lossy().to_consensus());
    }

    #[test]
    fn test_calc_next_work_required_underflow() {
        let first_block: BlockHeader = deserialize_hex("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a008f4d5fae77031e8ad22203").unwrap();
        let last_block: BlockHeader = deserialize_hex("00000020dec6741f7dc5df6661bcb2d3ec2fceb14bd0e6def3db80da904ed1eeb8000000d1f308132e6a72852c04b059e92928ea891ae6d513cd3e67436f908c804ec7be51df535fae77031e4d00f800").unwrap();

        // The "first" block has a timestamp later than the "last" one, so computing the
        // timespan `last.time - first.time` should underflow.
        //
        // The saturating subtraction yields a timespan of 0, which gets clamped to
        // `pow_target_timespan / 4`, quartering the target.
        let next_target = Consensus::calc_next_work_required(
            &first_block,
            &last_block,
            ChainParams::from(Network::Signet),
        );

        assert!((first_block.time as i32 - last_block.time as i32) < 0);

        assert_eq!(0x1e00ddeb, next_target.to_compact_lossy().to_consensus());

        let mut new_last = last_block;
        new_last.time = first_block.time;

        assert!(first_block.time - new_last.time == 0);

        // Since the time value is saturated and clamped, them being exactly equal should
        // make the result to be the same.
        let next_target = Consensus::calc_next_work_required(
            &first_block,
            &new_last,
            ChainParams::from(Network::Signet),
        );

        assert_eq!(0x1e00ddeb, next_target.to_compact_lossy().to_consensus());
    }

    #[test]
    fn test_reorg() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();
        let mut fork_acc = Stump::default();

        let parse_blocks = |blocks: &[&str]| {
            blocks
                .iter()
                .map(|s| deserialize_hex(s).unwrap())
                .collect::<Vec<Block>>()
        };

        let short_chain = parse_blocks(&blocks[0]);
        let long_chain = parse_blocks(&blocks[1]);

        assert_eq!(
            short_chain[0].header.prev_blockhash,
            genesis_block(Network::Regtest).block_hash(),
            "The first block we have is the next after genesis",
        );
        assert_eq!(short_chain.len(), 10, "First 10 blocks");
        assert_eq!(long_chain.len(), 11, "Next 11 fork blocks after the fifth");
        assert_eq!(
            long_chain[0].header.prev_blockhash,
            short_chain[4].block_hash(),
        );

        // Connect the first 10 blocks after genesis
        for block in short_chain {
            chain.accept_header(block.header).unwrap();
            chain
                .connect_block(&block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();

            if block.block_hash()
                == bhash!("45c74beefa2a110715377e023d4260168b4cafbb0891f3b0869aea30867acc87")
            {
                // This is the block we will reorg to
                fork_acc = chain.acc();
            }
        }

        let expected = (
            10,
            bhash!("6e9c49a19038f7db8d13f6c2e70566385536ea11975528b557799e08a014e784"),
        );

        assert_eq!(chain.get_best_block().unwrap(), expected);

        // Then accept a fork chain with 11 new blocks, building on the previous height 5 block
        for fork_block in long_chain.iter() {
            chain.accept_header(fork_block.header).unwrap();
        }

        let expected = (
            16,
            bhash!("4572ac401b94915dde6c4957b706abdb13b5824b000cad7f6065ebd9aea6dad1"),
        );

        assert_eq!(chain.get_best_block().unwrap(), expected);
        assert_eq!(
            chain.acc(),
            fork_acc,
            "The accumulator should not change when accepting headers only",
        );

        // Actually connect the fork chain
        for fork in long_chain {
            chain
                .connect_block(&fork, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
        }

        for i in 1..=chain.get_height().unwrap() {
            let hash = chain.get_block_hash(i).unwrap();
            let header = chain.get_disk_block_header(&hash).unwrap();
            let header_by_height = chain.get_header_by_height(i).unwrap();

            assert_eq!(header, header_by_height);
            assert_eq!(header.prev_blockhash, chain.get_block_hash(i - 1).unwrap());

            if let DiskBlockHeader::FullyValid(..) = header {
                continue;
            } else {
                panic!("Expected block at height {i} to be FullyValid, got: {header:?}");
            }
        }
    }

    #[test]
    fn open_resumes_existing_chain_state() {
        let file = include_bytes!("../../testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(Cursor::new(file)).unwrap();
        let mut buffer = uncompressed.as_slice();

        let test_id = rand::random::<u64>();
        let path = format!("./tmp-db/{test_id}/");

        let config = crate::FlatChainStoreConfig {
            block_index_size: Some(32_768),
            headers_file_size: Some(32_768),
            fork_file_size: Some(10_000),
            cache_size: Some(10),
            file_permission: Some(0o660),
            path: path.clone().into(),
        };
        let chain = ChainState::open(
            crate::FlatChainStore::new(config).unwrap(),
            Network::Signet,
            AssumeValidArg::Hardcoded,
        )
        .unwrap();

        // Accept 10 headers; accept_header updates inner.best_block
        let mut count = 0;
        while let Ok(header) = BlockHeader::consensus_decode(&mut buffer) {
            chain.accept_header(header).unwrap();
            count += 1;
            if count == 10 {
                break;
            }
        }

        let height = chain.get_height().unwrap();
        assert!(height > 0, "chain did not advance past genesis");
        chain.flush().unwrap();
        drop(chain);

        // Reopen: must resume at the same height, not reset to genesis
        let config2 = crate::FlatChainStoreConfig {
            block_index_size: Some(32_768),
            headers_file_size: Some(32_768),
            fork_file_size: Some(10_000),
            cache_size: Some(10),
            file_permission: Some(0o660),
            path: path.into(),
        };
        let chain2 = ChainState::open(
            crate::FlatChainStore::new(config2).unwrap(),
            Network::Signet,
            AssumeValidArg::Hardcoded,
        )
        .unwrap();
        assert_eq!(chain2.get_height().unwrap(), height);
    }

    #[test]
    fn test_fork_tips() {
        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();

        let parse_blocks = |blocks: &[&str]| {
            blocks
                .iter()
                .map(|s| deserialize_hex(s).unwrap())
                .collect::<Vec<Block>>()
        };

        let short_chain = parse_blocks(&blocks[0]); // 10 blocks on genesis
        let long_chain = parse_blocks(&blocks[1]); // 11 blocks forking from block 5

        // Accept all 10 main-chain headers
        for block in &short_chain {
            chain.accept_header(block.header).unwrap();
        }

        // Only one tip so far (the main chain)
        let tips = chain.get_chain_tips().unwrap();
        assert_eq!(tips.len(), 1);

        // Accept only 4 fork headers (less work than main chain: 4 < 5 blocks after fork point)
        for block in &long_chain[..4] {
            chain.accept_header(block.header).unwrap();
        }

        // Now we should have 2 tips: main chain + fork
        let tips = chain.get_chain_tips().unwrap();
        assert_eq!(tips.len(), 2);

        // First tip is always the best block (main chain tip at height 10)
        let (best_height, best_hash) = chain.get_best_block().unwrap();
        assert_eq!(best_height, 10);
        assert_eq!(tips[0], best_hash);

        // Second tip is the fork tip (4th fork block)
        assert_eq!(tips[1], long_chain[3].block_hash());
    }

    #[test]
    fn test_chainstate_functions() {
        let file = include_bytes!("../../testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(Cursor::new(file)).unwrap();
        let mut buffer = uncompressed.as_slice();

        let chain = setup_test_chain(Network::Signet, AssumeValidArg::Hardcoded, None);
        let mut headers: Vec<BlockHeader> = Vec::new();
        while let Ok(header) = BlockHeader::consensus_decode(&mut buffer) {
            headers.push(header);
        }

        headers.remove(0);

        // push_headers
        assert_ok!(chain.push_headers(headers.clone(), 1));

        // get_header_by_height
        assert_eq!(*chain.get_header_by_height(1).unwrap(), headers[0]);

        // find_best_chain
        assert_eq!(chain.find_best_chain().depth, 2015);

        // get_block_locator_for_tip
        assert!(
            !chain
                .get_block_locator_for_tip(read_lock!(chain).best_block.best_block)
                .unwrap()
                .is_empty()
        );

        // get_block_locator
        assert!(!chain.get_block_locator().unwrap().is_empty());

        // invalidate_block
        let random_height = rand::rng().random_range(1..=2014);

        chain
            .invalidate_block(headers[random_height].prev_blockhash)
            .unwrap();

        assert_eq!(chain.get_height().unwrap() as usize, random_height - 1);

        // update_tip
        chain.update_tip(headers[1].prev_blockhash, 1);
        assert_eq!(
            read_lock!(chain).best_block.best_block,
            headers[1].prev_blockhash
        );
    }

    #[test]
    fn test_calculate_chain_work() {
        let mut chainstore = FlatChainStore::new(FlatChainStoreConfig::new(
            "../../testdata/signet_headers.zst",
        ))
        .unwrap();

        let mut headers = vec![];
        let genesis_header: BlockHeader = deserialize_hex("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c").unwrap();
        let mut prev_blockhash = genesis_header.block_hash();

        chainstore
            .save_header(&DiskBlockHeader::FullyValid(genesis_header, 0))
            .unwrap();

        chainstore.update_block_index(0, prev_blockhash).unwrap();
        headers.push(genesis_header);

        for i in 1..3_000 {
            let header = BlockHeader {
                time: 1231006505 + i * 600,
                prev_blockhash,
                ..genesis_header
            };

            headers.push(header);
            let hash = header.block_hash();

            chainstore
                .save_header(&DiskBlockHeader::FullyValid(header, i))
                .unwrap();
            chainstore.update_block_index(i, hash).unwrap();

            prev_blockhash = header.block_hash();
        }

        chainstore
            .save_height(&crate::BestChain {
                best_block: prev_blockhash,
                depth: 3_000,
                validation_index: prev_blockhash,
                alternative_tips: vec![],
            })
            .unwrap();

        let chainstate = ChainState::new(chainstore, Network::Bitcoin, AssumeValidArg::Disabled);
        let header = headers[headers.len() - 1];
        let fork = headers[headers.len() / 2];

        let work = chainstate
            .get_work(header.block_hash())
            .expect("Failed to calculate chain work");

        let fork_work = chainstate
            .get_fork_work(header, fork.block_hash())
            .expect("Failed to calculate fork work");

        let expected_hex_string =
            "00000000000000000000000000000000000000000000000000000bb80bb80bb8";
        let expected_work = Work::from_hex(&format!("0x{expected_hex_string}")).unwrap();

        assert_eq!(work.to_string_hex(), expected_hex_string);
        assert_eq!(fork_work, work);
        assert_eq!(work, expected_work);
    }

    fn connect_reorg_chains(
        chain: &ChainState<FlatChainStore>,
        short_chain: &[Block],
        long_chain: &[Block],
    ) -> HashMap<BlockHash, Stump> {
        let mut accs = HashMap::new();
        accs.insert(chain.get_block_hash(0).unwrap(), chain.acc());

        for block in short_chain {
            chain.accept_header(block.header).unwrap();
            chain
                .connect_block(block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
            accs.insert(block.block_hash(), chain.acc());
        }

        for block in long_chain {
            chain.accept_header(block.header).unwrap();
        }

        for block in long_chain {
            chain
                .connect_block(block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
            accs.insert(block.block_hash(), chain.acc());
        }

        accs
    }

    #[test]
    fn reorg_publishes_the_validation_index_and_the_accumulator_together() {
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();

        let parse_blocks = |blocks: &[&str]| {
            blocks
                .iter()
                .map(|s| deserialize_hex(s).unwrap())
                .collect::<Vec<Block>>()
        };

        let short_chain = parse_blocks(&blocks[0]);
        let long_chain = parse_blocks(&blocks[1]);

        let reference = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);
        let accs = connect_reorg_chains(&reference, &short_chain, &long_chain);

        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);
        let done = AtomicBool::new(false);

        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    while !done.load(Ordering::Acquire) {
                        let (validation_index, acc) = {
                            let inner = chain.inner.read();
                            (inner.best_block.validation_index, inner.acc.clone())
                        };

                        assert_eq!(
                            accs.get(&validation_index),
                            Some(&acc),
                            "the accumulator must be the one left by block {validation_index}",
                        );
                    }
                });
            }

            connect_reorg_chains(&chain, &short_chain, &long_chain);
            done.store(true, Ordering::Release);
        });

        assert_eq!(chain.get_validation_index().unwrap(), 16);
    }

    #[test]
    fn reorg_above_the_validation_index_keeps_the_accumulator() {
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();

        let parse_blocks = |blocks: &[&str]| {
            blocks
                .iter()
                .map(|s| deserialize_hex(s).unwrap())
                .collect::<Vec<Block>>()
        };

        let short_chain = parse_blocks(&blocks[0]);
        let long_chain = parse_blocks(&blocks[1]);

        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);

        // Take the headers all the way to the tip of the short chain, but only validate the
        // first four blocks. The fork point, block 5, is left as `HeadersOnly`, which is the
        // ordinary state during IBD: headers run ahead of block validation.
        for block in &short_chain {
            chain.accept_header(block.header).unwrap();
        }

        for block in short_chain.iter().take(4) {
            chain
                .connect_block(block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
        }

        assert_eq!(chain.get_validation_index().unwrap(), 4);

        let acc = chain.acc();
        assert_ne!(acc, Stump::new(), "we validated four blocks");

        // The long chain forks at block 5, above our validation index, so the reorg doesn't
        // undo any validated block and must leave the accumulator alone.
        for block in &long_chain {
            chain.accept_header(block.header).unwrap();
        }

        assert_eq!(chain.get_validation_index().unwrap(), 4);
        assert_eq!(
            chain.acc(),
            acc,
            "a reorg above the validation index must not change the accumulator",
        );
    }

    #[test]
    fn a_failed_reorg_leaves_the_chain_state_untouched() {
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();

        let parse_blocks = |blocks: &[&str]| {
            blocks
                .iter()
                .map(|s| deserialize_hex(s).unwrap())
                .collect::<Vec<Block>>()
        };

        let short_chain = parse_blocks(&blocks[0]);
        let long_chain = parse_blocks(&blocks[1]);

        let chain = setup_test_chain(Network::Regtest, AssumeValidArg::Hardcoded, None);

        for block in &short_chain {
            chain.accept_header(block.header).unwrap();
        }

        for block in short_chain.iter().take(4) {
            chain
                .connect_block(block, Proof::default(), HashMap::new(), Vec::new())
                .unwrap();
        }

        // `mark_chain_as_assumed` takes a whole range as `FullyValid` but only saves the roots
        // for genesis, so every assumed block above our validation index has none. This is the
        // state an assume-utreexo node runs in, and a reorg landing on one of those blocks
        // can't find the accumulator it has to publish.
        let acc = chain.acc();
        chain
            .mark_chain_as_assumed(acc.clone(), short_chain[9].block_hash())
            .unwrap();

        assert!(chain.get_roots_for_block(5).unwrap().is_none());

        let best_block = chain.get_best_block().unwrap();
        let validation_index = chain.get_validation_index().unwrap();
        let block_after_fork = chain.get_block_hash(6).unwrap();

        // The long chain forks at block 5, one of the assumed blocks, so this reorg fails when
        // it looks for the accumulator that goes with the new validation index.
        let error = long_chain
            .iter()
            .find_map(|block| chain.accept_header(block.header).err());

        assert!(
            matches!(error, Some(BlockchainError::BadValidationIndex)),
            "expected the reorg to fail with BadValidationIndex, got {error:?}",
        );

        // A reorg that fails must not leave half of itself behind: the branches keep the state
        // they had, and the tip, validation index and accumulator still describe the old chain.
        assert_eq!(chain.get_best_block().unwrap(), best_block);
        assert_eq!(
            chain.get_validation_index().ok(),
            Some(validation_index),
            "the validation index must still point to a block we validated",
        );
        assert_eq!(chain.get_block_hash(6).unwrap(), block_after_fork);
        assert_eq!(chain.acc(), acc);
    }

    #[test]
    fn an_assumed_chain_finds_its_accumulator_again_after_a_restart() {
        let json_blocks = include_str!("../../testdata/test_reorg.json");
        let blocks: Vec<Vec<&str>> = serde_json::from_str(json_blocks).unwrap();
        let chain_blocks: Vec<Block> = blocks[0]
            .iter()
            .map(|s| deserialize_hex(s).unwrap())
            .collect();

        let test_id = rand::random::<u64>();
        let path = format!("./tmp-db/{test_id}/");
        let config = || FlatChainStoreConfig {
            block_index_size: Some(DEFAULT_TEST_CHAINSTORE_SIZE),
            headers_file_size: Some(DEFAULT_TEST_CHAINSTORE_SIZE),
            fork_file_size: Some(TEST_FORK_FILE_SIZE),
            cache_size: Some(10),
            file_permission: Some(0o660),
            path: path.clone().into(),
        };

        let acc = Stump {
            leaves: 42,
            roots: vec![BitcoinNodeHash::Some([1; 32]); 3],
        };

        let chain = ChainState::open(
            FlatChainStore::new(config()).unwrap(),
            Network::Regtest,
            AssumeValidArg::Hardcoded,
        )
        .unwrap();

        for block in &chain_blocks {
            chain.accept_header(block.header).unwrap();
        }

        let assumed_hash = chain_blocks[9].block_hash();
        chain
            .mark_chain_as_assumed(acc.clone(), assumed_hash)
            .unwrap();

        // The roots have to sit at the height the validation index points to, because that's
        // where `open` looks for them.
        let assumed_height = chain.get_validation_index().unwrap();
        assert_eq!(assumed_height, 10);
        assert_eq!(
            chain.get_roots_for_block(assumed_height).unwrap(),
            Some(acc.clone())
        );

        drop(chain);

        let chain = ChainState::open(
            FlatChainStore::new(config()).unwrap(),
            Network::Regtest,
            AssumeValidArg::Hardcoded,
        )
        .unwrap();

        assert_eq!(chain.get_validation_index().unwrap(), assumed_height);
        assert_eq!(
            chain.acc(),
            acc,
            "a restart must not throw the assumed accumulator away",
        );
    }
}
