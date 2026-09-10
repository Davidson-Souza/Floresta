// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pruned utreexo module handles the full blockchain logic: validation, state tracking and
//! interfacing. This blockchain backend does not store the historical blocks, it's pruned.
//!
//! This module file defines the main traits for an utreexo-enabled chain backend:
//!
//! - [BlockchainInterface]: The main interface for interacting with the backend
//! - [UpdatableChainstate]: Trait defining methods for updating the chain state

extern crate alloc;

pub mod chain_state;
pub mod chain_state_builder;
pub mod chainparams;
pub mod chainstore;
#[macro_use]
pub mod error;
pub mod consensus;
#[cfg(feature = "flat-chainstore")]
pub mod flat_chain_store;
pub mod merkle;
pub mod partial_chain;
pub mod udata;

use alloc::sync::Arc;
use core::error::Error;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::OutPoint;
use bitcoin::Work;
use bitcoin::block::Header as BlockHeader;
use bitcoin::hashes::sha256;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::proof::Proof;
use rustreexo::stump::Stump;

use self::chainstore::ChainStoreWarning;
use self::partial_chain::PartialChainState;
use crate::BlockConsumer;
use crate::BlockchainError;
use crate::prelude::*;
use crate::pruned_utreexo::utxo_data::UtxoData;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
/// Our current IBD state, meaning which startup phase are we, if any.
///
/// During startup, our node will go through a bootstrap process called Initial Block Download,
/// where it will catch up with the network. This enum is a simple state machine that represents
/// which state we are currently in.
pub enum IBDState {
    #[default]
    /// Downloading headers to establish which is the most-work chain.
    ///
    /// During this phase, we only download and check headers. We will finish when we are convinced
    /// this is the most work chain available.
    HeadersSync,

    /// Downloading and checking blocks.
    ///
    /// After we find the most work chain, we start downloading blocks and connecting them to our
    /// chain. This step usually takes the longest time.
    DownloadingBlocks,

    /// We've finished IBD and are now listening for new blocks as they are found.
    Done,
}

/// This trait is the main interface between our blockchain backend and other services.
/// It'll be useful for transitioning from rpc to a p2p based node
pub trait BlockchainInterface {
    type Error: Error + Send + Sync + 'static;

    /// Returns the block with a given height in our current tip.
    fn get_block_hash(&self, height: u32) -> Result<bitcoin::BlockHash, Self::Error>;

    /// Returns a bitcoin [`bitcoin::Transaction`] given its txid.
    fn get_tx(&self, txid: &bitcoin::Txid) -> Result<Option<bitcoin::Transaction>, Self::Error>;

    /// Get the height of our best know chain.
    fn get_height(&self) -> Result<u32, Self::Error>;

    /// Returns fee estimation for inclusion in `target` blocks.
    fn estimate_fee(&self, target: usize) -> Result<f64, Self::Error>;

    /// Returns a block with a given `hash` if any.
    fn get_block(&self, hash: &BlockHash) -> Result<Block, Self::Error>;

    /// Returns the best known block
    fn get_best_block(&self) -> Result<(u32, BlockHash), Self::Error>;

    /// Returns associated header for block with `hash`
    fn get_block_header(&self, hash: &BlockHash) -> Result<BlockHeader, Self::Error>;

    /// Register for receiving notifications for some event. Right now it only works for
    /// new blocks, but may work with transactions in the future too.
    /// if a module performs some heavy-lifting on the block's data, it should pass in a
    /// vector or a channel where data can be transferred to the actual worker, otherwise
    /// chainstate will be stuck for as long as you have work to do.
    fn subscribe(&self, tx: Arc<dyn BlockConsumer>);

    /// Tells whether or not we are on IBD
    fn is_in_ibd(&self) -> bool;

    /// Checks if a coinbase is mature
    fn is_coinbase_mature(&self, height: u32, block: BlockHash) -> Result<bool, Self::Error>;

    /// Returns a block locator
    fn get_block_locator(&self) -> Result<Vec<BlockHash>, Self::Error>;

    /// Returns a block locator from a given tip
    ///
    /// This method may be used to get the locator from a tip that's not the best one
    fn get_block_locator_for_tip(&self, tip: BlockHash) -> Result<Vec<BlockHash>, BlockchainError>;

    /// Returns the last block we validated
    fn get_validation_index(&self) -> Result<u32, Self::Error>;

    /// Returns the height of a block, given it's hash
    fn get_block_height(&self, hash: &BlockHash) -> Result<Option<u32>, Self::Error>;

    /// Applies a changeset to an accumulator and returns the resulting one
    fn update_acc(
        &self,
        acc: Stump,
        block: &Block,
        height: u32,
        proof: Proof,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<Stump, Self::Error>;

    /// Returns all known chain tips, including the best one and forks
    fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error>;

    /// Validates a block according to Bitcoin's rules, without modifying our chain
    fn validate_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
        acc: Stump,
    ) -> Result<(), Self::Error>;

    /// Find the last common ancestor between the current best chain and `block`
    fn get_fork_point(&self, block: BlockHash) -> Result<BlockHash, Self::Error>;

    /// Returns this chain's params
    fn get_params(&self) -> bitcoin::params::Params;

    /// Returns our current acc
    fn acc(&self) -> Stump;

    /// Returns the amount of [`Work`] associated with a given chain tip
    fn get_work(&self, tip: BlockHash) -> Result<Work, Self::Error>;

    /// Returns the total size on disk, in bytes, of the chain data persisted by this backend.
    fn size_on_disk(&self) -> Result<u64, Self::Error>;

    /// Returns the current state of our chain.
    fn ibd_state(&self) -> IBDState;

    /// Returns accumulated chain-store health warnings (e.g. index full).
    ///
    /// Warnings persist for the process lifetime. Mirrors the `warnings` field returned by
    /// Bitcoin Core's `getblockchaininfo`.
    fn get_warnings(&self) -> Vec<ChainStoreWarning> {
        vec![]
    }
}

/// [UpdatableChainstate] is a contract that a is expected from a chainstate
/// implementation, that wishes to be updated. Using those methods, a backend like the p2p-node,
/// can notify new blocks and transactions to a chainstate, allowing it to update it's state.
pub trait UpdatableChainstate {
    /// This is one of the most important methods for a ChainState,
    /// it gets a block and some utreexo data, validates this block and
    /// connects to our chain of blocks. This function is meant to be atomic
    /// and prone of running in parallel.
    fn connect_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<u32, BlockchainError>;

    fn switch_chain(&self, new_tip: BlockHash) -> Result<(), BlockchainError>;
    /// Accepts a new header to our chain. This method is called before connect_block, and
    /// makes some basic checks on a header and saves it on disk. We only accept a block as
    /// valid after calling connect_block.
    ///
    /// This function returns whether this block is on our best-known chain, or in a fork
    fn accept_header(&self, header: BlockHeader) -> Result<(), BlockchainError>;
    /// Inserts a prevalidated contiguous header range directly at `height`.
    ///
    /// This bypasses `accept_header`; callers must already have verified proof of work,
    /// difficulty transitions, continuity, and the range's committed endpoint.
    fn push_headers(&self, headers: Vec<BlockHeader>, height: u32) -> Result<(), BlockchainError>;
    /// Not used for now, but in a future blockchain with mempool, we can process transactions
    /// that are not in a block yet.
    fn handle_transaction(&self) -> Result<(), BlockchainError>;
    /// Persists our data. Should be invoked periodically.
    fn flush(&self) -> Result<(), BlockchainError>;
    /// Update IBD state
    fn update_ibd(&self, ibd_state: IBDState);
    /// Tells this blockchain to consider this block invalid, and not build on top of it
    fn invalidate_block(&self, block: BlockHash) -> Result<(), BlockchainError>;
    /// Marks one block as being fully validated, this overrides a block that was explicitly
    /// marked as invalid.
    fn mark_block_as_valid(&self, block: BlockHash) -> Result<(), BlockchainError>;
    /// Returns the root hashes of our utreexo forest
    fn get_root_hashes(&self) -> Vec<BitcoinNodeHash>;
    /// Returns a partial chainstate from a range of blocks.
    ///
    /// [PartialChainState] is a simplified version of `ChainState` that is used during IBD.
    /// It doesn't support reorgs, only hold headers for a subset of blocks and isn't [Sync].
    /// The idea here is that you take a OS thread or some async task that will drive one
    /// [PartialChainState] to completion by downloading blocks inside that chainstate's range.
    /// If all goes right, it'll end without error, and you should mark blocks in this range as
    /// valid.
    /// Since this chainstate may start from a height with an existing UTXO set, you need to
    /// provide a [Stump] for that block.
    fn get_partial_chain(
        &self,
        initial_height: u32,
        final_height: u32,
        acc: Stump,
    ) -> Result<PartialChainState, BlockchainError>;
    /// Marks a chain as fully-valid
    ///
    /// This mimics the behaviour of checking every block before this block, and continues
    /// from this point
    fn mark_chain_as_assumed(&self, acc: Stump, tip: BlockHash) -> Result<bool, BlockchainError>;
    /// Returns the current accumulator
    fn get_acc(&self) -> Stump;
}

#[derive(Debug, Clone)]
/// A notification is a hook that a type implementing [BlockchainInterface] sends each
/// time the given event happens. This is use to notify new blocks to the Electrum server.
/// In the future, it can be expanded to send more data, like transactions.
pub enum Notification {
    NewBlock((Block, u32)),
}

impl<T: UpdatableChainstate> UpdatableChainstate for Arc<T> {
    fn flush(&self) -> Result<(), BlockchainError> {
        T::flush(self)
    }

    fn get_acc(&self) -> Stump {
        T::get_acc(self)
    }

    fn update_ibd(&self, ibd_state: IBDState) {
        T::update_ibd(self, ibd_state)
    }

    fn connect_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<u32, BlockchainError> {
        T::connect_block(self, block, proof, inputs, del_hashes)
    }

    fn accept_header(&self, header: BlockHeader) -> Result<(), BlockchainError> {
        T::accept_header(self, header)
    }

    fn push_headers(&self, headers: Vec<BlockHeader>, height: u32) -> Result<(), BlockchainError> {
        T::push_headers(self, headers, height)
    }

    fn get_root_hashes(&self) -> Vec<BitcoinNodeHash> {
        T::get_root_hashes(self)
    }

    fn invalidate_block(&self, block: BlockHash) -> Result<(), BlockchainError> {
        T::invalidate_block(self, block)
    }

    fn get_partial_chain(
        &self,
        initial_height: u32,
        final_height: u32,
        acc: Stump,
    ) -> Result<PartialChainState, BlockchainError> {
        T::get_partial_chain(self, initial_height, final_height, acc)
    }

    fn handle_transaction(&self) -> Result<(), BlockchainError> {
        T::handle_transaction(self)
    }

    fn switch_chain(&self, new_tip: BlockHash) -> Result<(), BlockchainError> {
        T::switch_chain(self, new_tip)
    }

    fn mark_block_as_valid(&self, block: BlockHash) -> Result<(), BlockchainError> {
        T::mark_block_as_valid(self, block)
    }

    fn mark_chain_as_assumed(&self, acc: Stump, tip: BlockHash) -> Result<bool, BlockchainError> {
        T::mark_chain_as_assumed(self, acc, tip)
    }
}

impl<T: BlockchainInterface> BlockchainInterface for Arc<T> {
    type Error = <T as BlockchainInterface>::Error;

    fn get_work(&self, tip: BlockHash) -> Result<Work, Self::Error> {
        T::get_work(self, tip)
    }

    fn get_tx(&self, txid: &bitcoin::Txid) -> Result<Option<bitcoin::Transaction>, Self::Error> {
        T::get_tx(self, txid)
    }

    fn get_params(&self) -> bitcoin::params::Params {
        T::get_params(self)
    }

    fn acc(&self) -> Stump {
        T::acc(self)
    }

    fn get_block(&self, hash: &BlockHash) -> Result<Block, Self::Error> {
        T::get_block(self, hash)
    }

    fn subscribe(&self, tx: Arc<dyn BlockConsumer>) {
        T::subscribe(self, tx)
    }

    fn is_in_ibd(&self) -> bool {
        T::is_in_ibd(self)
    }

    fn get_height(&self) -> Result<u32, Self::Error> {
        T::get_height(self)
    }

    fn estimate_fee(&self, target: usize) -> Result<f64, Self::Error> {
        T::estimate_fee(self, target)
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
        T::get_block_hash(self, height)
    }

    fn get_best_block(&self) -> Result<(u32, BlockHash), Self::Error> {
        T::get_best_block(self)
    }

    fn get_block_header(&self, hash: &BlockHash) -> Result<BlockHeader, Self::Error> {
        T::get_block_header(self, hash)
    }

    fn get_block_height(&self, hash: &BlockHash) -> Result<Option<u32>, Self::Error> {
        T::get_block_height(self, hash)
    }

    fn get_block_locator(&self) -> Result<Vec<BlockHash>, Self::Error> {
        T::get_block_locator(self)
    }

    fn is_coinbase_mature(&self, height: u32, block: BlockHash) -> Result<bool, Self::Error> {
        T::is_coinbase_mature(self, height, block)
    }

    fn get_validation_index(&self) -> Result<u32, Self::Error> {
        T::get_validation_index(self)
    }

    fn get_block_locator_for_tip(&self, tip: BlockHash) -> Result<Vec<BlockHash>, BlockchainError> {
        T::get_block_locator_for_tip(self, tip)
    }

    fn update_acc(
        &self,
        acc: Stump,
        block: &Block,
        height: u32,
        proof: Proof,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<Stump, Self::Error> {
        T::update_acc(self, acc, block, height, proof, del_hashes)
    }

    fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error> {
        T::get_chain_tips(self)
    }

    fn validate_block(
        &self,
        block: &Block,
        proof: Proof,
        inputs: HashMap<OutPoint, UtxoData>,
        del_hashes: Vec<sha256::Hash>,
        acc: Stump,
    ) -> Result<(), Self::Error> {
        T::validate_block(self, block, proof, inputs, del_hashes, acc)
    }

    fn get_fork_point(&self, block: BlockHash) -> Result<BlockHash, Self::Error> {
        T::get_fork_point(self, block)
    }

    fn size_on_disk(&self) -> Result<u64, Self::Error> {
        T::size_on_disk(self)
    }

    fn ibd_state(&self) -> IBDState {
        T::ibd_state(self)
    }

    fn get_warnings(&self) -> Vec<ChainStoreWarning> {
        T::get_warnings(self)
    }
}

/// This module defines an [UtxoData] struct, helpful for transaction validation
pub mod utxo_data {
    use bitcoin::TxOut;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(
        any(test, feature = "test-utils"),
        derive(serde::Serialize, serde::Deserialize)
    )]
    /// Represents an unspent transaction output (UTXO) with additional metadata for validation.
    pub struct UtxoData {
        /// The unspent transaction output.
        pub txout: TxOut,
        /// Whether this output was created by a coinbase transaction.
        pub is_coinbase: bool,
        /// The block height at which the UTXO was confirmed.
        pub creation_height: u32,
        /// The creation time of the UTXO, defined by BIP 68 as the median time past (MTP) of the
        /// block preceding the confirming block.
        pub creation_time: u32,
    }
}

/// [`ChainBackend`] is a trait alias for the [`BlockchainInterface`] and [`UpdatableChainstate`] combo meant to be used
/// to specify a generic blockchain backend.
///
/// Useful to avoid trait bounds verbosity.
pub trait ChainBackend: BlockchainInterface + UpdatableChainstate {}

impl<T: BlockchainInterface + UpdatableChainstate> ChainBackend for T {}

/// [`ThreadSafeChain`] is a trait alias for the [`BlockchainInterface`], [`UpdatableChainstate`], [`Sync`] and [`Send`] combo
/// and has a static lifetime. It is meant to be used to specify thread-safe blockchain backends.
///
/// Useful to avoid code verbosity.
pub trait ThreadSafeChain: ChainBackend + Sync + Send + 'static {}

impl<T: ChainBackend + Sync + Send + 'static> ThreadSafeChain for T {}
