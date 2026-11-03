// SPDX-License-Identifier: MIT OR Apache-2.0

//! An in-memory [`ChainBackend`] mock for unit tests.
//!
//! This backend does no validation and stores nothing: it records the calls that
//! matter to the code under test (e.g. [`UpdatableChainstate::invalidate_block`])
//! and returns trivial results, so tests can assert *what the node asked the chain
//! to do*, instead of depending on a real chainstate.
//!
//! The mock is `Clone`: cloning it before handing it to the node keeps a handle
//! through which tests can inspect the recorded calls.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::OutPoint;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::Work;
use bitcoin::block::Header as BlockHeader;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use bitcoin::params::Params;
use floresta_chain::BlockConsumer;
use floresta_chain::BlockchainError;
use floresta_chain::UtxoData;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::IBDState;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::pruned_utreexo::partial_chain::PartialChainState;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::proof::Proof;
use rustreexo::stump::Stump;

#[derive(Default)]
struct MockChainState {
    /// What [`MockChain::get_validation_index`] reports.
    validation_index: u32,

    /// Every block hash passed to [`UpdatableChainstate::invalidate_block`], in order.
    invalidated_blocks: Vec<BlockHash>,
}

/// A [`ChainBackend`](floresta_chain::ChainBackend) that records calls instead of
/// validating anything.
#[derive(Clone, Default)]
pub struct MockChain {
    state: Arc<Mutex<MockChainState>>,
}

impl MockChain {
    /// Creates a mock reporting `validation_index` as its validation index.
    pub fn with_validation_index(validation_index: u32) -> Self {
        let chain = Self::default();
        chain.state.lock().unwrap().validation_index = validation_index;
        chain
    }

    /// Returns every block hash passed to [`UpdatableChainstate::invalidate_block`],
    /// in call order.
    pub fn invalidated_blocks(&self) -> Vec<BlockHash> {
        self.state.lock().unwrap().invalidated_blocks.clone()
    }
}

impl BlockchainInterface for MockChain {
    type Error = BlockchainError;

    fn get_block_hash(&self, _height: u32) -> Result<BlockHash, Self::Error> {
        Ok(BlockHash::all_zeros())
    }

    fn get_tx(&self, _txid: &Txid) -> Result<Option<Transaction>, Self::Error> {
        Ok(None)
    }

    fn get_height(&self) -> Result<u32, Self::Error> {
        Ok(self.state.lock().unwrap().validation_index)
    }

    fn estimate_fee(&self, _target: usize) -> Result<f64, Self::Error> {
        Ok(0.0)
    }

    fn get_block(&self, _hash: &BlockHash) -> Result<Block, Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn get_mtp_by_height(&self, _height: u32) -> Result<u32, Self::Error> {
        // Only used by `process_proof`'s MTP lookups, not exercised here.
        Ok(0)
    }

    fn get_best_block(&self) -> Result<(u32, BlockHash), Self::Error> {
        let index = self.state.lock().unwrap().validation_index;
        Ok((index, BlockHash::all_zeros()))
    }

    fn get_block_header(&self, _hash: &BlockHash) -> Result<BlockHeader, Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn subscribe(&self, _tx: Arc<dyn BlockConsumer>) {
        unimplemented!("not needed by the unit tests")
    }

    fn is_in_ibd(&self) -> bool {
        false
    }

    fn is_coinbase_mature(&self, _height: u32, _block: BlockHash) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn get_block_locator(&self) -> Result<Vec<BlockHash>, Self::Error> {
        Ok(Vec::new())
    }

    fn get_block_locator_for_tip(
        &self,
        _tip: BlockHash,
    ) -> Result<Vec<BlockHash>, BlockchainError> {
        Ok(Vec::new())
    }

    fn get_validation_index(&self) -> Result<u32, Self::Error> {
        Ok(self.state.lock().unwrap().validation_index)
    }

    fn get_block_height(&self, _hash: &BlockHash) -> Result<Option<u32>, Self::Error> {
        Ok(None)
    }

    fn update_acc(
        &self,
        _acc: Stump,
        _block: &Block,
        _height: u32,
        _proof: Proof,
        _del_hashes: Vec<sha256::Hash>,
    ) -> Result<Stump, Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error> {
        Ok(Vec::new())
    }

    fn validate_block(
        &self,
        _block: &Block,
        _proof: Proof,
        _inputs: HashMap<OutPoint, UtxoData>,
        _del_hashes: Vec<sha256::Hash>,
        _acc: Stump,
    ) -> Result<(), Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn get_fork_point(&self, _block: BlockHash) -> Result<BlockHash, Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn get_params(&self) -> Params {
        unimplemented!("not needed by the unit tests")
    }

    fn acc(&self) -> Stump {
        unimplemented!("not needed by the unit tests")
    }

    fn get_work(&self, _tip: BlockHash) -> Result<Work, Self::Error> {
        unimplemented!("not needed by the unit tests")
    }

    fn size_on_disk(&self) -> Result<u64, Self::Error> {
        Ok(0)
    }

    fn ibd_state(&self) -> IBDState {
        IBDState::Done
    }
}

impl UpdatableChainstate for MockChain {
    fn connect_block(
        &self,
        _block: &Block,
        _proof: Proof,
        _inputs: HashMap<OutPoint, UtxoData>,
        _del_hashes: Vec<sha256::Hash>,
    ) -> Result<u32, BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn switch_chain(&self, _new_tip: BlockHash) -> Result<(), BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn accept_header(&self, _header: BlockHeader) -> Result<(), BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn handle_transaction(&self) -> Result<(), BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn flush(&self) -> Result<(), BlockchainError> {
        Ok(())
    }

    fn update_ibd(&self, _ibd_state: IBDState) {}

    fn invalidate_block(&self, block: BlockHash) -> Result<(), BlockchainError> {
        self.state.lock().unwrap().invalidated_blocks.push(block);
        Ok(())
    }

    fn mark_block_as_valid(&self, _block: BlockHash) -> Result<(), BlockchainError> {
        Ok(())
    }

    fn get_root_hashes(&self) -> Vec<BitcoinNodeHash> {
        Vec::new()
    }

    fn get_partial_chain(
        &self,
        _initial_height: u32,
        _final_height: u32,
        _acc: Stump,
    ) -> Result<PartialChainState, BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn mark_chain_as_assumed(&self, _acc: Stump, _tip: BlockHash) -> Result<bool, BlockchainError> {
        unimplemented!("not needed by the unit tests")
    }

    fn get_acc(&self) -> Stump {
        unimplemented!("not needed by the unit tests")
    }
}
