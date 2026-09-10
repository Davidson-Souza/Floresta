// SPDX-License-Identifier: MIT OR Apache-2.0

//! A partial chain is a chain that only contains a subset of the blocks in the
//! full chain. We use multiple partial chains to sync up with the full chain,
//! and then merge them together to get the full chain. This allows us to make
//! Initial Block Download in parallel.
//!
//! We use a [PartialChainState] instead of the useal ChainState, mainly for
//! performance. Because we assume that only one worker will hold a [PartialChainState]
//! at a given time, we can drop all synchronization primitives and make a really performatic
//! ChainState that will consume and validate blocks as fast as we possibly can.
//!
//! This choice removes the use of costly atomic operations, but opens space for design flaws
//! and memory unsoundness, so here are some tips about this module and how people looking for
//! extend or use this code should proceed:
//!
//!   - Shared ownership is forbidden: if you have two threads or tasks owning this, you'll have
//!     data race. If you want to hold shared ownership for this module, you need to place a
//!     [PartialChainState] inside an `Arc<Mutex>` yourself. Don't just Arc this and expect it to
//!     work, as you are guaranteed to have data races.
//!   - The interior is toxic, so no peeking: no references, mutable or not, to any field should
//!     leak through the API, as we are not enforcing lifetime or borrowing rules at compile time.
//!   - Sending is fine: There's nothing in this module that makes it not sendable to between
//!     threads, as long as the origin thread gives away the ownership.

use core::cell::UnsafeCell;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::block::Header as BlockHeader;
use floresta_common::prelude::*;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::stump::Stump;
use tracing::info;

use super::BlockchainInterface;
use super::UpdatableChainstate;
use super::chainparams::ChainParams;
use super::consensus::Consensus;
use super::error::BlockValidationErrors;
use super::error::BlockchainError;
use crate::extensions::HeaderExt;
use crate::pruned_utreexo::IBDState;
use crate::pruned_utreexo::utxo_data::UtxoData;

#[doc(hidden)]
#[derive(Debug)]
pub(crate) struct PartialChainStateInner {
    /// The current accumulator state, it starts with a hardcoded value and
    /// gets checked against the result of the previous partial chainstate.
    pub(crate) current_acc: Stump,
    /// The block headers in this interval, we need this to verify the blocks
    /// and to build the accumulator. We assume this is sorted by height, and
    /// should contains all blocks in this interval.
    pub(crate) blocks: Vec<BlockHeader>,
    /// The height we are on right now, this is used to keep track of the progress
    /// of the sync.
    pub(crate) current_height: u32,
    /// The height we are syncing up to, trying to push more blocks than this will
    /// result in an error.
    pub(crate) final_height: u32,
    /// The error that occurred during validation, if any. It is here so we can
    /// pull that afterwards.
    pub(crate) error: Option<BlockValidationErrors>,
    /// The consensus parameters, we need this to validate the blocks.
    pub(crate) consensus: Consensus,
    /// Whether we assume the signatures in this interval as valid, this is used to
    /// speed up syncing, by assuming signatures in old blocks are valid.
    pub(crate) assume_valid: bool,
}

/// A partial chain is a chain that only contains a subset of the blocks in the
/// full chain. We use multiple partial chains to sync up with the full chain,
/// and then merge them together to get the full chain. This allows us to conduct
/// the sync in parallel. To build one, we need to know the initial
/// height, the final height, and the block headers in between.
///
/// We need to modify our current state as-we-go, but we also need to use the main
/// traits that define a chainstate. Most cruccially, both crates don't take a mutable
/// reference in any method, so we need some form of interior mutability.
/// We could just use a mutex, but this is not required and very wateful. Partial chains
/// differ from the normal chain because they only have one owner, the worker responsible
/// for driving this chain to it's completion. Because of that, we can simply use a UnsafeCell
/// and forbid shared access between threads by not implementing [Clone].
pub struct PartialChainState(pub(crate) UnsafeCell<PartialChainStateInner>);

/// We need to send [PartialChainState] between threads/tasks, because the worker thread, once it
/// finishes, needs to notify the main task and pass the final partial chain.
/// # Safety
///
/// All items inside the [UnsafeCell] are [Send], most importantly, there are no references or
/// smart pointers inside it, so sending shouldn't be a problem.
unsafe impl Send for PartialChainState {}
unsafe impl Sync for PartialChainState {}

impl PartialChainStateInner {
    /// Whether or not we have synced up to the final height
    pub fn is_sync(&self) -> bool {
        self.current_height == self.final_height
    }

    pub fn get_block(&self, height: u32) -> Option<&BlockHeader> {
        if height >= self.blocks.len() as u32 {
            return None;
        }

        self.blocks.get(height as usize)
    }

    #[inline]
    /// Update our internal state, given a new height and accumulator
    fn update_state(&mut self, height: u32, acc: Stump) {
        self.current_height = height;
        self.current_acc = acc;
    }

    #[inline]
    /// Returns the parameters for this chain
    fn chain_params(&self) -> ChainParams {
        self.consensus.parameters.clone()
    }

    #[inline]
    /// Returns the ancestor for a given block header
    fn get_ancestor(&self, height: u32) -> Result<BlockHeader, BlockchainError> {
        let prev = self
            .get_block(height - 1)
            .ok_or(BlockchainError::BlockNotPresent)?;

        Ok(*prev)
    }

    fn median_time_past(&self, height: u32) -> Result<u32, BlockchainError> {
        let header = self
            .get_block(height)
            .ok_or(BlockchainError::BlockNotPresent)?;
        let mut current_height = height;

        header.median_time_past_with(|_| {
            current_height = current_height
                .checked_sub(1)
                .expect("height 0 must be genesis and short-circuit MTP computation");

            self.get_block(current_height)
                .copied()
                .ok_or(BlockchainError::BlockNotPresent)
        })
    }

    fn previous_median_time_past(&self, height: u32) -> Result<u32, BlockchainError> {
        self.median_time_past(height.saturating_sub(1))
    }

    /// Process a block, given the proof, inputs, and deleted hashes.
    /// If we find an error that proves the assumed chain invalid, we save it.
    pub fn process_block(
        &mut self,
        block: &bitcoin::Block,
        proof: rustreexo::proof::Proof,
        inputs: HashMap<bitcoin::OutPoint, UtxoData>,
        del_hashes: Vec<bitcoin::hashes::sha256::Hash>,
    ) -> Result<u32, BlockchainError> {
        let height = self.current_height + 1;

        if let Err(BlockchainError::BlockValidation(e)) = self.validate_block(block, height, inputs)
        {
            // These errors may describe a mutated payload for an otherwise valid header.
            // Another peer can still provide the valid block, so don't poison this chain.
            if !matches!(
                e,
                BlockValidationErrors::BadMerkleRoot | BlockValidationErrors::BadWitnessCommitment
            ) {
                self.error = Some(e.clone());
            }
            return Err(BlockchainError::BlockValidation(e));
        }

        let acc = match Consensus::update_acc(&self.current_acc, block, height, proof, del_hashes) {
            Ok(acc) => acc,
            Err(_) => {
                self.error = Some(BlockValidationErrors::InvalidUtreexoProof);
                return Err(BlockchainError::InvalidUtreexoProof);
            }
        };

        // ... If we came this far, we consider this block valid ...

        if height % 10_000 == 0 {
            info!(
                "Downloading blocks: height={height} hash={}",
                block.block_hash()
            );
        }

        self.update_state(height, acc);

        Ok(height)
    }

    /// Check whether a block is valid
    fn validate_block(
        &self,
        block: &bitcoin::Block,
        height: u32,
        inputs: HashMap<bitcoin::OutPoint, UtxoData>,
    ) -> Result<(), BlockchainError> {
        self.consensus.check_block(block, height)?;

        let prev_block = self.get_ancestor(height)?;
        if block.header.prev_blockhash != prev_block.block_hash() {
            Err(BlockValidationErrors::BlockExtendsAnOrphanChain)?;
        }

        // Validate block transactions
        let subsidy = self.consensus.get_subsidy(height);
        let lock_time_cutoff =
            self.consensus
                .block_lock_time_cutoff(height, &block.header, || {
                    self.previous_median_time_past(height)
                })?;
        let verify_script = self.assume_valid;

        #[cfg(feature = "bitcoinkernel")]
        let flags = self
            .chain_params()
            .get_validation_flags(height, block.block_hash());

        #[cfg(not(feature = "bitcoinkernel"))]
        let flags = 0;

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

impl PartialChainState {
    /// Borrows the inner content as immutable reference.
    ///
    /// # Safety
    /// We can assume this [UnsafeCell] is initialized because the only way to get a
    /// [PartialChainState] is through our APIs, and we make sure this [UnsafeCell] is
    /// always valid.
    /// The reference returned here **should not** leak through the API, as there's no
    /// synchronization mechanisms for it.
    #[inline(always)]
    #[must_use]
    #[doc(hidden)]
    fn inner(&self) -> &PartialChainStateInner {
        unsafe { self.0.get().as_ref().expect("this pointer is valid") }
    }

    /// Borrows the inner content as a mutable reference.
    ///
    /// # Safety
    /// We can assume this [UnsafeCell] is initialized because the only way to get a
    /// [PartialChainState] is through our APIs, and we make sure this [UnsafeCell] is
    /// always valid.
    /// The reference returned here **should not** leak through the API, as there's no
    /// synchronization mechanisms for it.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    #[must_use]
    #[doc(hidden)]
    fn inner_mut(&self) -> &mut PartialChainStateInner {
        unsafe { self.0.get().as_mut().expect("this pointer is valid") }
    }

    /// Returns all blocks in this partial chain
    pub fn list_blocks(&self) -> &[BlockHeader] {
        &self.inner().blocks
    }

    /// Returns all block we have validated so far in this chain
    pub fn list_valid_blocks(&self) -> Vec<&BlockHeader> {
        self.inner()
            .blocks
            .iter()
            .take(self.inner().current_height as usize)
            .collect()
    }

    /// Returns whether any block inside this interval is invalid
    pub fn has_invalid_blocks(&self) -> bool {
        self.inner().error.is_some()
    }
}

impl UpdatableChainstate for PartialChainState {
    fn connect_block(
        &self,
        block: &bitcoin::Block,
        proof: rustreexo::proof::Proof,
        inputs: HashMap<bitcoin::OutPoint, UtxoData>,
        del_hashes: Vec<bitcoin::hashes::sha256::Hash>,
    ) -> Result<u32, BlockchainError> {
        self.inner_mut()
            .process_block(block, proof, inputs, del_hashes)
    }

    fn get_root_hashes(&self) -> Vec<BitcoinNodeHash> {
        self.inner().current_acc.roots.clone()
    }

    fn get_acc(&self) -> Stump {
        self.inner().current_acc.clone()
    }

    //these are no-ops, you can call them, but they won't do anything

    fn flush(&self) -> Result<(), BlockchainError> {
        // no-op: we keep everything on memory
        Ok(())
    }

    fn update_ibd(&self, _ibd_state: IBDState) {
        // no-op: we are only used for IBD, so we are always in IBD, and we don't need to update
        // anything
    }

    // these are unimplemented, and will panic if called

    fn push_headers(
        &self,
        _headers: Vec<BlockHeader>,
        _height: u32,
    ) -> Result<(), BlockchainError> {
        unimplemented!("partialChainState shouldn't be used to push headers")
    }

    fn accept_header(&self, _header: BlockHeader) -> Result<(), BlockchainError> {
        unimplemented!("partialChainState shouldn't be used to accept new headers")
    }

    fn switch_chain(&self, _new_tip: BlockHash) -> Result<(), BlockchainError> {
        unimplemented!("partialChainState shouldn't be used to switch chains")
    }

    fn get_partial_chain(
        &self,
        _initial_height: u32,
        _final_height: u32,
        _acc: Stump,
    ) -> Result<PartialChainState, BlockchainError> {
        unimplemented!("We are a partial chain")
    }

    fn invalidate_block(&self, _block: BlockHash) -> Result<(), BlockchainError> {
        unimplemented!(
            "we know if a block is invalid, just break out of your loop and use the is_valid() method"
        )
    }

    fn handle_transaction(&self) -> Result<(), BlockchainError> {
        unimplemented!("we don't do transactions")
    }

    fn mark_chain_as_assumed(&self, _acc: Stump, _tip: BlockHash) -> Result<bool, BlockchainError> {
        unimplemented!("no need to mark as valid")
    }

    fn mark_block_as_valid(&self, _block: BlockHash) -> Result<(), BlockchainError> {
        unimplemented!("no need to mark as valid")
    }
}

impl BlockchainInterface for PartialChainState {
    type Error = BlockchainError;

    fn ibd_state(&self) -> IBDState {
        IBDState::DownloadingBlocks
    }

    fn get_params(&self) -> bitcoin::params::Params {
        self.inner().chain_params().params
    }

    fn acc(&self) -> Stump {
        self.inner().current_acc.clone()
    }

    fn size_on_disk(&self) -> Result<u64, Self::Error> {
        unimplemented!("partialChainState has no on-disk presence")
    }

    fn get_height(&self) -> Result<u32, Self::Error> {
        Ok(self.inner().current_height)
    }

    fn get_block_hash(&self, height: u32) -> Result<bitcoin::BlockHash, BlockchainError> {
        self.inner()
            .blocks
            .get(height as usize)
            .map(|b| b.block_hash())
            .ok_or(BlockchainError::BlockNotPresent)
    }

    fn get_best_block(&self) -> Result<(u32, bitcoin::BlockHash), Self::Error> {
        Ok((
            self.inner().final_height,
            self.get_block_hash(self.inner().final_height)?,
        ))
    }

    fn is_coinbase_mature(
        &self,
        height: u32,
        _block: bitcoin::BlockHash,
    ) -> Result<bool, Self::Error> {
        let current_height = self.inner().current_height;
        let coinbase_maturity = self.inner().chain_params().coinbase_maturity;

        Ok(height + coinbase_maturity > current_height)
    }

    fn get_validation_index(&self) -> Result<u32, Self::Error> {
        Ok(self.inner().current_height)
    }

    fn is_in_ibd(&self) -> bool {
        !self.inner().is_sync()
    }

    fn get_block_locator(&self) -> Result<Vec<bitcoin::BlockHash>, Self::Error> {
        let mut hashes = vec![];
        let mut step = 1;
        let mut height = self.inner().current_height;

        while height > 0 {
            hashes.push(self.get_block_hash(height)?);
            if hashes.len() > 10 {
                step *= 2;
            }
            height = height.saturating_sub(step);
        }

        Ok(hashes)
    }

    // partial chain states are only used for IBD, so we don't need to implement these

    fn get_block_header(&self, _height: &BlockHash) -> Result<BlockHeader, Self::Error> {
        unimplemented!("PartialChainState::get_block_header")
    }

    fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error> {
        unimplemented!("PartialChainState::get_chain_tips")
    }

    fn validate_block(
        &self,
        _block: &bitcoin::Block,
        _proof: rustreexo::proof::Proof,
        _inputs: HashMap<bitcoin::OutPoint, UtxoData>,
        _del_hashes: Vec<bitcoin::hashes::sha256::Hash>,
        _acc: Stump,
    ) -> Result<(), Self::Error> {
        unimplemented!("PartialChainState::validate_block")
    }

    fn get_fork_point(&self, _block: BlockHash) -> Result<BlockHash, Self::Error> {
        unimplemented!("PartialChainState::get_fork_point")
    }

    fn update_acc(
        &self,
        _acc: Stump,
        _block: &Block,
        _height: u32,
        _proof: rustreexo::proof::Proof,
        _del_hashes: Vec<bitcoin::hashes::sha256::Hash>,
    ) -> Result<Stump, Self::Error> {
        unimplemented!("PartialChainState::update_acc")
    }

    fn get_block_locator_for_tip(
        &self,
        _tip: BlockHash,
    ) -> Result<Vec<BlockHash>, BlockchainError> {
        unimplemented!("PartialChainState::get_block_locator_for_tip")
    }

    fn get_block(&self, _hash: &bitcoin::BlockHash) -> Result<bitcoin::Block, Self::Error> {
        unimplemented!("PartialChainState::get_block")
    }

    fn get_tx(&self, _txid: &bitcoin::Txid) -> Result<Option<bitcoin::Transaction>, Self::Error> {
        unimplemented!("partialChainState::get_tx")
    }

    fn subscribe(&self, _tx: sync::Arc<dyn crate::BlockConsumer>) {
        unimplemented!("partialChainState::subscribe")
    }

    fn estimate_fee(&self, _target: usize) -> Result<f64, Self::Error> {
        unimplemented!("partialChainState::estimate_fee")
    }

    fn get_block_height(&self, _hash: &bitcoin::BlockHash) -> Result<Option<u32>, Self::Error> {
        unimplemented!("partialChainState::get_block_height")
    }

    fn get_work(&self, _tip: BlockHash) -> Result<bitcoin::Work, Self::Error> {
        unimplemented!()
    }
}

// mainly for tests
impl From<PartialChainStateInner> for PartialChainState {
    fn from(value: PartialChainStateInner) -> Self {
        Self(UnsafeCell::new(value))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bitcoin::Block;
    use bitcoin::CompactTarget;
    use bitcoin::Network;
    use bitcoin::block::Header;
    use bitcoin::block::Version;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::constants::genesis_block;
    use floresta_common::acchashes;
    use rustreexo::node_hash::BitcoinNodeHash;
    use rustreexo::proof::Proof;
    use rustreexo::stump::Stump;

    use super::PartialChainState;
    use crate::BlockchainError;
    use crate::pruned_utreexo::UpdatableChainstate;
    use crate::pruned_utreexo::chainparams::ChainParams;
    use crate::pruned_utreexo::consensus::Consensus;
    use crate::pruned_utreexo::error::BlockValidationErrors;
    use crate::pruned_utreexo::partial_chain::PartialChainStateInner;

    const EASIEST_REGTEST_TARGET_BITS: u32 = 0x207f_ffff;

    #[test]
    fn test_with_invalid_block() {
        fn run(block: &str, reason: BlockValidationErrors, caches_error: bool) {
            let genesis = parse_block(
                "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4adae5494dffff7f20020000000101000000010000000000000000000000000000000000000000000000000000000000000000ffffffff4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73ffffffff0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000",
            );
            let block = parse_block(block);

            let chainstate = get_empty_pchain(vec![genesis.header, block.header]);
            let res = chainstate.connect_block(&block, Proof::default(), HashMap::new(), vec![]);

            match res {
                Err(BlockchainError::BlockValidation(error)) if error == reason => {}
                other => panic!("unexpected {other:?}"),
            };

            assert_eq!(chainstate.has_invalid_blocks(), caches_error);
        }
        run(
            "0000002000226e46111a0b59caaf126043eb5bbf28c34f3a5e332a1fc7b2b73cf188910f39adbcd7823048d34357bdca86cd47172afe2a4af8366b5b34db36df89386d49b23ec964ffff7f20000000000101000000010000000000000000000000000000000000000000000000000000000000000000ffffffff165108feddb99c6b8435060b2f503253482f627463642fffffffff0100f2052a01000000160014806cef41295922d32ddfca09c26cc4acd36c3ed000000000",
            BlockValidationErrors::BlockExtendsAnOrphanChain,
            true,
        );
        run(
            "0000002000226e46111a0b59caaf126043eb5bbf28c34f3a5e332a1fc7b2b73cf188910f40adbcd7823048d34357bdca86cd47172afe2a4af8366b5b34db36df89386d49b23ec964ffff7f20000000000101000000010000000000000000000000000000000000000000000000000000000000000000ffffffff165108feddb99c6b8435060b2f503253482f627463642fffffffff0100f2052a01000000160014806cef41295922d32ddfca09c26cc4acd36c3ed000000000",
            BlockValidationErrors::BadMerkleRoot,
            false, // Potentially mutated block, original txdata may be valid
        );
        // Valid Merkle root, but the coinbase has witness data without a witness commitment
        run(
            "0000002000226e46111a0b59caaf126043eb5bbf28c34f3a5e332a1fc7b2b73cf188910f39adbcd7823048d34357bdca86cd47172afe2a4af8366b5b34db36df89386d49b23ec964ffff7f200000000001010000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff165108feddb99c6b8435060b2f503253482f627463642fffffffff0100f2052a01000000160014806cef41295922d32ddfca09c26cc4acd36c3ed001010000000000",
            BlockValidationErrors::BadWitnessCommitment,
            false,
        );
    }

    fn parse_block(hex: &str) -> Block {
        deserialize_hex(hex).unwrap()
    }

    fn get_empty_pchain(blocks: Vec<Header>) -> PartialChainState {
        PartialChainStateInner {
            assume_valid: true,
            consensus: Consensus {
                parameters: ChainParams::from(Network::Regtest),
            },
            current_height: 0,
            current_acc: Stump::default(),
            final_height: 1,
            blocks,
            error: None,
        }
        .into()
    }

    fn test_pchain_inner_with_times(times: &[u32]) -> PartialChainStateInner {
        let genesis = genesis_block(Network::Regtest);
        let mut prev_hash = genesis.header.prev_blockhash;
        let mut blocks = Vec::with_capacity(times.len());

        for (height, &time) in (0u32..).zip(times) {
            let header = Header {
                version: Version::TWO,
                prev_blockhash: prev_hash,
                merkle_root: genesis.header.merkle_root,
                time,
                bits: CompactTarget::from_consensus(EASIEST_REGTEST_TARGET_BITS),
                nonce: height,
            };
            prev_hash = header.block_hash();
            blocks.push(header);
        }

        PartialChainStateInner {
            assume_valid: true,
            consensus: Consensus::from(Network::Regtest),
            current_height: 0,
            current_acc: Stump::default(),
            final_height: times.len().saturating_sub(1) as u32,
            blocks,
            error: None,
        }
    }

    #[test]
    fn partial_chain_median_time_past_uses_available_headers() {
        let chainstate = test_pchain_inner_with_times(&[90, 120, 100, 110]);

        assert_eq!(chainstate.median_time_past(3).unwrap(), 110);
    }

    #[test]
    fn partial_chain_median_time_past_uses_last_eleven_headers() {
        let times = [
            300, 301, 311, 303, 309, 307, 305, 310, 302, 308, 304, 306, 312,
        ];
        let chainstate = test_pchain_inner_with_times(&times);

        assert_eq!(chainstate.median_time_past(12).unwrap(), 307);
    }

    #[test]
    fn partial_chain_median_time_past_requires_window_headers() {
        let chainstate = test_pchain_inner_with_times(&[400, 401, 402]);

        assert!(matches!(
            chainstate.median_time_past(10),
            Err(BlockchainError::BlockNotPresent)
        ));
    }

    #[test]
    fn partial_chain_previous_median_time_past_uses_previous_height() {
        let times = [
            500, 511, 501, 509, 507, 503, 505, 510, 502, 508, 504, 506, 900,
        ];
        let chainstate = test_pchain_inner_with_times(&times);

        assert_eq!(chainstate.previous_median_time_past(12).unwrap(), 506);
    }

    #[test]
    fn test_updating_single_chain() {
        let blocks = include_str!("../../testdata/regtest_blocks.txt");
        let mut parsed_blocks = vec![];
        for (i, block) in blocks.lines().enumerate() {
            if i > 100 {
                break;
            }
            parsed_blocks.push(parse_block(block));
        }
        let chainstate: PartialChainState = PartialChainStateInner {
            assume_valid: true,
            consensus: Consensus {
                parameters: ChainParams::from(Network::Regtest),
            },
            current_height: 0,
            current_acc: Stump::default(),
            final_height: 100,
            blocks: parsed_blocks.iter().map(|block| block.header).collect(),
            error: None,
        }
        .into();
        parsed_blocks.remove(0);
        for block in parsed_blocks {
            let proof = Proof::default();
            let inputs = HashMap::new();
            let del_hashes = Vec::new();
            chainstate
                .connect_block(&block, proof, inputs, del_hashes)
                .unwrap();
        }
        assert_eq!(chainstate.inner().current_height, 100);
    }

    #[test]
    fn test_updating_multiple_chains() {
        // We have two chains, one with 100 blocks, one with 50 blocks. We expect the
        // accumulator to be what we expect after 100 blocks and after 150 blocks.
        let blocks = include_str!("../../testdata/regtest_blocks.txt");
        let mut parsed_blocks = vec![];
        for block in blocks.lines() {
            parsed_blocks.push(parse_block(block));
        }
        // The file contains 150 blocks, we split them into two chains.
        let split = parsed_blocks.clone();
        let (blocks1, blocks2) = split.split_at(101);
        let mut chainstate1 = PartialChainStateInner {
            assume_valid: true,
            consensus: Consensus {
                parameters: ChainParams::from(Network::Regtest),
            },
            current_height: 0,
            current_acc: Stump::default(),
            final_height: 100,
            blocks: parsed_blocks.iter().map(|block| block.header).collect(),
            error: None,
        };

        // We need to add the last block of the first chain to the second chain, so that
        // the second chain can validate all its blocks.
        for (height, block) in blocks1.iter().enumerate() {
            // skip the genesis block
            if height == 0 {
                continue;
            }

            let proof = Proof::default();
            let inputs = HashMap::new();
            let del_hashes = vec![];
            chainstate1
                .process_block(block, proof, inputs, del_hashes)
                .unwrap();
        }

        // The state after 100 blocks, computed ahead of time.
        let roots = acchashes![
            "a2f1e6db842e13c7480c8d80f29ca2db5f9b96e1b428ebfdbd389676d7619081",
            "b21aae30bc74e9aef600a5d507ef27d799b9b6ba08e514656d34d717bdb569d2",
            "bedb648c9a3c5741660f926c1552d83ebb4cb1842cca6855b6d1089bb4951ce1",
        ]
        .to_vec();

        let acc2 = Stump { roots, leaves: 100 };

        // acc2 is hard-coded, while chainstate1.current_acc is calculated.
        // after catching up in the first half, the accumulator should be the same.
        // We can have the speedup of doing it in parallel, without needing to trust
        // the hard-coded values.
        assert_eq!(chainstate1.current_acc, acc2);

        let chainstate2: PartialChainState = PartialChainStateInner {
            assume_valid: true,
            consensus: Consensus {
                parameters: ChainParams::from(Network::Regtest),
            },
            current_height: 100,
            current_acc: acc2,
            final_height: 150,
            blocks: parsed_blocks.iter().map(|block| block.header).collect(),
            error: None,
        }
        .into();

        for block in blocks2 {
            let proof = Proof::default();
            let inputs = HashMap::new();
            let del_hashes = vec![];
            chainstate2
                .connect_block(block, proof, inputs, del_hashes)
                .unwrap();
        }

        let roots = acchashes![
            "e00b4ecc7c30865af0ac3b0c7c1b996015f51d6a6577ee6f52cc04b55933eb91",
            "9bf9659f93e246e0431e228032cd4b3a4d8a13e57f3e08a221e61f3e0bae657f",
            "e329a7ddcc888130bb6e4f82ce9f5cf5a712a7b0ae05a1aaf21b363866a9b05e",
            "1864a4982532447dcb3d9a5d2fea9f8ed4e3b1e759d55b8a427fb599fed0c302",
        ]
        .to_vec();

        let expected_acc: Stump = Stump { leaves: 150, roots };

        assert_eq!(chainstate2.inner().current_height, 150);
        assert_eq!(chainstate2.inner().current_acc, expected_acc);
    }
}
