// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error::Error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::Weight;
use bitcoin::Work;
use bitcoin::block::Header;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::constants::MAX_BLOCK_SIGOPS_COST;
use bitcoin::hashes::Hash;
use bitcoin::transaction::Version;
use floresta_common::bhash;
use floresta_common::prelude::Box;
use floresta_common::prelude::String;
use floresta_common::prelude::Vec;

use crate::BlockchainInterface;
use crate::pruned_utreexo::consensus::Consensus;
use crate::pruned_utreexo::sigops::SigopExt;

const MEDIAN_TIME_PAST_BLOCK_COUNT: usize = 11;

pub trait Bip30UnspendableExt {
    /// Returns true if the coinbase output in this block is BIP-30 unspendable.
    fn is_bip30_unspendable(&self, height: u32) -> bool;
}

impl Bip30UnspendableExt for Block {
    fn is_bip30_unspendable(&self, height: u32) -> bool {
        let bhash_91722 =
            bhash!("00000000000271a2dc26e7667f8419f2e15416dc6955e5a6c6cdf3f2574dd08e");
        let bhash_91812 =
            bhash!("00000000000af0aed4792b1acee3d966af36cf5def14935db8de83d6f9306f2f");

        match height {
            91722 => self.block_hash() == bhash_91722,
            91812 => self.block_hash() == bhash_91812,
            _ => false,
        }
    }
}

/// Provides additional methods for working with [`Transaction`] objects,
pub trait TransactionExt {
    /// Returns whether this transaction's version enforces BIP68 relative lock-time semantics.
    ///
    /// BIP68 only applies to version >= 2 transactions.
    fn is_bip68_enabled(&self) -> bool;
}

impl TransactionExt for Transaction {
    fn is_bip68_enabled(&self) -> bool {
        // Bitcoin Core treats nVersion as a uint32_t (bitcoin/bitcoin#29325), but rust-bitcoin's
        // `Version` still wraps an i32, so a signed comparison here would misclassify versions
        // whose top bit is set (they'd read as negative and wrongly skip the BIP68 check).
        // Cast to u32 on both sides to compare using the actual consensus semantics.
        // TODO(jaoleal): remove this cast when https://github.com/rust-bitcoin/rust-bitcoin/pull/4040 lands in a release.
        self.version.0 as u32 >= Version::TWO.0 as u32
    }
}

pub trait BlockExt {
    /// Runs the non-contextual block checks, which only require the block
    /// itself: the merkle root, the witness commitment and the block weight.
    /// Bitcoin Core performs them in
    /// [`CheckBlock`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/validation.cpp#L3950)
    /// and
    /// [`ContextualCheckBlock`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/validation.cpp#L4161).
    /// If successful, returns the list of [`Txid`]s computed for the merkle
    /// root check.
    fn check_block_structure(&self) -> Result<Vec<Txid>, BlockExtError>;
}

impl BlockExt for Block {
    fn check_block_structure(&self) -> Result<Vec<Txid>, BlockExtError> {
        let Some(txids) = Consensus::check_merkle_root(self) else {
            Err(BlockExtError::BadMerkleRoot)?
        };

        if !self.check_witness_commitment() {
            Err(BlockExtError::BadWitnessCommitment)?;
        }

        if self.weight() > Weight::MAX_BLOCK {
            Err(BlockExtError::BlockTooBig)?;
        }

        // Match Bitcoin Core's CheckBlock: sum every transaction's legacy sigops at four cost
        // units each. P2SH and witness costs are added later once prevouts are available.
        // https://github.com/bitcoin/bitcoin/blob/v31.0/src/validation.cpp#L4001-L4009
        let sigop_cost: usize = self.txdata.iter().map(|tx| tx.legacy_sigop_cost()).sum();
        if sigop_cost > usize::try_from(MAX_BLOCK_SIGOPS_COST).expect("non-negative") {
            Err(BlockExtError::TooManySigops)?;
        }

        Ok(txids)
    }
}

#[derive(Debug)]
pub enum BlockExtError {
    BadMerkleRoot,
    BadWitnessCommitment,
    BlockTooBig,
    TooManySigops,
}

/// Provides additional methods for working with [`Header`] objects,
pub trait HeaderExt {
    /// Calculates the Median Time Past (MTP) for the block.
    fn calculate_median_time_past(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<u32, HeaderExtError>;

    /// Calculates Median Time Past using a caller-provided previous-header lookup.
    fn median_time_past_with<E>(
        &self,
        previous_header: impl FnMut(&Self) -> Result<Self, E>,
    ) -> Result<u32, E>
    where
        Self: Sized;

    /// Calculates the total accumulated chain work up to the current block.
    fn calculate_chain_work(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Work, HeaderExtError>;

    /// Retrieves the hash of the next block in the chain, if it exists.
    ///
    /// Returns `None` if the block is the tip of the chain.
    fn get_next_block_hash(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Option<BlockHash>, HeaderExtError>;

    /// Retrieves the header of the previous block in the chain.
    fn get_previous_block_header(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Header, HeaderExtError>;

    /// Returns the block's "bits" field as a hexadecimal string.
    fn get_bits_hex(&self) -> String;

    /// Calculates the number of confirmations for the current block.
    fn get_confirmations(&self, chain: &impl BlockchainInterface) -> Result<u32, HeaderExtError>;

    /// Returns the block's difficulty as a floating-point number.
    fn get_difficulty(&self) -> f64;

    /// Retrieves the height of the block in the blockchain.
    fn get_height(&self, chain: &impl BlockchainInterface) -> Result<u32, HeaderExtError>;

    /// Returns the block's target as a hexadecimal string.
    ///
    /// In `rust-bitcoin`, calling `to_string` on `Target` returns the value in decimal
    /// because it wraps a `U256`, which defaults to decimal string conversion. However,
    /// Bitcoin Core represents targets in hexadecimal. This method ensures the target
    /// is returned in hexadecimal format, consistent with Bitcoin Core.
    fn get_target_hex(&self) -> String;

    /// Returns the block's version as a hexadecimal string.
    ///
    /// Bitcoin Core represents the block version as a 32-bit unsigned integer (`u32`)
    /// in hexadecimal format. This method ensures the version is returned as a
    /// properly formatted hexadecimal string, consistent with Bitcoin Core.
    fn get_version_hex(&self) -> String;
}

/// Errors that can occur when using the `HeaderExt` methods.
#[derive(Debug)]
pub enum HeaderExtError {
    /// An error related to the blockchain interface, wrapping the actual error.
    Chain(Box<dyn Error + Send + Sync>),

    /// Indicates that the block could not be found in the blockchain.
    BlockNotFound,

    /// You got an overflow while calculating the chain work.
    ChainWorkOverflow,
}

impl Display for HeaderExtError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chain(e) => write!(f, "Chain error: {e}"),
            Self::BlockNotFound => write!(f, "Block not found"),
            Self::ChainWorkOverflow => write!(f, "Chain work overflow"),
        }
    }
}

impl HeaderExt for Header {
    fn calculate_median_time_past(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<u32, HeaderExtError> {
        self.median_time_past_with(|current_header| current_header.get_previous_block_header(chain))
    }

    fn median_time_past_with<E>(
        &self,
        mut previous_header: impl FnMut(&Self) -> Result<Self, E>,
    ) -> Result<u32, E> {
        let mut block_timestamps = Vec::with_capacity(MEDIAN_TIME_PAST_BLOCK_COUNT);
        let mut current_header = *self;

        for _ in 0..MEDIAN_TIME_PAST_BLOCK_COUNT {
            block_timestamps.push(current_header.time);
            if block_timestamps.len() == MEDIAN_TIME_PAST_BLOCK_COUNT
                || current_header.prev_blockhash == BlockHash::all_zeros()
            {
                break;
            }

            current_header = previous_header(&current_header)?;
        }

        block_timestamps.sort();
        Ok(block_timestamps[block_timestamps.len() / 2])
    }

    fn calculate_chain_work(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Work, HeaderExtError> {
        chain
            .get_work(self.block_hash())
            .map_err(|err| HeaderExtError::Chain(Box::new(err)))
    }

    fn get_next_block_hash(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Option<BlockHash>, HeaderExtError> {
        let height = self.get_height(chain)?;

        // If obtaining the next block hash fails, treat it as "no next block" and return Ok(None)
        match chain.get_block_hash(height + 1) {
            Ok(opt_hash) => Ok(Some(opt_hash)),
            Err(_) => Ok(None),
        }
    }

    fn get_previous_block_header(
        &self,
        chain: &impl BlockchainInterface,
    ) -> Result<Header, HeaderExtError> {
        let prev_header = chain
            .get_block_header(&self.prev_blockhash)
            .map_err(|e| HeaderExtError::Chain(Box::new(e)))?;
        Ok(prev_header)
    }

    fn get_bits_hex(&self) -> String {
        serialize_hex(&self.bits.to_consensus().to_be())
    }

    fn get_confirmations(&self, chain: &impl BlockchainInterface) -> Result<u32, HeaderExtError> {
        let height = self.get_height(chain)?;

        let chain_height = chain
            .get_height()
            .map_err(|e| HeaderExtError::Chain(Box::new(e)))?;

        Ok(chain_height - height + 1)
    }

    fn get_difficulty(&self) -> f64 {
        self.difficulty_float()
    }

    fn get_height(&self, chain: &impl BlockchainInterface) -> Result<u32, HeaderExtError> {
        let height = match chain.get_block_height(&self.block_hash()) {
            Ok(Some(height)) => height,
            Ok(None) => return Err(HeaderExtError::BlockNotFound),
            Err(e) => return Err(HeaderExtError::Chain(Box::new(e))),
        };

        Ok(height)
    }

    fn get_target_hex(&self) -> String {
        serialize_hex(&self.target().to_be_bytes())
    }

    fn get_version_hex(&self) -> String {
        serialize_hex(&(self.version.to_consensus() as u32).to_be())
    }
}

impl From<ChainWorkOverflow> for HeaderExtError {
    fn from(_: ChainWorkOverflow) -> Self {
        Self::ChainWorkOverflow
    }
}

#[derive(Debug, PartialEq)]
pub struct ChainWorkOverflow;

pub trait WorkExt {
    /// Multiplies the Work by a u32 factor, returning an error if overflow occurs.
    fn multiply_work_by_u32(self, factor: u32) -> Result<Work, ChainWorkOverflow>;

    /// Returns the hexadecimal string representation of the Work.
    ///
    /// In `rust-bitcoin`, calling `to_string` on `Work` returns the value in decimal
    /// because it wraps a `U256`, which defaults to decimal string conversion. However,
    /// Bitcoin Core represents targets in hexadecimal. This method ensures the `Work``
    /// is returned in hexadecimal format, consistent with Bitcoin Core.
    fn to_string_hex(&self) -> String;
}

impl WorkExt for Work {
    fn multiply_work_by_u32(self, factor: u32) -> Result<Work, ChainWorkOverflow> {
        if factor == 0 {
            return Ok(Self::from_be_bytes([0u8; 32]));
        }

        if factor == 1 {
            return Ok(self);
        }

        // Convert Work to little-endian bytes for easier manipulation (least significant byte first)
        let work_bytes = self.to_le_bytes();
        let mut carry_high: u64 = 0;
        let mut result_bytes = [0u8; 32];
        let word_size = 4_usize;

        // Multiply each 4-byte word (u32) of Work by the factor, propagating carry
        // Work is processed in little-endian order (from least significant byte to most significant byte),
        // but result is stored in big-endian
        let by_chunks: Vec<u32> = work_bytes
            .chunks_exact(word_size)
            .map(|chunk| {
                let mut array = [0u8; 4];
                array.copy_from_slice(chunk);
                u32::from_le_bytes(array)
            })
            .collect();

        for (word_index, word) in by_chunks.iter().enumerate() {
            // Multiply the word by factor and add carry from previous step
            // Use u64 to avoid overflow during multiplication
            let product: u64 = (*word as u64) * (factor as u64) + carry_high;
            carry_high = product >> 32;

            // Store the low 32 bits of the product in the result
            // Result is built in big-endian order, so calculate the index accordingly
            let byte_index = by_chunks.len() - word_index;
            result_bytes[(byte_index - 1) * word_size..byte_index * word_size]
                .copy_from_slice(&(product as u32).to_be_bytes());
        }

        if carry_high > 0 {
            return Err(ChainWorkOverflow);
        }

        Ok(Self::from_be_bytes(result_bytes))
    }

    fn to_string_hex(&self) -> String {
        serialize_hex(&self.to_be_bytes())
    }
}

#[cfg(test)]
mod tests {
    use core::fmt;
    use core::fmt::Display;
    use core::fmt::Formatter;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::Arc;

    use bitcoin::Block;
    use bitcoin::BlockHash;
    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Transaction;
    use bitcoin::TxMerkleNode;
    use bitcoin::Txid;
    use bitcoin::absolute::LockTime;
    use bitcoin::block::Header;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::constants::genesis_block;
    use bitcoin::hashes::Hash;
    use bitcoin::hashes::sha256::Hash as Sha256Hash;
    use bitcoin::opcodes::Opcode;
    use bitcoin::opcodes::all::OP_CHECKMULTISIG;
    use bitcoin::opcodes::all::OP_CHECKSIG;
    use bitcoin::opcodes::all::OP_CHECKSIGADD;
    use bitcoin::opcodes::all::OP_NOP;
    use bitcoin::params::Params;
    use rustreexo::proof::Proof;
    use rustreexo::stump::Stump;

    use super::*;
    use crate::BlockConsumer;
    use crate::BlockchainError;
    use crate::UtxoData;
    use crate::pruned_utreexo::IBDState;
    use crate::test_utils::build_oversized_866_342;
    use crate::test_utils::build_tx;
    use crate::test_utils::decode_block_866342;
    use crate::test_utils::mutate_block;
    use crate::test_utils::update_witness_commitment;
    use crate::txin;
    use crate::txout;

    const SAMPLE_WORK_BYTES: [u8; 32] = [
        0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
        0, 4,
    ];

    #[derive(Debug)]
    pub enum MockBlockchainError {
        NotFound,
        Storage,
    }

    impl Display for MockBlockchainError {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "MockBlockchainError")
        }
    }

    impl core::error::Error for MockBlockchainError {}

    pub struct MockBlockchainInterface {
        pub headers: HashMap<BlockHash, Header>,
        pub heights: HashMap<BlockHash, u32>,
        pub storage_errors: HashSet<BlockHash>,
        pub chain_height: u32,
        pub fail_get_height: bool,
    }

    impl MockBlockchainInterface {
        pub fn new() -> Self {
            Self {
                headers: HashMap::new(),
                heights: HashMap::new(),
                storage_errors: HashSet::new(),
                chain_height: 0,
                fail_get_height: false,
            }
        }

        pub fn add_block(&mut self, hash: BlockHash, header: Header, height: u32) {
            self.headers.insert(hash, header);
            self.heights.insert(hash, height);
            self.chain_height = self.chain_height.max(height);
        }
    }

    impl BlockchainInterface for MockBlockchainInterface {
        type Error = MockBlockchainError;

        fn size_on_disk(&self) -> Result<u64, Self::Error> {
            unimplemented!("MockBlockchainInterface has no on-disk presence")
        }

        fn get_block_header(&self, hash: &BlockHash) -> Result<Header, Self::Error> {
            if self.storage_errors.contains(hash) {
                return Err(MockBlockchainError::Storage);
            }

            self.headers
                .get(hash)
                .cloned()
                .ok_or(MockBlockchainError::NotFound)
        }

        fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
            let hash = self
                .heights
                .iter()
                .find(|(_, h)| **h == height)
                .map(|(hash, _)| *hash)
                .ok_or(MockBlockchainError::NotFound)?;

            if self.storage_errors.contains(&hash) {
                return Err(MockBlockchainError::Storage);
            }

            Ok(hash)
        }

        fn get_mtp_by_height(&self, height: u32) -> Result<u32, Self::Error> {
            let hash = self.get_block_hash(height)?;
            let header = self.get_block_header(&hash)?;

            header.median_time_past_with(|current| self.get_block_header(&current.prev_blockhash))
        }

        fn get_block_height(&self, hash: &BlockHash) -> Result<Option<u32>, Self::Error> {
            if self.storage_errors.contains(hash) {
                return Err(MockBlockchainError::Storage);
            }

            Ok(self.heights.get(hash).cloned())
        }

        fn get_height(&self) -> Result<u32, Self::Error> {
            if self.fail_get_height {
                return Err(MockBlockchainError::Storage);
            }

            Ok(self.chain_height)
        }

        fn get_work(&self, tip: BlockHash) -> Result<Work, Self::Error> {
            if self.storage_errors.contains(&tip) {
                return Err(MockBlockchainError::Storage);
            }

            let work_hex = "00000000000000000000000000000000000000000000000000000bb80bb80bb8";
            Ok(Work::from_hex(&format!("0x{work_hex}")).expect("hardcoded work"))
        }

        fn get_tx(&self, _: &Txid) -> Result<Option<Transaction>, Self::Error> {
            unimplemented!()
        }

        fn estimate_fee(&self, _: usize) -> Result<f64, Self::Error> {
            unimplemented!()
        }

        fn get_block(&self, _: &BlockHash) -> Result<Block, Self::Error> {
            unimplemented!()
        }

        fn get_best_block(&self) -> Result<(u32, BlockHash), Self::Error> {
            unimplemented!()
        }

        fn subscribe(&self, _: Arc<dyn BlockConsumer>) {
            unimplemented!()
        }

        fn is_in_ibd(&self) -> bool {
            unimplemented!()
        }

        fn is_coinbase_mature(&self, _: u32, _: BlockHash) -> Result<bool, Self::Error> {
            unimplemented!()
        }

        fn get_block_locator(&self) -> Result<Vec<BlockHash>, Self::Error> {
            unimplemented!()
        }

        fn get_block_locator_for_tip(
            &self,
            _: BlockHash,
        ) -> Result<Vec<BlockHash>, BlockchainError> {
            unimplemented!()
        }

        fn get_validation_index(&self) -> Result<u32, Self::Error> {
            unimplemented!()
        }

        fn update_acc(
            &self,
            _: Stump,
            _: &Block,
            _: u32,
            _: Proof,
            _: Vec<Sha256Hash>,
        ) -> Result<Stump, Self::Error> {
            unimplemented!()
        }

        fn get_chain_tips(&self) -> Result<Vec<BlockHash>, Self::Error> {
            unimplemented!()
        }

        fn validate_block(
            &self,
            _: &Block,
            _: Proof,
            _: HashMap<OutPoint, UtxoData>,
            _: Vec<Sha256Hash>,
            _: Stump,
        ) -> Result<(), Self::Error> {
            unimplemented!()
        }

        fn get_fork_point(&self, _: BlockHash) -> Result<BlockHash, Self::Error> {
            unimplemented!()
        }

        fn get_params(&self) -> Params {
            unimplemented!()
        }

        fn acc(&self) -> Stump {
            unimplemented!()
        }

        fn ibd_state(&self) -> IBDState {
            unimplemented!()
        }
    }

    fn get_genesis_header() -> Header {
        let genesis_header = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";
        let header: Header = deserialize_hex(genesis_header).expect("Failed to deserialize header");
        header
    }

    fn get_chain_and_headers(height: u32) -> (MockBlockchainInterface, Vec<Header>) {
        let mut mock_chain = MockBlockchainInterface::new();

        let mut headers = vec![];
        let mut prev_blockhash = get_genesis_header().block_hash();
        let genesis_header = get_genesis_header();
        mock_chain.add_block(prev_blockhash, genesis_header, 0);
        headers.push(genesis_header);

        for i in 1..height {
            let header = Header {
                time: 1231006505 + i * 600,
                prev_blockhash,
                ..genesis_header
            };
            headers.push(header);
            let hash = header.block_hash();
            mock_chain.add_block(hash, header, i);
            prev_blockhash = header.block_hash();
        }

        (mock_chain, headers)
    }

    /// Real mainnet headers of the two BIP-30 duplicated blocks.
    const BLOCK_91722_HEADER_HEX: &str = "0100000042ba7629c32525ff7c74ca323fdc4c6d5b5c4410901aeb4f04300a000000000068b45f58b674e94eb881cd67b04c2cba07fe5552dbf1d5385637b0d4073dbfe3c89fdf4c56720e1ba67373ee";

    const BLOCK_91812_HEADER_HEX: &str = "010000000f362bdc16f2f880097c71fd3296c01b835c8b034e4d2939e8af02000000000065a62f2f6b9102d6eb5eee95be5ec3fcdfa27cf2117deeebefc6be53761d99499423e04c56720e1bb4518a45";

    fn bip30_block(header_hex: &str) -> Block {
        let header: Header = deserialize_hex(header_hex).expect("Failed to deserialize header");
        Block {
            header,
            txdata: vec![],
        }
    }

    #[test]
    fn test_bip30_height_91722_true() {
        let block = bip30_block(BLOCK_91722_HEADER_HEX);
        assert!(block.is_bip30_unspendable(91722));
    }

    #[test]
    fn test_bip30_height_91812_true() {
        let block = bip30_block(BLOCK_91812_HEADER_HEX);
        assert!(block.is_bip30_unspendable(91812));
    }

    #[test]
    fn test_bip30_wrong_height_returns_false() {
        let block_91722 = bip30_block(BLOCK_91722_HEADER_HEX);
        let block_91812 = bip30_block(BLOCK_91812_HEADER_HEX);

        // Each hash is only "unspendable" at its own height
        assert!(!block_91722.is_bip30_unspendable(91812));
        assert!(!block_91812.is_bip30_unspendable(91722));
    }

    #[test]
    fn test_bip30_other_heights_return_false() {
        let block_91722 = bip30_block(BLOCK_91722_HEADER_HEX);
        let block_91812 = bip30_block(BLOCK_91812_HEADER_HEX);

        for height in [0, 1, 91721, 91723, 91811, 91813] {
            assert!(
                !block_91722.is_bip30_unspendable(height),
                "91722 hash should not be unspendable at height {height}"
            );
            assert!(
                !block_91812.is_bip30_unspendable(height),
                "91812 hash should not be unspendable at height {height}"
            );
        }
    }

    #[test]
    fn test_bip30_unrelated_block_returns_false() {
        let block = Block {
            header: get_genesis_header(),
            txdata: vec![],
        };

        for height in [0, 91722, 91812] {
            assert!(!block.is_bip30_unspendable(height));
        }
    }

    #[test]
    fn test_calculate_median_time_past_more_than_11_blocks() {
        let (mock_chain, headers) = get_chain_and_headers(21);

        let median_header = headers[headers.len() - 1];
        let mtp = median_header
            .calculate_median_time_past(&mock_chain)
            .expect("Failed to calculate MTP");

        let mut times = headers
            .iter()
            .rev()
            .take(11)
            .map(|h| h.time)
            .collect::<Vec<_>>();
        times.sort();
        let expected_mtp = times[times.len() / 2];

        assert_eq!(mtp, expected_mtp);
    }

    #[test]
    fn test_calculate_median_time_past_less_than_11_blocks() {
        let (mock_chain, headers) = get_chain_and_headers(7);

        let median_header = headers[headers.len() - 1];
        let mtp = median_header
            .calculate_median_time_past(&mock_chain)
            .expect("Failed to calculate MTP");

        let mut times = headers.iter().map(|h| h.time).collect::<Vec<_>>();
        times.sort();
        let expected_mtp = times[times.len() / 2];

        assert_eq!(mtp, expected_mtp);
    }

    #[test]
    fn test_calculate_median_time_past_genesis_only() {
        let (mock_chain, headers) = get_chain_and_headers(1);

        // Test the MTP calculation
        let median_header = headers[0];
        let mtp = median_header
            .calculate_median_time_past(&mock_chain)
            .expect("Failed to calculate MTP");

        let expected_mtp = headers[0].time;

        assert_eq!(mtp, expected_mtp);
    }

    #[test]
    fn test_calculate_median_time_past_propagates_storage_errors() {
        let (mut mock_chain, headers) = get_chain_and_headers(12);
        let median_header = headers[headers.len() - 1];
        let missing_hash = headers[headers.len() - 2].block_hash();
        mock_chain.storage_errors.insert(missing_hash);

        let result = median_header.calculate_median_time_past(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "MTP lookup failure should be propagated, got {result:?}"
        );
    }

    #[test]
    fn test_calculate_median_time_past_does_not_fetch_past_window() {
        let (mut mock_chain, headers) = get_chain_and_headers(12);
        let median_header = headers[headers.len() - 1];
        let outside_window_hash = headers[0].block_hash();
        mock_chain.storage_errors.insert(outside_window_hash);

        let mtp = median_header
            .calculate_median_time_past(&mock_chain)
            .expect("MTP should not fetch outside the 11-header window");
        let mut times = headers
            .iter()
            .rev()
            .take(11)
            .map(|h| h.time)
            .collect::<Vec<_>>();
        times.sort();

        assert_eq!(mtp, times[times.len() / 2]);
    }

    #[test]
    fn test_get_next_block_hash() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        let header = headers[2];
        let next_hash = header
            .get_next_block_hash(&mock_chain)
            .expect("Failed to get next block hash")
            .expect("Next block hash is None");

        let expected_hash = headers[3].block_hash();

        assert_eq!(next_hash, expected_hash);

        let last_header = headers[headers.len() - 1];
        let next_hash = last_header
            .get_next_block_hash(&mock_chain)
            .expect("Failed to get next block hash");

        assert!(next_hash.is_none());
    }

    #[test]
    fn test_get_next_block_hash_height_error() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);

        let header = headers[2];
        mock_chain.storage_errors.insert(header.block_hash());

        let error = header.get_next_block_hash(&mock_chain).unwrap_err();

        assert!(matches!(error, HeaderExtError::Chain(_)));
    }

    #[test]
    fn test_get_next_block_hash_swallows_errors() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);

        // The block at height 3 fails to load, so requesting the next hash
        // of the block at height 2 must be treated as "no next block".
        let next_hash = headers[3].block_hash();
        mock_chain.storage_errors.insert(next_hash);

        let header = headers[2];
        let next = header
            .get_next_block_hash(&mock_chain)
            .expect("Lookup errors should be swallowed and return Ok");

        assert!(next.is_none());
    }

    #[test]
    fn test_get_previous_block_header() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        // headers[1].prev_blockhash points to the genesis header
        let header = headers[1];
        let prev_header = header
            .get_previous_block_header(&mock_chain)
            .expect("Failed to get previous block header");

        assert_eq!(prev_header, headers[0]);
    }

    #[test]
    fn test_get_previous_block_header_missing() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        // Point the header to a block that is not in the chain
        let mut header = headers[1];
        header.prev_blockhash = BlockHash::from_byte_array([0xabu8; 32]);

        let result = header.get_previous_block_header(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "Missing previous header should be propagated as a Chain error, got {result:?}"
        );
    }

    #[test]
    fn test_get_previous_block_header_storage_error() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);

        let genesis_hash = headers[0].block_hash();
        mock_chain.storage_errors.insert(genesis_hash);

        // headers[1].prev_blockhash = genesis hash, which now fails to load
        let header = headers[1];
        let result = header.get_previous_block_header(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "Storage failure should be propagated as a Chain error, got {result:?}"
        );
    }

    #[test]
    fn test_get_bits() {
        let header = get_genesis_header();
        let bits_hex = header.get_bits_hex();
        assert_eq!(bits_hex, "1d00ffff");
    }

    #[test]
    fn test_get_confirmations() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        let header = headers[2];
        let confirmations = header
            .get_confirmations(&mock_chain)
            .expect("Failed to get confirmations");

        let expected_confirmations = headers.len() - 2;

        assert_eq!(confirmations, expected_confirmations as u32);
    }

    #[test]
    fn test_get_difficulty() {
        let header = get_genesis_header();
        let difficulty = header.get_difficulty();
        assert_eq!(difficulty, 1.0);
    }

    #[test]
    fn test_get_confirmations_at_tip() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        let tip = headers[headers.len() - 1];
        let confirmations = tip
            .get_confirmations(&mock_chain)
            .expect("Failed to get confirmations");

        assert_eq!(confirmations, 1);
    }

    #[test]
    fn test_get_confirmations_propagates_chain_height_error() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);
        mock_chain.fail_get_height = true;

        let header = headers[2];
        let result = header.get_confirmations(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "Chain height failure should be propagated, got {result:?}"
        );
    }

    #[test]
    fn test_get_height() {
        let (mock_chain, headers) = get_chain_and_headers(5);
        let height_expected = 3;

        let header = headers[height_expected];
        let height = header
            .get_height(&mock_chain)
            .expect("Failed to get block height");

        assert_eq!(height, height_expected as u32);

        let mut header_missing = headers[0];
        header_missing.nonce = 0;
        let result = header_missing.get_height(&mock_chain);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_height_returns_block_not_found() {
        let (mock_chain, headers) = get_chain_and_headers(5);

        // A header that is not in the chain at all
        let mut header = headers[0];
        header.nonce = 0;

        let result = header.get_height(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::BlockNotFound)),
            "Unknown block should map to BlockNotFound, got {result:?}"
        );
    }

    #[test]
    fn test_get_height_propagates_storage_error() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);

        let hash = headers[2].block_hash();
        mock_chain.storage_errors.insert(hash);

        let header = headers[2];
        let result = header.get_height(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "Storage failure should be propagated as a Chain error, got {result:?}"
        );
    }

    #[test]
    fn test_get_target() {
        let header = get_genesis_header();
        let target_hex = header.get_target_hex();
        assert_eq!(
            target_hex,
            "00000000ffff0000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_get_version_hex() {
        let header = get_genesis_header();
        let version_hex = header.get_version_hex();
        assert_eq!(version_hex, "00000001");
    }

    #[test]
    fn test_calculate_chain_work_propagates_storage_error() {
        let (mut mock_chain, headers) = get_chain_and_headers(5);

        let hash = headers[2].block_hash();
        mock_chain.storage_errors.insert(hash);

        let header = headers[2];
        let result = header.calculate_chain_work(&mock_chain);

        assert!(
            matches!(result, Err(HeaderExtError::Chain(_))),
            "Work lookup failure should be propagated, got {result:?}"
        );
    }

    #[test]
    fn test_multiply_work_by_u32_success() {
        let work = Work::from_be_bytes(SAMPLE_WORK_BYTES);
        let factor = 2;

        let result = work.multiply_work_by_u32(factor).unwrap();

        let expected_bytes: [u8; 32] = [
            0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0,
            0, 0, 8,
        ];
        let expected = Work::from_be_bytes(expected_bytes);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiply_work_by_u32_overflow() {
        let work_bytes: [u8; 32] = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF,
        ];
        let work = Work::from_be_bytes(work_bytes);
        let factor = u32::MAX;

        let result = work.multiply_work_by_u32(factor);

        assert_eq!(result, Err(ChainWorkOverflow));
    }

    #[test]
    fn test_multiply_work_by_u32_factor_zero() {
        let work = Work::from_be_bytes([0xffu8; 32]);
        let result = work.multiply_work_by_u32(0).unwrap();

        assert_eq!(result, Work::from_be_bytes([0u8; 32]));
    }

    #[test]
    fn test_multiply_work_by_u32_factor_one() {
        let work = Work::from_be_bytes(SAMPLE_WORK_BYTES);

        let result = work.multiply_work_by_u32(1).unwrap();

        assert_eq!(result, work);
    }

    #[test]
    fn test_calculate_chain_work() {
        let (mock_chain, headers) = get_chain_and_headers(3000);
        let header = headers[headers.len() - 1];

        let work = header
            .calculate_chain_work(&mock_chain)
            .expect("Failed to calculate chain work");

        let expected_hex_string =
            "00000000000000000000000000000000000000000000000000000bb80bb80bb8";
        let expected_work = Work::from_hex(&format!("0x{expected_hex_string}")).unwrap();

        assert_eq!(work.to_string_hex(), expected_hex_string);
        assert_eq!(work, expected_work);
    }

    #[test]
    fn test_work_to_string_hex() {
        let work = Work::from_be_bytes(SAMPLE_WORK_BYTES);

        assert_eq!(
            work.to_string_hex(),
            "0000000300000001000000000000000200000000000000030000000000000004"
        );
    }

    fn tx_with_version(version: Version) -> Transaction {
        Transaction {
            version,
            lock_time: LockTime::ZERO,
            input: Vec::new(),
            output: Vec::new(),
        }
    }

    #[test]
    fn test_is_bip68_enabled() {
        assert!(!tx_with_version(Version::ONE).is_bip68_enabled());
        assert!(tx_with_version(Version::TWO).is_bip68_enabled());
        assert!(tx_with_version(Version::non_standard(3)).is_bip68_enabled());
    }

    #[test]
    fn test_is_bip68_enabled_unsigned_comparison() {
        // A version whose top bit is set reads as negative in rust-bitcoin's i32-backed
        // `Version`, but Bitcoin Core treats nVersion as a uint32_t, so this must still
        // enforce BIP68 (see the TODO on `TransactionExt::is_bip68_enabled`).
        let high_bit_version = Version::non_standard(-1);
        assert!(tx_with_version(high_bit_version).is_bip68_enabled());
    }
    #[test]
    fn test_check_block_structure_valid_block() {
        let block = decode_block_866342();

        let txids = block.check_block_structure().unwrap();

        let expected_txids: Vec<Txid> = block.txdata.iter().map(|tx| tx.compute_txid()).collect();
        assert_eq!(txids, expected_txids);
    }

    #[test]
    fn test_check_block_structure_valid_block_without_witness() {
        // The genesis block is pre-SegWit: witness commitments are optional when
        // no transaction in the block has witness data
        let block = genesis_block(Network::Bitcoin);

        let txids = block.check_block_structure().unwrap();

        let expected_txids: Vec<Txid> = block.txdata.iter().map(|tx| tx.compute_txid()).collect();
        assert_eq!(txids, expected_txids);
    }

    #[test]
    fn test_check_block_structure_bad_merkle_root() {
        let mut block = decode_block_866342();
        mutate_block(&mut block);

        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::BadMerkleRoot)
        ));
    }

    #[test]
    fn test_check_block_structure_mutated_merkle_tree() {
        let mut block = decode_block_866342();
        block.txdata.truncate(6);
        // Truncating a SegWit block breaks its witness commitment, so update it
        // to keep every check but the merkle tree intact
        update_witness_commitment(&mut block).unwrap();
        block.header.merkle_root = block.compute_merkle_root().unwrap();

        // Sanity check: the truncated block is structurally valid
        assert!(block.check_block_structure().is_ok());

        // CVE-2012-2459: `[1, 2, 3, 4, 5, 6]` and `[1, 2, 3, 4, 5, 6, 5, 6]` have
        // the same Merkle root
        block.txdata.extend_from_within(4..6);

        // A root-only check accepts it, but the consensus mutation detection
        // must reject it
        assert!(block.check_merkle_root());

        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::BadMerkleRoot)
        ));
    }

    #[test]
    fn test_check_block_structure_bad_witness_commitment() {
        let mut block = decode_block_866342();
        block.txdata.truncate(6);
        // Fix the merkle root so that only the witness commitment is broken
        block.header.merkle_root = block.compute_merkle_root().unwrap();

        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::BadWitnessCommitment)
        ));
    }

    #[test]
    fn test_check_block_structure_block_too_big() {
        let block = build_oversized_866_342();

        // Both the merkle root and the witness commitment are valid, so only the
        // weight check can reject it
        assert!(block.check_merkle_root());
        assert!(block.check_witness_commitment());
        assert!(block.weight() > Weight::MAX_BLOCK);

        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::BlockTooBig)
        ));
    }

    #[test]
    fn test_check_block_structure_empty_txdata_with_zero_merkle_root() {
        let mut block = decode_block_866342();
        block.txdata.clear();
        // Bitcoin Core defines the merkle root of an empty transaction list as zero
        block.header.merkle_root = TxMerkleNode::all_zeros();

        let txids = block.check_block_structure().unwrap();

        assert!(txids.is_empty());
    }

    /// Sums the legacy sigop cost of every transaction in `block`, mirroring the
    /// accounting `check_block_structure` applies against `MAX_BLOCK_SIGOPS_COST`.
    fn block_sigop_cost(block: &Block) -> usize {
        block.txdata.iter().map(|tx| tx.legacy_sigop_cost()).sum()
    }

    /// Returns a genesis-based block whose coinbase carries no sigops, so that the sigop
    /// cost of any block built on top of it is entirely explained by the added transactions.
    ///
    /// The genesis coinbase pays to a P2PK script (a single `OP_CHECKSIG`), which would
    /// otherwise add a constant four units to every assertion below.
    fn block_without_coinbase_sigops() -> Block {
        let mut block = genesis_block(Network::Bitcoin);
        block.txdata[0].output.clear();
        block
    }

    /// Builds a pre-SegWit block with no coinbase sigops where each entry of `output_ops`
    /// appends a transaction with that many `opcode`s in its scriptPubKey and, when
    /// non-zero, `input_ops` repetitions of `opcode` in its scriptSig.
    ///
    /// The merkle root is recomputed so the block passes every other non-contextual check,
    /// leaving the sigop budget as the only check that can reject it. Starting from the
    /// genesis block keeps the witness commitment check trivially satisfied, since none of
    /// the added transactions carry witness data.
    fn block_with_sigop_txs(opcode: Opcode, output_ops: &[usize], input_ops: usize) -> Block {
        let mut block = block_without_coinbase_sigops();
        let script = |op_count: usize| ScriptBuf::from_bytes(vec![opcode.to_u8(); op_count]);

        for (index, &op_count) in output_ops.iter().enumerate() {
            // A distinct outpoint per transaction keeps their txids unique, so the
            // CVE-2012-2459 merkle mutation check cannot reject the block first
            let outpoint = OutPoint {
                txid: Txid::all_zeros(),
                vout: index as u32,
            };
            let tx = build_tx(
                vec![txin!(outpoint, script(input_ops))],
                vec![txout!(0, script(op_count))],
            );
            block.txdata.push(tx);
        }

        block.header.merkle_root = block.compute_merkle_root().unwrap();
        block
    }

    #[test]
    fn test_check_block_structure_sigops_at_limit_is_accepted() {
        // 20,000 legacy sigops cost four units each, hitting the 80,000 budget exactly
        let block = block_with_sigop_txs(OP_CHECKSIG, &[10_000, 10_000], 0);

        assert_eq!(block_sigop_cost(&block), 80_000);
        assert!(block.check_block_structure().is_ok());
    }

    #[test]
    fn test_check_block_structure_sigops_one_over_limit_is_rejected() {
        let block = block_with_sigop_txs(OP_CHECKSIG, &[10_000, 10_000, 1], 0);

        assert_eq!(block_sigop_cost(&block), 80_004);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));
    }

    #[test]
    fn test_check_block_structure_sigops_are_summed_across_transactions() {
        // Each transaction costs 28,000 units on its own, well below the budget, so only
        // the block-wide sum can reject this block
        let block = block_with_sigop_txs(OP_CHECKSIG, &[7_000, 7_000, 7_000], 0);

        assert!(
            block
                .txdata
                .iter()
                .all(|tx| tx.legacy_sigop_cost() < 80_000)
        );
        assert_eq!(block_sigop_cost(&block), 84_000);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));
    }

    #[test]
    fn test_check_block_structure_counts_script_sig_sigops() {
        // Legacy counting also covers scriptSigs: an input with 20,001 OP_CHECKSIGs
        // overflows the budget without any sigop in an output script
        let block = block_with_sigop_txs(OP_CHECKSIG, &[0], 20_001);

        assert_eq!(block_sigop_cost(&block), 80_004);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));
    }

    #[test]
    fn test_check_block_structure_counts_coinbase_script_sig_sigops() {
        // The coinbase is not exempt: its scriptSig sigops count towards the budget too
        let mut block = block_without_coinbase_sigops();
        block.txdata[0].input[0].script_sig =
            ScriptBuf::from_bytes(vec![OP_CHECKSIG.to_u8(); 20_001]);
        block.header.merkle_root = block.compute_merkle_root().unwrap();

        assert_eq!(block_sigop_cost(&block), 80_004);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));
    }

    #[test]
    fn test_check_block_structure_counts_checkmultisig_as_twenty() {
        // Without a preceding `OP_N`, a bare CHECKMULTISIG counts as 20 sigops, so 1,000
        // of them fill the 80,000 budget and a single extra one overflows it
        let at_limit = block_with_sigop_txs(OP_CHECKMULTISIG, &[1_000], 0);
        assert_eq!(block_sigop_cost(&at_limit), 80_000);
        assert!(at_limit.check_block_structure().is_ok());

        let over_limit = block_with_sigop_txs(OP_CHECKMULTISIG, &[1_001], 0);
        assert_eq!(block_sigop_cost(&over_limit), 80_080);
        assert!(matches!(
            over_limit.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));
    }

    #[test]
    fn test_check_block_structure_ignores_non_sigop_opcodes() {
        // Only CHECKSIG/CHECKMULTISIG count; any amount of other opcodes must not consume
        // the sigop budget
        let block = block_with_sigop_txs(OP_NOP, &[20_001, 20_001], 0);

        assert_eq!(block_sigop_cost(&block), 0);
        assert!(block.check_block_structure().is_ok());
    }

    #[test]
    fn test_check_block_structure_ignores_taproot_checksigadd() {
        // `OP_CHECKSIGADD` is deliberately excluded from the legacy count: BIP342 gives
        // taproot its own per-input budget, enforced during script validation rather than
        // by this block-wide check
        let block = block_with_sigop_txs(OP_CHECKSIGADD, &[20_001], 0);

        assert_eq!(block_sigop_cost(&block), 0);
        assert!(block.check_block_structure().is_ok());
    }
}
