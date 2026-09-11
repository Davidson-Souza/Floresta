// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error::Error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;

#[cfg(feature = "bitcoinkernel")]
use bitcoin::Amount;
use bitcoin::Block;
use bitcoin::BlockHash;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::Network;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::OutPoint;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::ScriptBuf;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::Sequence;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::Transaction;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::TxIn;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::TxOut;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::Witness;
use bitcoin::Work;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::absolute;
use bitcoin::block::Header;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::blockdata::opcodes::OP_0;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::blockdata::opcodes::Opcode;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::blockdata::opcodes::all::OP_RETURN;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::blockdata::script::Instruction;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::consensus::Decodable;
use bitcoin::consensus::encode::serialize_hex;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::transaction::Version;
use floresta_common::bhash;
use floresta_common::prelude::Box;
use floresta_common::prelude::String;
use floresta_common::prelude::Vec;
#[cfg(feature = "bitcoinkernel")]
use floresta_common::prelude::vec;

use crate::BlockchainInterface;
#[cfg(feature = "bitcoinkernel")]
use crate::ChainParams;
#[cfg(feature = "bitcoinkernel")]
use crate::pruned_utreexo::merkle::ConsensusMerkle;

const MEDIAN_TIME_PAST_BLOCK_COUNT: usize = 11;
#[cfg(feature = "bitcoinkernel")]
const SIGNET_HEADER: [u8; 4] = [0xec, 0xc7, 0xda, 0xa2];
#[cfg(feature = "bitcoinkernel")]
const SIGNET_VERIFY_FLAGS: bitcoinkernel::ScriptVerificationFlags = bitcoinkernel::VERIFY_P2SH
    | bitcoinkernel::VERIFY_WITNESS
    | bitcoinkernel::VERIFY_DERSIG
    | bitcoinkernel::VERIFY_NULLDUMMY;
#[cfg(feature = "bitcoinkernel")]
const WITNESS_COMMITMENT_HEADER: [u8; 6] = [0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];

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

#[cfg(feature = "bitcoinkernel")]
/// Provides BIP-325 signet validation for [`Block`]s.
pub trait SignetBlockExt {
    /// Returns whether this block satisfies the signet challenge in `chain_params`.
    ///
    /// Genesis blocks and blocks for non-signet networks are always valid.
    fn is_signet_challenge_valid(&self, chain_params: &ChainParams) -> bool;
}

#[cfg(feature = "bitcoinkernel")]
impl SignetBlockExt for Block {
    fn is_signet_challenge_valid(&self, chain_params: &ChainParams) -> bool {
        if self.block_hash() == chain_params.genesis.block_hash() {
            return true;
        }

        if chain_params.network != Network::Signet {
            return true;
        }

        let Some(challenge) = chain_params.signet_challenge.as_ref() else {
            return false;
        };

        let Some((to_spend, to_sign)) = create_signet_transactions(self, challenge) else {
            return false;
        };

        let serialized = serialize(&to_sign);
        let Ok(to_sign) = bitcoinkernel::Transaction::try_from(serialized.as_slice()) else {
            return false;
        };
        let output = &to_spend.output[0];
        let Ok(amount) = i64::try_from(output.value.to_sat()) else {
            return false;
        };
        let Ok(challenge) = bitcoinkernel::ScriptPubkey::try_from(output.script_pubkey.as_bytes())
        else {
            return false;
        };
        let previous_output = bitcoinkernel::TxOut::new(&challenge, amount);
        let Ok(tx_data) = bitcoinkernel::PrecomputedTransactionData::new(
            &to_sign,
            core::slice::from_ref(&previous_output),
        ) else {
            return false;
        };

        bitcoinkernel::verify(
            &challenge,
            Some(amount),
            &to_sign,
            0,
            Some(SIGNET_VERIFY_FLAGS),
            &tx_data,
        )
        .is_ok()
    }
}

#[cfg(feature = "bitcoinkernel")]
fn create_signet_transactions(
    block: &Block,
    challenge: &ScriptBuf,
) -> Option<(Transaction, Transaction)> {
    let mut coinbase = block.txdata.first()?.clone();
    let commitment_index = coinbase.output.iter().rposition(|output| {
        let script = output.script_pubkey.as_bytes();
        script.len() >= 38 && script.starts_with(&WITNESS_COMMITMENT_HEADER)
    })?;

    let solution =
        fetch_and_clear_signet_solution(&mut coinbase.output[commitment_index].script_pubkey);
    let (script_sig, witness) = match solution {
        Some(solution) => {
            let mut reader = solution.as_slice();
            let script_sig = ScriptBuf::consensus_decode(&mut reader).ok()?;
            let witness = Witness::consensus_decode(&mut reader).ok()?;
            if !reader.is_empty() {
                return None;
            }
            (script_sig, witness)
        }
        None => (ScriptBuf::new(), Witness::new()),
    };

    let mut txids = Vec::with_capacity(block.txdata.len());
    txids.push(coinbase.compute_txid());
    txids.extend(block.txdata.iter().skip(1).map(Transaction::compute_txid));
    let (signet_merkle_root, _) = ConsensusMerkle::calculate_root(&txids)?;

    let mut block_data = [0; 72];
    block_data[..4].copy_from_slice(&block.header.version.to_consensus().to_le_bytes());
    block_data[4..36].copy_from_slice(&block.header.prev_blockhash.to_byte_array());
    block_data[36..68].copy_from_slice(&signet_merkle_root.to_byte_array());
    block_data[68..].copy_from_slice(&block.header.time.to_le_bytes());

    let to_spend = Transaction {
        version: Version(0),
        lock_time: absolute::LockTime::ZERO,
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            script_sig: ScriptBuf::builder()
                .push_opcode(OP_0)
                .push_slice(block_data)
                .into_script(),
            sequence: Sequence::ZERO,
            witness: Witness::new(),
        }],
        output: vec![TxOut {
            value: Amount::ZERO,
            script_pubkey: challenge.clone(),
        }],
    };

    let to_sign = Transaction {
        version: Version(0),
        lock_time: absolute::LockTime::ZERO,
        input: vec![TxIn {
            previous_output: OutPoint::new(to_spend.compute_txid(), 0),
            script_sig,
            sequence: Sequence::ZERO,
            witness,
        }],
        output: vec![TxOut {
            value: Amount::ZERO,
            script_pubkey: ScriptBuf::builder().push_opcode(OP_RETURN).into_script(),
        }],
    };

    Some((to_spend, to_sign))
}

#[cfg(feature = "bitcoinkernel")]
fn fetch_and_clear_signet_solution(witness_commitment: &mut ScriptBuf) -> Option<Vec<u8>> {
    let original = witness_commitment.as_bytes();
    let mut replacement = ScriptBuf::new();
    let mut solution = None;

    for instruction in witness_commitment.instruction_indices() {
        let Ok((index, instruction)) = instruction else {
            break;
        };

        match instruction {
            Instruction::PushBytes(bytes) if bytes.is_empty() => {
                replacement.push_opcode(Opcode::from(original[index]));
            }
            Instruction::PushBytes(bytes) => {
                if solution.is_none()
                    && bytes.len() > SIGNET_HEADER.len()
                    && bytes.as_bytes().starts_with(&SIGNET_HEADER)
                {
                    solution = Some(bytes.as_bytes()[SIGNET_HEADER.len()..].to_vec());
                    replacement.push_slice(SIGNET_HEADER);
                } else {
                    replacement.push_slice(bytes);
                }
            }
            Instruction::Op(opcode) => replacement.push_opcode(opcode),
        }
    }

    if solution.is_some() {
        *witness_commitment = replacement;
    }

    solution
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
    #[cfg(feature = "bitcoinkernel")]
    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::Transaction;
    use bitcoin::Txid;
    use bitcoin::block::Header;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::hashes::sha256::Hash as Sha256Hash;
    use bitcoin::params::Params;
    use rustreexo::proof::Proof;
    use rustreexo::stump::Stump;

    use super::*;
    use crate::BlockConsumer;
    #[cfg(feature = "bitcoinkernel")]
    use crate::BlockValidationErrors;
    use crate::BlockchainError;
    use crate::UtxoData;
    use crate::pruned_utreexo::IBDState;
    #[cfg(feature = "bitcoinkernel")]
    use crate::pruned_utreexo::consensus::Consensus;

    const SAMPLE_WORK_BYTES: [u8; 32] = [
        0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
        0, 4,
    ];

    #[cfg(feature = "bitcoinkernel")]
    const SIGNET_BLOCK_ONE: &str = concat!(
        "00000020f61eee3b63a380a477a063af32b2bbc97c9ff9f01f2c4225e973988108000000f575c832",
        "35984e7dc4afc1f30944c170462e84437ab6f2d52e16878a79e4678bd1914d5fae77031eccf40700",
        "01010000000001010000000000000000000000000000000000000000000000000000000000000000",
        "ffffffff025151feffffff0200f2052a010000001600149243f727dd5343293eb83174324019ec16",
        "c2630f0000000000000000776a24aa21a9ede2f61c3f71d1defd3fa999dfa36953755c6906897999",
        "62b48bebd836974e8cf94c4fecc7daa2490047304402205e423a8754336ca99dbe16509b877ef1bf",
        "98d008836c725005b3c787c41ebe46022047246e4467ad7cc7f1ad98662afcaf14c115e0095a227c",
        "7b05c5182591c23e7e01000120000000000000000000000000000000000000000000000000000000",
        "000000000000000000",
    );

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

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn default_signet_block_satisfies_challenge() {
        let block: Block = deserialize_hex(SIGNET_BLOCK_ONE).expect("valid signet block");
        let consensus = Consensus::from(Network::Signet);

        assert!(block.is_signet_challenge_valid(&consensus.parameters));
        consensus
            .check_block(&block, 1)
            .expect("valid signet block must pass contextual checks");
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn signet_parameters_require_a_challenge() {
        let block: Block = deserialize_hex(SIGNET_BLOCK_ONE).expect("valid signet block");
        let mut params = ChainParams::from(Network::Signet);
        params.signet_challenge = None;

        assert!(!block.is_signet_challenge_valid(&params));
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn modified_signet_solution_fails_consensus_validation() {
        let mut block: Block = deserialize_hex(SIGNET_BLOCK_ONE).expect("valid signet block");
        let script = &mut block.txdata[0].output[1].script_pubkey;
        let header_position = script
            .as_bytes()
            .windows(SIGNET_HEADER.len())
            .position(|window| window == SIGNET_HEADER)
            .expect("signet header is present");
        script.as_mut_bytes()[header_position + SIGNET_HEADER.len() + 10] ^= 1;
        block.header.merkle_root = block.compute_merkle_root().expect("block is not empty");

        let consensus = Consensus::from(Network::Signet);
        let result = consensus.check_block(&block, 1);

        assert!(matches!(
            result,
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::InvalidSignetBlockSolution
            ))
        ));
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn empty_solution_satisfies_op_true_challenge() {
        let mut block: Block = deserialize_hex(SIGNET_BLOCK_ONE).expect("valid signet block");
        let script = &mut block.txdata[0].output[1].script_pubkey;
        assert!(fetch_and_clear_signet_solution(script).is_some());
        block.header.merkle_root = block.compute_merkle_root().expect("block is not empty");

        let mut params = ChainParams::from(Network::Signet);
        params.signet_challenge = Some(ScriptBuf::from_bytes(vec![0x51]));
        assert!(block.is_signet_challenge_valid(&params));

        params.signet_challenge = Some(ScriptBuf::from_bytes(vec![0x00]));
        assert!(!block.is_signet_challenge_valid(&params));
    }
}
