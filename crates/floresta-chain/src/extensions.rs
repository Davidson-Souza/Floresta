// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error::Error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::Weight;
use bitcoin::Work;
use bitcoin::block::Header;
use bitcoin::consensus::encode::serialize_hex;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::consensus::serialize;
use bitcoin::constants::MAX_BLOCK_SIGOPS_COST;
use bitcoin::hashes::Hash;
use bitcoin::p2p::ServiceFlags;
use bitcoin::transaction::Version;
use floresta_common::bhash;
use floresta_common::prelude::Box;
use floresta_common::prelude::HashSet;
use floresta_common::prelude::String;
use floresta_common::prelude::ToString;
use floresta_common::prelude::Vec;
use floresta_common::prelude::vec;
use floresta_common::service_flags;

use crate::BlockchainInterface;
use crate::ChainParams;
use crate::pruned_utreexo::consensus::Consensus;
use crate::pruned_utreexo::sigops::SigopExt;

/// How many blocks are using for computing the Median Time Past.
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

    /// Classifies an observed missing input using only the block's transactions.
    ///
    /// A missing output created by a transaction in this block, or an outpoint
    /// spent more than once in this block, proves the block invalid. Other missing
    /// inputs may be caused by incomplete peer-supplied leaf data.
    fn is_invalid_missing_input(&self, outpoint: OutPoint) -> bool;
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

    fn is_invalid_missing_input(&self, outpoint: OutPoint) -> bool {
        let mut spends = 0;
        for tx in &self.txdata {
            if tx.compute_txid() == outpoint.txid {
                return true;
            }
            spends += tx
                .input
                .iter()
                .filter(|input| input.previous_output == outpoint)
                .count();
            if spends > 1 {
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
pub enum BlockExtError {
    BadMerkleRoot,
    BadWitnessCommitment,
    BlockTooBig,
    TooManySigops,
}

#[cfg(feature = "bitcoinkernel")]
/// Provides BIP-325 signet validation for [`Block`]s.
pub trait SignetBlockExt {
    /// Returns whether this block satisfies the signet challenge in `chain_params`.
    ///
    /// Genesis blocks and blocks for non-signet networks are always valid.
    fn check_signet_challenge(&self, chain_params: &ChainParams) -> bool;
}

#[cfg(feature = "bitcoinkernel")]
impl SignetBlockExt for Block {
    fn check_signet_challenge(&self, chain_params: &ChainParams) -> bool {
        if self.block_hash() == chain_params.genesis.block_hash() {
            return true;
        }

        if chain_params.network != Network::Signet {
            return true;
        }

        let Some(challenge) = chain_params.signet_challenge.as_ref() else {
            return false;
        };

        let Some((to_spend, to_sign)) = signet_utils::create_signet_transactions(self, challenge)
        else {
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
            Some(signet_utils::SIGNET_VERIFY_FLAGS),
            &tx_data,
        )
        .is_ok()
    }
}

#[cfg(feature = "bitcoinkernel")]
mod signet_utils {
    use bitcoin::Amount;
    use bitcoin::Block;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Sequence;
    use bitcoin::Transaction;
    use bitcoin::TxIn;
    use bitcoin::TxOut;
    use bitcoin::Witness;
    use bitcoin::absolute;
    use bitcoin::blockdata::opcodes::OP_0;
    use bitcoin::blockdata::opcodes::Opcode;
    use bitcoin::blockdata::opcodes::all::OP_RETURN;
    use bitcoin::blockdata::script::Instruction;
    use bitcoin::consensus::Decodable;
    use bitcoin::hashes::Hash;
    use bitcoin::transaction::Version;
    use floresta_common::prelude::Vec;
    use floresta_common::prelude::vec;

    use crate::pruned_utreexo::merkle::ConsensusMerkle;

    /// BIP-325 defined header, meaning this output is the signet challenge response.
    pub const SIGNET_HEADER: [u8; 4] = [0xec, 0xc7, 0xda, 0xa2];
    pub const SIGNET_VERIFY_FLAGS: bitcoinkernel::ScriptVerificationFlags =
        bitcoinkernel::VERIFY_P2SH
            | bitcoinkernel::VERIFY_WITNESS
            | bitcoinkernel::VERIFY_DERSIG
            | bitcoinkernel::VERIFY_NULLDUMMY;

    const WITNESS_COMMITMENT_HEADER: [u8; 6] = [0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];

    pub fn create_signet_transactions(
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

    pub fn fetch_and_clear_signet_solution(witness_commitment: &mut ScriptBuf) -> Option<Vec<u8>> {
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

/// A dns seed is a authoritative DNS server that returns the IP addresses of nodes that are
/// likely to be accepting incoming connections. This is our preferred way of finding new peers
/// on the first startup, as peers returned by seeds are likely to be online and accepting
/// connections. We may use this as a fallback if we don't have any peers to connect in
/// subsequent startups.
///
/// Some seeds allow filtering by service flags, so we may use this to find peers that are
/// likely to be running Utreexo, for example.
pub struct DnsSeed {
    /// The domain name of the seed
    pub seed: &'static str,

    /// Useful filters we can use to find relevant peers
    pub filters: ServiceFlags,
}

/// This functionality is used to create a new DNS seed with possible filters.
impl DnsSeed {
    /// Create a new DNS seed
    pub fn new(seed: &'static str, filters: ServiceFlags) -> Self {
        Self { seed, filters }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuriedDeployment {
    Bip34,
    Bip65,
    Bip66,
    Csv,
    Segwit,
}

impl BuriedDeployment {
    /// Name used in the `getdeploymentinfo` RPC response.
    pub fn to_deployment_name(self) -> &'static str {
        match self {
            Self::Bip34 => "bip34",
            Self::Bip65 => "bip65",
            Self::Bip66 => "bip66",
            Self::Csv => "csv",
            Self::Segwit => "segwit",
        }
    }

    /// Script flags enabled by this deployment. Empty for deployments
    /// that do not change script validation (BIP34 only changes coinbase structure).
    pub fn to_script_flags(self) -> &'static [&'static str] {
        match self {
            Self::Bip34 => &[],
            Self::Bip65 => &["CHECKLOCKTIMEVERIFY"],
            Self::Bip66 => &["DERSIG"],
            Self::Csv => &["CHECKSEQUENCEVERIFY"],
            Self::Segwit => &["WITNESS", "NULLDUMMY"],
        }
    }
}

pub trait ChainParamsExt {
    /// Get a list of [`DnsSeed`]s for a given [`Network`].
    ///
    /// Some DNS seeds allow requesting addresses using a [`ServiceFlags`] filter.
    /// Here we define `x9`, `x49`, and `x1009` (the relevant services for this node
    /// to operate), and use them to request addresses from the DNS seeds that support it.
    fn get_chain_dns_seeds(&self) -> Vec<DnsSeed>;

    /// Returns the buried deployment list for a network, as `(name, activation_height)` pairs.
    ///
    /// Heights are sourced from Bitcoin Core's `chainparams.cpp` at v30.2
    /// (commit `4d7d5f6b79d4c11c47e7a828d81296918fd11d4d`):
    /// <https://github.com/bitcoin/bitcoin/blob/4d7d5f6b79d4c11c47e7a828d81296918fd11d4d/src/kernel/chainparams.cpp>
    //
    // TODO: also emit BIP9 deployments (`taproot`, `testdummy`); requires the versionbits state machine.
    fn buried_deployments_for(&self) -> Vec<(BuriedDeployment, u32)>;

    /// Returns the set of script-validation flags active for this network at the given block hash
    /// and height, including historical exception blocks.
    fn get_script_flags(&self, hash: BlockHash, height: u32) -> HashSet<String>;
}

impl ChainParamsExt for ChainParams {
    fn get_chain_dns_seeds(&self) -> Vec<DnsSeed> {
        let mut seeds = Vec::new();

        let none = ServiceFlags::NONE;
        let x9 = ServiceFlags::NETWORK | ServiceFlags::WITNESS;
        let x49 = ServiceFlags::NETWORK | ServiceFlags::WITNESS | ServiceFlags::COMPACT_FILTERS;
        let x1009 = ServiceFlags::NETWORK | ServiceFlags::WITNESS | service_flags::UTREEXO.into();
        let x1000 = service_flags::UTREEXO.into();

        #[rustfmt::skip]
        match self.network {
            Network::Bitcoin => {
                seeds.push(DnsSeed::new("seed.calvinkim.info", x1009));
                seeds.push(DnsSeed::new("seed.bitcoin.luisschwab.com", x1009));
                seeds.push(DnsSeed::new("seed.bitcoin.sipa.be", x9));
                seeds.push(DnsSeed::new("dnsseed.bluematt.me", x49));
                seeds.push(DnsSeed::new("seed.bitcoinstats.com", x49));
                seeds.push(DnsSeed::new("seed.btc.petertodd.org", x49));
                seeds.push(DnsSeed::new("seed.bitcoin.sprovoost.nl", x49));
                seeds.push(DnsSeed::new("dnsseed.emzy.de", x49));
                seeds.push(DnsSeed::new("seed.bitcoin.wiz.biz", x49));
                seeds.push(DnsSeed::new("bitcoin.seed.dlsouza.lol", x1000));
            }
            Network::Signet => {
                seeds.push(DnsSeed::new("signet.seed.dlsouza.lol", x1000));
                seeds.push(DnsSeed::new("seed.signet.bitcoin.sprovoost.nl", x49));
                seeds.push(DnsSeed::new("signet.seed.utreexo.net", x1009));
            }
            Network::Testnet => {
                seeds.push(DnsSeed::new("testnet-seed.bitcoin.jonasschnelli.ch", x49));
                seeds.push(DnsSeed::new("testnet.seed.dlsouza.lol", x1000));
                seeds.push(DnsSeed::new("seed.tbtc.petertodd.org", x49));
                seeds.push(DnsSeed::new("seed.testnet.bitcoin.sprovoost.nl", x49));
                seeds.push(DnsSeed::new("testnet-seed.bluematt.me", none));
            }
            Network::Testnet4 => {
                seeds.push(DnsSeed::new("seed.testnet4.bitcoin.sprovoost.nl", none));
                seeds.push(DnsSeed::new("seed.testnet4.wiz.biz", none));
            }
            Network::Regtest => {}
        };

        seeds
    }

    fn buried_deployments_for(&self) -> Vec<(BuriedDeployment, u32)> {
        let mut bip34_height = self.params.bip34_height;
        let mut bip65_height = self.params.bip65_height;
        let mut bip66_height = self.params.bip66_height;

        // TODO: remove this regtest override once the rust-bitcoin dependency includes
        // the fix from https://github.com/rust-bitcoin/rust-bitcoin/pull/6998.
        if self.network == Network::Regtest {
            bip34_height = 1;
            bip65_height = 1;
            bip66_height = 1;
        }

        vec![
            (BuriedDeployment::Bip34, bip34_height),
            (BuriedDeployment::Bip65, bip65_height),
            (BuriedDeployment::Bip66, bip66_height),
            (BuriedDeployment::Csv, self.csv_activation_height),
            (BuriedDeployment::Segwit, self.segwit_activation_height),
        ]
    }

    fn get_script_flags(&self, hash: BlockHash, height: u32) -> HashSet<String> {
        //Helper function to create a `HashSet` from an array of str.
        fn flags_set(flags: &[&str]) -> HashSet<String> {
            flags.iter().map(|f| (*f).to_string()).collect()
        }

        // BIP16 didn't become active until Apr 1 2012 (on mainnet, and retroactively applied to testnet)
        // However, only one historical block violated the P2SH rules (on both mainnet and testnet).
        // Similarly, only one historical block violated the TAPROOT rules on mainnet.
        // For simplicity, always leave P2SH+WITNESS+TAPROOT on except for the two violating blocks.
        let mut flags = match (self.network, hash) {
            (Network::Bitcoin, h)
                if h == bhash!(
                    "00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22"
                ) =>
            {
                HashSet::new()
            }
            (Network::Bitcoin, h)
                if h == bhash!(
                    "0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad"
                ) =>
            {
                flags_set(&["P2SH", "WITNESS"])
            }
            (Network::Testnet, h)
                if h == bhash!(
                    "00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105"
                ) =>
            {
                HashSet::new()
            }
            // Only mainnet and testnet3 require additional validation. The other networks do not.
            _ => flags_set(&["P2SH", "WITNESS", "TAPROOT"]),
        };

        for (deployment, activation_height) in self.buried_deployments_for() {
            if height >= activation_height {
                flags.extend(
                    deployment
                        .to_script_flags()
                        .iter()
                        .map(|f| (*f).to_string()),
                );
            }
        }

        flags
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
    #[cfg(feature = "bitcoinkernel")]
    use crate::BlockValidationErrors;
    use crate::BlockchainError;
    use crate::UtxoData;
    #[cfg(feature = "bitcoinkernel")]
    use crate::extensions::signet_utils::SIGNET_HEADER;
    #[cfg(feature = "bitcoinkernel")]
    use crate::extensions::signet_utils::fetch_and_clear_signet_solution;
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

        fn get_chain_params(&self) -> ChainParams {
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

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn default_signet_block_satisfies_challenge() {
        let block: Block = deserialize_hex(SIGNET_BLOCK_ONE).expect("valid signet block");
        let consensus = Consensus::from(Network::Signet);

        assert!(block.check_signet_challenge(&consensus.parameters));
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

        assert!(!block.check_signet_challenge(&params));
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
        assert!(block.check_signet_challenge(&params));

        params.signet_challenge = Some(ScriptBuf::from_bytes(vec![0x00]));
        assert!(!block.check_signet_challenge(&params));
    }
}
