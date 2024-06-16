//! A collection of functions that implement the consensus rules for the Bitcoin Network.
//! This module contains functions that are used to verify blocks and transactions, and doesn't
//! assume anything about the chainstate, so it can be used in any context.
//! We use this to avoid code reuse among the different implementations of the chainstate.

extern crate alloc;

use core::ffi::c_uint;
use core::ops::Mul;

use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::Encodable;
use bitcoin::hashes::sha256;
use bitcoin::hashes::Hash;
use bitcoin::pow::U256;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::Target;
use bitcoin::Transaction;
use bitcoin::TxOut;
use floresta_common::prelude::*;
use rustreexo::accumulator::node_hash::NodeHash;
use rustreexo::accumulator::proof::Proof;
use rustreexo::accumulator::stump::Stump;
use sha2::Digest;
use sha2::Sha512_256;

use super::chainparams::ChainParams;
use super::error::BlockValidationErrors;
use super::error::BlockchainError;

/// The value of a single coin in satoshis.
pub const COIN_VALUE: u64 = 100_000_000;

/// The version tag to be prepended to the leafhash. It's just the sha512 hash of the string
/// `UtreexoV1` represented as a vector of [u8] ([85 116 114 101 101 120 111 86 49]).
/// The same tag is "5574726565786f5631" as a hex string.
pub const UTREEXO_TAG_V1: [u8; 64] = [
    0x5b, 0x83, 0x2d, 0xb8, 0xca, 0x26, 0xc2, 0x5b, 0xe1, 0xc5, 0x42, 0xd6, 0xcc, 0xed, 0xdd, 0xa8,
    0xc1, 0x45, 0x61, 0x5c, 0xff, 0x5c, 0x35, 0x72, 0x7f, 0xb3, 0x46, 0x26, 0x10, 0x80, 0x7e, 0x20,
    0xae, 0x53, 0x4d, 0xc3, 0xf6, 0x42, 0x99, 0x19, 0x99, 0x31, 0x77, 0x2e, 0x03, 0x78, 0x7d, 0x18,
    0x15, 0x6e, 0xb3, 0x15, 0x1e, 0x0e, 0xd1, 0xb3, 0x09, 0x8b, 0xdc, 0x84, 0x45, 0x86, 0x18, 0x85,
];

/// This struct contains all the information and methods needed to validate a block,
/// it is used by the [ChainState] to validate blocks and transactions.
#[derive(Debug, Clone)]
pub struct Consensus {
    /// The parameters of the chain we are validating, it is usually hardcoded
    /// constants. See [ChainParams] for more information.
    pub parameters: ChainParams,
}

impl Consensus {
    /// Returns the amount of block subsidy to be paid in a block, given it's height.
    /// Bitcoin Core source: https://github.com/bitcoin/bitcoin/blob/2b211b41e36f914b8d0487e698b619039cc3c8e2/src/validation.cpp#L1501-L1512
    pub fn get_subsidy(&self, height: u32) -> u64 {
        let halvings = height / self.parameters.subsidy_halving_interval as u32;
        // Force block reward to zero when right shift is undefined.
        if halvings >= 64 {
            return 0;
        }
        let mut subsidy = 50 * COIN_VALUE;
        // Subsidy is cut in half every 210,000 blocks which will occur approximately every 4 years.
        subsidy >>= halvings;
        subsidy
    }

    /// Returns the hash of a leaf node in the utreexo accumulator.
    #[inline]
    fn get_leaf_hashes(
        transaction: &Transaction,
        vout: u32,
        height: u32,
        block_hash: BlockHash,
    ) -> sha256::Hash {
        let header_code = height << 1;

        let mut ser_utxo = Vec::new();
        let utxo = transaction.output.get(vout as usize).unwrap();
        utxo.consensus_encode(&mut ser_utxo).unwrap();
        let header_code = if transaction.is_coinbase() {
            header_code | 1
        } else {
            header_code
        };

        let leaf_hash = Sha512_256::new()
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(block_hash)
            .chain_update(transaction.txid())
            .chain_update(vout.to_le_bytes())
            .chain_update(header_code.to_le_bytes())
            .chain_update(ser_utxo)
            .finalize();
        sha256::Hash::from_slice(leaf_hash.as_slice())
            .expect("parent_hash: Engines shouldn't be Err")
    }
    /// Verify if all transactions in a block are valid. Here we check the following:
    /// - The block must contain at least one transaction, and this transaction must be coinbase
    /// - The first transaction in the block must be coinbase
    /// - The coinbase transaction must have the correct value (subsidy + fees)
    /// - The block must not create more coins than allowed
    /// - All transactions must be valid:
    ///     - The transaction must not be coinbase (already checked)
    ///     - The transaction must not have duplicate inputs
    ///     - The transaction must not spend more coins than it claims in the inputs
    ///     - The transaction must have valid scripts
    #[allow(unused)]
    pub fn verify_block_transactions(
        mut utxos: HashMap<OutPoint, TxOut>,
        transactions: &[Transaction],
        subsidy: u64,
        verify_script: bool,
        flags: c_uint,
    ) -> Result<(), BlockchainError> {
        // Blocks must contain at least one transaction
        if transactions.is_empty() {
            return Err(BlockValidationErrors::EmptyBlock.into());
        }
        let mut fee = 0;
        // Skip the coinbase tx
        for (n, transaction) in transactions.iter().enumerate() {
            // We don't need to verify the coinbase inputs, as it spends newly generated coins
            if transaction.is_coinbase() {
                if n == 0 {
                    continue;
                }
                // A block must contain only one coinbase, and it should be the fist thing inside it
                return Err(BlockValidationErrors::FirstTxIsnNotCoinbase.into());
            }
            // Amount of all outputs
            let output_value = transaction
                .output
                .iter()
                .fold(0, |acc, tx| acc + tx.value.to_sat());
            // Amount of all inputs
            let in_value = transaction.input.iter().fold(0, |acc, input| {
                acc + utxos
                    .get(&input.previous_output)
                    .expect("We have all prevouts here")
                    .value
                    .to_sat()
            });
            // Value in should be greater or equal to value out. Otherwise, inflation.
            if output_value > in_value {
                return Err(BlockValidationErrors::NotEnoughMoney.into());
            }
            // Fee is the difference between inputs and outputs
            fee += in_value - output_value;
            // Verify the tx script
            #[cfg(feature = "bitcoinconsensus")]
            if verify_script {
                transaction
                    .verify_with_flags(|outpoint| utxos.remove(outpoint), flags)
                    .map_err(|err| BlockValidationErrors::InvalidTx(alloc::format!("{:?}", err)))?;
            }
        }
        // In each block, the first transaction, and only the first, should be coinbase
        if !transactions[0].is_coinbase() {
            return Err(BlockValidationErrors::FirstTxIsnNotCoinbase.into());
        }
        // Checks if the miner isn't trying to create inflation
        if fee + subsidy
            < transactions[0]
                .output
                .iter()
                .fold(0, |acc, out| acc + out.value.to_sat())
        {
            return Err(BlockValidationErrors::BadCoinbaseOutValue.into());
        }
        Ok(())
    }
    /// Calculates the next target for the proof of work algorithm, given the
    /// current target and the time it took to mine the last 2016 blocks.
    pub fn calc_next_work_required(
        last_block: &BlockHeader,
        first_block: &BlockHeader,
        params: ChainParams,
    ) -> Target {
        let cur_target = last_block.target().0;

        let expected_timespan = U256::from(params.pow_target_timespan);
        let mut actual_timespan = last_block.time - first_block.time;

        // Difficulty adjustments are limited, to prevent large swings in difficulty
        // caused by malicious miners.
        if actual_timespan < params.pow_target_timespan as u32 / 4 {
            actual_timespan = params.pow_target_timespan as u32 / 4;
        }
        if actual_timespan > params.pow_target_timespan as u32 * 4 {
            actual_timespan = params.pow_target_timespan as u32 * 4;
        }

        let new_target = cur_target.mul(actual_timespan.into());
        let new_target = new_target / expected_timespan;
        Target(new_target)
    }
    /// Updates our accumulator with the new block. This is done by calculating the new
    /// root hash of the accumulator, and then verifying the proof of inclusion of the
    /// deleted nodes. If the proof is valid, we return the new accumulator. Otherwise,
    /// we return an error.
    /// This function is pure, it doesn't modify the accumulator, but returns a new one.
    pub fn update_acc(
        acc: &Stump,
        block: &Block,
        height: u32,
        proof: Proof,
        del_hashes: Vec<sha256::Hash>,
    ) -> Result<Stump, BlockchainError> {
        let block_hash = block.block_hash();
        let mut leaf_hashes = Vec::new();
        let del_hashes = del_hashes
            .iter()
            .map(|hash| NodeHash::from(hash.as_byte_array()))
            .collect::<Vec<_>>();
        // Verify the proof of inclusion of the deleted nodes
        if !acc.verify(&proof, &del_hashes)? {
            return Err(BlockValidationErrors::InvalidProof.into());
        }
        // Get inputs from the block, we'll need this HashSet to check if an output is spent
        // in the same block. If it is, we don't need to add it to the accumulator.
        let mut block_inputs = HashSet::new();
        for transaction in block.txdata.iter() {
            for input in transaction.input.iter() {
                block_inputs.insert((input.previous_output.txid, input.previous_output.vout));
            }
        }

        // Get all leaf hashes that will be added to the accumulator
        for transaction in block.txdata.iter() {
            for (i, output) in transaction.output.iter().enumerate() {
                if !Self::is_unspendable(&output.script_pubkey)
                    && !block_inputs.contains(&(transaction.txid(), i as u32))
                {
                    leaf_hashes.push(Self::get_leaf_hashes(
                        transaction,
                        i as u32,
                        height,
                        block_hash,
                    ))
                }
            }
        }
        // Convert the leaf hashes to NodeHashes used in Rustreexo
        let hashes: Vec<NodeHash> = leaf_hashes
            .iter()
            .map(|&hash| NodeHash::from(hash.as_byte_array()))
            .collect();
        // Update the accumulator
        let acc = acc.modify(&hashes, &del_hashes, &proof)?.0;
        Ok(acc)
    }

    fn is_unspendable(script: &ScriptBuf) -> bool {
        if script.len() > 10_000 {
            return true;
        }

        if !script.is_empty() && script.as_bytes()[0] == 0x6a {
            return true;
        }

        false
    }
}
