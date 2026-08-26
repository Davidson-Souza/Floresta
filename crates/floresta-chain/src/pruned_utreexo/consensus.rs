// SPDX-License-Identifier: MIT OR Apache-2.0

//! A collection of functions that implement the consensus rules for the Bitcoin Network.
//! This module contains functions that are used to verify blocks and transactions, and doesn't
//! assume anything about the chainstate, so it can be used in any context.
//! We use this to avoid code reuse among the different implementations of the chainstate.

extern crate alloc;

use core::ffi::c_uint;

use bitcoin::Amount;
use bitcoin::Block;
use bitcoin::CompactTarget;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::Target;
use bitcoin::Transaction;
use bitcoin::TxIn;
use bitcoin::Txid;
use bitcoin::absolute;
use bitcoin::block::Header as BlockHeader;
use bitcoin::blockdata::Weight;
#[cfg(feature = "bitcoinkernel")]
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use bitcoin::merkle_tree;
use bitcoin::script;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::BLOCK_CHECK_BASE;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::BLOCK_CHECK_MERKLE;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::BlockCheckResult;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::BlockValidationResult;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::PrecomputedTransactionData;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::TxCheckResult;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::prelude::BlockValidationStateExt as _;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::prelude::TransactionExt as _;
use floresta_common::prelude::*;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::proof::Proof;
use rustreexo::stump::Stump;
use swift_sync_agg::SwiftSyncAgg;

use super::chainparams::ChainParams;
use super::error::BlockValidationErrors;
use super::error::BlockchainError;
use super::udata;
use crate::TransactionError;
use crate::extensions::Bip30UnspendableExt;
use crate::pruned_utreexo::utxo_data::UtxoData;
use crate::swift_sync_agg::SipHashKeys;
use crate::swift_sync_agg::TxidHashMidstate;

/// Maximum halving count before the subsidy shift exceeds a `u64`.
const MAX_SUBSIDY_HALVINGS: u32 = u64::BITS;

/// Maximum script length in bytes, as defined [in Bitcoin Core](https://github.com/bitcoin/bitcoin/blob/v30.0/src/script/script.h#L40).
const MAX_SCRIPT_SIZE: usize = 10_000;

/// The version tag to be prepended to the leafhash. It's just the sha512 hash of the string
/// `UtreexoV1` represented as a vector of [u8] ([85 116 114 101 101 120 111 86 49]).
/// The same tag is "5574726565786f5631" as a hex string.
pub const UTREEXO_TAG_V1: [u8; 64] = [
    0x5b, 0x83, 0x2d, 0xb8, 0xca, 0x26, 0xc2, 0x5b, 0xe1, 0xc5, 0x42, 0xd6, 0xcc, 0xed, 0xdd, 0xa8,
    0xc1, 0x45, 0x61, 0x5c, 0xff, 0x5c, 0x35, 0x72, 0x7f, 0xb3, 0x46, 0x26, 0x10, 0x80, 0x7e, 0x20,
    0xae, 0x53, 0x4d, 0xc3, 0xf6, 0x42, 0x99, 0x19, 0x99, 0x31, 0x77, 0x2e, 0x03, 0x78, 0x7d, 0x18,
    0x15, 0x6e, 0xb3, 0x15, 0x1e, 0x0e, 0xd1, 0xb3, 0x09, 0x8b, 0xdc, 0x84, 0x45, 0x86, 0x18, 0x85,
];

/// The unspendable UTXO on block 91_722 that exists because of the historical
/// [BIP30 violation](https://bips.dev/30/). For Utreexo, this UTXO is not overwritten
/// as we commit the block hash in the leafhash. But since non-Utreexo nodes consider
/// this as unspendable as it's already been overwritten, we also need to make it not spendable.
///
/// Encoded in hex string is 84b3af0783b410b4564c5d1f361868559f7cf77cfc65ce2be951210357022fe3.
pub const UNSPENDABLE_BIP30_UTXO_91722: [u8; 32] = [
    0x84, 0xb3, 0xaf, 0x07, 0x83, 0xb4, 0x10, 0xb4, 0x56, 0x4c, 0x5d, 0x1f, 0x36, 0x18, 0x68, 0x55,
    0x9f, 0x7c, 0xf7, 0x7c, 0xfc, 0x65, 0xce, 0x2b, 0xe9, 0x51, 0x21, 0x03, 0x57, 0x02, 0x2f, 0xe3,
];

/// The unspendable UTXO on block 91_812 that exists because of the historical
/// [BIP30 violation](https://bips.dev/30/). For Utreexo, this UTXO is not overwritten
/// as we commit the block hash in the leafhash. But since non-Utreexo nodes consider
/// this as unspendable as it's already been overwritten, we also need to make it not spendable.
///
/// Encoded in hex string is bc6b4bf7cebbd33a18d6b0fe1f8ecc7aa5403083c39ee343b985d51fd0295ad8.
pub const UNSPENDABLE_BIP30_UTXO_91812: [u8; 32] = [
    0xbc, 0x6b, 0x4b, 0xf7, 0xce, 0xbb, 0xd3, 0x3a, 0x18, 0xd6, 0xb0, 0xfe, 0x1f, 0x8e, 0xcc, 0x7a,
    0xa5, 0x40, 0x30, 0x83, 0xc3, 0x9e, 0xe3, 0x43, 0xb9, 0x85, 0xd5, 0x1f, 0xd0, 0x29, 0x5a, 0xd8,
];

/// This struct contains all the information and methods needed to validate a block,
/// it is used by the [ChainState](crate::ChainState) to validate blocks and transactions.
#[derive(Debug, Clone)]
pub struct Consensus {
    /// The parameters of the chain we are validating, it is usually hardcoded
    /// constants. See [ChainParams] for more information.
    pub parameters: ChainParams,
}

impl From<Network> for Consensus {
    fn from(network: Network) -> Self {
        Self {
            parameters: network.into(),
        }
    }
}

impl Consensus {
    /// Returns the block-finality lock-time cutoff for a candidate block.
    ///
    /// Before CSV/BIP113 activation this is the block header time. After activation
    /// it is the previous block's Median Time Past (MTP).
    pub(crate) fn block_lock_time_cutoff<E>(
        &self,
        height: u32,
        header: &BlockHeader,
        previous_median_time_past: impl FnOnce() -> Result<u32, E>,
    ) -> Result<u32, E> {
        if height >= self.parameters.csv_activation_height {
            previous_median_time_past()
        } else {
            Ok(header.time)
        }
    }

    /// Returns the amount of block subsidy to be paid in a block, given its height.
    ///
    /// The Bitcoin Core source can be found [here](https://github.com/bitcoin/bitcoin/blob/2b211b41e36f914b8d0487e698b619039cc3c8e2/src/validation.cpp#L1501-L1512).
    pub fn get_subsidy(&self, height: u32) -> Amount {
        let halvings = height / self.parameters.subsidy_halving_interval.get();
        // Force block reward to zero when right shift is undefined.
        if halvings >= MAX_SUBSIDY_HALVINGS {
            return Amount::ZERO;
        }

        let mut subsidy = 50 * Amount::ONE_BTC.to_sat();
        // Subsidy is cut in half every 210,000 blocks which will occur approximately every 4 years.
        subsidy >>= halvings;
        Amount::from_sat(subsidy)
    }

    /// Maximum theoretical supply at the given height. Excludes the unspendable genesis subsidy.
    ///
    /// # Panics
    ///
    /// Panics if `height` is `u32::MAX`.
    pub fn max_supply_at_height(&self, height: u32) -> Amount {
        let blocks_per_epoch = self.parameters.subsidy_halving_interval.get();

        // Block count includes the genesis block
        let block_count = height.checked_add(1).expect("height must be < u32::MAX");

        // Full epochs completely included + remainder blocks in the next epoch
        let full_epochs = block_count / blocks_per_epoch;
        let rem_blocks = block_count % blocks_per_epoch;

        let epoch_subsidy = |epoch: u32| self.get_subsidy(blocks_per_epoch * epoch).to_sat();

        let mut total: u64 = 0;

        // Sum full epochs before the subsidy shift limit
        for epoch in 0..full_epochs.min(MAX_SUBSIDY_HALVINGS) {
            total += u64::from(blocks_per_epoch) * epoch_subsidy(epoch);
        }

        // Add remainder in the current epoch, if before the subsidy shift limit
        if full_epochs < MAX_SUBSIDY_HALVINGS {
            total += u64::from(rem_blocks) * epoch_subsidy(full_epochs);
        }

        // Exclude the unspendable genesis subsidy, which was included above
        total -= epoch_subsidy(0);

        Amount::from_sat(total)
    }

    /// A script is unspendable if its length is larger than 10,000 bytes or if it starts with an
    /// `OP_RETURN`. This follows the
    /// [Bitcoin Core implementation](https://github.com/bitcoin/bitcoin/blob/v30.0/src/script/script.h#L571).
    pub fn is_unspendable(script: &ScriptBuf) -> bool {
        script.len() > MAX_SCRIPT_SIZE || script.is_op_return()
    }

    /// Verify if all transactions in a block are valid. Here we check the following:
    /// - The block must contain at least one transaction, and this transaction must be coinbase
    /// - The first transaction in the block must be coinbase
    /// - The coinbase transaction must have the correct value (subsidy + fees)
    /// - The block must not create more coins than allowed
    /// - No output may be spent more than once within the block
    /// - All transactions must be valid, as verified by [`Consensus::verify_transaction`]
    #[allow(unused)]
    pub fn verify_block_transactions(
        height: u32,
        lock_time_cutoff: u32,
        mut utxos: HashMap<OutPoint, UtxoData>,
        transactions: &[Transaction],
        subsidy: Amount,
        verify_script: bool,
        flags: c_uint,
    ) -> Result<(), BlockchainError> {
        // Blocks must contain at least one transaction (i.e., the coinbase)
        if transactions.is_empty() {
            Err(BlockValidationErrors::EmptyBlock)?;
        }

        // Total block fees that the miner can claim in the coinbase
        let mut fee = Amount::ZERO;

        for (n, transaction) in transactions.iter().enumerate() {
            if !Self::is_final_transaction(transaction, height, lock_time_cutoff) {
                Err(BlockValidationErrors::NonFinalTransaction)?;
            }

            if n == 0 {
                if !transaction.is_coinbase() {
                    Err(BlockValidationErrors::FirstTxIsNotCoinbase)?;
                }
                Self::verify_coinbase(transaction)?;
                // Skip next checks: coinbase input is exempt, coinbase reward checked later
                continue;
            }

            // Actually verify the transaction
            let (in_value, out_value) =
                Self::verify_transaction(transaction, &mut utxos, height, verify_script, flags)?;

            // Fee is the difference between inputs and outputs. In the above function call we have
            // verified that `out_value <= in_value` (no underflow risk).
            fee = fee
                .checked_add(in_value - out_value)
                .ok_or(BlockValidationErrors::TooManyCoins)?;
        }

        // Check coinbase output values to ensure the miner isn't producing excess coins
        let allowed_reward = fee
            .checked_add(subsidy)
            .ok_or(BlockValidationErrors::TooManyCoins)?;

        let coinbase_total = Self::total_out_value(&transactions[0])?;

        if coinbase_total > allowed_reward {
            Err(BlockValidationErrors::BadCoinbaseOutValue)?;
        }

        Ok(())
    }

    /// Performs all transaction checks that are independent of the spent outputs and produces
    /// a [`SwiftSyncAgg`] given the `unspent_indexes` hints and a secret `salt`. This is the
    /// AssumeValid SwiftSync version of [`Consensus::verify_block_transactions`].
    ///
    /// This function calls [`Consensus::check_transaction_context_free`] since previous outputs
    /// are not available (we assume the unlocking script is valid). Then, it removes all input
    /// `OutPoint`s from the aggregator and adds all hinted-as-spent outputs to it.
    ///
    /// Returns the resulting aggregator and the total unspent amount that has been locked in
    /// this block.
    fn verify_block_transactions_swiftsync(
        height: u32,
        block: &Block,
        txids: Vec<Txid>,
        unspent_indexes: HashSet<u32>,
        salt: &SipHashKeys,
    ) -> Result<(SwiftSyncAgg, Amount), BlockchainError> {
        let transactions = &block.txdata;
        assert_eq!(transactions.len(), txids.len());

        // Blocks must contain at least one transaction (i.e., the coinbase)
        if transactions.is_empty() {
            Err(BlockValidationErrors::EmptyBlock)?;
        }

        // The block-wide output index to compare against unspent output hints
        let mut output_index = 0;
        let mut unspent_amount = Amount::ZERO;
        let mut agg = SwiftSyncAgg::zero();

        for (n, (transaction, txid)) in transactions.iter().zip(txids).enumerate() {
            if n == 0 {
                if !transaction.is_coinbase() {
                    Err(BlockValidationErrors::FirstTxIsNotCoinbase)?;
                }
                Self::verify_coinbase(transaction)?;
                let coinbase_total = Self::total_out_value(transaction)?;

                // We don't know how much money is paid in fees (it would require input amounts),
                // so we can't check the exact amount here
                if coinbase_total > Amount::MAX_MONEY {
                    Err(BlockValidationErrors::TooManyCoins)?;
                }

                // Skip BIP-30 unspendable coinbase outputs
                if block.is_bip30_unspendable(height) {
                    continue;
                }
            } else {
                // Verify the non-coinbase transaction and remove the inputs from the aggregator
                Self::check_transaction_context_free(transaction)?;

                for input in transaction.input.iter() {
                    agg.remove(salt, &input.previous_output);
                }
            }

            let mut spent_vouts = Vec::new();
            for (vout, out) in transaction.output.iter().enumerate() {
                // Special case: unspendable outputs do not count for the block `output_index`
                if Self::is_unspendable(&out.script_pubkey) {
                    unspent_amount += out.value;
                    continue;
                }

                // According to the hints, is this output unspent? If not, add it to the aggregator
                let hinted_unspent = unspent_indexes.contains(&output_index);

                if hinted_unspent {
                    unspent_amount += out.value;
                } else {
                    spent_vouts.push(vout as u32);
                }
                output_index += 1;
            }
            // Only add spent outputs to the aggregator
            Self::add_outputs_to_agg(salt, &mut agg, txid, spent_vouts);
        }

        Ok((agg, unspent_amount))
    }

    /// Helper to compute the outputs `OutPoint` hashes efficiently and add them to the aggregator.
    ///
    /// Returns immediately if `spent_vouts` is empty. Otherwise, adds the corresponding
    /// `txid:vout` outpoints. For multiple `vout`s, it computes the `txid` hash midstate once and
    /// reuses it for each `vout`.
    fn add_outputs_to_agg(
        salt: &SipHashKeys,
        agg: &mut SwiftSyncAgg,
        txid: Txid,
        spent_vouts: Vec<u32>,
    ) {
        match spent_vouts.len() {
            // Nothing needs to be added to the aggregator
            0 => {}

            // Only one `OutPoint` to add
            1 => agg.add(salt, &OutPoint::new(txid, spent_vouts[0])),

            // All `OutPoints` to hash here will share the same txid, only differing in the vout.
            // Thus, we can compute the midstate once, amortizing this cost for all `OutPoints`,
            // which is around 57% less work given enough spent outputs.
            _ => {
                let midstate = TxidHashMidstate::new(salt, txid.as_byte_array());
                for vout in spent_vouts {
                    agg.add_with_vout(midstate.clone(), vout);
                }
            }
        }
    }

    /// Returns the total sum of money in the outputs of the given transaction.
    fn total_out_value(transaction: &Transaction) -> Result<Amount, BlockchainError> {
        let mut value = Amount::ZERO;
        for out in &transaction.output {
            value = value
                .checked_add(out.value)
                .ok_or(BlockValidationErrors::TooManyCoins)?;
        }

        Ok(value)
    }

    /// Verifies a single, non-coinbase transaction. To verify (the structure of) a coinbase
    /// transaction, use [`Consensus::verify_coinbase`].
    ///
    /// This function checks that the transaction:
    ///   - Has at least one input and one output
    ///   - Doesn't have null PrevOuts (reserved only for coinbase transactions)
    ///   - Doesn't spend more coins than it claims in the inputs
    ///   - Doesn't "move" more coins than allowed (at most 21 million)
    ///   - Spends mature coins, in case any input refers to a coinbase transaction
    ///   - Has valid scripts (if we don't assume them), and within the allowed size
    ///
    /// Inputs are removed from `utxos` as they are loaded, independently of script validation.
    /// This makes them unavailable to later transactions in the same block. Callers should treat
    /// `utxos` as scratch validation state and discard it if this function returns an error.
    pub fn verify_transaction(
        transaction: &Transaction,
        utxos: &mut HashMap<OutPoint, UtxoData>,
        height: u32,
        _verify_script: bool,
        _flags: c_uint,
    ) -> Result<(Amount, Amount), BlockchainError> {
        let txid = || transaction.compute_txid();

        let out_value = Self::check_transaction_context_free(transaction)?;

        // Take ownership of the inputs so later transactions cannot spend them again.
        // This vector keeps the UTXOs in tx-input order, as required by bitcoinkernel.
        let mut spent_utxos = Vec::with_capacity(transaction.input.len());

        let mut in_value = Amount::ZERO;
        for input in &transaction.input {
            let utxo = Self::take_utxo(input, utxos, txid)?;
            let txout = &utxo.txout;

            // A coinbase output created at height n can only be spent at height >= n + 100
            if utxo.is_coinbase && (height < utxo.creation_height + 100) {
                Err(tx_err!(txid, CoinbaseNotMatured))?;
            }

            // Check script sizes (spent txo pubkey, inputs are covered already)
            Self::validate_script_size(&txout.script_pubkey, txid)?;

            in_value = in_value
                .checked_add(txout.value)
                .ok_or(BlockValidationErrors::TooManyCoins)?;

            spent_utxos.push(utxo);
        }

        // Sanity check
        if in_value > Amount::MAX_MONEY {
            Err(BlockValidationErrors::TooManyCoins)?;
        }

        // Value in should be greater or equal to value out. Otherwise, inflation.
        if out_value > in_value {
            Err(tx_err!(txid, NotEnoughMoney))?;
        }

        // Verify the tx script
        #[cfg(feature = "bitcoinkernel")]
        if _verify_script {
            Self::verify_input_scripts(transaction, &spent_utxos, _flags)?;
        }

        Ok((in_value, out_value))
    }

    #[cfg(feature = "bitcoinkernel")]
    /// Verifies each input script against the transaction-local UTXOs, in input order.
    fn verify_input_scripts(
        transaction: &Transaction,
        spent_utxos: &[UtxoData],
        flags: c_uint,
    ) -> Result<(), BlockchainError> {
        let tx = serialize(&transaction);
        let txid = || transaction.compute_txid();

        let tx = bitcoinkernel::Transaction::try_from(tx.as_slice())
            .map_err(|e| tx_err!(txid, ScriptValidationError, e.to_string()))?;

        let mut prevouts = Vec::with_capacity(spent_utxos.len());
        let mut prevout_data = Vec::with_capacity(spent_utxos.len());

        for utxo in spent_utxos {
            let txout = &utxo.txout;

            let value =
                i64::try_from(txout.value.to_sat()).map_err(|_| tx_err!(txid, TooManyCoins))?;
            let spk = bitcoinkernel::ScriptPubkey::try_from(txout.script_pubkey.as_bytes())
                .map_err(|e| tx_err!(txid, ScriptValidationError, e.to_string()))?;

            prevouts.push(bitcoinkernel::TxOut::new(&spk, value));
            prevout_data.push((spk, value));
        }

        let tx_data = PrecomputedTransactionData::new(&tx, &prevouts)
            .map_err(|e| tx_err!(txid, ScriptValidationError, e.to_string()))?;

        for (input_index, (script, amount)) in prevout_data.iter().enumerate() {
            bitcoinkernel::verify(
                script,
                Some(*amount),
                &tx,
                input_index,
                Some(flags),
                &tx_data,
            )
            .map_err(|e| tx_err!(txid, ScriptValidationError, e.to_string()))?;
        }

        Ok(())
    }

    /// Returns `true` if the transaction contains duplicate inputs
    /// (the same `OutPoint` is spent more than once).
    ///
    /// Optimized for the most common cases: over 80% of Bitcoin transactions
    /// have a single input (see <https://mainnet.observer/charts/transactions-1in/>),
    /// which is handled with no allocation. Two-input transactions are handled
    /// with a single equality check. Only transactions with three or more inputs
    /// fall back to a `HashSet`.
    fn has_duplicate_inputs(inputs: &[TxIn]) -> bool {
        match inputs.len() {
            1 => false,
            2 => inputs[0].previous_output == inputs[1].previous_output,
            _ => {
                let mut seen = HashSet::with_capacity(inputs.len());
                for input in inputs {
                    if !seen.insert(&input.previous_output) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Performs consensus checks that are independent of the spent outputs (non-coinbase only).
    /// Returns the total output value as an [`Amount`].
    pub fn check_transaction_context_free(
        transaction: &Transaction,
    ) -> Result<Amount, BlockchainError> {
        #[cfg(feature = "bitcoinkernel")]
        {
            let txid = || transaction.compute_txid();
            let transaction_bytes = serialize(transaction);
            let kernel_transaction =
                bitcoinkernel::Transaction::try_from(transaction_bytes.as_slice())
                    .map_err(|e| tx_err!(txid, ContextFreeValidation, e.to_string()))?;

            match kernel_transaction.check() {
                TxCheckResult::Valid => {
                    // `CheckTransaction` accepts coinbase transactions, while this entry point is
                    // specifically for non-coinbase transactions.
                    if transaction.is_coinbase() {
                        Err(tx_err!(txid, NullPrevOut))?;
                    }

                    Self::total_out_value(transaction)
                }
                TxCheckResult::Invalid(result) => {
                    // Preserve the more specific errors exposed by Floresta where possible.
                    Self::diagnose_transaction_context_free(transaction)?;

                    Err(tx_err!(txid, ContextFreeValidation, format!("{result:?}")).into())
                }
            }
        }

        #[cfg(not(feature = "bitcoinkernel"))]
        Self::diagnose_transaction_context_free(transaction)
    }

    /// Diagnoses context-free transaction failures using Floresta's error variants.
    fn diagnose_transaction_context_free(
        transaction: &Transaction,
    ) -> Result<Amount, BlockchainError> {
        let txid = || transaction.compute_txid();

        if transaction.input.is_empty() {
            Err(tx_err!(txid, EmptyInputs))?;
        }
        if transaction.output.is_empty() {
            Err(tx_err!(txid, EmptyOutputs))?;
        }

        for input in &transaction.input {
            // Null PrevOuts are only allowed in coinbase inputs
            if input.previous_output.is_null() {
                Err(tx_err!(txid, NullPrevOut))?;
            }

            // Witness size is intentionally not checked here — witness-specific
            // limits are enforced during script execution, not in context-free checks.
            // This matches Bitcoin Core's CheckTransaction() which explicitly skips
            // witness in context-free checks because witness data has not been
            // checked for malleability at this point.
            // See: https://github.com/bitcoin/bitcoin/blob/master/src/consensus/tx_check.cpp
            Self::validate_script_size(&input.script_sig, txid)?;
        }

        // Check for duplicate inputs (CVE-2018-17144).
        // UpdateCoins does not detect duplicates — a duplicate prevout causes either
        // a crash or an inflation bug depending on the coins database implementation.
        // Bitcoin Core catches this explicitly in CheckTransaction() for the same reason.
        // after:
        if Self::has_duplicate_inputs(&transaction.input) {
            Err(tx_err!(txid, DuplicateInput))?;
        }
        let out_value = Self::total_out_value(transaction)?;

        // Sanity check
        if out_value > Amount::MAX_MONEY {
            Err(BlockValidationErrors::TooManyCoins)?;
        }

        Ok(out_value)
    }

    /// Runs inexpensive, consensus-critical block checks that don't require script execution. If
    /// successful, returns the list of [`Txid`]s computed for the merkle root check.
    ///
    /// This verifies:
    /// - base context-free block rules (size, coinbase, transactions, and sigops)
    /// - the header merkle root matches the block's txids
    /// - BIP34 coinbase-encoded height once activated (at `bip34_height`)
    /// - if there are SegWit transactions, the witness commitment is present and correct
    /// - total block weight is within the 4,000,000 WU limit
    pub fn check_block(&self, block: &Block, height: u32) -> Result<Vec<Txid>, BlockchainError> {
        #[cfg(feature = "bitcoinkernel")]
        {
            let block_bytes = serialize(block);
            let kernel_block = bitcoinkernel::Block::try_from(block_bytes.as_slice())
                .map_err(|e| BlockValidationErrors::ContextFreeValidation(e.to_string()))?;
            let chain_params = bitcoinkernel::ChainParams::new(match self.parameters.network {
                Network::Bitcoin => bitcoinkernel::ChainType::Mainnet,
                Network::Testnet => bitcoinkernel::ChainType::Testnet,
                Network::Testnet4 => bitcoinkernel::ChainType::Testnet4,
                Network::Signet => bitcoinkernel::ChainType::Signet,
                Network::Regtest => bitcoinkernel::ChainType::Regtest,
            });

            // Header validation already checks proof of work, so only run the block-level checks.
            match kernel_block.check(&chain_params, BLOCK_CHECK_BASE | BLOCK_CHECK_MERKLE) {
                BlockCheckResult::Valid => {}
                BlockCheckResult::Invalid(state) => {
                    if state.result() == BlockValidationResult::Mutated {
                        Err(BlockValidationErrors::BadMerkleRoot)?;
                    }

                    // Keep established errors for checks Floresta can identify precisely.
                    if Self::check_merkle_root(block).is_none() {
                        Err(BlockValidationErrors::BadMerkleRoot)?;
                    }
                    if block.txdata.is_empty() {
                        Err(BlockValidationErrors::EmptyBlock)?;
                    }
                    if !block.txdata[0].is_coinbase() {
                        Err(BlockValidationErrors::FirstTxIsNotCoinbase)?;
                    }
                    if block.weight() > Weight::MAX_BLOCK {
                        Err(BlockValidationErrors::BlockTooBig)?;
                    }

                    Self::verify_coinbase(&block.txdata[0])?;
                    for transaction in &block.txdata[1..] {
                        if transaction.is_coinbase() {
                            Err(BlockValidationErrors::InvalidCoinbase(
                                "Block contains multiple coinbase transactions".into(),
                            ))?;
                        }
                        Self::check_transaction_context_free(transaction)?;
                    }

                    return Err(
                        BlockValidationErrors::ContextFreeValidation(format!("{state:?}")).into(),
                    );
                }
            }
        }

        let Some(txids) = Self::check_merkle_root(block) else {
            Err(BlockValidationErrors::BadMerkleRoot)?
        };

        let bip34_height = self.parameters.params.bip34_height;
        // If bip34 is active, check that the encoded block height is correct
        if height >= bip34_height && Self::get_bip34_height(block) != Some(height) {
            Err(BlockValidationErrors::BadBip34)?;
        }

        if !block.check_witness_commitment() {
            Err(BlockValidationErrors::BadWitnessCommitment)?;
        }

        if block.weight() > Weight::MAX_BLOCK {
            Err(BlockValidationErrors::BlockTooBig)?;
        }

        Ok(txids)
    }

    /// Checks if the merkle root of the header matches the merkle root of the transaction list.
    ///
    /// Unlike [`Block::check_merkle_root`], this function returns the list of computed [`Txid`]s
    /// if the merkle roots matched, or `None` otherwise.
    ///
    /// The merkle root is computed in the same way as [`Block::compute_merkle_root`].
    pub fn check_merkle_root(block: &Block) -> Option<Vec<Txid>> {
        let txids: Vec<_> = block.txdata.iter().map(|obj| obj.compute_txid()).collect();

        // Copy the hashes into an iterator, as `calculate_root` requires ownership
        let hashes_iter = txids.iter().copied().map(|txid| txid.to_raw_hash());

        let calculated = merkle_tree::calculate_root(hashes_iter).map(|h| h.into());
        match calculated {
            Some(merkle_root) if block.header.merkle_root == merkle_root => Some(txids),
            _ => None,
        }
    }

    /// Validates a block under AssumeValid SwiftSync, where previous outputs are unavailable,
    /// using the given `unspent_indexes` hints and the secret, uniformly random `salt`
    /// chosen for this IBD session.
    ///
    /// Returns the resulting [`SwiftSyncAgg`], together with the total unspent amount that
    /// has been locked in this block (i.e., the amount from outputs that remain unspent
    /// during SwiftSync).
    ///
    /// This performs all consensus checks that do not require previous outputs, then:
    /// - adds hinted spent outputs to the aggregator,
    /// - removes all non-coinbase inputs from the aggregator,
    /// - sums the value locked in hinted unspent outputs **and** unspendable outputs.
    ///
    /// The returned amount is used at the end of SwiftSync to check that the total created
    /// supply does not exceed the expected limit.
    ///
    /// #### Regarding amount checks
    ///
    /// Since previous outputs are unavailable, this does **not** verify the coinbase reward,
    /// which depends on total fees (out - in amounts). As a result, the total supply check
    /// described above is weaker than the amount checks performed in traditional AssumeValid.
    ///
    /// In theory, an attacker with majority hashpower and the ability to insert an invalid
    /// AssumeValid hash into the Floresta codebase could make us accept a chain with excess
    /// coins by claiming historically unclaimed block rewards, while still staying below the
    /// maximum theoretical supply that we check.
    ///
    /// In practice, this does not materially change the trust model, since an attacker with
    /// those capabilities could already make us accept invalid scripts and thereby steal far
    /// more coins than could be created from historically unclaimed block rewards.
    pub fn process_block_swiftsync(
        &self,
        block: &Block,
        height: u32,
        unspent_indexes: HashSet<u32>,
        salt: &SipHashKeys,
    ) -> Result<(SwiftSyncAgg, Amount), BlockchainError> {
        let txids = self.check_block(block, height)?;

        Self::verify_block_transactions_swiftsync(height, block, txids, unspent_indexes, salt)
    }

    /// Removes and returns the UTXO spent by `input`.
    ///
    /// Removing inputs here is consensus-critical: a later transaction in the same block must not
    /// be able to spend the same output. Fails if the UTXO is missing or was already spent.
    fn take_utxo<F: Fn() -> Txid>(
        input: &TxIn,
        utxos: &mut HashMap<OutPoint, UtxoData>,
        txid: F,
    ) -> Result<UtxoData, TransactionError> {
        utxos
            .remove(&input.previous_output)
            .ok_or_else(|| tx_err!(txid, UtxoNotFound, input.previous_output))
    }

    /// Returns whether a transaction's absolute lock time is final for a candidate block.
    ///
    /// Zero `nLockTime` is final. Otherwise, height and time locks must be below
    /// `height` and `lock_time_cutoff`, respectively. `nLockTime` is disabled when
    /// every input has a final `nSequence`.
    ///
    /// The cutoff is the block timestamp before BIP113 and the previous block's
    /// Median Time Past (MTP) afterward.
    ///
    /// Mirrors Bitcoin Core's IsFinalTx:
    /// <https://github.com/bitcoin/bitcoin/blob/v31.0/src/consensus/tx_verify.cpp#L17-L37>
    fn is_final_transaction(transaction: &Transaction, height: u32, lock_time_cutoff: u32) -> bool {
        if transaction.lock_time == absolute::LockTime::ZERO {
            return true;
        }

        let lock_time_reached = match transaction.lock_time {
            absolute::LockTime::Blocks(lock_height) => lock_height.to_consensus_u32() < height,
            absolute::LockTime::Seconds(lock_time) => {
                lock_time.to_consensus_u32() < lock_time_cutoff
            }
        };

        lock_time_reached
            || transaction
                .input
                .iter()
                .all(|input| input.sequence.is_final())
    }

    /// Validates the script size and the number of sigops in a prevout scriptPubKey or scriptSig.
    fn validate_script_size<F: Fn() -> Txid>(
        script: &ScriptBuf,
        txid: F,
    ) -> Result<(), TransactionError> {
        if Self::is_unspendable(script) {
            return Err(tx_err!(txid, ScriptError));
        }
        if script.count_sigops() > 80_000 {
            return Err(tx_err!(txid, ScriptError));
        }
        Ok(())
    }

    /// Performs context-free validation of a coinbase transaction. The reward check requires
    /// context about the block and is performed by [`Consensus::verify_block_transactions`].
    pub fn verify_coinbase(tx: &Transaction) -> Result<(), TransactionError> {
        #[cfg(feature = "bitcoinkernel")]
        {
            let txid = || tx.compute_txid();
            let transaction_bytes = serialize(tx);
            let kernel_transaction =
                bitcoinkernel::Transaction::try_from(transaction_bytes.as_slice())
                    .map_err(|e| tx_err!(txid, ContextFreeValidation, e.to_string()))?;

            match kernel_transaction.check() {
                TxCheckResult::Valid if tx.is_coinbase() => Ok(()),
                TxCheckResult::Valid => Self::diagnose_coinbase(tx),
                TxCheckResult::Invalid(result) => {
                    Self::diagnose_coinbase(tx)?;

                    Err(tx_err!(txid, ContextFreeValidation, format!("{result:?}")))
                }
            }
        }

        #[cfg(not(feature = "bitcoinkernel"))]
        Self::diagnose_coinbase(tx)
    }

    /// Diagnoses coinbase failures using Floresta's error variants.
    fn diagnose_coinbase(tx: &Transaction) -> Result<(), TransactionError> {
        let txid = || tx.compute_txid();
        let input = match tx.input.as_slice() {
            [i] => i,
            _ => return Err(tx_err!(txid, InvalidCoinbase, "Coinbase must have 1 input")),
        };

        // The PrevOut of the coinbase input must be null
        if !input.previous_output.is_null() {
            return Err(tx_err!(txid, InvalidCoinbase, "Invalid Coinbase PrevOut"));
        }

        // The scriptsig size must be between 2 and 100 bytes
        // https://github.com/bitcoin/bitcoin/blob/v28.0/src/consensus/tx_check.cpp#L49
        let size = input.script_sig.len();
        if !(2..=100).contains(&size) {
            return Err(tx_err!(txid, InvalidCoinbase, "Invalid ScriptSig size"));
        }

        Ok(())
    }

    // TODO remove this once https://github.com/rust-bitcoin/rust-bitcoin/pull/3585 makes it into a
    // rust-bitcoin stable release (i.e., 0.33).
    pub fn get_bip34_height(block: &Block) -> Option<u32> {
        let cb = block.coinbase()?;
        let input = cb.input.first()?;
        let push = input.script_sig.instructions_minimal().next()?;

        match push {
            Ok(script::Instruction::PushBytes(b)) => {
                let h = script::read_scriptint(b.as_bytes()).ok()?;
                Some(h as u32)
            }

            Ok(script::Instruction::Op(opcode)) => {
                let opcode = opcode.to_u8();
                if (0x51..=0x60).contains(&opcode) {
                    Some(opcode as u32 - 0x50)
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Checks if a testnet4 block is compliant with the anti-timewarp rules of BIP94.
    ///
    /// a. The block's nTime field MUST be greater than or equal to the nTime
    /// field of the immediately prior block minus 600 seconds
    pub fn check_bip94_time(
        block: &BlockHeader,
        prev_block: &BlockHeader,
    ) -> Result<(), BlockValidationErrors> {
        if block.time < (prev_block.time - 600) {
            return Err(BlockValidationErrors::BIP94TimeWarp);
        }

        Ok(())
    }

    /// Calculates the next target for the proof of work algorithm, given the
    /// first and last block headers inside a difficulty adjustment period.
    pub fn calc_next_work_required(
        last_block: &BlockHeader,
        first_block: &BlockHeader,
        params: ChainParams,
    ) -> Target {
        let actual_timespan = last_block.time.saturating_sub(first_block.time);
        // from bip 94:
        //  a. The base difficulty value MUST be taken from the first block of the previous
        //     difficulty period
        //
        //  b. NOT from the last block as in previous implementations
        let bits = match params.enforce_bip94 {
            true => first_block.bits,
            false => last_block.bits,
        };

        CompactTarget::from_next_work_required(bits, actual_timespan as u64, params).into()
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

        // Check if there is a spend of an unspendable UTXO (BIP30)
        if Self::contains_unspendable_utxo(&del_hashes) {
            Err(BlockValidationErrors::UnspendableUTXO)?;
        }

        // Convert to BitcoinNodeHash, from rustreexo
        let del_hashes: Vec<_> = del_hashes
            .into_iter()
            .map(|hash| BitcoinNodeHash::Some(hash.to_byte_array()))
            .collect();

        let adds = udata::proof_util::get_block_adds(block, height, block_hash);

        // Update the accumulator
        let acc = acc.modify(&adds, &del_hashes, &proof)?.0;
        Ok(acc)
    }

    fn contains_unspendable_utxo(del_hashes: &[sha256::Hash]) -> bool {
        del_hashes.iter().any(|hash| {
            let bytes = hash.as_ref();
            bytes == UNSPENDABLE_BIP30_UTXO_91722 || bytes == UNSPENDABLE_BIP30_UTXO_91812
        })
    }
}

/// An order-independent 128-bit SwiftSync aggregator over `OutPoint`s.
pub mod swift_sync_agg {
    use core::ops::Add;
    use core::ops::AddAssign;

    use bitcoin::OutPoint;
    use bitcoin::hashes::Hash;
    use bitcoin::hashes::HashEngine;
    use bitcoin::hashes::siphash24;

    #[derive(Default)]
    /// A pair of `SipHash24` secret keys, used as the [`SwiftSyncAgg`] session salt.
    pub struct SipHashKeys {
        k0: u64,
        k1: u64,
        k2: u64,
        k3: u64,
    }

    impl SipHashKeys {
        #[inline]
        /// Construct a new pair of `SipHash24` keys from `(k0, k1)` and `(k2, k3)`.
        pub fn new(k0: u64, k1: u64, k2: u64, k3: u64) -> Self {
            Self { k0, k1, k2, k3 }
        }
    }

    #[derive(Clone)]
    /// Cached SipHash state after hashing a `txid`.
    ///
    /// This allows hashing multiple outpoints of the form `txid || vout_le` efficiently by
    /// reusing the work for the shared 32-byte `txid` prefix.
    pub struct TxidHashMidstate {
        base0: siphash24::HashEngine,
        base1: siphash24::HashEngine,
    }

    impl TxidHashMidstate {
        /// Create a midstate by initializing both SipHash engines and inputting `txid`.
        pub fn new(keys: &SipHashKeys, txid: &[u8; 32]) -> Self {
            let mut midstate = Self {
                base0: siphash24::HashEngine::with_keys(keys.k0, keys.k1),
                base1: siphash24::HashEngine::with_keys(keys.k2, keys.k3),
            };

            midstate.base0.input(txid);
            midstate.base1.input(txid);

            midstate
        }

        /// Finalize the hash for an `OutPoint` with this cached `txid` and the provided `vout`.
        pub fn finalize_with_vout(mut self, vout: u32) -> (u64, u64) {
            let vout_le = vout.to_le_bytes();

            self.base0.input(&vout_le);
            self.base1.input(&vout_le);

            let a = siphash24::Hash::from_engine_to_u64(self.base0);
            let b = siphash24::Hash::from_engine_to_u64(self.base1);

            (a, b)
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    /// An order-independent 128-bit SwiftSync aggregator.
    ///
    /// Adds and removes `OutPoint`s by hashing the preimage `txid || vout_le` with **two**
    /// `SipHash24` instances. Each instance uses its own independent 128-bit key (together,
    /// a 32-byte secret), which must be uniformly random to avoid adversarially chosen
    /// collisions.
    ///
    /// The two resulting `u64` values are accumulated into the two 64-bit limbs using
    /// wrapping add/sub. If the same multiset of outpoints is added and removed under the
    /// same secret, the accumulator returns to zero.
    ///
    /// The 32-byte [`SipHashKeys`] **must remain constant for the entire session**.
    /// Changing it breaks cancellation.
    ///
    /// # Example
    /// ```
    /// use bitcoin::OutPoint;
    /// use bitcoin::Txid;
    /// use bitcoin::hashes::Hash;
    /// use floresta_chain::swift_sync_agg::SipHashKeys;
    /// use floresta_chain::swift_sync_agg::SwiftSyncAgg;
    ///
    /// let keys = SipHashKeys::new(1, 2, 3, 4); // use uniformly random keys in production
    /// let outpoint = OutPoint::new(Txid::all_zeros(), 0);
    /// let mut agg = SwiftSyncAgg::zero();
    ///
    /// agg.add(&keys, &outpoint);
    /// assert!(!agg.is_zero());
    ///
    /// agg.remove(&keys, &outpoint);
    /// assert!(agg.is_zero());
    /// ```
    pub struct SwiftSyncAgg(u64, u64);

    impl SwiftSyncAgg {
        #[inline]
        /// Initializes an aggregator with zero value.
        pub const fn zero() -> Self {
            Self(0, 0)
        }

        #[inline]
        /// Whether the aggregator is zero.
        pub fn is_zero(&self) -> bool {
            self.0 == 0 && self.1 == 0
        }

        /// Adds an `OutPoint` to the aggregator.
        pub fn add(&mut self, salt: &SipHashKeys, outpoint: &OutPoint) {
            let hash = Self::hash_outpoint(salt, outpoint);

            *self = self.wrapping_add(hash);
        }

        /// Adds an `OutPoint` to the aggregator using a `SipHash24` midstate for the `txid`.
        pub fn add_with_vout(&mut self, hasher: TxidHashMidstate, vout: u32) {
            let hash = hasher.finalize_with_vout(vout);

            *self = self.wrapping_add(hash);
        }

        /// Removes an `OutPoint` from the aggregator.
        pub fn remove(&mut self, salt: &SipHashKeys, outpoint: &OutPoint) {
            let hash = Self::hash_outpoint(salt, outpoint);

            *self = self.wrapping_sub(hash);
        }

        /// Hashes the given `OutPoint` to a pair of `u64` values.
        pub(crate) fn hash_outpoint(keys: &SipHashKeys, outpoint: &OutPoint) -> (u64, u64) {
            let mut bytes = [0u8; 36];
            bytes[..32].copy_from_slice(outpoint.txid.as_byte_array());
            bytes[32..36].copy_from_slice(&outpoint.vout.to_le_bytes());

            let a = Self::sip64(keys.k0, keys.k1, &bytes);
            let b = Self::sip64(keys.k2, keys.k3, &bytes);
            (a, b)
        }

        fn sip64(k0: u64, k1: u64, msg: &[u8]) -> u64 {
            let mut h = siphash24::HashEngine::with_keys(k0, k1);
            h.input(msg);
            siphash24::Hash::from_engine_to_u64(h)
        }

        fn wrapping_add(&self, rhs: (u64, u64)) -> Self {
            Self(self.0.wrapping_add(rhs.0), self.1.wrapping_add(rhs.1))
        }

        fn wrapping_sub(&self, rhs: (u64, u64)) -> Self {
            Self(self.0.wrapping_sub(rhs.0), self.1.wrapping_sub(rhs.1))
        }
    }

    impl Add for SwiftSyncAgg {
        type Output = Self;

        #[inline]
        fn add(self, rhs: Self) -> Self::Output {
            self.wrapping_add((rhs.0, rhs.1))
        }
    }

    impl AddAssign for SwiftSyncAgg {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            *self = self.wrapping_add((rhs.0, rhs.1));
        }
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;
    use std::fs::File;

    use bitcoin::Amount;
    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Sequence;
    use bitcoin::Transaction;
    use bitcoin::TxIn;
    use bitcoin::TxOut;
    use bitcoin::Txid;
    use bitcoin::absolute::LockTime;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::constants::genesis_block;
    use bitcoin::hashes::Hash;
    use bitcoin::opcodes::OP_TRUE;
    use bitcoin::opcodes::all::OP_NOP;
    use bitcoin::transaction::Version;
    use floresta_common::assert_err;
    use floresta_common::assert_ok;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::prelude::IndexedMutRandom;
    use rand::rand_core::UnwrapErr;
    use rand::rngs::StdRng;
    use rand::rngs::SysRng;
    use rand::seq::SliceRandom;

    use super::*;

    #[macro_export]
    /// Macro for creating a TxOut
    macro_rules! txout {
        ($sats:expr, $script:expr) => {
            bitcoin::TxOut {
                value: bitcoin::Amount::from_sat($sats),
                script_pubkey: $script,
            }
        };
    }

    #[macro_export]
    /// Macro for constructing a legacy [`TxIn`] with optional scriptSig and sequence number.
    /// Needs the outpoint and, if not provided, defaults to empty scriptSig and `Sequence::MAX`.
    macro_rules! txin {
        ($outpoint:expr) => {
            bitcoin::TxIn {
                previous_output: $outpoint,
                script_sig: bitcoin::ScriptBuf::new(),
                sequence: bitcoin::Sequence::MAX,
                witness: bitcoin::Witness::new(),
            }
        };
        ($outpoint:expr, $script:expr) => {
            bitcoin::TxIn {
                previous_output: $outpoint,
                script_sig: $script,
                sequence: bitcoin::Sequence::MAX,
                witness: bitcoin::Witness::new(),
            }
        };
        ($outpoint:expr, $script:expr, $sequence:expr) => {
            bitcoin::TxIn {
                previous_output: $outpoint,
                script_sig: $script,
                sequence: $sequence,
                witness: bitcoin::Witness::new(),
            }
        };
    }

    /// Helper for building a zero-locktime transaction given the input and output list.
    fn build_tx(input: Vec<TxIn>, output: Vec<TxOut>) -> Transaction {
        Transaction {
            version: Version::TWO,
            lock_time: LockTime::ZERO,
            input,
            output,
        }
    }

    /// Helper to avoid boilerplate in test cases. Note this is not a null outpoint, restricted to
    /// coinbase transactions only, as that requires the vout to be `u32::MAX`.
    fn dummy_outpoint() -> OutPoint {
        OutPoint {
            txid: Txid::all_zeros(),
            vout: 0,
        }
    }

    fn tx_with_lock_time(lock_time: LockTime, sequence: Sequence) -> Transaction {
        Transaction {
            version: Version::TWO,
            lock_time,
            input: vec![txin!(dummy_outpoint(), ScriptBuf::new(), sequence)],
            output: vec![txout!(1, ScriptBuf::new())],
        }
    }

    const ZERO_HEIGHT: u32 = 0;
    const ZERO_BLOCK_TIME: u32 = 0;

    #[test]
    fn final_transaction_accepts_final_sequence() {
        let height_locked = tx_with_lock_time(LockTime::from_height(101).unwrap(), Sequence::MAX);
        let time_locked =
            tx_with_lock_time(LockTime::from_time(500_000_001).unwrap(), Sequence::MAX);

        assert!(Consensus::is_final_transaction(
            &height_locked,
            ZERO_HEIGHT,
            ZERO_BLOCK_TIME
        ));
        assert!(Consensus::is_final_transaction(
            &time_locked,
            ZERO_HEIGHT,
            ZERO_BLOCK_TIME
        ));
    }

    #[test]
    fn final_transaction_checks_height_locks_strictly() {
        let height = 100;

        let below_height = tx_with_lock_time(
            LockTime::from_height(height - 1).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );
        let at_height = tx_with_lock_time(
            LockTime::from_height(height).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );
        let above_height = tx_with_lock_time(
            LockTime::from_height(height + 1).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );

        assert!(Consensus::is_final_transaction(
            &below_height,
            height,
            ZERO_BLOCK_TIME
        ));
        assert!(!Consensus::is_final_transaction(
            &at_height,
            height,
            ZERO_BLOCK_TIME
        ));
        assert!(!Consensus::is_final_transaction(
            &above_height,
            height,
            ZERO_BLOCK_TIME
        ));
    }

    #[test]
    fn final_transaction_checks_time_locks_strictly() {
        let cutoff = 500_000_100;

        let below_cutoff = tx_with_lock_time(
            LockTime::from_time(cutoff - 1).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );
        let at_cutoff = tx_with_lock_time(
            LockTime::from_time(cutoff).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );
        let above_cutoff = tx_with_lock_time(
            LockTime::from_time(cutoff + 1).unwrap(),
            Sequence::ENABLE_LOCKTIME_NO_RBF,
        );

        assert!(Consensus::is_final_transaction(
            &below_cutoff,
            ZERO_HEIGHT,
            cutoff
        ));
        assert!(!Consensus::is_final_transaction(
            &at_cutoff,
            ZERO_HEIGHT,
            cutoff
        ));
        assert!(!Consensus::is_final_transaction(
            &above_cutoff,
            ZERO_HEIGHT,
            cutoff
        ));
    }

    #[cfg(feature = "bitcoinkernel")]
    /// Some made up transactions that test our script limits checks.
    /// Here's what is wrong with each transaction:
    ///     - tx1: Too many ops (512, should be <= 201)
    ///     - tx2: It's ok, just huge
    ///     - tx3: Script push too big (520, should be <= 520)
    ///     - tx4: Script sig is too big
    ///     - tx5: Also Ok, but a big script on p2sh
    ///     - tx6: Too many sig ops (201, should be <= 201)
    ///     - tx7: Executed OPs limit exceeded
    const TX_VALIDATION_CASES_LEGACY: &[&str] = &[
        "0200000001fdaf053eeaeed2e96594b542792417c6c223fa8571e88d31e296b1a655c81eb500000000fd0306012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294d000275757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575750000000001260200000000000017a9144e0c22952fa87a99064f93f77a74fb2e0184f04d8700000000:260200000000000017a9144e0c22952fa87a99064f93f77a74fb2e0184f04d87",
        "02000000013b4568b6d740e1625710ec49e6ce994e79ad55d7d1eef03cd945d5667229c05200000000fdcb04012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294cc97575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575750000000001260200000000000017a91436f39a404b67ec67516b353b30c3766b33609dec8700000000:260200000000000017a91436f39a404b67ec67516b353b30c3766b33609dec87",
        "02000000010f5f070f769b7d6290882dfd8659150e781f4ea7f19c2f5d93a9917aebd7d54500000000fd970501290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294d0004757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575750000000001260200000000000017a914dba622860b2edd8be5861f0cb204248f6a9f0b9d8700000000:260200000000000017a914dba622860b2edd8be5861f0cb204248f6a9f0b9d87",
        "0200000001540b20c497074bcb8c83869601566749f210ce7965f4e55ec1be9a8f648d743800000000fd9a0801290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294cc875757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575757575750000000001260200000000000017a9141ea5312cab0ad9a531c35b9051dc136bebc0669e8700000000:260200000000000017a9141ea5312cab0ad9a531c35b9051dc136bebc0669e87",
        "02000000019ebe6327436c18f978611d3fae6c2f6834cd6fbc08471134bc4bee16f8b9062400000000fdec03012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294cc86d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d6d0000000001260200000000000017a914a73ee4302bb9a6e5b7c42cae84b7c548638bc0148700000000:260200000000000017a914a73ee4302bb9a6e5b7c42cae84b7c548638bc01487",
        "0200000001cbbe326a4360ed38487a6fd1a091da0c22fd5084127401c3fba1ad52b8a37f3300000000fdec03012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901290129012901294cc8acacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacac0000000001260200000000000017a914436c244dc646042e3bafb50ff5729a0e80153e708700000000:260200000000000017a914436c244dc646042e3bafb50ff5729a0e80153e7087",
        "020000000139cf57739cb5d08335b7ed529792de34987d763d4856957dcc4b258c9cb1d0d300000000d601ff4cd276767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767676767600000000012602000000000000160014ef814b32b68a6138974b1144603b5c10569eeef600000000:2602000000000000160014ef814b32b68a6138974b1144603b5c10569eeef6",
    ];

    fn coinbase(is_valid: bool) -> Transaction {
        // This coinbase transaction was retrieved from https://learnmeabitcoin.com/explorer/block/0000000000000a0f82f8be9ec24ebfca3d5373fde8dc4d9b9a949d538e9ff679

        // Create input
        let script_sig = if is_valid {
            ScriptBuf::from_hex("03f0a2a4d9f0a2").unwrap()
        } else {
            // This must invalidate the coinbase transaction since it's a big script.
            ScriptBuf::from_hex(&format!("{:0>420}", "")).unwrap()
        };
        let input = txin!(OutPoint::null(), script_sig);

        // Create outputs
        let output_script = ScriptBuf::from_hex("41047eda6bd04fb27cab6e7c28c99b94977f073e912f25d1ff7165d9c95cd9bbe6da7e7ad7f2acb09e0ced91705f7616af53bee51a238b7dc527f2be0aa60469d140ac").unwrap();
        let output = txout!(5_000_350_000, output_script);

        Transaction {
            version: Version::ONE,
            lock_time: LockTime::from_height(150_007).unwrap(),
            input: vec![input],
            output: vec![output],
        }
    }

    /// Modifies a block to have a different output script (txdata is tampered with).
    fn mutate_block(block: &mut Block) {
        let mut rng = StdRng::seed_from_u64(0x_bebe_cafe);

        let tx = block.txdata.choose_mut(&mut rng).unwrap();
        let out = tx.output.choose_mut(&mut rng).unwrap();
        let spk = out.script_pubkey.as_mut_bytes();
        // Random byte from a random scriptPubKey
        let byte = spk.choose_mut(&mut rng).unwrap();

        *byte += 1;
    }

    /// Test helper to update the witness commitment in a block, assuming txdata was modified.
    /// This ensures `block` is not considered mutated, so we can exercise other error cases.
    fn update_witness_commitment(block: &mut Block) -> Option<()> {
        // BIP141 witness-commitment prefix, where the full commitment data is:
        //  1-byte - OP_RETURN (0x6a)
        //  1-byte - Push the following 36 bytes (0x24)
        //  4-byte - Commitment header (0xaa21a9ed)
        // 32-byte - Commitment hash: Double-SHA256(witness root hash|witness reserved value)
        const MAGIC: [u8; 6] = [0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];

        let coinbase = &block.txdata[0];

        // Commitment is in the last coinbase output that starts with magic bytes.
        let pos = coinbase.output.iter().rposition(|out| {
            let spk = out.script_pubkey.as_bytes();
            spk.len() >= 38 && spk[0..6] == MAGIC
        })?;

        // Witness reserved value is in coinbase input witness.
        let witness_rv: &[u8; 32] = {
            let mut it = coinbase.input[0].witness.iter();
            match (it.next(), it.next()) {
                (Some(rv), None) => rv.try_into().ok()?,
                _ => return None,
            }
        };

        let root = block.witness_root()?;
        let c = *Block::compute_witness_commitment(&root, witness_rv).as_byte_array();

        block.txdata[0].output[pos].script_pubkey.as_mut_bytes()[6..38].copy_from_slice(&c);
        Some(())
    }

    /// Decode and deserialize a zstd-compressed block in the given file path.
    fn decode_block(file_path: &str) -> Block {
        let block_file = File::open(file_path).unwrap();
        let block_bytes = zstd::decode_all(block_file).unwrap();
        deserialize(&block_bytes).unwrap()
    }

    #[test]
    fn test_check_merkle_root() {
        let blocks = [
            genesis_block(Network::Bitcoin),
            genesis_block(Network::Testnet),
            genesis_block(Network::Testnet4),
            genesis_block(Network::Signet),
            genesis_block(Network::Regtest),
            decode_block("./testdata/block_866342/raw.zst"),
            decode_block("./testdata/block_367891/raw.zst"),
        ];

        for mut block in blocks {
            assert!(block.check_merkle_root());
            let txids = Consensus::check_merkle_root(&block).expect("merkle roots match");

            // Sanity check: the returned txids are the correct ones
            for (txid, tx) in txids.into_iter().zip(&block.txdata) {
                assert_eq!(txid, tx.compute_txid());
            }

            // Modifying the txdata should invalidate the block
            mutate_block(&mut block);

            assert!(!block.check_merkle_root());
            if Consensus::check_merkle_root(&block).is_some() {
                panic!("merkle roots shouldn't match");
            }
        }
    }

    /// Modifies historical block at height 866,342 by adding one extra transaction so that the
    /// updated block weight is 4,000,001 WUs. The block merkle roots are updated accordingly.
    fn build_oversized_866_342() -> Block {
        let mut block = decode_block("./testdata/block_866342/raw.zst");

        let consensus = Consensus::from(Network::Bitcoin);
        consensus.check_block(&block, 866_342).expect("valid block");

        // This block is close but below to the max weight
        assert_eq!(block.weight().to_wu(), 3_993_209);

        // Modify the block by adding one transaction that makes it exceed the weight limit
        let mut script_out = ScriptBuf::default();
        for _ in 0..1_636 {
            script_out.push_opcode(OP_NOP);
        }
        let out = txout!(1, script_out);
        let tx = build_tx(vec![txin!(dummy_outpoint())], vec![out]);

        block.txdata.insert(1, tx);

        // Update the witness commitment, and then the merkle root, which depends on the former
        update_witness_commitment(&mut block).expect("should be able to update");
        block.header.merkle_root = block.compute_merkle_root().unwrap();

        block
    }

    #[test]
    fn test_block_too_big() {
        let height = 866_342;
        let consensus = Consensus::from(Network::Bitcoin);
        let block = build_oversized_866_342();

        // This block is now just over the weight limit, by one unit!
        assert_eq!(block.weight().to_wu(), 4_000_001);
        assert!(block.weight() > Weight::MAX_BLOCK);
        assert_eq!(Weight::MAX_BLOCK.to_wu(), 4_000_000);

        // The txdata commitments match
        assert!(block.check_merkle_root());
        assert!(block.check_witness_commitment());
        Consensus::check_merkle_root(&block).expect("merkle root matches");

        match consensus.check_block(&block, height) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::BlockTooBig)) => (),
            other => panic!("We should have `BlockValidationErrors::BlockTooBig`, got {other:?}"),
        }
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn context_free_block_check_rejects_mutated_merkle_tree() {
        let consensus = Consensus::from(Network::Regtest);
        let mut block = genesis_block(Network::Regtest);
        block.txdata.push(block.txdata[0].clone());
        block.header.merkle_root = block.compute_merkle_root().unwrap();

        match consensus.check_block(&block, 0) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::BadMerkleRoot)) => {}
            other => panic!("Expected BadMerkleRoot, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_script_size() {
        use bitcoin::hashes::Hash;
        let dummy_txid = Txid::all_zeros;

        // Generate a script larger than 10,000 bytes (e.g., 10,001 bytes)
        let large_script = ScriptBuf::from_hex(&format!("{:0>20002}", "")).unwrap();
        assert_eq!(large_script.len(), 10_001);

        let small_script =
            ScriptBuf::from_hex("76a9149206a30c09cc853bb03bd917a4f9f29b089c1bc788ac").unwrap();

        assert_ok!(Consensus::validate_script_size(&small_script, dummy_txid));
        assert_err!(Consensus::validate_script_size(&large_script, dummy_txid));
    }

    #[test]
    fn test_validate_coinbase() {
        let valid_one = coinbase(true);
        let invalid_one = coinbase(false);
        let regular_tx = build_tx(
            vec![txin!(dummy_outpoint())],
            vec![txout!(0, ScriptBuf::new())],
        );
        // The case that should be valid
        assert_ok!(Consensus::verify_coinbase(&valid_one));
        assert_err!(Consensus::verify_coinbase(&regular_tx));
        // Invalid coinbase script
        assert_eq!(
            Consensus::verify_coinbase(&invalid_one)
                .unwrap_err()
                .error
                .to_string(),
            "Invalid coinbase: \"Invalid ScriptSig size\""
        );
    }

    #[test]
    fn test_coinbase_maturity() {
        // Helper function to test coinbase spending
        // Requires the coinbase tx to have the relevant output at vout=0, and be spent at vin=0
        fn test_case(
            coinbase_tx: &Transaction,
            spending_tx: &Transaction,
            expected_coinbase: &str,
            expected_spending: &str,
            heights: (u32, u32),
            expected_ok: bool,
        ) {
            let (creation_height, spending_height) = heights;

            assert_eq!(coinbase_tx.compute_txid().to_string(), expected_coinbase);
            assert_eq!(spending_tx.compute_txid().to_string(), expected_spending);
            assert_ok!(Consensus::verify_coinbase(coinbase_tx));

            let mut utxos = HashMap::new();
            utxos.insert(
                spending_tx.input[0].previous_output,
                UtxoData {
                    txout: coinbase_tx.output[0].clone(),
                    is_coinbase: true,
                    creation_height,
                    creation_time: 0, // Use a dummy time
                },
            );

            let spend_result =
                Consensus::verify_transaction(spending_tx, &mut utxos, spending_height, true, 0);

            if expected_ok {
                assert_ok!(spend_result);
            } else {
                match spend_result.unwrap_err() {
                    BlockchainError::TransactionError(inner) => {
                        let txid = || spending_tx.compute_txid();
                        assert_eq!(inner, tx_err!(txid, CoinbaseNotMatured));
                    }
                    e => panic!("Expected a TransactionError, but got: {e:?}"),
                }
            }
        }

        // The first ever coinbase coins to be spent 101 blocks later are from this coinbase P2PK
        // output (bd4d7b0bf9341e575ab7f17b3e3187b70635fd2b99e0dc24d4c3f55e9e358115:0)
        let coinbase_74_547 = deserialize_hex("01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff080418ba001c02c205ffffffff0100f2052a010000004341040e6d9c4cea77c12fab14f4ef5a7713ed50a5c847e4060ae513ea672427babf878ee2b88a0f515157b39b706b1c799cdc42edac2554b205a6e61bf4bcc8ca595aac00000000").unwrap();
        let spending_tx = deserialize_hex("01000000011581359e5ef5c3d424dce0992bfd3506b787313e7bf1b75a571e34f90b7b4dbd0000000049483045022100a0bba3f5731b0d89af8a4fa9f20ccdd89a0758b9eee909db1f8a3eb55d8c907c02201e52ff6abc5c88c09f48dcd40a44beb25f6715aac01ab1e3b9b3e12b2509ab4c01ffffffff0100f2052a010000001976a914ee3a639d407116b0debb484f0335144d794f57e188ac00000000").unwrap();

        test_case(
            &coinbase_74_547,
            &spending_tx,
            // The expected txids
            "bd4d7b0bf9341e575ab7f17b3e3187b70635fd2b99e0dc24d4c3f55e9e358115",
            "315b8651901195deed71f830cb37f33b23eab9c524bd2f2cca32d5b7273a3528",
            // Creation and spending heights
            (74_547, 74_648),
            true,
        );

        // This is the 11th time a coinbase output was spent just 100 blocks later (the very first
        // case ever was 223ccdd55e625d4a82d616d3fbdcded0c47f237360098334430242c2262ac6ee:4)
        let coinbase_232_709 = deserialize_hex("01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff2703058d03062f503253482f04a9357651086007873a080000020d3430363638332f736c7573682f0000000001cbef8f96000000001976a914e285a29e0704004d4e95dbb7c57a98563d9fb2eb88ac00000000").unwrap();
        let spending_tx = deserialize_hex("01000000014c901737cca754ad3d0c96fb3e8523d8b00380babf3c20d35779316392b40767000000006a473044022026f8b003ce423ba36fd95f48d3bf297dfb37cd5533a635b76a1f47c76c30e99402205113058dfc25f821905d4ba26a05e870722fb0877e1406605a857f1d71ea13f5012102c37e0f2966c6f154fd43dc45fc3b5fcfdd2d85ed3c95961d6d07a60409e6e4c1ffffffff02cbf68c01000000001976a9145c0727387071c75451b500aacd7077744d6dfbea88ac80626a94000000001976a914e1c9b052561cf0a1da9ee3175df7d5a2d7ff7dd488ac00000000").unwrap();

        test_case(
            &coinbase_232_709,
            &spending_tx,
            "6707b49263317957d3203cbfba8003b0d823853efb960c3dad54a7cc3717904c",
            "0a5672eecf25f809d11053d7dc9098d8db3c98387d51052b499a4c79eef41b61",
            (232_709, 232_809), // Spent just at block n + 100
            true,
        );

        // These coins couldn't be spent at the previous block (not matured)
        test_case(
            &coinbase_232_709,
            &spending_tx,
            "6707b49263317957d3203cbfba8003b0d823853efb960c3dad54a7cc3717904c",
            "0a5672eecf25f809d11053d7dc9098d8db3c98387d51052b499a4c79eef41b61",
            (232_709, 232_808), // Spent at block n + 99
            false,
        );
    }

    #[test]
    #[cfg(feature = "bitcoinkernel")]
    fn script_verified_transaction_consumes_inputs() {
        // Transaction extracted from https://learnmeabitcoin.com/explorer/tx/0094492b6f010a5e39c2aacc97396ce9b6082dc733a7b4151ccdbd580f789278
        // Mock data for testing

        let mut utxos = HashMap::new();
        let tx: Transaction = deserialize_hex("0100000001bd597773d03dcf6e22ba832f2387152c9ab69d250a8d86792bdfeb690764af5b010000006c493046022100841d4f503f44dd6cef8781270e7260db73d0e3c26c4f1eea61d008760000b01e022100bc2675b8598773984bcf0bb1a7cad054c649e8a34cb522a118b072a453de1bf6012102de023224486b81d3761edcd32cedda7cbb30a4263e666c87607883197c914022ffffffff021ee16700000000001976a9144883bb595608dcfe882aea5f7c579ef107a4fb5b88ac52a0aa00000000001976a914782231de72adb5c9df7367ab0c21c7b44bbd743188ac00000000").unwrap();
        let txid = || tx.compute_txid();

        assert_eq!(tx.input.len(), 1, "We only spend one utxo in this tx");
        let outpoint = tx.input[0].previous_output;

        let output_script =
            ScriptBuf::from_hex("76a9149206a30c09cc853bb03bd917a4f9f29b089c1bc788ac").unwrap();

        utxos.insert(
            outpoint,
            UtxoData {
                txout: txout!(18000000, output_script),
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        );
        let flags = bitcoinkernel::VERIFY_P2SH;
        Consensus::verify_transaction(&tx, &mut utxos, 0, true, flags)
            .expect("Transaction should be valid");

        // Script verification uses the transaction-local UTXOs, while the block-wide map retains
        // only outputs available to later transactions.
        assert!(utxos.is_empty(), "UTXO should be consumed");

        // Trying to verify again with an empty UTXO map must fail with this error
        let expected = tx_err!(txid, UtxoNotFound, outpoint);

        match Consensus::verify_transaction(&tx, &mut utxos, 0, true, flags) {
            Err(BlockchainError::TransactionError(e)) => assert_eq!(e, expected),
            other => panic!("Expected TransactionError, got: {other:?}"),
        }
    }

    #[test]
    fn rejects_cross_transaction_double_spend_without_script_checks() {
        let outpoint = dummy_outpoint();
        let mut utxos = HashMap::from([(
            outpoint,
            UtxoData {
                txout: txout!(1, ScriptBuf::new()),
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        )]);

        let first = build_tx(vec![txin!(outpoint)], vec![txout!(1, ScriptBuf::new())]);
        let second = build_tx(vec![txin!(outpoint)], vec![txout!(0, ScriptBuf::new())]);

        // Block validation carries this scratch map from one transaction to the next, including
        // when script checks are skipped during AssumeValid validation.
        Consensus::verify_transaction(&first, &mut utxos, 0, false, 0)
            .expect("first spend should be valid");
        assert!(utxos.is_empty(), "spent input must leave the UTXO map");

        // Trying to verify again with an empty UTXO map must fail with this error
        let expected = tx_err!(|| second.compute_txid(), UtxoNotFound, outpoint);

        match Consensus::verify_transaction(&second, &mut utxos, 0, false, 0) {
            Err(BlockchainError::TransactionError(e)) => assert_eq!(e, expected),
            other => panic!("Expected TransactionError, got: {other:?}"),
        }
    }

    #[test]
    fn test_output_value_overflow() {
        let ins = vec![txin!(dummy_outpoint())];
        let outs = vec![
            txout!(u64::MAX, ScriptBuf::new()),
            txout!(1, ScriptBuf::new()),
        ];
        let tx = build_tx(ins, outs);

        match Consensus::check_transaction_context_free(&tx) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::TooManyCoins)) => (),
            other => panic!("Expected TooManyCoins, got: {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_inputs_rejected() {
        let outpoint = dummy_outpoint();
        let outpoint2 = OutPoint {
            txid: Txid::all_zeros(),
            vout: 1,
        };

        // 2-input fast path
        let tx = build_tx(
            vec![txin!(outpoint), txin!(outpoint)],
            vec![txout!(0, ScriptBuf::new())],
        );

        assert!(matches!(
            Consensus::check_transaction_context_free(&tx),
            Err(BlockchainError::TransactionError(TransactionError {
                error: BlockValidationErrors::DuplicateInput,
                ..
            }))
        ));

        // 3+-input HashSet path
        let tx = build_tx(
            vec![txin!(outpoint), txin!(outpoint2), txin!(outpoint)],
            vec![txout!(0, ScriptBuf::new())],
        );

        assert!(matches!(
            Consensus::check_transaction_context_free(&tx),
            Err(BlockchainError::TransactionError(TransactionError {
                error: BlockValidationErrors::DuplicateInput,
                ..
            }))
        ));
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn context_free_transaction_allows_op_return_script_sig() {
        let tx = build_tx(
            vec![txin!(dummy_outpoint(), ScriptBuf::from_bytes(vec![0x6a]))],
            vec![txout!(0, ScriptBuf::new())],
        );

        Consensus::check_transaction_context_free(&tx)
            .expect("OP_RETURN in scriptSig is valid before script execution");
    }

    #[test]
    fn test_input_value_above_max_money() {
        let outpoint = dummy_outpoint();

        let excess_money = (Amount::MAX_MONEY + Amount::ONE_SAT).to_sat();
        assert_eq!(excess_money, 100_000_000 * 21_000_000 + 1); // sanity check

        let mut utxos = HashMap::new();
        utxos.insert(
            outpoint,
            UtxoData {
                txout: txout!(excess_money, ScriptBuf::new()),
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        );

        let tx = build_tx(vec![txin!(outpoint)], vec![txout!(1, ScriptBuf::new())]);

        match Consensus::verify_transaction(&tx, &mut utxos, 0, false, 0) {
            Err(BlockchainError::BlockValidation(BlockValidationErrors::TooManyCoins)) => (),
            other => panic!("Expected TooManyCoins, got: {other:?}"),
        }
    }

    #[test]
    fn test_total_out_value() {
        let tx: Transaction = deserialize_hex("010000000001018723232750a7a1ec07f650535a9f5d1ccbfdb311919dd7e58d04b61460acb3761600000000fdffffff0310aa0200000000001600144a36b0334c4f32fb7cdc0dbceaae0d61dd08db32f9bb08000000000016001497e7645f99db8913ae7eb08f06a8b735ef97bcd1df61040200000000220020099ee9d3c6fb2d278ab0b602db3dca4eef0a09368ab472dabe7b0df599b92e490400473044022003d0e0fb7ed7723e65bae57ad848f0efacd96eaa97d5fc789ebd19eb84a39dfa022044f58aa40845017779260a827583bb88e8065fb582222588513ce135a614849a014730440220501c20145b38d50abe25ef4719b447cc58075ee0c2e7a701abdc7a9e78d9752402206d64a8790010fa4a27ecf27473eb4f8195e1f6e28583a073c56dc58d577652ad01695221038aa0f2da0ba95cc2b75ccb7e8492a7fb74fe74f60fe90983b4b7f3ddc109088c210235df4fc22cf11f810b1fe3d98f53bb21252add45b9062f4df8a52b07d377a61f2103b65ac7a33844ecdcdef741afee2b009432fa2fc486af4d5155c12d0cf122158253ae00000000").unwrap();
        let expected_txid =
            Txid::from_str("28b5daa19149e892000bef5d1f53fdb8e366a6c7da2b10cd6f9d4ff4a33c667d")
                .unwrap();

        assert_eq!(tx.compute_txid(), expected_txid, "real tx mined at 936,305");

        let total_outs = Consensus::total_out_value(&tx).unwrap();
        assert_eq!(total_outs, Amount::from_sat(34_588_648));
    }

    // Test cases for Bitcoin script limits in the format <spending_tx>:<prevout>.
    #[cfg(feature = "bitcoinkernel")]
    fn create_case(case: &str) -> (Transaction, HashMap<OutPoint, UtxoData>) {
        let Some((spending, prevout)) = case.split_once(':') else {
            panic!("Invalid case: {case}");
        };

        let spending_tx: Transaction = deserialize_hex(spending).unwrap();
        let txout: TxOut = deserialize_hex(prevout).unwrap();

        let mut utxos = HashMap::new();
        utxos.insert(
            spending_tx.input[0].previous_output,
            UtxoData {
                txout,
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        );

        (spending_tx, utxos)
    }

    #[cfg(feature = "bitcoinkernel")]
    #[test]
    fn test_transaction_validation_legacy() {
        let expected = [false, true, false, false, true, false, false];
        let mut valid = expected.into_iter();

        for case in TX_VALIDATION_CASES_LEGACY.iter() {
            let (transaction, mut utxos) = create_case(case);
            let dummy_height = 0;

            let result = Consensus::verify_transaction(
                &transaction,
                &mut utxos,
                dummy_height,
                true,
                bitcoinkernel::VERIFY_ALL_PRE_TAPROOT,
            );

            let expected = valid.next().unwrap();
            assert_eq!(result.is_ok(), expected, "{case} {result:?}");
        }
    }

    pub fn true_script() -> ScriptBuf {
        let mut script = ScriptBuf::default();
        script.push_opcode(OP_TRUE);
        script
    }

    pub fn oversized_script() -> ScriptBuf {
        let mut script = ScriptBuf::default();
        for _ in 0..MAX_SCRIPT_SIZE {
            script.push_opcode(OP_NOP);
        }
        script.push_opcode(OP_TRUE);
        script
    }

    #[test]
    // Bitcoin Consensus rules dictate that a scriptPubKey that's more than 10_000 bytes long
    // won't be spendable. However, such an output **can** be created. We only check those
    // sizes when it gets spent.
    //
    // This test creates an over-sized script, make sure that transaction containing it is valid.
    // Then we try to spend this output, and verify if this causes an error.
    fn test_spending_script_too_big() {
        let outpoint = dummy_outpoint();
        let flags = 0;
        let dummy_height = 0;

        let mut utxos = HashMap::new();
        utxos.insert(
            outpoint,
            UtxoData {
                txout: txout!(0, true_script()),
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        );

        // 1. Build a valid transaction that produces an oversized, unspendable output.
        let dummy_in = txin!(outpoint);
        let oversized_out = txout!(0, oversized_script());
        let tx_with_oversized = build_tx(vec![dummy_in], vec![oversized_out.clone()]);

        Consensus::verify_transaction(&tx_with_oversized, &mut utxos, dummy_height, false, flags)
            .unwrap();

        // 2. Register the oversized output as an available UTXO.
        let prevout = OutPoint::new(tx_with_oversized.compute_txid(), 0);
        utxos.insert(
            prevout,
            UtxoData {
                txout: oversized_out,
                is_coinbase: false,
                creation_height: 0,
                creation_time: 0,
            },
        );

        // 3. Attempt to spend the oversized output.
        let spending_in = txin!(prevout);
        let spending_tx = build_tx(vec![spending_in], vec![txout!(0, true_script())]);
        let err =
            Consensus::verify_transaction(&spending_tx, &mut utxos, dummy_height, false, flags)
                .unwrap_err();

        // Check that the error is exactly what we expect.
        match err {
            BlockchainError::TransactionError(inner) => {
                assert_eq!(inner, tx_err!(|| spending_tx.compute_txid(), ScriptError));
            }
            e => panic!("Expected a TransactionError, but got: {e:?}"),
        }
    }

    /// The mainnet blocks up to height 175. At height 9 we find the first ever spent `TxOut`,
    /// which is spent at height 170. These two blocks are especially useful for testing.
    fn read_blocks_txt() -> Vec<Block> {
        include_str!("../../testdata/mainnet_blocks.txt")
            .lines()
            .map(|b| deserialize_hex(b).unwrap())
            .collect()
    }

    #[test]
    fn test_swift_sync_agg_blocks() {
        let consensus = Consensus::from(Network::Bitcoin);
        let mainnet_blocks = read_blocks_txt();
        assert_eq!(mainnet_blocks.len(), 176);
        // All blocks except 9 and 170 just have a single, unspent TxOut
        let default_unspent_idx = HashSet::from_iter(vec![0]);

        let mut rng = UnwrapErr(SysRng);
        let salt = SipHashKeys::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        );

        let mut supply = Amount::ZERO;
        let mut agg = SwiftSyncAgg::zero();
        for (i, block) in mainnet_blocks.iter().enumerate().skip(1) {
            match i {
                // We add the only TxOut in this block to the aggregator (spent later).
                9 => {
                    let unspent_indexes = HashSet::new();
                    let (agg_blk_9, amount) = consensus
                        .process_block_swiftsync(block, 9, unspent_indexes, &salt)
                        .unwrap();

                    assert!(!agg_blk_9.is_zero(), "block aggregator shouldn't be zero");
                    assert!(agg.is_zero());
                    agg += agg_blk_9;
                    assert!(!agg.is_zero());

                    supply += amount;
                }
                // This block spends the TxOut that was added to the aggregator in block 9.
                170 => {
                    let unspent_indexes = HashSet::from_iter(vec![0, 1, 2]);
                    let (agg_blk_170, amount) = consensus
                        .process_block_swiftsync(block, 170, unspent_indexes, &salt)
                        .unwrap();

                    assert!(!agg_blk_170.is_zero(), "block aggregator shouldn't be zero");
                    assert!(!agg.is_zero(), "global aggregator shouldn't be zero");
                    agg += agg_blk_170;
                    assert!(agg.is_zero(), "aggregators should cancel out to zero");

                    supply += amount;
                }
                i => {
                    let unspent_indexes = default_unspent_idx.clone();
                    let (agg_i, amount) = consensus
                        .process_block_swiftsync(block, i as u32, unspent_indexes, &salt)
                        .unwrap();

                    assert!(agg_i.is_zero());
                    agg += agg_i;

                    if i < 9 {
                        assert!(agg.is_zero(), "we don't have TxOuts, agg should be zero");
                    } else if i < 170 {
                        assert!(
                            !agg.is_zero(),
                            "we added TxOut from block 9, agg should be non-zero"
                        );
                    } else {
                        assert!(
                            agg.is_zero(),
                            "we remove the element after finding the input, agg should be zero"
                        );
                    }
                    supply += amount;
                }
            }
        }

        // After the first 175 non-genesis blocks, the supply was 175 * 50 BTC
        assert_eq!(supply, Amount::ONE_BTC * 50 * 175);
        assert_eq!(supply, consensus.max_supply_at_height(175));

        // Repeat the check using only the relevant blocks
        let block_9 = &mainnet_blocks[9];
        let block_170 = &mainnet_blocks[170];

        let (agg_9, _) = consensus
            .process_block_swiftsync(block_9, 9, HashSet::new(), &salt)
            .unwrap();
        let (agg_170, _) = consensus
            .process_block_swiftsync(block_170, 170, HashSet::from_iter(vec![0, 1, 2]), &salt)
            .unwrap();

        assert!(!agg_9.is_zero(), "block aggregator shouldn't be zero");
        assert!(!agg_170.is_zero(), "block aggregator shouldn't be zero");

        let agg = agg_9 + agg_170;
        assert!(agg.is_zero(), "aggregators should cancel out to zero");
    }

    #[test]
    fn test_swift_sync_agg_shuffled() {
        let mut rng = StdRng::seed_from_u64(0xdefecade);
        let salt = SipHashKeys::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        );

        // Generate a set of random unique OutPoints
        const N: usize = 1_000;
        let mut set: HashSet<OutPoint> = HashSet::with_capacity(N);

        while set.len() < N {
            let mut txid_bytes = [0u8; 32];
            rng.fill_bytes(&mut txid_bytes);

            let txid = Txid::from_byte_array(txid_bytes);
            let vout = rng.next_u32() % 64;

            set.insert(OutPoint::new(txid, vout));
        }

        let mut add_order: Vec<OutPoint> = set.iter().copied().collect();
        let mut remove_order = add_order.clone();
        // Shuffle the orders of the two identical sets
        add_order.shuffle(&mut rng);
        remove_order.shuffle(&mut rng);

        let mut agg = SwiftSyncAgg::zero();

        // We can remove before adding, or vice versa
        for op in &remove_order {
            agg.remove(&salt, op);
            assert!(!agg.is_zero(), "not zero while removing");
        }

        for (i, op) in add_order.iter().enumerate() {
            agg.add(&salt, op);
            if i != 999 {
                assert!(!agg.is_zero(), "not zero while adding");
            }
        }

        assert!(agg.is_zero(), "zero at the end");
    }

    #[test]
    fn test_swift_sync_hash_midstate() {
        let mut rng = UnwrapErr(SysRng);

        for _ in 0..10_000 {
            let keys = SipHashKeys::new(
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            );

            let mut arr = [0u8; 32];
            rng.fill_bytes(&mut arr);
            let txid = Txid::from_byte_array(arr);
            let vout = rng.next_u32();

            let outpoint = OutPoint::new(txid, vout);
            let expected = SwiftSyncAgg::hash_outpoint(&keys, &outpoint);

            let mid = TxidHashMidstate::new(&keys, txid.as_byte_array());
            let got = mid.clone().finalize_with_vout(vout); // clone to match real code behavior

            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_max_supply_at_height() {
        let consensus = Consensus::from(Network::Bitcoin);
        assert_eq!(consensus.parameters.subsidy_halving_interval.get(), 210_000);

        let subsidy_0 = consensus.get_subsidy(0);
        let subsidy_1 = consensus.get_subsidy(210_000);
        let subsidy_2 = consensus.get_subsidy(420_000);
        let full_epoch_0 = subsidy_0 * 209_999; // No coinbase
        let full_epoch_1 = subsidy_1 * 210_000;

        assert_eq!(consensus.max_supply_at_height(0), Amount::from_sat(0));
        assert_eq!(consensus.max_supply_at_height(1), subsidy_0);
        assert_eq!(consensus.max_supply_at_height(209_999), full_epoch_0);
        assert_eq!(
            consensus.max_supply_at_height(210_000),
            full_epoch_0 + subsidy_1
        );
        assert_eq!(
            consensus.max_supply_at_height(310_006),
            full_epoch_0 + subsidy_1 * 100_007,
        );
        assert_eq!(
            consensus.max_supply_at_height(310_007),
            full_epoch_0 + subsidy_1 * 100_008,
        );
        assert_eq!(
            consensus.max_supply_at_height(420_001),
            full_epoch_0 + full_epoch_1 + (subsidy_2 * 2),
        );

        let mut max_coins = Amount::ZERO;
        for epoch in 0..=64 {
            // Last height in this halving epoch
            let height = (epoch + 1) * 210_000 - 1;
            let subsidy = consensus.get_subsidy(height);
            let blocks = match epoch {
                0 => 209_999,
                _ => 210_000,
            };
            max_coins += subsidy * blocks;
            assert_eq!(consensus.max_supply_at_height(height), max_coins);
        }

        assert_eq!(consensus.max_supply_at_height(13_440_000), max_coins);
        assert_eq!(consensus.max_supply_at_height(13_440_001), max_coins);
        assert_eq!(consensus.max_supply_at_height(13_444_444), max_coins);
        assert_eq!(consensus.max_supply_at_height(1_300_440_000), max_coins);
        assert_eq!(consensus.max_supply_at_height(u32::MAX - 1), max_coins);
    }
}
