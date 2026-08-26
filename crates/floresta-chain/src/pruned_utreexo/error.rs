// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module defines error types specific to the blockchain validation and database operations, along with conversion between types.
//!
//! The main error types are:
//! - [BlockchainError]: High-level error type that encapsulates all the error kinds from our node chain backend operation.
//! - [TransactionError]: Represents errors in transaction validation
//! - [BlockValidationErrors]: Errors encountered during block validation that are not tied to any specific transaction
//!
//! Each error type implements `Display` and `Debug` for error reporting.

extern crate alloc;

use core::error::Error;
use core::fmt;
use core::fmt::Debug;
use core::fmt::Display;
use core::fmt::Formatter;

use bitcoin::OutPoint;
use bitcoin::Txid;
use floresta_common::impl_error_from;
use floresta_common::prelude::*;
use rustreexo::stump::StumpError;

use crate::extensions::ChainWorkOverflow;
use crate::proof_util::UtreexoLeafError;
use crate::pruned_utreexo::chain_state_builder::BlockchainBuilderError;

pub trait DatabaseError: Debug + Display + Send + Sync + 'static {}

#[derive(Debug)]
/// Errors that can happen whilst interacting with the local blockchain.
///
/// It's the highest level error type in [`floresta_chain`](crate),
/// and is returned by [`ChainState`](crate::ChainState) methods.
pub enum BlockchainError {
    /// The block is not present in the [`ChainState`](crate::ChainState).
    BlockNotPresent,

    /// The block is an orphan or is invalid.
    OrphanOrInvalidBlock,

    /// The block failed validation.
    BlockValidation(BlockValidationErrors),

    /// The block contains invalid transaction(s).
    TransactionError(TransactionError),

    /// The Utreexo proof for this block is invalid.
    InvalidUtreexoProof,

    /// Error whilst interacting with the [accumulator](rustreexo::stump::Stump).
    AccumulatorError(StumpError),

    /// Failed to reconstruct a scriptpubkey from a [leaf](crate::pruned_utreexo::udata::CompactLeafData).
    UtreexoLeaf(UtreexoLeafError),

    /// Error whilst interacting with the the [`ChainStore`](crate::ChainStore).
    Database(Box<dyn DatabaseError>),

    /// The [`ChainState`](crate::ChainState) is not initialized.
    ChainNotInitialized,

    /// The [`ChainState`](crate::ChainState)'s tip is invalid.
    InvalidTip(String),

    /// The [`ChainState`](crate::ChainState)'s validation index is invalid.
    BadValidationIndex,

    /// A [`ChainState`](crate::ChainState) operation overflowed.
    OperationOverflow(ChainWorkOverflow),
}

impl_error_from!(BlockchainError, ChainWorkOverflow, OperationOverflow);
impl Display for BlockchainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockNotPresent => write!(f, "The block is not present in the ChainState"),
            Self::OrphanOrInvalidBlock => write!(f, "The block was orphaned or is invalid"),
            Self::BlockValidation(e) => write!(f, "Failed to validate the block: {e}"),
            Self::TransactionError(e) => {
                write!(f, "The block contains invalid transaction(s): {e}")
            }
            Self::InvalidUtreexoProof => write!(f, "The Utreexo proof for this block is invalid"),
            Self::AccumulatorError(e) => {
                write!(f, "Error whilst interacting with the accumulator: {e:?}")
            }
            Self::UtreexoLeaf(e) => write!(
                f,
                "Failed to reconstruct a scriptpubkey from Compact Leaf Data: {e}"
            ),
            Self::Database(e) => {
                write!(f, "Error whilst interacting with the the ChainState: {e}")
            }
            Self::ChainNotInitialized => write!(f, "The ChainState is not initialized"),
            Self::InvalidTip(e) => write!(f, "The ChainState's tip is invalid: {e}"),
            Self::BadValidationIndex => write!(f, "The ChainState's validation index is invalid"),
            Self::OperationOverflow(_) => write!(f, "A ChainState operation overflowed"),
        }
    }
}

impl Error for BlockchainError {}

impl<T: DatabaseError> From<T> for BlockchainError {
    fn from(value: T) -> Self {
        Self::Database(Box::new(value))
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Represents errors encountered during transaction validation.
pub struct TransactionError {
    /// The id of the transaction that caused this error
    pub txid: Txid,

    /// The error we've encountered
    pub error: BlockValidationErrors,
}

#[derive(Clone, Debug, PartialEq)]
/// Represents errors encountered during block validation.
pub enum BlockValidationErrors {
    BlockDoesntExtendTip,
    ContextFreeValidation(String),
    InvalidCoinbase(String),
    UtxoNotFound(OutPoint),
    ScriptValidationError(String),
    NullPrevOut,
    EmptyInputs,
    EmptyOutputs,
    ScriptError,
    BlockTooBig,
    TooManyCoins,
    NotEnoughPow,
    BadMerkleRoot,
    BadWitnessCommitment,
    NotEnoughMoney,
    FirstTxIsNotCoinbase,
    BadCoinbaseOutValue,
    EmptyBlock,
    BlockExtendsAnOrphanChain,
    BadBip34,
    InvalidUtreexoProof,
    CoinbaseNotMatured,
    NonFinalTransaction,
    UnspendableUTXO,
    BIP94TimeWarp,
    DuplicateInput,
}

// Helpful macro for generating a TransactionError
macro_rules! tx_err {
    ($txid_fn:expr, $variant:ident, $msg:expr) => {
        TransactionError {
            txid: ($txid_fn)(),
            error: BlockValidationErrors::$variant($msg.into()),
        }
    };
    ($txid_fn:expr, $variant:ident) => {
        TransactionError {
            txid: ($txid_fn)(),
            error: BlockValidationErrors::$variant,
        }
    };
}

impl Display for TransactionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Transaction {} is invalid: {}", self.txid, self.error)
    }
}

impl Display for BlockValidationErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockDoesntExtendTip => {
                write!(f, "This block doesn't build directly on the tip")
            }
            Self::ContextFreeValidation(e) => {
                write!(f, "Context-free consensus validation failed: {e}")
            }
            Self::ScriptValidationError(e) => {
                write!(f, "{e}")
            }
            Self::UtxoNotFound(outpoint) => {
                write!(f, "Utxo referenced by {outpoint:?} not found")
            }
            Self::NullPrevOut => {
                write!(
                    f,
                    "This transaction has a null PrevOut but it's not coinbase"
                )
            }
            Self::EmptyInputs => {
                write!(f, "This transaction has no inputs")
            }
            Self::EmptyOutputs => {
                write!(f, "This transaction has no outputs")
            }
            Self::BlockTooBig => write!(f, "Block too big"),
            Self::InvalidCoinbase(e) => {
                write!(f, "Invalid coinbase: {e:?}")
            }
            Self::TooManyCoins => write!(f, "Moving more coins that exists"),
            Self::ScriptError => {
                write!(
                    f,
                    "Script does not follow size requirements of 2>= and <=520"
                )
            }
            Self::NotEnoughPow => {
                write!(f, "This block doesn't have enough proof-of-work")
            }
            Self::BadMerkleRoot => write!(f, "Wrong merkle root"),
            Self::BadWitnessCommitment => write!(f, "Wrong witness commitment"),
            Self::NotEnoughMoney => {
                write!(f, "A transaction spends more than it should")
            }
            Self::FirstTxIsNotCoinbase => {
                write!(f, "The first transaction in a block isn't a coinbase")
            }
            Self::BadCoinbaseOutValue => {
                write!(f, "Coinbase claims more bitcoins than it should")
            }
            Self::EmptyBlock => {
                write!(f, "This block is empty (doesn't have a coinbase tx)")
            }
            Self::BlockExtendsAnOrphanChain => {
                write!(f, "This block extends a chain we don't have the ancestors")
            }
            Self::BadBip34 => write!(f, "BIP34 commitment mismatch"),
            Self::InvalidUtreexoProof => write!(f, "Invalid proof"),
            Self::CoinbaseNotMatured => {
                write!(f, "Coinbase not matured yet")
            }
            Self::NonFinalTransaction => {
                write!(f, "Block contains a non-final transaction")
            }
            Self::UnspendableUTXO => {
                write!(
                    f,
                    "Attempts to spend unspendable UTXO that was overwritten by the historical BIP30 violation"
                )
            }
            Self::BIP94TimeWarp => {
                write!(f, "BIP94 time warp detected")
            }
            Self::DuplicateInput => {
                write!(f, "This transaction has duplicate inputs")
            }
        }
    }
}

impl<T: DatabaseError> From<T> for BlockchainBuilderError {
    fn from(value: T) -> Self {
        Self::Database(Box::new(value))
    }
}

impl_error_from!(BlockchainError, TransactionError, TransactionError);
impl_error_from!(BlockchainError, BlockValidationErrors, BlockValidation);
impl_error_from!(BlockchainError, StumpError, AccumulatorError);
