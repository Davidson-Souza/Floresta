// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module defines the [ChainStore] trait, which provides the Floresta node API
//! for persisting and retrieving blockchain data (headers, block hashes linked to a
//! height, the best chain data, and the accumulator for each block).
//!
//! It also defines two important types for our storage format:
//! - [DiskBlockHeader]: A block header linked to its validation-state metadata
//! - [BestChain]: Tracks the current best chain, last valid block, and fork tips
//!
//! # Error types
//!
//! This module centralises the error types used by chainstore implementations:
//! - [ChainstoreError]: The public-facing error enum for [ChainStore] operations

use core::error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;

use bitcoin::BlockHash;
use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::consensus::encode;

use crate::BlockchainError;
use crate::DatabaseError;
use crate::prelude::*;

/// The maximum theoretical size in bytes of the Utreexo accumulator
///
/// In the worst case that all leaves are filled:
/// * The accumulator can have up to 64 roots
/// * Each root is 32 bytes
/// * The number of leaves is expressed as a [`u64`]
///
/// 64 (MAX_ROOTS) * 32 bytes (ROOT_SIZE) + 8 bytes (LEAF_COUNT) = 2056 bytes
pub const MAX_ACCUMULATOR_SIZE: usize = 2056;

#[derive(Debug)]
/// Errors that can happen whilst interacting with a [`ChainStore`] implementation
///
/// Variants that carry domain meaning ([`HeaderNotFound`], [`OversizedAccumulator`],
/// [`CorruptedDatabase`], [`InvalidValidationIndex`], [`FullIndex`]) allow callers to react
/// programmatically. Unrecoverable implementation-level failures are wrapped inside
/// [`Other`] as a plain diagnostic string.
///
/// [`HeaderNotFound`]: ChainstoreError::HeaderNotFound
/// [`OversizedAccumulator`]: ChainstoreError::OversizedAccumulator
/// [`CorruptedDatabase`]: ChainstoreError::CorruptedDatabase
/// [`InvalidValidationIndex`]: ChainstoreError::InvalidValidationIndex
/// [`FullIndex`]: ChainstoreError::FullIndex
/// [`Other`]: ChainstoreError::Other
pub enum ChainstoreError {
    /// The requested block header was not found in the store
    HeaderNotFound,

    /// The accumulator data exceeds the maximum allowed size of
    /// [`MAX_ACCUMULATOR_SIZE`] bytes
    OversizedAccumulator,

    /// The database is corrupted
    CorruptedDatabase,

    /// The validation index references a block without a known height
    ///
    /// Usually indicates the block is orphaned or on an invalid chain;
    /// Can be remedied by a full reindex
    InvalidValidationIndex,

    /// The store has reached its capacity; no new headers or index entries can be saved
    ///
    /// Surfaced to users as a [`ChainStoreWarning`]
    FullIndex,

    /// An unrecoverable error with a plain diagnostic message
    ///
    /// Covers implementation details such as I/O failures, lock poisoning,
    /// schema mismatches, and other non-actionable
    /// conditions. Consumers should treat this as fatal and surface the
    /// message for diagnostic purposes only.
    Other(String),
}

impl Display for ChainstoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderNotFound => {
                write!(f, "the requested block header was not found")
            }
            Self::OversizedAccumulator => write!(
                f,
                "accumulator exceeds the maximum size of {} bytes",
                MAX_ACCUMULATOR_SIZE
            ),
            Self::CorruptedDatabase => write!(f, "the database is corrupted"),
            Self::InvalidValidationIndex => {
                write!(
                    f,
                    "validation index has no known height; can be remedied by a full reindex"
                )
            }
            Self::FullIndex => write!(f, "the chainstore is full; no new entries can be saved"),
            Self::Other(msg) => write!(f, "other error: {msg}"),
        }
    }
}

impl error::Error for ChainstoreError {}

/// Allows [`ChainstoreError`] to be used as the associated `Error` type of [`ChainStore`]
impl DatabaseError for ChainstoreError {}

/// A trait defining methods for interacting with our chain database. These methods will be used by
/// the [ChainState](super::chain_state::ChainState) to save and retrieve data about the blockchain,
/// likely on disk.
///
/// This trait requires an associated error type that implements [DatabaseError]; a marker trait
/// satisfied by any `T: Display + Error`. This is useful to abstract the database implementation
/// from the blockchain.
pub trait ChainStore {
    type Error: DatabaseError;

    /// Saves the accumulator state for a given block height.
    fn save_roots_for_block(&mut self, roots: Vec<u8>, height: u32) -> Result<(), Self::Error>;

    /// Loads the state of our accumulator for a given block height.
    ///
    /// This is the state of the resulting accumulator after we process the block at `height`. If you
    /// need the accumulator used to validate a block at height `n`, you should get the accumulator
    /// from block `n - 1`.
    fn load_roots_for_block(&mut self, height: u32) -> Result<Option<Vec<u8>>, Self::Error>;

    /// Loads the blockchain height
    fn load_height(&self) -> Result<Option<BestChain>, Self::Error>;

    /// Saves the blockchain height.
    fn save_height(&mut self, height: &BestChain) -> Result<(), Self::Error>;

    /// Get a block header from our database. See [DiskBlockHeader] for more info about
    /// the data we save.
    fn get_header(&self, block_hash: &BlockHash) -> Result<Option<DiskBlockHeader>, Self::Error>;

    /// Get a block header by its height in our database.
    fn get_header_by_height(&self, height: u32) -> Result<Option<DiskBlockHeader>, Self::Error>;

    /// Saves a block header to our database. See [DiskBlockHeader] for more info about
    /// the data we save.
    fn save_header(&mut self, header: &DiskBlockHeader) -> Result<(), Self::Error>;

    /// Returns the block hash for a given height.
    fn get_block_hash(&self, height: u32) -> Result<Option<BlockHash>, Self::Error>;

    /// Flushes write buffers to disk, this is called periodically by the [ChainState](crate::ChainState),
    /// so in case of a crash, we don't lose too much data. If the database doesn't support
    /// write buffers, this method can be a no-op.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Associates a block hash with a given height, so we can retrieve it later.
    fn update_block_index(&mut self, height: u32, hash: BlockHash) -> Result<(), Self::Error>;

    /// Checks if our database didn't get corrupted, and if it has, it returns
    /// an error.
    ///
    /// If you're using a database that already checks for integrity by itself,
    /// this can safely be a no-op.
    fn check_integrity(&self) -> Result<(), Self::Error>;

    /// Returns the total size on disk, in bytes, of the data this store persists.
    ///
    /// The returned value is the sum of the sizes of each backing file managed by
    /// this store. Implementations should not include the size of any enclosing
    /// directory or unrelated data.
    fn size_on_disk(&self) -> Result<u64, Self::Error>;

    /// Returns accumulated health warnings about this store (e.g. index full).
    ///
    /// Warnings persist for the process lifetime and are never cleared.
    fn get_warnings(&self) -> Vec<ChainStoreWarning> {
        vec![]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A health warning raised by a [`ChainStore`], mapping to a message via [`Display`].
pub enum ChainStoreWarning {
    /// The header storage is full; new block headers cannot be stored.
    HeaderStorageFull,

    /// The block index is full; new blocks cannot be indexed.
    BlockIndexFull,
}

impl Display for ChainStoreWarning {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderStorageFull => write!(
                f,
                "Warning: header storage is full; new blocks cannot be stored."
            ),
            Self::BlockIndexFull => write!(
                f,
                "Warning: block index is full; new blocks cannot be stored."
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// This enum is used to store a block header in the database. It contains the header along with
/// metadata about the validation state of the block, and, if applicable, also its height.
pub enum DiskBlockHeader {
    /// Represents a fully validated block header in the current best chain.
    FullyValid(BlockHeader, u32),

    /// Represents an assumed valid block header.
    AssumedValid(BlockHeader, u32),

    /// Represents an orphan block header.
    Orphan(BlockHeader),

    /// Represents a block header in the current best chain whose block is pending validation.
    HeadersOnly(BlockHeader, u32),

    /// Represents a block header in a fork.
    InFork(BlockHeader, u32),

    /// Represents an invalid chain block header.
    InvalidChain(BlockHeader),
}

impl DiskBlockHeader {
    /// Computes the block hash.
    pub fn block_hash(&self) -> BlockHash {
        self.deref().block_hash()
    }

    /// Gets the block height or returns `None` if the block is orphaned or on an invalid chain.
    pub fn height(&self) -> Option<u32> {
        match self {
            Self::InFork(_, height) => Some(*height),
            Self::FullyValid(_, height) => Some(*height),
            Self::HeadersOnly(_, height) => Some(*height),
            Self::AssumedValid(_, height) => Some(*height),
            // These two cases don't store the block height
            Self::Orphan(_) => None,
            Self::InvalidChain(_) => None,
        }
    }

    /// Gets the block height or returns `BlockchainError::OrphanOrInvalidBlock` if the block is
    /// orphaned or on an invalid chain (the height is not stored).
    pub fn try_height(&self) -> Result<u32, BlockchainError> {
        self.height().ok_or(BlockchainError::OrphanOrInvalidBlock)
    }
}

// `DiskBlockHeader` dereferences to the inner header type.
impl Deref for DiskBlockHeader {
    type Target = BlockHeader;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::FullyValid(header, _) => header,
            Self::Orphan(header) => header,
            Self::HeadersOnly(header, _) => header,
            Self::InFork(header, _) => header,
            Self::InvalidChain(header) => header,
            Self::AssumedValid(header, _) => header,
        }
    }
}

impl Decodable for DiskBlockHeader {
    /// Decodes a `DiskBlockHeader` from a reader.
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, encode::Error> {
        let tag = u8::consensus_decode(reader)?;
        let header = BlockHeader::consensus_decode(reader)?;

        match tag {
            0x00 => {
                let height = u32::consensus_decode(reader)?;
                Ok(Self::FullyValid(header, height))
            }
            0x01 => Ok(Self::Orphan(header)),
            0x02 => {
                let height = u32::consensus_decode(reader)?;
                Ok(Self::HeadersOnly(header, height))
            }
            0x03 => {
                let height = u32::consensus_decode(reader)?;

                Ok(Self::InFork(header, height))
            }
            0x04 => Ok(Self::InvalidChain(header)),
            0x05 => {
                let height = u32::consensus_decode(reader)?;
                Ok(Self::AssumedValid(header, height))
            }
            _ => Err(encode::Error::ParseFailed("DiskBlockHeader: invalid tag")),
        }
    }
}

impl Encodable for DiskBlockHeader {
    /// Encodes a `DiskBlockHeader` to a writer using the consensus encoding.
    fn consensus_encode<W: bitcoin::io::Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> bitcoin::io::Result<usize> {
        let mut len = 80 + 1; // Header + tag
        match self {
            Self::FullyValid(header, height) => {
                0x00_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            Self::Orphan(header) => {
                0x01_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
            }
            Self::HeadersOnly(header, height) => {
                0x02_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            Self::InFork(header, height) => {
                0x03_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            Self::InvalidChain(header) => {
                0x04_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
            }
            Self::AssumedValid(header, height) => {
                0x05_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
        };
        Ok(len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Internal representation of the chain we are in
pub struct BestChain {
    /// Hash of the last block in the chain we believe has more work on
    pub best_block: BlockHash,

    /// How many blocks are pilled on this chain?
    pub depth: u32,

    /// We actually validated blocks up to this point
    pub validation_index: BlockHash,

    /// Blockchains are not fast-forward only, they might have "forks", sometimes it's useful
    /// to keep track of them, in case they become the best one. This keeps track of some
    /// tips we know about, but are not the best one. We don't keep tips that are too deep
    /// or have too little work if compared to our best one
    pub alternative_tips: Vec<BlockHash>,
}

impl BestChain {
    pub(super) fn new_block(&mut self, block_hash: BlockHash, height: u32) {
        self.best_block = block_hash;
        self.depth = height;
    }

    pub(super) fn valid_block(&mut self, block_hash: BlockHash) {
        self.validation_index = block_hash;
    }
}

impl Encodable for BestChain {
    fn consensus_encode<W: bitcoin::io::Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> bitcoin::io::Result<usize> {
        let mut len = 0;
        len += self.best_block.consensus_encode(writer)?;
        len += self.depth.consensus_encode(writer)?;
        len += self.validation_index.consensus_encode(writer)?;
        len += self.alternative_tips.consensus_encode(writer)?;
        Ok(len)
    }
}

impl Decodable for BestChain {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, encode::Error> {
        let best_block = BlockHash::consensus_decode(reader)?;
        let depth = u32::consensus_decode(reader)?;
        let validation_index = BlockHash::consensus_decode(reader)?;

        let alternative_tips = <Vec<BlockHash>>::consensus_decode(reader)?;
        Ok(Self {
            alternative_tips,
            best_block,
            depth,
            validation_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use bitcoin::BlockHash;
    use bitcoin::CompactTarget;
    use bitcoin::TxMerkleNode;
    use bitcoin::block::Header as BlockHeader;
    use bitcoin::block::Version;
    use bitcoin::consensus::Decodable;
    use bitcoin::consensus::Encodable;
    use bitcoin::hashes::Hash;
    use rand;
    use rand::RngExt;

    use super::*;

    /// Build a header with random values
    fn gen_header() -> BlockHeader {
        let mut rng = rand::rng();

        BlockHeader {
            version: Version::from_consensus(2),
            prev_blockhash: BlockHash::from_byte_array(rng.random()),
            merkle_root: TxMerkleNode::from_byte_array(rng.random()),
            time: rng.random(),
            bits: CompactTarget::from_consensus(rng.random()),
            nonce: rng.random(),
        }
    }

    #[test]
    fn disk_block_header_height() {
        let header = gen_header();
        let h = 100;

        // fully valid
        let fv = DiskBlockHeader::FullyValid(header, h);
        assert_eq!(fv.height(), Some(h));
        assert_eq!(fv.try_height().unwrap(), h);

        // assumed valid
        let av = DiskBlockHeader::AssumedValid(header, h);
        assert_eq!(av.height(), Some(h));
        assert_eq!(av.try_height().unwrap(), h);

        // headers only
        let ho = DiskBlockHeader::HeadersOnly(header, h);
        assert_eq!(ho.height(), Some(h));
        assert_eq!(ho.try_height().unwrap(), h);

        // in‐fork
        let ifk = DiskBlockHeader::InFork(header, h);
        assert_eq!(ifk.height(), Some(h));
        assert_eq!(ifk.try_height().unwrap(), h);

        // orphaned → no height
        let orp = DiskBlockHeader::Orphan(header);
        assert_eq!(orp.height(), None);
        assert!(matches!(
            orp.try_height(),
            Err(BlockchainError::OrphanOrInvalidBlock)
        ));

        // invalid chain → no height
        let inv = DiskBlockHeader::InvalidChain(header);
        assert_eq!(inv.height(), None);
        assert!(matches!(
            inv.try_height(),
            Err(BlockchainError::OrphanOrInvalidBlock)
        ));
    }

    #[test]
    fn disk_block_header_deref() {
        let header = gen_header();
        let disk_header = DiskBlockHeader::FullyValid(header, 1);
        // Must dereference to the same `BlockHeader`
        assert_eq!(*disk_header, header);
        assert_eq!(disk_header.block_hash(), header.block_hash());
    }

    #[test]
    fn encode_decode_disk_block_header() {
        let header = gen_header();
        let h = 123_456;
        let all_variants = vec![
            DiskBlockHeader::FullyValid(header, h),
            DiskBlockHeader::AssumedValid(header, h),
            DiskBlockHeader::HeadersOnly(header, h),
            DiskBlockHeader::InFork(header, h),
            DiskBlockHeader::Orphan(header),
            DiskBlockHeader::InvalidChain(header),
        ];

        for original in all_variants {
            let mut buf = Vec::new();
            let written = original.consensus_encode(&mut buf).unwrap();
            assert_eq!(buf.len(), written, "length matches returned size");

            let mut cursor = Cursor::new(&buf);
            let decoded = DiskBlockHeader::consensus_decode(&mut cursor).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn best_chain_methods() {
        let hash = BlockHash::from_byte_array(rand::random());

        let mut best = BestChain {
            best_block: hash,
            depth: 10,
            validation_index: BlockHash::all_zeros(),
            alternative_tips: Vec::new(),
        };

        // new_block(...)
        let new_hash = BlockHash::from_byte_array(rand::random());
        let new_depth = 20;
        best.new_block(new_hash, new_depth);
        assert_eq!(
            best,
            BestChain {
                best_block: new_hash,
                depth: new_depth,
                validation_index: BlockHash::all_zeros(),
                alternative_tips: Vec::new(),
            }
        );

        // valid_block(...)
        best.valid_block(hash);
        assert_eq!(
            best,
            BestChain {
                best_block: new_hash,
                depth: new_depth,
                validation_index: hash,
                alternative_tips: Vec::new(),
            }
        );
    }

    #[test]
    fn encode_decode_best_chain() {
        let mut rng = rand::rng();
        let alternative_tips = (0..64)
            .map(|_| BlockHash::from_byte_array(rng.random()))
            .collect();

        let best = BestChain {
            best_block: BlockHash::from_byte_array(rng.random()),
            depth: rng.random(),
            validation_index: BlockHash::from_byte_array(rng.random()),
            alternative_tips,
        };

        let mut buf = Vec::new();
        let written = best.consensus_encode(&mut buf).unwrap();
        assert_eq!(buf.len(), written, "written length matches buffer");

        let mut cursor = Cursor::new(&buf);
        let decoded = BestChain::consensus_decode(&mut cursor).unwrap();
        assert_eq!(decoded, best);
    }
}
