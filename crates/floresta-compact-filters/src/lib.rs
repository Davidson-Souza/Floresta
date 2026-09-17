// SPDX-License-Identifier: MIT OR Apache-2.0

//! BIP157/158 compact block-filter synchronization, storage, and rescanning.
//!
//! Filter headers are persisted for every block. Full filters are fetched or built only when
//! needed and retained in a bounded in-memory cache.

// cargo docs customization
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/249173822")]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/getfloresta/floresta-media/master/logo_png/Icon-Green(main).png"
)]

use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;

use bitcoin::BlockHash;
use bitcoin::FilterHeader;
use bitcoin::bip158::BlockFilter;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::consensus::encode;
use floresta_common::impl_error_from;

pub mod filters_man;

/// Number of recent full filters retained by [`FlatFilterStore::new`].
pub const DEFAULT_FILTER_CACHE_SIZE: usize = 1_000;

/// Errors returned by [`FlatFilterStore`].
#[derive(Debug)]
pub enum FlatFilterStoreError {
    /// No filter-header entry exists at the requested height.
    NotFound,

    /// Bitcoin consensus encoding or decoding failed.
    Encode(encode::Error),

    /// A Bitcoin I/O operation failed.
    BitcoinIo(bitcoin::io::Error),

    /// A filesystem operation failed.
    StdIo(std::io::Error),

    /// The persisted file does not contain complete fixed-size records.
    CorruptedFile,
}

impl PartialEq for FlatFilterStoreError {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::NotFound, Self::NotFound)
                | (Self::Encode(_), Self::Encode(_))
                | (Self::BitcoinIo(_), Self::BitcoinIo(_))
                | (Self::StdIo(_), Self::StdIo(_))
                | (Self::CorruptedFile, Self::CorruptedFile)
        )
    }
}

impl Eq for FlatFilterStoreError {}

impl Display for FlatFilterStoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "filter header not found"),
            Self::Encode(error) => write!(f, "filter-header encoding error: {error}"),
            Self::BitcoinIo(error) => write!(f, "filter-header Bitcoin I/O error: {error}"),
            Self::StdIo(error) => write!(f, "filter-header filesystem error: {error}"),
            Self::CorruptedFile => write!(f, "filter-header file is truncated or too large"),
        }
    }
}

impl std::error::Error for FlatFilterStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(error) => Some(error),
            Self::BitcoinIo(error) => Some(error),
            Self::StdIo(error) => Some(error),
            Self::NotFound | Self::CorruptedFile => None,
        }
    }
}

impl_error_from!(FlatFilterStoreError, bitcoin::io::Error, BitcoinIo);
impl_error_from!(FlatFilterStoreError, encode::Error, Encode);
impl_error_from!(FlatFilterStoreError, std::io::Error, StdIo);

/// A persisted filter header and the block to which it belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterHeaderEntry {
    /// Hash of the block committed to by `filter_header`.
    pub block_hash: BlockHash,

    /// BIP157 filter header for `block_hash`.
    pub filter_header: FilterHeader,
}

impl FilterHeaderEntry {
    /// Serialized size of one entry.
    pub const SERIALIZED_SIZE: u64 = 64;
}

impl Encodable for FilterHeaderEntry {
    fn consensus_encode<W: bitcoin::io::Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> Result<usize, bitcoin::io::Error> {
        let mut written = self.block_hash.consensus_encode(writer)?;
        written += self.filter_header.consensus_encode(writer)?;
        Ok(written)
    }
}

impl Decodable for FilterHeaderEntry {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, encode::Error> {
        Ok(Self {
            block_hash: BlockHash::consensus_decode(reader)?,
            filter_header: FilterHeader::consensus_decode(reader)?,
        })
    }
}

/// Storage required by the compact-filter manager.
pub trait FilterHeadersStore: Send + 'static {
    /// Appends a filter-header entry at the next height.
    fn put_filter_header(
        &mut self,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<(), FlatFilterStoreError>;

    /// Returns the filter header at `height`.
    fn get_filter_header(&mut self, height: u32) -> Result<FilterHeader, FlatFilterStoreError>;

    /// Returns the block hash associated with the filter header at `height`.
    fn get_block_hash(&mut self, height: u32) -> Result<BlockHash, FlatFilterStoreError>;

    /// Replaces a filter-header entry and returns the previous filter header.
    fn update_filter_header(
        &mut self,
        height: u32,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<FilterHeader, FlatFilterStoreError>;

    /// Returns the last stored height, or `None` when the store is empty.
    fn get_height(&self) -> Result<Option<u32>, FlatFilterStoreError>;

    /// Removes every entry after `height`, or clears the store when it is `None`.
    fn truncate(&mut self, height: Option<u32>) -> Result<(), FlatFilterStoreError>;

    /// Adds a validated full filter to the bounded cache.
    fn put_filter(&mut self, height: u32, filter: BlockFilter) -> Result<(), FlatFilterStoreError>;

    /// Returns a cached full filter, if present.
    fn get_filter(&self, height: u32) -> Result<Option<BlockFilter>, FlatFilterStoreError>;

    /// Flushes persisted filter headers to disk.
    fn flush(&mut self) -> Result<(), FlatFilterStoreError>;
}

/// Flat-file filter-header storage with a bounded in-memory full-filter cache.
#[derive(Debug)]
pub struct FlatFilterStore {
    reader: BufReader<File>,
    writer: BufWriter<File>,
    len: u64,
    filter_cache: HashMap<u32, BlockFilter>,
    cache_order: VecDeque<u32>,
    cache_capacity: usize,
}

impl FlatFilterStore {
    /// Opens a filter-header file and retains the latest 1,000 fetched filters in memory.
    pub fn new(file: &Path) -> Result<Self, FlatFilterStoreError> {
        Self::with_cache_capacity(file, DEFAULT_FILTER_CACHE_SIZE)
    }

    /// Opens a filter-header file with an explicit full-filter cache capacity.
    pub fn with_cache_capacity(
        file: &Path,
        cache_capacity: usize,
    ) -> Result<Self, FlatFilterStoreError> {
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(file)?;
        let len = file.metadata()?.len();

        if len % FilterHeaderEntry::SERIALIZED_SIZE != 0 {
            return Err(FlatFilterStoreError::CorruptedFile);
        }

        let reader = BufReader::new(file.try_clone()?);
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;

        let store = Self {
            reader,
            writer,
            len,
            filter_cache: HashMap::with_capacity(cache_capacity),
            cache_order: VecDeque::with_capacity(cache_capacity),
            cache_capacity,
        };

        store.entry_count()?;
        Ok(store)
    }

    fn entry_count(&self) -> Result<u32, FlatFilterStoreError> {
        let count = self.len / FilterHeaderEntry::SERIALIZED_SIZE;
        u32::try_from(count).map_err(|_| FlatFilterStoreError::CorruptedFile)
    }

    fn read_entry(&mut self, height: u32) -> Result<FilterHeaderEntry, FlatFilterStoreError> {
        let offset = u64::from(height)
            .checked_mul(FilterHeaderEntry::SERIALIZED_SIZE)
            .ok_or(FlatFilterStoreError::CorruptedFile)?;
        let end = offset
            .checked_add(FilterHeaderEntry::SERIALIZED_SIZE)
            .ok_or(FlatFilterStoreError::CorruptedFile)?;

        if end > self.len {
            return Err(FlatFilterStoreError::NotFound);
        }

        self.writer.flush()?;
        self.reader.seek(SeekFrom::Start(offset))?;
        Ok(FilterHeaderEntry::consensus_decode(&mut self.reader)?)
    }

    fn write_entry(
        &mut self,
        height: u32,
        entry: FilterHeaderEntry,
    ) -> Result<(), FlatFilterStoreError> {
        let offset = u64::from(height)
            .checked_mul(FilterHeaderEntry::SERIALIZED_SIZE)
            .ok_or(FlatFilterStoreError::CorruptedFile)?;

        if offset >= self.len {
            return Err(FlatFilterStoreError::NotFound);
        }

        self.writer.seek(SeekFrom::Start(offset))?;
        entry.consensus_encode(&mut self.writer)?;
        self.writer.flush()?;
        Ok(())
    }
}

impl FilterHeadersStore for FlatFilterStore {
    fn put_filter_header(
        &mut self,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<(), FlatFilterStoreError> {
        self.writer.seek(SeekFrom::End(0))?;
        FilterHeaderEntry {
            block_hash,
            filter_header,
        }
        .consensus_encode(&mut self.writer)?;
        self.len = self
            .len
            .checked_add(FilterHeaderEntry::SERIALIZED_SIZE)
            .ok_or(FlatFilterStoreError::CorruptedFile)?;
        Ok(())
    }

    fn get_filter_header(&mut self, height: u32) -> Result<FilterHeader, FlatFilterStoreError> {
        Ok(self.read_entry(height)?.filter_header)
    }

    fn get_block_hash(&mut self, height: u32) -> Result<BlockHash, FlatFilterStoreError> {
        Ok(self.read_entry(height)?.block_hash)
    }

    fn update_filter_header(
        &mut self,
        height: u32,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<FilterHeader, FlatFilterStoreError> {
        let previous = self.read_entry(height)?.filter_header;
        self.write_entry(
            height,
            FilterHeaderEntry {
                block_hash,
                filter_header,
            },
        )?;
        Ok(previous)
    }

    fn get_height(&self) -> Result<Option<u32>, FlatFilterStoreError> {
        let count = self.entry_count()?;
        Ok(count.checked_sub(1))
    }

    fn truncate(&mut self, height: Option<u32>) -> Result<(), FlatFilterStoreError> {
        let entries = height.map_or(0_u64, |height| u64::from(height) + 1);
        let len = entries
            .checked_mul(FilterHeaderEntry::SERIALIZED_SIZE)
            .ok_or(FlatFilterStoreError::CorruptedFile)?;

        if len > self.len {
            return Err(FlatFilterStoreError::NotFound);
        }

        self.writer.flush()?;
        self.writer.get_ref().set_len(len)?;
        self.writer.seek(SeekFrom::End(0))?;
        self.reader.seek(SeekFrom::Start(0))?;
        self.len = len;

        self.filter_cache
            .retain(|cached_height, _| height.is_some_and(|height| *cached_height <= height));
        self.cache_order
            .retain(|cached_height| height.is_some_and(|height| *cached_height <= height));
        Ok(())
    }

    fn put_filter(&mut self, height: u32, filter: BlockFilter) -> Result<(), FlatFilterStoreError> {
        if self.cache_capacity == 0 {
            return Ok(());
        }

        if self.filter_cache.contains_key(&height) {
            self.cache_order
                .retain(|cached_height| *cached_height != height);
        }

        self.filter_cache.insert(height, filter);
        self.cache_order.push_back(height);

        while self.cache_order.len() > self.cache_capacity {
            if let Some(evicted_height) = self.cache_order.pop_front() {
                self.filter_cache.remove(&evicted_height);
            }
        }

        Ok(())
    }

    fn get_filter(&self, height: u32) -> Result<Option<BlockFilter>, FlatFilterStoreError> {
        Ok(self.filter_cache.get(&height).cloned())
    }

    fn flush(&mut self) -> Result<(), FlatFilterStoreError> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;

    use bitcoin::Network;
    use bitcoin::blockdata::constants::genesis_block;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::serialize;
    use bitcoin::hashes::Hash;
    use tempfile::NamedTempFile;

    use super::*;

    fn entry(byte: u8) -> FilterHeaderEntry {
        FilterHeaderEntry {
            block_hash: BlockHash::from_byte_array([byte; 32]),
            filter_header: FilterHeader::from_byte_array([byte.wrapping_add(1); 32]),
        }
    }

    #[test]
    fn filter_header_entry_roundtrips() {
        let entry = entry(42);
        let encoded = serialize(&entry);

        assert_eq!(encoded.len() as u64, FilterHeaderEntry::SERIALIZED_SIZE);
        assert_eq!(deserialize::<FilterHeaderEntry>(&encoded).unwrap(), entry);
    }

    #[test]
    fn filter_headers_persist_and_truncate() {
        let file = NamedTempFile::new().unwrap();
        let mut store = FlatFilterStore::new(file.path()).unwrap();
        let first = entry(1);
        let second = entry(2);

        store
            .put_filter_header(first.block_hash, first.filter_header)
            .unwrap();
        store
            .put_filter_header(second.block_hash, second.filter_header)
            .unwrap();
        store.flush().unwrap();
        assert_eq!(store.get_height().unwrap(), Some(1));
        assert_eq!(store.get_block_hash(1).unwrap(), second.block_hash);

        store.truncate(Some(0)).unwrap();
        drop(store);

        let mut reopened = FlatFilterStore::new(file.path()).unwrap();
        assert_eq!(reopened.get_height().unwrap(), Some(0));
        assert_eq!(reopened.get_filter_header(0).unwrap(), first.filter_header);
        assert_eq!(
            reopened.get_filter_header(1),
            Err(FlatFilterStoreError::NotFound)
        );
    }

    #[test]
    fn rejects_partial_filter_header_record() {
        let file = NamedTempFile::new().unwrap();
        OpenOptions::new()
            .write(true)
            .open(file.path())
            .unwrap()
            .set_len(FilterHeaderEntry::SERIALIZED_SIZE - 1)
            .unwrap();

        assert!(matches!(
            FlatFilterStore::new(file.path()),
            Err(FlatFilterStoreError::CorruptedFile)
        ));
    }

    #[test]
    fn full_filter_cache_is_bounded() {
        let file = NamedTempFile::new().unwrap();
        let mut store = FlatFilterStore::with_cache_capacity(file.path(), 2).unwrap();
        let block = genesis_block(Network::Regtest);
        let filter = BlockFilter::new_script_filter(&block, |outpoint| {
            Err::<bitcoin::ScriptBuf, _>(bitcoin::bip158::Error::UtxoMissing(*outpoint))
        })
        .unwrap();

        store.put_filter(1, filter.clone()).unwrap();
        store.put_filter(2, filter.clone()).unwrap();
        store.put_filter(3, filter.clone()).unwrap();

        assert_eq!(store.get_filter(1).unwrap(), None);
        assert_eq!(store.get_filter(2).unwrap(), Some(filter.clone()));
        assert_eq!(store.get_filter(3).unwrap(), Some(filter));
    }
}
