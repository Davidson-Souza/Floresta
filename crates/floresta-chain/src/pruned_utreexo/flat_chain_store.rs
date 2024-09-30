//! A probably over-engenering database for the chainstore
//!
//! In it's infancy, floresta-chain used `kv` as it's database, since kv is a small and efficient
//! embeded database that doesn't require any runtime dependency. However, floresta-chain uses the
//! database in a very unusual way: it downloads a bunch of small chunks of data that needs to be
//! indexed and retrieved, all at once (~800k for mainnet at the time of writing). If we simply
//! keep evething in memory, and then make one big batch, most embedded databases will see a big
//! spike in heap usage. This would be OK for regular desktops, but floresta aims to run in small,
//! lowe-power devices too, so we can't just assume we have two gigs of RAM to spare. We could make
//! these writes more commonly, but then we reach an I/O bottleneck in those lower-power systems,
//! where usually we won't see high-quality ssds that can make billions of transfers per second.
//!
//! This chainstore was designed to reduce the over-usage of both. We do rely on any extra RAM as
//! kernel buffers, but we also do a decent level of I/O. We get a better perfomance by using
//! a ad-hock storage that exploits the fact that the data we keep is canonical and monotonically
//! increasing, so we keep all headers in a simple flat file, one after the other. So pos(h) = h *
//! size_of(DiskBlockHeader), with a overhead factor of 1. We also need a way to map block hashes
//! into the given header, we do this by keeping a persistent, open-addressing hash map that map block
//! hashes -> heights. Then from the height we can work out the block header in the headers file.
//!
//! ## Calculations
//!
//! We want to keep the load factor for the hash map as low as possible, while also avoiding
//! re-hashing. So we pick a fairly big initial space to it, say 10 million buckets. Each bucket is
//! 4 bytes long, so we have 40 MiB of map. Each [DiskBlockHeader] is 84 bytes long, we only
//! allocate space for a header as needed, so the assimtoptical size for it, assuming 10 million
//! headers, is 84 MiB So 40 + 84 = 134 MiB. The smallest device I know that can run floresta has
//! ~250 MiB of RAM, so we could fit both the index and flat file in RAM all the time. Moreover,
//! we have some newer CPUs that have over a 100MB of L3 cache, we could fit all of this
//! in the L3 cache of some future CPU, and make the process a whole lot faster in the future.
//!
//! The longest chain we have right now is testnet, with about 3 million blocks. That yields a load
//! factor of 0.3. With that load factor, there's a ~1/3 probability of collision, we are expected
//! to have a worst-case search ~2. So we'll need to fetch two nodes to find the one we want. Since
//! each node is a u32, most of the time we'll pull the second node too (64 bits machines can't
//! pull 32 bits values from memory).
//!
//! # Safety
//!
//! This is completely reckless and bare-bones, so it needs some safety precautions:
//!     (i): we must make sure that the map and flat file are initialized correctly
//!     (ii): we must make sure that the map won't give us any height greater than the current tip
//!     (iii): we must make sure that the load factor **never** reaches one
//! i and ii will cause a segfault, iii will turn the addition (or search for non-existent values)
//! an infinite loop. If we are about to reach the map's capacity, we should re-hash with a new
//! capacity.
use core::mem::size_of;
use std::fs::DirBuilder;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;

use bitcoin::consensus::ReadExt;
use bitcoin::hashes::Hash;
use bitcoin::BlockHash;
use floresta_common::impl_error_from;
use memmap2::MmapMut;
use memmap2::MmapOptions;

use super::ChainStore;
use crate::BestChain;
use crate::DatabaseError;
use crate::DiskBlockHeader;

#[derive(Clone)]
pub struct FlatChainStoreConfig {
    pub index_mmap_size: usize,
    pub headers_file_map_size: usize,
    pub _cache_size: usize,
    pub _file_permission: usize,
    pub path: String,
}

pub enum Entry {
    Empty(*mut u32),
    Occupied(*mut u32),
}

#[derive(Debug)]
pub struct DiskBlockHeaderAndHash {
    header: DiskBlockHeader,
    hash: BlockHash,
}

#[derive(Debug, Clone, Copy)]
struct Metadata {
    /// Hash of the last block in the chain we believe has more work on
    best_block: BlockHash,
    /// How many blocks are pilled on this chain?
    depth: u32,
    /// We actually validated blocks up to this point
    validation_index: BlockHash,
    /// Blockchains are not fast-forward only, they might have "forks", sometimes it's useful
    /// to keep track of them, in case they become the best one. This keeps track of some
    /// tips we know about, but are not the best one. We don't keep tips that are too deep
    /// or has too little work if compared to our best one
    alternative_tips: [BlockHash; 64], // we hope to never have more than 64 alt-chains
    /// Saves the height occupied by the assume valid block
    assume_valid_index: u32,

    acc: [u8; 32 * 64 + 8],
}

#[derive(Debug)]
pub enum FlatChainstoreError {
    Io(std::io::Error),
    NotFound,
}

impl DatabaseError for FlatChainstoreError {}

impl_error_from!(FlatChainstoreError, std::io::Error, Io);

pub struct FlatChainStore {
    headers_map: MmapMut,
    index_map: MmapMut,
    metadata_map: MmapMut,
    _index_file: File,
}

unsafe impl Send for FlatChainStore {}
unsafe impl Sync for FlatChainStore {}

impl FlatChainStore {
    /// Creates a new storage, given a set of configs
    ///
    /// If any of the I/O operations fail, this function should return an error
    pub fn new(config: FlatChainStoreConfig) -> Result<Self, FlatChainstoreError> {
        DirBuilder::new().recursive(true).create(&config.path)?;

        let index_file = format!("{}/blocks_index.bin", config.path);
        let headers_file = format!("{}/headers.bin", config.path);
        let metadata_file = format!("{}/metadata.bin", config.path);

        let mut index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&index_file)?;

        // this creates a "sparse file", a file that pretends to have a given size, but won't
        // occupy all that size on disk. Reads to unpoppulated parts of that file will return 0x00
        index_file.seek(SeekFrom::Start(1_000_000_000 * size_of::<u32>() as u64))?;
        // if the file isn't initialized, seek to the end and write a singe zero by to create the
        // sparse file.
        if index_file.read_u8().is_err() {
            index_file.write(&[0])?;
        }

        let mut headers_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&headers_file)?;

        headers_file.seek(SeekFrom::Start(1_000_000_000 * size_of::<u32>() as u64))?;
        // if the file isn't initialized, seek to the end and write a singe zero by to create the
        // sparse file.
        if headers_file.read_u8().is_err() {
            headers_file.write(&[0])?;
        }

        let metadata_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&metadata_file)?;

        metadata_file.set_len(size_of::<Metadata>() as u64)?;

        let headers_map = unsafe {
            MmapOptions::new()
                .len(config.headers_file_map_size)
                .map_mut(&headers_file)?
        };
        let index_map = unsafe {
            MmapOptions::default()
                .len(config.index_mmap_size)
                .map_mut(&index_file)?
        };
        let metadata_map = unsafe { MmapOptions::default().len(10_000).map_mut(&metadata_file)? };

        Ok(Self {
            headers_map,
            metadata_map,
            _index_file: index_file,
            index_map,
        })
    }

    /// Returns a [DiskBlockHeader] given a block height. This height must be less than or equal to
    /// the total headers inside our storage. It is UB to call this function with a height >
    /// total_headers
    unsafe fn get_header_by_height(
        &self,
        height: u32,
    ) -> Result<DiskBlockHeaderAndHash, FlatChainstoreError> {
        let pos = height as usize * size_of::<DiskBlockHeaderAndHash>();
        let header =
            self.headers_map.as_ptr().wrapping_add(pos as usize) as *mut DiskBlockHeaderAndHash;

        Ok(header.read())
    }

    unsafe fn get_acc_inner(&self) -> Result<Vec<u8>, FlatChainstoreError> {
        let ptr = self.metadata_map.as_ptr() as *mut Metadata;
        let metadata = *ptr;

        Ok(metadata.acc.into_iter().collect())
    }

    unsafe fn do_save_height(&self, best_block: BestChain) -> Result<(), FlatChainstoreError> {
        let ptr = self.metadata_map.as_ptr() as *mut Metadata;
        let metadata = &mut *ptr;

        metadata.assume_valid_index = best_block.assume_valid_index;
        metadata.best_block = best_block.best_block;
        metadata.depth = best_block.depth;
        metadata.validation_index = best_block.validation_index;
        assert!(best_block.alternative_tips.len() <= 64);
        metadata
            .alternative_tips
            .as_mut_ptr()
            .copy_from_nonoverlapping(
                best_block.alternative_tips.as_ptr(),
                best_block.alternative_tips.len(),
            );
        Ok(())
    }

    unsafe fn get_best_chain(&self) -> Result<BestChain, FlatChainstoreError> {
        let ptr = self.metadata_map.as_ptr() as *mut Metadata;
        let metadata = *ptr;

        Ok(BestChain {
            best_block: metadata.best_block,
            depth: metadata.depth,
            validation_index: metadata.validation_index,
            alternative_tips: metadata
                .alternative_tips
                .into_iter()
                .take_while(|tip| *tip != BlockHash::all_zeros())
                .collect(),
            assume_valid_index: metadata.assume_valid_index,
        })
    }

    /// The (short) hash function we use to compute where is the map a given block height should be
    ///
    /// In our normal operation, we sometime need to retrieve a header based on a block hash,
    /// rather than height. Block hashes are 256 bits long, so we can't really use them to index
    /// here. Truncating the sha256 is one option, this short hash function will give us a better
    /// randomization over the data, and it's super easy to compute anyways.
    fn index_hash_fn(block_hash: BlockHash) -> u32 {
        let mut hash: u32 = 0;
        for i in block_hash.to_byte_array() {
            hash = hash.wrapping_add(i as u32);
            hash = hash.wrapping_add(hash << 10);
            hash ^= hash >> 6;
        }

        hash = hash.wrapping_add(hash << 3);
        hash ^= hash >> 11;
        hash = hash.wrapping_add(hash << 15);

        hash
    }

    /// Returns the block header, given a block hash
    ///
    /// If the header doesn't exist in our index, it'll return an error
    unsafe fn get_header_by_hash(
        &self,
        hash: BlockHash,
    ) -> Result<Option<DiskBlockHeader>, FlatChainstoreError> {
        let height = self
            .get_index_for_hash(hash)?
            .map(|height| self.get_header_by_height(height));
        Ok(height.transpose()?.map(|x| x.header))
    }

    /// Writes a block header in our storage
    ///
    /// This function will allocate size_of(DiskBlockHeader) bytes in our file and write the raw
    /// header there. May return an error if we can't grow the file
    unsafe fn write_header_to_storage(
        &self,
        header: DiskBlockHeader,
    ) -> Result<(), FlatChainstoreError> {
        let offset = size_of::<DiskBlockHeaderAndHash>() * header.height().unwrap() as usize;
        let ptr = self.headers_map.as_ptr().wrapping_add(offset) as *mut DiskBlockHeaderAndHash;
        *ptr = DiskBlockHeaderAndHash {
            header,
            hash: header.block_hash(),
        };

        Ok(())
    }

    /// Updates our index to map a block hash to a block height
    ///
    /// After accepting a new block, this should be updated to tell us in what position in the
    /// chain that block is.
    unsafe fn set_index_for_hash(
        &self,
        hash: BlockHash,
        index: u32,
    ) -> Result<(), FlatChainstoreError> {
        let pos = self.hash_map_find_pos(hash)?;
        match pos {
            Entry::Empty(pos) | Entry::Occupied(pos) => pos.write(index),
        }

        Ok(())
    }

    /// Return the block height of a given block hash
    unsafe fn get_index_for_hash(
        &self,
        hash: BlockHash,
    ) -> Result<Option<u32>, FlatChainstoreError> {
        match self.hash_map_find_pos(hash)? {
            Entry::Empty(_) => Ok(None),
            Entry::Occupied(index) => Ok(Some(*index)),
        }
    }

    /// Returns the position inside the hash map where a given hash should be
    ///
    /// This function will compute the short hash for this header, look up the position inside the
    /// hash map, and if the position is occupied, return keep incrementing the count, until we
    /// either find the record or find a vacant position. If we can't find the original thing, we
    /// return where it would be added in the index. So if you're adding a new record, call this
    /// function (it will return a vacant position) and just write the height there.
    unsafe fn hash_map_find_pos(
        &self,
        block_hash: BlockHash,
    ) -> Result<Entry, FlatChainstoreError> {
        let mut hash = Self::index_hash_fn(block_hash) as usize;
        loop {
            let index = self
                .index_map
                .as_ptr()
                .wrapping_add((hash % 10_000_000) * size_of::<u32>())
                as *mut u32;

            let header = self.get_header_by_height(*index)?;
            if header.hash == block_hash {
                return Ok(Entry::Occupied(index));
            }

            if *index == 0 {
                return Ok(Entry::Empty(index));
            }

            hash += 1;
        }
    }

    unsafe fn do_save_roots(&self, roots: Vec<u8>) -> Result<(), FlatChainstoreError> {
        let ptr = self.metadata_map.as_ptr() as *mut Metadata;
        let mut metadata = *ptr;

        metadata
            .acc
            .as_mut_ptr()
            .copy_from_nonoverlapping(roots.as_ptr(), roots.len());

        Ok(())
    }

    unsafe fn do_flush(&self) -> Result<(), FlatChainstoreError> {
        self.headers_map.flush()?;
        self.index_map.flush()?;
        self.metadata_map.flush()?;

        Ok(())
    }
}

impl ChainStore for FlatChainStore {
    type Error = FlatChainstoreError;

    fn flush(&self) -> Result<(), Self::Error> {
        unsafe { self.do_flush() }
    }

    fn save_roots(&self, roots: Vec<u8>) -> Result<(), Self::Error> {
        unsafe { self.do_save_roots(roots) }
    }

    fn load_roots(&self) -> Result<Option<Vec<u8>>, Self::Error> {
        unsafe { self.get_acc_inner().map(|x| Some(x)) }
    }

    fn get_header(&self, block_hash: &BlockHash) -> Result<Option<DiskBlockHeader>, Self::Error> {
        unsafe { self.get_header_by_hash(*block_hash) }
    }

    fn load_height(&self) -> Result<Option<crate::BestChain>, Self::Error> {
        unsafe {
            self.get_best_chain()
                .map(|x| if x.depth == 0 { None } else { Some(x) })
        }
    }

    fn save_height(&self, height: &crate::BestChain) -> Result<(), Self::Error> {
        unsafe { self.do_save_height(height.clone()) }
    }

    fn save_header(&self, header: &DiskBlockHeader) -> Result<(), Self::Error> {
        unsafe { self.write_header_to_storage(*header) }
    }

    fn get_block_hash(&self, height: u32) -> Result<Option<BlockHash>, Self::Error> {
        let header = unsafe { Self::get_header_by_height(&self, height)? };
        Ok(Some(header.hash))
    }

    fn update_block_index(&self, height: u32, hash: BlockHash) -> Result<(), Self::Error> {
        unsafe { self.set_index_for_hash(hash, height) }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use bitcoin::block::Header;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::Decodable;
    use bitcoin::constants::genesis_block;
    use bitcoin::Block;

    use super::FlatChainStore;
    use crate::pruned_utreexo::ChainStore;
    use crate::pruned_utreexo::UpdatableChainstate;
    use crate::AssumeValidArg;
    use crate::BestChain;
    use crate::ChainState;
    use crate::DiskBlockHeader;

    fn get_test_chainstore() -> FlatChainStore {
        let test_id = rand::random::<u64>();
        let config = super::FlatChainStoreConfig {
            index_mmap_size: 10_00_000_000,
            headers_file_map_size: 100_000_000,
            _cache_size: 10,
            _file_permission: 660,
            path: format!("./data/{test_id}/"),
        };

        FlatChainStore::new(config).unwrap()
    }

    #[test]
    fn test_create_chainstore() {
        let config = super::FlatChainStoreConfig {
            index_mmap_size: 1_000_000,
            headers_file_map_size: 1_000_000,
            _cache_size: 10,
            _file_permission: 660,
            path: ".".into(),
        };
        let _store = FlatChainStore::new(config).unwrap();
    }

    #[test]
    fn test_save_headers() {
        let store = get_test_chainstore();

        let blocks = include_str!("./testdata/blocks.txt");
        for (i, line) in blocks.lines().enumerate() {
            let block = hex::decode(line).unwrap();
            let block: Block = deserialize(&block).unwrap();

            store
                .save_header(&DiskBlockHeader::FullyValid(block.header, i as u32))
                .unwrap();
        }
    }

    #[test]
    fn test_save_height() {
        let store = get_test_chainstore();
        let height = BestChain {
            alternative_tips: Vec::new(),
            assume_valid_index: 0,
            validation_index: genesis_block(bitcoin::Network::Signet).block_hash(),
            depth: 1,
            best_block: genesis_block(bitcoin::Network::Signet).block_hash(),
        };

        store.save_height(&height).unwrap();

        let recovered = store.load_height().unwrap().unwrap();

        assert_eq!(recovered, height);
    }

    #[test]
    fn test_index() {
        let store = get_test_chainstore();
        let mut hashes = Vec::new();
        let blocks = include_str!("./testdata/blocks.txt");
        for (i, line) in blocks.lines().enumerate() {
            let block = hex::decode(line).unwrap();
            let block: Block = deserialize(&block).unwrap();
            hashes.push(block.block_hash());
            store
                .save_header(&DiskBlockHeader::FullyValid(block.header, i as u32))
                .unwrap();
            store
                .update_block_index(i as u32, block.block_hash())
                .unwrap();
        }

        for hash in hashes {
            if hash == genesis_block(bitcoin::Network::Regtest).block_hash() {
                continue;
            }
            let header = store.get_header(&hash).unwrap().unwrap();
            assert_eq!(header.block_hash(), hash);
        }
    }

    #[test]
    fn accept_mainnet_headers() {
        // Accepts the first 10235 mainnet headers
        let file = include_bytes!("./testdata/headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
        let mut cursor = Cursor::new(uncompressed);
        let store = get_test_chainstore();
        let chain = ChainState::new(
            store,
            crate::Network::Bitcoin.into(),
            AssumeValidArg::Hardcoded,
        );

        while let Ok(header) = Header::consensus_decode(&mut cursor) {
            chain.accept_header(header).unwrap();
        }
    }

    #[test]
    fn accept_first_signet_headers() {
        // Accepts the first 2016 signet headers
        let file = include_bytes!("./testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
        let mut cursor = Cursor::new(uncompressed);

        let store = get_test_chainstore();
        let chain = ChainState::new(
            store,
            crate::Network::Signet.into(),
            AssumeValidArg::Hardcoded,
        );

        while let Ok(header) = Header::consensus_decode(&mut cursor) {
            chain.accept_header(header).unwrap();
        }
    }
}
