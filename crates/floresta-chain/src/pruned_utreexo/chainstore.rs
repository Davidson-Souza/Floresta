// Written in 2022 by Davidson Souza

//! This is a basic kv database that stores all metadata about our blockchain and utreexo
//! state.
use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::serialize;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::BlockHash;
use kv::Batch;
use kv::Bucket;
use spin::RwLock;

use crate::prelude::*;
#[derive(Debug, Clone, Copy)]
pub enum DiskBlockHeader {
    FullyValid(BlockHeader, u32),
    AssumedValid(BlockHeader, u32),
    Orphan(BlockHeader),
    HeadersOnly(BlockHeader, u32),
    InFork(BlockHeader, u32),
    InvalidChain(BlockHeader),
}

impl DiskBlockHeader {
    pub fn block_hash(&self) -> BlockHash {
        self.deref().block_hash()
    }
    pub fn height(&self) -> Option<u32> {
        match self {
            DiskBlockHeader::FullyValid(_, height) => Some(*height),
            DiskBlockHeader::Orphan(_) => None,
            DiskBlockHeader::HeadersOnly(_, height) => Some(*height),
            DiskBlockHeader::InFork(_, height) => Some(*height),
            DiskBlockHeader::InvalidChain(_) => None,
            DiskBlockHeader::AssumedValid(_, height) => Some(*height),
        }
    }
}
impl Deref for DiskBlockHeader {
    type Target = BlockHeader;
    fn deref(&self) -> &Self::Target {
        match self {
            DiskBlockHeader::FullyValid(header, _) => header,
            DiskBlockHeader::Orphan(header) => header,
            DiskBlockHeader::HeadersOnly(header, _) => header,
            DiskBlockHeader::InFork(header, _) => header,
            DiskBlockHeader::InvalidChain(header) => header,
            DiskBlockHeader::AssumedValid(header, _) => header,
        }
    }
}
impl Decodable for DiskBlockHeader {
    fn consensus_decode<R: Read + ?Sized>(
        reader: &mut R,
    ) -> core::result::Result<Self, bitcoin::consensus::encode::Error> {
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
            _ => unreachable!(),
        }
    }
}
impl Encodable for DiskBlockHeader {
    fn consensus_encode<W: Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> core::result::Result<usize, ioError> {
        let mut len = 80 + 1; // Header + tag
        match self {
            DiskBlockHeader::FullyValid(header, height) => {
                0x00_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            DiskBlockHeader::Orphan(header) => {
                0x01_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
            }
            DiskBlockHeader::HeadersOnly(header, height) => {
                0x02_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            DiskBlockHeader::InFork(header, height) => {
                0x03_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
            DiskBlockHeader::InvalidChain(header) => {
                0x04_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
            }
            DiskBlockHeader::AssumedValid(header, height) => {
                0x05_u8.consensus_encode(writer)?;
                header.consensus_encode(writer)?;
                height.consensus_encode(writer)?;
                len += 4;
            }
        };
        Ok(len)
    }
}
use kv::Config;
use kv::Integer;
use kv::Store;

use super::chain_state::BestChain;
use super::ChainStore;

pub struct KvChainStore<'a> {
    _store: Store,
    headers: Bucket<'a, Vec<u8>, Vec<u8>>,
    index: Bucket<'a, Integer, Vec<u8>>,
    meta: Bucket<'a, &'a str, Vec<u8>>,
    headers_cache: RwLock<HashMap<BlockHash, DiskBlockHeader>>,
    index_cache: RwLock<HashMap<u32, BlockHash>>,
}

impl<'a> KvChainStore<'a> {
    pub fn new(datadir: String) -> Result<KvChainStore<'a>, kv::Error> {
        // Configure the database
        let cfg = Config::new(datadir + "/chain_data").cache_capacity(100_000_000);

        // Open the key/value store
        let store = Store::new(cfg)?;

        Ok(KvChainStore {
            headers: store.bucket(Some("headers"))?,
            index: store.bucket(Some("index"))?,
            meta: store.bucket(None)?,
            _store: store,
            headers_cache: RwLock::new(HashMap::new()),
            index_cache: RwLock::new(HashMap::new()),
        })
    }
}

impl<'a> ChainStore for KvChainStore<'a> {
    type Error = kv::Error;
    fn load_roots(&self) -> Result<Option<Vec<u8>>, Self::Error> {
        self.meta.get(&"roots")
    }

    fn save_roots(&self, roots: Vec<u8>) -> Result<(), Self::Error> {
        self.meta.set(&"roots", &roots)?;
        Ok(())
    }

    fn load_height(&self) -> Result<Option<BestChain>, Self::Error> {
        let height = self.meta.get(&"height")?;
        if let Some(height) = height {
            return Ok(Some(deserialize(&height).unwrap()));
        }

        Ok(None)
    }

    fn save_height(&self, height: &BestChain) -> Result<(), Self::Error> {
        let height = serialize(height);
        self.meta.set(&"height", &height)?;
        Ok(())
    }

    fn get_header(&self, block_hash: &BlockHash) -> Result<Option<DiskBlockHeader>, Self::Error> {
        match self.headers_cache.read().get(block_hash) {
            Some(header) => Ok(Some(*header)),
            None => {
                let block_hash = serialize(&block_hash);
                Ok(self
                    .headers
                    .get(&block_hash)?
                    .and_then(|b| deserialize(&b).ok()))
            }
        }
    }

    fn flush(&self) -> Result<(), Self::Error> {
        // save all headers in batch
        let mut batch = Batch::new();
        for header in self.headers_cache.read().iter() {
            let ser_header = serialize(header.1);
            let block_hash = serialize(&header.1.block_hash());
            batch.set(&block_hash, &ser_header)?;
        }
        self.headers.batch(batch)?;
        self.headers_cache.write().clear();

        // save all index in batch
        let mut batch = Batch::new();
        for (height, hash) in self.index_cache.read().iter() {
            let ser_hash = serialize(hash);
            batch.set(&Integer::from(*height), &ser_hash)?;
        }
        self.index.batch(batch)?;
        self.index_cache.write().clear();

        // Flush the header bucket
        self.headers.flush()?;
        // Flush the block index
        self.index.flush()?;
        // Flush the default bucket with meta-info
        self.meta.flush()?;
        Ok(())
    }

    fn save_header(&self, header: &DiskBlockHeader) -> Result<(), Self::Error> {
        self.headers_cache
            .write()
            .insert(header.block_hash(), *header);
        Ok(())
    }

    fn get_block_hash(&self, height: u32) -> Result<Option<BlockHash>, Self::Error> {
        match self.index_cache.read().get(&height).cloned() {
            Some(hash) => Ok(Some(hash)),
            None => Ok(self
                .index
                .get(&Integer::from(height))?
                .and_then(|b| deserialize(&b).ok())),
        }
    }

    fn update_block_index(&self, height: u32, hash: BlockHash) -> Result<(), Self::Error> {
        self.index_cache.write().insert(height, hash);
        Ok(())
    }
}
