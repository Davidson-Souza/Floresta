use core::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::path::PathBuf;

use kv::Bucket;
use kv::Config;
use kv::Integer;

use crate::BlockFilter;
use crate::BlockFilterStore;

/// Stores the block filters insinde a kv database
#[derive(Clone)]
pub struct KvFilterStore {
    bucket: Bucket<'static, Integer, Vec<u8>>,
}

impl Debug for KvFilterStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvFilterStore").finish()
    }
}

impl KvFilterStore {
    /// Creates a new [KvFilterStore] that stores it's content in `datadir`.
    ///
    /// If the path does't exist it'll be created. This store uses compression by default, if you
    /// want to make more granular configuration over the underlying Kv database, use `with_config`
    /// instead.
    pub fn new(datadir: &PathBuf) -> Self {
        let store = kv::Store::new(kv::Config {
            path: datadir.to_owned(),
            temporary: false,
            use_compression: false,
            flush_every_ms: None,
            cache_capacity: None,
            segment_size: None,
        })
        .expect("Could not open store");

        let bucket = store.bucket(Some("cfilters")).unwrap();
        KvFilterStore { bucket }
    }

    /// Creates a new [KvFilterStore] that stores it's content with a given config
    pub fn with_config(config: Config) -> Self {
        let store = kv::Store::new(config).expect("Could not open database");
        let bucket = store.bucket(Some("cffilters")).unwrap();
        KvFilterStore { bucket }
    }
}

impl BlockFilterStore for KvFilterStore {
    fn get_filter(&self, block_height: u32) -> Option<BlockFilter> {
        let value = self
            .bucket
            .get(&Integer::from(block_height))
            .ok()
            .flatten()?;
        Some(BlockFilter::new(&value))
    }

    fn put_filter(&self, block_height: u32, block_filter: BlockFilter) {
        self.bucket
            .set(&Integer::from(block_height), &block_filter.content)
            .expect("Bucket should be open");
    }

    fn get_height(&self) -> Option<u32> {
        // A bit of a hack to avoid opening a new bucket just for the height
        // write the height as the 0th block
        self.bucket
            .get(&Integer::from(0))
            .ok()
            .flatten()
            .map(|height| {
                let mut _height = [0u8; 4];
                _height.copy_from_slice(&height);
                u32::from_le_bytes(_height)
            })
    }

    fn put_height(&self, height: u32) {
        self.bucket
            .set(&Integer::from(0), &height.to_le_bytes().to_vec())
            .expect("Bucket should be open");
    }
}
