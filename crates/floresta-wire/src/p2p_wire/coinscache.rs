use std::collections::HashMap;
use std::fmt::Display;
use std::num::NonZeroUsize;

use bitcoin::OutPoint;
use bitcoin::TxOut;
use lru::LruCache;

#[derive(Debug, Clone)]
pub enum CacheEntry {
    Spent,
    Unspent(TxOut),
}

pub struct CoinsCache {
    capacity: usize,
    cache: LruCache<OutPoint, CacheEntry>,
}

impl Default for CoinsCache {
    fn default() -> Self {
        Self {
            capacity: 1_000_000,
            cache: LruCache::new(NonZeroUsize::new(1_000_000).unwrap()),
        }
    }
}

impl CoinsCache {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity: capacity.into(),
            cache: LruCache::new(capacity),
        }
    }

    pub fn spend(&mut self, outpoint: &OutPoint) -> Option<TxOut> {
        let Some(entry) = self.cache.pop(&outpoint) else {
            self.cache.put(*outpoint, CacheEntry::Spent);
            return None;
        };

        match entry {
            CacheEntry::Unspent(entry) => Some(entry),

            // TODO(@davidson): Is this reachable?
            CacheEntry::Spent => panic!("Called remove twice for the same UTXO"),
        }
    }

    pub fn create(&mut self, outpoint: OutPoint, output: TxOut) {
        if self.cache.contains(&outpoint) {
            self.cache.pop(&outpoint);
            return;
        };

        self.cache.put(outpoint, CacheEntry::Unspent(output));
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}
