//! An in-memory database to store addresses data. Being in-memory means this database is
//! volatile, and all data is lost after the database is dropped or the process is terminated.
//! It's not meant to use in production, but for the integrated testing framework
//!
//! For actual databases that can be used for production code, see [KvDatabase].
use bitcoin::hashes::sha256;
use bitcoin::Txid;
use floresta_common::prelude::sync::RwLock;
use floresta_common::prelude::*;

use super::AddressCacheDatabase;
use super::CachedAddress;
use super::CachedTransaction;
use super::Stats;
#[derive(Debug, Default)]
struct Inner {
    addresses: HashMap<sha256::Hash, CachedAddress>,
    transactions: HashMap<Txid, CachedTransaction>,
    stats: Stats,
    height: u32,
    descriptors: Vec<String>,
}

#[derive(Debug)]
pub enum MemoryDatabaseError {
    PoisonedLock,
}
#[derive(Debug, Default)]
pub struct MemoryDatabase {
    inner: RwLock<Inner>,
}

type Result<T> = floresta_common::prelude::Result<T, MemoryDatabaseError>;

impl MemoryDatabase {
    fn get_inner(&self) -> Result<sync::RwLockReadGuard<Inner>> {
        self.inner
            .read()
            .map_err(|_| MemoryDatabaseError::PoisonedLock)
    }
    fn get_inner_mut(&self) -> Result<sync::RwLockWriteGuard<Inner>> {
        self.inner
            .write()
            .map_err(|_| MemoryDatabaseError::PoisonedLock)
    }
    pub fn new() -> MemoryDatabase {
        MemoryDatabase {
            inner: Default::default(),
        }
    }
}
impl AddressCacheDatabase for MemoryDatabase {
    type Error = MemoryDatabaseError;
    fn save(&self, address: &CachedAddress) {
        self.get_inner_mut()
            .map(|mut inner| {
                inner
                    .addresses
                    .insert(address.script_hash, address.to_owned())
            })
            .unwrap();
    }

    fn load(&self) -> Result<Vec<CachedAddress>> {
        Ok(self.get_inner()?.addresses.values().cloned().collect())
    }

    fn get_stats(&self) -> Result<super::Stats> {
        Ok(self.get_inner()?.stats.to_owned())
    }

    fn save_stats(&self, stats: &super::Stats) -> Result<()> {
        self.get_inner_mut().map(|mut inner| {
            inner.stats.clone_from(stats);
        })?;
        Ok(())
    }

    fn update(&self, address: &super::CachedAddress) {
        self.get_inner_mut()
            .map(|mut inner| {
                inner
                    .addresses
                    .entry(address.script_hash)
                    .and_modify(|addr| addr.clone_from(address));
            })
            .unwrap();
    }

    fn get_cache_height(&self) -> Result<u32> {
        Ok(self.get_inner()?.height)
    }

    fn set_cache_height(&self, height: u32) -> Result<()> {
        self.get_inner_mut()?.height = height;
        Ok(())
    }

    fn desc_save(&self, descriptor: &str) -> Result<()> {
        self.get_inner_mut().map(|mut inner| {
            inner.descriptors.push(descriptor.into());
        })
    }

    fn descs_get(&self) -> Result<Vec<String>> {
        Ok(self.get_inner()?.descriptors.to_owned())
    }

    fn get_transaction(&self, txid: &bitcoin::Txid) -> Result<super::CachedTransaction> {
        if let Some(tx) = self.get_inner()?.transactions.get(txid) {
            return Ok(tx.clone());
        }
        Err(MemoryDatabaseError::PoisonedLock)
    }

    fn save_transaction(&self, tx: &super::CachedTransaction) -> Result<()> {
        self.get_inner_mut()?
            .transactions
            .insert(tx.hash, tx.to_owned());
        Ok(())
    }

    fn list_transactions(&self) -> Result<Vec<Txid>> {
        Ok(self.get_inner()?.transactions.keys().copied().collect())
    }
}
