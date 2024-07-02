use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::Error;
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
use bitcoin::Txid;
use floresta_common::impl_error_from;
use floresta_common::prelude::*;
use kv::Bucket;
use kv::Config;
use kv::Store;

use super::AddressCacheDatabase;
use super::Stats;

pub struct KvDatabase(Store, Bucket<'static, String, Vec<u8>>);
impl KvDatabase {
    pub fn new(datadir: String) -> Result<KvDatabase> {
        // Configure the database
        let cfg = Config::new(datadir);

        // Open the key/value store
        let store = Store::new(cfg)?;
        let bucket = store.bucket::<String, Vec<u8>>(Some("addresses"))?;
        Ok(KvDatabase(store, bucket))
    }
}
#[derive(Debug)]
pub enum KvDatabaseError {
    KvError(kv::Error),
    SerdeJsonError(serde_json::Error),
    WalletNotInitialized,
    DeserializeError(Error),
    TransactionNotFound,
}
impl_error_from!(KvDatabaseError, serde_json::Error, SerdeJsonError);
impl_error_from!(KvDatabaseError, kv::Error, KvError);
impl_error_from!(KvDatabaseError, Error, DeserializeError);

impl Display for KvDatabaseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            KvDatabaseError::KvError(e) => write!(f, "KvError: {}", e),
            KvDatabaseError::SerdeJsonError(e) => write!(f, "SerdeJsonError: {}", e),
            KvDatabaseError::WalletNotInitialized => write!(f, "WalletNotInitialized"),
            KvDatabaseError::DeserializeError(e) => write!(f, "DeserializeError: {}", e),
            KvDatabaseError::TransactionNotFound => write!(f, "TransactionNotFound"),
        }
    }
}

impl floresta_common::prelude::Error for KvDatabaseError {}

type Result<T> = floresta_common::prelude::Result<T, KvDatabaseError>;

impl AddressCacheDatabase for KvDatabase {
    type Error = KvDatabaseError;
    fn load(&self) -> Result<Vec<super::CachedAddress>> {
        let mut addresses = Vec::new();
        for item in self.1.iter() {
            let item = item?;
            let key = item.key::<String>()?;
            if *"height" == key || *"desc" == key {
                continue;
            }
            let value: Vec<u8> = item.value().unwrap();
            let value = serde_json::from_slice(&value)?;
            addresses.push(value);
        }
        Ok(addresses)
    }
    fn save(&self, address: &super::CachedAddress) {
        let key = address.script_hash.to_string();
        let value = serde_json::to_vec(&address).expect("Invalid object serialization");

        self.1
            .set(&key, &value)
            .expect("Fatal: Database isn't working");
        self.1.flush().expect("Could not write to disk");
    }
    fn update(&self, address: &super::CachedAddress) {
        self.save(address);
    }
    fn get_cache_height(&self) -> Result<u32> {
        let height = self.1.get(&String::from("height"))?;
        if let Some(height) = height {
            return Ok(deserialize(&height)?);
        }
        Err(KvDatabaseError::WalletNotInitialized)
    }
    fn set_cache_height(&self, height: u32) -> Result<()> {
        self.1.set(&String::from("height"), &serialize(&height))?;
        self.1.flush()?;
        Ok(())
    }

    fn desc_save(&self, descriptor: &str) -> Result<()> {
        let mut descs = self.descs_get()?;
        descs.push(String::from(descriptor));
        self.1
            .set(&String::from("desc"), &serde_json::to_vec(&descs).unwrap())?;
        self.1.flush()?;

        Ok(())
    }

    fn descs_get(&self) -> Result<Vec<String>> {
        let res = self.1.get(&String::from("desc"))?;
        if let Some(res) = res {
            return Ok(serde_json::de::from_slice(&res)?);
        }
        Ok(Vec::new())
    }

    fn get_transaction(&self, txid: &bitcoin::Txid) -> Result<super::CachedTransaction> {
        let store = self.0.bucket::<&[u8], Vec<u8>>(Some("transactions"))?;
        let res = store.get(&txid.as_byte_array().to_vec().as_slice())?;
        if let Some(res) = res {
            return Ok(serde_json::de::from_slice(&res)?);
        }
        Err(KvDatabaseError::TransactionNotFound)
    }

    fn save_transaction(&self, tx: &super::CachedTransaction) -> Result<()> {
        let store = self.0.bucket::<&[u8], Vec<u8>>(Some("transactions"))?;
        let ser_tx = serde_json::to_vec(&tx)?;
        store.set(&tx.tx.txid().as_byte_array().to_vec().as_slice(), &ser_tx)?;
        self.1.flush()?;

        Ok(())
    }

    fn get_stats(&self) -> Result<super::Stats> {
        let store = self.0.bucket::<&[u8], Vec<u8>>(Some("stats"))?;
        let res = store.get(&String::from("stats").as_bytes())?;
        if let Some(res) = res {
            return Ok(serde_json::de::from_slice(&res)?);
        }
        Err(KvDatabaseError::TransactionNotFound)
    }

    fn save_stats(&self, stats: &Stats) -> Result<()> {
        let store = self.0.bucket::<&[u8], Vec<u8>>(Some("stats"))?;
        let ser_stats = serde_json::to_vec(&stats)?;
        store.set(&String::from("stats").as_bytes(), &ser_stats)?;
        self.1.flush()?;

        Ok(())
    }

    fn list_transactions(&self) -> Result<Vec<bitcoin::Txid>> {
        let mut transactions = Vec::new();
        let store = self.0.bucket::<&[u8], Vec<u8>>(Some("transactions"))?;

        for item in store.iter() {
            let item = item?;
            let key = item.key::<&[u8]>()?;
            transactions.push(Txid::from_slice(key).unwrap());
        }
        Ok(transactions)
    }
}

#[cfg(test)]
mod test {
    use core::str::FromStr;

    use bitcoin::address::NetworkUnchecked;
    use bitcoin::consensus::deserialize;
    use bitcoin::hashes::hex::FromHex;
    use bitcoin::hashes::sha256;
    use bitcoin::Address;
    use bitcoin::Transaction;
    use floresta_common::get_spk_hash;

    use super::KvDatabase;
    use crate::AddressCacheDatabase;
    use crate::CachedAddress;
    use crate::CachedTransaction;
    use crate::Stats;

    fn get_test_db() -> KvDatabase {
        let test_id = rand::random::<u32>();
        let kv_database = KvDatabase::new(format!("./data/{test_id}.floresta/")).unwrap();
        kv_database
    }
    fn get_test_address() -> (Address<NetworkUnchecked>, sha256::Hash) {
        let address = Address::from_str("tb1q9d4zjf92nvd3zhg6cvyckzaqumk4zre26x02q9").unwrap();
        let script_hash = get_spk_hash(&address.payload().script_pubkey());
        (address, script_hash)
    }
    #[test]
    fn test_kv_database() {
        let db = get_test_db();
        let (_, script_hash) = get_test_address();

        let cache_address = CachedAddress {
            script_hash: script_hash,
            balance: 0,
            transactions: Vec::new(),
            utxos: Vec::new(),
        };
        let test_height: u32 = rand::random();
        let desc = "wsh(sortedmulti(1,[54ff5a12/48h/1h/0h/2h]tpubDDw6pwZA3hYxcSN32q7a5ynsKmWr4BbkBNHydHPKkM4BZwUfiK7tQ26h7USm8kA1E2FvCy7f7Er7QXKF8RNptATywydARtzgrxuPDwyYv4x/<0;1>/*,[bcf969c0/48h/1h/0h/2h]tpubDEFdgZdCPgQBTNtGj4h6AehK79Jm4LH54JrYBJjAtHMLEAth7LuY87awx9ZMiCURFzFWhxToRJK6xp39aqeJWrG5nuW3eBnXeMJcvDeDxfp/<0;1>/*))#fuw35j0q";

        let transaction = "020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100";
        let transaction = Vec::from_hex(transaction).unwrap();
        let transaction: Transaction = deserialize(&transaction).unwrap();

        let merkle_block = "0100000000000000ea530307089e3e6f6e8997a0ae48e1dc2bee84635bc4e6c6ecdcc7225166b06b010000000000000034086ef398efcdec47b37241221c8f4613e02bc31026cc74d07ddb3092e6d6e7";
        let merkle_block = Vec::from_hex(merkle_block).unwrap();
        let merkle_block = deserialize(&merkle_block).unwrap();

        let cache_tx = CachedTransaction {
            tx: transaction.clone(),
            height: 118511,
            merkle_block: Some(merkle_block),
            hash: transaction.txid(),
            position: 1,
        };

        let stats = Stats {
            address_count: 11,
            transaction_count: 11,
            utxo_count: 11,
            cache_height: 11,
            txo_count: 11,
            balance: 11,
            derivation_index: 11,
        };

        db.save_stats(&stats).unwrap();
        assert_eq!(db.get_stats().unwrap().address_count, 11);

        db.save_transaction(&cache_tx).unwrap();
        assert_eq!(db.get_transaction(&cache_tx.hash).unwrap(), cache_tx);
        assert_eq!(db.list_transactions().unwrap(), vec![cache_tx.hash]);

        db.set_cache_height(test_height).unwrap();
        assert_eq!(db.get_cache_height().unwrap(), test_height);

        db.desc_save(desc).unwrap();
        assert_eq!(db.descs_get().unwrap(), vec![desc]);

        db.update(&cache_address);
        assert_eq!(db.load().unwrap()[0].script_hash, cache_address.script_hash);
    }
}
