#![cfg_attr(feature = "no-std", no_std)]
use core::cmp::Ordering;
use core::fmt::Debug;

use bitcoin::hashes::sha256;
use bitcoin::ScriptBuf;
use floresta_common::get_spk_hash;
use floresta_common::parse_descriptors;
use floresta_common::prelude::ToString;

pub mod kv_database;
#[cfg(any(test, feature = "memory-database"))]
pub mod memory_database;
pub mod merkle;

use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::hash_types::Txid;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hashes::sha256::Hash;
use bitcoin::hashes::Hash as HashTrait;
use bitcoin::Block;
use bitcoin::OutPoint;
use bitcoin::Transaction;
use bitcoin::TxOut;
use floresta_common::prelude::*;
use merkle::MerkleProof;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug)]
pub enum WatchOnlyError<DatabaseError: fmt::Debug> {
    WalletNotInitialized,
    TransactionNotFound,
    DatabaseError(DatabaseError),
}
impl<DatabaseError: fmt::Debug> Display for WatchOnlyError<DatabaseError> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatchOnlyError::WalletNotInitialized => {
                write!(f, "Wallet isn't initialized")
            }
            WatchOnlyError::TransactionNotFound => {
                write!(f, "Transaction not found")
            }
            WatchOnlyError::DatabaseError(e) => {
                write!(f, "Database error: {:?}", e)
            }
        }
    }
}
impl<DatabaseError: fmt::Debug> From<DatabaseError> for WatchOnlyError<DatabaseError> {
    fn from(e: DatabaseError) -> Self {
        WatchOnlyError::DatabaseError(e)
    }
}
impl<T: Debug> floresta_common::prelude::Error for WatchOnlyError<T> {}

/// Every address contains zero or more associated transactions, this struct defines what
/// data we store for those.
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct CachedTransaction {
    pub tx: Transaction,
    pub height: u32,
    pub merkle_block: Option<MerkleProof>,
    pub hash: Txid,
    pub position: u32,
}

impl Ord for CachedTransaction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.height.cmp(&other.height)
    }
}

impl PartialOrd for CachedTransaction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CachedTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
    }
}
impl Default for CachedTransaction {
    fn default() -> Self {
        CachedTransaction {
            // A placeholder transaction with no input and no outputs, the bare-minimum to be
            // serializable
            tx: deserialize(&Vec::from_hex("010000000000ffffffff").unwrap()).unwrap(),
            height: 0,
            merkle_block: None,
            hash: Txid::all_zeros(),
            position: 0,
        }
    }
}

/// An address inside our cache, contains all information we need to satisfy electrum's requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAddress {
    script_hash: Hash,
    balance: u64,
    script: ScriptBuf,
    transactions: Vec<Txid>,
    utxos: Vec<OutPoint>,
}

/// Holds some useful data about our wallet, like how many addresses we have, how many
/// transactions we have, etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stats {
    pub address_count: usize,
    pub transaction_count: usize,
    pub utxo_count: usize,
    pub cache_height: u32,
    pub txo_count: usize,
    pub balance: u64,
    pub derivation_index: u32,
}
/// Public trait defining a common interface for databases to be used with our cache
pub trait AddressCacheDatabase {
    type Error: fmt::Debug + Send + Sync + 'static;
    /// Saves a new address to the database. If the address already exists, `update` should
    /// be used instead
    fn save(&self, address: &CachedAddress);
    /// Loads all addresses we have cached so far
    fn load(&self) -> Result<Vec<CachedAddress>, Self::Error>;
    /// Loads the data associated with our watch-only wallet.
    fn get_stats(&self) -> Result<Stats, Self::Error>;
    /// Saves the data associated with our watch-only wallet.
    fn save_stats(&self, stats: &Stats) -> Result<(), Self::Error>;
    /// Updates an address, probably because a new transaction arrived
    fn update(&self, address: &CachedAddress);
    /// TODO: Maybe turn this into another db
    /// Returns the height of the last block we filtered
    fn get_cache_height(&self) -> Result<u32, Self::Error>;
    /// Saves the height of the last block we filtered
    fn set_cache_height(&self, height: u32) -> Result<(), Self::Error>;
    /// Saves the descriptor of associated cache
    fn desc_save(&self, descriptor: &str) -> Result<(), Self::Error>;
    /// Get associated descriptors
    fn descs_get(&self) -> Result<Vec<String>, Self::Error>;
    /// Get a transaction from the database
    fn get_transaction(&self, txid: &Txid) -> Result<CachedTransaction, Self::Error>;
    /// Saves a transaction to the database
    fn save_transaction(&self, tx: &CachedTransaction) -> Result<(), Self::Error>;
    /// Returns all transaction we have cached so far
    fn list_transactions(&self) -> Result<Vec<Txid>, Self::Error>;
}
/// Holds all addresses and associated transactions. We need a database with some basic
/// methods, to store all data
pub struct AddressCache<D: AddressCacheDatabase> {
    /// A database that will be used to persist all needed to get our address history
    database: D,
    /// Maps a hash to a cached address struct, this is basically an in-memory version
    /// of our database, used for speeding up processing a block. This hash is the electrum's
    /// script hash
    address_map: HashMap<Hash, CachedAddress>,
    /// Holds all scripts we are interested in
    script_set: HashSet<sha256::Hash>,
    /// Keeps track of all utxos we own, and the script hash they belong to
    utxo_index: HashMap<OutPoint, Hash>,
}

impl<D: AddressCacheDatabase> AddressCache<D> {
    /// Iterates through a block, finds transactions destined to ourselves.
    /// Returns all transactions we found.
    pub fn block_process(&mut self, block: &Block, height: u32) -> Vec<(Transaction, TxOut)> {
        let mut my_transactions = Vec::new();
        // Check if this transaction spends from one of our utxos
        for (position, transaction) in block.txdata.iter().enumerate() {
            for (vin, txin) in transaction.input.iter().enumerate() {
                if let Some(script) = self.utxo_index.get(&txin.previous_output) {
                    let script = self
                        .address_map
                        .get(script)
                        .expect("Can't cache a utxo for a address we don't have")
                        .to_owned();
                    let tx = self
                        .get_transaction(&txin.previous_output.txid)
                        .expect("We cached a utxo for a transaction we don't have");

                    let utxo = tx
                        .tx
                        .output
                        .get(txin.previous_output.vout as usize)
                        .expect("Did we cache an invalid utxo?");
                    let merkle_block = MerkleProof::from_block(block, position as u64);

                    self.cache_transaction(
                        transaction,
                        height,
                        utxo.value.to_sat(),
                        merkle_block,
                        position as u32,
                        vin,
                        true,
                        script.script_hash,
                    )
                }
            }
            // Checks if one of our addresses is the recipient of this transaction
            for (vout, output) in transaction.output.iter().enumerate() {
                let hash = get_spk_hash(&output.script_pubkey);
                if self.script_set.contains(&hash) {
                    my_transactions.push((transaction.clone(), output.clone()));

                    let merkle_block = MerkleProof::from_block(block, position as u64);

                    self.cache_transaction(
                        transaction,
                        height,
                        output.value.to_sat(),
                        merkle_block,
                        position as u32,
                        vout,
                        false,
                        hash,
                    );
                }
            }
        }
        my_transactions
    }

    pub fn get_cached_addresses(&self) -> Vec<ScriptBuf> {
        self.address_map
            .values()
            .map(|address| address.script.clone())
            .collect()
    }

    fn get_stats(&self) -> Stats {
        self.database
            .get_stats()
            .expect("Could not get stats from database")
    }

    pub fn bump_height(&self, height: u32) {
        self.database
            .set_cache_height(height)
            .expect("Database is not working");
    }

    pub fn get_cache_height(&self) -> u32 {
        self.database.get_cache_height().unwrap_or(0)
    }

    pub fn new(database: D) -> AddressCache<D> {
        let scripts = database.load().expect("Could not load database");
        if database.get_stats().is_err() {
            database
                .save_stats(&Stats::default())
                .expect("Could not save stats");
        }
        let mut address_map = HashMap::new();
        let mut script_set = HashSet::new();
        let mut utxo_index = HashMap::new();
        for address in scripts {
            for utxo in address.utxos.iter() {
                utxo_index.insert(*utxo, address.script_hash);
            }
            script_set.insert(address.script_hash);
            address_map.insert(address.script_hash, address);
        }
        AddressCache {
            database,
            address_map,
            script_set,
            utxo_index,
        }
    }
    pub fn get_address_utxos(&self, script_hash: &Hash) -> Option<Vec<(TxOut, OutPoint)>> {
        let address = self.address_map.get(script_hash)?;
        let utxos = &address.utxos;
        let mut address_utxos = Vec::new();
        for utxo in utxos {
            let tx = self.get_transaction(&utxo.txid)?;
            let txout = tx.tx.output.get(utxo.vout as usize)?;
            address_utxos.push((txout.clone(), *utxo));
        }

        Some(address_utxos)
    }
    pub fn get_transaction(&self, txid: &Txid) -> Option<CachedTransaction> {
        self.database.get_transaction(txid).ok()
    }
    /// Returns all transactions this address has, both input and outputs
    pub fn get_address_history(&self, script_hash: &Hash) -> Option<Vec<CachedTransaction>> {
        let cached_script = self.address_map.get(script_hash)?;
        let mut transactions: Vec<_> = cached_script
            .transactions
            .iter()
            .filter_map(|txid| self.get_transaction(txid))
            .collect();
        let mut unconfirmed = transactions.clone();

        transactions.retain(|tx| tx.height != 0);
        transactions.sort();
        unconfirmed.retain(|tx| tx.height == 0);
        transactions.extend(unconfirmed);
        Some(transactions)
    }
    /// Returns the balance of this address, debts (spends) are taken in account
    pub fn get_address_balance(&self, script_hash: &Hash) -> u64 {
        if let Some(cached_script) = self.address_map.get(script_hash) {
            return cached_script.balance;
        }

        0
    }
    /// Returns the Merkle Proof for a given address
    pub fn get_merkle_proof(&self, txid: &Txid) -> Option<(Vec<String>, u32)> {
        let mut hashes = Vec::new();
        let tx = self.get_transaction(txid)?;
        // If a given transaction is cached, but the merkle tree doesn't exist, that means
        // an unconfirmed transaction.
        tx.merkle_block.as_ref()?;
        for hash in tx.merkle_block?.hashes() {
            hashes.push(hash.to_string());
        }
        Some((hashes, tx.position))
    }
    pub fn get_position(&self, txid: &Txid) -> Option<u32> {
        Some(self.get_transaction(txid)?.position)
    }
    pub fn get_height(&self, txid: &Txid) -> Option<u32> {
        Some(self.get_transaction(txid)?.height)
    }
    pub fn get_cached_transaction(&self, txid: &Txid) -> Option<String> {
        let tx = self.get_transaction(txid)?;
        Some(serialize_hex(&tx.tx))
    }
    /// Adds a new address to track, should be called at wallet setup and every once in a while
    /// to cache new addresses, as we use the first ones. Only requires a script to cache.
    pub fn cache_address(&mut self, script_pk: ScriptBuf) {
        let hash = get_spk_hash(&script_pk);
        if self.address_map.contains_key(&hash) {
            return;
        }
        let new_address = CachedAddress {
            balance: 0,
            script: script_pk,
            script_hash: hash,
            transactions: Vec::new(),
            utxos: Vec::new(),
        };
        self.database.save(&new_address);

        self.address_map.insert(hash, new_address);
        self.script_set.insert(hash);
    }

    /// Setup is the first command that should be executed. In a new cache. It sets our wallet's
    /// state, like the height we should start scanning and the wallet's descriptor.
    pub fn setup(&self) -> Result<(), WatchOnlyError<D::Error>> {
        if self.database.descs_get().is_err() {
            self.database.set_cache_height(0)?;
        }
        Ok(())
    }
    /// Tells whether or not a descriptor is already cached
    pub fn is_cached(&self, desc: &String) -> Result<bool, WatchOnlyError<D::Error>> {
        let known_descs = self.database.descs_get()?;
        Ok(known_descs.contains(desc))
    }
    /// Tells wheter an address is already cached
    pub fn is_address_cached(&self, script_hash: &Hash) -> bool {
        self.address_map.contains_key(script_hash)
    }
    pub fn push_descriptor(&self, descriptor: &str) -> Result<(), WatchOnlyError<D::Error>> {
        Ok(self.database.desc_save(descriptor)?)
    }
    fn derive_addresses(&mut self) -> Result<(), WatchOnlyError<D::Error>> {
        let mut stats = self.get_stats();
        let descriptors = self.database.descs_get()?;
        let descriptors = parse_descriptors(&descriptors).expect("We validate those descriptors");
        for desc in descriptors {
            let index = stats.derivation_index;
            for idx in index..(index + 100) {
                let script = desc
                    .at_derivation_index(idx)
                    .expect("We validate those descriptors before saving")
                    .script_pubkey();
                self.cache_address(script);
            }
        }
        stats.derivation_index += 100;
        Ok(self.database.save_stats(&stats)?)
    }
    fn maybe_derive_addresses(&mut self) {
        let stats = self.get_stats();
        if stats.transaction_count > (stats.derivation_index as usize * 100) {
            let res = self.derive_addresses();
            if res.is_err() {
                log::error!("Error deriving addresses: {:?}", res);
            }
        }
    }
    pub fn find_unconfirmed(&self) -> Result<Vec<Transaction>, WatchOnlyError<D::Error>> {
        let transactions = self.database.list_transactions()?;
        let mut unconfirmed = Vec::new();

        for tx in transactions {
            let tx = self.database.get_transaction(&tx)?;
            if tx.height == 0 {
                unconfirmed.push(tx.tx);
            }
        }
        Ok(unconfirmed)
    }
    fn find_spend(&self, transaction: &Transaction) -> Vec<(usize, TxOut)> {
        let mut spends = Vec::new();
        for (idx, input) in transaction.input.iter().enumerate() {
            if self.utxo_index.contains_key(&input.previous_output) {
                let prev_tx = self.get_transaction(&input.previous_output.txid).unwrap();
                spends.push((
                    idx,
                    prev_tx.tx.output[input.previous_output.vout as usize].clone(),
                ));
            }
        }
        spends
    }
    pub fn cache_mempool_transaction(&mut self, transaction: &Transaction) -> Vec<TxOut> {
        let mut coins = self.find_spend(transaction);
        for (idx, spend) in coins.iter() {
            let script = self
                .address_map
                .get(&get_spk_hash(&spend.script_pubkey))
                .unwrap()
                .to_owned();
            self.cache_transaction(
                transaction,
                0,
                spend.value.to_sat(),
                MerkleProof::default(),
                0,
                *idx,
                true,
                script.script_hash,
            )
        }
        for (idx, out) in transaction.output.iter().enumerate() {
            let spk_hash = get_spk_hash(&out.script_pubkey);
            if self.script_set.contains(&spk_hash) {
                let script = self.address_map.get(&spk_hash).unwrap().to_owned();
                coins.push((idx, out.clone()));
                self.cache_transaction(
                    transaction,
                    0,
                    out.value.to_sat(),
                    MerkleProof::default(),
                    0,
                    idx,
                    true,
                    script.script_hash,
                )
            }
        }
        coins
            .iter()
            .cloned()
            .unzip::<usize, TxOut, Vec<usize>, Vec<TxOut>>()
            .1
    }
    fn save_mempool_tx(&mut self, hash: Hash, transaction_to_cache: CachedTransaction) {
        if let Some(address) = self.address_map.get_mut(&hash) {
            if !address.transactions.contains(&transaction_to_cache.hash) {
                address.transactions.push(transaction_to_cache.hash);
                self.database.update(address);
            }
        }
    }
    fn save_non_mempool_tx(
        &mut self,
        transaction: &Transaction,
        is_spend: bool,
        value: u64,
        index: usize,
        hash: Hash,
        transaction_to_cache: CachedTransaction,
    ) {
        if let Some(address) = self.address_map.get_mut(&hash) {
            // This transaction is spending from this address, so we should remove the UTXO
            if is_spend {
                assert!(value <= address.balance);
                address.balance -= value;
                let input = transaction
                    .input
                    .get(index)
                    .expect("Malformed call, index is bigger than the output vector");
                let idx = address
                    .utxos
                    .iter()
                    .position(|utxo| *utxo == input.previous_output);
                if let Some(idx) = idx {
                    let utxo = address.utxos.remove(idx);
                    self.utxo_index.remove(&utxo);
                }
            } else {
                // This transaction is creating a new utxo for this address
                let utxo = OutPoint {
                    txid: transaction.txid(),
                    vout: index as u32,
                };
                address.utxos.push(utxo);
                self.utxo_index.insert(utxo, hash);
                address.balance += value;
            }

            if !address.transactions.contains(&transaction_to_cache.hash) {
                address.transactions.push(transaction_to_cache.hash);
                self.database.update(address);
            }
        }
    }
    /// Caches a new transaction. This method may be called for addresses we don't follow yet,
    /// this automatically makes we follow this address.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_transaction(
        &mut self,
        transaction: &Transaction,
        height: u32,
        value: u64,
        merkle_block: MerkleProof,
        position: u32,
        index: usize,
        is_spend: bool,
        hash: sha256::Hash,
    ) {
        let transaction_to_cache = CachedTransaction {
            height,
            merkle_block: Some(merkle_block),
            tx: transaction.clone(),
            hash: transaction.txid(),
            position,
        };
        self.database
            .save_transaction(&transaction_to_cache)
            .expect("Database not working");

        if let Entry::Vacant(e) = self.address_map.entry(hash) {
            let script = transaction.output[index].script_pubkey.clone();
            // This means `cache_transaction` have been called with an address we don't
            // follow. This may be useful for caching new addresses without re-scanning.
            // We can track this address from now onwards, but the past history is only
            // available with full rescan
            let new_address = CachedAddress {
                balance: 0,
                script,
                script_hash: hash,
                transactions: Vec::new(),
                utxos: Vec::new(),
            };
            self.database.save(&new_address);

            e.insert(new_address);
            self.script_set.insert(hash);
        }
        self.maybe_derive_addresses();
        // Confirmed transaction
        if height > 0 {
            return self.save_non_mempool_tx(
                transaction,
                is_spend,
                value,
                index,
                hash,
                transaction_to_cache,
            );
        }
        // Unconfirmed transaction
        self.save_mempool_tx(hash, transaction_to_cache);
    }
}

#[cfg(test)]
mod test {
    use bitcoin::address::NetworkUnchecked;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::Decodable;
    use bitcoin::hashes::hex::FromHex;
    use bitcoin::hashes::sha256;
    use bitcoin::Address;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Txid;
    use floresta_common::get_spk_hash;
    use floresta_common::prelude::*;

    const BLOCK_FIRST_UTXO: &str = "00000020b4f594a390823c53557c5a449fa12413cbbae02be529c11c4eb320ff8e000000dd1211eb35ca09dc0ee519b0f79319fae6ed32c66f8bbf353c38513e2132c435474d81633c4b011e195a220002010000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff0403edce01feffffff028df2052a0100000016001481113cad52683679a83e76f76f84a4cfe36f75010000000000000000776a24aa21a9ed67863b4f356b7b9f3aab7a2037615989ef844a0917fb0a1dcd6c23a383ee346b4c4fecc7daa2490047304402203768ff10a948a2dd1825cc5a3b0d336d819ea68b5711add1390b290bf3b1cba202201d15e73791b2df4c0904fc3f7c7b2f22ab77762958e9bc76c625138ad3a04d290100012000000000000000000000000000000000000000000000000000000000000000000000000002000000000101be07b18750559a418d144f1530be380aa5f28a68a0269d6b2d0e6ff3ff25f3200000000000feffffff0240420f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a326f55d94c060000160014c2ed86a626ee74d854a12c9bb6a9b72a80c0ddc50247304402204c47f6783800831bd2c75f44d8430bf4d962175349dc04d690a617de6c1eaed502200ffe70188a6e5ad89871b2acb4d0f732c2256c7ed641d2934c6e84069c792abc012103ba174d9c66078cf813d0ac54f5b19b5fe75104596bdd6c1731d9436ad8776f41ecce0100";
    const BLOCK_SPEND: &str = "000000203ea734fa2c8dee7d3194878c9eaf6e83a629f79b3076ec857793995e01010000eb99c679c0305a1ac0f5eb2a07a9f080616105e605b92b8c06129a2451899225ab5481633c4b011e0b26720102020000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff0403efce01feffffff026ef2052a01000000225120a1a1b1376d5165617a50a6d2f59abc984ead8a92df2b25f94b53dbc2151824730000000000000000776a24aa21a9ed1b4c48a7220572ff3ab3d2d1c9231854cb62542fbb1e0a4b21ebbbcde8d652bc4c4fecc7daa2490047304402204b37c41fce11918df010cea4151737868111575df07f7f2945d372e32a6d11dd02201658873a8228d7982df6bdbfff5d0cad1d6f07ee400e2179e8eaad8d115b7ed001000120000000000000000000000000000000000000000000000000000000000000000000000000020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100";
    fn deserialize_from_str<T: Decodable>(thing: &str) -> T {
        let hex = Vec::from_hex(thing).unwrap();
        deserialize(&hex).unwrap()
    }
    use super::memory_database::MemoryDatabase;
    use super::AddressCache;
    use crate::merkle::MerkleProof;
    fn get_test_cache() -> AddressCache<MemoryDatabase> {
        let database = MemoryDatabase::new();
        AddressCache::new(database)
    }
    fn get_test_address() -> (Address<NetworkUnchecked>, sha256::Hash) {
        let address = Address::from_str("tb1q9d4zjf92nvd3zhg6cvyckzaqumk4zre26x02q9").unwrap();
        let script_hash = get_spk_hash(&address.payload().script_pubkey());
        (address, script_hash)
    }
    #[test]
    fn test_create() {
        let _ = get_test_cache();
    }
    #[test]
    fn test_cache_address() {
        let (address, script_hash) = get_test_address();
        let mut cache = get_test_cache();
        // Should have no address before caching
        assert_eq!(cache.address_map.len(), 0);

        cache.cache_address(address.payload().script_pubkey());
        // Assert we indeed have one cached address
        assert_eq!(cache.address_map.len(), 1);
        assert_eq!(cache.get_address_balance(&script_hash), 0);
        assert_eq!(cache.get_address_history(&script_hash), Some(Vec::new()));
    }

    #[test]
    fn test_cache_transaction() {
        // Signet transaction with id 6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea
        // block hash 0000009298f9e75a91fa763c78b66d1555cb059d9ca9d45601eed2b95166a151.
        let transaction = "020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100";
        let transaction = Vec::from_hex(transaction).unwrap();
        let transaction = deserialize(&transaction).unwrap();

        let merkle_block = "0100000000000000ea530307089e3e6f6e8997a0ae48e1dc2bee84635bc4e6c6ecdcc7225166b06b010000000000000034086ef398efcdec47b37241221c8f4613e02bc31026cc74d07ddb3092e6d6e7";
        let merkle_block = Vec::from_hex(merkle_block).unwrap();
        let merkle_block = deserialize(&merkle_block).unwrap();

        let (_, script_hash) = get_test_address();
        let mut cache = get_test_cache();

        cache.cache_transaction(
            &transaction,
            118511,
            transaction.output[0].value.to_sat(),
            merkle_block,
            1,
            0,
            false,
            get_spk_hash(&transaction.output[0].script_pubkey),
        );

        assert_eq!(
            script_hash,
            get_spk_hash(&transaction.output[0].script_pubkey)
        );

        let balance = cache.get_address_balance(&script_hash);
        let history = cache.get_address_history(&script_hash).unwrap();
        let cached_merkle_block = cache.get_merkle_proof(&transaction.txid()).unwrap();
        assert_eq!(balance, 999890);
        assert_eq!(
            Ok(history[0].hash),
            Txid::from_str("6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea")
        );
        let expected_hashes = Vec::from([String::from(
            "e7d6e69230db7dd074cc2610c32be013468f1c224172b347eccdef98f36e0834",
        )]);
        assert_eq!(cached_merkle_block, (expected_hashes, 1));

        // TESTS FOR SMALL, HELPER FUNCTIONS

        // [get_position]
        assert_eq!(cache.get_position(&transaction.txid()).unwrap(), 1);

        // [get_height]
        assert_eq!(cache.get_height(&transaction.txid()).unwrap(), 118511);

        // [get_cached_transaction]
        assert!(cache.get_cached_transaction(&transaction.txid()).is_some());

        // [get_address_utxos]
        let tx_out = transaction.output[0].clone();
        let outpoint = OutPoint {
            txid: transaction.txid(),
            vout: 0,
        };
        assert_eq!(
            cache.get_address_utxos(&script_hash).unwrap(),
            vec![(tx_out, outpoint)]
        );

        // [find_unconfirmed] Caching am unconfirmed transaction
        let transaction = "01000000010b7e3ac7e68944dc7a7115362391c3b7975d60f4fbe4af0ca924a172bfe7a7d9000000006b483045022100e0ff6984e5c2e16df6f309b759b75e04adf6930593b6043cd9134f87efb7e07c02206544a9f265f6041f0e3e2bd11a95ea75a112d3dc05647a9b01eca0d352feeb380121024f9c3deb05e81a3ddb17dadcf283fb132894aa70ab127395a03a3e9d382f13a3ffffffff022c92ae00000000001976a914ca9755ffb8f0e5aeca43478d8620e1a35b3baada88acc0894601000000001976a914b62ad08a3ffc469e9c0df75d1ceca49a88345fc888ac00000000";
        let transaction = Vec::from_hex(transaction).unwrap();
        let transaction = deserialize(&transaction).unwrap();

        cache.cache_transaction(
            &transaction,
            0,
            transaction.output[1].value.to_sat(),
            MerkleProof::default(),
            2,
            1,
            false,
            get_spk_hash(&transaction.output[1].script_pubkey),
        );

        assert_eq!(
            cache.find_unconfirmed().unwrap()[0].txid(),
            transaction.txid()
        );
    }
    #[test]
    fn test_process_block() {
        let (address, script_hash) = get_test_address();
        let mut cache = get_test_cache();
        cache.cache_address(address.payload().script_pubkey());

        let block = "000000203ea734fa2c8dee7d3194878c9eaf6e83a629f79b3076ec857793995e01010000eb99c679c0305a1ac0f5eb2a07a9f080616105e605b92b8c06129a2451899225ab5481633c4b011e0b26720102020000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff0403efce01feffffff026ef2052a01000000225120a1a1b1376d5165617a50a6d2f59abc984ead8a92df2b25f94b53dbc2151824730000000000000000776a24aa21a9ed1b4c48a7220572ff3ab3d2d1c9231854cb62542fbb1e0a4b21ebbbcde8d652bc4c4fecc7daa2490047304402204b37c41fce11918df010cea4151737868111575df07f7f2945d372e32a6d11dd02201658873a8228d7982df6bdbfff5d0cad1d6f07ee400e2179e8eaad8d115b7ed001000120000000000000000000000000000000000000000000000000000000000000000000000000020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100";
        let block = deserialize(&Vec::from_hex(block).unwrap()).unwrap();
        cache.block_process(&block, 118511);

        let balance = cache.get_address_balance(&script_hash);
        let history = cache.get_address_history(&script_hash).unwrap();
        let transaction_id =
            Txid::from_str("6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea")
                .unwrap();
        let cached_merkle_block = cache.get_merkle_proof(&transaction_id).unwrap();
        assert_eq!(balance, 999890);
        assert_eq!(
            history[0].hash,
            Txid::from_str("6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea")
                .unwrap()
        );
        let expected_hashes = Vec::from([String::from(
            "e7d6e69230db7dd074cc2610c32be013468f1c224172b347eccdef98f36e0834",
        )]);
        assert_eq!(cached_merkle_block, (expected_hashes, 1));

        // TESTS FOR SMALL HELPER FUNCTIONS

        // [bump_height], [get_cache_height], [set_cache_height]
        cache.bump_height(118511);
        assert_eq!(cache.get_cache_height(), 118511);

        // [is_cached], [push_descriptor]
        let desc = "wsh(sortedmulti(1,[54ff5a12/48h/1h/0h/2h]tpubDDw6pwZA3hYxcSN32q7a5ynsKmWr4BbkBNHydHPKkM4BZwUfiK7tQ26h7USm8kA1E2FvCy7f7Er7QXKF8RNptATywydARtzgrxuPDwyYv4x/<0;1>/*,[bcf969c0/48h/1h/0h/2h]tpubDEFdgZdCPgQBTNtGj4h6AehK79Jm4LH54JrYBJjAtHMLEAth7LuY87awx9ZMiCURFzFWhxToRJK6xp39aqeJWrG5nuW3eBnXeMJcvDeDxfp/<0;1>/*))#fuw35j0q";
        cache.push_descriptor(desc).unwrap();
        assert!(cache.is_cached(&desc.to_string()).unwrap());

        // [derive_addresses]
        cache.derive_addresses().unwrap();
        assert_eq!(cache.get_stats().derivation_index, 100);
    }
    #[test]
    fn test_multiple_transaction() {
        let block1 = deserialize_from_str(BLOCK_FIRST_UTXO);
        let block2 = deserialize_from_str(BLOCK_SPEND);

        let spk = ScriptBuf::from_hex("00142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a")
            .expect("Valid address");
        let script_hash = get_spk_hash(&spk);
        let mut cache = get_test_cache();

        cache.cache_address(spk);

        cache.block_process(&block1, 118511);
        cache.block_process(&block2, 118509);

        let address = cache.address_map.get(&script_hash).unwrap();

        assert_eq!(address.transactions.len(), 2);
        assert_eq!(address.utxos.len(), 1);
    }
}
