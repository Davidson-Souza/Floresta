use std::fmt::Debug;

use bitcoin::block::Header as BlockHeader;
use bitcoin::BlockHash;
use bitcoin::Txid;
use serde_json::Number;
use serde_json::Value;

use crate::rpc_types;
use crate::rpc_types::*;

type Result<T> = std::result::Result<T, rpc_types::Error>;

/// A trait specifying all possible methods for floresta's json-rpc
pub trait FlorestaRPC {
    /// Get the BIP158 filter for a given block height
    ///
    /// BIP158 filters are a compact representation of the set of transactions in a block,
    /// designed for efficient light client synchronization. This method returns the filter
    /// for a given block height, encoded as a hexadecimal string.
    /// You need to have enabled block filters by setting the `blockfilters=1` option
    fn get_block_filter(&self, heigth: u32) -> Result<String>;
    /// Returns general information about the chain we are on
    ///
    /// This method returns a bunch of information about the chain we are on, including
    /// the current height, the best block hash, the difficulty, and whether we are
    /// currently in IBD (Initial Block Download) mode.
    fn get_blockchain_info(&self) -> Result<GetBlockchainInfoRes>;
    /// Returns the hash of the block at the given height
    ///
    /// This method returns the hash of the block at the given height. If the height is
    /// invalid, an error is returned.
    fn get_block_hash(&self, height: u32) -> Result<BlockHash>;
    /// Returns the block header for the given block hash
    ///
    /// This method returns the block header for the given block hash, as defined
    /// in the Bitcoin protocol specification. A header contains the block's version,
    /// the previous block hash, the merkle root, the timestamp, the difficulty target,
    /// and the nonce.
    fn get_block_header(&self, hash: BlockHash) -> Result<BlockHeader>;
    /// Gets a transaction from the blockchain
    ///
    /// This method returns a transaction that's cached in our wallet. If the verbosity flag is
    /// set to false, the transaction is returned as a hexadecimal string. If the verbosity
    /// flag is set to true, the transaction is returned as a json object.
    fn get_transaction(&self, tx_id: Txid, verbosity: Option<bool>) -> Result<Value>;
    /// Returns the proof that one or more transactions were included in a block
    ///
    /// This method returns the Merkle proof, showing that a transaction was included in a block.
    /// The pooof is returned as a vector hexadecimal string.
    fn get_tx_proof(&self, tx_id: Txid) -> Result<Vec<String>>;
    /// Loads up a descriptor into the wallet
    ///
    /// This method loads up a descriptor into the wallet. If the rescan option is not None,
    /// the wallet will be rescanned for transactions matching the descriptor. If you have
    /// compact block filters enabled, this process will be much faster and use less bandwidth.
    /// The rescan parameter is the height at which to start the rescan, and should be at least
    /// as old as the oldest transaction this descriptor could have been used in.
    fn load_descriptor(&self, descriptor: String, rescan: Option<u32>) -> Result<bool>;
    /// Trigger a rescan of the wallet
    ///
    /// This method triggers a rescan of the wallet. If you have compact block filters enabled,
    /// this process will be much faster and use less bandwidth. If you don't have compact block
    /// filters, we'll need to download the entire blockchain again, which will take a while.
    /// The rescan parameter is the height at which to start the rescan, and should be at least
    /// as old as the oldest transaction this descriptor could have been used in.
    fn rescan(&self, rescan: u32) -> Result<bool>;
    /// Returns the current height of the blockchain
    fn get_height(&self) -> Result<u32>;
    /// Sends a hex-encoded transaction to the network
    ///
    /// This method sends a transaction to the network. The transaction should be encoded as a
    /// hexadecimal string. If the transaction is valid, it will be broadcast to the network, and
    /// return the transaction id. If the transaction is invalid, an error will be returned.
    fn send_raw_transaction(&self, tx: String) -> Result<Txid>;
    /// Gets the current accumulator for the chain we're on
    ///
    /// This method returns the current accumulator for the chain we're on. The accumulator is
    /// a set of roots, that let's us prove that a UTXO exists in the chain. This method returns
    /// a vector of hexadecimal strings, each of which is a root in the accumulator.
    fn get_roots(&self) -> Result<Vec<String>>;
    /// Gets information about the peers we're connected with
    ///
    /// This method returns information about the peers we're connected with. This includes
    /// the peer's IP address, the peer's version, the peer's user agent, and the peer's
    /// current height.
    fn get_peer_info(&self) -> Result<Vec<PeerInfo>>;
    /// Returns a block, given a block hash
    ///
    /// This method returns a block, given a block hash. If the verbosity flag is 0, the block
    /// is returned as a hexadecimal string. If the verbosity flag is 1, the block is returned
    /// as a json object.
    fn get_block(&self, hash: BlockHash) -> Result<GetBlockRes>;
    /// Finds an specific utxo in the chain
    ///
    /// You can use this to look for a utxo. If it exists, it will return the amount and
    /// scriptPubKey of this utxo. It returns an empty object if the utxo doesn't exist.
    /// You must have enabled block filters by setting the `blockfilters=1` option.
    fn get_tx_out(&self, tx_id: Txid, outpoint: u32) -> Result<Value>;
    /// Stops the florestad process
    ///
    /// This can be used to gracefully stop the florestad process.
    fn stop(&self) -> Result<bool>;
    /// Tells florestad to connect with a peer
    ///
    /// You can use this to connect with a given node, providing it's IP address and port.
    fn add_node(&self, node: String) -> Result<bool>;
}

/// Since the workflow for jsonrpc is the same for all methods, we can implement a trait
/// that will let us call any method on the client, and then implement the methods on any
/// client that implements this trait.
pub trait JsonRPCClient: Sized {
    /// Calls a method on the client
    ///
    /// This should call the appropriated rpc method and return a parsed response or error.
    fn call<T>(&self, method: &str, params: &[Value]) -> Result<T>
    where
        T: for<'a> serde::de::Deserialize<'a> + serde::de::DeserializeOwned + Debug;
}

impl<T: JsonRPCClient> FlorestaRPC for T {
    fn add_node(&self, node: String) -> Result<bool> {
        self.call("addnode", &[Value::String(node)])
    }

    fn stop(&self) -> Result<bool> {
        self.call("stop", &[])
    }

    fn rescan(&self, rescan: u32) -> Result<bool> {
        self.call("rescan", &[Value::Number(Number::from(rescan))])
    }

    fn get_roots(&self) -> Result<Vec<String>> {
        self.call("getroots", &[])
    }

    fn get_block(&self, hash: BlockHash) -> Result<GetBlockRes> {
        let verbosity = 1; // Return the block in json format
        self.call(
            "getblock",
            &[
                Value::String(hash.to_string()),
                Value::Number(Number::from(verbosity)),
            ],
        )
    }

    fn get_height(&self) -> Result<u32> {
        self.call("getheight", &[])
    }

    fn get_tx_out(&self, tx_id: Txid, outpoint: u32) -> Result<Value> {
        self.call(
            "gettxout",
            &[
                Value::String(tx_id.to_string()),
                Value::Number(Number::from(outpoint)),
            ],
        )
    }

    fn get_tx_proof(&self, tx_id: Txid) -> Result<Vec<String>> {
        self.call("gettxoutproof", &[Value::String(tx_id.to_string())])
    }

    fn get_peer_info(&self) -> Result<Vec<PeerInfo>> {
        self.call("getpeerinfo", &[])
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash> {
        self.call("getblockhash", &[Value::Number(Number::from(height))])
    }

    fn get_transaction(&self, tx_id: Txid, verbosity: Option<bool>) -> Result<Value> {
        let verbosity = verbosity.unwrap_or(false);
        self.call(
            "getrawtransaction",
            &[Value::String(tx_id.to_string()), Value::Bool(verbosity)],
        )
    }

    fn load_descriptor(&self, descriptor: String, rescan: Option<u32>) -> Result<bool> {
        let rescan = rescan.unwrap_or(0);
        self.call(
            "loaddescriptor",
            &[
                Value::String(descriptor),
                Value::Number(Number::from(rescan)),
            ],
        )
    }

    fn get_block_filter(&self, heigth: u32) -> Result<String> {
        self.call("getblockfilter", &[Value::Number(Number::from(heigth))])
    }

    fn get_block_header(&self, hash: BlockHash) -> Result<BlockHeader> {
        self.call("getblockheader", &[Value::String(hash.to_string())])
    }

    fn get_blockchain_info(&self) -> Result<GetBlockchainInfoRes> {
        self.call("getblockchaininfo", &[])
    }

    fn send_raw_transaction(&self, tx: String) -> Result<Txid> {
        self.call("sendrawtransaction", &[Value::String(tx)])
    }
}
