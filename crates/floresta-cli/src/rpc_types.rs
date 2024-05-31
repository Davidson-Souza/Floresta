use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Deserialize, Serialize)]
pub struct GetBlockchainInfoRes {
    /// The best block we know about
    ///
    /// This should be the hash of the latest block in the most PoW chain we know about. We may
    /// or may not have fully-validated it yet
    pub best_block: String,
    /// The depth of the most-PoW chain we know about
    pub height: u32,
    /// Whether we are on Initial Block Download
    pub ibd: bool,
    /// How many blocks we have fully-validated so far? This number will be smaller than
    /// height during IBD, and should be equal to height otherwise
    pub validated: u32,
    /// The work performed by the last block
    ///
    /// This is the estimated amount of hashes the miner of this block had to perform
    /// before mining that block, on average
    pub latest_work: String,
    /// The UNIX timestamp for the latest block, as reported by the block's header
    pub latest_block_time: u32,
    /// How many leaves we have in the utreexo accumulator so far
    ///
    /// This should be equal to the number of UTXOs returned by core's `gettxoutsetinfo`
    pub leaf_count: u32,
    /// How many roots we have in the acc
    pub root_count: u32,
    /// The actual hex-encoded roots
    pub root_hashes: Vec<String>,
    /// A short string representing the chain we're in
    pub chain: String,
    /// The validation progress
    ///
    /// 0% means we didn't validate any block. 100% means we've validated all blocks, so
    /// validated == height
    pub progress: Option<f32>,
    /// Current network "difficulty"
    ///
    /// On average, miners needs to make `difficulty` hashes before finding one that
    /// solves a block's PoW
    pub difficulty: u64,
}

/// The information returned by a get_raw_tx
#[derive(Deserialize, Serialize)]
pub struct RawTx {
    /// Whether this tx is in our best known chain
    pub in_active_chain: bool,
    /// The hex-encoded tx
    pub hex: String,
    /// Tha sha256d of the serialized transaction without witness
    pub txid: String,
    /// The sha256d of the serialized transaction including witness
    pub hash: String,
    /// The size this transaction occupies on disk
    pub size: u32,
    /// The virtual size of this transaction, as define by the segwit soft-fork
    pub vsize: u32,
    /// The weight of this transacion, as defined by the segwit soft-fork
    pub weight: u32,
    /// This transaction's version. The current bigger version is 2
    pub version: u32,
    /// This transaction's locktime
    pub locktime: u32,
    /// A list of inputs being spent by this transaction
    ///
    /// See [TxIn] for more information about the contents of this
    pub vin: Vec<TxIn>,
    /// A list of outputs being created by this tx
    ///
    /// Se [TxOut] for more information
    pub vout: Vec<TxOut>,
    /// The hash of the block that included this tx, if any
    pub blockhash: String,
    /// How many blocks have been mined after this transaction's confirmation
    /// including the block that confirms it. A zero value means this tx is unconfirmed
    pub confirmations: u32,
    /// The timestamp for the block confirming this tx, if confirmed
    pub blocktime: u32,
    /// Same as blocktime
    pub time: u32,
}

/// A transaction output returned by some RPCs like getrawtransaction and getblock
#[derive(Deserialize, Serialize)]
pub struct TxOut {
    /// The amount in sats locked in this UTXO
    pub value: u64,
    /// This utxo's index inside the transaction
    pub n: u32,
    /// The loking script of this utxo
    pub script_pub_key: ScriptPubKey,
}

/// The locking script inside a txout
#[derive(Deserialize, Serialize)]
pub struct ScriptPubKey {
    /// A ASM representation for this script
    ///
    /// Assembly is a high-level representation of a lower level code. Instructions
    /// are turned into OP_XXXXX and data is hex-encoded.
    /// E.g: OP_DUP OP_HASH160 <0000000000000000000000000000000000000000> OP_EQUALVERIFY OP_CHECKSIG
    pub asm: String,
    /// The hex-encoded raw script
    pub hex: String,
    /// How many signatures are required to spend this UTXO.
    ///
    /// This field is deprecated and is here for compatibility with Core
    pub req_sigs: u32,
    #[serde(rename = "type")]
    /// The type of this spk. E.g: PKH, SH, WSH, WPKH, TR, non-standard...
    pub type_: String,
    /// Encode this script using one of the standard address types, if possible
    pub address: String,
}

/// A transaction input returned by some rpcs, like getrawtransaction and getblock
#[derive(Deserialize, Serialize)]
pub struct TxIn {
    /// The txid that created this UTXO
    pub txid: String,
    /// The index of this UTXO inside the tx that created it
    pub vout: u32,
    /// Unlocking script that should solve the challenge and prove ownership over
    /// that UTXO
    pub script_sig: ScriptSigJson,
    /// The nSequence field, used in relative and absolute lock-times
    pub sequence: u32,
    /// A vector of witness elements for this input
    pub witness: Vec<String>,
}

/// A representation for the transaction ScriptSig, returned by some rpcs
/// like getrawtransaction and getblock
#[derive(Deserialize, Serialize)]
pub struct ScriptSigJson {
    /// A ASM representation for this scriptSig
    ///
    /// Assembly is a high-level representation of a lower level code. Instructions
    /// are turned into OP_XXXXX and data is hex-encoded.
    /// E.g: OP_PUSHBYTES32 <000000000000000000000000000000000000000000000000000000000000000000>
    pub asm: String,
    /// The hex-encoded script sig
    pub hex: String,
}

/// General information about our peers. Returned by get_peer_info
#[derive(Debug, Deserialize, Serialize)]
pub struct PeerInfo {
    /// The network address for this peer.
    pub address: String,
    /// A string with the services this peer advertises. E.g. NODE_NETWORK, UTREEXO, WITNESS...
    pub services: String,
    /// User agent is a string that represents the client being used by our peer. E.g.
    /// /Satoshi-26.0/ for bitcoin core version 26
    pub user_agent: String,
    /// This peer's height at the time we've openned a connection with them
    pub initial_height: u32,
}

/// A full bitcoin block, returned by get_block
#[derive(Debug, Deserialize, Serialize)]
pub struct GetBlockRes {
    /// This block's hash.
    pub hash: String,
    /// How many blocks have been added to the chain, after this one have been found. This is
    /// inclusive, so it starts with one when this block is the latest. If another one is found,
    /// then it increments to 2 and so on...
    pub confirmations: u32,
    /// The size of this block, without the witness
    pub strippedsize: usize,
    /// This block's size, with the witness
    pub size: usize,
    /// This block's weight.
    ///
    /// Data inside a segwit block is counted differently, 'base data' has a weight of 4, while
    /// witness only counts 1. This is (3 * base_size) + size
    pub weight: usize,
    /// How many blocks there are before this block
    pub height: u32,
    /// This block's version field
    ///
    /// Currently, blocks have version 2 (see BIP34), but it may also flip some of the LSB for
    /// either consensus reason (see BIPs 8 and 9) or for version rolling mining, usually bits
    /// after the 24th are not touched. Therefore, the actual version is likelly the result of
    /// version & ~(1 << 24).
    /// This is encoded as a number, see `version_hex` for a hex-encoded version
    pub version: i32,
    #[serde(rename = "versionHex")]
    /// Same as `version` by hex-encoded
    pub version_hex: String,
    /// This block's merkle root
    ///
    /// A Merkle Tree is a binary tree where every leaf is some data, and the branches are pairwise
    /// hashes util reaching the root. This allows for compact proof of inclusion in the original
    /// set. This merkle tree commits to the txid of all transactions in a block, and is used by
    /// some light clients to determine whether a transaction is in a given block
    pub merkleroot: String,
    /// A list of hex-encoded transaction id for the tx's in this block
    pub tx: Vec<String>,
    /// The timestamp commited to in this block's header
    ///
    /// Since there's no central clock that can tell time precisely in Bitcoin, this value is
    /// reported by miners and only constrained by a couple of consensus rules. More sensibly, it
    /// is **not** garanteed to be monotonical. So a block n might have a lower timestamp than
    /// block `n - 1`.
    /// If you need it to be monotonical, see `mediantime` insted
    pub time: u32,
    /// The meadian of the last 11 blocktimes.
    ///
    /// This is a monotonically increasing number that bounds how old a block can be. Blocks may
    /// not have a timestamp less than the current `mediantime`. This is also used in relative
    /// timelocks.
    pub mediantime: u32,
    /// The nonce used to mine this block.
    ///
    /// Blocks are mined by increasing this value until you find a hash that is less than a network
    /// defined target. This number has no meaning in itself and is just a random u32.
    pub nonce: u32,
    /// Bits is a compact representation for the target.
    ///
    /// This is a exponential format (with well-define rouding) used by openssl that Satoshi
    /// decided to make consensus critical :/
    pub bits: String,
    /// The difficulty is derived from the current target and is defined as how many hashes, on
    /// average, one has to make before finding a valid block
    ///
    /// This is computed as 1 / (target / 2 ^ 256). In most softwares (this one inclued) the
    /// difficulty is a multiple of the smallest possible difficulty. So to find the actual
    /// difficulty you have to multiply this by the min_diff.
    /// For mainnet, mindiff is 2 ^ 32
    pub difficulty: u128,
    /// Commullative work in this network
    ///
    /// This is a estimate of how many hashes the network has ever made to produce this chain
    pub chainwork: String,
    /// How many transactions in this block
    pub n_tx: usize,
    /// The hash of the block comming before this one
    pub previousblockhash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// The hash of the block comming after this one, if any
    pub nextblockhash: Option<String>,
}

#[derive(Error, Debug)]
/// All possible errors returned by the jsonrpc
pub enum Error {
    #[error(transparent)]
    /// An error while deserializing our response
    Serde(#[from] serde_json::Error),
    #[cfg(feature = "with-reqwest")]
    /// An internal reqwest error
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    /// An error internal to our jsonrpc server
    #[error(transparent)]
    Api(#[from] serde_json::Value),
    /// The server sent an empty response
    #[error("Empty response")]
    EmtpyResponse,
}
