// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use std::path::PathBuf;

pub use RpcTypes::AddrManInfoNetwork;
pub use RpcTypes::CoinbaseTransaction;
pub use RpcTypes::DeploymentInfo;
pub use RpcTypes::GetAddrManInfo;
pub use RpcTypes::GetBlockHeaderVerbose;
pub use RpcTypes::GetBlockVerboseOne;
pub use RpcTypes::GetBlockchainInfo;
pub use RpcTypes::GetDeploymentInfo;
pub use RpcTypes::GetNetworkInfo;
pub use RpcTypes::GetNetworkInfoNetwork;
pub use RpcTypes::GetRawTransactionVerbose;
pub use RpcTypes::GetTxOut;
pub use RpcTypes::RawTransactionInput;
pub use RpcTypes::RawTransactionOutput;
pub use corepc_types::ScriptPubKey;
pub use corepc_types::ScriptSig;
pub use corepc_types::v31 as RpcTypes;
use floresta_proc_macro::enum_str_map;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum GetRawTransactionRes {
    Zero(String),

    One(Box<GetRawTransactionVerbose>),
}

/// General information about our peers. Returned by get_peer_info
#[derive(Debug, Deserialize, Serialize)]
pub struct PeerInfo {
    /// This peer's ID in the peer manager.
    pub id: u32,
    /// The network address for this peer.
    pub address: String,
    /// Hex-encoded bitfield with the services this peer advertises.
    pub services: String,
    /// Human-readable names for the recognized services this peer advertises.
    #[serde(rename = "servicesnames")]
    pub services_names: Vec<String>,
    /// Whether this peer requested mempool transaction relay from us.
    #[serde(rename = "relaytxes")]
    pub relay_txs: bool,
    /// User agent is a string that represents the client being used by our peer. E.g.
    /// /Satoshi-26.0/ for bitcoin core version 26
    pub user_agent: String,
    /// Whether this peer connection is inbound.
    pub inbound: bool,
    /// Whether we selected this peer as a BIP152 high-bandwidth compact block relay peer.
    pub bip152_hb_to: bool,
    /// Whether this peer selected us as a BIP152 high-bandwidth compact block relay peer.
    pub bip152_hb_from: bool,
    /// This peer's height at the time we've opened a connection with them
    pub initial_height: u32,
    /// The peer time offset in seconds.
    #[serde(rename = "timeoffset")]
    pub time_offset: i64,
    /// The connection type of this peer
    ///
    /// We can connect with peers for different reasons. E.g. we can connect to a peer to
    /// see if it has a block we're missing, or just to check if that address is still alive.
    /// Possible values are: outbound-full-relay, block-relay-only, manual, feeler, and addr-fetch.
    pub kind: String,
    /// The state of this peer
    ///
    /// Can be either Ready, Connecting or Banned
    pub state: String,
    /// Special permissions granted to this peer.
    pub permissions: Vec<String>,
    /// The transport protocol used with peer.
    pub transport_protocol: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum GetBlockRes {
    Zero(String),

    One(Box<GetBlockVerboseOne>),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
/// The response for getblockheader, which can be either a raw hex-encoded block header or a verbose
/// one with all the fields parsed and decoded.
pub enum GetBlockHeaderRes {
    /// The raw hex-encoded block header, as returned by getblockheader with verbosity false
    Raw(String),

    /// A verbose block header, as returned by getblockheader with verbosity true
    Verbose(Box<GetBlockHeaderVerbose>),
}

/// A confidence enum to auxiliate rescan timestamp values.
///
/// Tells how much confidence you need for this rescan request. That is, the how conservative you want floresta to be when determining which block to start the rescan.
/// will make the rescan to start in a block that have an lower timestamp than the given in order to be more certain
/// about finding addresses and relevant transactions, a lower confidence will make the rescan to be closer to the given value.
///
/// This input is necessary to cover network variancy specially in testnet, for mainnet you can safely use low or medium confidences
/// depending on how much sure you are about the given timestamp covering the addresses you need.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum RescanConfidence {
    /// `high`: 99% confidence interval. Meaning 46 minutes in seconds.
    High,

    /// `medium` (default): 95% confidence interval. Meaning 30 minutes in seconds.
    Medium,

    /// `low`: 90% confidence interval. Meaning 23 minutes in seconds.
    Low,

    /// `exact`: Removes any lookback addition. Meaning 0 in seconds.
    Exact,
}

impl RescanConfidence {
    pub const fn as_secs(&self) -> u32 {
        match self {
            Self::Exact => 0,
            Self::Low => 1_380,
            Self::Medium => 1_800,
            Self::High => 2_760,
        }
    }
}

#[derive(Debug)]
/// All possible errors returned by the jsonrpc
pub enum Error {
    /// An error while deserializing our response
    Serde(serde_json::Error),

    #[cfg(feature = "with-jsonrpc")]
    /// An internal reqwest error
    JsonRpc(jsonrpc::Error),

    /// An error internal to our jsonrpc server
    Api(serde_json::Value),

    /// The server sent an empty response
    EmptyResponse,

    /// The provided verbosity level is invalid
    InvalidVerbosity,

    /// The user requested a rescan based on invalid values.
    InvalidRescanVal,

    /// The requested transaction output was not found
    TxOutNotFound,
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}

#[cfg(feature = "with-jsonrpc")]
impl From<jsonrpc::Error> for Error {
    fn from(value: jsonrpc::Error) -> Self {
        Self::JsonRpc(value)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "with-jsonrpc")]
            Self::JsonRpc(e) => write!(f, "JsonRpc returned an error {e}"),
            Self::Api(e) => write!(f, "general jsonrpc error: {e}"),
            Self::Serde(e) => write!(f, "error while deserializing the response: {e}"),
            Self::EmptyResponse => write!(f, "got an empty response from server"),
            Self::InvalidVerbosity => write!(f, "invalid verbosity level"),
            Self::InvalidRescanVal => write!(f, "Invalid rescan values"),
            Self::TxOutNotFound => write!(f, "Transaction output was not found"),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GetMemInfoStats {
    pub locked: MemInfoLocked,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MemInfoLocked {
    /// Memory currently in use, in bytes
    pub used: u64,
    /// Memory currently free, in bytes
    pub free: u64,
    /// Total memory allocated, in bytes
    pub total: u64,
    /// Total memory locked, in bytes
    ///
    /// If total is less than total, then some pages may be on swap or not philysically allocated
    /// yet
    pub locked: u64,
    /// How many chunks are currently in use
    pub chunks_used: u64,
    /// How many chunks are currently free
    pub chunks_free: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GetMemInfoRes {
    Stats(GetMemInfoStats),
    MallocInfo(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActiveCommand {
    pub method: String,
    pub duration: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetRpcInfoRes {
    pub active_commands: Vec<ActiveCommand>,
    pub logpath: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
#[enum_str_map(case = "lower", separator = "")]
/// Enum to represent the different subcommands for the addnode command
pub enum AddNodeCommand {
    /// Add a node to the addnode list (but not connect to it)
    Add,

    /// Remove a node from the addnode list (but not necessarily disconnect from it)
    Remove,

    /// Connect to a node once, but don't add it to the addnode list
    Onetry,
}

/// A simple implementation to convert the enum to a string.
/// Useful for get the subcommand name of addnode with
/// command.to_string()
impl Display for AddNodeCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let cmd = match self {
            Self::Add => "add",
            Self::Remove => "remove",
            Self::Onetry => "onetry",
        };
        write!(f, "{cmd}")
    }
}

impl error::Error for Error {}
