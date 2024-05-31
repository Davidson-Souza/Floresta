use std::fmt::Debug;

use crate::error::AppError;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::Txid;
use clap::Parser;
use clap::Subcommand;
use floresta_cli::reqwest_client::ReqwestClient;
use floresta_cli::rpc::FlorestaRPC;

mod error;
mod reqwest_client;
mod rpc;
mod rpc_types;

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();

    let client = ReqwestClient::new(get_host(&cli));
    let res = do_request(&cli, client)?;

    println!("{}", res);

    Ok(())
}

fn get_host(cmd: &Cli) -> String {
    if let Some(host) = cmd.rpc_host.clone() {
        return host;
    }

    match cmd.network {
        Network::Bitcoin => "http://127.0.0.1:8332".into(),
        Network::Testnet => "http://127.0.0.1:18332".into(),
        Network::Signet => "http://127.0.0.1:38332".into(),
        Network::Regtest => "http://127.0.0.1:18442".into(),
        _ => "http://127.0.0.1:8332".into(),
    }
}

fn do_request(cmd: &Cli, client: ReqwestClient) -> Result<String, AppError> {
    Ok(match cmd.methods.clone() {
        Methods::GetBlockchainInfo => serde_json::to_string_pretty(&client.get_blockchain_info()?)?,
        Methods::GetBlockHash { height } => {
            serde_json::to_string_pretty(&client.get_block_hash(height)?)?
        }
        Methods::GetTxOut { txid, vout } => {
            serde_json::to_string_pretty(&client.get_tx_out(txid, vout)?)?
        }
        Methods::GetTxProof { txids, .. } => {
            serde_json::to_string_pretty(&client.get_tx_proof(txids)?)?
        }
        Methods::GetRawTransaction { txid, .. } => {
            serde_json::to_string_pretty(&client.get_transaction(txid, Some(true))?)?
        }
        Methods::RescanBlockchain { start_height } => {
            serde_json::to_string_pretty(&client.rescan(start_height)?)?
        }
        Methods::SendRawTransaction { tx } => {
            serde_json::to_string_pretty(&client.send_raw_transaction(tx)?)?
        }
        Methods::GetBlockHeader { hash } => {
            serde_json::to_string_pretty(&client.get_block_header(hash)?)?
        }
        Methods::LoadDescriptor { desc, rescan } => {
            serde_json::to_string_pretty(&client.load_descriptor(desc, rescan)?)?
        }
        Methods::GetRoots => serde_json::to_string_pretty(&client.get_roots()?)?,
        Methods::GetBlock { hash, .. } => serde_json::to_string_pretty(&client.get_block(hash)?)?,
        Methods::GetPeerInfo => serde_json::to_string_pretty(&client.get_peer_info()?)?,
        Methods::Stop => serde_json::to_string_pretty(&client.stop()?)?,
        Methods::AddNode { node } => serde_json::to_string_pretty(&client.add_node(node)?)?,
        Methods::GetFilters { height } => {
            serde_json::to_string_pretty(&client.get_block_filter(height)?)?
        }
    })
}

#[derive(Debug, Parser)]
#[command(author = "Davidson Souza", version = "0.1.0", about = r#"
    A simple command line interface to the Floresta JSON RPC interface.
"#, long_about = None)]
pub struct Cli {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    pub config_file: Option<String>,
    /// Which network should we use
    #[arg(short, long, default_value_t=Network::Bitcoin)]
    pub network: Network,
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,
    /// The RPC host to connect to
    #[arg(short = 'H', long, value_name = "URL")]
    pub rpc_host: Option<String>,
    /// The RPC username to use
    #[arg(short = 'u', long, value_name = "USERNAME")]
    pub rpc_user: Option<String>,
    /// The RPC password to use
    #[arg(short = 'P', long, value_name = "PASSWORD")]
    pub rpc_password: Option<String>,
    /// An actual RPC command to run
    #[command(subcommand)]
    pub methods: Methods,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Methods {
    /// Returns information about the current state of the blockchain
    #[command(name = "getblockchaininfo")]
    GetBlockchainInfo,
    /// Returns the hash of the block associated with height
    #[command(name = "getblockhash")]
    GetBlockHash { height: u32 },
    /// Returns the proof that one or more transactions were included in a block
    #[command(name = "gettxproof")]
    GetTxProof {
        txids: Txid,
        blockhash: Option<BlockHash>,
    },
    /// Returns the raw transaction, assuming it is cached by our watch only wallet
    #[command(name = "getrawtransaction")]
    GetRawTransaction { txid: Txid, verbose: Option<bool> },
    /// Ask the node to rescan the blockchain for transactions
    #[command(name = "rescan")]
    RescanBlockchain { start_height: u32 },
    /// Submits a raw transaction to the network
    #[command(name = "sendrawtransaction")]
    SendRawTransaction { tx: String },
    /// Returns the block header for the given block hash
    #[command(name = "getblockheader")]
    GetBlockHeader { hash: BlockHash },
    /// Loads a new descriptor to the watch only wallet
    #[command(name = "loaddescriptor")]
    LoadDescriptor { desc: String, rescan: Option<u32> },
    /// Returns the roots of the current utreexo forest
    #[command(name = "getroots")]
    GetRoots,
    /// Returns a block
    #[command(name = "getblock")]
    GetBlock { hash: BlockHash, verbosity: u32 },
    /// Returns information about the peers we are connected to
    #[command(name = "getpeerinfo")]
    GetPeerInfo,
    /// Returns the value associated with a UTXO, if it's still not spent.
    /// This function only works properly if we have the compact block filters
    /// feature enabled
    #[command(name = "gettxout")]
    GetTxOut { txid: Txid, vout: u32 },
    /// Stops the node
    #[command(name = "stop")]
    Stop,
    /// Connects with a peer, given its address and port
    /// Usage: addnode <ip:[port]>
    #[command(name = "addnode")]
    AddNode { node: String },
    /// Returns the filters for a given block
    #[command(name = "getfilter")]
    GetFilters { height: u32 },
}
