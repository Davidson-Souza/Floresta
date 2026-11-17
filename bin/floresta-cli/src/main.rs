// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt::Debug;
use std::path::PathBuf;

use anyhow::Ok;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::ScriptBuf;
use bitcoin::Txid;
use clap::Parser;
use clap::Subcommand;
use floresta_common::NetworkExt;
use floresta_rpc::jsonrpc_client::Client;
use floresta_rpc::jsonrpc_client::JsonRPCConfig;
use floresta_rpc::rpc_interfaces::BlockchainRpc;
use floresta_rpc::rpc_interfaces::ControlRpc;
use floresta_rpc::rpc_interfaces::NetworkRpc;
use floresta_rpc::rpc_interfaces::RawTransactionRpc;
use floresta_rpc::rpc_interfaces::WalletRpc;
use floresta_rpc::rpc_types::AddNodeCommand;
use floresta_rpc::rpc_types::RescanConfidence;

// `floresta-cli` cannot be built with the `async` feature. If another crate in the same
// build enables it (feature unification), build this binary alone.
const _: () = assert!(
    !floresta_rpc::IS_ASYNC,
    "floresta-cli requires `floresta-rpc` without the `async` feature, but a crate built in \
     the same invocation (e.g. `floresta-node`) enables it through feature unification. \
     Build this binary on its own: `cargo build -p floresta-cli`."
);

// Main function that runs the CLI application
fn main() -> anyhow::Result<()> {
    // Parse command line arguments into a Cli struct
    let cli = Cli::parse();

    // Resolve basic-auth credentials: explicit flags win, otherwise read the
    // cookie file written by florestad at startup.
    let (user, pass) = credentials::resolve(&cli)?;

    // Create a new JSON-RPC client using the host and credentials.
    let client = Client::new_with_config(JsonRPCConfig {
        url: get_host(&cli),
        user: Some(user),
        pass: Some(pass),
    });

    // Perform the requested RPC call and get the result
    let res = do_request(&cli, client)?;

    // Print the result to the console
    println!("{res}");

    // Return Ok to indicate the program ran successfully
    anyhow::Ok(())
}

/// Basic-auth credential resolution for RPC calls.
mod credentials {
    use std::path::Path;

    use anyhow::Context;
    use floresta_rpc::net_datadir;
    use floresta_rpc::read_cookie_file;
    use floresta_rpc::resolve_cookie_path;

    use crate::Cli;

    /// Resolve the basic-auth credentials for the RPC call.
    ///
    /// The password decides, as in `bitcoin-cli` (bitcoin-cli.cpp:869) and
    /// florestad:
    ///   1. `--rpc-password` non-empty: send it with `--rpc-user`, or an empty
    ///      user when absent. Such a daemon writes no cookie to fall back to.
    ///   2. Otherwise read the cookie at `--rpc-cookie-file` (or the network
    ///      default) and split on the first `:`.
    ///
    /// An empty `--rpc-password` counts as unset, so `-P "$UNSET_VAR"` reaches
    /// the cookie instead of 401ing. A lone `--rpc-user` warns and falls
    /// through, since the daemon ignores it too.
    pub(super) fn resolve(cli: &Cli) -> anyhow::Result<(String, String)> {
        if let Some(pass) = cli.rpc_password.clone().filter(|p| !p.is_empty()) {
            return Ok((cli.rpc_user.clone().unwrap_or_default(), pass));
        }
        if cli.rpc_user.is_some() {
            // "non-empty" because an empty `-P` lands here too.
            eprintln!(
                "warning: --rpc-user set without a non-empty --rpc-password; falling back to cookie auth"
            );
        }
        let datadir = net_datadir(cli.data_dir.as_deref(), cli.network);
        let cookie_path = resolve_cookie_path(&datadir, cli.rpc_cookie_file.as_deref());
        read_cookie(&cookie_path)
    }

    /// Wrap `floresta_rpc::read_cookie_file` with CLI-specific error context.
    fn read_cookie(path: &Path) -> anyhow::Result<(String, String)> {
        read_cookie_file(path).with_context(|| {
            format!(
                "failed to read RPC cookie file at {}; start florestad first or pass --rpc-user/--rpc-password",
                path.display()
            )
        })
    }

    #[cfg(test)]
    mod tests {
        use clap::Parser;

        use super::resolve;
        use crate::Cli;

        fn parse(args: &[&str]) -> Cli {
            let mut argv = vec!["floresta-cli"];
            argv.extend_from_slice(args);
            argv.push("getblockchaininfo");
            Cli::parse_from(argv)
        }

        #[test]
        fn password_without_user_sends_an_empty_user() {
            // A password-only daemon writes no cookie, so falling back would
            // leave it unreachable. `bitcoin-cli` sends `:<pass>` here too.
            // The empty data dir makes a fallback fail rather than pick up a
            // real cookie from the developer's home directory.
            let cli = parse(&[
                "--rpc-password=hunter2",
                "--data-dir=/nonexistent-floresta-datadir",
            ]);

            let (user, pass) = resolve(&cli).expect("password alone resolves without a cookie");

            assert_eq!(user, "");
            assert_eq!(pass, "hunter2");
        }

        #[test]
        fn both_flags_are_passed_through() {
            let cli = parse(&["--rpc-user=alice", "--rpc-password=hunter2"]);

            let (user, pass) = resolve(&cli).expect("an explicit pair resolves");

            assert_eq!(user, "alice");
            assert_eq!(pass, "hunter2");
        }

        #[test]
        fn user_without_password_falls_back_to_the_cookie() {
            // An empty data dir makes the fallback observable as a read error.
            let cli = parse(&[
                "--rpc-user=alice",
                "--data-dir=/nonexistent-floresta-datadir",
            ]);

            let err = resolve(&cli).expect_err("no cookie exists to fall back to");

            assert!(
                err.to_string().contains("failed to read RPC cookie file"),
                "expected the cookie path to be attempted, got: {err}"
            );
        }

        #[test]
        fn empty_password_falls_back_to_the_cookie() {
            // `-P "$UNSET_VAR"` must not send a blank password and 401. Core
            // (bitcoin-cli.cpp:869) and florestad both read empty as unset.
            for args in [
                vec![
                    "--rpc-password=",
                    "--data-dir=/nonexistent-floresta-datadir",
                ],
                vec![
                    "--rpc-user=alice",
                    "--rpc-password=",
                    "--data-dir=/nonexistent-floresta-datadir",
                ],
            ] {
                let err = resolve(&parse(&args)).expect_err("no cookie to fall back to");

                assert!(
                    err.to_string().contains("failed to read RPC cookie file"),
                    "expected the cookie path to be attempted for {args:?}, got: {err}"
                );
            }
        }
    }
}

// Function to determine the RPC host based on CLI arguments and network type
fn get_host(cmd: &Cli) -> String {
    // If a specific RPC host is provided, use it
    if let Some(host) = cmd.rpc_host.clone() {
        return host;
    }

    // Otherwise, use the default host based on the network type
    format!("http://127.0.0.1:{}", cmd.network.default_rpc_port())
}

// Function to perform the requested RPC call based on CLI arguments
fn do_request(cmd: &Cli, client: Client) -> anyhow::Result<String> {
    Ok(match cmd.methods.clone() {
        // Handle each possible RPC method and serialize the result to a pretty JSON string
        Methods::GetBlockchainInfo => serde_json::to_string_pretty(&client.get_blockchain_info()?)?,
        Methods::GetBlockHash { height } => {
            serde_json::to_string_pretty(&client.get_block_hash(height)?)?
        }
        Methods::GetBestBlockHash => serde_json::to_string_pretty(&client.get_best_block_hash()?)?,
        Methods::GetBlockCount => serde_json::to_string_pretty(&client.get_block_count()?)?,
        Methods::GetDifficulty => serde_json::to_string_pretty(&client.get_difficulty()?)?,
        Methods::GetTxOut {
            txid,
            vout,
            include_mempool,
        } => serde_json::to_string_pretty(&client.get_tx_out(txid, vout, include_mempool)?)?,
        Methods::GetTxOutProof { txids, blockhash } => {
            serde_json::to_string_pretty(&client.get_txout_proof(&txids, blockhash)?)?
        }
        Methods::GetRawTransaction { txid, verbosity } => {
            serde_json::to_string_pretty(&client.get_raw_transaction(txid, verbosity)?)?
        }
        Methods::RescanBlockchain {
            start_block,
            stop_block,
            use_timestamp,
            confidence,
        } => serde_json::to_string_pretty(&client.rescan_blockchain(
            start_block,
            stop_block,
            use_timestamp,
            confidence,
        )?)?,
        Methods::SendRawTransaction { tx } => {
            serde_json::to_string_pretty(&client.send_raw_transaction(tx)?)?
        }
        Methods::GetBlockHeader { hash, verbosity } => {
            serde_json::to_string_pretty(&client.get_block_header(hash, verbosity)?)?
        }
        Methods::LoadDescriptor { desc } => {
            serde_json::to_string_pretty(&client.load_descriptor(desc)?)?
        }
        Methods::GetRoots => serde_json::to_string_pretty(&client.get_roots()?)?,
        Methods::GetBlock { hash, verbosity } => {
            serde_json::to_string_pretty(&client.get_block(hash, verbosity)?)?
        }
        Methods::GetPeerInfo => serde_json::to_string_pretty(&client.get_peer_info()?)?,
        Methods::GetConnectionCount => {
            serde_json::to_string_pretty(&client.get_connection_count()?)?
        }
        Methods::Stop => serde_json::to_string_pretty(&client.stop()?)?,
        Methods::AddNode {
            node,
            command,
            v2transport,
        } => serde_json::to_string_pretty(&client.add_node(node, command, v2transport)?)?,
        Methods::DisconnectNode {
            node_address,
            node_id,
        } => serde_json::to_string_pretty(&client.disconnect_node(node_address, node_id)?)?,
        Methods::FindTxOut {
            txid,
            vout,
            script,
            height_hint,
        } => {
            let script = ScriptBuf::from_hex(&script).map_err(|e| anyhow::anyhow!("{e}"))?;
            serde_json::to_string_pretty(&client.find_tx_out(txid, vout, script, height_hint)?)?
        }
        Methods::GetMemoryInfo { mode } => {
            serde_json::to_string_pretty(&client.get_memory_info(mode.as_deref())?)?
        }
        Methods::GetRpcInfo => serde_json::to_string_pretty(&client.get_rpc_info()?)?,
        Methods::Uptime => serde_json::to_string_pretty(&client.uptime()?)?,
        Methods::ListDescriptors => serde_json::to_string_pretty(&client.list_descriptors()?)?,
        Methods::Ping => serde_json::to_string_pretty(&client.ping()?)?,
        Methods::GetNetworkInfo => serde_json::to_string_pretty(&client.get_network_info()?)?,
        Methods::GetDeploymentInfo { blockhash } => {
            serde_json::to_string_pretty(&client.get_deployment_info(blockhash)?)?
        }
        Methods::GetAddrManInfo => serde_json::to_string_pretty(&client.get_addrman_info()?)?,
    })
}

#[derive(Debug, Parser)]
#[command(author = "Davidson Souza", version = env!("CARGO_PKG_VERSION"), about = r#"
    A simple command line interface to the Floresta JSON RPC interface.
"#, long_about = None)]
pub struct Cli {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    pub config_file: Option<PathBuf>,
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
    /// Data directory florestad is using. Defaults to `~/.floresta`.
    #[arg(long, value_name = "PATH")]
    pub data_dir: Option<PathBuf>,
    /// Path to the RPC cookie file. Defaults to `<datadir>/[<net>/].cookie`.
    ///
    /// Relative paths are prefixed by the network-specific data directory,
    /// matching how florestad resolves the same flag.
    #[arg(long, value_name = "PATH")]
    pub rpc_cookie_file: Option<PathBuf>,
    /// An actual RPC command to run
    #[command(subcommand)]
    pub methods: Methods,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Methods {
    #[doc = include_str!("../../../doc/rpc/getblockchaininfo.md")]
    #[command(
        name = "getblockchaininfo",
        about = "Returns an object containing various state info regarding blockchain processing.",
        long_about = Some(include_str!("../../../doc/rpc/getblockchaininfo.md")),
        disable_help_subcommand = true
    )]
    GetBlockchainInfo,

    #[doc = include_str!("../../../doc/rpc/getblockhash.md")]
    #[command(
        name = "getblockhash",
        about = "Returns the hash of the block at the given height.",
        long_about = Some(include_str!("../../../doc/rpc/getblockhash.md")),
        disable_help_subcommand = true
    )]
    GetBlockHash { height: u32 },

    #[doc = include_str!("../../../doc/rpc/getbestblockhash.md")]
    #[command(
        name = "getbestblockhash",
        about = "Returns the hash of the best (tip) block in the most-work chain.",
        long_about = Some(include_str!("../../../doc/rpc/getbestblockhash.md")),
        disable_help_subcommand = true
    )]
    GetBestBlockHash,

    #[doc = include_str!("../../../doc/rpc/getblockcount.md")]
    #[command(
        name = "getblockcount",
        about = "Returns the height of the most-work chain.",
        long_about = Some(include_str!("../../../doc/rpc/getblockcount.md")),
        disable_help_subcommand = true
    )]
    GetBlockCount,

    #[doc = include_str!("../../../doc/rpc/getdifficulty.md")]
    #[command(
        name = "getdifficulty",
        about = "Returns the proof-of-work difficulty as a multiple of the minimum difficulty.",
        long_about = Some(include_str!("../../../doc/rpc/getdifficulty.md")),
        disable_help_subcommand = true
    )]
    GetDifficulty,

    #[doc = include_str!("../../../doc/rpc/gettxoutproof.md")]
    #[command(
        name = "gettxoutproof",
        about = "Returns a hex-encoded Merkle proof showing that one or more transactions were included in a block.",
        long_about = Some(include_str!("../../../doc/rpc/gettxoutproof.md")),
        disable_help_subcommand = true
    )]
    GetTxOutProof {
        /// The transaction IDs to prove
        #[arg(required = true, value_parser = |s: &str| serde_json::from_str::<Vec<Txid>>(s).map_err(|e| e.to_string()))]
        txids: std::vec::Vec<Txid>, // you need to specify the path of Vec https://github.com/clap-rs/clap/discussions/4695

        /// The block in which to look for the transactions
        #[arg(required = false)]
        blockhash: Option<BlockHash>,
    },

    /// Returns the transaction, assuming it is cached by our watch only wallet
    #[doc = include_str!("../../../doc/rpc/getrawtransaction.md")]
    #[command(
        name = "getrawtransaction",
        about = "Returns raw transaction data for a given txid from wallet cache (controlled by verbosity level)",
        long_about = Some(include_str!("../../../doc/rpc/getrawtransaction.md")),
        disable_help_subcommand = true
    )]
    GetRawTransaction { txid: Txid, verbosity: Option<u8> },

    #[doc = include_str!("../../../doc/rpc/rescanblockchain.md")]
    #[command(
        name = "rescanblockchain",
        about = "Sends a request to the node to rescan the blockchain searching for transactions related to the wallet's cached addresses.",
        long_about = Some(include_str!("../../../doc/rpc/rescanblockchain.md")),
        disable_help_subcommand = true
    )]
    RescanBlockchain {
        /// The starting point for the rescan. (optional)
        start_block: Option<u32>,

        /// The stopping height for the rescan. (optional)
        stop_block: Option<u32>,

        /// Treat the start parameter as a UNIX timestamp instead of block height.
        #[arg(short = 't', long = "timestamp", required = false)]
        use_timestamp: Option<bool>,

        #[arg(short = 'c', long = "confidence", required = false, value_enum)]
        confidence: Option<RescanConfidence>,
    },

    /// Submits a raw transaction to the network
    #[doc = include_str!("../../../doc/rpc/sendrawtransaction.md")]
    #[command(
        name = "sendrawtransaction",
        about = "Broadcast a serialized transaction to the P2P network",
        long_about = Some(include_str!("../../../doc/rpc/sendrawtransaction.md")),
        disable_help_subcommand = true
    )]
    SendRawTransaction { tx: String },

    /// Returns the block header for the given block hash
    #[doc = include_str!("../../../doc/rpc/getblockheader.md")]
    #[command(
        name = "getblockheader",
        about = "Returns the block header for the given block hash",
        long_about = Some(include_str!("../../../doc/rpc/getblockheader.md")),
        disable_help_subcommand = true
    )]
    GetBlockHeader {
        hash: BlockHash,
        verbosity: Option<bool>,
    },

    #[doc = include_str!("../../../doc/rpc/getdeploymentinfo.md")]
    #[command(
        name = "getdeploymentinfo",
        about = "Returns data regarding deployments of consensus changes.",
        long_about = Some(include_str!("../../../doc/rpc/getdeploymentinfo.md")),
        disable_help_subcommand = true
    )]
    GetDeploymentInfo { blockhash: Option<BlockHash> },

    /// Loads a new descriptor to the watch only wallet
    #[doc = include_str!("../../../doc/rpc/loaddescriptor.md")]
    #[command(
        name = "loaddescriptor",
        about = "Loads a descriptor into the watch-only wallet, extracting the address(es) for being cached",
        long_about = Some(include_str!("../../../doc/rpc/loaddescriptor.md")),
        disable_help_subcommand = true
    )]
    LoadDescriptor { desc: String },

    #[doc = include_str!("../../../doc/rpc/getroots.md")]
    #[command(
        name = "getroots",
        about = "Returns the roots of the current utreexo forest",
        long_about = Some(include_str!("../../../doc/rpc/getroots.md")),
        disable_help_subcommand = true
    )]
    GetRoots,

    /// Returns a block
    #[doc = include_str!("../../../doc/rpc/getblock.md")]
    #[command(
        name = "getblock",
        about = "Returns information about a block",
        long_about = Some(include_str!("../../../doc/rpc/getblock.md")),
        disable_help_subcommand = true
    )]
    GetBlock {
        hash: BlockHash,
        verbosity: Option<u32>,
    },

    #[doc = include_str!("../../../doc/rpc/getpeerinfo.md")]
    #[command(
        name = "getpeerinfo",
        about = "Returns information about the peers we are connected to",
        long_about = Some(include_str!("../../../doc/rpc/getpeerinfo.md")),
        disable_help_subcommand = true
    )]
    GetPeerInfo,

    #[doc = include_str!("../../../doc/rpc/getconnectioncount.md")]
    #[command(
        name = "getconnectioncount",
        about = "Returns the number of connections to other nodes",
        long_about = Some(include_str!("../../../doc/rpc/getconnectioncount.md")),
        disable_help_subcommand = true
    )]
    GetConnectionCount,

    #[doc = include_str!("../../../doc/rpc/gettxout.md")]
    #[command(
        name = "gettxout",
        about = "Returns details about an unspent transaction output.",
        long_about = Some(include_str!("../../../doc/rpc/gettxout.md")),
        disable_help_subcommand = true
    )]
    GetTxOut {
        txid: Txid,
        vout: u32,
        include_mempool: Option<bool>,
    },

    #[doc = include_str!("../../../doc/rpc/stop.md")]
    #[command(
        name = "stop",
        about = "Request a graceful shutdown of Floresta.",
        long_about = Some(include_str!("../../../doc/rpc/stop.md")),
        disable_help_subcommand = true
    )]
    Stop,

    #[doc = include_str!("../../../doc/rpc/addnode.md")]
    #[command(
        name = "addnode",
        about = "Attempts to add or remove a node from the list of addnodes",
        long_about = Some(include_str!("../../../doc/rpc/addnode.md")),
        disable_help_subcommand = true
    )]
    AddNode {
        node: String,
        command: AddNodeCommand,
        v2transport: Option<bool>,
    },

    #[doc = include_str!("../../../doc/rpc/disconnectnode.md")]
    #[command(
        name = "disconnectnode",
        about = "Immediately disconnect from a peer",
        long_about = Some(include_str!("../../../doc/rpc/disconnectnode.md")),
        disable_help_subcommand = true
    )]
    DisconnectNode {
        node_address: String,
        node_id: Option<u32>,
    },

    #[command(name = "findtxout")]
    FindTxOut {
        txid: Txid,
        vout: u32,
        script: String,
        height_hint: Option<u32>,
    },

    #[doc = include_str!("../../../doc/rpc/getmemoryinfo.md")]
    #[command(
        name = "getmemoryinfo",
        about = "Returns statistics about Floresta's memory usage.",
        long_about = Some(include_str!("../../../doc/rpc/getmemoryinfo.md")),
        disable_help_subcommand = true
    )]
    GetMemoryInfo { mode: Option<String> },

    #[doc = include_str!("../../../doc/rpc/getrpcinfo.md")]
    #[command(
        name = "getrpcinfo",
        about = "Returns information about the RPC server",
        long_about = Some(include_str!("../../../doc/rpc/getrpcinfo.md")),
        disable_help_subcommand = true
    )]
    GetRpcInfo,

    #[doc = include_str!("../../../doc/rpc/uptime.md")]
    #[command(
        name = "uptime",
        about = "Returns for how long the node has been running, in seconds",
        long_about = Some(include_str!("../../../doc/rpc/uptime.md")),
        disable_help_subcommand = true
    )]
    Uptime,

    #[doc = include_str!("../../../doc/rpc/listdescriptors.md")]
    #[command(
        name = "listdescriptors",
        about = "Returns a list of all descriptors currently loaded in the wallet",
        long_about = Some(include_str!("../../../doc/rpc/listdescriptors.md")),
        disable_help_subcommand = true
    )]
    ListDescriptors,

    #[doc = include_str!("../../../doc/rpc/ping.md")]
    #[command(
        name = "ping",
        about = "Sends a ping to all peers, checking if they are still alive",
        long_about = Some(include_str!("../../../doc/rpc/ping.md")),
        disable_help_subcommand = true
    )]
    Ping,

    #[doc = include_str!("../../../doc/rpc/getnetworkinfo.md")]
    #[command(
        name = "getnetworkinfo",
        about = "Returns information about the network and connected peers",
        long_about = Some(include_str!("../../../doc/rpc/getnetworkinfo.md")),
        disable_help_subcommand = true
    )]
    GetNetworkInfo,
    #[doc = include_str!("../../../doc/rpc/getaddrmaninfo.md")]
    #[command(
        name = "getaddrmaninfo",
        about = "Returns address manager statistics broken down by network type",
        long_about = Some(include_str!("../../../doc/rpc/getaddrmaninfo.md")),
        disable_help_subcommand = true
    )]
    GetAddrManInfo,
}
