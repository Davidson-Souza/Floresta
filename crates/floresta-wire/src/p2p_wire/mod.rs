//! Main module for the p2p chain. This is a blockchain provider, just like cli-chain, but it's
//! backed by p2p Bitcoin's p2p network.

use std::net::SocketAddr;

use bitcoin::Network;
use floresta_chain::AssumeUtreexoValue;

use self::address_man::LocalAddress;

#[derive(Debug, Clone)]
/// Configuration for the Utreexo node.
pub struct UtreexoNodeConfig {
    /// The blockchain we are in, defaults to Bitcoin. Possible values are Bitcoin,
    /// Testnet, Regtest and Signet.
    pub network: Network,
    /// Whether to use PoW fraud proofs. Defaults to false.
    ///
    /// PoW fraud proof is a mechanism to skip the verification of the whole blockchain,
    /// but while also giving a better security than simple SPV. Check out the documentation
    /// in `pow_fraud_proofs.md` under the `docs` folder.
    pub pow_fraud_proofs: bool,
    /// Whether to use compact filters. Defaults to false.
    ///
    /// Compact filters are useful to rescan the blockchain for a specific address, without
    /// needing to download the whole chain. It will download ~1GB of filters, and then
    /// download the blocks that match the filters.
    pub compact_filters: bool,
    /// Fixed peers to connect to. Defaults to None.
    ///
    /// If you want to connect to a specific peer, you can set this to a string with the
    /// format `ip:port`. For example, `localhost:8333`.
    pub fixed_peer: Option<LocalAddress>,
    /// Maximum ban score. Defaults to 100.
    ///
    /// If a peer misbehaves, we increase its ban score. If the ban score reaches this value,
    /// we disconnect from the peer.
    pub max_banscore: u32,
    /// Maximum number of outbound connections. Defaults to 8.
    pub max_outbound: u32,
    /// Maximum number of inflight requests. Defaults to 10.
    ///
    /// More inflight requests means more memory usage, but also more parallelism.
    pub max_inflight: u32,
    /// Data directory for the node. Defaults to `.floresta-node`.
    pub datadir: String,
    /// A SOCKS5 proxy to use. Defaults to None.
    pub proxy: Option<SocketAddr>,
    /// If enabled, the node will assume that the provided Utreexo state is valid, and will
    /// start running from there
    pub assume_utreexo: Option<AssumeUtreexoValue>,
    /// If we assumeutreexo or pow_fraud_proof, we can skip the IBD and make our node usable
    /// faster, with the tradeoff of security. If this is enabled, we will still download the
    /// blocks in the background, and verify the final Utreexo state. So, the worse case scenario
    /// is that we are vulnerable to a fraud proof attack for a few hours, but we can spot it
    /// and react in a couple of hours at most, so the attack window is very small.
    pub backfill: bool,
}

impl Default for UtreexoNodeConfig {
    fn default() -> Self {
        UtreexoNodeConfig {
            network: Network::Bitcoin,
            pow_fraud_proofs: false,
            compact_filters: false,
            fixed_peer: None,
            max_banscore: 100,
            max_outbound: 8,
            max_inflight: 10,
            datadir: ".floresta-node".to_string(),
            proxy: None,
            backfill: false,
            assume_utreexo: None,
        }
    }
}

pub mod address_man;
pub mod chain_selector;
pub mod error;
pub mod mempool;
pub mod node;
pub mod node_context;
pub mod node_interface;
pub mod peer;
pub mod running_node;
pub mod socks;
pub mod stream_reader;
pub mod sync_node;
