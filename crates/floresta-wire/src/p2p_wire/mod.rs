// SPDX-License-Identifier: MIT OR Apache-2.0

//! Main module for the p2p chain. This is a blockchain provider, just like cli-chain, but it's
//! backed by p2p Bitcoin's p2p network.

use core::net::SocketAddr;
use std::path::PathBuf;

use bitcoin::Network;
use bitcoin::Script;
use bitcoin::ScriptBuf;
use bitcoin::consensus::Encodable;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256d;
use bitcoin::p2p::Magic;
use floresta_chain::AssumeUtreexoValue;

#[derive(Debug, Clone)]
/// Configuration for the Utreexo node.
pub struct UtreexoNodeConfig {
    /// The blockchain we are in, defaults to Bitcoin. Possible values are Bitcoin,
    /// Testnet, Regtest and Signet.
    pub network: Network,
    /// The BIP-325 challenge used to derive custom signet message-start bytes.
    ///
    /// [`None`] selects the default message-start bytes for `network`.
    pub signet_challenge: Option<ScriptBuf>,
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
    /// Fixed peers to connect to. Defaults to an empty list.
    ///
    /// Each entry is `host[:port]`, where `host` is an IPv4 address, a bracketed IPv6 address (`[::1]`), or a hostname;
    /// `port` is optional and defaults to the network's default port (for example, `"localhost"` or `"127.0.0.1:8333"`).
    pub fixed_peers: Vec<String>,
    /// Maximum ban score. Defaults to 100.
    ///
    /// If a peer misbehaves, we increase its ban score. If the ban score reaches this value,
    /// we disconnect from the peer.
    pub max_banscore: u32,
    /// Data directory for the node. Defaults to `.floresta-node`.
    pub datadir: PathBuf,
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
    /// If we are using network-provided block filters, we may not need to download the whole
    /// chain of filters, as our wallets may not have been created at the beginning of the chain.
    /// With this option, we can make a rough estimate of the block height we need to start
    /// and only download the filters from that height.
    ///
    /// If the value is negative, it's relative to the current tip. For example, if the current
    /// tip is at height 1000, and we set this value to -100, we will start downloading filters
    /// from height 900.
    pub filter_start_height: Option<i32>,
    /// The user agent that we will advertise to our peers. Defaults to `floresta:<version>`.
    pub user_agent: String,
    /// Whether to allow fallback to v1 transport if v2 connection fails.
    /// Defaults to true.
    pub allow_v1_fallback: bool,
    /// Whether to disable DNS seeds. Defaults to false.
    pub disable_dns_seeds: bool,
}

impl UtreexoNodeConfig {
    /// Returns the message-start bytes used by this node.
    pub fn network_magic(&self) -> Magic {
        match (self.network, self.signet_challenge.as_deref()) {
            (Network::Signet, Some(challenge)) => signet_magic(challenge),
            _ => self.network.magic(),
        }
    }
}

/// Derives the BIP-325 message-start bytes from a signet challenge.
pub fn signet_magic(challenge: &Script) -> Magic {
    let mut engine = sha256d::Hash::engine();
    challenge
        .consensus_encode(&mut engine)
        .expect("hash engines are infallible");
    let hash = sha256d::Hash::from_engine(engine).to_byte_array();

    Magic::from_bytes([hash[0], hash[1], hash[2], hash[3]])
}

impl Default for UtreexoNodeConfig {
    fn default() -> Self {
        Self {
            disable_dns_seeds: false,
            network: Network::Bitcoin,
            signet_challenge: None,
            pow_fraud_proofs: false,
            compact_filters: false,
            fixed_peers: Vec::new(),
            max_banscore: 100,
            datadir: ".floresta-node".into(),
            proxy: None,
            backfill: false,
            assume_utreexo: None,
            filter_start_height: None,
            user_agent: format!("floresta:{}", env!("CARGO_PKG_VERSION")),
            allow_v1_fallback: true,
        }
    }
}

pub mod address_man;
pub mod bitcoin_socket_addr;
pub mod block_proof;
pub mod error;
pub mod network_message_ext;
pub mod node;
pub mod node_context;
pub mod node_handle;
pub mod node_interface;
pub mod onion;
pub mod peer;
pub mod socks;
mod stump_updater;
#[cfg(test)]
#[doc(hidden)]
pub mod tests;
pub mod transport;

#[cfg(test)]
mod magic_tests {
    use floresta_chain::ChainParams;

    use super::*;

    #[test]
    fn signet_magic_is_derived_from_serialized_challenge() {
        let challenge = ScriptBuf::from_hex(
            "512103ad5e0edad18cb1f0fc0d28a3d4f1f3e445640337489abb10404f2d1e086be43051ae",
        )
        .expect("BIP-325 example challenge");
        let config = UtreexoNodeConfig {
            network: Network::Signet,
            signet_challenge: Some(challenge),
            ..Default::default()
        };

        assert_eq!(
            config.network_magic(),
            Magic::from_bytes([0x7e, 0xc6, 0x53, 0xa5])
        );
    }

    #[test]
    fn default_signet_challenge_derives_builtin_magic() {
        let params = ChainParams::from(Network::Signet);
        let challenge = params
            .signet_challenge
            .as_deref()
            .expect("signet has a challenge");

        assert_eq!(signet_magic(challenge), Magic::SIGNET);
    }
}
