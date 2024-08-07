extern crate alloc;
use alloc::vec::Vec;
use core::ffi::c_uint;

#[cfg(feature = "bitcoinconsensus")]
use bitcoin::bitcoinconsensus::VERIFY_NONE;
#[cfg(feature = "bitcoinconsensus")]
use bitcoin::bitcoinconsensus::VERIFY_P2SH;
#[cfg(feature = "bitcoinconsensus")]
use bitcoin::bitcoinconsensus::VERIFY_WITNESS;
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::p2p::ServiceFlags;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Target;
use rustreexo::accumulator::node_hash::NodeHash;

use crate::prelude::*;
use crate::Network;
#[derive(Clone, Debug)]
pub struct ChainParams {
    /// The network's first block, also called genesis block.
    pub genesis: Block,
    /// Wheter or not to verify script. Requires "bitcoinconsensus" feature.
    pub verify_script: bool,
    /// Whether or not we are allowed to mine blocks with the network's smallest diff, this
    /// is used in testnet, if a block takes more than 20 minutes to be mined
    pub pow_allow_min_diff: bool,
    /// Whether or not we are allowed to mine without retargets, this is used in regtests
    pub pow_allow_no_retarget: bool,
    /// This is the maximum possible target (i.e, minimum possible difficulty), and in mainnet
    /// it's defined as ~((uint256)0 >> 32).
    pub max_target: Target,
    /// Interval of blocks until the block reward halves
    pub subsidy_halving_interval: u64,
    /// We expect blocks to take this many seconds to be found, on average
    pub pow_target_spacing: u64,
    /// When we retarget we expect this many seconds to be elapsed since last time. If
    /// it's more, we decrease difficulty, if it's less we increase difficulty
    pub pow_target_timespan: u64,
    /// We wait this many blocks before a coinbase output can be spent
    pub coinbase_maturity: u32,
    /// The height at which bip32 is activated
    pub bip34_activation_height: u32,
    /// The height at which bip65 is activated
    pub bip65_activation_height: u32,
    /// The height at which bip66 is activated
    pub bip66_activation_height: u32,
    /// The height at which segwit is activated
    pub segwit_activation_height: u32,
    /// The height at which csv(CHECK_SEQUENCE_VERIFY) is activated
    pub csv_activation_height: u32,
    /// A list of exceptions to the rules, where the key is the block hash and the value is the
    /// verification flags
    pub exceptions: HashMap<BlockHash, c_uint>,
}

/// A dns seed is a authoritative DNS server that returns the IP addresses of nodes that are
/// likely to be accepting incoming connections. This is our prefered way of finding new peers
/// on the first startup, as peers returned by seeds are likely to be online and accepting
/// connections. We may use this as a fallback if we don't have any peers to connect in
/// subsequent startups.
///
/// Some seeds allow filtering by service flags, so we may use this to find peers that are
/// likely to be running Utreexo, for example.
pub struct DnsSeed {
    /// The network this peer supports (e.g, mainnet, testnet, etc)
    pub network: Network,
    /// The domain name of the seed
    pub seed: &'static str,
    /// Useful filters we can use to find relevant peers
    pub filters: ServiceFlags,
}

impl DnsSeed {
    pub fn new(network: Network, seed: &'static str, filters: ServiceFlags) -> Self {
        DnsSeed {
            network,
            seed,
            filters,
        }
    }
}

/// If enabled, the node will assume that the provided Utreexo state is valid, and will
/// start running from there. You may use this to make your node start faster, but you
/// should be sure that the provided state is valid. You may or not verify the state,
/// by downloading all blocks on background, and then verifying the final Utreexo state.
#[derive(Debug, Clone)]
pub struct AssumeUtreexoValue {
    /// The latest block assumed to be valid. This acc is the roots at this block
    pub block_hash: BlockHash,
    /// Same as block_hash, but in height
    pub height: u32,
    /// The roots of the Utreexo accumulator at this block
    pub roots: Vec<NodeHash>,
    /// The number of leaves in the Utreexo accumulator at this block
    pub leaves: u64,
}

impl ChainParams {
    pub fn get_assumeutreexo_value(network: Network) -> AssumeUtreexoValue {
        match network {
            Network::Bitcoin => AssumeUtreexoValue {
                block_hash: BlockHash::from_str(
                    "00000000000000000002447c802760582b80022f937d754aece888ec9e96f212",
                )
                .unwrap(),
                height: 849070,
                roots: [
                    "d9eb4f63c9c302154cfde1d7c36f20674877653883f7d857550351ed3558545c",
                    "7519893244433168ef4dc3a21e3221b87fa5b3802f4a745be33baa9defa7ac6b",
                    "59be29510b035212adcffc970f71a889e81db2298ab3bff4d0e3e1b56caa8976",
                    "3e4833c382dbfa4fd5e496b37ac6769036bba655534249a9d7d5a2a221718a39",
                    "4569761976542d95af3c765b6fc7e149b71659e5f89ad1cc13a7b00bbd7e9d8d",
                    "33fbd5f96c50327d309722970378bea7474b171097ba40bc98b360871a1f38c2",
                    "d2955774ac37daa1e16a2fd9d419906acf6bdf6ffd6f171a9cf72935175cec6f",
                    "ccac02fac6922f5082e893b83c0898636be24f47dd6c00f502425c9dedab2c25",
                    "e418ad2e3b516c54a873cd89aec3bbf88513e16192f817b150f168a939f3ad3c",
                    "1931c7dbc568c4fe92cceab72e870da307df88c8f6e7a01bf6dbfdac4dd88073",
                ]
                .into_iter()
                .map(|x| NodeHash::from_str(x).unwrap())
                .collect(),
                leaves: 2554606658,
            },
            Network::Testnet => AssumeUtreexoValue {
                block_hash: genesis_block(network.into()).block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
            Network::Signet => AssumeUtreexoValue {
                block_hash: genesis_block(network.into()).block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
            Network::Regtest => AssumeUtreexoValue {
                block_hash: genesis_block(network.into()).block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
        }
    }
}

impl ChainParams {
    fn max_target(net: Network) -> Target {
        match net {
            Network::Bitcoin => Target::MAX_ATTAINABLE_MAINNET,
            Network::Testnet => Target::MAX_ATTAINABLE_TESTNET,
            Network::Signet => Target::MAX_ATTAINABLE_SIGNET,
            Network::Regtest => Target::MAX_ATTAINABLE_REGTEST,
        }
    }
}

#[cfg(feature = "bitcoinconsensus")]
fn get_exceptions() -> HashMap<BlockHash, c_uint> {
    // For some reason, some blocks in the mainnet and testnet have different rules than it should
    // be, so we need to keep a list of exceptions and treat them differently
    let mut exceptions = HashMap::new();
    exceptions.insert(
        BlockHash::from_str("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22")
            .unwrap(),
        VERIFY_NONE,
    ); // BIP16 exception on main net
    exceptions.insert(
        BlockHash::from_str("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad")
            .unwrap(),
        VERIFY_P2SH | VERIFY_WITNESS,
    ); // Taproot exception on main net
    exceptions.insert(
        BlockHash::from_str("00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105")
            .unwrap(),
        VERIFY_NONE,
    ); // BIP16 exception on test net
    exceptions
}

#[cfg(not(feature = "bitcoinconsensus"))]
fn get_exceptions() -> HashMap<BlockHash, c_uint> {
    HashMap::new()
}

impl From<Network> for ChainParams {
    fn from(net: Network) -> Self {
        let genesis = genesis_block(net.into());
        let max_target = ChainParams::max_target(net);
        let exceptions = get_exceptions();
        match net {
            Network::Bitcoin => ChainParams {
                genesis,
                max_target,
                verify_script: true,
                pow_allow_min_diff: false,
                pow_allow_no_retarget: false,
                pow_target_spacing: 10 * 60, // One block every 600 seconds (10 minutes)
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: 210_000,
                coinbase_maturity: 100,
                bip34_activation_height: 227931,
                bip65_activation_height: 388381,
                bip66_activation_height: 363725,
                segwit_activation_height: 481824,
                csv_activation_height: 419328,
                exceptions,
            },
            Network::Testnet => ChainParams {
                genesis,
                max_target,
                verify_script: true,
                pow_allow_min_diff: true,
                pow_allow_no_retarget: false,
                pow_target_spacing: 10 * 60, // One block every 600 seconds (10 minutes)
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: 210_000,
                coinbase_maturity: 100,
                bip34_activation_height: 211_111,
                bip65_activation_height: 581_885,
                bip66_activation_height: 330_776,
                segwit_activation_height: 834_624,
                csv_activation_height: 770_112,
                exceptions,
            },
            Network::Signet => ChainParams {
                genesis,
                max_target,
                verify_script: true,
                pow_allow_min_diff: false,
                pow_allow_no_retarget: false,
                pow_target_spacing: 10 * 60, // One block every 600 seconds (10 minutes)
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: 210_000,
                coinbase_maturity: 100,
                bip34_activation_height: 500,
                csv_activation_height: 1,
                bip65_activation_height: 1,
                bip66_activation_height: 1,
                segwit_activation_height: 1,
                exceptions,
            },
            Network::Regtest => ChainParams {
                genesis,
                max_target,
                verify_script: false,
                pow_allow_min_diff: false,
                pow_allow_no_retarget: true,
                pow_target_spacing: 10 * 60, // One block every 600 seconds (10 minutes)
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: 150,
                coinbase_maturity: 100,
                bip34_activation_height: 500,
                csv_activation_height: 0,
                bip65_activation_height: 0,
                bip66_activation_height: 0,
                segwit_activation_height: 0,
                exceptions,
            },
        }
    }
}

pub fn get_chain_dns_seeds(network: Network) -> Vec<DnsSeed> {
    // DNS seeds taken from Bitcoin Core at commit 382b692a503355df7347efd9c128aff465b5583e
    let mut seeds = Vec::new();

    // Some dns seeds lets us filter the returned peers by advertised services. We are interested
    // in peers with: UTREEXO, COMPACT_FILTERS, WITNESS and NETWORK. Not all seeds supports all
    // bits, so from this list, we pick the ones they support, and ask for this

    // x9 or 0x09 means NETWORK + WITNESS
    let x9: ServiceFlags = ServiceFlags::from(0x9);
    // this mean NETWORK + WITNESS + COMPACT_FILTERS
    let x49 = ServiceFlags::from(0x49);
    // this mean NETWORK + WITNESS + COMPACT_FILTERS + UTREEXO
    let x1000049 = ServiceFlags::from(0x1000049);
    // filters aren't supported (usually returns a static list of peers)
    let none = ServiceFlags::NONE;

    match network {
        Network::Bitcoin => {
            seeds.push(DnsSeed::new(
                Network::Bitcoin,
                "seed.bitcoin.sipa.be",
                x9, // no COMPACT_FILTERS
            ));
            seeds.push(DnsSeed::new(Network::Bitcoin, "dnsseed.bluematt.me", x49));
            seeds.push(DnsSeed::new(
                Network::Bitcoin,
                "dnsseed.bitcoin.dashjr.org",
                none, // no filter
            ));
            seeds.push(DnsSeed::new(Network::Bitcoin, "seed.bitcoinstats.com", x49));
            seeds.push(DnsSeed::new(
                Network::Bitcoin,
                "seed.bitcoin.jonasschnelli.ch",
                x49,
            ));
            seeds.push(DnsSeed::new(
                Network::Bitcoin,
                "seed.btc.petertodd.org",
                x49,
            ));
            seeds.push(DnsSeed::new(
                Network::Bitcoin,
                "seed.bitcoin.sprovoost.nl",
                x49,
            ));
            seeds.push(DnsSeed::new(Network::Bitcoin, "dnsseed.emzy.de", x49));
            seeds.push(DnsSeed::new(Network::Bitcoin, "seed.bitcoin.wiz.biz", x49));
        }
        Network::Testnet => {
            seeds.push(DnsSeed::new(
                Network::Testnet,
                "testnet-seed.bitcoin.jonasschnelli.ch",
                x49,
            ));
            seeds.push(DnsSeed::new(
                Network::Testnet,
                "seed.tbtc.petertodd.org",
                x49,
            ));
            seeds.push(DnsSeed::new(
                Network::Testnet,
                "seed.testnet.bitcoin.sprovoost.nl",
                x49,
            ));
            seeds.push(DnsSeed::new(
                Network::Testnet,
                "testnet-seed.bluematt.me",
                none,
            ));
        }
        Network::Signet => {
            seeds.push(DnsSeed::new(
                Network::Signet,
                "seed.signet.bitcoin.sprovoost.nl",
                x49,
            ));
            seeds.push(DnsSeed::new(Network::Signet, "seed.dlsouza.lol", x1000049));
        }
        Network::Regtest => {
            // No seeds for regtest
        }
    };
    seeds
}
