// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module provides configuration and parameters for different Bitcoin networks (mainnet,
//! testnet, signet, and regtest).
//!
//! It includes:
//! - Network-specific parameters like block reward halving intervals and maturity periods
//! - DNS seeds for peer discovery
//! - Assumable validation states for Utreexo
//! - Block verification flag exceptions
//!
//! The main struct [`ChainParams`] encapsulates all chain-specific parameters while
//! [`DnsSeed`] handles peer discovery through DNS.

extern crate alloc;
use alloc::vec::Vec;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::Work;
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::constants::SUBSIDY_HALVING_INTERVAL;
use bitcoin::p2p::ServiceFlags;
use bitcoin::params::Params;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::ScriptVerificationFlags;
use floresta_common::acchashes;
use floresta_common::bhash;
use floresta_common::service_flags;
use rustreexo::node_hash::BitcoinNodeHash;

use crate::AssumeValidArg;
use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsidyHalvingInterval {
    /// Bitcoin, testnet, testnet4, and signet: 210,000 blocks.
    Bitcoin,

    /// Regtest: 150 blocks.
    Regtest,
}

impl SubsidyHalvingInterval {
    pub const fn get(self) -> u32 {
        match self {
            Self::Bitcoin => SUBSIDY_HALVING_INTERVAL,
            Self::Regtest => 150,
        }
    }
}

#[derive(Clone, Debug)]
/// This struct encapsulates all chain-specific parameters.
pub struct ChainParams {
    /// Field to store parameters related to the chain consensus.
    pub params: Params,

    /// The network's first block, also called genesis block.
    pub genesis: Block,

    /// Interval of blocks until the block reward halves
    pub subsidy_halving_interval: SubsidyHalvingInterval,

    /// When we retarget we expect this many seconds to be elapsed since last time. If
    /// it's more, we decrease difficulty, if it's less we increase difficulty
    pub pow_target_timespan: u64,

    /// We wait this many blocks before a coinbase output can be spent
    pub coinbase_maturity: u32,

    /// The height at which segwit is activated
    pub segwit_activation_height: u32,

    /// The height at which csv(CHECK_SEQUENCE_VERIFY) is activated
    pub csv_activation_height: u32,

    /// A list of exceptions to the rules, where the key is the block hash and the value is the
    /// verification flags
    pub exceptions: HashMap<BlockHash, u32>,

    /// The network this chain params is for
    pub network: Network,

    /// Whether we should enforce BIP-094 "Testnet 4" rules
    pub enforce_bip94: bool,
}

/// A dns seed is a authoritative DNS server that returns the IP addresses of nodes that are
/// likely to be accepting incoming connections. This is our preferred way of finding new peers
/// on the first startup, as peers returned by seeds are likely to be online and accepting
/// connections. We may use this as a fallback if we don't have any peers to connect in
/// subsequent startups.
///
/// Some seeds allow filtering by service flags, so we may use this to find peers that are
/// likely to be running Utreexo, for example.
pub struct DnsSeed {
    /// The domain name of the seed
    pub seed: &'static str,

    /// Useful filters we can use to find relevant peers
    pub filters: ServiceFlags,
}

/// This functionality is used to create a new DNS seed with possible filters.
impl DnsSeed {
    /// Create a new DNS seed
    pub fn new(seed: &'static str, filters: ServiceFlags) -> Self {
        Self { seed, filters }
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
    pub roots: Vec<BitcoinNodeHash>,

    /// The number of leaves in the Utreexo accumulator at this block
    pub leaves: u64,
}

impl ChainParams {
    /// This method is called when Assume Utreexo is set to true. It means that the user will accept the hardcoded utreexo state for the specified block, if it is found in the best chain. We can then sync rapidly from this state.
    pub fn get_assume_utreexo(network: Network) -> AssumeUtreexoValue {
        let genesis = genesis_block(Params::new(network));
        match network {
            Network::Bitcoin => AssumeUtreexoValue {
                height: 939969,
                block_hash: bhash!(
                    "000000000000000000009d36aae180d04aeac872adb14e22f65c8b6647a8bf79"
                ),
                roots: acchashes![
                    "08daaf0c6bc41531885cfcfdeb89c34bd4d06ab4b105cf0e81bd74ab082693f5",
                    "8d4166d0303d41f7023cd35b95b24455b99b2f4a2728083bba3d172727900bed",
                    "08d95bc9b7bc0bc07c9f626322c0092bd16c198fcb96d290fe1a191e9719b4c9",
                    "e663da82fd6523124b4ae5b52d1790460cfdd16ae7733fb5ef7d9d0e8911f516",
                    "87a9dd5e49e99b29c394622207a27f0c2ba51a8325eedccc8f604b07b3dafe23",
                    "174bed42eb80a6e5adec1e97f7fc8e736e96d99d0f4d6cdcfed0343281934aa3",
                    "16a4b567f72e39d928b8fc84afc50e035bba5639c5bdaa01b53ddb806190f96a",
                    "d1a2df40385aa7c5ac3b9fc9f1ebd4fa9c2d007842fa70e3f60a434762ae8855",
                    "cf985706bc58a2fb89edc2f28b7c4bf04af440c0534da01b27a2a7eea5f98391",
                    "6d3c3406b7b99e476a754f3d04a1bec2442b2699c8b7d317ee0aca877b59a74f",
                    "f166e22db463c0c1ec2c374d6ddff56699ae27d52443622154a4f6f29b2d6bac",
                    "11270cc9d63da0cf47b2a75e8948576fecc9f6405981f7306c2b58b0fcb37ef6",
                    "fc9bcdcf61bf2ee5480326d24a460ef520f4b02d1cec8a309784795ddcc4bcd9",
                    "c97717ab33820c8efe9d464ff48a1d593e06e4c99135686ae96cc7328a290a79",
                    "5b3c899b033b43989ed57db0854e2deb084de27fcf8596e9c60738ccd08d16c1",
                    "745b78a30e984590b7ceb3a433adb363862cf84b747cb55e500eb25164b5f71f",
                    "d338cb3e81a902b4e0301f69ac9443f7a9bb423446bbac41306c7e839bae4866",
                    "d3078b9c3bb20af622559fc2837d339210cd4e1e13ebdb66d19ca70e83e91a00",
                ]
                .to_vec(),
                leaves: 3066490760,
            },
            Network::Signet => AssumeUtreexoValue {
                height: 296870,
                block_hash: bhash!(
                    "000000068d38e9cfa53268a08b32dab55118a58e5212729b016ee3c9c66387a6"
                ),
                roots: acchashes![
                    "2f50b3b1ed71ab671b8f9e90dcd1a73aba00ee9e8441f34c388efe224956257c",
                    "cfbc9f1139665175ffdf6f949ec8e82956172e128ca4c1fca228ef5b6816c196",
                    "31bcb14a70b18a8c22a4144e270dcdae8795cda26e938ed9c676dbba4443b8ad",
                    "cb90b5ad4df6ff872f315575a8dda593f6d1382349f6f2e41e315f7ceb8a084f",
                    "ca0db15d9e59ae69691e82b3e9b2fb65b7b932275cd86795337a066fb1f371a7",
                    "d802536c657b0b52e34929ebf26033f0354d1ac65f2186f34e967fafa5b0c3fd",
                    "fe157607ceec8e0f46d401f0f1a0d3421bfe40abaef708d9694246e512be27da",
                    "87a75c823e13bb8d0111c238a0a9b5d8d89ba79395f7970418d9981ad5b0ffc3",
                    "4670816e7c71a989e2d4da8dd491a045681536276e386913dca98c7f6fbe5156",
                    "ed1e927d905d01616e066492d1f4a014151b74801299a596918d103c438e2551",
                    "5c8e0755507f65c79972aaadf49a86f3c5f02a75d855a6a3380707830cc20ae2"
                ]
                .to_vec(),
                leaves: 126113034,
            },
            Network::Testnet => AssumeUtreexoValue {
                block_hash: genesis.block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
            Network::Testnet4 => AssumeUtreexoValue {
                block_hash: genesis.block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
            Network::Regtest => AssumeUtreexoValue {
                block_hash: genesis.block_hash(),
                height: 0,
                leaves: 0,
                roots: Vec::new(),
            },
        }
    }

    /// Returns the [`BlockHash`] to use as the assume-valid checkpoint,
    /// or [`None`] if script validation should run on all blocks.
    ///
    /// Blocks at and before this checkpoint skip script execution during IBD.
    /// This argument does not influence chain selection; if the best chain doesn't
    /// include this block, we will verify all the historical scripts.
    ///
    /// # Variants
    /// - [`AssumeValidArg::Disabled`] — no checkpoint; all scripts are validated.
    /// - [`AssumeValidArg::UserInput`] — use the provided hash.
    /// - [`AssumeValidArg::Hardcoded`] — use a release-time checkpoint per [`Network`]:
    ///   - **Bitcoin**: block [939,969](https://mempool.space/block/939969)
    ///   - **Signet**: block [296,870](https://mempool.space/signet/block/296870)
    ///   - **Testnet**: block [4,887,983](https://mempool.space/testnet/block/4887983)
    ///   - **Testnet4**: block [126,514](https://mempool.space/testnet4/block/126514)
    ///   - **Regtest**: genesis block
    pub fn get_assume_valid(network: Network, arg: AssumeValidArg) -> Option<BlockHash> {
        match arg {
            AssumeValidArg::Disabled => None,
            AssumeValidArg::UserInput(hash) => Some(hash),
            AssumeValidArg::Hardcoded => match network {
                Network::Bitcoin => Some(bhash!(
                    "000000000000000000009d36aae180d04aeac872adb14e22f65c8b6647a8bf79" // 939_969
                )),
                Network::Signet => Some(bhash!(
                    "000000068d38e9cfa53268a08b32dab55118a58e5212729b016ee3c9c66387a6" // 296_870
                )),
                Network::Testnet => Some(bhash!(
                    "000000005cf458fb1f79c8fee78822eead52aee40530a6bbe018cd61f22d6bb1" // 4_887_983
                )),
                Network::Testnet4 => Some(bhash!(
                    "000000000066d17b237cd1ac323526731084c3eed82caeacd1ec028c6fea7276" // 126_514
                )),
                Network::Regtest => Some(bhash!(
                    "0f9188f13cb7b2c71f2a335e3a4fc328bf5beb436012afca590b1a11466e2206" // 0
                )),
            },
        }
    }

    #[cfg(feature = "bitcoinkernel")]
    /// Returns the validation flags for a given block hash and height
    pub fn get_validation_flags(&self, height: u32, hash: BlockHash) -> ScriptVerificationFlags {
        if let Some(flag) = self.exceptions.get(&hash) {
            return *flag;
        }

        // From Bitcoin Core:
        // BIP16 didn't become active until Apr 1 2012 (on mainnet, and
        // retroactively applied to testnet)
        // However, only one historical block violated the P2SH rules (on both
        // mainnet and testnet).
        // Similarly, only one historical block violated the TAPROOT rules on
        // mainnet.
        // For simplicity, always leave P2SH+WITNESS+TAPROOT on except for the two
        // violating blocks.
        let mut flags = bitcoinkernel::VERIFY_P2SH
            | bitcoinkernel::VERIFY_WITNESS
            | bitcoinkernel::VERIFY_TAPROOT;

        if height >= self.params.bip65_height {
            flags |= bitcoinkernel::VERIFY_CHECKLOCKTIMEVERIFY;
        }
        if height >= self.params.bip66_height {
            flags |= bitcoinkernel::VERIFY_DERSIG;
        }
        if height >= self.csv_activation_height {
            flags |= bitcoinkernel::VERIFY_CHECKSEQUENCEVERIFY;
        }
        if height >= self.segwit_activation_height {
            flags |= bitcoinkernel::VERIFY_NULLDUMMY;
        }

        flags
    }
}

#[cfg(feature = "bitcoinkernel")]
/// There's almost no transactions in the chain that
/// "looks like segwit but are not segwit". We pretend segwit
/// was enabled since genesis, and only skip this for blocks
/// that have such transactions using hardcoded values.
fn get_exceptions() -> HashMap<BlockHash, ScriptVerificationFlags> {
    use bitcoinkernel::VERIFY_NONE;
    use bitcoinkernel::VERIFY_P2SH;
    use bitcoinkernel::VERIFY_WITNESS;

    let mut exceptions = HashMap::new();
    exceptions.insert(
        bhash!("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22"),
        VERIFY_NONE,
    ); // BIP16 exception on main net
    exceptions.insert(
        bhash!("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad"),
        VERIFY_P2SH | VERIFY_WITNESS,
    ); // Taproot exception on main net
    exceptions.insert(
        bhash!("00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105"),
        VERIFY_NONE,
    ); // BIP16 exception on test net
    exceptions
}

#[cfg(not(feature = "bitcoinkernel"))]
fn get_exceptions() -> HashMap<BlockHash, u32> {
    HashMap::new()
}

impl AsRef<Params> for ChainParams {
    fn as_ref(&self) -> &Params {
        &self.params
    }
}

impl From<Network> for ChainParams {
    fn from(network: Network) -> Self {
        let genesis = genesis_block(Params::new(network));
        let exceptions = get_exceptions();

        match network {
            Network::Bitcoin => Self {
                params: Params::new(network),
                network,
                genesis,
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: SubsidyHalvingInterval::Bitcoin,
                coinbase_maturity: 100,
                segwit_activation_height: 481_824,
                csv_activation_height: 419_328,
                exceptions,
                enforce_bip94: false,
            },
            Network::Testnet => Self {
                params: Params::new(network),
                network,
                genesis,
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: SubsidyHalvingInterval::Bitcoin,
                coinbase_maturity: 100,
                segwit_activation_height: 834_624,
                csv_activation_height: 770_112,
                exceptions,
                enforce_bip94: false,
            },
            Network::Testnet4 => Self {
                params: Params::new(network),
                network,
                genesis,
                pow_target_timespan: 14 * 24 * 60 * 60,
                subsidy_halving_interval: SubsidyHalvingInterval::Bitcoin,
                coinbase_maturity: 100,
                segwit_activation_height: 1,
                csv_activation_height: 1,
                exceptions,
                enforce_bip94: true,
            },
            Network::Signet => Self {
                params: Params::new(network),
                network,
                genesis,
                pow_target_timespan: 14 * 24 * 60 * 60, // two weeks
                subsidy_halving_interval: SubsidyHalvingInterval::Bitcoin,
                coinbase_maturity: 100,
                csv_activation_height: 1,
                segwit_activation_height: 1,
                exceptions,
                enforce_bip94: false,
            },
            Network::Regtest => Self {
                params: Params::new(network),
                network,
                genesis,
                pow_target_timespan: 24 * 60 * 60, // one day
                subsidy_halving_interval: SubsidyHalvingInterval::Regtest,
                coinbase_maturity: 100,
                csv_activation_height: 1,
                segwit_activation_height: 0,
                exceptions,
                enforce_bip94: false,
            },
        }
    }
}

/// Get a list of [`DnsSeed`]s for a given [`Network`].
///
/// Some DNS seeds allow requesting addresses using a [`ServiceFlags`] filter.
/// Here we define `x9`, `x49`, and `x1009` (the relevant services for this node
/// to operate), and use them to request addresses from the DNS seeds that support it.
pub fn get_chain_dns_seeds(network: Network) -> Vec<DnsSeed> {
    let mut seeds = Vec::new();

    let none = ServiceFlags::NONE;
    let x9 = ServiceFlags::NETWORK | ServiceFlags::WITNESS;
    let x49 = ServiceFlags::NETWORK | ServiceFlags::WITNESS | ServiceFlags::COMPACT_FILTERS;
    let x1009 = ServiceFlags::NETWORK | ServiceFlags::WITNESS | service_flags::UTREEXO.into();
    let x1000 = service_flags::UTREEXO.into();

    #[rustfmt::skip]
    match network {
        Network::Bitcoin => {
            seeds.push(DnsSeed::new("seed.calvinkim.info", x1009));
            seeds.push(DnsSeed::new("seed.bitcoin.luisschwab.com", x1009));
            seeds.push(DnsSeed::new("seed.bitcoin.sipa.be", x9));
            seeds.push(DnsSeed::new("dnsseed.bluematt.me", x49));
            seeds.push(DnsSeed::new("seed.bitcoinstats.com", x49));
            seeds.push(DnsSeed::new("seed.btc.petertodd.org", x49));
            seeds.push(DnsSeed::new("seed.bitcoin.sprovoost.nl", x49));
            seeds.push(DnsSeed::new("dnsseed.emzy.de", x49));
            seeds.push(DnsSeed::new("seed.bitcoin.wiz.biz", x49));
            seeds.push(DnsSeed::new("bitcoin.seed.dlsouza.lol", x1000));
        }
        Network::Signet => {
            seeds.push(DnsSeed::new("signet.seed.dlsouza.lol", x1000));
            seeds.push(DnsSeed::new("seed.signet.bitcoin.sprovoost.nl", x49));
            seeds.push(DnsSeed::new("signet.seed.utreexo.net", x1009));
        }
        Network::Testnet => {
            seeds.push(DnsSeed::new("testnet-seed.bitcoin.jonasschnelli.ch", x49));
            seeds.push(DnsSeed::new("testnet.seed.dlsouza.lol", x1000));
            seeds.push(DnsSeed::new("seed.tbtc.petertodd.org", x49));
            seeds.push(DnsSeed::new("seed.testnet.bitcoin.sprovoost.nl", x49));
            seeds.push(DnsSeed::new("testnet-seed.bluematt.me", none));
        }
        Network::Testnet4 => {
            seeds.push(DnsSeed::new("seed.testnet4.bitcoin.sprovoost.nl", none));
            seeds.push(DnsSeed::new("seed.testnet4.wiz.biz", none));
        }
        Network::Regtest => {}
    };

    seeds
}

/// Returns the buried deployment list for a network, as `(name, activation_height)` pairs.
///
/// Heights are sourced from Bitcoin Core's `chainparams.cpp` at v30.2
/// (commit `4d7d5f6b79d4c11c47e7a828d81296918fd11d4d`):
/// <https://github.com/bitcoin/bitcoin/blob/4d7d5f6b79d4c11c47e7a828d81296918fd11d4d/src/kernel/chainparams.cpp>
//
// TODO: also emit BIP9 deployments (`taproot`, `testdummy`); requires the versionbits state machine.
pub fn buried_deployments_for(network: Network) -> &'static [(&'static str, u32)] {
    const BITCOIN_BURIED: &[(&str, u32)] = &[
        ("bip34", 227_931),
        ("bip66", 363_725),
        ("bip65", 388_381),
        ("csv", 419_328),
        ("segwit", 481_824),
    ];

    const TESTNET_BURIED: &[(&str, u32)] = &[
        ("bip34", 21_111),
        ("bip66", 330_776),
        ("bip65", 581_885),
        ("csv", 770_112),
        ("segwit", 834_624),
    ];

    const TESTNET4_BURIED: &[(&str, u32)] = &[
        ("bip34", 1),
        ("bip66", 1),
        ("bip65", 1),
        ("csv", 1),
        ("segwit", 1),
    ];

    const SIGNET_BURIED: &[(&str, u32)] = &[
        ("bip34", 1),
        ("bip66", 1),
        ("bip65", 1),
        ("csv", 1),
        ("segwit", 1),
    ];

    const REGTEST_BURIED: &[(&str, u32)] = &[
        ("bip34", 1),
        ("bip66", 1),
        ("bip65", 1),
        ("csv", 1),
        ("segwit", 0),
    ];

    match network {
        Network::Bitcoin => BITCOIN_BURIED,
        Network::Testnet => TESTNET_BURIED,
        Network::Testnet4 => TESTNET4_BURIED,
        Network::Signet => SIGNET_BURIED,
        Network::Regtest => REGTEST_BURIED,
    }
}

/// Returns the minimum cumulative chainwork a valid chain must demonstrate.
/// Mirrors [`nMinimumChainWork`](https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp) in Bitcoin Core.
pub fn minimum_chain_work(network: Network) -> Work {
    // TODO(@Micah-Shallom): sync this value with nMinimumChainWork in Bitcoin Core's chainparams.cpp on each Core release.
    match network {
        // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L117
        Network::Bitcoin => {
            Work::from_hex("0x0000000000000000000000000000000000000001128750f82f4c366153a3a030")
                .expect("hardcoded work")
        }
        // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L248
        Network::Testnet => {
            Work::from_hex("0x0000000000000000000000000000000000000000000017dde1c649f3708d14b6")
                .expect("hardcoded work")
        }
        // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L356
        Network::Testnet4 => {
            Work::from_hex("0x0000000000000000000000000000000000000000000009a0fe15d0177d086304")
                .expect("hardcoded work")
        }
        // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L447
        // Custom -signetchallenge signets use 0; Floresta targets the default signet.
        Network::Signet => {
            Work::from_hex("0x00000000000000000000000000000000000000000000000000000b463ea0a4b8")
                .expect("hardcoded work")
        }
        _ => Work::from_be_bytes([0u8; 32]),
    }
}
