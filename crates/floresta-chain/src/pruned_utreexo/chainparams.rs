// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module provides configuration and parameters for different Bitcoin networks (mainnet,
//! testnet, signet, and regtest).
//!
//! It includes:
//! - Network-specific parameters like block reward halving intervals and maturity periods
//! - Assumable validation states for Utreexo
//! - Block verification flag exceptions

extern crate alloc;
use alloc::vec::Vec;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::Script;
use bitcoin::ScriptBuf;
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::constants::SUBSIDY_HALVING_INTERVAL;
use bitcoin::params::Params;
#[cfg(feature = "bitcoinkernel")]
use bitcoinkernel::ScriptVerificationFlags;
use floresta_common::acchashes;
use floresta_common::bhash;
use rustreexo::node_hash::BitcoinNodeHash;

use crate::AssumeValidArg;
use crate::prelude::*;

// Bitcoin Core script verification flags used by GetTransactionSigOpCost:
// https://github.com/bitcoin/bitcoin/blob/v31.0/src/kernel/bitcoinkernel.h#L468-L479
const VERIFY_NONE: u32 = 0;
pub(super) const VERIFY_P2SH: u32 = 1 << 0;
pub(super) const VERIFY_WITNESS: u32 = 1 << 11;

#[cfg(feature = "bitcoinkernel")]
const _: () = {
    assert!(VERIFY_NONE == bitcoinkernel::VERIFY_NONE);
    assert!(VERIFY_P2SH == bitcoinkernel::VERIFY_P2SH);
    assert!(VERIFY_WITNESS == bitcoinkernel::VERIFY_WITNESS);
};

const DEFAULT_SIGNET_CHALLENGE: &[u8] = &[
    0x51, 0x21, 0x03, 0xad, 0x5e, 0x0e, 0xda, 0xd1, 0x8c, 0xb1, 0xf0, 0xfc, 0x0d, 0x28, 0xa3, 0xd4,
    0xf1, 0xf3, 0xe4, 0x45, 0x64, 0x03, 0x37, 0x48, 0x9a, 0xbb, 0x10, 0x40, 0x4f, 0x2d, 0x1e, 0x08,
    0x6b, 0xe4, 0x30, 0x21, 0x03, 0x59, 0xef, 0x50, 0x21, 0x96, 0x4f, 0xe2, 0x2d, 0x6f, 0x8e, 0x05,
    0xb2, 0x46, 0x3c, 0x95, 0x40, 0xce, 0x96, 0x88, 0x3f, 0xe3, 0xb2, 0x78, 0x76, 0x0f, 0x04, 0x8f,
    0x51, 0x89, 0xf2, 0xe6, 0xc4, 0x52, 0xae,
];

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

    /// The challenge script that signet blocks must satisfy.
    ///
    /// This is [`None`] for networks other than signet.
    pub signet_challenge: Option<ScriptBuf>,

    /// Whether we should enforce BIP-094 "Testnet 4" rules
    pub enforce_bip94: bool,
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
    /// Returns the default Bitcoin signet challenge.
    pub fn default_signet_challenge() -> &'static Script {
        Script::from_bytes(DEFAULT_SIGNET_CHALLENGE)
    }

    /// Returns whether these parameters select a custom signet challenge.
    pub fn is_custom_signet(&self) -> bool {
        self.network == Network::Signet
            && self.signet_challenge.as_deref() != Some(Self::default_signet_challenge())
    }

    /// This method is called when Assume Utreexo is set to true. It means that the user will accept the hardcoded utreexo state for the specified block, if it is found in the best chain. We can then sync rapidly from this state.
    pub fn get_assume_utreexo(network: Network) -> AssumeUtreexoValue {
        let genesis = genesis_block(Params::new(network));
        match network {
            Network::Bitcoin => AssumeUtreexoValue {
                height: 976679,
                block_hash: bhash!(
                    "000000000000000000000d9c98eed0d249631c1326408b194321690611bf52b9"
                ),
                roots: acchashes![
                    "7d9bdee21d75a5c2b4eb9e3a9300b37d14b1c89b0e1c5af9fd41417fc4ce4abc",
                    "de2d96c559d1bb933f0d194cfb36180f2e20204766b40e8d9509e673f3f41f8a",
                    "f425aaddf40db26b4884892f2b1371d37733aa9c68bdec21190f58a73e1f3d98",
                    "e2793a974c9b39aa2c27a014d67dfcc16dd72e3e083e20d9d04d7cf9f7d266ec",
                    "acbc10a5a8506d0ff217b664661d19cd177e1352de6f7f0b4376bee7849e0f12",
                    "8cb1a6153fa5c359b837e84401600923b46cce1b03da792764566498d6e56239",
                    "b480e08f8333e39c715183339a215bcec3eb6f6d3106eeb6d031370f699eb058",
                    "f14d29c159669f93e62eba0d1a317a0bd13e0192f0cc915254b537535c0598b6",
                    "4ef58ba52ce2db8be116bbc6148a181af018642f954c61c555ac8ccae888dd6a",
                    "7cf06c7367e9b25edc7981b8afc07d14fc4449995ee441f3f0ea13228907dcbe",
                    "a01c424e89fef3f455cf4caec71dc0959d66f9687c2a6316f6b8a1fab97fa67a",
                    "a964e30a5b3504661a1bd1047ba6242a6e4dea8a5a1f931b641934f0f15d4cd7",
                    "438ad4152fcab5de786e75cca4928654239ccfad550a9557b56a6ca61f4c1a78",
                    "c65700fab927ff98fcd21c9541e46a23d1ffee4b8e7b5f987fa6c7191dda833b",
                    "fd6db3280d75bb5a796e730b9d1aaa5ba0595a6167ce75f9320149873b6e8c8a",
                ]
                .to_vec(),
                leaves: 3243591085,
            },
            Network::Signet => AssumeUtreexoValue {
                height: 331881,
                block_hash: bhash!(
                    "00000013de70c4cc657575e55ff9a6acc932729128b9434d31679178f71a1ca1"
                ),
                roots: acchashes![
                    "819fe2ab44dd87a33dcc262d29890c6fad0111fa4a13507fc28fa63907f10cec",
                    "3a08e8403f9f8740d076724dec22123ce6ea1dd5b5387d0889e3229a3cecd8a9",
                    "2a990a42dc75ff5e0b1d5ce7b0aa52f3db5c58cc0f3c3fdfc7da4935edcb93c4",
                    "78f67308229225f60ce781e24b2898d2fffc350774259eac6df999f3d7caf334",
                    "f5cc57665fb9e9f017336c6795ecfa20ebcca8e372b2270a45b0b719e5d1b828",
                    "3aa05ca40b5851de3711693f4c26d2ad0ea0e47fa7f0bd060a197fe69fa9eef7",
                    "b4762ae42f3d017ba3fa403338275db4852868a67befa13454839d6052c4aaf2",
                    "3eacdf34fce736ee009959138be9478288b6696a702dbf39ebc532ccdd46cad1",
                    "aa07cb033f653d5d446c0a15599f67193ca82f436258a1ba0f1968e46cf5e7de",
                    "4a4523c54f472ed4c696bc7abbc1918304d0559586dfbaa9f1dd3c06674c5833",
                    "6ca17573455d699f95449b48cd559bf2d1ae7937c77ac672335c023dc77b7338",
                    "e885d24c919131bc3ce641c74c493e2fd70fea5fe250f40d7b7e3e16137827f5",
                    "f88eabe952734dd3655cce64d8af89bf85ddefe12a174aa525febcecb6f06605",
                    "0fcc9ffe10130000dc09791272f0eafb45a3e7c0e69dab0392fbee8b8f217a34",
                    "948a885d7ca57fd9cb9832a96bff33c30ccdb2613ceef5bc9c18aaa7e864da5d",
                    "40ca03221215b734ae246941502b1696ba159271bcc7decf9b58906229c9843d",
                ]
                .to_vec(),
                leaves: 163448509,
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
    ///   Custom signets have no hardcoded checkpoint.
    ///   - **Bitcoin**: block [971,420](https://mempool.space/block/971420)
    ///   - **Signet**: block [326,326](https://mempool.space/signet/block/326326)
    ///   - **Testnet**: block [4,887,983](https://mempool.space/testnet/block/4887983)
    ///   - **Testnet4**: block [126,514](https://mempool.space/testnet4/block/126514)
    ///   - **Regtest**: genesis block
    pub fn get_assume_valid(&self, arg: AssumeValidArg) -> Option<BlockHash> {
        match arg {
            AssumeValidArg::Disabled => None,
            AssumeValidArg::UserInput(hash) => Some(hash),
            AssumeValidArg::Hardcoded if self.is_custom_signet() => None,
            AssumeValidArg::Hardcoded => match self.network {
                Network::Bitcoin => Some(bhash!(
                    "000000000000000000000d9c98eed0d249631c1326408b194321690611bf52b9" // 976_679
                )),
                Network::Signet => Some(bhash!(
                    "00000013de70c4cc657575e55ff9a6acc932729128b9434d31679178f71a1ca1" // 331_881
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

    #[cfg(not(feature = "bitcoinkernel"))]
    /// Returns the flags relevant to contextual sigop accounting.
    pub(crate) fn get_sigop_flags(&self, hash: BlockHash) -> u32 {
        self.exceptions
            .get(&hash)
            .copied()
            .unwrap_or(VERIFY_P2SH | VERIFY_WITNESS)
    }
}

/// There's almost no transactions in the chain that
/// "looks like segwit but are not segwit". We pretend segwit
/// was enabled since genesis, and only skip this for blocks
/// that have such transactions using hardcoded values.
fn get_exceptions() -> HashMap<BlockHash, u32> {
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

impl AsRef<Params> for ChainParams {
    fn as_ref(&self) -> &Params {
        &self.params
    }
}

impl From<Network> for ChainParams {
    fn from(network: Network) -> Self {
        let genesis = genesis_block(Params::new(network));
        let exceptions = get_exceptions();
        let signet_challenge = match network {
            Network::Signet => Some(ScriptBuf::from_bytes(DEFAULT_SIGNET_CHALLENGE.to_vec())),
            _ => None,
        };

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
                signet_challenge,
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
                signet_challenge,
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
                signet_challenge,
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
                signet_challenge,
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
                signet_challenge,
                enforce_bip94: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_signet_challenge_matches_bitcoin_core() {
        let params = ChainParams::from(Network::Signet);

        assert_eq!(
            params
                .signet_challenge
                .as_deref()
                .map(|script| script.as_bytes()),
            Some(DEFAULT_SIGNET_CHALLENGE)
        );
    }

    #[test]
    fn distinguishes_default_and_custom_signet() {
        let mut params = ChainParams::from(Network::Signet);
        assert!(!params.is_custom_signet());

        params.signet_challenge = Some(ScriptBuf::from_bytes(vec![0x51]));
        assert!(params.is_custom_signet());
        assert!(!ChainParams::from(Network::Bitcoin).is_custom_signet());
    }

    #[test]
    fn custom_signet_disables_only_hardcoded_assume_valid() {
        let mut params = ChainParams::from(Network::Signet);
        params.signet_challenge = Some(ScriptBuf::from_bytes(vec![0x51]));
        let explicit_checkpoint = params.genesis.block_hash();

        assert_eq!(params.get_assume_valid(AssumeValidArg::Hardcoded), None);
        assert_eq!(
            params.get_assume_valid(AssumeValidArg::UserInput(explicit_checkpoint)),
            Some(explicit_checkpoint)
        );
    }

    #[test]
    fn non_signet_networks_have_no_signet_challenge() {
        for network in [
            Network::Bitcoin,
            Network::Testnet,
            Network::Testnet4,
            Network::Regtest,
        ] {
            assert!(ChainParams::from(network).signet_challenge.is_none());
        }
    }
}
