// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Floresta Common
//! Provides utility functions, macros and modules to be
//! used in other Floresta crates.

// cargo docs customization
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/249173822")]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/getfloresta/floresta-media/master/logo_png/Icon-Green(main).png"
)]
#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use bitcoin::Network;
use bitcoin::ScriptBuf;
use bitcoin::VarInt;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::encode;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use bitcoin::p2p::ServiceFlags;
#[cfg(feature = "std")]
mod ema;
pub mod macros;
pub mod merkle;
pub mod spsc;

#[cfg(feature = "std")]
pub use ema::Ema;
pub use merkle::MerkleBackend;
pub use spsc::Channel;
#[cfg(feature = "std")]
pub trait ChainMethods {
    type Error: core::error::Error;

    /// Gets a block by its hash.
    fn get_block(
        &self,
        block: bitcoin::BlockHash,
    ) -> impl core::future::Future<Output = Result<Option<bitcoin::Block>, Self::Error>> + Send;

    /// Returns a list of Compact Block Filters headers for the requested block range.
    fn get_cfilters_headers(
        &self,
        start_height: u32,
        stop_hash: bitcoin::BlockHash,
    ) -> impl core::future::Future<
        Output = Result<bitcoin::p2p::message_filter::CFHeaders, Self::Error>,
    > + Send;

    /// Returns basic compact block filters for consecutive `block_hashes`.
    fn get_cfilter(
        &self,
        start_height: u32,
        block_hashes: Vec<bitcoin::BlockHash>,
    ) -> impl core::future::Future<Output = Result<Vec<bitcoin::bip158::BlockFilter>, Self::Error>> + Send;

    /// Returns BIP157 filter-header checkpoints through `stop_hash`.
    fn get_cfcheckpt(
        &self,
        stop_hash: bitcoin::BlockHash,
    ) -> impl core::future::Future<
        Output = Result<bitcoin::p2p::message_filter::CFCheckpt, Self::Error>,
    > + Send;
}

/// Computes the SHA-256 digest of the byte slice data and returns a [Hash] from `bitcoin_hashes`.
///
/// [Hash]: https://docs.rs/bitcoin_hashes/latest/bitcoin_hashes/sha256/struct.Hash.html
pub fn get_hash_from_u8(data: &[u8]) -> sha256::Hash {
    sha256::Hash::hash(data)
}

/// Computes the SHA-256 digest of a script, reverses its bytes, and returns a [Hash] from
/// `bitcoin_hashes`.
///
/// The source to the specification can be found in the Electrum protocol [documentation], and it is
/// used to identify scripts in the Electrum Protocol.
///
/// [documentation]: https://electrum-protocol.readthedocs.io/en/latest/protocol-basics.html#script-hashes
/// [Hash]: https://docs.rs/bitcoin_hashes/latest/bitcoin_hashes/sha256/struct.Hash.html
pub fn get_spk_hash(spk: &ScriptBuf) -> sha256::Hash {
    let data = spk.as_bytes();
    let mut bytes = sha256::Hash::hash(data).to_byte_array();
    bytes.reverse();
    sha256::Hash::from_byte_array(bytes)
}
/// Reads a VarInt from the given reader and ensures it is less than or equal to `max`.
///
/// Returns an error if the VarInt is larger than `max`.
pub fn read_bounded_len<R: bitcoin::io::Read + ?Sized>(
    reader: &mut R,
    max: usize,
) -> Result<usize, encode::Error> {
    let n64 = VarInt::consensus_decode(reader)?.0;
    if n64 > max as u64 {
        return Err(encode::Error::OversizedVectorAllocation {
            requested: n64 as usize,
            max,
        });
    }
    Ok(n64 as usize)
}

/// Utreexo-specific service flags.
///
/// TODO(@luisschwab): remove this once <https://github.com/rust-bitcoin/rust-bitcoin/pull/5009> is merged.
pub mod service_flags {
    /// `UTREEXO`: the node is capable of serving inclusion proofs for new
    /// blocks and transactions, and for their other advertised services.
    pub const UTREEXO: u64 = 1 << 12;

    /// `UTREEXO_ARCHIVE`: the node is capable of serving historical
    /// inclusion proofs for all blocks, but not necessarily historical blocks.
    pub const UTREEXO_ARCHIVE: u64 = 1 << 13;
}

/// Extension trait for [`bitcoin::Network`] providing network-specific defaults.
// TODO(@luisschwab): get rid of this once
// https://github.com/rust-bitcoin/rust-bitcoin/pull/6502 makes it into a release.
// TODO: move to a dedicated network utilities crate if needed.
pub trait NetworkExt {
    /// Returns the default RPC port for the given network.
    fn default_rpc_port(&self) -> u16;
}

impl NetworkExt for Network {
    fn default_rpc_port(&self) -> u16 {
        match self {
            Self::Bitcoin => 8332,
            Self::Signet => 38332,
            Self::Testnet => 18332,
            Self::Testnet4 => 48332,
            Self::Regtest => 18442,
        }
    }
}

/// The P2P protocol version Floresta speaks.
pub const PROTOCOL_VERSION: u32 = 70016;

/// The services advertised by this node.
///
///   - `WITNESS`: SegWit blocks and transactions (BIP-0144).
///   - `P2P_V2`: Encrypted transport (BIP-0324).
///   - `UTREEXO`: Utreexo inclusion proofs (BIP-0183).
pub fn advertised_services() -> ServiceFlags {
    ServiceFlags::WITNESS | ServiceFlags::P2P_V2 | ServiceFlags::from(service_flags::UTREEXO)
}

/// Returns string names for all known service flags that are set in `flags`.
pub fn service_flags_strings(flags: &ServiceFlags) -> Vec<String> {
    let known_flags = [
        (ServiceFlags::NETWORK, "NETWORK"),
        (ServiceFlags::GETUTXO, "GETUTXO"),
        (ServiceFlags::BLOOM, "BLOOM"),
        (ServiceFlags::WITNESS, "WITNESS"),
        (ServiceFlags::COMPACT_FILTERS, "COMPACT_FILTERS"),
        (ServiceFlags::NETWORK_LIMITED, "NETWORK_LIMITED"),
        (ServiceFlags::P2P_V2, "P2P_V2"),
        (service_flags::UTREEXO.into(), "UTREEXO"),
        (service_flags::UTREEXO_ARCHIVE.into(), "UTREEXO_ARCHIVE"),
    ];

    known_flags
        .iter()
        .filter(|(flag, _)| flags.has(*flag))
        .map(|(_, name)| name.to_string())
        .collect()
}

#[cfg(not(feature = "std"))]
pub mod prelude {
    extern crate alloc;
    pub use alloc::borrow::ToOwned;
    pub use alloc::boxed::Box;
    pub use alloc::format;
    pub use alloc::string::String;
    pub use alloc::string::ToString;
    pub use alloc::vec;
    pub use alloc::vec::Vec;
    pub use core::cmp;
    pub use core::convert;
    pub use core::iter;
    pub use core::mem;
    pub use core::ops;
    pub use core::ops::Deref;
    pub use core::ops::DerefMut;
    pub use core::option;
    pub use core::result;
    pub use core::slice;

    pub use bitcoin::io::Error as ioError;
    pub use bitcoin::io::Read;
    pub use bitcoin::io::Write;
    pub use hashbrown::HashMap;
    pub use hashbrown::HashSet;
}

#[cfg(feature = "std")]
/// Provides implementation for basic `std` types, without assuming we have a `std` library.
///
/// This module is used to avoid having `#[cfg(feature = "no-std")]` sprinkled
/// around all crates that support `no-std`. It imports all types we would use
/// from the `stdlib`, either from the lib itself, or from other sources in case
/// `stdlib` isn't available.
pub mod prelude {
    extern crate alloc;
    extern crate std;
    pub use alloc::format;
    pub use alloc::string::ToString;
    pub use std::borrow::ToOwned;
    pub use std::boxed::Box;
    pub use std::collections::HashMap;
    pub use std::collections::HashSet;
    pub use std::collections::hash_map::Entry;
    pub use std::io::Error as ioError;
    pub use std::io::Read;
    pub use std::io::Write;
    pub use std::ops::Deref;
    pub use std::ops::DerefMut;
    pub use std::result::Result;
    pub use std::string::String;
    pub use std::sync;
    pub use std::vec;
    pub use std::vec::Vec;
}

#[cfg(test)]
mod tests {
    use bitcoin::Network;
    use bitcoin::ScriptBuf;
    use bitcoin::hashes::Hash;
    use bitcoin::hex::DisplayHex;

    use super::NetworkExt;
    use super::prelude::*;

    #[test]
    fn test_get_hash_from_u8() {
        let data = b"Hello, world!";
        let hash = super::get_hash_from_u8(data);
        let expected =
            String::from("315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3");
        assert_eq!(hash.as_byte_array().to_lower_hex_string(), expected);
    }

    #[test]
    fn test_get_spk_hash() {
        // Example taken from Electrum protocol documentation
        // https://electrum-protocol.readthedocs.io/en/latest/protocol-basics.html#script-hashes

        let spk =
            ScriptBuf::from_hex("76a91462e907b15cbf27d5425399ebf6f0fb50ebb88f1888ac").unwrap(); // P2PKH script
        let hash = super::get_spk_hash(&spk);
        let expected =
            String::from("8b01df4e368ea28f8dc0423bcf7a4923e3a12d307c875e47a0cfbf90b5c39161");

        assert_eq!(hash.as_byte_array().to_lower_hex_string(), expected);
    }

    #[test]
    fn test_default_rpc_port() {
        assert_eq!(Network::Bitcoin.default_rpc_port(), 8332);
        assert_eq!(Network::Testnet.default_rpc_port(), 18332);
        assert_eq!(Network::Testnet4.default_rpc_port(), 48332);
        assert_eq!(Network::Signet.default_rpc_port(), 38332);
        assert_eq!(Network::Regtest.default_rpc_port(), 18442);
    }
}
