// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Floresta Chain
//! This crate provides the core validation logic for a full node using libfloresta.
//! It is maintained as a separate crate to allow other projects to build on it,
//! independent of the libfloresta P2P network or libfloresta wallet.
//! The main entry point is the [ChainState] struct, that keeps track of the current
//! blockchain state, like headers and utreexo accumulator.
//!
//! All data is stored in a `ChainStore` implementation, which is generic over the
//! underlying database. See the ChainStore trait for more information. For a
//! ready-to-use implementation, see the [`FlatChainStore`] struct.

// cargo docs customization
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/249173822")]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/getfloresta/floresta-media/master/logo_png/Icon-Green(main).png"
)]
#![allow(clippy::manual_is_multiple_of)]
#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

pub mod extensions;

pub mod pruned_utreexo;

#[cfg(test)]
pub(crate) mod test_utils;
pub(crate) use floresta_common::prelude;
pub use pruned_utreexo::BlockchainInterface;
pub use pruned_utreexo::ChainBackend;
pub use pruned_utreexo::Notification;
pub use pruned_utreexo::ThreadSafeChain;
pub use pruned_utreexo::chain_state::*;
pub use pruned_utreexo::chainparams::*;
pub use pruned_utreexo::chainstore::*;
pub use pruned_utreexo::consensus::swift_sync_agg;
pub use pruned_utreexo::error::*;
#[cfg(feature = "flat-chainstore")]
pub use pruned_utreexo::flat_chain_store::*;
pub use pruned_utreexo::udata::*;
pub use pruned_utreexo::utxo_data::*;
