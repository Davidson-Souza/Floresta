// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Floresta Wire
//! This crate provides the core networking logic for a full node using libfloresta,
//! including the P2P network and the mempool. You can easily integrate it with any
//! other crate that provides a `BlockchainInterface` and `UpdatableChainstate`
//! implementation.
//!
//! A node also gives you a `handle` that you can use to send messages to the node,
//! like requesting blocks, mempool transactions or asking to connect with a given
//! peer.

// cargo docs customization
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/249173822")]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/getfloresta/floresta-media/master/logo_png/Icon-Green(main).png"
)]

use bitcoin::Block;
use bitcoin::Transaction;
use bitcoin::block::Header as BlockHeader;
pub use rustreexo;
#[cfg(not(target_arch = "wasm32"))]
mod p2p_wire;
pub use p2p_wire::UtreexoNodeConfig;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::address_man;
pub use p2p_wire::bitcoin_socket_addr;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::block_proof;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::error;
#[cfg(all(feature = "fuzzing", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use p2p_wire::fuzz;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::network_message_ext;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node_context;
pub use p2p_wire::node_handle;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node_interface;
pub use p2p_wire::onion;
pub use p2p_wire::transport::TransportProtocol;

/// NodeHooks is a trait that defines the hooks that a node can use to interact with the network
/// and the blockchain. Every time an event happens, the node will call the corresponding hook.
pub trait NodeHooks {
    /// We've received a new block
    fn on_block_received(&mut self, block: &Block);
    /// We've received a new transaction
    fn on_transaction_received(&mut self, transaction: &Transaction);
    /// We've received a new peer
    fn on_peer_connected(&mut self, peer: &u32);
    /// We've lost a peer
    fn on_peer_disconnected(&mut self, peer: &u32);
    /// We've received a new header
    fn on_header_received(&mut self, header: &BlockHeader);
}
