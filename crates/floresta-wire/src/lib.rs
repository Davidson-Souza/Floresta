// SPDX-License-Identifier: MIT
//! # Floresta Wire
//! This crate provides the core networking logic for a full node using libfloresta,
//! including the P2P network and the mempool. You can easily integrate it with any
//! other crate that provides a `BlockchainInterface` and `UpdatableChainstate`
//! implementation.
//!
//! A node also gives you a `handle` that you can use to send messages to the node,
//! like requesting blocks, mempool transactions or asking to connect with a given
//! peer.

#[cfg(not(target_arch = "wasm32"))]
mod p2p_wire;
use bitcoin::block::Header as BlockHeader;
use bitcoin::Block;
use bitcoin::Transaction;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::address_man;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::mempool;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node_context;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::node_interface;
#[cfg(not(target_arch = "wasm32"))]
pub use p2p_wire::running_node;
pub use p2p_wire::UtreexoNodeConfig;
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
