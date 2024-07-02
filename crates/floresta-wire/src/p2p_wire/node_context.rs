//! During the lifetime of a Bitcoin client, we have a couple of phases that are slightly different
//! from each other, having to implement their own state-machines and logic for handing requests.
//! While we could simply put everything in one struct and have a single `impl` block, that would
//! create a massive amount of if's in the code, taking different paths depending on which state
//! are we in. For that reason, we define the basics of a node, like code shared by all the
//! states into one base struct called `UtreexoNode`, we then further refine this struct using
//! fine-tunned `Contexts`, that should implement [NodeContext] and are passed-in as a generic
//! parameter by the caller.
//!
//! The three flavors of node are:
//!  - ChainSelector: This finds the best PoW chain, by downloding multiple candidates and taking
//!                   the one with more PoW. It should do it's job quickly, as it blocks our main
//!                   client and can't proceed without this information.
//!  - SyncNode: Used to download and verify all blocks in a chain. This is computationally
//!              expensive and may take a while to run. After this ends it's job, it gives us 100%
//!              centanty that this chain is valid.
//!  - Running Node: This is the one that users interacts with, and should be the one running most
//!                  of the time. This node is started right after `ChainSelector` returns, and
//!                  will handle new blocks (even if `SyncNode` haven't returned) and handle
//!                  requests by users.

use bitcoin::p2p::ServiceFlags;

/// This trait mainly defines a bunch of constants that we need for the node, but we may tweak
/// those values for each one. It's also an organized way of defining those constants anyway.
pub trait NodeContext {
    const REQUEST_TIMEOUT: u64;
    /// Max number of simultaneous connections we initiates we are willing to hold
    const MAX_OUTGOING_PEERS: usize = 10;
    /// We ask for peers every ASK_FOR_PEERS_INTERVAL seconds
    const ASK_FOR_PEERS_INTERVAL: u64 = 60 * 60; // One hour
    /// Save our database of peers every PEER_DB_DUMP_INTERVAL seconds
    const PEER_DB_DUMP_INTERVAL: u64 = 30; // 30 seconds
    /// Attempt to open a new connection (if needed) every TRY_NEW_CONNECTION seconds
    const TRY_NEW_CONNECTION: u64 = 10; // 10 seconds
    /// If ASSUME_STALE seconds passed since our last tip update, treat it as stale
    const ASSUME_STALE: u64 = 15 * 60; // 15 minutes
    /// While on IBD, if we've been without blocks for this long, ask for headers again
    const IBD_REQUEST_BLOCKS_AGAIN: u64 = 30; // 30 seconds
    /// How often we broadcast transactions
    const BROADCAST_DELAY: u64 = 30; // 30 seconds
    /// Max number of simultaneous inflight requests we allow
    const MAX_INFLIGHT_REQUESTS: usize = 1_000;
    /// Interval at which we open new feeler connections
    const FEELER_INTERVAL: u64 = 30; // 30 seconds
    /// Interval at which we rearrange our addresses
    const ADDRESS_REARRANGE_INTERVAL: u64 = 60 * 60; // 1 hour
    /// How long we ban a peer for
    const BAN_TIME: u64 = 60 * 60 * 24;
    /// How often we check if we haven't missed a block
    const BLOCK_CHECK_INTERVAL: u64 = 60 * 5; // 5 minutes
    /// How often we send our addresses to our peers
    const SEND_ADDRESSES_INTERVAL: u64 = 60 * 60; // 1 hour
    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::NETWORK
    }
}

pub(crate) type PeerId = u32;
