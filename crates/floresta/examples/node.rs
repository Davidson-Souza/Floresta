// SPDX-License-Identifier: MIT

//! A simple example of a Bitcoin node using the floresta crates. It connects to the Bitcoin
//! network, downloads the blockchain, and prints the current state of the accumulator.
//! This will validate all blocks from genesis to the current tip, so it will take a while
//! to sync.

use std::str::FromStr;
use std::sync::Arc;

use bitcoin::BlockHash;
use floresta::chain::pruned_utreexo::BlockchainInterface;
use floresta::chain::ChainState;
use floresta::chain::KvChainStore;
use floresta::chain::Network;
use floresta::wire::mempool::Mempool;
use floresta::wire::node::UtreexoNode;
use floresta_chain::AssumeValidArg;
use floresta_wire::node_interface::NodeMethods;
use floresta_wire::running_node::RunningNode;
use floresta_wire::UtreexoNodeConfig;
use tokio::sync::RwLock;

const DATA_DIR: &str = "./data";

#[tokio::main]
async fn main() {
    // Create a new chain state, which will store the accumulator and the headers chain.
    // It will be stored in the DATA_DIR directory. With this chain state, we don't keep
    // the block data after we validated it. This saves a lot of space, but it means that
    // we can't serve blocks to other nodes or rescan the blockchain without downloading
    // it again.
    let chain_store =
        KvChainStore::new(DATA_DIR.into()).expect("failed to open the blockchain database");

    // The actual chainstate. It will keep track of the current state of the accumulator
    // and the headers chain. It will also validate new blocks and headers as we receive them.
    // The last parameter is the assume valid block. We assume that all blocks before this
    // one have valid signatures. This is a performance optimization, as we don't need to validate all
    // signatures in the blockchain, just the ones after the assume valid block. We are giving a Disabled
    // value, so we will validate all signatures regardless.
    // We place the chain state in an Arc, so we can share it with other components.
    let chain = Arc::new(ChainState::<KvChainStore>::new(
        chain_store,
        Network::Bitcoin,
        AssumeValidArg::Disabled,
    ));

    // Create a new node. It will connect to the Bitcoin network and start downloading the blockchain.
    // It will also start a mempool, which will keep track of the current mempool state, this
    // particular mempool doesn't store other's transactions, it just keeps track of our own, to
    // perform broadcast. We always rebroadcast our own transactions every hour.
    // Note that we are using the RunningNode context, which is a state optimized for a node that
    // already has the blockchain synced. You don't need to worry about this, because internally
    // the node will automatically switch to the IBD context and back once it's finished.
    // If you want a node to IBD only, you can use the IBDNode context.
    // Finally, we are using the chain state created above, the node will use it to determine
    // what blocks and headers to download, and hand them to it to validate.
    let config = UtreexoNodeConfig::default();
    let p2p: UtreexoNode<RunningNode, Arc<ChainState<KvChainStore>>> = UtreexoNode::new(
        config,
        chain.clone(),
        Arc::new(RwLock::new(Mempool::new())),
        None,
    );
    // A handle is a simple way to interact with the node. It implements a queue of requests
    // that will be processed by the node.
    let handle = p2p.get_handle();

    let (sender, _receiver) = futures::channel::oneshot::channel();

    // Start the node. This will start the IBD process, and will return once the node is synced.
    // It will also start the mempool, which will start rebroadcasting our transactions every hour.
    // The node will keep running until the process is killed, by setting kill_signal to true. In
    // this example, we don't kill the node, so it will keep running forever.
    p2p.run(Arc::new(RwLock::new(false)), sender).await;

    // That's it! The node is now running, and will keep running until the process is killed.
    // You can now use the chain state to query the current state of the accumulator, or the
    // mempool to query the current state of the mempool. You may also ask the node to grab some
    // blocks or headers for you, or to send a transaction to the network, rescan the blockchain,
    // etc. Check the documentation of the node for more information.

    // You can't request blocks or headers from the node until it's synced. You can check if it's
    // synced by calling the is_in_ibd method.
    loop {
        // Wait till the node is synced
        if !chain.is_in_idb() {
            break;
        }
        // Sleep for 10 seconds, and check again
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

    // Here we ask the node to grab the block with the given hash.
    let block = handle
        .get_block(
            BlockHash::from_str("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f")
                .unwrap(),
        )
        .unwrap();
    println!("Block: {:?}", block);
}
