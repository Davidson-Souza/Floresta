// SPDX-License-Identifier: MIT

//! In node.rs we created a node that connects to the Bitcoin network and downloads the blockchain.
//! We use the default chainstate, which starts at genesis and validates all blocks. This is
//! the simplest way to create a node, but you can also create a node that starts at a given
//! block, or that doesn't validate all signatures. All customizations are done through the
//! ChainStateBuilder struct. This example shows how to use it.
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::hashes::Hash;
use bitcoin::BlockHash;
use floresta::chain::ChainState;
use floresta::chain::KvChainStore;
use floresta::chain::Network;
use floresta_chain::pruned_utreexo::chain_state_builder::ChainStateBuilder;
use floresta_chain::ChainParams;
use rustreexo::accumulator::stump::Stump;

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

    // Create a new chain state builder. We can use it to customize the chain state.
    // Assume valid is the same as in node.rs, it's the block that we assume that all
    // blocks before it have valid signatures.
    //
    // Tip is the block that we consider to be the tip of the chain. If you want to
    // start the chainstate at a given block, you can use this. If you don't set it,
    // it will start at genesis.
    //
    // We also set the chain params, which are the parameters of the network that we
    // are connecting to. We use the Bitcoin network here, but you can also use
    // Testnet, Signet or Regtest.
    //
    // Finally, we set the utreexo accumulator. This is the accumulator that we use
    // to validate the blockchain. If you set the chain height, you should update
    // the accumulator to the state of the blockchain at that height too.
    let _chain: ChainState<KvChainStore> = ChainStateBuilder::new()
        .with_assume_valid(BlockHash::all_zeros())
        .with_chain_params(ChainParams::from(Network::Bitcoin))
        .with_tip(
            (genesis_block(bitcoin::Network::Bitcoin).block_hash(), 0),
            genesis_block(bitcoin::Network::Bitcoin).header,
        )
        .assume_utreexo(Stump::new())
        .with_chainstore(chain_store)
        .build()
        .unwrap();

    // ... If you want to drive the chainstate, you can use the BlockchainInterface trait.
    // See node.rs for an example on how to do it ...
}
