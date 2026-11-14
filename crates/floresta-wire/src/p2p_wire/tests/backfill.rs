// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests for the backfill node: a [`SyncNode`] running over a [`PartialChainState`], built like
//! `running_ctx::backfill` does in production.
//!
//! [`SyncNode`]: crate::node::sync_ctx::SyncNode
//! [`PartialChainState`]: floresta_chain::pruned_utreexo::partial_chain::PartialChainState

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::mem::discriminant;
    use std::time::Instant;

    use bitcoin::Network;
    use floresta_chain::extensions::BlockExt;
    use floresta_chain::extensions::BlockExtError;
    use floresta_common::acchashes;
    use rustreexo::node_hash::BitcoinNodeHash;
    use rustreexo::stump::Stump;

    use crate::node::NodeNotification;
    use crate::p2p_wire::peer::PeerMessages;
    use crate::p2p_wire::tests::utils::BackfillOutcome;
    use crate::p2p_wire::tests::utils::BlockDefect;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::mutate_block;
    use crate::p2p_wire::tests::utils::run_backfill;
    use crate::p2p_wire::tests::utils::run_backfill_node;
    use crate::p2p_wire::tests::utils::setup_backfill_node;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;
    use crate::p2p_wire::tests::utils::synthetic_block;

    const NUM_BLOCKS: usize = 9;

    /// The peer that sends the unsolicited block in the tests below.
    const OFFENDER: u32 = 0;

    /// Expected accumulator at height 9 after syncing the pristine signet blocks,
    /// used as the expected result of the `PartialChainState` (backfill) path.
    fn honest_acc() -> Stump {
        Stump {
            leaves: 9,
            roots: acchashes![
                "73fa24017bd8ead2baef8d17fc154f8cb22fc4b0d945faee6ba41941379b993e",
                "397fb8fc9377dc0fd014fd327f6bd588b2ff8bf8a4132a8c07d93d6ca6a1bc9e"
            ]
            .to_vec(),
        }
    }

    /// Asserts the backfill reached the same state as the `ChainState` path.
    fn assert_honest_outcome(outcome: BackfillOutcome) {
        assert_eq!(outcome.height, 9);
        assert!(!outcome.is_in_ibd);
        assert!(!outcome.has_invalid_blocks);
        assert_eq!(outcome.acc, honest_acc());
    }

    /// Runs a backfill node, which syncs over a `PartialChainState`, and checks it reaches the
    /// same state as the `ChainState` path. If `mutated`, nine extra peers send a mutated block
    /// at height 7, and only one peer has the original txdata.
    async fn check_backfill(mutated: bool) {
        let datadir = format!("./tmp-db/{}.backfill", rand::random::<u32>());
        let honest_peer = PeerData::new(Vec::new(), signet_blocks(), HashMap::new());

        let mut peers = Vec::new();
        if mutated {
            let headers = signet_headers();
            let mut blocks = signet_blocks();
            mutate_block(blocks.get_mut(&headers[7].block_hash()).unwrap());

            peers = vec![PeerData::new(Vec::new(), blocks, HashMap::new()); 9];
        }
        peers.push(honest_peer);

        let args = SetupNodeArgs::new(peers, false, Network::Signet, datadir, NUM_BLOCKS);
        assert_honest_outcome(run_backfill(args).await);
    }

    #[tokio::test]
    async fn test_backfill_valid_blocks() {
        check_backfill(false).await;
    }

    #[tokio::test]
    async fn test_backfill_mutated_block() {
        check_backfill(true).await;
    }

    /// Builds a backfill node whose `num_peers` peers serve the pristine signet blocks, and
    /// queues a [`synthetic_block`] with `defect` as an unsolicited block from [`OFFENDER`],
    /// through the same channel a real peer uses, before the node starts running.
    ///
    /// `expected_err` is what [`BlockExt::check_block_structure`] must return for the block:
    /// `None` if the block is structurally valid.
    ///
    /// The block isn't the next block of the backfill interval, so it's never processed: asserts
    /// the backfill still completes with the honest state.
    async fn check_unsolicited_block(
        defect: BlockDefect,
        expected_err: Option<BlockExtError>,
        num_peers: usize,
    ) {
        let block = synthetic_block(defect);
        let err = block.check_block_structure().err();
        assert_eq!(
            err.as_ref().map(discriminant),
            expected_err.as_ref().map(discriminant),
            "got {err:?}, expected {expected_err:?}"
        );

        let datadir = format!("./tmp-db/{}.backfill", rand::random::<u32>());
        let peers = vec![PeerData::new(Vec::new(), signet_blocks(), HashMap::new()); num_peers];
        let args = SetupNodeArgs::new(peers, false, Network::Signet, datadir, NUM_BLOCKS);
        let node = setup_backfill_node(args);

        let message =
            NodeNotification::FromPeer(OFFENDER, PeerMessages::Block(block), Instant::now());
        node.node_tx.send(message).unwrap();

        assert_honest_outcome(run_backfill_node(node).await);
    }

    #[tokio::test]
    async fn test_backfill_unsolicited_invalid_block_too_many_sigops() {
        check_unsolicited_block(
            BlockDefect::TooManySigops,
            Some(BlockExtError::TooManySigops),
            1,
        )
        .await;
    }

    #[tokio::test]
    async fn test_backfill_unsolicited_invalid_block_too_big() {
        check_unsolicited_block(
            BlockDefect::BlockTooBig,
            Some(BlockExtError::BlockTooBig),
            1,
        )
        .await;
    }

    #[tokio::test]
    async fn test_backfill_unsolicited_mutated_block_bad_witness_commitment() {
        check_unsolicited_block(
            BlockDefect::WitnessCommitment,
            Some(BlockExtError::BadWitnessCommitment),
            2,
        )
        .await;
    }

    #[tokio::test]
    async fn test_backfill_unsolicited_mutated_block_bad_merkle_root() {
        check_unsolicited_block(
            BlockDefect::MerkleRoot,
            Some(BlockExtError::BadMerkleRoot),
            2,
        )
        .await;
    }

    #[tokio::test]
    async fn test_backfill_unsolicited_unconnected_block() {
        check_unsolicited_block(BlockDefect::None, None, 1).await;
    }
}
